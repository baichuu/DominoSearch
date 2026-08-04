"""Monotonic gradual magnitude pruning utilities.

The functions in this module are intentionally independent from the ImageNet
training entry point so the schedule and exact mask updates can be tested with
small synthetic models.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch


@dataclass(frozen=True)
class GradualMagnitudeConfig:
    """Configuration for an inclusive epoch-based cubic pruning schedule."""

    target_sparsity: float
    start_epoch: int
    end_epoch: int
    frequency: int = 1
    power: float = 3.0
    scope: str = "global"
    prune_first: bool = False
    prune_last: bool = False

    def validate(self, total_epochs: int) -> None:
        if not 0.0 < self.target_sparsity < 1.0:
            raise ValueError("Gradual target sparsity must be between 0 and 1.")
        if self.start_epoch < 0:
            raise ValueError("Gradual start epoch cannot be negative.")
        if self.end_epoch < self.start_epoch:
            raise ValueError("Gradual end epoch must be >= start epoch.")
        if self.end_epoch >= total_epochs:
            raise ValueError("Gradual end epoch must be smaller than total epochs.")
        if self.frequency <= 0:
            raise ValueError("Gradual pruning frequency must be positive.")
        if self.power <= 0:
            raise ValueError("Gradual pruning power must be positive.")
        if self.scope not in {"global", "local"}:
            raise ValueError("Gradual pruning scope must be 'global' or 'local'.")

    def to_dict(self) -> dict:
        return asdict(self)


def scheduled_sparsity(config: GradualMagnitudeConfig, epoch: int) -> float:
    """Return target sparsity for ``epoch`` using a cubic-style schedule."""

    if epoch < config.start_epoch:
        return 0.0
    if epoch >= config.end_epoch or config.end_epoch == config.start_epoch:
        return config.target_sparsity
    progress = (epoch - config.start_epoch) / (
        config.end_epoch - config.start_epoch
    )
    return config.target_sparsity * (
        1.0 - math.pow(1.0 - progress, config.power)
    )


def should_update_masks(config: GradualMagnitudeConfig, epoch: int) -> bool:
    if epoch < config.start_epoch:
        return False
    if epoch >= config.end_epoch:
        return epoch == config.end_epoch
    return (epoch - config.start_epoch) % config.frequency == 0


def eligible_weight_parameters(
    model,
    prune_first: bool = False,
    prune_last: bool = False,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[str]]:
    """Return Conv/Linear weights eligible for pruning and protected names."""

    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    if not layers:
        raise ValueError("Model contains no Conv2d or Linear layers to prune.")

    eligible = []
    protected = []
    for index, (module_name, module) in enumerate(layers):
        parameter_name = f"{module_name}.weight" if module_name else "weight"
        protect = (
            (index == 0 and not prune_first)
            or (index == len(layers) - 1 and not prune_last)
        )
        if protect:
            protected.append(parameter_name)
        else:
            eligible.append((parameter_name, module.weight))
    if not eligible:
        raise ValueError("No eligible weights remain after boundary protection.")
    return eligible, protected


def initialize_masks(
    eligible: Iterable[tuple[str, torch.nn.Parameter]],
    preserve_existing_zeros: bool = False,
) -> Dict[str, torch.Tensor]:
    """Create device-local boolean masks, optionally restoring zeroed weights."""

    masks = {}
    for name, parameter in eligible:
        if preserve_existing_zeros:
            masks[name] = parameter.detach().ne(0)
        else:
            masks[name] = torch.ones_like(parameter, dtype=torch.bool)
    return masks


def restore_masks_from_training_checkpoint(
    model_dir: Path,
    eligible: Iterable[tuple[str, torch.nn.Parameter]],
    config: GradualMagnitudeConfig,
) -> Dict[str, torch.Tensor] | None:
    """Restore exact gradual masks and reject a changed resume schedule."""

    pointer = Path(model_dir) / "checkpoint"
    if not pointer.exists():
        return None
    pointer_lines = pointer.read_text(encoding="utf-8").splitlines()
    if not pointer_lines:
        raise ValueError(f"Empty checkpoint pointer: {pointer}")
    first_line = pointer_lines[0]
    if ":" not in first_line:
        raise ValueError(f"Invalid checkpoint pointer: {pointer}")
    checkpoint_path = Path(first_line.split(":", 1)[1])
    if not checkpoint_path.exists():
        candidate = Path(model_dir) / checkpoint_path.name
        checkpoint_path = candidate if candidate.exists() else checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_metadata = checkpoint.get("gradual_pruning")
    saved_masks = checkpoint.get("parameter_masks")
    if saved_metadata is None or saved_masks is None:
        raise ValueError(
            "Cannot resume gradual pruning: checkpoint has no gradual schedule "
            "and exact parameter masks. Use an empty --model_dir for a new run."
        )
    if saved_metadata.get("schedule") != config.to_dict():
        raise ValueError(
            "Gradual pruning schedule differs from the checkpoint being resumed."
        )

    eligible = list(eligible)
    expected = {name for name, _ in eligible}
    if set(saved_masks) != expected:
        raise ValueError("Checkpoint masks do not exactly cover eligible parameters.")
    restored = {}
    for name, parameter in eligible:
        mask = saved_masks[name]
        if not torch.is_tensor(mask) or mask.shape != parameter.shape:
            raise ValueError(f"Invalid checkpoint mask for '{name}'.")
        mask = mask.to(device=parameter.device, dtype=torch.bool)
        restored[name] = mask
    return restored


def _additional_prune_count(mask: torch.Tensor, sparsity: float) -> int:
    target = round(mask.numel() * sparsity)
    already_pruned = mask.numel() - int(mask.count_nonzero().item())
    return max(0, target - already_pruned)


def _prune_local(
    eligible: Iterable[tuple[str, torch.nn.Parameter]],
    masks: Dict[str, torch.Tensor],
    sparsity: float,
) -> None:
    for name, parameter in eligible:
        mask = masks[name]
        additional = _additional_prune_count(mask, sparsity)
        if additional == 0:
            continue
        scores = parameter.detach().abs().reshape(-1).clone()
        flat_mask = mask.reshape(-1)
        scores.masked_fill_(~flat_mask, torch.inf)
        indices = torch.topk(
            scores, k=additional, largest=False, sorted=False
        ).indices
        flat_mask[indices] = False


def _prune_global(
    eligible: Iterable[tuple[str, torch.nn.Parameter]],
    masks: Dict[str, torch.Tensor],
    sparsity: float,
) -> None:
    eligible = list(eligible)
    total = sum(parameter.numel() for _, parameter in eligible)
    already_pruned = sum(
        mask.numel() - int(mask.count_nonzero().item()) for mask in masks.values()
    )
    additional = max(0, round(total * sparsity) - already_pruned)
    if additional == 0:
        return

    scores = torch.cat(
        [parameter.detach().abs().reshape(-1) for _, parameter in eligible]
    )
    flat_mask = torch.cat([masks[name].reshape(-1) for name, _ in eligible])
    scores.masked_fill_(~flat_mask, torch.inf)
    indices = torch.topk(scores, k=additional, largest=False, sorted=False).indices
    flat_mask[indices] = False

    offset = 0
    for name, parameter in eligible:
        count = parameter.numel()
        masks[name].copy_(flat_mask[offset : offset + count].reshape(parameter.shape))
        offset += count


def apply_gradual_pruning(
    eligible: Iterable[tuple[str, torch.nn.Parameter]],
    masks: Dict[str, torch.Tensor],
    sparsity: float,
    scope: str,
) -> dict:
    """Monotonically prune to an exact rounded target and apply the masks."""

    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Scheduled sparsity must satisfy 0 <= sparsity < 1.")
    eligible = list(eligible)
    expected_names = {name for name, _ in eligible}
    if set(masks) != expected_names:
        raise ValueError("Gradual masks do not exactly cover eligible parameters.")
    for name, parameter in eligible:
        if masks[name].shape != parameter.shape:
            raise ValueError(f"Mask shape mismatch for '{name}'.")

    before = {name: mask.clone() for name, mask in masks.items()}
    if scope == "global":
        _prune_global(eligible, masks, sparsity)
    elif scope == "local":
        _prune_local(eligible, masks, sparsity)
    else:
        raise ValueError("Gradual pruning scope must be 'global' or 'local'.")

    with torch.no_grad():
        for name, parameter in eligible:
            if torch.any(masks[name] & ~before[name]):
                raise RuntimeError("Gradual pruning attempted to regrow a masked weight.")
            parameter.mul_(masks[name].to(dtype=parameter.dtype))
    return mask_statistics(eligible, masks)


def mask_statistics(
    eligible: Iterable[tuple[str, torch.nn.Parameter]],
    masks: Dict[str, torch.Tensor],
) -> dict:
    eligible = list(eligible)
    total = sum(parameter.numel() for _, parameter in eligible)
    kept = sum(int(masks[name].count_nonzero().item()) for name, _ in eligible)
    layers = []
    for name, parameter in eligible:
        layer_kept = int(masks[name].count_nonzero().item())
        layers.append(
            {
                "parameter": name,
                "weights": parameter.numel(),
                "nonzero_mask": layer_kept,
                "sparsity_percent": 100.0 * (1.0 - layer_kept / parameter.numel()),
            }
        )
    return {
        "eligible_weights": total,
        "nonzero_mask": kept,
        "eligible_sparsity": 1.0 - kept / total,
        "layers": layers,
    }


def save_mask_artifact(
    path: Path,
    masks: Dict[str, torch.Tensor],
    config: GradualMagnitudeConfig,
    epoch: int,
    scheduled_target: float,
    statistics: dict,
    protected_parameters: list[str],
) -> None:
    """Write one immutable, epoch-specific mask artifact idempotently."""

    path = Path(path)
    payload = {
        "parameter_masks": {
            name: mask.detach().to(device="cpu", dtype=torch.bool)
            for name, mask in masks.items()
        },
        "method": "unstructured-gradual-magnitude",
        "schedule": config.to_dict(),
        "epoch": epoch,
        "scheduled_target": scheduled_target,
        "statistics": statistics,
        "protected_parameters": protected_parameters,
    }
    if path.exists():
        existing = torch.load(path, map_location="cpu")
        metadata_matches = all(
            existing.get(key) == payload[key]
            for key in (
                "method",
                "schedule",
                "epoch",
                "scheduled_target",
                "statistics",
                "protected_parameters",
            )
        )
        existing_masks = existing.get("parameter_masks", {})
        masks_match = set(existing_masks) == set(payload["parameter_masks"]) and all(
            torch.equal(existing_masks[name], mask)
            for name, mask in payload["parameter_masks"].items()
        )
        if not metadata_matches or not masks_match:
            raise FileExistsError(
                f"Existing gradual mask artifact does not match resumed state: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
