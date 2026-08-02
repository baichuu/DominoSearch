#!/usr/bin/env python3
"""Create exact global or layer-local unstructured magnitude masks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "train"
MODEL_ROOT = TRAIN_ROOT / "classification_sparsity_level"
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(TRAIN_ROOT))

try:
    import torch

    import models
    from devkit.sparse_ops import SparseConv, SparseLinear
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install them with: pip install -r requirements.txt"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot magnitude pruning.")
    parser.add_argument("--model", default="resnet18_sparse")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--scope", choices=("global", "local"), default="global")
    parser.add_argument("--prune-first", action="store_true")
    parser.add_argument("--prune-last", action="store_true")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path)
    parser.add_argument("--scheme-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def load_model(args: argparse.Namespace):
    if not 0.0 < args.sparsity < 1.0:
        raise ValueError("--sparsity must be between 0 and 1 (exclusive).")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive.")
    if args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("Refusing to overwrite the input checkpoint.")
    constructor = models.__dict__.get(args.model)
    if not callable(constructor):
        raise ValueError(f"Unknown model constructor: {args.model}")
    model = constructor(
        pretrained=False,
        N=args.group_size,
        M=args.group_size,
        num_classes=args.num_classes,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict or contain 'state_dict'.")
    state = {name.removeprefix("module."): value for name, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"Checkpoint mismatch; missing={list(incompatible.missing_keys)[:5]}, "
            f"unexpected={list(incompatible.unexpected_keys)[:5]}"
        )
    return model, checkpoint


def eligible_weights(model, prune_first: bool, prune_last: bool):
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    eligible = []
    protected = []
    for index, (name, module) in enumerate(layers):
        is_protected = (
            (index == 0 and not prune_first)
            or (index == len(layers) - 1 and not prune_last)
        )
        (protected if is_protected else eligible).append((name, module))
    if not eligible:
        raise ValueError("No eligible Conv2d/Linear weights remain after protection rules.")
    return eligible, protected


def exact_smallest_mask(weight, prune_count: int):
    flat = weight.detach().abs().reshape(-1)
    mask = torch.ones(flat.numel(), dtype=torch.bool)
    if prune_count:
        indices = torch.topk(flat, k=prune_count, largest=False, sorted=False).indices
        mask[indices] = False
    return mask.reshape(weight.shape)


def make_masks(eligible, sparsity: float, scope: str):
    masks = {}
    if scope == "local":
        for name, module in eligible:
            prune_count = round(module.weight.numel() * sparsity)
            masks[f"{name}.weight"] = exact_smallest_mask(module.weight, prune_count)
        return masks

    sizes = [module.weight.numel() for _, module in eligible]
    scores = torch.cat([module.weight.detach().abs().reshape(-1) for _, module in eligible])
    prune_count = round(scores.numel() * sparsity)
    global_mask = torch.ones(scores.numel(), dtype=torch.bool)
    if prune_count:
        indices = torch.topk(scores, k=prune_count, largest=False, sorted=False).indices
        global_mask[indices] = False
    offset = 0
    for (name, module), size in zip(eligible, sizes):
        masks[f"{name}.weight"] = global_mask[offset : offset + size].reshape(
            module.weight.shape
        )
        offset += size
    return masks


def dense_scheme(model, group_size: int) -> dict[str, list[int]]:
    return {
        layer.get_name(): [group_size, group_size]
        for layer in model.modules()
        if isinstance(layer, (SparseConv, SparseLinear))
    }


def main() -> None:
    args = parse_args()
    model, original_checkpoint = load_model(args)
    eligible, protected = eligible_weights(model, args.prune_first, args.prune_last)
    masks = make_masks(eligible, args.sparsity, args.scope)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            parameters[name].mul_(mask.to(dtype=parameters[name].dtype))

    output = args.output.expanduser().resolve()
    mask_output = (
        args.mask_output.expanduser().resolve()
        if args.mask_output
        else output.with_suffix(output.suffix + ".masks.pth")
    )
    scheme_output = (
        args.scheme_output.expanduser().resolve()
        if args.scheme_output
        else output.with_suffix(output.suffix + ".dense-scheme.txt")
    )
    manifest_output = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output.with_suffix(output.suffix + ".json")
    )
    for path in (output, mask_output, scheme_output, manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    output_payload = (
        dict(original_checkpoint)
        if isinstance(original_checkpoint, dict) and "state_dict" in original_checkpoint
        else {}
    )
    output_payload["state_dict"] = model.state_dict()
    output_payload["pruning_method"] = "unstructured-magnitude"
    torch.save(output_payload, output)
    torch.save({"parameter_masks": masks, "method": "unstructured-magnitude"}, mask_output)
    scheme_output.write_text(repr(dense_scheme(model, args.group_size)) + "\n", encoding="utf-8")

    all_weights = [
        module.weight
        for module in model.modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    total_weights = sum(weight.numel() for weight in all_weights)
    nonzero_weights = sum(torch.count_nonzero(weight).item() for weight in all_weights)
    layer_stats = []
    for name, module in eligible:
        count = module.weight.numel()
        nonzero = torch.count_nonzero(module.weight).item()
        layer_stats.append(
            {
                "layer": name,
                "weights": count,
                "nonzero": nonzero,
                "sparsity_percent": 100.0 * (1.0 - nonzero / count),
            }
        )
    manifest = {
        "schema_version": 1,
        "method": "unstructured-magnitude",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "model": args.model,
        "scope": args.scope,
        "requested_eligible_sparsity": args.sparsity,
        "protected_layers": [name for name, _ in protected],
        "input_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "output_checkpoint": str(output),
        "mask_file": str(mask_output),
        "dense_scheme_file": str(scheme_output),
        "overall_conv_linear_weight_sparsity_percent": 100.0
        * (1.0 - nonzero_weights / total_weights),
        "layers": layer_stats,
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Scope: {args.scope}; eligible layers: {len(eligible)}")
    print(f"Materialized checkpoint: {output}")
    print(f"Persistent masks: {mask_output}")
    print(f"Dense N:M scheme: {scheme_output}")
    print(f"Manifest: {manifest_output}")


if __name__ == "__main__":
    main()
