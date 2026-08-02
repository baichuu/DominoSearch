#!/usr/bin/env python3
"""Materialize safe hidden-channel pruning masks for ResNet residual blocks."""

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
    parser = argparse.ArgumentParser(
        description="Prune hidden channels inside ResNet blocks using L1 or BN scores."
    )
    parser.add_argument("--model", default="resnet18_sparse")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--score", choices=("l1", "bn"), default="l1")
    parser.add_argument("--alignment", type=int, default=8)
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
    if not 0.0 < args.ratio < 1.0:
        raise ValueError("--ratio must be between 0 and 1 (exclusive).")
    if args.alignment <= 0 or args.group_size <= 0:
        raise ValueError("--alignment and --group-size must be positive.")
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


def kept_indices(conv, bn, ratio: float, alignment: int, score_method: str):
    channels = conv.out_channels
    if alignment > channels:
        raise ValueError(f"Alignment {alignment} exceeds channel count {channels}.")
    requested = max(1, round(channels * (1.0 - ratio)))
    kept = max(alignment, round(requested / alignment) * alignment)
    kept = min(channels, kept)
    if score_method == "bn":
        scores = bn.weight.detach().abs()
    else:
        scores = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    return torch.topk(scores, k=kept, largest=True, sorted=True).indices.sort().values


def apply_site_masks(
    conv,
    bn,
    next_conv,
    keep,
    parameter_names,
    masks,
):
    if next_conv.groups != 1:
        raise ValueError("Grouped convolutions are not supported by this channel pruner.")
    channel_mask = torch.zeros(conv.out_channels, dtype=torch.bool)
    channel_mask[keep] = True
    removed = ~channel_mask

    def get_mask(parameter):
        name = parameter_names[id(parameter)]
        if name not in masks:
            masks[name] = torch.ones_like(parameter, dtype=torch.bool, device="cpu")
        return masks[name]

    get_mask(conv.weight)[removed, ...] = False
    if conv.bias is not None:
        get_mask(conv.bias)[removed] = False
    get_mask(bn.weight)[removed] = False
    get_mask(bn.bias)[removed] = False
    get_mask(next_conv.weight)[:, removed, ...] = False

    with torch.no_grad():
        bn.running_mean[removed] = 0
        bn.running_var[removed] = 1
    return int(channel_mask.sum().item()), int(channel_mask.numel())


def prune_model(model, ratio: float, alignment: int, score_method: str):
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    masks = {}
    sites = []
    for block_name, block in model.named_modules():
        if block.__class__.__name__ not in {"BasicBlock", "Bottleneck"}:
            continue
        pairs = [("conv1_to_conv2", block.conv1, block.bn1, block.conv2)]
        if block.__class__.__name__ == "Bottleneck":
            pairs.append(("conv2_to_conv3", block.conv2, block.bn2, block.conv3))
        for site_name, conv, bn, next_conv in pairs:
            keep = kept_indices(conv, bn, ratio, alignment, score_method)
            kept, total = apply_site_masks(
                conv, bn, next_conv, keep, parameter_names, masks
            )
            sites.append(
                {
                    "block": block_name,
                    "site": site_name,
                    "kept_channels": kept,
                    "original_channels": total,
                    "channel_reduction_percent": 100.0 * (1.0 - kept / total),
                }
            )

    if not sites:
        raise ValueError("No supported ResNet BasicBlock/Bottleneck pruning sites found.")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            parameters[name].mul_(mask.to(dtype=parameters[name].dtype))
    return masks, sites


def dense_scheme(model, group_size: int) -> dict[str, list[int]]:
    return {
        layer.get_name(): [group_size, group_size]
        for layer in model.modules()
        if isinstance(layer, (SparseConv, SparseLinear))
    }


def main() -> None:
    args = parse_args()
    model, original_checkpoint = load_model(args)
    masks, sites = prune_model(model, args.ratio, args.alignment, args.score)

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
    output_payload["pruning_method"] = "structured-channel"
    torch.save(output_payload, output)
    torch.save({"parameter_masks": masks, "method": "structured-channel"}, mask_output)
    scheme_output.write_text(repr(dense_scheme(model, args.group_size)) + "\n", encoding="utf-8")

    conv_linear_weights = [
        module.weight
        for module in model.modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    total_weights = sum(weight.numel() for weight in conv_linear_weights)
    nonzero_weights = sum(torch.count_nonzero(weight).item() for weight in conv_linear_weights)
    manifest = {
        "schema_version": 1,
        "method": "structured-channel",
        "stage": "materialized-mask-not-compact-export",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "model": args.model,
        "input_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "output_checkpoint": str(output),
        "mask_file": str(mask_output),
        "dense_scheme_file": str(scheme_output),
        "requested_channel_ratio": args.ratio,
        "alignment": args.alignment,
        "score": args.score,
        "pruning_sites": sites,
        "conv_linear_weight_sparsity_percent": 100.0 * (1.0 - nonzero_weights / total_weights),
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Pruned {len(sites)} safe hidden-channel site(s).")
    print(f"Materialized checkpoint: {output}")
    print(f"Persistent masks: {mask_output}")
    print(f"Dense N:M scheme: {scheme_output}")
    print(f"Manifest: {manifest_output}")


if __name__ == "__main__":
    main()
