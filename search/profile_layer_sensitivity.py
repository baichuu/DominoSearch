#!/usr/bin/env python3
"""Measure one-layer-at-a-time N:M accuracy sensitivity on validation data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parent
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from benchmark_model import (  # noqa: E402
    build_model,
    build_validation_loader,
    resolve_device,
    seed_everything,
)
from devkit.sparse_ops import SparseConv, SparseLinear  # noqa: E402
from imagenet_data import parquet_manifest  # noqa: E402
from layer_sensitivity import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_write_json,
    file_sha256,
    load_partial_profile,
    load_nm_scheme,
    partial_profile_path,
    validate_scheme_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet18_sparse")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-scheme-file",
        type=Path,
        help=(
            "Measure every candidate while all other layers use this scheme. "
            "Use the checkpoint fine-tuned for the same scheme."
        ),
    )
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--candidate-n", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--layout", choices=("NCHW", "NHWC"), default="NHWC")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-format", choices=("imagefolder", "meta", "parquet"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--val-root", type=Path)
    parser.add_argument("--val-source", type=Path)
    parser.add_argument("--parquet-root", type=Path)
    parser.add_argument("--parquet-pattern", default="data/validation-*.parquet")
    parser.add_argument("--dataset-num-samples", type=int, default=50_000)
    parser.add_argument("--accuracy-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=5_000,
        help="Use the same deterministic prefix for every candidate; 0 means full split.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume candidate measurements from <output>.partial.json. The partial "
            "profile is validated against the current checkpoint, scheme, dataset, "
            "candidate set, and seed before any rows are reused."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[int]:
    candidates = sorted(set(args.candidate_n))
    if args.m <= 0 or not candidates or any(n <= 0 or n > args.m for n in candidates):
        raise ValueError("Every candidate must satisfy 0 < N <= M.")
    if args.m not in candidates:
        raise ValueError("--candidate-n must include dense N=M.")
    if args.input_size <= 0 or args.accuracy_batch_size <= 0 or args.workers < 0:
        raise ValueError("Input/batch size must be positive and workers non-negative.")
    if args.max_eval_samples < 0 or args.dataset_num_samples <= 0:
        raise ValueError("Sample counts are invalid.")
    if args.dataset_format == "imagefolder" and not args.data_root:
        raise ValueError("--data-root is required for imagefolder.")
    if args.dataset_format == "meta" and (not args.val_root or not args.val_source):
        raise ValueError("--val-root and --val-source are required for meta.")
    if args.dataset_format == "parquet" and not args.parquet_root:
        raise ValueError("--parquet-root is required for parquet.")
    return candidates


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() or None


def measure(model: torch.nn.Module, loader, device: torch.device, max_samples: int) -> dict[str, float | int]:
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    samples = 0
    with torch.inference_mode():
        for images, targets in loader:
            remaining = max_samples - samples if max_samples else targets.numel()
            if max_samples and remaining <= 0:
                break
            if max_samples:
                images, targets = images[:remaining], targets[:remaining]
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images)
            total_loss += F.cross_entropy(output, targets, reduction="sum").item()
            predictions = output.topk(min(5, output.shape[1]), dim=1).indices
            matches = predictions.eq(targets.view(-1, 1))
            correct1 += matches[:, :1].sum().item()
            correct5 += matches.sum().item()
            samples += targets.numel()
    if samples == 0:
        raise RuntimeError("Validation loader produced no samples.")
    return {
        "samples": samples,
        "cross_entropy": total_loss / samples,
        "top1_percent": 100.0 * correct1 / samples,
        "top5_percent": 100.0 * correct5 / samples,
    }


def dataset_manifest(args: argparse.Namespace, loader) -> dict[str, Any]:
    result: dict[str, Any] = {
        "format": args.dataset_format,
        "max_eval_samples": args.max_eval_samples,
        "batch_size": args.accuracy_batch_size,
        "workers": args.workers,
        "preprocessing": {
            "resize": round(args.input_size * 256 / 224),
            "center_crop": args.input_size,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
    }
    if args.dataset_format == "parquet":
        result.update(parquet_manifest(args.parquet_root, args.parquet_pattern))
    else:
        result["root"] = str((args.data_root or args.val_root).expanduser().resolve())
        result["dataset_length"] = len(loader.dataset)
        if args.val_source:
            result["source"] = str(args.val_source.expanduser().resolve())
            result["source_sha256"] = file_sha256(args.val_source)
    return result


def main() -> None:
    args = parse_args()
    candidates = validate_args(args)
    output = args.output.expanduser().resolve()
    partial_output = partial_profile_path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite sensitivity profile: {output}")
    if partial_output.exists() and not args.resume:
        raise FileExistsError(
            f"Partial profile already exists: {partial_output}. Use --resume or a new output path."
        )
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # build_model requires these benchmark-only fields but does not use them here.
    args.n = args.m
    args.pretrained = False
    args.scheme_file = None
    model, load_info = build_model(args)
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        raise RuntimeError("Checkpoint did not load exactly.")
    model.to(device).eval()
    loader = build_validation_loader(args, device)
    if loader is None:
        raise RuntimeError("A validation dataset is required.")

    layers = [module for module in model.modules() if isinstance(module, (SparseConv, SparseLinear))]
    names = [module.get_name() for module in layers]
    if not layers or len(set(names)) != len(names):
        raise ValueError("Sparse layer names must be non-empty and unique.")
    base_scheme = (
        load_nm_scheme(args.base_scheme_file)
        if args.base_scheme_file is not None
        else {name: [args.m, args.m] for name in names}
    )
    validate_scheme_layers(base_scheme, names, "Base scheme")
    for module in layers:
        if module.weight.numel() % args.m:
            raise ValueError(f"Layer {module.get_name()} weight count is not divisible by M={args.m}.")
        n, m = base_scheme[module.get_name()]
        if m != args.m or n not in candidates:
            raise ValueError(
                f"Base scheme {module.get_name()}={n}:{m} must use M={args.m} "
                "and an N listed in --candidate-n."
            )
        module.apply_N_M(n, m)

    baseline = measure(model, loader, device, args.max_eval_samples)
    profile = {
        "schema_version": SCHEMA_VERSION,
        "method": (
            "conditioned-one-layer-at-a-time-nm-sensitivity"
            if args.base_scheme_file is not None
            else "one-layer-at-a-time-nm-sensitivity"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "model": {
            "name": args.model,
            "layout": args.layout,
            "input_size": args.input_size,
            "checkpoint": {
                "path": str(args.checkpoint.expanduser().resolve()),
                "sha256": file_sha256(args.checkpoint),
                "missing_keys": [],
                "unexpected_keys": [],
            },
        },
        "dataset": dataset_manifest(args, loader),
        "measurement": {
            "seed": args.seed,
            "baseline": baseline,
            "protocol": (
                "one candidate layer at a time; all other sparse layers use base_scheme"
                if args.base_scheme_file is not None
                else "one sparse layer at a time; all other sparse layers dense"
            ),
            "warning": "Layer deltas are selection estimates, not additive accuracy guarantees.",
        },
        "base_scheme_file": (
            {
                "path": str(args.base_scheme_file.expanduser().resolve()),
                "sha256": file_sha256(args.base_scheme_file),
            }
            if args.base_scheme_file is not None
            else None
        ),
        "base_scheme": base_scheme if args.base_scheme_file is not None else None,
        "candidate_n": candidates,
        "m": args.m,
        "layers": [],
        "progress": {"status": "incomplete", "completed_candidates": 0},
    }
    if args.resume:
        if not partial_output.exists():
            raise FileNotFoundError(f"No partial sensitivity profile to resume: {partial_output}")
        profile = load_partial_profile(partial_output, profile)
        if profile["measurement"]["baseline"]["samples"] != baseline["samples"]:
            raise ValueError("Resume baseline evaluated a different sample count.")
        profile["measurement"]["baseline"] = baseline
        print(
            f"Resuming {profile['progress']['completed_candidates']} candidate(s) "
            f"from {partial_output}"
        )

    saved_layers = {row["name"]: row for row in profile["layers"]}
    profile_layers = []
    for index, module in enumerate(layers, start=1):
        base_n, base_m = base_scheme[module.get_name()]
        saved_layer = saved_layers.get(module.get_name(), {})
        rows = list(saved_layer.get("candidates", []))
        completed_pairs = {(int(row["n"]), int(row["m"])) for row in rows}
        for n in candidates:
            if args.base_scheme_file is not None and n / args.m > base_n / base_m:
                continue
            if (n, args.m) in completed_pairs:
                print(f"[{index}/{len(layers)}] {module.get_name()} {n}:{args.m} resumed")
                continue
            module.apply_N_M(n, args.m)
            observed = measure(model, loader, device, args.max_eval_samples)
            if observed["samples"] != baseline["samples"]:
                raise RuntimeError("Candidate and baseline did not evaluate the same sample count.")
            rows.append(
                {
                    "n": n,
                    "m": args.m,
                    "density": n / args.m,
                    "observed": observed,
                    "sensitivity": {
                        "loss_increase": observed["cross_entropy"] - baseline["cross_entropy"],
                        "top1_drop_percent": baseline["top1_percent"] - observed["top1_percent"],
                        "top5_drop_percent": baseline["top5_percent"] - observed["top5_percent"],
                    },
                }
            )
            print(
                f"[{index}/{len(layers)}] {module.get_name()} {n}:{args.m} "
                f"top1={observed['top1_percent']:.3f}% "
                f"drop={baseline['top1_percent'] - observed['top1_percent']:.3f} pp"
            )
            current_layer = {
                "name": module.get_name(),
                "type": "conv2d" if isinstance(module, SparseConv) else "linear",
                "dense_parameters": module.weight.numel(),
                "candidates": rows,
            }
            remaining_layers = [
                layer for layer in profile_layers if layer["name"] != module.get_name()
            ]
            profile["layers"] = remaining_layers + [current_layer]
            profile["progress"] = {
                "status": "incomplete",
                "completed_candidates": sum(
                    len(layer["candidates"]) for layer in profile["layers"]
                ),
                "last_layer": module.get_name(),
                "last_candidate": [n, args.m],
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(partial_output, profile)
        module.apply_N_M(base_n, base_m)
        profile_layers.append(
            {
                "name": module.get_name(),
                "type": "conv2d" if isinstance(module, SparseConv) else "linear",
                "dense_parameters": module.weight.numel(),
                "candidates": rows,
            }
        )

    profile["layers"] = profile_layers
    profile["progress"] = {
        "status": "complete",
        "completed_candidates": sum(len(layer["candidates"]) for layer in profile_layers),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, profile)
    if partial_output.exists():
        partial_output.unlink()
    print(f"Sensitivity profile: {output}")


if __name__ == "__main__":
    main()
