#!/usr/bin/env python3
"""Generate an exact, validated uniform N:M scheme for DominoSearch models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "train"
MODEL_ROOT = TRAIN_ROOT / "classification_sparsity_level"
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(TRAIN_ROOT))

try:
    import models
    from devkit.sparse_ops import SparseConv, SparseLinear
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install them with: pip install -r requirements.txt"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a uniform N:M scheme.")
    parser.add_argument("--model", default="resnet18_sparse")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--keep-first-dense", action="store_true")
    parser.add_argument("--keep-last-dense", action="store_true")
    parser.add_argument("--keep-1x1-dense", action="store_true")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="LAYER",
        help="Keep an exact sparse-layer name dense; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def validate_ratio(n: int, m: int) -> None:
    if not 0 < n <= m:
        raise ValueError(f"Invalid N:M ratio {n}:{m}; expected 0 < N <= M.")


def build_model(name: str, num_classes: int, m: int):
    constructor = models.__dict__.get(name)
    if not callable(constructor):
        raise ValueError(f"Unknown model constructor: {name}")
    return constructor(pretrained=False, N=m, M=m, num_classes=num_classes)


def generate(args: argparse.Namespace) -> tuple[dict[str, list[int]], dict[str, Any]]:
    validate_ratio(args.n, args.m)
    model = build_model(args.model, args.num_classes, args.m)
    layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, (SparseConv, SparseLinear))
    ]
    if not layers:
        raise ValueError(f"Model {args.model} contains no supported sparse layers.")

    known_names = {layer.get_name() for layer in layers}
    unknown_exclusions = sorted(set(args.exclude) - known_names)
    if unknown_exclusions:
        raise ValueError(f"Unknown --exclude layer(s): {unknown_exclusions}")

    scheme: dict[str, list[int]] = {}
    protected: list[str] = []
    dense_weight_count = 0
    effective_weight_count = 0.0
    for index, layer in enumerate(layers):
        name = layer.get_name()
        keep_dense = (
            (args.keep_first_dense and index == 0)
            or (args.keep_last_dense and index == len(layers) - 1)
            or (
                args.keep_1x1_dense
                and isinstance(layer, SparseConv)
                and tuple(layer.kernel_size) == (1, 1)
            )
            or name in args.exclude
        )
        layer_n = args.m if keep_dense else args.n
        if layer.weight.numel() % args.m != 0:
            raise ValueError(
                f"Layer {name} has {layer.weight.numel()} weights, not divisible by M={args.m}."
            )
        scheme[name] = [layer_n, args.m]
        if keep_dense:
            protected.append(name)
        count = layer.weight.numel()
        dense_weight_count += count
        effective_weight_count += count * layer_n / args.m

    manifest = {
        "schema_version": 1,
        "method": "uniform-nm",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "model": args.model,
        "num_classes": args.num_classes,
        "requested_n_m": [args.n, args.m],
        "protected_layers": protected,
        "layer_count": len(layers),
        "dense_sparse_layer_weights": dense_weight_count,
        "effective_sparse_layer_weights": round(effective_weight_count),
        "sparsity_percent": 100.0 * (1.0 - effective_weight_count / dense_weight_count),
        "scheme_file": str(args.output.expanduser().resolve()),
    }
    return scheme, manifest


def main() -> None:
    args = parse_args()
    scheme, manifest = generate(args)
    output = args.output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output.with_suffix(output.suffix + ".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(repr(scheme) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Generated {manifest['layer_count']} layer entries: {output}")
    print(f"Theoretical sparse-layer sparsity: {manifest['sparsity_percent']:.2f}%")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
