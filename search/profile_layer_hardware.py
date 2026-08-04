#!/usr/bin/env python3
"""Benchmark every sparse layer/configuration and create a hardware profile."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_ROOT.parent
TRAIN_ROOT = REPO_ROOT / "train"
MODEL_ROOT = TRAIN_ROOT / "classification_sparsity_level"
sys.path.insert(0, str(TRAIN_ROOT))
sys.path.insert(0, str(MODEL_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import models  # noqa: E402
from devkit.sparse_ops import SparseConv, SparseLinear  # noqa: E402
from hardware_cost import (  # noqa: E402
    DEFAULT_WEIGHTS,
    SCHEMA_VERSION,
    file_sha256,
    fit_cost_predictor,
    validate_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile layer-wise N:M cost on one target device."
    )
    parser.add_argument("--model", default="resnet18_sparse")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--candidate-n", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--layout", choices=("NCHW", "NHWC"), default="NHWC")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--latency-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--bandwidth-weight", type=float, default=0.0)
    parser.add_argument("--memory-weight", type=float, default=0.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA profiling requested but CUDA is unavailable.")
    return device


def load_checkpoint_exact(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict or contain state_dict.")
    state = {name.removeprefix("module."): value for name, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint does not exactly match profile model. Missing: {}; unexpected: {}".format(
                list(incompatible.missing_keys)[:5],
                list(incompatible.unexpected_keys)[:5],
            )
        )
    return {
        "path": str(path.expanduser().resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "missing_keys": [],
        "unexpected_keys": [],
    }


def git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def benchmark_module(
    module: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            module(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        per_operation_ms = []
        peak_bytes = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    module(sample)
                end.record()
                torch.cuda.synchronize(device)
                elapsed_ms = start.elapsed_time(end)
                peak_bytes.append(max(0, torch.cuda.max_memory_allocated(device) - baseline))
            else:
                started = time.perf_counter()
                for _ in range(iterations):
                    module(sample)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                peak_bytes.append(0)
            per_operation_ms.append(elapsed_ms / iterations)
    return {
        "latency_ms": statistics.median(per_operation_ms),
        "latency_p95_ms": percentile(per_operation_ms, 0.95),
        "peak_temporary_bytes": max(peak_bytes),
    }


def dense_macs(module: torch.nn.Module, input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> int:
    if isinstance(module, SparseConv):
        kernel_h, kernel_w = module.kernel_size
        output_elements = int(torch.tensor(output_shape).prod().item())
        return output_elements * (module.in_channels // module.groups) * kernel_h * kernel_w
    return int(torch.tensor(output_shape).prod().item()) * module.in_features


def hardware_manifest(device: torch.device) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        manifest.update(
            {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    else:
        manifest["name"] = platform.processor() or platform.machine()
    return manifest


def main() -> None:
    args = parse_args()
    if args.m <= 0:
        raise ValueError("--m must be positive.")
    candidates = sorted(set(args.candidate_n))
    if not candidates or any(n <= 0 or n > args.m for n in candidates):
        raise ValueError("Every candidate N must satisfy 0 < N <= M.")
    if args.m not in candidates:
        raise ValueError("--candidate-n must include dense N=M.")
    if args.batch_size <= 0 or args.input_size <= 0:
        raise ValueError("Batch size and input size must be positive.")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("Warmup cannot be negative; iterations/repeats must be positive.")
    weights = validate_weights(
        {
            "latency_ms": args.latency_weight,
            "energy_mj": args.energy_weight,
            "bandwidth_bytes": args.bandwidth_weight,
            "memory_bytes": args.memory_weight,
        }
    )
    if weights["energy_mj"] > 0.0:
        raise ValueError(
            "This profiler does not claim layer energy measurements. Keep --energy-weight 0 "
            "until a synchronized board power sensor is integrated."
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    model = models.__dict__[args.model](pretrained=False, N=args.m, M=args.m, search=False)
    if hasattr(model, "set_datalayout"):
        model.set_datalayout(args.layout)
    checkpoint_manifest = None
    if args.checkpoint:
        checkpoint_manifest = load_checkpoint_exact(model, args.checkpoint)
    model.to(device).eval()

    sparse_modules = [
        module for module in model.modules() if isinstance(module, (SparseConv, SparseLinear))
    ]
    if not sparse_modules:
        raise ValueError("Model contains no SparseConv/SparseLinear layers.")
    captures: dict[str, dict[str, Any]] = {}
    handles = []
    for module in sparse_modules:
        def capture(current, inputs, output, name=module.get_name()):
            captures[name] = {
                "input": inputs[0].detach(),
                "input_shape": tuple(inputs[0].shape),
                "output_shape": tuple(output.shape),
            }
        handles.append(module.register_forward_hook(capture))
    sample = torch.randn(
        args.batch_size, 3, args.input_size, args.input_size, device=device
    )
    with torch.inference_mode():
        model(sample)
    for handle in handles:
        handle.remove()
    expected = {module.get_name() for module in sparse_modules}
    if set(captures) != expected:
        raise RuntimeError("Failed to capture every sparse layer input/output shape.")

    layers = []
    for layer_index, module in enumerate(sparse_modules, start=1):
        name = module.get_name()
        capture = captures[name]
        layer_input = capture["input"]
        input_shape = capture["input_shape"]
        output_shape = capture["output_shape"]
        element_bytes = module.weight.element_size()
        if module.weight.numel() % args.m:
            raise ValueError(f"Layer {name} weight count is not divisible by M={args.m}.")
        if isinstance(module, SparseConv):
            kernel_h, kernel_w = module.kernel_size
            features = {
                "batch_size": args.batch_size,
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "input_elements": layer_input.numel(),
                "output_elements": int(torch.tensor(output_shape).prod().item()),
                "kernel_elements": kernel_h * kernel_w,
                "groups": module.groups,
                "stride": list(module.stride),
                "padding": list(module.padding),
                "dilation": list(module.dilation),
                "dense_parameters": module.weight.numel(),
                "dense_macs": dense_macs(module, input_shape, output_shape),
            }
            layer_type = "conv2d"
        else:
            features = {
                "batch_size": args.batch_size,
                "in_channels": module.in_features,
                "out_channels": module.out_features,
                "input_elements": layer_input.numel(),
                "output_elements": int(torch.tensor(output_shape).prod().item()),
                "kernel_elements": 1,
                "groups": 1,
                "stride": [1, 1],
                "padding": [0, 0],
                "dilation": [1, 1],
                "dense_parameters": module.weight.numel(),
                "dense_macs": dense_macs(module, input_shape, output_shape),
            }
            layer_type = "linear"

        candidate_rows = []
        for n in candidates:
            module.apply_N_M(n, args.m)
            measured = benchmark_module(
                module,
                layer_input,
                device,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            density = n / args.m
            bias_elements = 0 if module.bias is None else module.bias.numel()
            effective_weight_elements = round(module.weight.numel() * density)
            bandwidth_bytes = (
                layer_input.numel()
                + int(torch.tensor(output_shape).prod().item())
                + effective_weight_elements
                + bias_elements
            ) * element_bytes
            memory_bytes = (effective_weight_elements + bias_elements) * element_bytes
            candidate_rows.append(
                {
                    "n": n,
                    "m": args.m,
                    "density": density,
                    "metrics": {
                        "latency_ms": measured["latency_ms"],
                        "energy_mj": None,
                        "bandwidth_bytes": bandwidth_bytes,
                        "memory_bytes": memory_bytes,
                    },
                    "diagnostics": {
                        "latency_p95_ms": measured["latency_p95_ms"],
                        "peak_temporary_bytes": measured["peak_temporary_bytes"],
                    },
                }
            )
            print(
                f"[{layer_index}/{len(sparse_modules)}] {name} {n}:{args.m} "
                f"median={measured['latency_ms']:.6f} ms"
            )
        layers.append(
            {
                "name": name,
                "type": layer_type,
                "input_shape": list(input_shape),
                "output_shape": list(output_shape),
                "features": features,
                "candidates": candidate_rows,
            }
        )

    profile = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "layer-wise-nm-hardware-profile",
        "source": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "model": {
            "name": args.model,
            "input_shape": [args.batch_size, 3, args.input_size, args.input_size],
            "layout": args.layout,
            "batch_size": args.batch_size,
            "checkpoint": checkpoint_manifest,
        },
        "hardware": hardware_manifest(device),
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
            "latency": "median of repeated synchronized per-operation block means",
            "energy": "not measured",
            "bandwidth": "estimated input + output + effective weights + bias bytes",
            "memory": "estimated effective weights + bias bytes",
        },
        "cost_definition": {
            "formula": "sum(weight[metric] * layer_metric / total_dense_metric)",
            "weights": weights,
            "notes": "Only metrics with non-zero weights affect search cost.",
        },
        "candidate_n": candidates,
        "m": args.m,
        "layers": layers,
    }
    profile["predictor"] = fit_cost_predictor(profile, ridge=args.ridge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite hardware profile: {args.output}")
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(profile, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(f"Hardware profile: {args.output}")
    print(json.dumps(profile["predictor"]["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
