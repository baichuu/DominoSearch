#!/usr/bin/env python3
"""Benchmark a DominoSearch model before and after N:M pruning.

The JSON output deliberately separates theoretical sparse complexity from measured
runtime.  DominoSearch masks weights in PyTorch but still calls dense operators, so
PyTorch latency is not evidence of FPGA latency.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
MODEL_ROOT = TRAIN_ROOT / "classification_sparsity_level"
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(TRAIN_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    import models
    from devkit.dataset.imagenet_dataset import ImagenetDataset
    from devkit.sparse_ops import SparseConv, SparseLinear
    from imagenet_data import ParquetImageNetDataset
except ImportError as exc:  # pragma: no cover - gives a useful message on a clean host
    raise SystemExit(
        "Missing benchmark dependencies. Install them with: "
        "pip install -r requirements.txt"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure accuracy, model complexity, latency, throughput and memory."
    )
    parser.add_argument("--run-name", default="benchmark")
    parser.add_argument("--model", default="resnet18_sparse")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--scheme-file", type=Path)
    parser.add_argument("--n", type=int, default=1, help="Uniform N (N=M means dense).")
    parser.add_argument("--m", type=int, default=1, help="Uniform M (N=M means dense).")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--layout", choices=("NCHW", "NHWC"), default="NHWC")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1, help="Performance batch size.")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:INDEX")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pruning-method",
        choices=(
            "dense",
            "uniform-nm",
            "domino-mixed-nm",
            "structured-channel",
            "unstructured-magnitude",
        ),
        default="dense",
    )
    parser.add_argument(
        "--density-source",
        choices=("nm", "nonzero"),
        default="nm",
        help="Use N:M ratios or materialized non-zero weights for effective metrics.",
    )
    parser.add_argument(
        "--experiment-status",
        choices=("debug", "candidate", "final"),
        default="debug",
        help="A debug run must not be used as final optimization evidence.",
    )

    parser.add_argument(
        "--dataset-format",
        choices=("none", "imagefolder", "meta", "parquet"),
        default="none",
    )
    parser.add_argument("--data-root", type=Path, help="ImageFolder validation directory.")
    parser.add_argument("--val-root", type=Path, help="Image root for a meta-file dataset.")
    parser.add_argument("--val-source", type=Path, help="Lines formatted as: path class_id.")
    parser.add_argument(
        "--parquet-root",
        type=Path,
        help="Directory containing local ImageNet Parquet shards (Google Drive is supported).",
    )
    parser.add_argument(
        "--parquet-pattern",
        default="data/validation-*.parquet",
        help="Glob relative to --parquet-root for validation shards.",
    )
    parser.add_argument("--dataset-num-samples", type=int, default=50_000)
    parser.add_argument("--accuracy-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 evaluates all.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.n <= args.m:
        raise ValueError("Uniform sparsity must satisfy 0 < N <= M.")
    for name in ("input_size", "batch_size", "iterations", "accuracy_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.warmup < 0 or args.max_eval_samples < 0:
        raise ValueError("--warmup and --max-eval-samples cannot be negative.")
    if args.checkpoint and args.pretrained:
        raise ValueError("Use either --checkpoint or --pretrained, not both.")
    if args.dataset_format == "imagefolder" and not args.data_root:
        raise ValueError("--data-root is required for --dataset-format imagefolder.")
    if args.dataset_format == "meta" and (not args.val_root or not args.val_source):
        raise ValueError("--val-root and --val-source are required for meta datasets.")
    if args.dataset_format == "parquet" and not args.parquet_root:
        raise ValueError("--parquet-root is required for --dataset-format parquet.")
    if args.dataset_num_samples <= 0:
        raise ValueError("--dataset-num-samples must be positive.")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_info() -> dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
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

    status = run_git("status", "--porcelain")
    return {
        "branch": run_git("branch", "--show-current"),
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def load_scheme(path: Path) -> dict[str, list[int]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Scheme file is empty: {path}")
    value = ast.literal_eval(raw.splitlines()[0])
    if not isinstance(value, dict):
        raise ValueError("Scheme must be a Python dictionary on the first line.")
    scheme: dict[str, list[int]] = {}
    for layer, pair in value.items():
        if not isinstance(layer, str) or not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid scheme entry: {layer!r}: {pair!r}")
        n, m = int(pair[0]), int(pair[1])
        if not 0 < n <= m:
            raise ValueError(f"Invalid N:M value for {layer}: {n}:{m}")
        scheme[layer] = [n, m]
    return scheme


def apply_scheme(model: nn.Module, scheme: dict[str, list[int]]) -> None:
    sparse_layers = {
        layer.get_name(): layer
        for layer in model.modules()
        if isinstance(layer, (SparseConv, SparseLinear))
    }
    unknown = sorted(set(scheme) - set(sparse_layers))
    missing = sorted(set(sparse_layers) - set(scheme))
    if unknown:
        raise ValueError(f"Scheme contains unknown layers: {unknown[:5]}")
    if missing:
        raise ValueError(
            f"Scheme is missing {len(missing)} sparse layer(s), for example: {missing[:3]}"
        )
    for name, layer in sparse_layers.items():
        layer.apply_N_M(*scheme[name])


def load_checkpoint(model: nn.Module, path: Path) -> dict[str, list[str]]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a state_dict or contain a 'state_dict' key.")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    load_info = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        raise ValueError(
            "Checkpoint does not exactly match the model. "
            f"Missing keys: {load_info['missing_keys'][:5]}; "
            f"unexpected keys: {load_info['unexpected_keys'][:5]}"
        )
    return load_info


def current_scheme(model: nn.Module) -> dict[str, list[int]]:
    return {
        layer.get_name(): [int(layer.N), int(layer.M)]
        for layer in model.modules()
        if isinstance(layer, (SparseConv, SparseLinear))
    }


def build_model(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    if args.model not in models.__dict__ or not callable(models.__dict__[args.model]):
        available = sorted(name for name, obj in models.__dict__.items() if callable(obj))
        raise ValueError(f"Unknown model {args.model!r}. Available examples: {available[:12]}")
    model = models.__dict__[args.model](
        pretrained=args.pretrained, N=args.n, M=args.m, num_classes=args.num_classes
    )
    if hasattr(model, "set_datalayout"):
        model.set_datalayout(args.layout)
    load_info: dict[str, Any] = {"missing_keys": [], "unexpected_keys": []}
    if args.checkpoint:
        load_info = load_checkpoint(model, args.checkpoint)
    if args.scheme_file:
        apply_scheme(model, load_scheme(args.scheme_file))
    return model, load_info


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure_complexity(
    model: nn.Module, device: torch.device, input_size: int, density_source: str
) -> dict[str, float | int]:
    macs_by_module: dict[nn.Module, int] = {}
    hooks = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        batch = output.shape[0]
        if isinstance(module, nn.Conv2d):
            output_elements = output.numel() // batch
            kernel_ops = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                // module.groups
            )
            macs_by_module[module] = output_elements * kernel_ops
        elif isinstance(module, nn.Linear):
            output_elements = output.numel() // batch
            macs_by_module[module] = output_elements * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(torch.zeros(1, 3, input_size, input_size, device=device))
    for item in hooks:
        item.remove()

    dense_macs = sum(macs_by_module.values())
    effective_macs = 0.0
    for module, module_macs in macs_by_module.items():
        if density_source == "nonzero":
            ratio = torch.count_nonzero(module.weight).item() / module.weight.numel()
        else:
            ratio = (
                module.N / module.M
                if isinstance(module, (SparseConv, SparseLinear))
                else 1.0
            )
        effective_macs += module_macs * ratio

    dense_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    effective_parameters = float(dense_parameters)
    sparse_weight_parameters = 0
    effective_sparse_weight_parameters = 0.0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            count = module.weight.numel()
            sparse_weight_parameters += count
            if density_source == "nonzero":
                effective = torch.count_nonzero(module.weight).item()
            elif isinstance(module, (SparseConv, SparseLinear)):
                effective = count * module.N / module.M
            else:
                effective = count
            effective_sparse_weight_parameters += effective
            effective_parameters -= count - effective

    return {
        "dense_parameters": dense_parameters,
        "trainable_parameters": trainable_parameters,
        "effective_parameters": round(effective_parameters),
        "sparse_weight_parameters": sparse_weight_parameters,
        "effective_sparse_weight_parameters": round(effective_sparse_weight_parameters),
        "parameter_reduction_percent": 100.0 * (1.0 - effective_parameters / dense_parameters),
        "dense_macs_per_sample": dense_macs,
        "effective_macs_per_sample": round(effective_macs),
        "mac_reduction_percent": 100.0 * (1.0 - effective_macs / dense_macs),
        "density_source": density_source,
    }


def build_validation_loader(args: argparse.Namespace, device: torch.device) -> DataLoader | None:
    if args.dataset_format == "none":
        return None
    resize_size = round(args.input_size * 256 / 224)
    transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    if args.dataset_format == "imagefolder":
        dataset = datasets.ImageFolder(str(args.data_root), transform=transform)
    elif args.dataset_format == "meta":
        dataset = ImagenetDataset(str(args.val_root), str(args.val_source), transform=transform)
    else:
        dataset = ParquetImageNetDataset(
            args.parquet_root,
            args.parquet_pattern,
            "validation",
            transform,
            args.dataset_num_samples,
            shuffle=False,
            seed=args.seed,
            shuffle_buffer=1,
        )
    if (
        args.dataset_format != "parquet"
        and args.max_eval_samples
        and args.max_eval_samples < len(dataset)
    ):
        dataset = Subset(dataset, range(args.max_eval_samples))
    return DataLoader(
        dataset,
        batch_size=args.accuracy_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )


def measure_accuracy(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    max_samples: int = 0,
) -> dict[str, Any]:
    if loader is None:
        return {"evaluated": False, "samples": 0, "top1_percent": None, "top5_percent": None}
    correct1 = 0
    correct5 = 0
    samples = 0
    with torch.inference_mode():
        for images, target in loader:
            if max_samples:
                remaining = max_samples - samples
                if remaining <= 0:
                    break
                images = images[:remaining]
                target = target[:remaining]
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            k = min(5, output.shape[1])
            predictions = output.topk(k, dim=1).indices
            matches = predictions.eq(target.view(-1, 1))
            correct1 += matches[:, :1].sum().item()
            correct5 += matches.sum().item()
            samples += target.numel()
    return {
        "evaluated": True,
        "samples": samples,
        "top1_percent": 100.0 * correct1 / samples,
        "top5_percent": 100.0 * correct5 / samples,
    }


def measure_performance(
    model: nn.Module, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    sample = torch.randn(
        args.batch_size, 3, args.input_size, args.input_size, device=device
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None

    latencies: list[float] = []
    with torch.inference_mode():
        for _ in range(args.iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(sample)
                end.record()
                end.synchronize()
                latencies.append(start.elapsed_time(end))
            else:
                start_time = time.perf_counter()
                model(sample)
                latencies.append((time.perf_counter() - start_time) * 1000.0)

    median_ms = statistics.median(latencies)
    result: dict[str, Any] = {
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "batch_size": args.batch_size,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": median_ms,
            "p95": percentile(latencies, 95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "throughput_samples_per_second": args.batch_size * 1000.0 / median_ms,
        "peak_device_memory_bytes": None,
        "incremental_peak_device_memory_bytes": None,
    }
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        result["peak_device_memory_bytes"] = peak
        result["incremental_peak_device_memory_bytes"] = max(0, peak - baseline_memory)
    return result


def environment_info(device: torch.device) -> dict[str, Any]:
    device_name = platform.processor() or platform.machine()
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "device_name": device_name,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }


def print_summary(result: dict[str, Any]) -> None:
    complexity = result["complexity"]
    accuracy = result["accuracy"]
    performance = result["performance"]
    print(f"Run: {result['run_name']}")
    print(
        "Parameters: "
        f"{complexity['dense_parameters']:,} dense -> "
        f"{complexity['effective_parameters']:,} effective "
        f"(-{complexity['parameter_reduction_percent']:.2f}%)"
    )
    print(
        "MACs/sample: "
        f"{complexity['dense_macs_per_sample']:,} dense -> "
        f"{complexity['effective_macs_per_sample']:,} effective "
        f"(-{complexity['mac_reduction_percent']:.2f}%)"
    )
    if accuracy["evaluated"]:
        print(
            f"Accuracy: top-1 {accuracy['top1_percent']:.3f}%, "
            f"top-5 {accuracy['top5_percent']:.3f}% ({accuracy['samples']} samples)"
        )
    else:
        print("Accuracy: not evaluated (synthetic input benchmark only)")
    latency = performance["latency_ms"]
    print(
        f"Latency: median {latency['median']:.3f} ms, p95 {latency['p95']:.3f} ms; "
        f"throughput {performance['throughput_samples_per_second']:.2f} samples/s"
    )
    print(f"JSON: {result['output_file']}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, load_info = build_model(args)
    model = model.to(device).eval()
    loader = build_validation_loader(args, device)
    complexity = measure_complexity(model, device, args.input_size, args.density_source)
    accuracy = measure_accuracy(model, loader, device, args.max_eval_samples)
    performance = measure_performance(model, args, device)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "run_name": args.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": str(output),
        "environment": environment_info(device),
        "source": git_info(),
        "experiment": {
            "status": args.experiment_status,
            "pruning_method": args.pruning_method,
            "seed": args.seed,
        },
        "model": {
            "name": args.model,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_size_bytes": os.path.getsize(args.checkpoint) if args.checkpoint else None,
            "pretrained": args.pretrained,
            "scheme_file": str(args.scheme_file.resolve()) if args.scheme_file else None,
            "uniform_n_m": [args.n, args.m],
            "applied_scheme": current_scheme(model),
            "layout": args.layout,
            "input_shape": [args.batch_size, 3, args.input_size, args.input_size],
            "checkpoint_load": load_info,
        },
        "dataset": {
            "format": args.dataset_format,
            "data_root": str(args.data_root.resolve()) if args.data_root else None,
            "val_root": str(args.val_root.resolve()) if args.val_root else None,
            "val_source": str(args.val_source.resolve()) if args.val_source else None,
            "parquet_root": (
                str(args.parquet_root.expanduser().resolve()) if args.parquet_root else None
            ),
            "parquet_pattern": (
                args.parquet_pattern if args.dataset_format == "parquet" else None
            ),
            "expected_samples": (
                args.dataset_num_samples if args.dataset_format == "parquet" else None
            ),
            "parquet_manifest": (
                loader.dataset.manifest()
                if args.dataset_format == "parquet" and loader is not None
                else None
            ),
            "max_eval_samples": args.max_eval_samples,
        },
        "complexity": complexity,
        "accuracy": accuracy,
        "performance": performance,
        "notes": {
            "mac_definition": "One multiply-accumulate is counted as one MAC.",
            "effective_metrics": "Theoretical N:M non-zero count, not compressed file size.",
            "runtime_warning": (
                "This repository masks weights then calls dense PyTorch operators. "
                "Measured runtime is host-framework runtime, not projected FPGA runtime."
            ),
        },
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(result)


if __name__ == "__main__":
    main()
