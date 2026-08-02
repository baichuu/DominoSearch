#!/usr/bin/env python3
"""Train a reproducible dense ResNet baseline on CIFAR-100 using CPU or CUDA."""

from __future__ import annotations

import argparse
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

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick dense CIFAR-100 baseline.")
    parser.add_argument("--model", default="resnet20_cifar_dense")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--split-file", type=Path, default=Path("data/cifar100-quick-split-seed42.json")
    )
    parser.add_argument(
        "--checkpoint-output", type=Path, default=Path("runs/cifar100-dense-quick.pth")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/cifar100-dense-quick.json")
    )
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-iterations", type=int, default=30)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "train_samples",
        "val_samples",
        "epochs",
        "batch_size",
        "threads",
        "latency_iterations",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.workers < 0 or args.latency_warmup < 0:
        raise ValueError("--workers and --latency-warmup cannot be negative.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(requested)


def create_or_load_split(
    path: Path, train_size: int, val_size: int, train_total: int, val_total: int, seed: int
) -> tuple[list[int], list[int]]:
    if train_size > train_total or val_size > val_total:
        raise ValueError(
            f"Requested subset exceeds CIFAR-100: train {train_size}/{train_total}, "
            f"validation {val_size}/{val_total}."
        )
    path = path.expanduser().resolve()
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        if split.get("seed") != seed:
            raise ValueError(f"Split seed mismatch in {path}.")
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        if len(train_indices) != train_size or len(val_indices) != val_size:
            raise ValueError(
                "Existing split size differs from --train-samples/--val-samples. "
                "Use another --split-file instead of overwriting it."
            )
        return train_indices, val_indices

    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(train_total, generator=generator)[:train_size].tolist()
    val_indices = torch.randperm(val_total, generator=generator)[:val_size].tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": "CIFAR100",
                "seed": seed,
                "train_indices": train_indices,
                "val_indices": val_indices,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return train_indices, val_indices


def build_loaders(args: argparse.Namespace, device: torch.device):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    root = args.data_root.expanduser().resolve()
    train_dataset = datasets.CIFAR100(
        root=str(root), train=True, transform=train_transform, download=args.download
    )
    val_dataset = datasets.CIFAR100(
        root=str(root), train=False, transform=val_transform, download=args.download
    )
    train_indices, val_indices = create_or_load_split(
        args.split_file,
        args.train_samples,
        args.val_samples,
        len(train_dataset),
        len(val_dataset),
        args.seed,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **common,
    )
    val_loader = DataLoader(Subset(val_dataset, val_indices), shuffle=False, **common)
    return train_loader, val_loader


def build_model(name: str) -> nn.Module:
    constructor = models.__dict__.get(name)
    if not callable(constructor):
        raise ValueError(f"Unknown model constructor: {name}")
    return constructor(num_classes=100)


def accuracy_counts(output: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    predictions = output.topk(5, dim=1).indices
    matches = predictions.eq(target.view(-1, 1))
    return int(matches[:, :1].sum().item()), int(matches.sum().item())


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    samples = 0
    start = time.perf_counter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        batch = targets.numel()
        total_loss += loss.item() * batch
        correct += accuracy_counts(output, targets)[0]
        samples += batch
    return {
        "loss": total_loss / samples,
        "top1_percent": 100.0 * correct / samples,
        "seconds": time.perf_counter() - start,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    samples = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, targets)
            batch = targets.numel()
            top1, top5 = accuracy_counts(output, targets)
            total_loss += loss.item() * batch
            correct1 += top1
            correct5 += top5
            samples += batch
    return {
        "loss": total_loss / samples,
        "top1_percent": 100.0 * correct1 / samples,
        "top5_percent": 100.0 * correct5 / samples,
        "samples": samples,
    }


def measure_macs(model: nn.Module, device: torch.device) -> int:
    macs = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal macs
        batch = output.shape[0]
        if isinstance(module, nn.Conv2d):
            kernel = math.prod(module.kernel_size) * module.in_channels // module.groups
            macs += output.numel() // batch * kernel
        elif isinstance(module, nn.Linear):
            macs += output.numel() // batch * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, 32, 32, device=device))
    for handle in handles:
        handle.remove()
    return macs


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure_latency(model, args, device):
    model.eval()
    sample = torch.randn(1, 3, 32, 32, device=device)
    with torch.inference_mode():
        for _ in range(args.latency_warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        values = []
        for _ in range(args.latency_iterations):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
    median = statistics.median(values)
    return {
        "iterations": args.latency_iterations,
        "median_ms": median,
        "p95_ms": percentile(values, 95),
        "mean_ms": statistics.fmean(values),
        "throughput_samples_per_second": 1000.0 / median,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    torch.set_num_threads(min(args.threads, os.cpu_count() or args.threads))
    device = resolve_device(args.device)
    model = build_model(args.model).to(device)
    train_loader, val_loader = build_loaders(args, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_top1 = -1.0
    best_state = None
    run_start = time.perf_counter()
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append(
            {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        )
        if val_metrics["top1_percent"] > best_top1:
            best_top1 = val_metrics["top1_percent"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train top1={train_metrics['top1_percent']:.2f}% "
            f"val top1={val_metrics['top1_percent']:.2f}% "
            f"val top5={val_metrics['top5_percent']:.2f}% "
            f"time={train_metrics['seconds']:.1f}s"
        )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state.")
    model.load_state_dict(best_state)
    final_validation = evaluate(model, val_loader, criterion, device)
    macs = measure_macs(model, device)
    latency = measure_latency(model, args, device)
    checkpoint_output = args.checkpoint_output.expanduser().resolve()
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model": args.model,
            "num_classes": 100,
            "best_top1": best_top1,
            "seed": args.seed,
            "status": "debug",
        },
        checkpoint_output,
    )

    source_status = git_value("status", "--porcelain")
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_name": "cifar100-dense-quick-baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "status": "debug",
            "pruning_method": "dense",
            "seed": args.seed,
            "purpose": "pipeline smoke test; not final pruning evidence",
        },
        "source": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(source_status) if source_status is not None else None,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "cpu": platform.processor(),
            "threads": torch.get_num_threads(),
        },
        "model": {
            "name": args.model,
            "checkpoint": str(checkpoint_output),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "macs_per_sample": macs,
            "input_shape": [1, 3, 32, 32],
        },
        "dataset": {
            "name": "CIFAR100",
            "train_samples": args.train_samples,
            "validation_samples": args.val_samples,
            "split_file": str(args.split_file.expanduser().resolve()),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "total_seconds": time.perf_counter() - run_start,
            "history": history,
        },
        "accuracy": final_validation,
        "performance": latency,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Best dense checkpoint: {checkpoint_output}")
    print(f"Debug baseline JSON: {output}")


if __name__ == "__main__":
    main()
