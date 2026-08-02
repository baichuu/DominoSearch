#!/usr/bin/env python3
"""Compare JSON files emitted by benchmark_model.py."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a baseline-vs-optimized table.")
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--csv", type=Path, help="Optionally save the same rows as CSV.")
    return parser.parse_args()


def value(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def make_row(result: dict[str, Any]) -> dict[str, Any]:
    complexity = result["complexity"]
    accuracy = result["accuracy"]
    performance = result["performance"]
    memory = performance.get("peak_device_memory_bytes")
    experiment = result.get("experiment", {})
    source = result.get("source", {})
    return {
        "run": result["run_name"],
        "method": experiment.get("pruning_method"),
        "status": experiment.get("status"),
        "branch": source.get("branch"),
        "top1_%": accuracy.get("top1_percent"),
        "top5_%": accuracy.get("top5_percent"),
        "effective_params_M": complexity["effective_parameters"] / 1e6,
        "effective_MACs_G": complexity["effective_macs_per_sample"] / 1e9,
        "MAC_reduction_%": complexity["mac_reduction_percent"],
        "median_ms": performance["latency_ms"]["median"],
        "p95_ms": performance["latency_ms"]["p95"],
        "samples_per_s": performance["throughput_samples_per_second"],
        "peak_memory_MB": memory / 1e6 if memory is not None else None,
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    labels = {
        "run": "Run",
        "method": "Method",
        "status": "Status",
        "branch": "Branch",
        "top1_%": "Top-1 %",
        "top5_%": "Top-5 %",
        "effective_params_M": "Eff. params M",
        "effective_MACs_G": "Eff. MACs G",
        "MAC_reduction_%": "MAC reduction %",
        "median_ms": "Median ms",
        "p95_ms": "P95 ms",
        "samples_per_s": "Samples/s",
        "peak_memory_MB": "Peak MB",
    }
    keys = list(labels)
    lines = [
        "| " + " | ".join(labels[key] for key in keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        rendered = []
        for key in keys:
            item = row[key]
            rendered.append(
                str(item) if key in {"run", "method", "status", "branch"} and item else value(item)
            )
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    rows = [make_row(result) for result in results]
    print(markdown_table(rows))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
