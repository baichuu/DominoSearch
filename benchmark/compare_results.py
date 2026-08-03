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
    parser.add_argument("--markdown", type=Path, help="Save a presentation-ready report.")
    parser.add_argument("--title", default="DominoSearch pruning experiment report")
    return parser.parse_args()


def value(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def make_row(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    complexity = result["complexity"]
    accuracy = result["accuracy"]
    performance = result["performance"]
    memory = performance.get("peak_device_memory_bytes")
    experiment = result.get("experiment", {})
    source = result.get("source", {})
    baseline_complexity = baseline["complexity"]
    baseline_accuracy = baseline["accuracy"]
    baseline_performance = baseline["performance"]
    top1 = accuracy.get("top1_percent")
    baseline_top1 = baseline_accuracy.get("top1_percent")
    median = performance["latency_ms"]["median"]
    baseline_median = baseline_performance["latency_ms"]["median"]
    return {
        "run": result["run_name"],
        "method": experiment.get("pruning_method"),
        "status": experiment.get("status"),
        "branch": source.get("branch"),
        "top1_%": accuracy.get("top1_percent"),
        "top5_%": accuracy.get("top5_percent"),
        "delta_top1_pp": (
            top1 - baseline_top1 if top1 is not None and baseline_top1 is not None else None
        ),
        "effective_params_M": complexity["effective_parameters"] / 1e6,
        "parameter_reduction_%": 100.0 * (
            1.0
            - complexity["effective_parameters"]
            / baseline_complexity["effective_parameters"]
        ),
        "effective_MACs_G": complexity["effective_macs_per_sample"] / 1e9,
        "MAC_reduction_%": complexity["mac_reduction_percent"],
        "median_ms": median,
        "p95_ms": performance["latency_ms"]["p95"],
        "host_speedup_x": baseline_median / median,
        "samples_per_s": performance["throughput_samples_per_second"],
        "peak_memory_MB": memory / 1e6 if memory is not None else None,
        "eval_samples": accuracy.get("samples"),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    labels = {
        "run": "Run",
        "method": "Method",
        "status": "Status",
        "branch": "Branch",
        "top1_%": "Top-1 %",
        "top5_%": "Top-5 %",
        "delta_top1_pp": "ΔTop-1 pp",
        "effective_params_M": "Eff. params M",
        "parameter_reduction_%": "Param reduction %",
        "effective_MACs_G": "Eff. MACs G",
        "MAC_reduction_%": "MAC reduction %",
        "median_ms": "Median ms",
        "p95_ms": "P95 ms",
        "host_speedup_x": "Host speedup ×",
        "samples_per_s": "Samples/s",
        "peak_memory_MB": "Peak MB",
        "eval_samples": "Eval samples",
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
            if key in {"run", "method", "status", "branch"} and item:
                rendered.append(str(item))
            elif key == "eval_samples" and item is not None:
                rendered.append(str(int(item)))
            else:
                rendered.append(value(item))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def find_baseline(results: list[dict[str, Any]]) -> dict[str, Any]:
    for result in results:
        if result.get("experiment", {}).get("pruning_method") == "dense":
            return result
    raise ValueError("At least one result with pruning_method='dense' is required.")


def comparison_warnings(results: list[dict[str, Any]], baseline: dict[str, Any]) -> list[str]:
    warnings = []
    baseline_dataset = baseline.get("dataset", {})
    baseline_model = baseline.get("model", {})
    baseline_environment = baseline.get("environment", {})
    baseline_performance = baseline.get("performance", {})
    expected = {
        "dataset format": baseline_dataset.get("format"),
        "evaluated samples": baseline.get("accuracy", {}).get("samples"),
        "input shape": baseline_model.get("input_shape"),
        "device": baseline_environment.get("device_name"),
        "performance batch size": baseline_performance.get("batch_size"),
        "warm-up iterations": baseline_performance.get("warmup_iterations"),
        "measured iterations": baseline_performance.get("measured_iterations"),
    }
    for result in results:
        actual = {
            "dataset format": result.get("dataset", {}).get("format"),
            "evaluated samples": result.get("accuracy", {}).get("samples"),
            "input shape": result.get("model", {}).get("input_shape"),
            "device": result.get("environment", {}).get("device_name"),
            "performance batch size": result.get("performance", {}).get("batch_size"),
            "warm-up iterations": result.get("performance", {}).get("warmup_iterations"),
            "measured iterations": result.get("performance", {}).get("measured_iterations"),
        }
        for field, expected_value in expected.items():
            if actual[field] != expected_value:
                warnings.append(
                    f"{result['run_name']}: {field}={actual[field]!r}, "
                    f"dense baseline={expected_value!r}."
                )
        if result.get("source", {}).get("dirty"):
            warnings.append(f"{result['run_name']}: worktree was dirty.")
        if result.get("experiment", {}).get("status") == "debug":
            warnings.append(f"{result['run_name']}: experiment status is debug.")
        checkpoint_load = result.get("model", {}).get("checkpoint_load", {})
        if checkpoint_load.get("missing_keys") or checkpoint_load.get("unexpected_keys"):
            warnings.append(f"{result['run_name']}: checkpoint did not load exactly.")
    return warnings


def markdown_report(
    title: str,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> str:
    warnings = comparison_warnings(results, baseline)
    baseline_source = baseline.get("source", {})
    baseline_dataset = baseline.get("dataset", {})
    baseline_environment = baseline.get("environment", {})
    validity = (
        "Các điều kiện so sánh chính khớp với dense baseline."
        if not warnings
        else "Có khác biệt điều kiện; các dòng liên quan chưa phải bằng chứng so sánh cuối cùng."
    )
    warning_lines = [f"- {warning}" for warning in warnings] or ["- Không phát hiện khác biệt."]
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Phạm vi",
            "",
            "Báo cáo này chỉ so sánh pruning. Effective parameter/MAC là chỉ số lý thuyết; "
            "latency Colab là runtime của PyTorch trên host và không chứng minh tốc độ FPGA.",
            "",
            "## Dense baseline và môi trường",
            "",
            f"- Run: `{baseline['run_name']}`",
            f"- Branch/commit: `{baseline_source.get('branch')}` / `{baseline_source.get('commit')}`",
            f"- Dataset: `{baseline_dataset.get('format')}`, "
            f"{baseline.get('accuracy', {}).get('samples')} validation samples",
            f"- Device: `{baseline_environment.get('device_name')}`",
            f"- PyTorch/CUDA: `{baseline_environment.get('torch')}` / "
            f"`{baseline_environment.get('cuda')}`",
            "",
            "## Kết quả",
            "",
            markdown_table(rows),
            "",
            "## Kiểm tra tính so sánh",
            "",
            validity,
            "",
            *warning_lines,
            "",
            "## Cách diễn giải",
            "",
            "- ΔTop-1 được tính so với dense baseline; giá trị âm là mất accuracy.",
            "- Host speedup chỉ có ý nghĩa khi device, batch size, warm-up và iterations giống nhau.",
            "- Chỉ kết luận target-hardware speedup sau khi đo trực tiếp latency, throughput, "
            "power, BRAM, DSP, LUT và bandwidth trên board/FPGA.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    baseline = find_baseline(results)
    rows = [make_row(result, baseline) for result in results]
    print(markdown_table(rows))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            markdown_report(args.title, rows, results, baseline), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
