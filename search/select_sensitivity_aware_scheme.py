#!/usr/bin/env python3
"""Select mixed N:M using measured hardware cost and layer sensitivity."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

try:  # Support both ``python search/file.py`` and package imports in tests.
    from .hardware_cost import HardwareCostModel
    from .layer_sensitivity import LayerSensitivityProfile, SENSITIVITY_METRICS
except ImportError:  # pragma: no cover - exercised by the CLI form
    from hardware_cost import HardwareCostModel
    from layer_sensitivity import LayerSensitivityProfile, SENSITIVITY_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-profile", type=Path, required=True)
    parser.add_argument("--sensitivity-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-reduction", type=float, required=True)
    parser.add_argument(
        "--target-metric", choices=("hardware-cost", "macs"), default="hardware-cost"
    )
    parser.add_argument(
        "--sensitivity-metric", choices=SENSITIVITY_METRICS, default="loss_increase"
    )
    parser.add_argument(
        "--max-estimated-sensitivity",
        type=float,
        default=None,
        help="Reject a scheme if its additive sensitivity estimate exceeds this value.",
    )
    parser.add_argument("--minimum-n", type=int, default=1)
    parser.add_argument("--protect-first-conv", action="store_true")
    parser.add_argument("--protect-linear", action="store_true")
    parser.add_argument("--latency-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--bandwidth-weight", type=float, default=None)
    parser.add_argument("--memory-weight", type=float, default=None)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False
    ).stdout.strip()


def choose_scheme(
    hardware: HardwareCostModel,
    sensitivity: LayerSensitivityProfile,
    target_reduction: float,
    target_metric: str,
    sensitivity_metric: str,
    minimum_n: int = 1,
    protect_first_conv: bool = False,
    protect_linear: bool = False,
) -> tuple[dict[str, list[int]], float]:
    """Multiple-choice Pareto search minimizing additive measured sensitivity."""

    if not 0.0 < target_reduction < 1.0:
        raise ValueError("Target reduction must be in (0, 1).")
    if minimum_n <= 0:
        raise ValueError("--minimum-n must be positive.")
    if target_metric not in {"hardware-cost", "macs"}:
        raise ValueError("Unknown target metric.")

    layer_items = list(hardware.layers.items())
    all_pairs = sorted(
        {
            (int(candidate["n"]), int(candidate["m"]))
            for layer in hardware.layers.values()
            for candidate in layer["candidates"]
        }
    )
    sensitivity.validate_candidates(hardware.layers, all_pairs)
    first_conv = next(
        (name for name, layer in layer_items if layer.get("type") == "conv2d"), None
    )

    dense_scheme: dict[str, list[int]] = {}
    dense_total = 0.0
    choices_by_layer: list[list[tuple[int, int, float, float]]] = []
    for name, layer in layer_items:
        pairs = sorted(
            {(int(row["n"]), int(row["m"])) for row in layer["candidates"]}
        )
        dense_pair = max(pairs, key=lambda pair: pair[0] / pair[1])
        if dense_pair[0] != dense_pair[1]:
            raise ValueError(f"Hardware profile has no dense N=M candidate for {name}.")
        dense_scheme[name] = list(dense_pair)
        dense_value = (
            hardware.cost(name, *dense_pair)
            if target_metric == "hardware-cost"
            else float(layer["features"]["dense_macs"])
        )
        dense_total += dense_value
        protected = (protect_first_conv and name == first_conv) or (
            protect_linear and layer.get("type") == "linear"
        )
        candidates = []
        for n, m in pairs:
            if n < minimum_n or (protected and n != m):
                continue
            candidate_value = (
                hardware.cost(name, n, m)
                if target_metric == "hardware-cost"
                else float(layer["features"]["dense_macs"]) * n / m
            )
            saving = dense_value - candidate_value
            # Negative deltas are sampling noise, not credit that can cancel damage elsewhere.
            penalty = max(0.0, sensitivity.value(name, n, m, sensitivity_metric))
            candidates.append((n, m, saving, penalty))
        if not candidates:
            raise ValueError(f"No allowed candidate remains for layer {name}.")
        choices_by_layer.append(candidates)

    required_saving = target_reduction * dense_total
    # saving, sensitivity, density loss, tuple of (N, M)
    frontier: list[tuple[float, float, float, tuple[tuple[int, int], ...]]] = [
        (0.0, 0.0, 0.0, ())
    ]
    for candidates in choices_by_layer:
        expanded = [
            (
                saving + candidate_saving,
                penalty + candidate_penalty,
                density_loss + (1.0 - n / m),
                choices + ((n, m),),
            )
            for saving, penalty, density_loss, choices in frontier
            for n, m, candidate_saving, candidate_penalty in candidates
        ]
        expanded.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        frontier = []
        best_quality: tuple[float, float] | None = None
        for row in expanded:
            quality = (row[1], row[2])
            if best_quality is None or quality < best_quality:
                frontier.append(row)
                best_quality = quality

    feasible = [row for row in frontier if row[0] + 1e-12 >= required_saving]
    if not feasible:
        maximum = max(row[0] for row in frontier) / dense_total
        raise ValueError(
            f"Target reduction {target_reduction:.6f} is infeasible with constraints; "
            f"maximum is {maximum:.6f}."
        )
    selected = min(
        feasible,
        key=lambda row: (row[1], row[0] - required_saving, row[2], row[3]),
    )
    scheme = {
        name: [n, m]
        for (name, _), (n, m) in zip(layer_items, selected[3], strict=True)
    }
    return scheme, selected[1]


def main() -> None:
    args = parse_args()
    overrides = {
        "latency_ms": args.latency_weight,
        "energy_mj": args.energy_weight,
        "bandwidth_bytes": args.bandwidth_weight,
        "memory_bytes": args.memory_weight,
    }
    supplied = [value is not None for value in overrides.values()]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide all four cost weights or none of them.")
    weights = None if not any(supplied) else {key: float(value) for key, value in overrides.items()}
    hardware = HardwareCostModel(args.hardware_profile, mode="lookup", weights=weights)
    sensitivity = LayerSensitivityProfile(args.sensitivity_profile)
    scheme, estimated_sensitivity = choose_scheme(
        hardware,
        sensitivity,
        args.target_reduction,
        args.target_metric,
        args.sensitivity_metric,
        args.minimum_n,
        args.protect_first_conv,
        args.protect_linear,
    )
    if (
        args.max_estimated_sensitivity is not None
        and estimated_sensitivity > args.max_estimated_sensitivity
    ):
        raise ValueError(
            f"Best feasible scheme sensitivity {estimated_sensitivity:.6f} exceeds "
            f"budget {args.max_estimated_sensitivity:.6f}."
        )

    dense_parameters = sum(float(row["features"]["dense_parameters"]) for row in hardware.layers.values())
    dense_macs = sum(float(row["features"]["dense_macs"]) for row in hardware.layers.values())
    effective_parameters = sum(
        float(row["features"]["dense_parameters"]) * scheme[name][0] / scheme[name][1]
        for name, row in hardware.layers.items()
    )
    effective_macs = sum(
        float(row["features"]["dense_macs"]) * scheme[name][0] / scheme[name][1]
        for name, row in hardware.layers.items()
    )
    achieved = {
        "hardware_cost_reduction": hardware.reduction(scheme),
        "parameter_reduction": 1.0 - effective_parameters / dense_parameters,
        "mac_reduction": 1.0 - effective_macs / dense_macs,
        "estimated_additive_sensitivity": estimated_sensitivity,
    }
    output = args.output.expanduser().resolve()
    manifest_path = Path(str(output) + ".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite scheme output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(repr(scheme) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": "sensitivity-aware-hardware-mixed-nm",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "objective": {
            "target_reduction": args.target_reduction,
            "target_metric": args.target_metric,
            "sensitivity_metric": args.sensitivity_metric,
            "max_estimated_sensitivity": args.max_estimated_sensitivity,
            "minimum_n": args.minimum_n,
            "protect_first_conv": args.protect_first_conv,
            "protect_linear": args.protect_linear,
            "note": "Layer sensitivities are additive estimates and require end-to-end validation.",
        },
        "hardware_profile": asdict(hardware.manifest()),
        "sensitivity_profile": sensitivity.manifest(),
        "achieved": achieved,
        "scheme_file": str(output),
        "scheme": scheme,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Scheme: {output}")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(achieved, sort_keys=True))


if __name__ == "__main__":
    main()
