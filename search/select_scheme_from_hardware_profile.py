#!/usr/bin/env python3
"""Select a mixed N:M scheme directly from a measured hardware profile."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from hardware_cost import HardwareCostModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-reduction", type=float, required=True)
    parser.add_argument(
        "--loss-metric", choices=("parameters", "macs"), default="parameters"
    )
    parser.add_argument("--latency-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--bandwidth-weight", type=float, default=None)
    parser.add_argument("--memory-weight", type=float, default=None)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False
    ).stdout.strip()


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
    weights = (
        None
        if not any(supplied)
        else {key: float(value) for key, value in overrides.items()}
    )
    model = HardwareCostModel(args.hardware_profile, mode="lookup", weights=weights)
    scheme = model.minimum_loss_scheme(args.target_reduction, args.loss_metric)

    dense_parameters = sum(
        float(layer["features"]["dense_parameters"]) for layer in model.layers.values()
    )
    dense_macs = sum(
        float(layer["features"]["dense_macs"]) for layer in model.layers.values()
    )
    effective_parameters = sum(
        float(layer["features"]["dense_parameters"]) * scheme[name][0] / scheme[name][1]
        for name, layer in model.layers.items()
    )
    effective_macs = sum(
        float(layer["features"]["dense_macs"]) * scheme[name][0] / scheme[name][1]
        for name, layer in model.layers.items()
    )
    achieved = {
        "hardware_cost_reduction": model.reduction(scheme),
        "parameter_reduction": 1.0 - effective_parameters / dense_parameters,
        "mac_reduction": 1.0 - effective_macs / dense_macs,
    }
    output = args.output.expanduser().resolve()
    manifest_path = Path(str(output) + ".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite scheme output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(repr(scheme) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": "hardware-profile-mixed-nm",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "objective": {
            "target_reduction": args.target_reduction,
            "loss_metric": args.loss_metric,
            "hardware_cost_mode": "lookup",
        },
        "hardware_profile": asdict(model.manifest()),
        "achieved": achieved,
        "scheme_file": str(output),
        "scheme": scheme,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Scheme: {output}")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(achieved, sort_keys=True))


if __name__ == "__main__":
    main()
