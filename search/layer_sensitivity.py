"""Validation and lookup helpers for layer-wise N:M sensitivity profiles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
SENSITIVITY_METRICS = ("loss_increase", "top1_drop_percent", "top5_drop_percent")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


class LayerSensitivityProfile:
    """Strict reader for one-layer-at-a-time pruning measurements."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.profile = json.loads(self.path.read_text(encoding="utf-8"))
        if self.profile.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported sensitivity schema {self.profile.get('schema_version')!r}."
            )
        if self.profile.get("method") != "one-layer-at-a-time-nm-sensitivity":
            raise ValueError("Not a layer-wise N:M sensitivity profile.")
        rows = self.profile.get("layers")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Sensitivity profile must contain a non-empty layers list.")
        self.layers: dict[str, dict[str, Any]] = {}
        self.lookup: dict[tuple[str, int, int], dict[str, float]] = {}
        for layer in rows:
            name = layer.get("name")
            if not isinstance(name, str) or not name or name in self.layers:
                raise ValueError(f"Invalid or duplicate sensitivity layer name: {name!r}.")
            candidates = layer.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Sensitivity layer {name} has no candidates.")
            self.layers[name] = layer
            for candidate in candidates:
                n, m = int(candidate.get("n", 0)), int(candidate.get("m", 0))
                if not 0 < n <= m:
                    raise ValueError(f"Invalid sensitivity candidate {n}:{m} for {name}.")
                key = (name, n, m)
                if key in self.lookup:
                    raise ValueError(f"Duplicate sensitivity candidate {name} {n}:{m}.")
                metrics = candidate.get("sensitivity")
                if not isinstance(metrics, dict) or set(metrics) != set(SENSITIVITY_METRICS):
                    raise ValueError(
                        f"Sensitivity metrics for {name} {n}:{m} must contain exactly "
                        f"{SENSITIVITY_METRICS}."
                    )
                self.lookup[key] = {
                    metric: _finite(metrics[metric], f"{name} {n}:{m} {metric}")
                    for metric in SENSITIVITY_METRICS
                }

    def validate_candidates(
        self, layer_names: Iterable[str], candidates: Iterable[tuple[int, int]]
    ) -> None:
        expected = set(layer_names)
        actual = set(self.layers)
        if expected != actual:
            raise ValueError(
                "Sensitivity profile does not exactly match hardware-profile layers. "
                f"Missing: {sorted(expected - actual)[:5]}; "
                f"unknown: {sorted(actual - expected)[:5]}."
            )
        missing = [
            f"{name} {n}:{m}"
            for name in sorted(expected)
            for n, m in candidates
            if (name, n, m) not in self.lookup
        ]
        if missing:
            raise ValueError(f"Sensitivity profile lacks required rows: {missing[:5]}.")

    def value(self, layer_name: str, n: int, m: int, metric: str) -> float:
        if metric not in SENSITIVITY_METRICS:
            raise ValueError(f"Unknown sensitivity metric {metric!r}.")
        try:
            return self.lookup[(layer_name, n, m)][metric]
        except KeyError as exc:
            raise ValueError(f"No sensitivity row for {layer_name} at {n}:{m}.") from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": file_sha256(self.path),
            "model": self.profile.get("model"),
            "dataset": self.profile.get("dataset"),
            "measurement": self.profile.get("measurement"),
        }
