"""Validation and lookup helpers for layer-wise N:M sensitivity profiles."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
SENSITIVITY_METRICS = ("loss_increase", "top1_drop_percent", "top5_drop_percent")
PROFILE_METHODS = (
    "one-layer-at-a-time-nm-sensitivity",
    "conditioned-one-layer-at-a-time-nm-sensitivity",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partial_profile_path(output: str | Path) -> Path:
    return Path(str(Path(output)) + ".partial.json")


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    """Write JSON through a sibling temporary file to survive interruption."""

    destination = Path(path)
    temporary = Path(str(destination) + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def resume_identity(profile: dict[str, Any]) -> dict[str, Any]:
    """Fields that must remain identical before partial measurements are reused."""

    return {
        "schema_version": profile["schema_version"],
        "method": profile["method"],
        "model": profile["model"],
        "dataset": profile["dataset"],
        "seed": profile["measurement"]["seed"],
        "base_scheme_file": profile["base_scheme_file"],
        "base_scheme": profile["base_scheme"],
        "candidate_n": profile["candidate_n"],
        "m": profile["m"],
    }


def load_partial_profile(path: str | Path, expected: dict[str, Any]) -> dict[str, Any]:
    source = Path(path)
    partial = json.loads(source.read_text(encoding="utf-8"))
    if partial.get("progress", {}).get("status") != "incomplete":
        raise ValueError(f"Resume profile is not marked incomplete: {source}")
    if resume_identity(partial) != resume_identity(expected):
        raise ValueError(
            "Partial sensitivity profile does not match the current checkpoint, "
            "scheme, dataset, candidates, or seed."
        )
    return partial


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def load_nm_scheme(path: str | Path) -> dict[str, list[int]]:
    """Load and strictly validate the first-line Python N:M scheme format."""

    source = Path(path).expanduser().resolve()
    raw = source.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Scheme file is empty: {source}")
    value = ast.literal_eval(raw.splitlines()[0])
    if not isinstance(value, dict) or not value:
        raise ValueError("Scheme must be a non-empty Python dictionary on the first line.")
    scheme: dict[str, list[int]] = {}
    for name, pair in value.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(pair, (list, tuple))
            or len(pair) != 2
        ):
            raise ValueError(f"Invalid scheme entry: {name!r}: {pair!r}")
        n, m = int(pair[0]), int(pair[1])
        if not 0 < n <= m:
            raise ValueError(f"Invalid N:M value for {name}: {n}:{m}")
        scheme[name] = [n, m]
    return scheme


def validate_scheme_layers(
    scheme: dict[str, list[int]], layer_names: Iterable[str], label: str = "Scheme"
) -> None:
    expected = set(layer_names)
    actual = set(scheme)
    if expected != actual:
        raise ValueError(
            f"{label} does not exactly match sparse layers. "
            f"Missing: {sorted(expected - actual)[:5]}; "
            f"unknown: {sorted(actual - expected)[:5]}."
        )


class LayerSensitivityProfile:
    """Strict reader for one-layer-at-a-time pruning measurements."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.profile = json.loads(self.path.read_text(encoding="utf-8"))
        if self.profile.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported sensitivity schema {self.profile.get('schema_version')!r}."
            )
        self.method = self.profile.get("method")
        if self.method not in PROFILE_METHODS:
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

        raw_base = self.profile.get("base_scheme")
        self.base_scheme: dict[str, list[int]] | None = None
        if self.method == "conditioned-one-layer-at-a-time-nm-sensitivity":
            if not isinstance(raw_base, dict) or not raw_base:
                raise ValueError("Conditioned sensitivity profile must contain base_scheme.")
            self.base_scheme = {}
            for name, pair in raw_base.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                ):
                    raise ValueError(f"Invalid conditioned base scheme entry: {name!r}: {pair!r}")
                n, m = int(pair[0]), int(pair[1])
                if not 0 < n <= m:
                    raise ValueError(f"Invalid conditioned base N:M for {name}: {n}:{m}")
                self.base_scheme[name] = [n, m]
            validate_scheme_layers(self.base_scheme, self.layers, "Conditioned base scheme")
        elif raw_base is not None:
            raise ValueError("Dense sensitivity profile must not contain base_scheme.")

    def validate_candidates(
        self, layer_names: Iterable[str], candidates: Iterable[tuple[int, int]]
    ) -> None:
        required = set(candidates)
        self.validate_candidate_map(
            {name: required for name in layer_names}
        )

    def validate_candidate_map(
        self, candidates_by_layer: dict[str, Iterable[tuple[int, int]]]
    ) -> None:
        """Validate exact layers and the candidate rows selection will consume."""

        expected = set(candidates_by_layer)
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
            for n, m in candidates_by_layer[name]
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

    def validate_conditioning(self, base_scheme: dict[str, list[int]] | None) -> None:
        """Prevent mixing dense and conditioned sensitivity semantics."""

        if self.base_scheme is None and base_scheme is not None:
            raise ValueError(
                "A base scheme requires a conditioned sensitivity profile measured from it."
            )
        if self.base_scheme is not None and base_scheme is None:
            raise ValueError(
                "Conditioned sensitivity profile requires its base scheme during selection."
            )
        if self.base_scheme != base_scheme:
            raise ValueError("Selected base scheme does not match the sensitivity profile.")

    def manifest(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": file_sha256(self.path),
            "method": self.method,
            "base_scheme": self.base_scheme,
            "model": self.profile.get("model"),
            "dataset": self.profile.get("dataset"),
            "measurement": self.profile.get("measurement"),
        }
