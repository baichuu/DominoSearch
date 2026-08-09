"""Validated hardware-cost profiles and a small reproducible cost predictor."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
METRICS = ("latency_ms", "energy_mj", "bandwidth_bytes", "memory_bytes")
def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def validate_weights(weights: dict[str, Any]) -> dict[str, float]:
    if set(weights) != set(METRICS):
        raise ValueError(f"Cost weights must contain exactly {METRICS}.")
    validated = {
        metric: _finite_nonnegative(weights[metric], f"weight {metric}")
        for metric in METRICS
    }
    if abs(sum(validated.values()) - 1.0) > 1e-8:
        raise ValueError("Hardware cost weights must sum to 1.")
    return validated


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_features(layer: dict[str, Any], n: int, m: int) -> np.ndarray:
    """Create stable numerical features for one layer/configuration pair."""

    features = layer["features"]
    layer_type = layer["type"]
    density = n / m
    batch = int(features["batch_size"])
    in_features = int(features["in_channels"])
    out_features = int(features["out_channels"])
    input_elements = int(features["input_elements"])
    output_elements = int(features["output_elements"])
    kernel_elements = int(features["kernel_elements"])
    groups = int(features.get("groups", 1))
    dense_parameters = int(features["dense_parameters"])
    dense_macs = int(features["dense_macs"])
    io_elements = input_elements + output_elements
    return np.asarray(
        [
            1.0 if layer_type == "conv2d" else 0.0,
            math.log1p(batch),
            math.log1p(in_features),
            math.log1p(out_features),
            math.log1p(input_elements),
            math.log1p(output_elements),
            math.log1p(kernel_elements),
            math.log1p(groups),
            density,
            math.log1p(dense_parameters),
            math.log1p(dense_macs),
            math.log1p(io_elements),
            density * math.log1p(dense_macs),
            density * math.log1p(dense_parameters),
        ],
        dtype=np.float64,
    )


FEATURE_NAMES = (
    "is_conv2d",
    "log_batch",
    "log_in_channels",
    "log_out_channels",
    "log_input_elements",
    "log_output_elements",
    "log_kernel_elements",
    "log_groups",
    "density",
    "log_dense_parameters",
    "log_dense_macs",
    "log_io_elements",
    "density_x_log_dense_macs",
    "density_x_log_dense_parameters",
)


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ np.log1p(y),
    )
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge": ridge,
    }


def _predict_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    return np.maximum(0.0, np.expm1(design @ coefficients))


def fit_cost_predictor(profile: dict[str, Any], ridge: float = 1e-3) -> dict[str, Any]:
    """Fit metric regressors and report leave-one-layer-out validation."""

    if ridge <= 0.0:
        raise ValueError("Predictor ridge must be positive.")
    samples: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for layer in profile["layers"]:
        for candidate in layer["candidates"]:
            samples.append(
                (
                    layer["name"],
                    candidate_features(layer, int(candidate["n"]), int(candidate["m"])),
                    candidate["metrics"],
                )
            )
    if len(samples) < 3:
        raise ValueError("At least three profile samples are required for a predictor.")

    x = np.stack([sample[1] for sample in samples])
    layer_names = np.asarray([sample[0] for sample in samples])
    predictor = {
        "kind": "ridge-log1p",
        "feature_names": list(FEATURE_NAMES),
        "metrics": {},
        "validation": {},
    }
    for metric in METRICS:
        available = np.asarray(
            [sample[2].get(metric) is not None for sample in samples], dtype=bool
        )
        if not available.any():
            continue
        y = np.asarray(
            [_finite_nonnegative(sample[2][metric], metric) for sample in samples if sample[2].get(metric) is not None],
            dtype=np.float64,
        )
        metric_x = x[available]
        metric_layers = layer_names[available]
        if len(metric_x) < 3:
            continue
        held_out_predictions = np.empty_like(y)
        for layer_name in sorted(set(metric_layers.tolist())):
            test = metric_layers == layer_name
            train = ~test
            if int(train.sum()) < 2:
                held_out_predictions[test] = y[train].mean() if train.any() else 0.0
            else:
                fold = _fit_ridge(metric_x[train], y[train], ridge)
                held_out_predictions[test] = _predict_ridge(fold, metric_x[test])
        absolute = np.abs(held_out_predictions - y)
        nonzero = y > 1e-12
        mape = float(np.mean(absolute[nonzero] / y[nonzero]) * 100.0) if nonzero.any() else None
        denominator = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((held_out_predictions - y) ** 2)) / denominator if denominator > 0.0 else None
        predictor["validation"][metric] = {
            "method": "leave-one-layer-out",
            "samples": int(len(y)),
            "mae": float(absolute.mean()),
            "mape_percent": mape,
            "r2": r2,
        }
        predictor["metrics"][metric] = _fit_ridge(metric_x, y, ridge)
    return predictor


@dataclass(frozen=True)
class HardwareCostManifest:
    path: str
    sha256: str
    device: dict[str, Any]
    model: dict[str, Any]
    weights: dict[str, float]
    predictor_validation: dict[str, Any] | None


class HardwareCostModel:
    """Resolve per-layer N:M cost from exact lookup rows or a predictor."""

    def __init__(
        self,
        profile_path: str | Path,
        mode: str = "lookup",
        weights: dict[str, float] | None = None,
    ) -> None:
        if mode not in {"lookup", "predictor"}:
            raise ValueError("Hardware cost mode must be 'lookup' or 'predictor'.")
        self.path = Path(profile_path).expanduser().resolve()
        with self.path.open(encoding="utf-8") as source:
            self.profile = json.load(source)
        self._validate_profile()
        self.mode = mode
        self.weights = validate_weights(
            weights or self.profile["cost_definition"]["weights"]
        )
        self.layers = {layer["name"]: layer for layer in self.profile["layers"]}
        self.lookup = {
            (layer["name"], int(candidate["n"]), int(candidate["m"])): candidate["metrics"]
            for layer in self.profile["layers"]
            for candidate in layer["candidates"]
        }
        self.predictor = self.profile.get("predictor")
        if mode == "predictor" and not self.predictor:
            raise ValueError("Profile has no fitted predictor.")
        self.normalizers = self._dense_normalizers()

    def _validate_profile(self) -> None:
        profile = self.profile
        if profile.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported hardware profile schema: {profile.get('schema_version')!r}."
            )
        validate_weights(profile.get("cost_definition", {}).get("weights", {}))
        layers = profile.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("Hardware profile must contain at least one layer.")
        names = [layer.get("name") for layer in layers]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("Hardware profile layer names must be unique and non-empty.")
        for layer in layers:
            candidates = layer.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Layer {layer['name']} has no candidate measurements.")
            keys = set()
            has_dense = False
            for candidate in candidates:
                n, m = int(candidate.get("n", 0)), int(candidate.get("m", 0))
                if not 0 < n <= m:
                    raise ValueError(f"Invalid N:M {n}:{m} for {layer['name']}.")
                if (n, m) in keys:
                    raise ValueError(f"Duplicate N:M {n}:{m} for {layer['name']}.")
                keys.add((n, m))
                has_dense = has_dense or n == m
                metrics = candidate.get("metrics", {})
                for metric in METRICS:
                    if metrics.get(metric) is not None:
                        _finite_nonnegative(metrics[metric], f"{layer['name']} {metric}")
            if not has_dense:
                raise ValueError(f"Layer {layer['name']} has no dense N=M reference.")

    def _dense_normalizers(self) -> dict[str, float | None]:
        normalizers: dict[str, float | None] = {}
        for metric in METRICS:
            values = []
            for layer in self.profile["layers"]:
                dense = next(
                    candidate
                    for candidate in layer["candidates"]
                    if int(candidate["n"]) == int(candidate["m"])
                )
                value = dense["metrics"].get(metric)
                if value is None:
                    values = []
                    break
                values.append(float(value))
            total = sum(values)
            normalizers[metric] = total if total > 0.0 else None
            if self.weights[metric] > 0.0 and normalizers[metric] is None:
                raise ValueError(
                    f"Cost weight for {metric} is positive but the profile has no additive dense reference."
                )
        return normalizers

    def validate_model_layers(
        self,
        layer_names: Iterable[str],
        m: int,
        candidate_n: Iterable[int] | None = None,
    ) -> None:
        expected = set(layer_names)
        actual = set(self.layers)
        if expected != actual:
            raise ValueError(
                "Hardware profile does not exactly match model sparse layers. "
                f"Missing: {sorted(expected - actual)[:5]}; unknown: {sorted(actual - expected)[:5]}."
            )
        if self.mode == "lookup":
            required_n = set(candidate_n or [m])
            missing = [
                f"{name} {n}:{m}"
                for name in sorted(expected)
                for n in sorted(required_n)
                if (name, n, m) not in self.lookup
            ]
            if missing:
                raise ValueError(f"Profile lacks required lookup rows: {missing[:5]}.")

    def metrics(self, layer_name: str, n: int, m: int) -> dict[str, float | None]:
        if layer_name not in self.layers:
            raise ValueError(f"Unknown hardware-profile layer: {layer_name}")
        if not 0 < n <= m:
            raise ValueError(f"Invalid N:M {n}:{m} for {layer_name}.")
        if self.mode == "lookup":
            try:
                values = self.lookup[(layer_name, n, m)]
            except KeyError as exc:
                raise ValueError(
                    f"No measured hardware cost for {layer_name} at {n}:{m}."
                ) from exc
            return {
                metric: (None if values.get(metric) is None else float(values[metric]))
                for metric in METRICS
            }

        x = candidate_features(self.layers[layer_name], n, m).reshape(1, -1)
        values = {}
        for metric in METRICS:
            metric_model = self.predictor["metrics"].get(metric)
            values[metric] = (
                None if metric_model is None else float(_predict_ridge(metric_model, x)[0])
            )
        return values

    def cost(self, layer_name: str, n: int, m: int) -> float:
        values = self.metrics(layer_name, n, m)
        return sum(
            self.weights[metric] * float(values[metric]) / float(self.normalizers[metric])
            for metric in METRICS
            if self.weights[metric] > 0.0
        )

    def total_cost(self, scheme: dict[str, list[int]]) -> float:
        if set(scheme) != set(self.layers):
            raise ValueError("Scheme does not exactly cover hardware-profile layers.")
        return sum(self.cost(name, int(n_m[0]), int(n_m[1])) for name, n_m in scheme.items())

    def reduction(self, scheme: dict[str, list[int]]) -> float:
        dense = {
            name: [max(int(candidate["m"]) for candidate in layer["candidates"]),
                   max(int(candidate["m"]) for candidate in layer["candidates"])]
            for name, layer in self.layers.items()
        }
        dense_cost = self.total_cost(dense)
        if dense_cost <= 0.0:
            raise ValueError("Dense hardware cost must be positive.")
        return 1.0 - self.total_cost(scheme) / dense_cost

    def best_independent_scheme(self) -> dict[str, list[int]]:
        """Return the lowest-cost measured candidate for each layer."""

        scheme = {}
        for name, layer in self.layers.items():
            candidates = [
                (int(candidate["n"]), int(candidate["m"]))
                for candidate in layer["candidates"]
            ]
            n, m = min(candidates, key=lambda n_m: self.cost(name, *n_m))
            scheme[name] = [n, m]
        return scheme

    def maximum_independent_reduction(self) -> float:
        return self.reduction(self.best_independent_scheme())

    def minimum_loss_scheme(
        self,
        target_reduction: float,
        loss_metric: str = "parameters",
    ) -> dict[str, list[int]]:
        """Meet a cost target with minimum theoretical pruning loss.

        This is a deterministic multiple-choice Pareto search. ``parameters``
        and ``macs`` are complexity proxies, not accuracy predictors.
        """

        if not 0.0 < target_reduction < 1.0:
            raise ValueError("Target hardware-cost reduction must be in (0, 1).")
        if loss_metric not in {"parameters", "macs"}:
            raise ValueError("Loss metric must be 'parameters' or 'macs'.")

        dense_scheme = {
            name: [
                max(int(candidate["m"]) for candidate in layer["candidates"]),
                max(int(candidate["m"]) for candidate in layer["candidates"]),
            ]
            for name, layer in self.layers.items()
        }
        dense_cost = self.total_cost(dense_scheme)
        required_saving = target_reduction * dense_cost
        # cost saving, primary loss, secondary loss, N choices
        frontier: list[tuple[float, float, float, tuple[int, ...]]] = [
            (0.0, 0.0, 0.0, ())
        ]
        layer_items = list(self.layers.items())
        for name, layer in layer_items:
            dense_n, dense_m = dense_scheme[name]
            dense_layer_cost = self.cost(name, dense_n, dense_m)
            features = layer["features"]
            dense_parameters = float(features["dense_parameters"])
            dense_macs = float(features["dense_macs"])
            candidates = []
            for candidate in layer["candidates"]:
                n, m = int(candidate["n"]), int(candidate["m"])
                density = n / m
                parameter_loss = dense_parameters * (1.0 - density)
                mac_loss = dense_macs * (1.0 - density)
                primary, secondary = (
                    (parameter_loss, mac_loss)
                    if loss_metric == "parameters"
                    else (mac_loss, parameter_loss)
                )
                candidates.append(
                    (dense_layer_cost - self.cost(name, n, m), primary, secondary, n)
                )

            expanded = [
                (saving + ds, loss + dl, secondary + dsecondary, choices + (n,))
                for saving, loss, secondary, choices in frontier
                for ds, dl, dsecondary, n in candidates
            ]
            expanded.sort(key=lambda state: (-state[0], state[1], state[2], state[3]))
            frontier = []
            best_loss: tuple[float, float] | None = None
            for state in expanded:
                state_loss = (state[1], state[2])
                if best_loss is None or state_loss < best_loss:
                    frontier.append(state)
                    best_loss = state_loss

        feasible = [state for state in frontier if state[0] + 1e-12 >= required_saving]
        if not feasible:
            maximum = self.maximum_independent_reduction()
            raise ValueError(
                f"Target reduction {target_reduction:.6f} is infeasible; "
                f"maximum independent reduction is {maximum:.6f}."
            )
        selected = min(
            feasible,
            key=lambda state: (state[1], state[2], state[0] - required_saving, state[3]),
        )
        return {
            name: [n, dense_scheme[name][1]]
            for (name, _), n in zip(layer_items, selected[3])
        }

    def normalization(self, scheme: dict[str, list[int]]) -> dict[str, float]:
        costs = {
            name: self.cost(name, int(n_m[0]), int(n_m[1]))
            for name, n_m in scheme.items()
        }
        maximum = max(costs.values())
        if maximum <= 0.0:
            raise ValueError("At least one layer hardware cost must be positive.")
        return {name: value / maximum for name, value in costs.items()}

    def manifest(self) -> HardwareCostManifest:
        return HardwareCostManifest(
            path=str(self.path),
            sha256=file_sha256(self.path),
            device=self.profile["hardware"],
            model=self.profile["model"],
            weights=self.weights,
            predictor_validation=(
                None if self.predictor is None else self.predictor.get("validation")
            ),
        )
