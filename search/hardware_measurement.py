"""Dependency-free statistics helpers for reproducible hardware measurements."""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable


def percentile(values: Iterable[float], fraction: float) -> float:
    samples = sorted(float(value) for value in values)
    if not samples:
        raise ValueError("At least one sample is required.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Percentile fraction must be in [0, 1].")
    position = (len(samples) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def bootstrap_median_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("At least one sample is required.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be in (0, 1).")
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive.")
    if len(samples) == 1:
        return samples[0], samples[0]
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choices(samples, k=len(samples)))
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    return percentile(medians, tail), percentile(medians, 1.0 - tail)


def summarize_samples(
    values: Iterable[float],
    bootstrap_resamples: int = 2_000,
    seed: int = 42,
) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples or any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("Measurement samples must be finite and non-negative.")
    ci_low, ci_high = bootstrap_median_ci(
        samples, resamples=bootstrap_resamples, seed=seed
    )
    return {
        "samples": len(samples),
        "median": statistics.median(samples),
        "p95": percentile(samples, 0.95),
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }
