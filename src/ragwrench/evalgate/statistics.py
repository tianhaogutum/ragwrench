"""Seeded paired inference using only the Python standard library.

All p-values are two-sided. Monte Carlo p-values use the add-one correction.
"""
from __future__ import annotations

import math
import random
import statistics

from .formats import InputError


def adjust_p_values(values: list[float], correction: str) -> list[float]:
    """Bonferroni or Holm step-down adjusted p-values, in original order."""
    if correction not in {"none", "bonferroni", "holm"}:
        raise InputError("correction must be none, bonferroni or holm")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise InputError("p-values must be finite and in [0, 1]")
    count = len(values)
    if correction == "none":
        return list(values)
    if correction == "bonferroni":
        return [min(1.0, p * count) for p in values]
    adjusted = [0.0] * count
    previous = 0.0
    for rank, index in enumerate(sorted(range(count), key=lambda i: (values[i], i))):
        previous = max(previous, min(1.0, (count - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def _beta_fraction(a, b, x):
    """Lentz continued fraction for the incomplete beta function."""
    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1001):
        for coefficient in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + coefficient * d
            c = 1.0 + coefficient / c
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = c if abs(c) > tiny else tiny
            change = d * c
            h *= change
        if abs(change - 1.0) < 3e-14:
            return h
    raise InputError("incomplete beta calculation failed to converge")


def _regularized_beta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        result = factor * _beta_fraction(a, b, x) / a
    else:
        result = 1.0 - factor * _beta_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, result))


def t_two_sided(t: float, degrees: int) -> float:
    return _regularized_beta(degrees / 2, 0.5, degrees / (degrees + t * t))


def _t_critical(confidence, degrees):
    lo, hi = 0.0, 1.0
    while t_two_sided(hi, degrees) > 1 - confidence:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_two_sided(mid, degrees) > 1 - confidence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _quantile(sorted_values, probability):
    pos = probability * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def validate_paired_options(query_count, method, *, samples, confidence):
    """Validate inference settings and work bounds without performing inference."""
    if method not in {"bootstrap", "randomization", "ttest"}:
        raise InputError(f"unknown statistical test: {method}")
    if query_count < 2:
        raise InputError("paired inference requires at least two qrels queries")
    try:
        valid_confidence = 0 < confidence < 1 and math.isfinite(confidence)
    except (TypeError, ValueError, OverflowError):
        valid_confidence = False
    if not valid_confidence:
        raise InputError("confidence must be finite and strictly between 0 and 1")
    if not isinstance(samples, int) or not 100 <= samples <= 1_000_000:
        raise InputError("samples must be an integer in [100, 1000000]")
    if method != "ttest" and query_count * samples > 50_000_000:
        raise InputError("resampling budget exceeds 50000000 query draws; reduce --samples or use --test ttest")


def paired_test(deltas: list[float], method="randomization", *, seed=42, samples=10000, confidence=0.95) -> dict:
    validate_paired_options(len(deltas), method, samples=samples, confidence=confidence)
    if not all(math.isfinite(d) for d in deltas):
        raise InputError("paired deltas must be finite")
    n = len(deltas)
    observed = statistics.fmean(deltas)
    warning = None
    if method == "ttest":
        se = statistics.stdev(deltas) / math.sqrt(n)
        if se == 0:
            p = 1.0 if observed == 0 else 0.0
            ci = [observed, observed]
            warning = "zero paired variance; p-value is the degenerate t-test limit"
        else:
            p = t_two_sided(observed / se, n - 1)
            margin = _t_critical(confidence, n - 1) * se
            ci = [observed - margin, observed + margin]
        ci_method = "student-t"
    else:
        rng = random.Random(seed)
        boot = sorted(statistics.fmean(rng.choices(deltas, k=n)) for _ in range(samples))
        alpha = (1 - confidence) / 2
        ci = [_quantile(boot, alpha), _quantile(boot, 1 - alpha)]
        ci_method = "paired-bootstrap-percentile"
        if method == "bootstrap":
            # Recenter bootstrap replicates to approximate the null mean of zero.
            extreme = sum(abs(value - observed) >= abs(observed) - 1e-15 for value in boot)
        else:
            # An independent stream keeps the permutation p-value independent of CI draws.
            rng = random.Random(f"randomization:{seed}")
            extreme = 0
            for _ in range(samples):
                permuted = math.fsum(d if rng.getrandbits(1) else -d for d in deltas) / n
                extreme += abs(permuted) >= abs(observed) - 1e-15
        p = (extreme + 1) / (samples + 1)
    return {"test": method, "seed": seed, "samples": samples if method != "ttest" else None,
            "confidence": confidence, "ci": ci, "ci_method": ci_method, "p_value": p,
            "warning": warning}
