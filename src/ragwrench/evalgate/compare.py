"""Pair queries by ID, report regressions, and apply absolute-drop gates."""
from __future__ import annotations

from collections.abc import Mapping
import math
import re

from .formats import InputError
from .metrics import evaluate, validate_metrics
from .saved import bind_evaluation, validate_evaluation
from .statistics import adjust_p_values, paired_test, validate_paired_options

TOLERANCE = 1e-12


class PreflightMismatch(InputError):
    """Valid bound audits drifted; ``diff`` is the standalone schema-1 diagnostic.

    Malformed audits and invalid actual families raise ordinary InputError.
    The diagnostic owns its values and contains no inference results.
    """

    def __init__(self, message, diff):
        super().__init__(message)
        self.diff = diff


def validate_candidate_names(names):
    if not names:
        raise InputError("at least one named candidate is required")
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise InputError("candidate names must match [A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_thresholds(values: list[str], metrics, *, allow_off=False) -> dict[str, float | None]:
    allowed = {str(m) for m in metrics}
    result = {}
    for text in values:
        try:
            metric, value = text.split("=")
            drop = None if allow_off and value == "off" else float(value)
        except (ValueError, TypeError, OverflowError):
            expected = "nonnegative-number or off" if allow_off else "nonnegative-number"
            raise InputError(f"invalid max-drop {text!r}; expected metric@k={expected}") from None
        if metric not in allowed:
            raise InputError(f"max-drop metric {metric!r} must appear in --metrics")
        if metric in result or (drop is not None and (not math.isfinite(drop) or drop < 0)):
            raise InputError("max-drop values must be unique, finite and nonnegative")
        result[metric] = drop
    return result


def parse_candidate_thresholds(values: list[str], candidates, metrics) -> dict[str, dict[str, float | None]]:
    result = {}
    for text in values:
        name, separator, threshold = text.partition(":")
        if not separator:
            raise InputError(f"invalid candidate max-drop {text!r}; expected NAME:metric@k=nonnegative-number")
        if name not in candidates:
            raise InputError(f"unknown max-drop candidate {name!r}; must name a --candidate")
        parsed = parse_thresholds([threshold], metrics, allow_off=True)
        overrides = result.setdefault(name, {})
        if overrides.keys() & parsed.keys():
            raise InputError(f"duplicate candidate max-drop for {name!r}: {threshold.split('=')[0]}")
        overrides.update(parsed)
    return result


def compare(qrels, baseline, candidate, metrics, *, gain="linear", method="randomization", seed=42,
            samples=10000, confidence=0.95, thresholds=None, top=10):
    _validate_options(metrics, thresholds, top)
    base = evaluate(qrels, baseline, metrics, gain)
    cand = evaluate(qrels, candidate, metrics, gain)
    return _compare_evaluated(base, cand, metrics, method=method, seed=seed, samples=samples,
                              confidence=confidence, thresholds=thresholds, top=top)


def compare_saved(baseline, candidate, *, method="randomization", seed=42,
                  samples=10000, confidence=0.95, thresholds=None, top=10):
    """Pair complete saved evaluations only when judgments and settings match."""
    metrics = validate_evaluation(baseline)
    validate_evaluation(candidate)
    _validate_saved_pair(baseline, candidate)
    _validate_options(metrics, thresholds, top)
    report = _compare_evaluated(baseline, candidate, metrics, method=method, seed=seed,
                                samples=samples, confidence=confidence, thresholds=thresholds, top=top)
    return _saved_report(report, baseline, candidate)


def _validate_saved_pair(baseline, candidate):
    base, cand = baseline["manifest"], candidate["manifest"]
    bq, cq = set(base["dataset"]["query_ids"]), set(cand["dataset"]["query_ids"])
    if bq != cq:
        raise InputError(f"judged query sets differ; baseline-only={sorted(bq - cq)}, candidate-only={sorted(cq - bq)}")
    if base["dataset"]["sha256"] != cand["dataset"]["sha256"]:
        raise InputError("dataset hashes differ: judgments must match, not just query IDs")
    if base["parameters"] != cand["parameters"]:
        raise InputError("evaluation parameters differ: metrics, gain, ranking and query policy must match")


def _saved_report(report, baseline, candidate):
    report["command"] = "compare-saved"
    report["evaluation_manifests"] = {"baseline": baseline["manifest"], "candidate": candidate["manifest"]}
    return report


def _validate_options(metrics, thresholds, top, *, allow_disabled=False):
    validate_metrics(metrics)
    if top < 0:
        raise InputError("top must be nonnegative")
    thresholds = thresholds or {}
    if set(thresholds) - {str(m) for m in metrics}:
        raise InputError("threshold metric must be evaluated")
    try:
        invalid = any(not math.isfinite(v) or v < 0 for v in thresholds.values()
                      if not (allow_disabled and v is None))
    except (TypeError, ValueError, OverflowError):
        invalid = True
    if invalid:
        raise InputError("thresholds must be finite and nonnegative")


def _compare_evaluated(base, cand, metrics, *, method, seed, samples, confidence, thresholds, top):
    thresholds = thresholds or {}
    per_query = {qid: {} for qid in sorted(base["per_query"])}
    rows = {}
    violations = []
    for metric in metrics:
        key = str(metric)
        deltas = []
        wins = losses = 0
        regressed = []
        for qid, row in per_query.items():
            b, c = base["per_query"][qid][key], cand["per_query"][qid][key]
            delta = c - b
            deltas.append(delta)
            outcome = "win" if delta > TOLERANCE else "loss" if delta < -TOLERANCE else "tie"
            row[key] = {"baseline": b, "candidate": c, "delta": delta, "outcome": outcome}
            wins += delta > TOLERANCE
            losses += delta < -TOLERANCE
            if delta < -TOLERANCE:
                regressed.append({"query_id": qid, **row[key]})
        delta = math.fsum(deltas) / len(deltas)
        threshold = thresholds.get(key)
        violated = threshold is not None and delta < -threshold - TOLERANCE
        if violated:
            violations.append({"metric": key, "drop": -delta, "max_drop": threshold})
        rows[key] = {"baseline": base["means"][key], "candidate": cand["means"][key], "delta": delta,
                     **paired_test(deltas, method, seed=seed, samples=samples, confidence=confidence),
                     "wins": wins, "ties": len(deltas) - wins - losses, "losses": losses,
                     "max_drop": threshold, "passed": not violated,
                     "top_regressed": sorted(regressed, key=lambda r: (r["delta"], r["query_id"]))[:top]}
    return {"schema_version": 1, "command": "compare", "passed": not violations,
            "query_count": base["query_count"], "gain": base["gain"], "tie_rule": base["tie_rule"],
            "baseline_missing_queries": base["missing_queries"], "candidate_missing_queries": cand["missing_queries"],
            "baseline_ignored_queries": base["ignored_queries"], "candidate_ignored_queries": cand["ignored_queries"],
            "metrics": rows, "per_query": per_query, "violations": violations}


def compare_many(qrels, baseline, candidates, metrics, *, correction, alpha=0.05,
                 gain="linear", method="randomization", seed=42, samples=10000,
                 confidence=0.95, thresholds=None, candidate_thresholds=None, top=10,
                 max_family_resamples=None, max_family_pairs=None, preflight_only=False, expected_preflight=None):
    """Compare a named family to one baseline; correct p-values across all pairs.

    Corrections label significance only. Absolute mean-drop gates and marginal
    confidence intervals retain the single-comparison semantics. Candidate
    thresholds override shared metric defaults; None disables an inherited gate.
    Unspecified pairs without a shared default have no gate. Optional total-family
    resampling and pair-count limits count every pair, including ungated and
    opted-out pairs. The pair-count limit also applies to analytic t-tests.
    preflight_only=True returns a JSON audit summary with canonical input
    bindings, without inference or gates. expected_preflight optionally requires
    the complete bound audit to match before inference (or preflight output).
    """
    _validate_preflight_mode(preflight_only)
    policy = _family_policy(candidates, metrics, correction, alpha, thresholds, candidate_thresholds, top)
    metrics = sorted(metrics, key=str)
    base = evaluate(qrels, baseline, metrics, gain)
    evaluated = {name: evaluate(qrels, candidates[name], metrics, gain) for name in sorted(candidates)}
    return _compare_many_evaluated(base, evaluated, metrics, policy=policy,
                                   method=method, seed=seed, samples=samples, confidence=confidence, top=top,
                                   max_family_resamples=max_family_resamples, max_family_pairs=max_family_pairs,
                                   preflight_only=preflight_only, expected_preflight=expected_preflight,
                                   binding_builder=lambda: _input_bindings(
                                       bind_evaluation(qrels, baseline, base),
                                       {name: bind_evaluation(qrels, candidates[name], evaluated[name])
                                        for name in sorted(candidates)}))


def compare_many_saved(baseline, candidates, *, correction, alpha=0.05,
                       method="randomization", seed=42, samples=10000,
                       confidence=0.95, thresholds=None, candidate_thresholds=None, top=10,
                       max_family_resamples=None, max_family_pairs=None, preflight_only=False, expected_preflight=None):
    """Validate an entire saved family before inference against its fixed baseline.

    preflight_only=True returns a JSON audit summary with validated artifact
    bindings, without inference or gates. expected_preflight requires a matching
    complete bound audit before inference.
    """
    _validate_preflight_mode(preflight_only)
    validate_candidate_names(candidates)
    try:
        metrics = validate_evaluation(baseline)
    except InputError as exc:
        raise InputError(f"baseline: {exc}") from None
    for name in sorted(candidates):
        try:
            validate_evaluation(candidates[name])
            _validate_saved_pair(baseline, candidates[name])
        except InputError as exc:
            raise InputError(f"candidate {name!r}: {exc}") from None
    policy = _family_policy(candidates, metrics, correction, alpha, thresholds, candidate_thresholds, top)
    report = _compare_many_evaluated(baseline, candidates, metrics, policy=policy,
                                     method=method, seed=seed, samples=samples, confidence=confidence, top=top,
                                     max_family_resamples=max_family_resamples, max_family_pairs=max_family_pairs,
                                     preflight_only=preflight_only, expected_preflight=expected_preflight,
                                     binding_builder=lambda: _input_bindings(baseline, candidates),
                                     preflight_command="compare-many-saved")
    report["command"] = "compare-many-saved"
    if not preflight_only:
        for name, comparison in report["candidates"].items():
            _saved_report(comparison, baseline, candidates[name])
    return report


def _input_bindings(baseline, candidates):
    manifest = baseline["manifest"]
    parameters = manifest["parameters"]
    return {
        "schema_version": 1,
        "evaluator": {"name": manifest["evaluator"]["name"], "version": manifest["evaluator"]["version"]},
        "dataset_sha256": manifest["dataset"]["sha256"],
        "parameters": {"metrics": list(parameters["metrics"]), "gain": parameters["gain"],
                       "tie_rule": parameters["tie_rule"], "query_policy": parameters["query_policy"]},
        "baseline": _artifact_binding(baseline),
        "candidates": {name: _artifact_binding(candidates[name]) for name in sorted(candidates)},
    }


def _artifact_binding(evaluation):
    manifest = evaluation["manifest"]
    return {"artifact_sha256": manifest["artifact_sha256"], "run_sha256": manifest["run"]["sha256"]}


def _family_policy(candidates, metrics, correction, alpha, thresholds, candidate_thresholds, top):
    validate_candidate_names(candidates)
    if correction not in {"none", "bonferroni", "holm"}:
        raise InputError("correction must be none, bonferroni or holm")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise InputError("alpha must be finite and strictly between 0 and 1")
    _validate_options(metrics, thresholds, top)
    candidate_thresholds = candidate_thresholds or {}
    for name, overrides in candidate_thresholds.items():
        if name not in candidates:
            raise InputError(f"unknown max-drop candidate {name!r}; must name a candidate")
        _validate_options(metrics, overrides, top, allow_disabled=True)
    shared = dict(sorted((thresholds or {}).items()))
    overrides = {name: dict(sorted(values.items())) for name, values in sorted(candidate_thresholds.items())}
    effective = {name: {metric: value for metric, value in sorted({**shared, **overrides.get(name, {})}.items())
                        if value is not None} for name in sorted(candidates)}
    return {"correction": correction, "alpha": alpha, "family": "all candidate-metric pairs",
            "family_size": len(candidates) * len(metrics), "ci_adjustment": "none", "gate": "absolute-mean-drop",
            "significance_affects_gate": False, "thresholds": shared, "candidate_thresholds": overrides,
            "threshold_precedence": "candidate override > shared metric default > no gate",
            "effective_thresholds": effective}


def _validate_preflight_mode(preflight_only):
    if type(preflight_only) is not bool:
        raise InputError("preflight_only must be a boolean")


def _family_limits(query_count, policy, method, samples, max_family_resamples, max_family_pairs):
    if max_family_pairs is not None:
        if type(max_family_pairs) is not int or max_family_pairs <= 0:
            raise InputError("--max-family-pairs must be a positive integer")
        required = policy["family_size"]
        if required > max_family_pairs:
            raise InputError(f"family pair limit exceeded: required={required}, allowed={max_family_pairs}; "
                             "reduce candidates or metrics, or raise --max-family-pairs")
        policy = {**policy, "max_family_pairs": max_family_pairs}
    if max_family_resamples is not None:
        if type(max_family_resamples) is not int or max_family_resamples <= 0:
            raise InputError("--max-family-resamples must be a positive integer")
        if method not in {"bootstrap", "randomization", "ttest"}:
            raise InputError(f"unknown statistical test: {method}")
        if not isinstance(samples, int) or not 100 <= samples <= 1_000_000:
            raise InputError("samples must be an integer in [100, 1000000]")
        # Normalized work units, not RNG calls: randomization has both a
        # bootstrap CI stream and a sign-flip stream. Count all hypotheses.
        required = 0 if method == "ttest" else query_count * samples * policy["family_size"]
        if required > max_family_resamples:
            raise InputError(f"family resampling budget exceeded: required={required}, allowed={max_family_resamples}; "
                             "reduce --samples or family size, raise --max-family-resamples, or use --test ttest")
        policy = {**policy, "estimated_family_resamples": required, "max_family_resamples": max_family_resamples}
    return policy


def _verify_expected_preflight(expected, current, path=()):
    """Strict structural/value equality; only root family command is normalized.

    Compare against the freshly validated audit rather than trusting audit counts,
    digests or policy in isolation. Unknown and missing fields fail at every depth.
    Never coerce bools or integer schema/count/option fields through numeric equality.
    """
    label = ".".join(path) or "audit"

    def mismatch():
        raise InputError(f"expected preflight mismatch at {label}; regenerate or use matching inputs/options")

    if isinstance(current, dict):
        if not isinstance(expected, Mapping) or set(expected) != set(current):
            mismatch()
        for key in current:
            _verify_expected_preflight(expected[key], current[key], (*path, key))
    elif isinstance(current, list):
        if not isinstance(expected, list) or len(expected) != len(current):
            mismatch()
        for index, (old, new) in enumerate(zip(expected, current)):
            _verify_expected_preflight(old, new, (*path, str(index)))
    elif path == ("command",):
        if not isinstance(expected, str) or expected not in ("compare-many", "compare-many-saved"):
            mismatch()
    elif type(current) in (int, float):
        integer_field = path[-1] in {"schema_version", "query_count", "family_size", "estimated_family_resamples",
                                     "seed", "samples", "max_family_pairs", "max_family_resamples"}
        if type(expected) not in ((int,) if integer_field else (int, float)) or expected != current:
            mismatch()
    elif type(expected) is not type(current) or expected != current:
        mismatch()


def _family_preflight(base, candidates, metrics, policy, method, seed, samples, confidence,
                      max_family_resamples, max_family_pairs, command, binding_builder):
    validate_paired_options(base["query_count"], method, samples=samples, confidence=confidence)
    if type(seed) is not int:
        raise InputError("preflight seed must be an integer")
    work = 0 if method == "ttest" else base["query_count"] * samples * policy["family_size"]
    return {"schema_version": 1, "command": command, "mode": "preflight-only",
            "query_count": base["query_count"], "candidate_names": sorted(candidates),
            "metric_names": [str(m) for m in metrics], "family_size": policy["family_size"],
            "estimated_family_resamples": work,
            "limits": {"max_family_pairs": max_family_pairs, "max_family_resamples": max_family_resamples},
            "settings": {"test": method, "seed": seed, "samples": samples, "confidence": confidence},
            "policy": policy, "input_bindings": binding_builder()}


def _compare_many_evaluated(base, candidates, metrics, *, policy, method, seed, samples, confidence, top,
                            max_family_resamples=None, max_family_pairs=None, preflight_only=False,
                            expected_preflight=None, binding_builder=None, preflight_command="compare-many"):
    metrics = sorted(metrics, key=str)
    policy = _family_limits(base["query_count"], policy, method, samples,
                            max_family_resamples, max_family_pairs)
    if preflight_only or expected_preflight is not None:
        audit = _family_preflight(base, candidates, metrics, policy, method, seed, samples, confidence,
                                  max_family_resamples, max_family_pairs, preflight_command, binding_builder)
        if expected_preflight is not None:
            try:
                _verify_expected_preflight(expected_preflight, audit)
            except InputError as mismatch:
                # Preserve the guard's existing first-mismatch message. Only
                # independently valid audits may expose a drift diagnostic.
                # Local import avoids the pure validator's compare dependency.
                from .preflight import diff_preflight
                try:
                    diagnostic = diff_preflight(expected_preflight, audit)
                except InputError:
                    raise mismatch from None
                raise PreflightMismatch(str(mismatch), diagnostic) from None
        if preflight_only:
            return audit
    reports = {}
    violations = []
    for name in sorted(candidates):
        cand = candidates[name]
        reports[name] = _compare_evaluated(base, cand, metrics, method=method, seed=seed,
                                         samples=samples, confidence=confidence, thresholds=policy["effective_thresholds"][name], top=top)
        violations.extend({"candidate": name, **v} for v in reports[name]["violations"])
    rows = [row for report in reports.values() for row in report["metrics"].values()]
    for row, adjusted in zip(rows, adjust_p_values([row["p_value"] for row in rows], policy["correction"])):
        row["adjusted_p_value"] = adjusted
        row["significant"] = adjusted <= policy["alpha"]
    return {"schema_version": 1, "command": "compare-many", "passed": not violations,
            "query_count": base["query_count"], "gain": base["gain"], "tie_rule": base["tie_rule"],
            "policy": policy,
            "candidates": reports, "violations": violations}
