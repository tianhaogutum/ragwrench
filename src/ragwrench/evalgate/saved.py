"""Versioned, self-contained evaluation artifacts with canonical SHA-256 bindings."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from . import __version__
from .formats import InputError, decode, identifier
from .metrics import TIE_RULE, evaluate, parse_metrics

QUERY_POLICY = "all judged queries; missing run queries score zero; extra run queries ignored"


def canonical_sha256(value):
    """Hash sorted compact ASCII JSON, not source bytes, paths or timestamps."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def _artifact_sha256(report):
    manifest = {k: v for k, v in report["manifest"].items() if k != "artifact_sha256"}
    return canonical_sha256({**report, "manifest": manifest})


def saved_evaluation(qrels, run, metrics, gain="linear"):
    """Evaluate and bind the complete score payload, inputs and scoring settings."""
    metrics = sorted(metrics, key=str)
    if not metrics:
        raise InputError("at least one metric is required")
    return bind_evaluation(qrels, run, evaluate(qrels, run, metrics, gain))


def bind_evaluation(qrels, run, evaluation):
    """Build a canonical artifact from an already-evaluated payload, without scoring.

    Internal callers supply the complete evaluation of these parsed inputs.
    The returned artifact owns its manifest and leaves the payload unchanged.
    """
    report = {"schema_version": 1, "command": "eval", **evaluation}
    report["manifest"] = {
        "schema_version": 1,
        "evaluator": {"name": "retgate", "version": __version__},
        "dataset": {"sha256": canonical_sha256(qrels), "query_ids": sorted(qrels)},
        "run": {"sha256": canonical_sha256({q: {d: float(s) for d, s in docs.items()}
                                             for q, docs in run.items()})},
        "parameters": {"metrics": sorted(evaluation["means"]), "gain": evaluation["gain"],
                       "tie_rule": TIE_RULE, "query_policy": QUERY_POLICY},
    }
    report["manifest"]["artifact_sha256"] = _artifact_sha256(report)
    return report


def _require(condition, message):
    if not condition:
        raise InputError(f"invalid saved evaluation: {message}")


def _keys(value, expected, label):
    _require(isinstance(value, dict) and set(value) == set(expected), f"{label} fields must be {', '.join(expected)}")


def _ids(value, label):
    _require(isinstance(value, list), f"{label} must be an ID array")
    for qid in value:
        identifier(qid, "saved evaluation", 1, label)
    _require(value == sorted(set(value)), f"{label} must be sorted and unique")
    return set(value)


def _digest(value, label):
    _require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), f"{label} must be a SHA-256 hex digest")


def _score(value):
    # Avoid coercing booleans/strings or overflowing float conversions of huge JSON integers.
    return type(value) in (int, float) and 0 <= value <= 1 + 1e-12


def validate_evaluation(report):
    """Refuse incomplete/corrupt artifacts before any pairing or aggregation."""
    _require(isinstance(report, dict) and "manifest" in report,
             "manifest required; regenerate with ragwrench eval run --format json")
    _keys(report, ("schema_version", "command", "query_count", "gain", "tie_rule", "means",
                   "per_query", "missing_queries", "ignored_queries", "manifest"), "evaluation")
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1 and report["command"] == "eval",
             "expected schema_version 1 eval report")
    manifest = report["manifest"]
    _keys(manifest, ("schema_version", "evaluator", "dataset", "run", "parameters", "artifact_sha256"), "manifest")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1, "unsupported manifest schema")
    _require(manifest["evaluator"] == {"name": "retgate", "version": __version__}, "unsupported evaluator/version")
    _keys(manifest["dataset"], ("sha256", "query_ids"), "dataset")
    _keys(manifest["run"], ("sha256",), "run")
    for label, digest in (("dataset", manifest["dataset"]["sha256"]),
                          ("run", manifest["run"]["sha256"]), ("artifact", manifest["artifact_sha256"])):
        _digest(digest, label)
    qids = _ids(manifest["dataset"]["query_ids"], "judged query IDs")
    _require(qids and type(report["query_count"]) is int and report["query_count"] == len(qids), "query count differs from judged query set")
    params = manifest["parameters"]
    _keys(params, ("metrics", "gain", "tie_rule", "query_policy"), "parameters")
    names = params["metrics"]
    _require(isinstance(names, list) and names and all(isinstance(m, str) for m in names), "metrics must be a nonempty array")
    metrics = parse_metrics(",".join(names))
    _require(names == sorted(str(m) for m in metrics), "metrics must be canonical and sorted")
    _require(params["gain"] in ("linear", "exponential") and params["gain"] == report["gain"], "gain inconsistent")
    _require(params["tie_rule"] == report["tie_rule"] == TIE_RULE and params["query_policy"] == QUERY_POLICY,
             "unsupported ranking/query policy")
    missing = _ids(report["missing_queries"], "missing queries")
    ignored = _ids(report["ignored_queries"], "ignored queries")
    _require(missing < qids and not ignored & qids, "missing/ignored queries inconsistent with judged query set")
    scores = report["per_query"]
    _require(isinstance(scores, dict), "per_query must be an object")
    _require(set(scores) == qids,
             f"per-query data must exactly cover judged query set; missing={sorted(qids - set(scores))}, extra={sorted(set(scores) - qids)}")
    _keys(report["means"], names, "means")
    for qid, row in scores.items():
        _keys(row, names, f"per_query[{qid!r}]")
        _require(all(_score(v) for v in row.values()), f"per_query[{qid!r}] scores must be finite numbers in [0, 1]")
        _require(qid not in missing or all(v == 0 for v in row.values()), f"missing query {qid!r} must score zero")
    for name, mean in report["means"].items():
        _require(_score(mean) and mean == math.fsum(scores[q][name] for q in sorted(qids)) / len(qids),
                 f"mean {name} inconsistent with complete per-query data")
    _require(manifest["artifact_sha256"] == _artifact_sha256(report), "artifact checksum mismatch")
    return metrics


def read_evaluation(path):
    report = decode(Path(path).read_text(encoding="utf-8"), path)
    try:
        validate_evaluation(report)
    except InputError as exc:
        raise InputError(f"{path}: {exc}") from None
    return report
