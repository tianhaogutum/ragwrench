"""Read-only plan comparison and deterministic, plan-bound review receipts."""
from __future__ import annotations

import math

from .core import SyncError, canonical, digest, validate_plan

GATES = ("max_deletion_fraction", "embed_model_changed")
CATEGORIES = ("upsert", "delete", "unchanged", "payload_only", "moved")


def deletion_fraction(plan):
    # Moves retire IDs but retain their content, so are not content deletions.
    count = len(plan["old"]["chunks"])
    return len(plan["delete"]) / count if count else 0.0


def compare_plans(baseline, candidate):
    """Compare competing plans from exactly the same source, without any I/O."""
    validate_plan(baseline)
    validate_plan(candidate)
    if baseline["old"]["fingerprint"] != candidate["old"]["fingerprint"]:
        raise SyncError("plan comparison requires the same source snapshot")

    def keys(plan, category):
        if category == "moved":
            return {(m["before"]["id"], m["after"]["id"]): m for m in plan[category]}
        return {c["id"]: c for c in plan[category]}

    operations = {}
    for category in CATEGORIES:
        before, after = keys(baseline, category), keys(candidate, category)
        operations[category] = {"added": sorted(after.keys() - before.keys()),
                                "removed": sorted(before.keys() - after.keys()),
                                "changed": sorted(k for k in before.keys() & after.keys()
                                                  if canonical(before[k]) != canonical(after[k]))}
    return {"kind": "chunksync.comparison", "version": 1,
            "source_fingerprint": baseline["old"]["fingerprint"],
            "baseline_fingerprint": baseline["fingerprint"],
            "candidate_fingerprint": candidate["fingerprint"],
            "identical": baseline["fingerprint"] == candidate["fingerprint"],
            "summary_delta": {k: candidate["summary"][k] - baseline["summary"][k]
                              for k in (*CATEGORIES, "new_chunks", "embeddings_avoided")},
            "deletion_fraction": {"baseline": deletion_fraction(baseline), "candidate": deletion_fraction(candidate)},
            "embed_model": {"source": baseline["old"]["embed_model"],
                            "baseline": baseline["new"]["embed_model"],
                            "candidate": candidate["new"]["embed_model"]},
            "operations": operations}


def review_plan(plan, *, max_deletion_fraction=1.0, overrides=(), reviewer=None, reason=None):
    """Evaluate gates; overrides are exact violations with caller-stated provenance.

    A receipt is an audit record, not a signature or an authorization boundary.
    Reproducibility lets the same receipt bind retries without timestamp drift.
    """
    validate_plan(plan)
    if (type(max_deletion_fraction) not in (int, float) or not 0 <= max_deletion_fraction <= 1
            or not math.isfinite(max_deletion_fraction)):
        raise SyncError("max-deletion-fraction must be finite and between 0 and 1")
    if (not isinstance(overrides, (list, tuple)) or any(g not in GATES for g in overrides)
            or len(set(overrides)) != len(overrides)):
        raise SyncError("overrides must name unique supported review gates")
    fraction = deletion_fraction(plan)
    violations = []
    if fraction > max_deletion_fraction:
        violations.append("max_deletion_fraction")
    if plan["summary"]["embed_model_changed"]:
        violations.append("embed_model_changed")
    if set(overrides) - set(violations):
        raise SyncError("override must name a gate violated by this exact plan and policy")
    if overrides:
        if any(not isinstance(v, str) or not v.strip() for v in (reviewer, reason)):
            raise SyncError("review overrides require a nonempty reviewer and reason")
        if len(reviewer) > 256 or len(reason) > 4096:
            raise SyncError("reviewer/reason exceed 256/4096 character limits")
    elif reviewer is not None or reason is not None:
        raise SyncError("reviewer and reason require an explicit override gate")
    blocked = [g for g in violations if g not in overrides]
    receipt = {"kind": "chunksync.review", "version": 1, "plan_fingerprint": plan["fingerprint"],
               "policy": {"max_deletion_fraction": float(max_deletion_fraction)},
               "observed": {"old_chunks": len(plan["old"]["chunks"]), "deletions": len(plan["delete"]),
                            "deletion_fraction": fraction, "old_embed_model": plan["old"]["embed_model"],
                            "new_embed_model": plan["new"]["embed_model"]},
               "violations": violations, "blocked": blocked, "allowed": not blocked,
               "overrides": sorted(overrides),
               "provenance": {"reviewer": reviewer, "reason": reason} if overrides else None}
    receipt["fingerprint"] = digest(receipt)
    return receipt


def validate_review(plan, receipt):
    """Recompute all observations, policy decisions, and plan/provenance hashes."""
    try:
        provenance = receipt["provenance"]
        expected = review_plan(plan, max_deletion_fraction=receipt["policy"]["max_deletion_fraction"],
                               overrides=receipt["overrides"],
                               reviewer=provenance["reviewer"] if provenance is not None else None,
                               reason=provenance["reason"] if provenance is not None else None)
        if canonical(receipt) != canonical(expected):
            raise SyncError("review receipt mismatch; regenerate for this exact plan")
        return receipt
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SyncError):
            raise
        raise SyncError(f"invalid review receipt: {exc}") from exc


def require_allowed(receipt):
    if not receipt["allowed"]:
        raise SyncError("review blocked: " + ", ".join(receipt["blocked"]))
