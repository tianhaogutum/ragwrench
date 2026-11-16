"""Collection deletion policy and separate, plan-bound policy exceptions."""
from __future__ import annotations

import math
import re

from .core import SyncError, canonical, digest, validate_plan


def _limit(value):
    if (type(value) not in (int, float) or not 0 <= value <= 1
            or not math.isfinite(value)):
        raise SyncError("collection review policy max-deletion-fraction must be finite and between 0 and 1")
    return float(value)


def policy_override(plan, collection, *, from_max_deletion_fraction, max_deletion_fraction, reviewer, reason):
    """Create an unsigned audit record, independently of a plan-gate override."""
    validate_plan(plan)
    return _override_record(plan["fingerprint"], collection,
                            from_max_deletion_fraction=from_max_deletion_fraction,
                            max_deletion_fraction=max_deletion_fraction, reviewer=reviewer, reason=reason)


def _override_record(plan_fingerprint, collection, *, from_max_deletion_fraction, max_deletion_fraction, reviewer, reason):
    if not isinstance(plan_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_fingerprint):
        raise SyncError("invalid policy override plan fingerprint")
    if not isinstance(collection, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", collection):
        raise SyncError("invalid policy override collection name")
    before, after = _limit(from_max_deletion_fraction), _limit(max_deletion_fraction)
    if after <= before:
        raise SyncError("policy override requires a strictly weaker deletion limit")
    if any(not isinstance(v, str) or not v.strip() for v in (reviewer, reason)):
        raise SyncError("policy override requires a nonempty reviewer and reason")
    if len(reviewer) > 256 or len(reason) > 4096:
        raise SyncError("policy override reviewer/reason exceed 256/4096 character limits")
    record = {"kind": "chunksync.policy_override", "version": 1, "collection": collection,
              "plan_fingerprint": plan_fingerprint,
              "from_policy": {"max_deletion_fraction": before},
              "to_policy": {"max_deletion_fraction": after},
              "provenance": {"reviewer": reviewer, "reason": reason}}
    record["fingerprint"] = digest(record)
    return record


def stored_policy_override(collection, record):
    """Validate a historical record's shape/hash, without its unavailable plan.

    This does not authenticate provenance or establish the current policy:
    a later transaction may have tightened it since the exception was admitted.
    """
    if record is None:
        return None
    try:
        expected = _override_record(
            record["plan_fingerprint"], collection,
            from_max_deletion_fraction=record["from_policy"]["max_deletion_fraction"],
            max_deletion_fraction=record["to_policy"]["max_deletion_fraction"],
            reviewer=record["provenance"]["reviewer"], reason=record["provenance"]["reason"])
        if canonical(record) != canonical(expected):
            raise SyncError("invalid stored policy override record or fingerprint")
        return expected
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SyncError):
            raise
        raise SyncError("invalid stored policy override record") from exc


def validate_policy_override(plan, collection, record, policy):
    try:
        expected = policy_override(
            plan, collection, from_max_deletion_fraction=record["from_policy"]["max_deletion_fraction"],
            max_deletion_fraction=record["to_policy"]["max_deletion_fraction"],
            reviewer=record["provenance"]["reviewer"], reason=record["provenance"]["reason"])
        if canonical(record) != canonical(expected) or record["to_policy"] != policy:
            raise SyncError("policy override mismatch; use the exact plan, collection and requested policy")
        return record
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SyncError):
            raise
        raise SyncError("invalid policy override record") from exc


def collection_policy(state):
    """Legacy collections inherit their latest recorded limit, else compatibility 1.0."""
    try:
        policy = (state["review_policy"] if "review_policy" in state else
                  state["last_review"]["policy"] if state.get("last_review") is not None else
                  {"max_deletion_fraction": 1.0})
        if not isinstance(policy, dict) or set(policy) != {"max_deletion_fraction"}:
            raise SyncError("invalid collection review policy")
        return {"max_deletion_fraction": _limit(policy["max_deletion_fraction"])}
    except (KeyError, TypeError) as exc:
        raise SyncError("invalid collection review policy") from exc


def admit_policy(state, review, override, *, replay=False):
    """Check live policy without mutating state; retries must reuse their binding."""
    current, requested = collection_policy(state), review["policy"]
    if override is not None:
        # Admission already persisted before a point-call failure or final commit.
        recorded = (state.get("last_policy_override") == override
                    and state.get("last_review") == review and current == requested)
        if not recorded and (replay or override["from_policy"] != current
                             or requested["max_deletion_fraction"] <= current["max_deletion_fraction"]):
            raise SyncError("policy override does not match the current collection review policy")
    elif requested["max_deletion_fraction"] > current["max_deletion_fraction"]:
        raise SyncError("collection review policy cannot be weakened without a separate policy override")
    return current if replay else requested
