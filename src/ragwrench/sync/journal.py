"""Write-ahead progress for one deterministic, single-writer operation sequence."""
from __future__ import annotations

import re
import struct

from .core import SyncError, digest


def point_fingerprint(point):
    if point is None:
        return None
    return digest({"payload": point.payload,
                   "vector_float32": struct.pack(f"<{len(point.vector)}f", *point.vector).hex()})


def _fingerprint(hashes):
    return digest(sorted(hashes.items()))


def _is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class ApplyJournal:
    """A completed prefix plus at most one ambiguous point mutation.

    Persist the intent before calling Qdrant, then the checkpoint after it.
    On reopen, accept either the before or after hash for that one point and
    require every other point to match the last checkpoint. This also handles
    a call that persists successfully and then raises before acknowledging.
    No vectors or document text are duplicated in the journal.
    """

    def __init__(self, pending, binding, operation_ids, points, save):
        self.hashes = {pid: point_fingerprint(p) for pid, p in points.items()}
        self.save = save
        if pending is None:
            self.entry = {"version": 1, **binding, "next_operation": 0,
                          "checkpoint": _fingerprint(self.hashes), "in_flight": None}
            self.save(self.entry)
            return
        required = {"version", *binding, "next_operation", "checkpoint", "in_flight"}
        if not isinstance(pending, dict) or set(pending) != required or type(pending["version"]) is not int or pending["version"] != 1:
            raise SyncError("unsupported apply journal schema/version")
        if pending["plan_fingerprint"] != binding["plan_fingerprint"]:
            raise SyncError("a different plan is pending; apply journal binding mismatch")
        if pending.get("review_fingerprint") != binding.get("review_fingerprint"):
            raise SyncError("apply journal review binding mismatch; reuse the original review receipt")
        if any(pending[k] != v for k, v in binding.items()):
            raise SyncError("apply journal binding mismatch: plan or source hashes changed")
        self.entry = pending
        cursor, flight = self.cursor, pending["in_flight"]
        if type(cursor) is not int or not 0 <= cursor <= len(operation_ids) or not _is_hash(pending["checkpoint"]):
            raise SyncError("invalid apply journal progress")
        if flight is None:
            if _fingerprint(self.hashes) != pending["checkpoint"]:
                raise SyncError("apply journal checkpoint fingerprint mismatch")
            return
        if (not isinstance(flight, dict) or set(flight) != {"id", "before", "after"}
                or cursor == len(operation_ids) or flight["id"] != operation_ids[cursor]
                or any(v is not None and not _is_hash(v) for v in (flight["before"], flight["after"]))):
            raise SyncError("invalid apply journal in-flight operation")
        pid = flight["id"]
        actual = self.hashes.get(pid)
        before = dict(self.hashes)
        if flight["before"] is None:
            before.pop(pid, None)
        else:
            before[pid] = flight["before"]
        if actual not in (flight["before"], flight["after"]) or _fingerprint(before) != pending["checkpoint"]:
            raise SyncError("apply journal in-flight fingerprint mismatch")
        if actual == flight["after"]:
            self._advance()

    @property
    def cursor(self):
        return self.entry["next_operation"]

    def prepare(self, pid, after):
        """Return False when the intended state already exists (legacy resume)."""
        intent = {"id": pid, "before": self.hashes.get(pid), "after": after}
        if self.entry["in_flight"] is not None:
            if self.entry["in_flight"] != intent:
                raise SyncError("apply journal intended fingerprint mismatch; embedding source changed")
        elif intent["before"] == after:
            self._advance()
            return False
        else:
            self.entry["in_flight"] = intent
            self.save(self.entry)
        return True

    def complete(self, pid, actual):
        if actual != self.entry["in_flight"]["after"]:
            raise SyncError("apply journal write fingerprint mismatch")
        if actual is None:
            self.hashes.pop(pid, None)
        else:
            self.hashes[pid] = actual
        self._advance()

    def _advance(self):
        self.entry.update(next_operation=self.cursor + 1, checkpoint=_fingerprint(self.hashes), in_flight=None)
        self.save(self.entry)
