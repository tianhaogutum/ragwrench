"""Strict, typed predicates and conservative Qdrant prefilters.

Native filters are only a superset optimization. The same predicate is always
applied afterwards, including to array-valued metadata and missing fields.
"""
import math
from datetime import datetime, timezone

from .errors import invalid

OPS = frozenset({"contains", "not contains", "start with", "end with", "is", "is not",
                 "in", "not in", "empty", "not empty", "=", "≠", ">", "<", "≥", "≤",
                 "before", "after"})
NUMERIC = {"=", "≠", ">", "<", "≥", "≤"}


def number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def date(value):
    if not isinstance(value, str):
        raise ValueError("date must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Dates and naive datetimes have an explicit UTC interpretation.
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


class Predicate:
    def __init__(self, raw, allowed):
        self.conditions = []
        self.logical = "and"
        if raw is None:
            return
        if not isinstance(raw, dict) or set(raw) - {"logical_operator", "conditions"}:
            raise invalid("metadata_condition must contain conditions and optional logical_operator")
        self.logical = raw.get("logical_operator", "and")
        conditions = raw.get("conditions")
        if self.logical not in ("and", "or") or not isinstance(conditions, list) or len(conditions) > 64:
            raise invalid("Expected and/or and at most 64 metadata conditions")
        for c in conditions:
            if not isinstance(c, dict) or set(c) - {"name", "comparison_operator", "value"}:
                raise invalid("Invalid metadata condition fields")
            name, op, value = c.get("name"), c.get("comparison_operator"), c.get("value")
            if not isinstance(name, str) or name not in allowed:
                raise invalid("Metadata field is not allowlisted")
            if not isinstance(op, str) or op not in OPS:
                raise invalid("Unknown comparison_operator")
            if op in NUMERIC and not number(value):
                raise invalid("Numeric comparison requires a finite number")
            if op in {"in", "not in"}:
                if not isinstance(value, list) or len(value) > 256 or not all(isinstance(v, str) and len(v) <= 4096 for v in value):
                    raise invalid("in/not in requires at most 256 strings of at most 4096 characters")
            elif op not in NUMERIC | {"empty", "not empty"} and not isinstance(value, str):
                raise invalid("String/date comparison requires a string")
            if isinstance(value, str) and len(value) > 4096:
                raise invalid("Comparison strings are limited to 4096 characters")
            if op in {"before", "after"}:
                try:
                    date(value)
                except ValueError:
                    raise invalid("Date comparison requires an ISO-8601 date") from None
            self.conditions.append((name, op, value))

    @staticmethod
    def compare(actual, op, value):
        if op == "empty":
            return actual is None or actual == "" or actual == [] or actual == {}
        if op == "not empty":
            return not Predicate.compare(actual, "empty", None)
        if op in NUMERIC:
            if not number(actual):
                return False
            return {"=": actual == value, "≠": actual != value, ">": actual > value,
                    "<": actual < value, "≥": actual >= value, "≤": actual <= value}[op]
        if op in {"before", "after"}:
            try:
                a, b = date(actual), date(value)
                return a < b if op == "before" else a > b
            except (ValueError, TypeError):
                return False
        # A missing or mismatched type does not satisfy even a negative operator.
        if op in {"in", "not in"}:
            if not isinstance(actual, str):
                return False
            return (actual in value) if op == "in" else (actual not in value)
        if op in {"contains", "not contains"} and isinstance(actual, list):
            return (value in actual) if op == "contains" else (value not in actual)
        if not isinstance(actual, str):
            return False
        return {"contains": value in actual, "not contains": value not in actual,
                "start with": actual.startswith(value), "end with": actual.endswith(value),
                "is": actual == value, "is not": actual != value}.get(op, False)

    def matches(self, metadata):
        values = [self.compare(metadata.get(name), op, value) for name, op, value in self.conditions]
        return (all(values) if self.logical == "and" else any(values)) if values else True

    def native(self, metadata_field):
        """Push scalar equality/list membership/ranges; leave text/date/negative to Python."""
        from qdrant_client import models as m
        terms = []
        for name, op, value in self.conditions:
            key = f"{metadata_field}.{name}"
            term = None
            if op == "is":
                term = m.FieldCondition(key=key, match=m.MatchValue(value=value))
            elif op == "in" and value:
                term = m.FieldCondition(key=key, match=m.MatchAny(any=value))
            elif op in {">", "<", "≥", "≤"} and abs(value) < 2**53:
                term = m.FieldCondition(key=key, range=m.Range(**{{">": "gt", "<": "lt", "≥": "gte", "≤": "lte"}[op]: value}))
            # Numeric equality remains a post-filter: Qdrant integer matching is
            # not guaranteed to include equivalent floating-point payload values.
            # Large numeric ranges also remain local to avoid float rounding loss.
            if term is None:
                if self.logical == "or":
                    return None
            else:
                terms.append(term)
        if not terms:
            return None
        return m.Filter(**{"must" if self.logical == "and" else "should": terms})
