"""Read-only comparison of saved resolution artifacts with independently replayed decisions."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .core import CiteError, MAX_JSON_BYTES, MAX_REGIONS, Region, parse_json, read_bytes
from .preflight import resolve_with_audit


class AuditMismatch(CiteError):
    """An artifact mismatch with a field path available to CLI consumers."""

    def __init__(self, path: str, problem: str):
        self.path = path
        super().__init__(f"{path}: {problem}; regenerate with ragwrench highlight resolve --audit "
                         "using the original PDF, regions and edited report")


def read_final_regions(path: Path) -> list[Any]:
    """Keep raw JSON fields/types: normalizing here could hide edited artifacts."""
    try:
        lines = read_bytes(path, MAX_JSON_BYTES).decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise CiteError("final regions must be UTF-8 JSONL") from exc
    result = []
    for line_no, line in enumerate(lines, 1):
        if line.strip():
            try:
                result.append(parse_json(line))
            except CiteError as exc:
                raise CiteError(f"final regions line {line_no}: {exc}") from exc
            if len(result) > MAX_REGIONS:
                raise CiteError(f"final regions exceed {MAX_REGIONS} entries")
    return result


def _match(actual: Any, expected: Any, path: str) -> None:
    """The regenerated artifact defines exact keys, array order and scalar types."""
    problem = None
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            problem = "object fields differ"
        else:
            for key, value in expected.items():
                _match(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            problem = f"expected {len(expected)} ordered entries"
        else:
            for i, (left, right) in enumerate(zip(actual, expected)):
                _match(left, right, f"{path}[{i}]")
    elif isinstance(expected, float):
        # Region coordinates are numbers; integer versions/counts/pages are strict.
        if (type(actual) not in (int, float) or actual != expected
                or not math.isfinite(actual)):
            problem = "numeric value differs"
    elif type(actual) is not type(expected) or actual != expected:
        problem = "value or type differs"
    if problem:
        raise AuditMismatch(path, problem)


def check_audit(reader: PdfReader, regions: list[Region], resolutions: Any,
                audit: Any, final_regions: list[Any]) -> dict:
    """Revalidate authorization, then require both saved artifacts to agree."""
    final, expected = resolve_with_audit(reader, regions, resolutions)
    _match(audit, expected, "audit")
    _match(final_regions, [region.to_dict() for region in final], "final regions")
    return expected["summary"]
