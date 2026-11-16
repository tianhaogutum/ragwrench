"""Read-only manifest comparison after independent PDF annotation verification."""
from __future__ import annotations

from typing import Any

from pypdf import PdfReader

from .core import CiteError, Region
from .pdf import manifest, verify


class ManifestMismatch(CiteError):
    """An exact artifact mismatch with a path independent of diagnostic text."""

    def __init__(self, path: str, problem: str):
        self.path = path
        super().__init__(f"{path}: {problem}; regenerate with ragwrench highlight manifest using "
                         "the current PDF and final resolved regions in the same directory")


def _match(actual: Any, expected: Any, path: str) -> None:
    """Require the writer's exact fields, array order and scalar types."""
    problem = None
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            problem = "object fields differ"
        else:
            for key, value in expected.items():
                _match(actual[key], value, f"{path}[{key!r}]")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            problem = f"expected {len(expected)} ordered entries"
        else:
            for i, (left, right) in enumerate(zip(actual, expected)):
                _match(left, right, f"{path}[{i}]")
    elif isinstance(expected, float):
        # Coordinates accept equivalent JSON numbers, without PDF tolerance.
        # Equality to a finite expected coordinate also excludes NaN/infinity.
        if type(actual) not in (int, float) or actual != expected:
            problem = "numeric value differs"
    elif type(actual) is not type(expected) or actual != expected:
        problem = "value or type differs"
    if problem:
        raise ManifestMismatch(path, problem)


def check_manifest(reader: PdfReader, regions: list[Region], saved: Any, pdf_name: str,
                   *, tolerance: float = 0.01, text_check: str = "best-effort") -> dict:
    """Saved fields never supply expected geometry, IDs or review decisions."""
    report = verify(reader, regions, tolerance=tolerance, text_check=text_check)
    expected = manifest(regions, pdf_name)
    _match(saved, expected, "manifest")
    return {"annotations": report["annotations"], "chunks": len(expected["chunks"])}
