"""Validated interchange types, bounded local I/O, and PDF coordinate geometry."""
from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_PDF_BYTES = 128 * 1024 * 1024
MAX_REGIONS = 100_000


class CiteError(ValueError):
    """Invalid or unsupported input; safe to display without a traceback."""


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CiteError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise CiteError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise CiteError(f"{label} must be {'positive and ' if positive else ''}finite")
    return result


def page_number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CiteError("page must be a positive 1-based integer")
    return value


def identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 8192:
        raise CiteError(f"{label} must be a nonempty string (at most 8192 characters)")
    return value


def obj(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise CiteError(f"{label} must be an object")
    return value


def seq(value: Any, label: str, *, nonempty: bool = True) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        raise CiteError(f"{label} must be a {'nonempty ' if nonempty else ''}array")
    return value


def text_value(value: Any) -> str:
    if not isinstance(value, str):
        raise CiteError("text must be a string")
    return value


def parse_json(data: str | bytes) -> Any:
    def bad_constant(token: str) -> None:
        raise CiteError(f"nonfinite JSON constant: {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise CiteError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data, parse_constant=bad_constant, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CiteError(f"invalid JSON: {exc}") from exc


def read_bytes(path: Path, limit: int) -> bytes:
    # Read limit+1 as well as checking stat, so growing/special files stay bounded.
    if not path.is_file() or path.stat().st_size > limit:
        raise CiteError(f"input must be a regular file of at most {limit} bytes: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise CiteError(f"input exceeds {limit} bytes: {path}")
    return data


def read_json(path: Path) -> Any:
    return parse_json(read_bytes(path, MAX_JSON_BYTES))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Region:
    chunk_id: str
    source_ref: str
    page: int
    rect: tuple[float, float, float, float]
    text: str = ""
    source_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Any) -> Region:
        value = obj(value, "region")
        rect = seq(value.get("rect"), "rect")
        if len(rect) != 4:
            raise CiteError("rect must contain four numbers")
        coords = tuple(number(n, "rect coordinate") for n in rect)
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            raise CiteError("rect must have positive width and height: [x0,y0,x1,y1]")
        refs = value.get("source_refs", [])
        seq(refs, "source_refs", nonempty=False)
        source_ref = identifier(value.get("source_ref"), "source_ref")
        normalized_refs = tuple(sorted(set(identifier(r, "source_refs entry") for r in refs)))
        if normalized_refs and source_ref not in normalized_refs:
            raise CiteError("source_refs must include source_ref")
        return cls(identifier(value.get("chunk_id"), "chunk_id"),
                   source_ref,
                   page_number(value.get("page")), coords,
                   text_value(value.get("text", "")),
                   normalized_refs)

    def to_dict(self) -> dict:
        result = {"chunk_id": self.chunk_id, "source_ref": self.source_ref,
                  "page": self.page, "rect": list(self.rect), "text": self.text}
        if self.source_refs:
            result["source_refs"] = list(self.source_refs)
        return result


def read_regions(path: Path, *, allow_empty: bool = False) -> list[Region]:
    try:
        lines = read_bytes(path, MAX_JSON_BYTES).decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise CiteError("regions must be UTF-8 JSONL") from exc
    result = []
    for line_no, line in enumerate(lines, 1):
        if line.strip():
            try:
                result.append(Region.from_dict(parse_json(line)))
            except CiteError as exc:
                raise CiteError(f"regions line {line_no}: {exc}") from exc
    if (not result and not allow_empty) or len(result) > MAX_REGIONS:
        raise CiteError(f"expected {0 if allow_empty else 1}..{MAX_REGIONS} regions")
    if len({canonical(r.to_dict()) for r in result}) != len(result):
        raise CiteError("duplicate identical regions are not allowed")
    return result


def regions_jsonl(regions: list[Region], *, allow_empty: bool = False) -> str:
    if (not regions and not allow_empty) or len(regions) > MAX_REGIONS:
        raise CiteError(f"expected {0 if allow_empty else 1}..{MAX_REGIONS} regions")
    # Normalize repeat provenance entries deterministically.
    unique = dict.fromkeys(canonical(r.to_dict()) for r in regions)
    return "\n".join(unique) + ("\n" if unique else "")


def preflight_outputs(outputs: list[Path], inputs: list[Path], force: bool = False) -> None:
    def aliases(a: Path, b: Path) -> bool:
        return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))

    if any(aliases(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1:]):
        raise CiteError("output paths must be distinct")
    for path in outputs:
        if any(aliases(path, source) for source in inputs):
            raise CiteError(f"output must not overwrite an input: {path}")
        if not path.parent.is_dir():
            raise CiteError(f"output parent directory does not exist: {path.parent}")
        if path.exists() and (not force or not path.is_file()):
            raise CiteError(f"output already exists (use --force for files): {path}")


def atomic_write(path: Path, write: Callable) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".citehl-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_text(path: Path, content: str) -> None:
    data = content.encode("utf-8")
    atomic_write(path, lambda stream: stream.write(data))


def open_pdf(path: Path) -> PdfReader:
    import io
    from pypdf.errors import PdfReadError
    try:
        reader = PdfReader(io.BytesIO(read_bytes(path, MAX_PDF_BYTES)), strict=True)
        if reader.is_encrypted:
            raise CiteError("encrypted PDFs are unsupported, including empty-password encryption")
        if not reader.pages:
            raise CiteError("PDF has no pages")
        return reader
    except CiteError:
        raise
    except (PdfReadError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise CiteError(f"invalid PDF: {exc}") from exc


@dataclass(frozen=True)
class Geometry:
    media: tuple[float, float, float, float]
    crop: tuple[float, float, float, float]
    rotation: int = 0

    @classmethod
    def from_page(cls, page: Any) -> Geometry:
        media = tuple(number(float(v), "MediaBox") for v in page.mediabox)
        crop = tuple(number(float(v), "CropBox") for v in page.cropbox)
        for box in (media, crop):
            if box[2] <= box[0] or box[3] <= box[1]:
                raise CiteError("PDF page box has nonpositive dimensions")
        raw_rotation = number(page.get("/Rotate", 0), "PDF /Rotate")
        if raw_rotation % 90:
            raise CiteError("only /Rotate multiples of 90 are supported")
        if number(page.get("/UserUnit", 1), "PDF /UserUnit") != 1:
            raise CiteError("PDF /UserUnit other than 1 is unsupported")
        return cls(media, crop, int(raw_rotation) % 360)

    @property
    def visible(self) -> tuple[float, float, float, float]:
        return (max(self.media[0], self.crop[0]), max(self.media[1], self.crop[1]),
                min(self.media[2], self.crop[2]), min(self.media[3], self.crop[3]))

    def dimensions(self, box: str, frame: str) -> tuple[float, float]:
        x0, y0, x1, y1 = getattr(self, box)
        width, height = x1 - x0, y1 - y0
        return (height, width) if frame == "display" and self.rotation in (90, 270) else (width, height)

    def to_pdf(self, x: float, y: float, width: float, height: float,
               origin: str, box: str, frame: str) -> tuple[float, float]:
        """Map source coordinates to *unrotated*, absolute PDF user space."""
        bx, by, right, top = getattr(self, box)
        w, h = right - bx, top - by
        dw, dh = self.dimensions(box, frame)
        # Source -> bottom-left coordinates of the chosen frame.
        u, v = x / width * dw, (height - y if origin == "TOPLEFT" else y) / height * dh
        rotation = self.rotation if frame == "display" else 0
        if rotation == 0:
            px, py = u, v
        elif rotation == 90:
            px, py = w - v, u
        elif rotation == 180:
            px, py = w - u, h - v
        else:  # 270
            px, py = v, h - u
        return bx + px, by + py

    def clamp(self, rect: tuple, *, strict: bool) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = rect
        left, bottom, right, top = self.visible
        clipped = (max(x0, left), max(y0, bottom), min(x1, right), min(y1, top))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise CiteError(f"bbox is wholly outside visible page box: {rect}")
        if any(abs(a - b) > 1e-6 for a, b in zip(rect, clipped)):
            message = f"bbox outside visible page box; clamped {rect} to {clipped}"
            if strict:
                raise CiteError(message)
            warnings.warn(message, stacklevel=2)
        return clipped


def geometries(reader: PdfReader) -> dict[int, Geometry]:
    return {i: Geometry.from_page(page) for i, page in enumerate(reader.pages, 1)}


def validate_regions(regions: list[Region], geometry: dict[int, Geometry], *, strict: bool) -> list[Region]:
    from dataclasses import replace
    result = []
    for r in regions:
        if r.page not in geometry:
            raise CiteError(f"page {r.page} outside PDF (1..{len(geometry)})")
        result.append(replace(r, rect=geometry[r.page].clamp(r.rect, strict=strict)))
    return result
