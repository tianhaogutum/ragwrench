"""Standard PDF highlight annotations, deep links, and independent readback checks."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DictionaryObject, FloatObject, NameObject,
                           NumberObject, TextStringObject)

from .core import (CiteError, Geometry, MAX_JSON_BYTES, Region, atomic_write, canonical,
                   obj, parse_json)

PREFIX = "citehl:v1:"
METADATA_KEY = "/CiteHL"


def embedded_region(annotation) -> Region | None:
    """Validate present metadata, including its binding to the PDF annotation ID.

    Absence is allowed for legacy annotations; callers decide whether to require it.
    Numeric JSON representations are retained because v1 IDs hash exact region JSON.
    """
    if METADATA_KEY not in annotation:
        return None
    try:
        raw = annotation[METADATA_KEY]
        if not isinstance(raw, TextStringObject):
            raise CiteError("payload must be a PDF text string")
        if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
            raise CiteError("payload exceeds 32 MiB")
        data = obj(parse_json(str(raw)), "payload")
        if (set(data) != {"format", "schema_version", "annotation_id", "region"}
                or data.get("format") != "citehl-region"
                or type(data.get("schema_version")) is not int or data["schema_version"] != 1):
            raise CiteError("unsupported region format/schema_version or fields")
        region = Region.from_dict(data["region"])
        if data["region"] != region.to_dict():
            raise CiteError("region must contain only normalized region fields")
        region = replace(region, rect=tuple(data["region"]["rect"]))
        name = annotation_id(region)
        # Dictionary indexing resolves PDF references; get() returns the stored
        # reference. Identity must be the persisted text string, without mutation.
        persisted = annotation["/NM"] if "/NM" in annotation else None
        if (not isinstance(persisted, TextStringObject)
                or data["annotation_id"] != name or persisted != name):
            raise CiteError("annotation_id/region/PDF /NM mismatch")
        return region
    except CiteError as exc:
        raise CiteError(f"invalid citehl region metadata: {exc}") from exc


def region_metadata(r: Region) -> str:
    """Serialize validated v1 provenance without changing stable region IDs."""
    if Region.from_dict(r.to_dict()) != r:
        raise CiteError("region metadata requires a normalized region")
    data = canonical({"format": "citehl-region", "schema_version": 1,
                      "annotation_id": annotation_id(r), "region": r.to_dict()})
    if len(data.encode("utf-8")) > MAX_JSON_BYTES:
        raise CiteError("region metadata exceeds 32 MiB")
    return data


def annotation_id(region: Region) -> str:
    return PREFIX + hashlib.sha256(canonical(region.to_dict()).encode("utf-8")).hexdigest()


def contents(region: Region) -> str:
    return f"{region.chunk_id}\n{region.text}"


def quad_points(rect: tuple) -> list[float]:
    x0, y0, x1, y1 = rect
    # Acrobat's widely supported Z order: upper-left, upper-right, lower-left, lower-right.
    return [x0, y1, x1, y1, x0, y0, x1, y0]


def floats(values) -> ArrayObject:
    return ArrayObject([FloatObject(v) for v in values])


def annotate(reader: PdfReader, regions: list[Region], output: Path) -> None:
    writer = PdfWriter(clone_from=reader)
    # Refresh our namespace, preserving unrelated annotation objects and original content.
    for page in writer.pages:
        if "/Annots" in page:
            retained = ArrayObject([a for a in page["/Annots"]
                                    if not str(a.get_object().get("/NM", "")).startswith(PREFIX)])
            page[NameObject("/Annots")] = retained
    for r in regions:
        add_highlight(writer, r)
    atomic_write(output, writer.write)


def add_highlight(writer: PdfWriter, r: Region) -> None:
    """Append one standard highlight without removing any existing annotation."""
    metadata = region_metadata(r)
    if r.page > len(writer.pages):
        raise CiteError(f"page {r.page} outside PDF")
    Geometry.from_page(writer.pages[r.page - 1]).clamp(r.rect, strict=True)
    annotation = DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Highlight"),
        NameObject("/Rect"): floats(r.rect),
        NameObject("/QuadPoints"): floats(quad_points(r.rect)),
        NameObject("/Contents"): TextStringObject(contents(r)),
        NameObject("/NM"): TextStringObject(annotation_id(r)),
        NameObject(METADATA_KEY): TextStringObject(metadata),
        NameObject("/T"): TextStringObject("citehl"),
        NameObject("/C"): floats([1, 1, 0]),
        NameObject("/CA"): FloatObject(0.35),
        NameObject("/F"): NumberObject(4),  # printable
    })
    writer.add_annotation(page_number=r.page - 1, annotation=annotation)


def manifest(regions: list[Region], pdf_name: str) -> dict:
    groups = defaultdict(list)
    for r in regions:
        groups[r.chunk_id].append(r)
    chunks = {}
    for chunk in sorted(groups):
        entries = groups[chunk]
        pages = sorted({r.page for r in entries})
        chunks[chunk] = {
            "pages": pages,
            "rects": [{"page": r.page, "rect": list(r.rect), "source_ref": r.source_ref,
                       "source_refs": list(r.source_refs or (r.source_ref,)),
                       "annotation_id": annotation_id(r)} for r in entries],
            "links": [f"{quote(pdf_name, safe='')}#page={page}" for page in pages],
        }
    return {"schema_version": 1, "coordinate_space": "unrotated-pdf-user-space",
            "pdf": pdf_name, "chunks": chunks}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE))


def text_fragments(page) -> list[tuple[str, tuple]]:
    """Best-effort word boxes for ordinary horizontal simple-font PDF text.

    Font widths are used when available; missing built-in font widths use 0.5em.
    Complex fonts, nonhorizontal matrices, and unknown glyphs are skipped. This is
    evidence for a text cross-check, not a general PDF text layout/OCR engine.
    """
    result = []

    def visitor(text, cm, tm, font, size):
        if not text.strip() or font is None or font.get("/Subtype") not in ("/Type1", "/TrueType"):
            return
        # PDF text matrix followed by current transformation matrix.
        a = tm[0] * cm[0] + tm[1] * cm[2]
        b = tm[0] * cm[1] + tm[1] * cm[3]
        c = tm[2] * cm[0] + tm[3] * cm[2]
        d = tm[2] * cm[1] + tm[3] * cm[3]
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        if abs(b) > 1e-6 or abs(c) > 1e-6 or a <= 0 or d <= 0 or size <= 0:
            return
        widths = font.get("/Widths")
        if widths is not None:
            widths = widths.get_object()
        first = int(font.get("/FirstChar", 0))
        cursor = x
        for word in re.finditer(r"\S+|\s+", text):
            fragment = word.group()
            advance = 0.0
            for char in fragment:
                if char in "\r\n":
                    continue
                index = ord(char) - first
                width = float(widths[index]) if widths is not None and 0 <= index < len(widths) else 500
                advance += width / 1000 * size * a
            if fragment.strip():
                result.append((fragment, (cursor, y - size * d * 0.2, cursor + advance, y + size * d * 0.8)))
            cursor += advance
            if "\n" in fragment or "\r" in fragment:
                cursor = x
                y -= size * d * 1.2

    page.extract_text(visitor_text=visitor)
    return result


def overlap(a: tuple, b: tuple) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def annotation_errors(annotation, page_no: int, r: Region, tolerance: float) -> list[str]:
    """Check one annotation's identity, geometry, contents, and visibility."""
    name = annotation_id(r)
    errors = []
    try:
        embedded_region(annotation)
    except CiteError as exc:
        errors.append(str(exc))
    if page_no != r.page or annotation.get("/Subtype") != "/Highlight":
        errors.append(f"wrong page or subtype for {name}")
    for key, desired in (("/Rect", r.rect), ("/QuadPoints", quad_points(r.rect))):
        actual = annotation.get(key, [])
        try:
            matches = len(actual) == len(desired) and all(
                abs(float(a) - b) <= tolerance for a, b in zip(actual, desired))
        except (ValueError, TypeError, OverflowError):
            matches = False
        if not matches:
            errors.append(f"{key} mismatch for {name}")
    if annotation.get("/Contents") != contents(r):
        errors.append(f"/Contents mismatch for {name}")
    try:
        flags = int(annotation.get("/F", 0))
        opacity = float(annotation.get("/CA", 1))
        if flags & (1 | 2 | 32) or opacity <= 0:
            errors.append(f"hidden/invisible annotation {name}")
    except (ValueError, TypeError):
        errors.append(f"invalid visibility for {name}")
    return errors


def verify(reader: PdfReader, regions: list[Region], *, tolerance: float = 0.01,
           text_check: str = "best-effort") -> dict:
    expected = {annotation_id(r): r for r in regions}
    if len(expected) != len(regions):
        raise CiteError("duplicate region annotation IDs")
    found = {}
    errors = []
    for page_no, page in enumerate(reader.pages, 1):
        for ref in page.get("/Annots", []):
            annotation = ref.get_object()
            # A renamed/missing /NM must not conceal a metadata-bearing annotation.
            if METADATA_KEY in annotation:
                embedded_region(annotation)
            name = annotation["/NM"] if "/NM" in annotation else ""
            # A PDF name spelled with the citehl prefix is our namespace, not foreign.
            if isinstance(name, NameObject) and str(name).lstrip("/").startswith(PREFIX):
                raise CiteError("citehl annotation /NM must be a PDF text string")
            if not isinstance(name, TextStringObject) or not name.startswith(PREFIX):
                continue
            if name in found:
                errors.append(f"duplicate annotation {name}")
            found[name] = (page_no, annotation)
    for name, r in expected.items():
        if name not in found:
            errors.append(f"missing annotation for chunk {r.chunk_id}, page {r.page}")
            continue
        page_no, annotation = found[name]
        errors.extend(annotation_errors(annotation, page_no, r, tolerance))
    for name in found.keys() - expected.keys():
        errors.append(f"unexpected citehl annotation {name}")
    if errors:
        raise CiteError("verification failed: " + "; ".join(errors[:20]))
    checked, skipped = 0, 0
    skip_reasons = []
    if text_check != "off":
        cache = {}
        for r in regions:
            expected_words = tokens(r.text)
            if r.page not in cache:
                try:
                    cache[r.page] = text_fragments(reader.pages[r.page - 1])
                except Exception as exc:
                    # Text extraction is explicitly best effort; annotation integrity is not.
                    cache[r.page] = []
                    skip_reasons.append(f"page {r.page}: text extraction unavailable ({type(exc).__name__})")
            actual_words = tokens(" ".join(text for text, box in cache[r.page] if overlap(r.rect, box)))
            if not expected_words or not actual_words:
                skipped += 1
                if text_check == "required":
                    errors.append(f"text unavailable under chunk {r.chunk_id}, page {r.page}")
            elif not expected_words & actual_words:
                errors.append(f"text mismatch under chunk {r.chunk_id}, page {r.page}")
            else:
                checked += 1
    if errors:
        raise CiteError("verification failed: " + "; ".join(errors[:20]))
    return {"annotations": len(regions), "text_checked": checked, "text_skipped": skipped,
            "text_check": text_check, "notes": skip_reasons}
