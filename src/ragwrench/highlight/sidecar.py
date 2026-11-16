"""Versioned portable citehl highlights bound to ordered PDF page fingerprints."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, BooleanObject, ByteStringObject, DictionaryObject,
                           FloatObject, IndirectObject, NameObject, NullObject, NumberObject,
                           StreamObject, TextStringObject)

from .core import (CiteError, MAX_JSON_BYTES, MAX_PDF_BYTES, MAX_REGIONS, Region,
                   atomic_write, canonical, geometries, number, obj, seq, validate_regions)
from .pdf import (METADATA_KEY, PREFIX, add_highlight, annotation_errors, annotation_id,
                  embedded_region, verify)

FORMAT = "citehl-annotation-sidecar"
ALGORITHM = "citehl-page-sha256-v1"
COORDINATE_SPACE = {
    "name": "unrotated-pdf-user-space", "units": "points", "origin": "bottom-left",
    "rect": "[x0,y0,x1,y1]", "page_index_base": 1,
}


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class _ObjectHasher:
    """Hash resolved, typed PDF values, independent of indirect object numbers.

    Container children are represented by their digests; dictionaries sort keys.
    Stream compression is excluded and decoded bytes are hashed instead. Cyclic
    resource graphs fail closed rather than silently omitting part of a page.
    """

    def __init__(self):
        self.memo: dict[int, str] = {}
        self.active: set[int] = set()
        self.nodes = 0
        self.stream_bytes = 0

    def hash(self, value: Any, depth: int = 0) -> str:
        if depth > 100:
            raise CiteError("page fingerprint object depth exceeds 100")
        if isinstance(value, IndirectObject):
            return self.hash(value.get_object(), depth + 1)
        if isinstance(value, (DictionaryObject, ArrayObject)):
            key = id(value)
            if key in self.active:
                raise CiteError("cyclic PDF resource graph unsupported for page fingerprints")
            if key in self.memo:
                return self.memo[key]
            self.nodes += 1
            if self.nodes > 100_000:
                raise CiteError("page fingerprint exceeds 100000 container objects")
            self.active.add(key)
            if isinstance(value, DictionaryObject):
                excluded = {"/Length", "/Filter", "/DecodeParms"} if isinstance(value, StreamObject) else set()
                entries = [[str(k), self.hash(v, depth + 1)] for k, v in sorted(value.items())
                           if k not in excluded]
                if isinstance(value, StreamObject):
                    data = value.get_data()
                    self.stream_bytes += len(data)
                    if self.stream_bytes > MAX_PDF_BYTES:
                        raise CiteError("page fingerprint decoded streams exceed 128 MiB")
                    normalized = ["stream", entries, hashlib.sha256(data).hexdigest()]
                else:
                    normalized = ["dict", entries]
            else:
                normalized = ["array", [self.hash(v, depth + 1) for v in value]]
            result = digest(normalized)
            self.active.remove(key)
            self.memo[key] = result
            return result
        if isinstance(value, NameObject):
            normalized = ["name", str(value)]
        elif isinstance(value, TextStringObject):
            normalized = ["text", str(value)]
        elif isinstance(value, ByteStringObject):
            normalized = ["bytes", bytes(value).hex()]
        elif isinstance(value, BooleanObject):
            normalized = ["bool", value.value]
        elif isinstance(value, NullObject):
            normalized = ["null"]
        elif isinstance(value, (NumberObject, FloatObject)):
            number(value, "PDF fingerprint number")  # reject nonfinite values
            # Equal PDF numbers hash alike without rounding large integers to float.
            normalized = ["number", *value.as_integer_ratio()]
        else:
            raise CiteError(f"unsupported PDF fingerprint object: {type(value).__name__}")
        return digest(normalized)


def document_fingerprint(reader: PdfReader) -> dict:
    geometry = geometries(reader)
    hasher = _ObjectHasher()
    pages = []
    # Page geometry is normalized separately, including inherited/default values.
    excluded = {"/Parent", "/Annots", "/MediaBox", "/CropBox", "/Rotate", "/UserUnit"}
    # Retain these dictionaries for the lifetime of the memo (keys use object id).
    roots = [DictionaryObject({k: v for k, v in page.items() if k not in excluded})
             for page in reader.pages]
    for page_no, root in enumerate(roots, 1):
        fingerprint = digest({"geometry": asdict(geometry[page_no]), "objects": hasher.hash(root)})
        pages.append({"page": page_no, "fingerprint": fingerprint})
    return {"algorithm": ALGORITHM, "fingerprint": digest(pages), "pages": pages}


def export_sidecar(reader: PdfReader, regions: list[Region] | None = None) -> dict:
    """Export the exact verified citehl set; foreign annotations stay in the PDF."""
    if regions is None:
        regions = []
        metadata_bytes = 0
        for page in reader.pages:
            for ref in page.get("/Annots", []):
                annotation = ref.get_object()
                if METADATA_KEY not in annotation and not str(annotation.get("/NM", "")).startswith(PREFIX):
                    continue
                region = embedded_region(annotation)
                if region is None:
                    raise CiteError("citehl annotation missing region metadata; supply the original regions JSONL")
                metadata_bytes += len(annotation[METADATA_KEY].encode("utf-8"))
                if metadata_bytes > MAX_JSON_BYTES or len(regions) >= MAX_REGIONS:
                    raise CiteError("embedded metadata exceeds 32 MiB or 100000 annotations")
                regions.append(region)
    regions = validate_regions(regions, geometries(reader), strict=True)
    verify(reader, regions, text_check="off")
    result = {
        "format": FORMAT, "schema_version": 1, "coordinate_space": dict(COORDINATE_SPACE),
        "document": document_fingerprint(reader),
        "annotations": [{"annotation_id": annotation_id(r), "region": r.to_dict()} for r in regions],
    }
    if len(canonical(result).encode("utf-8")) > MAX_JSON_BYTES:
        raise CiteError("sidecar exceeds 32 MiB JSON input limit")
    return result


class SidecarMismatch(CiteError):
    """The first saved artifact mismatch, separate from PDF integrity errors."""

    def __init__(self, path: str, problem: str):
        self.path = path
        super().__init__(f"{path}: {problem}; regenerate with ragwrench highlight export using "
                         "the current PDF and the same PDF-only or final JSONL export order")


def _match_sidecar(actual: Any, expected: Any, path: str, *, coordinate: bool = False) -> None:
    """Compare complete exported fields without normalizing the saved artifact."""
    problem = None
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            problem = "object fields differ"
        else:
            for key, value in expected.items():
                _match_sidecar(actual[key], value, f"{path}[{key!r}]",
                               coordinate=key == "rect" and isinstance(value, list))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            problem = f"expected {len(expected)} ordered entries"
        else:
            for i, (left, right) in enumerate(zip(actual, expected)):
                _match_sidecar(left, right, f"{path}[{i}]", coordinate=coordinate)
    elif coordinate:
        # Equal finite JSON numbers are allowed, but never bools or PDF tolerance.
        if type(actual) not in (int, float) or actual != expected:
            problem = "numeric value differs"
    elif type(actual) is not type(expected) or actual != expected:
        problem = "value or type differs"
    if problem:
        raise SidecarMismatch(path, problem)


def check_sidecar(reader: PdfReader, saved: Any, regions: list[Region] | None = None) -> dict:
    """Verify the current PDF first; saved regions/fingerprints authorize nothing."""
    expected = export_sidecar(reader, regions)
    _match_sidecar(saved, expected, "sidecar")
    return {"annotations": len(expected["annotations"]), "pages": len(expected["document"]["pages"])}


def import_sidecar(reader: PdfReader, data: Any, output: Path) -> dict:
    """Validate fully, then add missing annotations; never replace existing ones."""
    data = obj(data, "sidecar")
    if data.get("format") != FORMAT or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise CiteError("unsupported annotation sidecar format/schema_version")
    if canonical(data.get("coordinate_space")) != canonical(COORDINATE_SPACE):
        raise CiteError("unsupported sidecar coordinate_space")
    if canonical(data.get("document")) != canonical(document_fingerprint(reader)):
        raise CiteError("document/page fingerprint mismatch; refusing annotation mapping")
    entries = seq(data.get("annotations"), "annotations", nonempty=False)
    if len(entries) > MAX_REGIONS:
        raise CiteError("too many sidecar annotations")
    expected = {}
    for entry in entries:
        entry = obj(entry, "annotation entry")
        region = Region.from_dict(entry.get("region"))
        # v1 IDs preserve numeric JSON representations, including Python-writer integers.
        region = replace(region, rect=tuple(entry["region"]["rect"]))
        name = annotation_id(region)
        if entry.get("annotation_id") != name:
            raise CiteError("sidecar annotation_id does not match its region")
        if name in expected:
            raise CiteError("duplicate sidecar annotation_id")
        expected[name] = region
    validate_regions(list(expected.values()), geometries(reader), strict=True)
    found = set()
    for page_no, page in enumerate(reader.pages, 1):
        for ref in page.get("/Annots", []):
            annotation = ref.get_object()
            if METADATA_KEY in annotation:
                embedded_region(annotation)
            name = str(annotation.get("/NM", ""))
            if name in expected:
                errors = annotation_errors(annotation, page_no, expected[name], tolerance=0.01)
                if name in found or errors:
                    raise CiteError(f"existing annotation ID conflict: {name}; " + "; ".join(errors))
                found.add(name)
    writer = PdfWriter(clone_from=reader)
    for name, region in expected.items():
        if name not in found:
            add_highlight(writer, region)
    atomic_write(output, writer.write)
    return {"added": len(expected) - len(found), "already_present": len(found)}
