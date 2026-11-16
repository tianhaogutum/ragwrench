"""Explicit selection of persisted citehl IDs; unlink only selected page annotations."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, TextStringObject

from .core import CiteError, MAX_REGIONS, atomic_write, geometries, obj, seq
from .pdf import PREFIX, embedded_region


def selection_ids(data: Any) -> set[str]:
    data = obj(data, "selection")
    if (set(data) != {"version", "annotation_ids"}
            or type(data.get("version")) is not int or data["version"] != 1):
        raise CiteError("selection requires exactly version (integer 1) and annotation_ids")
    entries = seq(data["annotation_ids"], "annotation_ids", nonempty=False)
    if len(entries) > MAX_REGIONS:
        raise CiteError("selection exceeds 100000 annotation IDs")
    selected = set()
    for name in entries:
        if not isinstance(name, str) or re.fullmatch(re.escape(PREFIX) + r"[0-9a-f]{64}", name) is None:
            raise CiteError("annotation_ids must contain exact citehl:v1: IDs with 64 lowercase hex digits")
        if name in selected:
            raise CiteError(f"duplicate requested annotation ID: {name}")
        selected.add(name)
    return selected


def remove_annotations(reader: PdfReader, selection: Any, output: Path) -> dict:
    """Validate all targets before cloning/writing; preserve retained objects verbatim.

    Ownership is the reserved /NM namespace, never title/style/text. Metadata is
    optional for legacy IDs, but selected present metadata must bind to /NM.
    Unselected payloads are not interpreted. Duplicate names in our namespace
    anywhere in the page arrays are ambiguous even for an empty selection.
    """
    selected = selection_ids(selection)
    geometries(reader)  # Enforce the same supported page geometry as other PDF commands.
    found = set()
    seen = set()
    removal_indices = []
    total = 0
    for page in reader.pages:
        annotations = page.get("/Annots", ArrayObject()).get_object()
        if not isinstance(annotations, ArrayObject):
            raise CiteError("PDF /Annots must be an array")
        indices = set()
        for index, ref in enumerate(annotations):
            annotation = ref.get_object()
            if not isinstance(annotation, DictionaryObject):
                raise CiteError("PDF annotation must be a dictionary")
            total += 1
            name = annotation.get("/NM")
            if name is not None:
                name = name.get_object()
            if isinstance(name, str) and name.startswith(PREFIX):
                if name in seen:
                    raise CiteError(f"ambiguous duplicate PDF annotation ID: {name}")
                seen.add(name)
            if not isinstance(name, str) or name not in selected:
                continue
            if not isinstance(name, TextStringObject):
                raise CiteError("selected annotation /NM must be a PDF text string")
            # Validate against the resolved persisted text, not an indirect object
            # reference. A shallow view leaves the reader and retained payloads intact.
            metadata_annotation = DictionaryObject(dict(annotation))
            metadata_annotation[NameObject("/NM")] = name
            embedded_region(metadata_annotation)
            found.add(name)
            indices.add(index)
        removal_indices.append(indices)
    missing = selected - found
    if missing:
        raise CiteError(f"selected annotation ID missing or not citehl-owned: {sorted(missing)[0]}")
    writer = PdfWriter(clone_from=reader)
    for page, indices in zip(writer.pages, removal_indices):
        if indices:
            page[NameObject("/Annots")] = ArrayObject([
                ref for index, ref in enumerate(page["/Annots"]) if index not in indices
            ])
    atomic_write(output, writer.write)
    return {"removed": len(found), "preserved": total - len(found)}
