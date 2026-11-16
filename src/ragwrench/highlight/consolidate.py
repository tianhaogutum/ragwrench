"""Chunk membership resolution and deterministic, page-local box consolidation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
from typing import Any

from .core import CiteError, MAX_REGIONS, Region, canonical, identifier, number, obj, seq, text_value
from .normalize import orig_elements


def chunk_memberships(data: Any) -> list[tuple[str, list[str], str | None]]:
    """Read a compact chunk->refs map or exported Docling/Unstructured chunk lists."""
    result = []
    if isinstance(data, dict):
        for chunk_id, refs in data.items():
            result.append((identifier(chunk_id, "chunk_id"),
                           [identifier(r, "element ref") for r in seq(refs, "chunk refs")], None))
    else:
        for chunk in seq(data, "chunks"):
            chunk = obj(chunk, "chunk")
            chunk_id = chunk.get("chunk_id", chunk.get("element_id"))
            if chunk_id is None and "doc_items" in obj(chunk.get("meta", {}), "chunk meta"):
                chunk_id = "docling:" + hashlib.sha256(canonical(chunk).encode()).hexdigest()[:24]
            chunk_id = identifier(chunk_id, "chunk_id / element_id")
            text = text_value(chunk["text"]) if "text" in chunk else None
            if "element_ids" in chunk:
                refs = chunk["element_ids"]
            elif "doc_items" in obj(chunk.get("meta", {}), "chunk meta"):
                refs = [obj(i, "doc_item").get("self_ref") for i in seq(chunk["meta"]["doc_items"], "doc_items")]
            else:
                metadata = obj(chunk.get("metadata", {}), "chunk metadata")
                if "element_ids" in metadata:
                    refs = metadata["element_ids"]
                elif "orig_elements" in metadata:
                    refs = [e.get("element_id") for e in orig_elements(metadata["orig_elements"])]
                else:
                    raise CiteError(f"chunk {chunk_id} has no element_ids, meta.doc_items, or metadata.orig_elements")
            result.append((chunk_id, [identifier(r, "element ref") for r in seq(refs, "chunk refs")], text))
    if not result or len({c for c, _, _ in result}) != len(result):
        raise CiteError("chunk IDs must be nonempty and unique")
    return result


def assign_chunks(regions: list[Region], data: Any) -> list[Region]:
    index = defaultdict(list)
    for region in regions:
        for ref in set(region.source_refs or (region.source_ref,)):
            index[ref].append(region)
    result = []
    for chunk, refs, text in chunk_memberships(data):
        seen = set()
        selected_refs = set(refs)
        for ref in dict.fromkeys(refs):
            if ref not in index:
                raise CiteError(f"chunk {chunk} references missing element {ref}")
            for region in index[ref]:
                if region.source_refs and not set(region.source_refs) <= selected_refs:
                    raise CiteError("cannot partially remap a consolidated region; use the original normalized regions")
                updated = replace(region, chunk_id=chunk, text=text if text is not None else region.text)
                if updated not in seen:
                    seen.add(updated)
                    result.append(updated)
                    if len(result) > MAX_REGIONS:
                        raise CiteError("chunk mapping exceeds maximum region count")
    return result


def union(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def combine(a: Region, b: Region) -> Region:
    refs = tuple(sorted(set((a.source_refs or (a.source_ref,)) + (b.source_refs or (b.source_ref,)))))
    texts = list(dict.fromkeys(t for t in (a.text, b.text) if t))
    return Region(a.chunk_id, refs[0], a.page, union(a.rect, b.rect), " ".join(texts), refs)


def consolidate(regions: list[Region], *, per_page: bool = False, line_tolerance: float = 2,
                max_gap: float = 8) -> list[Region]:
    """Merge adjacent horizontal boxes only; each group is a (chunk,page) pair."""
    number(line_tolerance, "line tolerance")
    number(max_gap, "max gap")
    if line_tolerance < 0 or max_gap < 0:
        raise CiteError("line tolerance and max gap must be nonnegative")
    groups = defaultdict(list)
    for region in regions:
        groups[(region.chunk_id, region.page)].append(region)
    result = []
    for key in sorted(groups):
        boxes = sorted(set(groups[key]), key=lambda r: (-r.rect[3], r.rect[0], r.source_ref))
        if not per_page and len(boxes) > 2000:
            raise CiteError("same-line merging supports at most 2000 boxes per (chunk,page); use --per-page or split chunks")
        if per_page:
            merged = boxes[0]
            for box in boxes[1:]:
                merged = combine(merged, box)
            result.append(merged)
            continue
        merged_boxes: list[Region] = []
        for box in boxes:
            # Repeat until fixed point: joining one adjacent fragment may bridge another.
            i = 0
            while i < len(merged_boxes):
                other = merged_boxes[i]
                a, b = box.rect, other.rect
                same_line = abs(a[1] - b[1]) <= line_tolerance and abs(a[3] - b[3]) <= line_tolerance
                gap = max(a[0], b[0]) - min(a[2], b[2])
                if same_line and gap <= max_gap:
                    box = combine(other, box)
                    merged_boxes.pop(i)
                    i = 0
                else:
                    i += 1
            merged_boxes.append(box)
        result.extend(sorted(merged_boxes, key=lambda r: (-r.rect[3], r.rect[0], r.source_ref)))
    return result
