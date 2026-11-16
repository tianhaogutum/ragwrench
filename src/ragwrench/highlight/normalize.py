"""Adapters for public DoclingDocument and Unstructured element JSON formats."""
from __future__ import annotations

import base64
import binascii
import math
import warnings
import zlib
from typing import Any

from .core import (CiteError, Geometry, MAX_JSON_BYTES, MAX_REGIONS, Region, identifier,
                   number, obj, page_number, parse_json, seq, text_value)


def orig_elements(value: Any) -> list[dict]:
    """Accept base64(zlib/gzip(JSON)) or the human-readable array variant."""
    if isinstance(value, str):
        if len(value) > MAX_JSON_BYTES:
            raise CiteError("orig_elements encoded input exceeds limit")
        try:
            compressed = base64.b64decode(value, validate=True)
            decoder = zlib.decompressobj(wbits=47)  # zlib and gzip headers
            decoded = decoder.decompress(compressed, MAX_JSON_BYTES + 1)
            if len(decoded) > MAX_JSON_BYTES:
                raise CiteError("orig_elements decompression exceeds 32 MiB limit")
            if not decoder.eof or decoder.unused_data:
                raise CiteError("orig_elements compressed stream is truncated or has trailing data")
            value = parse_json(decoded)
        except (binascii.Error, zlib.error) as exc:
            raise CiteError("orig_elements must be base64-encoded zlib/gzip JSON or an array") from exc
    return [obj(e, "orig_elements entry") for e in seq(value, "orig_elements")]


def mapped_rect(points: list, geometry: Geometry, width: float, height: float,
                origin: str, box: str, frame: str, strict: bool) -> tuple:
    converted = []
    for point in points:
        pair = seq(point, "coordinate point")
        if len(pair) != 2:
            raise CiteError("each coordinate point must contain exactly two numbers")
        converted.append(geometry.to_pdf(number(pair[0], "x"), number(pair[1], "y"),
                                         width, height, origin, box, frame))
    if any(not math.isfinite(v) for point in converted for v in point):
        raise CiteError("coordinate conversion overflow; check page dimensions and bbox")
    rect = (min(p[0] for p in converted), min(p[1] for p in converted),
            max(p[0] for p in converted), max(p[1] for p in converted))
    if rect[0] >= rect[2] or rect[1] >= rect[3]:
        raise CiteError("bbox must have positive width and height")
    return geometry.clamp(rect, strict=strict)


def normalize_docling(data: Any, pdf_geometry: dict[int, Geometry] | None = None, *,
                      strict: bool = False, page_box: str | None = None,
                      coordinate_frame: str | None = None) -> list[Region]:
    data = obj(data, "Docling document")
    page_box, coordinate_frame = page_box or "media", coordinate_frame or "unrotated"
    pages = obj(data.get("pages", {}), "Docling pages")
    result = []
    for collection in ("texts", "tables", "pictures"):
        for item in seq(data.get(collection, []), collection, nonempty=False):
            item = obj(item, collection + " item")
            ref = identifier(item.get("self_ref"), "self_ref")
            text = text_value(item.get("text", ""))
            if not text and collection == "tables":
                cells = obj(item.get("data", {}), "table data").get("table_cells", [])
                text = " ".join(text_value(obj(c, "table cell").get("text", ""))
                                for c in seq(cells, "table_cells", nonempty=False))
            for prov in seq(item.get("prov"), f"{ref} prov"):
                prov = obj(prov, "provenance")
                page = page_number(prov.get("page_no"))
                if pdf_geometry is not None and page not in pdf_geometry:
                    raise CiteError(f"page {page} outside PDF")
                page_data = obj(pages.get(str(page), {}), "Docling page")
                size = page_data.get("size")
                geometry = pdf_geometry[page] if pdf_geometry is not None else None
                if size is None:
                    if geometry is None:
                        raise CiteError(f"Docling page {page} lacks size; provide --pdf for MediaBox fallback")
                    width, height = geometry.dimensions(page_box, coordinate_frame)
                    warnings.warn(f"Docling page {page} lacks size; using PDF {page_box.title()}Box dimensions", stacklevel=2)
                else:
                    size = obj(size, "page size")
                    width = number(size.get("width"), "page width", positive=True)
                    height = number(size.get("height"), "page height", positive=True)
                if geometry is None:
                    geometry = Geometry((0, 0, width, height), (0, 0, width, height))
                bbox = obj(prov.get("bbox"), "bbox")
                origin = bbox.get("coord_origin", "TOPLEFT")
                if origin not in ("TOPLEFT", "BOTTOMLEFT"):
                    raise CiteError(f"unknown Docling coord_origin: {origin!r}")
                left, top, right, bottom = [number(bbox.get(k), f"bbox {k}") for k in ("l", "t", "r", "b")]
                if left >= right or (origin == "TOPLEFT" and top >= bottom) or (origin == "BOTTOMLEFT" and bottom >= top):
                    raise CiteError("Docling bbox has inverted or zero dimensions for its coord_origin")
                rect = mapped_rect([[left, top], [right, bottom]], geometry, width, height,
                                   origin, page_box, coordinate_frame, strict)
                region_text = text
                if "charspan" in prov and text:
                    span = seq(prov["charspan"], "charspan")
                    if (len(span) != 2 or any(type(v) is not int for v in span)
                            or not 0 <= span[0] <= span[1] <= len(text)):
                        raise CiteError(f"invalid charspan for {ref}")
                    region_text = text[span[0]:span[1]]
                result.append(Region(ref, ref, page, rect, region_text))
                if len(result) > MAX_REGIONS:
                    raise CiteError("too many regions")
    if not result:
        raise CiteError("Docling document has no supported items with provenance")
    return result


def normalize_unstructured(data: Any, pdf_geometry: dict[int, Geometry] | None = None, *,
                           strict: bool = False, page_box: str | None = None,
                           coordinate_frame: str | None = None) -> list[Region]:
    result = []
    # A chunk carrying orig_elements is expanded to real source geometry; no fake chunk box.
    for raw in seq(data, "Unstructured elements"):
        raw = obj(raw, "Unstructured element")
        chunk_id = identifier(raw.get("element_id"), "element_id")
        metadata = obj(raw.get("metadata"), "metadata")
        elements = orig_elements(metadata["orig_elements"]) if "orig_elements" in metadata else [raw]
        for element in elements:
            ref = identifier(element.get("element_id"), "element_id")
            meta = obj(element.get("metadata"), "metadata")
            if "orig_elements" in meta:
                raise CiteError("nested orig_elements are unsupported; flatten source elements first")
            page = page_number(meta.get("page_number"))
            coords = obj(meta.get("coordinates"), f"{ref} metadata.coordinates")
            system = coords.get("system")
            if system not in ("PixelSpace", "PointSpace"):
                raise CiteError(f"unknown Unstructured coordinate system: {system!r}")
            width = number(coords.get("layout_width"), "layout_width", positive=True)
            height = number(coords.get("layout_height"), "layout_height", positive=True)
            if pdf_geometry is None:
                if system == "PixelSpace":
                    raise CiteError("PixelSpace requires --pdf to map pixels to actual PDF points")
                geometry = Geometry((0, 0, width, height), (0, 0, width, height))
            else:
                if page not in pdf_geometry:
                    raise CiteError(f"page {page} outside PDF")
                geometry = pdf_geometry[page]
            # PointSpace describes native PDF axes; pixels describe a rendered page.
            box = page_box or ("crop" if system == "PixelSpace" else "media")
            frame = coordinate_frame or ("display" if system == "PixelSpace" else "unrotated")
            points = seq(coords.get("points"), "coordinates.points")
            if len(points) < 2:
                raise CiteError("coordinates.points needs at least two points")
            rect = mapped_rect(points, geometry, width, height,
                               "TOPLEFT" if system == "PixelSpace" else "BOTTOMLEFT",
                               box, frame, strict)
            result.append(Region(chunk_id, ref, page, rect, text_value(element.get("text", ""))))
            if len(result) > MAX_REGIONS:
                raise CiteError("too many regions")
    if not result:
        raise CiteError("no Unstructured regions")
    return result
