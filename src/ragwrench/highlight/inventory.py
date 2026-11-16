"""Read-only inventory of persisted IDs, current geometry and optional provenance."""
from __future__ import annotations

import re

from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, TextStringObject

from .core import CiteError, MAX_JSON_BYTES, MAX_REGIONS, canonical, geometries, number
from .pdf import METADATA_KEY, PREFIX, embedded_region
from .sidecar import COORDINATE_SPACE


def annotation_inventory(reader: PdfReader) -> dict:
    """Inventory page entries, without verifying current geometry/style/text.

    /NM alone establishes namespace ownership. Metadata is checked even on a
    renamed entry so a conflict cannot silently become a foreign/legacy entry.
    The recovered region is provenance; page/rect describe the current annotation.
    """
    geometries(reader)
    entries = []
    seen = set()
    foreign = metadata_bytes = 0
    for page_no, page in enumerate(reader.pages, 1):
        annotations = page.get('/Annots', ArrayObject()).get_object()
        if not isinstance(annotations, ArrayObject):
            raise CiteError('PDF /Annots must be an array')
        for ref in annotations:
            annotation = ref.get_object()
            if not isinstance(annotation, DictionaryObject):
                raise CiteError('PDF annotation must be a dictionary')
            name = annotation.get('/NM')
            if name is not None:
                name = name.get_object()
            owned = isinstance(name, str) and name.startswith(PREFIX)
            if owned:
                if (not isinstance(name, TextStringObject)
                        or re.fullmatch(re.escape(PREFIX) + r'[0-9a-f]{64}', name) is None):
                    raise CiteError('owned /NM must be a PDF text string with an exact citehl:v1: ID')
                if name in seen:
                    raise CiteError(f'ambiguous duplicate PDF annotation ID: {name}')
                seen.add(name)
                if len(seen) > MAX_REGIONS:
                    raise CiteError('inventory exceeds 100000 owned annotations')
            region = None
            if METADATA_KEY in annotation:
                # Resolve indirect /NM without mutating the reader's dictionary.
                metadata_annotation = DictionaryObject(dict(annotation))
                if name is not None:
                    metadata_annotation[NameObject('/NM')] = name
                region = embedded_region(metadata_annotation)
                metadata_bytes += len(annotation[METADATA_KEY].encode('utf-8'))
                if metadata_bytes > MAX_JSON_BYTES:
                    raise CiteError('inventory embedded metadata exceeds 32 MiB')
            if not owned:
                foreign += 1
                continue
            rect = annotation.get('/Rect')
            if rect is not None:
                rect = rect.get_object()
            if not isinstance(rect, ArrayObject) or len(rect) != 4:
                raise CiteError('owned annotation /Rect must contain four numbers')
            coords = [number(value.get_object(), 'annotation /Rect coordinate') for value in rect]
            if coords[0] >= coords[2] or coords[1] >= coords[3]:
                raise CiteError('owned annotation /Rect must have positive width and height')
            entries.append({'annotation_id': str(name), 'page': page_no, 'rect': coords,
                            'metadata_available': region is not None,
                            'region': region.to_dict() if region is not None else None})
    result = {'format': 'citehl-annotation-inventory', 'schema_version': 1,
              'coordinate_space': dict(COORDINATE_SPACE), 'annotations': entries,
              'summary': {'owned': len(entries), 'foreign': foreign}}
    if len(canonical(result).encode('utf-8')) + 1 > MAX_JSON_BYTES:
        raise CiteError('inventory exceeds 32 MiB JSON output limit')
    return result
