"""Shape adapters, written independently from upstream implementations.

Only complete, non-streaming textual exports are supported. Unknown metadata
is ignored, never evaluated, dereferenced or included in diagnostics.
"""
from .model import InputError, Record, Source, MAX_SOURCES, validate

FORMATS = ("generic", "llamaindex", "dify", "ragflow")


def obj(value, path):
    if not isinstance(value, dict):
        raise InputError(f"{path} must be an object")
    return value


def required(mapping, key, path):
    if key not in mapping:
        raise InputError(f"missing {path}.{key}; check --format and export complete text")
    return mapping[key]


def text(value, path):
    if not isinstance(value, str):
        raise InputError(f"{path} must be a string (multimodal/streaming values are unsupported)")
    return value


def array(value, path):
    if not isinstance(value, list):
        raise InputError(f"{path} must be an array; aggregate streaming events first")
    if len(value) > MAX_SOURCES:
        raise InputError(f"{path}: at most {MAX_SOURCES} sources are supported")
    return value


def adapt(payload, format="generic") -> Record:
    root = obj(payload, "record")
    if format not in FORMATS:
        raise InputError("unsupported format; choose generic, llamaindex, dify or ragflow")
    sources = []
    if format == "generic":
        answer = required(root, "answer", "record")
        for n, raw in enumerate(array(required(root, "sources", "record"), "sources"), 1):
            raw = obj(raw, f"sources[{n}]")
            sources.append(Source(required(raw, "idx", "source"),
                                  text(raw.get("id", str(n)), "source.id"),
                                  text(raw.get("title", ""), "source.title"),
                                  text(required(raw, "text", "source"), "source.text")))
    elif format == "llamaindex":
        answer = required(root, "response", "record")
        for n, raw in enumerate(array(required(root, "source_nodes", "record"), "source_nodes"), 1):
            raw = obj(raw, "source_nodes[]")
            node = obj(required(raw, "node", "source_nodes[]"), "source_nodes[].node")
            metadata = obj(node.get("metadata", {}), "node.metadata")
            # CitationQueryEngine numbers the citation nodes in response order,
            # NOT their original document ids or retrieval rank metadata.
            sources.append(Source(n, text(node.get("id_", str(n)), "node.id_"),
                                  text(metadata.get("file_name", metadata.get("title", "")), "node title"),
                                  text(required(node, "text", "node"), "node.text")))
    elif format == "dify":
        answer = required(root, "answer", "record")
        metadata = obj(required(root, "metadata", "record"), "metadata")
        for raw in array(required(metadata, "retriever_resources", "metadata"), "metadata.retriever_resources"):
            raw = obj(raw, "retriever_resources[]")
            # Position is explicit and may be sparse/out of order. Segment id
            # identifies chunks; document_id alone is not unique per chunk.
            position = required(raw, "position", "retriever_resource")
            sources.append(Source(position,
                                  text(raw.get("segment_id", raw.get("document_id", str(position))), "resource id"),
                                  text(raw.get("document_name", ""), "document_name"),
                                  text(required(raw, "content", "retriever_resource"), "resource.content")))
    else:
        if "data" in root:
            if type(root.get("code", 0)) is not int or root.get("code", 0) != 0:
                raise InputError("RAGFlow error envelope; a successful completion is required")
            root = obj(root["data"], "data")
        answer = required(root, "answer", "record/data")
        reference = obj(required(root, "reference", "record/data"), "reference")
        for n, raw in enumerate(array(required(reference, "chunks", "reference"), "reference.chunks"), 1):
            raw = obj(raw, "reference.chunks[]")
            content = raw.get("content", raw.get("content_with_weight"))
            if content is None:
                raise InputError("RAGFlow chunk needs content or content_with_weight; tokenized content is insufficient")
            sources.append(Source(n, text(raw.get("id", raw.get("chunk_id", str(n))), "chunk id"),
                                  text(raw.get("document_name", raw.get("docnm_kwd", "")), "chunk title"),
                                  text(content, "chunk content")))
    return validate(Record(text(answer, "answer/response"), tuple(sources)))
