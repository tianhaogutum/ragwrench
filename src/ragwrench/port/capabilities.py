"""Describe the pinned import contract without loading a bundle or migration state."""


def capabilities(system, profile, method):
    """Return fresh JSON data; resolution and optional GET discovery happen in CLI."""
    dify = system == "dify"

    def field(status, detail):
        return {"import": status, "detail": detail}

    def metadata():
        return field("requires_allow_lossy", "Nonempty metadata is rejected by default; --allow-lossy drops it from the target, retaining it in the bundle only.")
    return {
        "system": system,
        "api_profile": profile,
        "bundle_format": "kbbundle/1",
        "scope": "import",
        "profile_resolution": {
            "method": method,
            "server_checked": method == "dataset_signatures",
            "exact_server_version_verified": False,
        },
        "fields": {
            "documents": {
                "id": field("remapped", "Fresh target IDs; source-to-target mapping is saved in migration state."),
                "name": field("renamed", "Up to 80 source-name characters plus a deterministic [kbport-…] recovery marker."),
                "doc_form": field("preserved" if dify else "text_model",
                                  "text_model and qa_model supported; hierarchical documents unsupported." if dify else
                                  "Target documents use text_model; separate QA structure is not restored."),
                "metadata": metadata(),
            },
            "chunks": {
                "id": field("remapped", "Target-generated ID saved in migration state."),
                "doc_id": field("remapped", "Resolved through the saved document mapping."),
                "position": field("not_restored", "Used for deterministic bundle/import traversal, not written as a target ordinal or verified."),
                "content": field("preserved", "Exact UTF-8 text, except the explicitly acknowledged RAGFlow QA projection; no automatic splitting."),
                "keywords": field("preserved", "Written as keywords." if dify else "Written as important_keywords."),
                "questions": field("requires_allow_lossy" if dify else "preserved",
                                   "Nonempty embedding questions rejected by default; --allow-lossy drops them." if dify else
                                   "Written as questions."),
                "answer": field("conditional" if dify else "requires_allow_lossy",
                                "Each qa_model chunk requires a nonempty answer; text_model chunks require no answer. --allow-lossy does not bypass this rule." if dify else
                                "Nonempty separate answers rejected by default; --allow-lossy replaces content with Question: <content>\\nAnswer: <answer> and removes the separate answer."),
                "enabled": field("preserved", "Applied with a segment update." if dify else "Applied with chunk PATCH available; PUT-only APIs unsupported."),
                "metadata": metadata(),
                "content_sha256": field("derived", "Local integrity hash recomputed after projection; verification hashes target content, not a restored API field."),
            },
        },
        "dataset_creation": ({"indexing_technique": "high_quality", "permission": "only_me"} if dify else
                             {"chunk_method": "naive"} if profile == "ragflow-0.27.3" else
                             {"parse_type": 1, "parser_id": "general"}),
        "limits": {
            "default_max_chunk_bytes": 16000,
            "max_chunk_bytes_ceiling": 1048576,
            "chunk_size_unit": "UTF-8 bytes after projection; not an embedding model token limit",
            "automatic_chunk_splitting": False,
            "repeated_content_in_document": "allowed" if dify else "rejected to avoid content-derived chunk ID collisions",
        },
        "loss_boundaries": [
            "This describes a pinned import contract, not bundle preflight, target write permissions, provider readiness or live-server compatibility.",
            "Default/explicit profiles are offline selections. Auto uses dataset field signatures to identify a contract family, not an exact installed release; empty or ambiguous discovery requires an explicit profile.",
            "--allow-lossy applies only to the stated metadata, questions and answer projections and emits warnings; it does not expand API support.",
            "Source dataset settings are descriptive and are not automatically applied. Dataset creation uses the reported defaults; Dify --indexing economy is optional. Target models/providers must already be configured.",
            "Document disabled/archive state is not restored. Export rejects disabled, archived or incompletely indexed Dify documents and disabled RAGFlow documents; hierarchical Dify documents are unsupported.",
            "Chunk layout, image, tag and attachment metadata are not reconstructed. Original binary files, vectors/embeddings, apps/workflows and database dumps are not transferred.",
            "Names, IDs, ordering and metadata are not verification criteria. Verify after import to check projected content, answers, keyword/question lists, enabled flags and multiplicity.",
        ],
    }
