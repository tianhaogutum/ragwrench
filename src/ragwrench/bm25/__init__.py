"""Offline BM25 for CJK and Qdrant sparse vectors."""

from .encoder import Encoder, SparseVector, token_hash

__all__ = ["Encoder", "SparseVector", "token_hash"]

# Listed by ``ragwrench bm25 --help`` when the qdrant extra is missing.
SUBCOMMANDS = {
    "encode": "encode text into BM25 sparse vectors",
    "index": "build a Qdrant local collection from a JSONL corpus",
    "search": "BM25 search over a Qdrant local collection",
    "update": "add/delete documents using saved corpus configuration",
    "recover": "replay a saved intended corpus after interrupted writes",
    "status": "read saved collection readiness without validating recovery",
    "verify-document": "compare one source document with its stored point",
    "verify-documents": "compare a bounded JSONL source batch with selected stored points",
    "export-documents": "export selected stored source documents as reindexable JSONL",
    "eval": "run bundled synthetic zh/ja/mixed comparison",
}
