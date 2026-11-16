"""JSON on stdout; diagnostics and skipped-empty warnings on stderr."""

from __future__ import annotations

import argparse
import json
from contextlib import closing
import os
import sys
import tempfile
from pathlib import Path

from . import SUBCOMMANDS
from .encoder import Encoder
from .evaluation import evaluate
from .store import (
    build_collection,
    explain_collection,
    export_documents,
    open_local,
    read_corpus,
    read_delete_ids,
    read_document,
    read_encoder_config,
    read_export_ids,
    recover_collection,
    search_collection,
    status_collection,
    update_collection,
    validate_collection_name,
    validate_documents,
    verify_document,
    verify_documents,
)
from .tokenizers import MAX_TEXT_CHARS, TOKENIZERS, read_stopwords

MAX_ENCODE_TEXT_BYTES = 4 * 1024 * 1024


def _read_encode_text(path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read(MAX_ENCODE_TEXT_BYTES + 1)
    if len(data) > MAX_ENCODE_TEXT_BYTES:
        raise ValueError("encode text file exceeds 4 MiB")
    # No newline conversion or BOM stripping: encode the exact decoded text.
    text = data.decode("utf-8")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters")
    return text


def _nonnegative_int(value: str) -> int:
    try:
        count = int(value)
        if count >= 0:
            return count
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected a nonnegative integer")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ragwrench bm25", description="Offline CJK BM25 with Qdrant local mode"
    )
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("encode", "index", "search"):
        cmd = sub.add_parser(name, help=SUBCOMMANDS[name])
        cmd.add_argument("--tokenizer", choices=TOKENIZERS, default=None)
        cmd.add_argument("--stopwords", type=Path, help="UTF-8 literal tokens, one per line")
        cmd.add_argument("--k1", type=float, default=None)
        cmd.add_argument("--b", type=float, default=None)
        cmd.add_argument("--avgdl", type=float, default=None)
        if name in ("encode", "index"):
            cmd.add_argument(
                "--encoder-config",
                type=Path,
                help=(
                    "strict saved Encoder.config JSON; no config overrides"
                    + ("; avgdl becomes fixed" if name == "index" else "")
                ),
            )
        if name == "encode":
            source = cmd.add_mutually_exclusive_group(required=True)
            source.add_argument("text", nargs="?")
            source.add_argument(
                "--text-file", type=Path, help="one literal UTF-8 text file, at most 4 MiB"
            )
            cmd.add_argument(
                "--document", action="store_true", help="TF weights instead of query ones"
            )
        else:
            cmd.add_argument("--path", required=True, type=Path)
            cmd.add_argument("--collection", required=True)
            if name == "index":
                cmd.add_argument("corpus", type=Path)
            else:
                cmd.add_argument("query")
                cmd.add_argument("--top-k", type=int, default=10)
                cmd.add_argument(
                    "--explain", action="store_true", help="include tokens and BM25 contributions"
                )
                cmd.add_argument(
                    "--explain-id",
                    metavar="JSON_ID",
                    help="also explain one source ID as JSON (7 or '\"7\"'); implies --explain",
                )
    cmd = sub.add_parser("update", help="add/delete documents using saved corpus configuration")
    cmd.add_argument("--path", required=True, type=Path)
    cmd.add_argument("--collection", required=True)
    cmd.add_argument("--add", type=Path, help="JSONL documents to add")
    cmd.add_argument("--delete-ids", type=Path, help="JSON array of source IDs to delete")
    cmd = sub.add_parser("recover", help="replay a saved intended corpus after interrupted writes")
    cmd.add_argument("--path", required=True, type=Path)
    cmd.add_argument("--collection", required=True)
    cmd.add_argument(
        "--dry-run", action="store_true", help="validate and count recovery work without writes"
    )
    cmd = sub.add_parser(
        "status", help="read saved collection readiness without validating recovery"
    )
    cmd.add_argument("--path", required=True, type=Path)
    cmd.add_argument("--collection", required=True)
    cmd.add_argument(
        "--require-ready", action="store_true", help="fail if the validated state is not ready"
    )
    cmd.add_argument(
        "--expect-documents",
        type=_nonnegative_int,
        metavar="NONNEGATIVE_INT",
        help="require the saved intended document count (not the live point count)",
    )
    cmd = sub.add_parser(
        "verify-document", help="compare one source document with its stored point"
    )
    cmd.add_argument("--path", required=True, type=Path)
    cmd.add_argument("--collection", required=True)
    cmd.add_argument("document", type=Path, help="one strict JSON document, not JSONL")
    cmd = sub.add_parser(
        "verify-documents", help="compare a bounded JSONL source batch with selected stored points"
    )
    cmd.add_argument("--path", required=True, type=Path)
    cmd.add_argument("--collection", required=True)
    cmd.add_argument("documents", type=Path, help="strict JSONL source documents")
    cmd.add_argument(
        "--require-complete",
        action="store_true",
        help="require supplied record count to equal the saved indexed document count",
    )
    cmd = sub.add_parser(
        "export-documents", help="export selected stored source documents as reindexable JSONL"
    )
    cmd.add_argument("--path", required=True, type=Path)
    cmd.add_argument("--collection", required=True)
    cmd.add_argument(
        "--ids", required=True, type=Path, help="strict nonempty JSON array of source IDs"
    )
    cmd.add_argument("--output", required=True, type=Path, help="fresh JSONL path outside storage")
    cmd = sub.add_parser("eval", help="run bundled synthetic zh/ja/mixed comparison")
    cmd.add_argument("--top-k", type=int, default=3)
    cmd.add_argument("--include-jieba", action="store_true")
    cmd.add_argument("--output", type=Path, default=Path(".artifacts/eval.json"))
    return root


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".qbm25-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _export_destination(output: Path, selection: Path, storage: Path) -> Path:
    """Resolve parent aliases before opening storage; never replace any entry."""
    resolved = output.resolve()
    store = storage.resolve()
    if resolved == selection.resolve() or resolved == store or store in resolved.parents:
        raise ValueError("export output must be distinct from selection and outside storage")
    if os.path.lexists(output):
        raise ValueError("export output already exists")
    if not resolved.parent.is_dir():
        raise ValueError("export output parent must be an existing directory")
    return resolved


def _write_export(path: Path, documents: list[dict]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".qbm25-export-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            for document in documents:
                stream.write(
                    (json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n").encode(
                        "utf-8"
                    )
                )
        # Same-directory hard link publishes the complete file and fails atomically
        # if another process created the destination after preflight validation.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace):
    if args.command == "eval":
        report = evaluate(top_k=args.top_k, include_jieba=args.include_jieba)
        write_report(args.output, report)
        return report
    if args.command == "recover":
        validate_collection_name(args.collection)
        with closing(open_local(args.path, must_exist=True)) as client:
            return recover_collection(client, args.collection, dry_run=args.dry_run)
    if args.command == "status":
        validate_collection_name(args.collection)
        with closing(open_local(args.path, must_exist=True)) as client:
            return status_collection(
                client,
                args.collection,
                require_ready=args.require_ready,
                expect_documents=args.expect_documents,
            )
    if args.command == "verify-document":
        validate_collection_name(args.collection)
        document = read_document(args.document)
        with closing(open_local(args.path, must_exist=True)) as client:
            return verify_document(client, args.collection, document)
    if args.command == "verify-documents":
        validate_collection_name(args.collection)
        documents = read_corpus(args.documents)
        with closing(open_local(args.path, must_exist=True)) as client:
            return verify_documents(
                client, args.collection, documents, require_complete=args.require_complete
            )
    if args.command == "export-documents":
        validate_collection_name(args.collection)
        source_ids = read_export_ids(args.ids)
        destination = _export_destination(args.output, args.ids, args.path)
        with closing(open_local(args.path, must_exist=True)) as client:
            documents = export_documents(client, args.collection, source_ids)
        _write_export(destination, documents)
        return {"schema": 1, "collection": args.collection, "exported": len(documents)}
    if args.command == "update":
        validate_collection_name(args.collection)
        additions = read_corpus(args.add) if args.add is not None else []
        delete_ids = read_delete_ids(args.delete_ids) if args.delete_ids is not None else []
        if not additions and not delete_ids:
            raise ValueError("update requires at least one addition or deletion")
        with closing(open_local(args.path, must_exist=True)) as client:
            return update_collection(
                client, args.collection, additions=additions, delete_ids=delete_ids
            )
    encoder = None
    if args.command in ("encode", "index") and args.encoder_config is not None:
        if any(
            getattr(args, name) is not None
            for name in ("tokenizer", "stopwords", "k1", "b", "avgdl")
        ):
            raise ValueError(
                "--encoder-config cannot be combined with --tokenizer/--stopwords/--k1/--b/--avgdl"
            )
        if args.command == "index":
            validate_collection_name(args.collection)
        encoder = read_encoder_config(args.encoder_config)
    if args.command == "index" and encoder is not None:
        documents = read_corpus(args.corpus)
        with closing(open_local(args.path)) as client:
            return build_collection(
                client,
                args.collection,
                documents,
                tokenizer=encoder.tokenizer.name,
                stopwords=tuple(encoder.tokenizer.stopwords),
                k1=encoder.k1,
                b=encoder.b,
                avgdl=encoder.avgdl,
            )
    stopwords = read_stopwords(args.stopwords) if args.stopwords is not None else None
    if args.command == "search":
        validate_collection_name(args.collection)
        selection = {}
        if args.explain_id is not None:
            source_id = json.loads(args.explain_id)
            validate_documents([{"id": source_id, "text": ""}])
            selection["source_id"] = source_id
        with closing(open_local(args.path, must_exist=True)) as client:
            search = explain_collection if args.explain or selection else search_collection
            return search(
                client,
                args.collection,
                args.query,
                top_k=args.top_k,
                tokenizer=args.tokenizer,
                k1=args.k1,
                b=args.b,
                avgdl=args.avgdl,
                stopwords=stopwords,
                **selection,
            )
    options = {
        "tokenizer": args.tokenizer or "cjk-bigram",
        "k1": args.k1 if args.k1 is not None else 1.2,
        "b": args.b if args.b is not None else 0.75,
        "stopwords": stopwords or (),
    }
    if args.command == "encode":
        if encoder is None:
            encoder = Encoder(**options, avgdl=args.avgdl if args.avgdl is not None else 100.0)
        text = _read_encode_text(args.text_file) if args.text_file is not None else args.text
        vector = (
            encoder.encode_documents([text])[0] if args.document else encoder.encode_query(text)
        )
        return {
            "tokens": encoder.tokenizer(text),
            **vector.as_dict(),
            "config": encoder.config,
            "collision_stats": encoder.collision_stats,
        }
    validate_collection_name(args.collection)
    documents = read_corpus(args.corpus)
    # Validate parameter/tokenizer errors before opening a store.
    Encoder(**options, avgdl=args.avgdl if args.avgdl is not None else 100.0)
    with closing(open_local(args.path)) as client:
        return build_collection(client, args.collection, documents, **options, avgdl=args.avgdl)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = run(args)
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))
        return (
            1
            if args.command in ("verify-document", "verify-documents") and not output["matches"]
            else 0
        )
    except (ValueError, OSError, RuntimeError, UnicodeError, TypeError, RecursionError) as exc:
        print(f"ragwrench bm25: error: {exc}", file=sys.stderr)
        return 2
