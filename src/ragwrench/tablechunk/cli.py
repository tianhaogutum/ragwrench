"""CLI uses UTF-8 text only; no external workflows or code evaluation."""
import argparse
from contextlib import nullcontext
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
from .models import TableChunkError
from .splitter import MAX_INPUT_BYTES, split_document
from .streaming import stream_document
from .tokens import tiktoken_counter
from .benchmark import run_benchmark, synthetic_fixture


def read_text(path: str) -> str:
    if path == "-":
        data = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    else:
        with open(path, "rb") as stream:
            data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise TableChunkError("input exceeds 10 MiB limit")
    return data.decode("utf-8")


def write_text(path: str | None, text: str):
    if not path or path == "-":
        sys.stdout.write(text)
        return
    # Atomic replacement avoids partial output on failure; no invented output paths.
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".tablechunk-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_chunks(path, chunks, adapter):
    """Incremental JSONL serialization; publish only after complete success."""
    destination = Path(path) if path and path != "-" else None
    temporary = None
    warnings = set()
    try:
        if destination is not None:
            output = tempfile.NamedTemporaryFile(mode="w+t", encoding="utf-8",
                                                 dir=destination.parent, prefix=".tablechunk-", delete=False)
            temporary = Path(output.name)
        else:
            output = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        with output:
            for chunk in chunks:
                output.write(json.dumps(chunk.as_dict(adapter), ensure_ascii=False) + "\n")
                warnings.update(chunk.metadata.get("warnings", []))
            if destination is None:
                output.seek(0)
                shutil.copyfileobj(output, sys.stdout, length=65536)
        if destination is not None:
            os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return warnings


def parser():
    root = argparse.ArgumentParser(prog="ragwrench table-chunk", description="Model-free, table-aware document chunking")
    root.add_argument("--version", action="version", version="ragwrench table-chunk 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="split Markdown or HTML into JSONL")
    split.add_argument("input", help="UTF-8 file, or - for stdin")
    split.add_argument("--stream", action="store_true",
                       help="disk-backed large Markdown or HTML input; bounded memory and atomic JSONL output")
    split.add_argument("--format", choices=("auto", "markdown", "html"), default="auto")
    split.add_argument("--max-tokens", type=int, default=256)
    split.add_argument("--overlap-rows", type=int, default=0,
                       help="repeat up to 0..100 whole suffix rows when a new row also fits (default: 0)")
    split.add_argument("--mode", choices=("markdown", "kv", "html"), default="markdown")
    split.add_argument("--adapter", choices=("jsonl", "langchain", "llamaindex"), default="jsonl")
    split.add_argument("--counter", choices=("regex", "tiktoken"), default="regex")
    split.add_argument("--encoding", default="cl100k_base")
    split.add_argument("-o", "--output")
    bench = commands.add_parser("bench", help="run pure-Python BM25 comparison, offline by default")
    bench.add_argument("--rows", type=int, default=60)
    bench.add_argument("--tables", type=int, default=4)
    bench.add_argument("--max-tokens", type=int, default=120)
    bench.add_argument("--k", type=int, default=3)
    bench.add_argument("--corpus", help="replay a Markdown corpus; requires --queries")
    bench.add_argument("--queries", help="JSON array with query/entity/column/answer")
    bench.add_argument("--fixture-dir", help="write generated corpus.md and queries.json into a new directory")
    bench.add_argument("-o", "--output")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "split":
            input_format = arguments.format
            if input_format == "auto":
                input_format = "html" if Path(arguments.input).suffix.lower() in (".html", ".htm") else "markdown"
            kwargs = {}
            if arguments.counter == "tiktoken":
                kwargs["counter"] = tiktoken_counter(arguments.encoding)
            if arguments.stream:
                with (nullcontext(sys.stdin.buffer) if arguments.input == "-" else open(arguments.input, "rb")) as stream:
                    chunks = stream_document(stream, max_tokens=arguments.max_tokens, mode=arguments.mode,
                                             input_format=input_format, source=arguments.input,
                                             overlap_rows=arguments.overlap_rows, **kwargs)
                    try:
                        warnings = write_chunks(arguments.output, chunks, arguments.adapter)
                    finally:
                        chunks.close()
            else:
                text = read_text(arguments.input)
                chunks = split_document(text, max_tokens=arguments.max_tokens, mode=arguments.mode,
                                        input_format=input_format, source=arguments.input,
                                        overlap_rows=arguments.overlap_rows, **kwargs)
                serialized = "".join(json.dumps(chunk.as_dict(arguments.adapter), ensure_ascii=False) + "\n" for chunk in chunks)
                write_text(arguments.output, serialized)
                warnings = {w for chunk in chunks for w in chunk.metadata.get("warnings", [])}
            for warning in sorted(warnings):
                print(f"ragwrench table-chunk warning: {warning}", file=sys.stderr)
        else:
            corpus, queries = None, None
            if bool(arguments.corpus) != bool(arguments.queries):
                raise TableChunkError("--corpus and --queries must be used together")
            if arguments.corpus:
                corpus = read_text(arguments.corpus)
                queries = json.loads(read_text(arguments.queries))
                if not isinstance(queries, list):
                    raise TableChunkError("queries must be a JSON array")
            if arguments.fixture_dir and arguments.corpus:
                raise TableChunkError("--fixture-dir only applies to generated fixtures")
            result = run_benchmark(rows=arguments.rows, tables=arguments.tables,
                                   max_tokens=arguments.max_tokens, k=arguments.k,
                                   corpus=corpus, queries=queries)
            if arguments.fixture_dir:
                directory = Path(arguments.fixture_dir)
                directory.mkdir(exist_ok=False)
                corpus, queries = synthetic_fixture(arguments.rows, arguments.tables)
                write_text(str(directory / "corpus.md"), corpus)
                write_text(str(directory / "queries.json"), json.dumps(queries, indent=2) + "\n")
            write_text(arguments.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(f"ragwrench table-chunk: {exc}", file=sys.stderr)
        return 2
    return 0
