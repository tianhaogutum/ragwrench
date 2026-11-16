"""Strict, offline readers and canonical writers for retrieval exchange formats."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]


class InputError(ValueError):
    """Invalid user input (safe to display without a traceback)."""


def fail(path: str | Path, line: int, message: str):
    raise InputError(f"{path}:{line}: {message}")


def identifier(value, path, line, field):
    if not isinstance(value, str) or not value or any(c.isspace() or unicodedata.category(c) in {"Cc", "Cs"} for c in value):
        fail(path, line, f"{field} must be a nonempty string without whitespace/control characters")
    return value


def grade(value, path, line):
    # Bounding grades prevents overflow in exponential nDCG; negatives are unsupported.
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        fail(path, line, "relevance must be an integer in [0, 100]")
    if len(str(value)) > 3 or not 0 <= int(value) <= 100:
        fail(path, line, "relevance must be an integer in [0, 100]")
    return int(value)


def score(value, path, line):
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError
        if isinstance(value, str) and not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
            raise ValueError
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        fail(path, line, "score must be a finite number")
    if not math.isfinite(result):
        fail(path, line, "score must be a finite number")
    return result


def insert(data, qid, docid, value, path, line):
    identifier(qid, path, line, "query ID")
    identifier(docid, path, line, "document ID")
    docs = data.setdefault(qid, {})
    if docid in docs:
        fail(path, line, f"duplicate (query, document): {qid!r}, {docid!r}")
    docs[docid] = value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def decode(text, path, line=1):
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except json.JSONDecodeError as exc:
        fail(path, line + exc.lineno - 1, f"invalid JSON: {exc.msg}")
    except (ValueError, RecursionError) as exc:
        fail(path, line, str(exc))


def lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if raw.strip():
                yield line, raw.rstrip("\r\n")


def read_qrels(path: str | Path, fmt="auto") -> Qrels:
    if fmt == "auto":
        first = next(lines(path), (1, ""))[1].lstrip()
        fmt = "llamaindex" if first.startswith("{") else "beir" if first.startswith("query-id\t") else "trec"
    if fmt == "llamaindex":
        return _read_llama(path)
    if fmt not in {"trec", "beir"}:
        raise InputError(f"unknown qrels format: {fmt}")
    data: Qrels = {}
    rows = iter(lines(path))
    if fmt == "beir":
        line, header = next(rows, (1, ""))
        if header != "query-id\tcorpus-id\tscore":
            fail(path, line, "expected BEIR header: query-id<TAB>corpus-id<TAB>score")
    for line, raw in rows:
        fields = raw.split("\t") if fmt == "beir" else raw.split()
        if len(fields) != (3 if fmt == "beir" else 4):
            fail(path, line, f"expected {3 if fmt == 'beir' else 4} qrels fields")
        qid, docid, rel = fields if fmt == "beir" else (fields[0], fields[2], fields[3])
        insert(data, qid, docid, grade(rel, path, line), path, line)
    if not data:
        fail(path, 1, "empty qrels")
    return data


def _read_llama(path) -> Qrels:
    obj = decode(Path(path).read_text(encoding="utf-8"), path)
    if not isinstance(obj, dict) or not all(isinstance(obj.get(k), dict) for k in ("queries", "corpus", "relevant_docs")):
        fail(path, 1, "LlamaIndex QA dataset requires queries, corpus, relevant_docs objects")
    if set(obj) - {"queries", "corpus", "relevant_docs", "mode"}:
        fail(path, 1, "unsupported LlamaIndex dataset fields")
    if "mode" in obj and obj["mode"] != "text":
        fail(path, 1, "only text LlamaIndex datasets are supported")
    for field in ("queries", "corpus"):
        for key, text in obj[field].items():
            identifier(key, path, 1, field + " ID")
            if not isinstance(text, str):
                fail(path, 1, f"{field} values must be strings")
    if set(obj["relevant_docs"]) != set(obj["queries"]):
        fail(path, 1, "relevant_docs keys must exactly match queries")
    data: Qrels = {}
    for qid, docs in obj["relevant_docs"].items():
        if not isinstance(docs, list):
            fail(path, 1, "relevant_docs values must be document ID arrays")
        data[qid] = {}
        for docid in docs:
            identifier(docid, path, 1, "document ID")
            if docid not in obj["corpus"]:
                fail(path, 1, f"relevant document {docid!r} absent from corpus")
            insert(data, qid, docid, 1, path, 1)
    if not data:
        fail(path, 1, "empty queries")
    return data


def read_run(path: str | Path, fmt="auto") -> Run:
    if fmt == "auto":
        fmt = "jsonl" if next(lines(path), (1, ""))[1].lstrip().startswith("{") else "trec"
    if fmt not in {"trec", "jsonl"}:
        raise InputError(f"unknown run format: {fmt}")
    data: Run = {}
    for line, raw in lines(path):
        if fmt == "jsonl":
            obj = decode(raw, path, line)
            if not isinstance(obj, dict) or set(obj) != {"query_id", "doc_id", "score"}:
                fail(path, line, "JSONL result requires exactly query_id, doc_id, score")
            qid, docid, value = obj["query_id"], obj["doc_id"], obj["score"]
        else:
            fields = raw.split()
            if len(fields) != 6:
                fail(path, line, "expected 6 TREC run fields")
            qid, placeholder, docid, rank, value, _tag = fields
            if placeholder != "Q0" or not re.fullmatch(r"[0-9]+", rank) or not rank.strip("0"):
                fail(path, line, "TREC run requires Q0 and a positive integer rank (ranking uses scores)")
        insert(data, qid, docid, score(value, path, line), path, line)
    if not data:
        fail(path, 1, "empty run")
    return data


def ranked(docs: dict[str, float]) -> list[str]:
    """trec_eval: descending score, then descending lexicographic document ID."""
    return sorted(docs, key=lambda d: (docs[d], d), reverse=True)


def write_qrels(data: Qrels, fmt: str) -> str:
    if fmt not in {"trec", "beir"}:
        raise InputError("qrels output must be trec or beir")
    if any(not docs for docs in data.values()):
        raise InputError("TREC/BEIR qrels cannot preserve queries with zero judgments")
    result = ["query-id\tcorpus-id\tscore\n"] if fmt == "beir" else []
    for qid in sorted(data):
        for docid, rel in sorted(data[qid].items()):
            result.append(f"{qid}\t{docid}\t{rel}\n" if fmt == "beir" else f"{qid} 0 {docid} {rel}\n")
    return "".join(result)


def write_run(data: Run, fmt: str) -> str:
    if fmt not in {"trec", "jsonl"}:
        raise InputError("run output must be trec or jsonl")
    result = []
    for qid in sorted(data):
        for rank, docid in enumerate(ranked(data[qid]), 1):
            value = data[qid][docid]
            result.append(f"{qid} Q0 {docid} {rank} {value!r} retgate\n" if fmt == "trec" else json.dumps({"query_id": qid, "doc_id": docid, "score": value}, ensure_ascii=False, allow_nan=False) + "\n")
    return "".join(result)
