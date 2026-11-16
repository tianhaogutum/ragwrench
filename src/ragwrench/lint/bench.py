"""Committed synthetic exports, measured by the same public lint/fix engine."""
from importlib.resources import files
from pathlib import Path

from .fix import fix
from .io import load_chunks, read_text
from .model import Config


def benchmark():
    root = files("ragwrench.lint").joinpath("data")
    source = read_text(Path(str(root.joinpath("doc.md"))))
    chunks = load_chunks(Path(str(root.joinpath("chunks_langchain.jsonl"))))
    config = Config(min_chars=0, max_chars=160, near_duplicate=0.95)
    result = fix(chunks, source, config)
    rerun = fix(result.chunks, source, config)
    return {"fixture": "synthetic-langchain-document-export-v1", "config": {
        "min_chars": config.min_chars, "max_chars": config.max_chars, "near_duplicate": config.near_duplicate},
        "before": result.before.stats, "after": result.after.stats,
        "changes": result.changes, "invariant": result.invariant,
        "idempotent": [c.record() for c in result.chunks] == [c.record() for c in rerun.chunks],
        "method": "Same rules/config before and after; Unicode characters; no timer or external splitter."}
