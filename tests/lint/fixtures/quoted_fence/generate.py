"""Regenerate with langchain-text-splitters==0.3.11, langchain-core==0.3.79.

Run this script in a separate environment containing those exact packages.
The chunklint runtime and test suite do not need LangChain.
"""
from importlib.metadata import version
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
PINS = {"langchain-text-splitters": "0.3.11", "langchain-core": "0.3.79"}
OPTIONS = {
    "separators": ["\n"],
    "keep_separator": "end",
    "chunk_size": 65,
    "chunk_overlap": 0,
    "strip_whitespace": False,
    "add_start_index": True,
}

if __name__ == "__main__":
    assert {name: version(name) for name in PINS} == PINS
    source = (ROOT / "source.md").read_text(encoding="utf-8")
    documents = RecursiveCharacterTextSplitter(**OPTIONS).create_documents(
        [source], metadatas=[{"source": "synthetic-quoted-guide.md"}])
    records = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    assert "".join(record["page_content"] for record in records) == source
    (ROOT / "chunks.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    (ROOT / "provenance.json").write_text(json.dumps({
        "packages": PINS,
        "splitter": "langchain_text_splitters.RecursiveCharacterTextSplitter",
        "options": OPTIONS,
        "synthetic_source": "source.md",
        "chunks": len(records),
    }, indent=2) + "\n", encoding="utf-8")
