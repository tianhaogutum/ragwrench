import json
from pathlib import Path

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check
from ragwrench.citecheck.model import InputError

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("format,expected", [
    ("llamaindex", [(1, "rover-chunk", "rover.txt"), (2, "battery-chunk", "battery.txt")]),
    ("dify", [(1, "warranty-1", "warranty.md"), (2, "returns-1", "returns.md")]),
    ("ragflow", [(1, "rover-chunk", "rover.txt"), (2, "battery-chunk", "battery.txt")]),
])
def test_framework_e2e_mapping(example, format, expected):
    record = example(format)
    assert [(s.idx, s.id, s.title) for s in record.sources] == expected
    report = check(record)
    assert report["findings"] == []
    assert [s["citation_indices"] for s in report["sentences"]] == [[1], [2]]


def test_generic_jsonl():
    for line in (ROOT / "examples/cite-check/generic.jsonl").read_text().splitlines():
        assert check(adapt(json.loads(line)))["findings"] == []


def test_sparse_out_of_order_dify_mapping():
    record = adapt({"answer": "The satellite uses solar panels [7].", "metadata": {"retriever_resources": [
        {"position": 7, "segment_id": "chunk7", "content": "The satellite uses solar panels."},
        {"position": 2, "content": "The ocean is blue."}]}}, "dify")
    assert [s.idx for s in record.sources] == [7, 2]
    assert [(f["code"], f["source_indices"]) for f in check(record)["findings"]] == [("CC007", (2,))]


def test_llamaindex_ignores_document_rank():
    record = adapt({"response": "The satellite uses solar panels [1].", "source_nodes": [
        {"node": {"id_": "f39", "text": "The satellite uses solar panels.", "metadata": {"position": 99}}}]}, "llamaindex")
    assert record.sources[0].idx == 1
    assert check(record)["findings"] == []


@pytest.mark.parametrize("format,payload,match", [
    ("llamaindex", {"answer": "a", "sources": []}, "response"),
    ("dify", {"answer": "a", "metadata": {}}, "retriever_resources"),
    ("ragflow", {"answer": "a", "reference": {"chunks": {}}}, "array"),
    ("ragflow", {"code": 100, "data": {}}, "error envelope"),
    ("ragflow", {"answer": "a", "reference": {"chunks": [{"content_ltks": "abc"}]}}, "content"),
    ("generic", [], "object"),
    ("generic", {"answer": 3, "sources": []}, "string"),
    ("generic", {"answer": "", "sources": []}, "non-empty"),
    ("generic", {"answer": "a", "sources": [{"idx": "1", "text": "x"}]}, "integer"),
    ("generic", {"answer": "a", "sources": [{"idx": True, "text": "x"}]}, "integer"),
    ("generic", {"answer": "a", "sources": [{"idx": 0, "text": "x"}]}, "integer"),
    ("generic", {"answer": "a", "sources": [{"idx": 1, "text": " "}]}, "empty"),
    ("generic", {"answer": "a", "sources": [{"idx": 1, "text": "x"}, {"idx": 1, "text": "y"}]}, "duplicate"),
    ("generic", {"answer": "a", "sources": [{"idx": 1, "id": 3, "text": "x"}]}, "string"),
    ("dify", {"answer": ["stream"], "metadata": {"retriever_resources": []}}, "string"),
])
def test_invalid_exports_fail_clearly(format, payload, match):
    with pytest.raises(InputError, match=match):
        adapt(payload, format)


def test_ragflow_newer_chunk_fields():
    record = adapt({"answer": "The rover uses solar panels ##0$$.", "reference": {"chunks": [
        {"id": "chunk", "content": "The rover uses solar panels.", "document_name": "rover.md"}]}}, "ragflow")
    assert (record.sources[0].id, record.sources[0].title) == ("chunk", "rover.md")
    assert check(record)["findings"] == []
