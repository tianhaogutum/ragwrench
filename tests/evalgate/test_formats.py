import json
from pathlib import Path

import pytest

from ragwrench.evalgate.formats import InputError, read_qrels, read_run, write_qrels, write_run

FIXTURES = Path(__file__).parent / "fixtures"


def put(tmp_path, text):
    path = tmp_path / "data"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("text,line,match", [
    ("q 0 d 1\nq 0 bad\n", 2, "expected 4"),
    ("q 0 d 1\nq 0 d 2\n", 2, "duplicate"),
    ("q 0 d -1\n", 1, "relevance"), ("q 0 d 1.5\n", 1, "relevance"),
    ("q 0 d NaN\n", 1, "relevance"), ("q 0 d 101\n", 1, "relevance"),
    ("", 1, "empty"), ("query-id\tcorpus-id\tscore\nq\td\tnonsense\n", 2, "relevance"),
    ("query-id\tcorpus-id\tscore\nq\td\t1\textra\n", 2, "expected 3"),
])
def test_qrels_errors_line_numbers(tmp_path, text, line, match):
    path = put(tmp_path, text)
    with pytest.raises(InputError, match=rf"{line}: .*{match}"):
        read_qrels(path)


@pytest.mark.parametrize("text,line,match", [
    ("q Q0 d 1 1 tag\nwrong\n", 2, "expected 6"),
    ("q Q0 d 1 1 tag\nq Q0 d 2 2 tag\n", 2, "duplicate"),
    ("q Q0 d 1 NaN tag\n", 1, "finite"), ("q Q0 d 1 inf tag\n", 1, "finite"),
    ("q Q0 d 1 1e999 tag\n", 1, "finite"), ("q Q0 d 1 nope tag\n", 1, "finite"),
    ("q Q0 d 0 1 tag\n", 1, "positive integer"), ("q Q0 d 1.5 1 tag\n", 1, "positive integer"),
    ("q 00 d 1 1 tag\n", 1, "Q0"), ("\n", 1, "empty"),
    ('{"query_id":"q","doc_id":"d","score":1}\n{"query_id":"q","doc_id":"d","score":2}\n', 2, "duplicate"),
    ('{"query_id":"q","doc_id":"d","score":true}\n', 1, "finite"),
    ('{"query_id":"q","doc_id":"d","score":NaN}\n', 1, "non-finite"),
    ('{"query_id":"q","doc_id":"d","score":1,"score":2}\n', 1, "duplicate JSON key"),
    ('{"query_id":"q","doc_id":"d","score":1,"extra":0}\n', 1, "exactly"),
    ('{"query_id":"q","doc_id":5,"score":1}\n', 1, "document ID"),
    ('{"query_id":"q","doc_id":"has space","score":1}\n', 1, "document ID"),
    ('{"query_id":"q","doc_id":"d","score":1}\n{broken}\n', 2, "invalid JSON"),
])
def test_run_errors_line_numbers(tmp_path, text, line, match):
    path = put(tmp_path, text)
    with pytest.raises(InputError, match=rf"{line}: .*{match}"):
        read_run(path)


def test_qrels_roundtrip(tmp_path):
    data = {"q2": {"d3": 0}, "q1": {"d2": 2, "d1": 1}}
    for fmt in ("trec", "beir"):
        assert read_qrels(put(tmp_path, write_qrels(data, fmt))) == data


def test_run_roundtrip_and_rank_ignored(tmp_path):
    data = {"q": {"z": -0.0, "b": -2e-100, "a": 1.2345678901234567}}
    for fmt in ("trec", "jsonl"):
        assert read_run(put(tmp_path, write_run(data, fmt))) == data
    assert read_run(put(tmp_path, "q Q0 a 99 2 tag\nq Q0 b 1 1 tag\n")) == {"q": {"a": 2, "b": 1}}


def test_llama_fixture_and_jsonl_fixture():
    assert read_qrels(FIXTURES / "llamaindex.json") == {"q1": {"d1": 1, "d3": 1}, "q2": {"d2": 1}}
    assert read_run(FIXTURES / "results.jsonl")["q1"] == {"d1": 2, "d3": 1}


@pytest.mark.parametrize("change", [
    {"queries": []}, {"relevant_docs": {}}, {"relevant_docs": {"q": ["missing"]}},
    {"relevant_docs": {"q": ["d", "d"]}}, {"relevant_docs": {"q": "d"}},
    {"corpus": {"d": 5}}, {"mode": "image"}, {"new_field": 1},
    {"relevant_docs": {"q": [[]]}},
])
def test_llama_mapping_boundaries(tmp_path, change):
    data = {"queries": {"q": "Question"}, "corpus": {"d": "Doc"}, "relevant_docs": {"q": ["d"]}}
    data.update(change)
    with pytest.raises(InputError, match="1:"):
        read_qrels(put(tmp_path, json.dumps(data)))


def test_empty_llama_judgments_cannot_be_silently_lost(tmp_path):
    data = {"queries": {"q": "Question"}, "corpus": {}, "relevant_docs": {"q": []}}
    parsed = read_qrels(put(tmp_path, json.dumps(data)))
    assert parsed == {"q": {}}
    with pytest.raises(InputError, match="cannot preserve"):
        write_qrels(parsed, "trec")


def test_json_syntax_error_line(tmp_path):
    with pytest.raises(InputError, match="3: invalid JSON"):
        read_qrels(put(tmp_path, '{\n "queries": {},\n broken\n}'))


def test_explicit_format_and_invalid_utf8(tmp_path):
    with pytest.raises(InputError, match="BEIR header"):
        read_qrels(put(tmp_path, "q\td\t1\n"), "beir")
    path = tmp_path / "invalid"
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        read_run(path)


@pytest.mark.parametrize("docid", ["\u0085", "\u009b31m", "\ud800", "line\nfeed", "tab\there"])
def test_control_and_surrogate_ids_are_rejected(tmp_path, docid):
    text = json.dumps({"query_id": "q", "doc_id": docid, "score": 1})
    with pytest.raises(InputError, match="document ID"):
        read_run(put(tmp_path, text))


@pytest.mark.parametrize("value", ["1_000", " 1 ", "0x10", "١٢", "1;exit", "1.0junk"])
def test_numeric_strings_use_decimal_not_python_expression_syntax(tmp_path, value):
    text = json.dumps({"query_id": "q", "doc_id": "d", "score": value})
    with pytest.raises(InputError, match="finite number"):
        read_run(put(tmp_path, text))


@pytest.mark.parametrize("value,expected", [("1e-3", .001), ("-.25", -.25), ("+3.", 3), ("-0", 0)])
def test_finite_decimal_score_forms(tmp_path, value, expected):
    assert read_run(put(tmp_path, f"q Q0 d 1 {value} tag\n")) == {"q": {"d": expected}}
