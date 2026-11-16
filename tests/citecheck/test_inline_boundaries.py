"""Contracts for prose escapes and the boundary between HTML tags and code."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.markers import parse_markers
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("body,visible", [
    (r"\`literal [9]\`", True),
    (r"`literal [9]\`", False),
    (r"\\`literal [9]`", False),
    ('<span title="`">literal [9]`', True),
    ('<span title="``">literal [9]``', True),
    ('<span title="\\\\`">literal [9]`', True),
    ('<span title="`">literal [9]', True),
    ('<span title="`">`literal [9]`', False),
    ('<span title="`">``literal ` [9]``', False),
    ('`<span title="[9]">`', False),
    ('`<span title="`">literal [9]', True),
    ('`<span title="`">`literal [9]`', False),
    ('<span title="`"><em title="``">literal [9]``', True),
])
def test_inline_boundary_citations_and_original_offsets(body, visible):
    answer = "说明🙂 " + body
    masked = mask_markdown(answer)
    assert len(masked) == len(answer)
    assert [m.indices for m in parse_markers(masked)] == ([(9,)] if visible else [])
    report = check(adapt({"answer": answer, "sources": []}))
    assert fails([report], "error") is visible
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
    if visible:
        finding, = report["findings"]
        assert finding["code"] == "CC001"
        assert finding["evidence"] == {"marker": "[9]", "start": answer.index("[9]"),
                                       "end": answer.index("[9]") + 3, "kind": "dangling"}
    else:
        assert report["findings"] == []


def test_html_backticks_do_not_contaminate_supported_prose():
    answer = 'Supported syntax <span title="` [8] 999 kg">is literal [1]`.'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 1, "text": "Supported syntax is literal."}]}))
    assert report["findings"] == []
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [1]
    assert sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_inline_boundary_cli_jsonl_keeps_unicode_offsets():
    fixture = ROOT / "examples/cite-check/inline_boundaries.jsonl"
    answers = [json.loads(line)["answer"] for line in fixture.read_text().splitlines()]
    assert len(answers) == 4
    result = cli("check", fixture, "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["record_count"] == len(answers) and output["gate_failed"]
    for answer, record, visible in zip(answers, output["records"], [True, False, True, False]):
        assert record["sentences"][0]["text"] == answer
        if visible:
            finding, = record["findings"]
            assert finding["code"] == "CC001"
            start, end = finding["evidence"]["start"], finding["evidence"]["end"]
            assert (start, end) == (answer.index("[9]"), answer.index("[9]") + 3)
            assert answer[start:end] == "[9]"
        else:
            assert record["findings"] == []
