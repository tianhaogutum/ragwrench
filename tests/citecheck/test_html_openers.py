"""Whitespace after < is prose; valid tag/code/link/URL masks keep precedence."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.markers import parse_markers
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("body", [
    "< limit [9] >", "x < y [9] > z", "< [9] >", "<  limit [9] >",
    "<\tlimit [9] >", "<\rlimit [9] >", "<\vlimit [9] >", "<\flimit [9] >",
    "<\u00a0limit [9] >", "<\u2003limit [9] >",
])
def test_whitespace_after_angle_opener_keeps_prose_and_citation(body):
    answer = "说明🙂 " + body
    assert mask_markdown(answer) == answer
    marker, = parse_markers(mask_markdown(answer))
    assert marker.indices == (9,) and marker.error is None
    report = check(adapt({"answer": answer, "sources": []}))
    assert fails([report], "error")
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
    assert sentence["citation_indices"] == [9]
    finding, = report["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"] == {"marker": "[9]", "start": answer.index("[9]"),
                                   "end": answer.index("[9]") + 3, "kind": "dangling"}


@pytest.mark.parametrize("body,expected", [
    ("<\nlimit [9] >", "<\nlimit [9] >"),
    ("<> [9]", "<> [9]"),
    ("<em>[9]</em>", " " * 4 + "[9]" + " " * 5),
    ("<img/>[9]", " " * 6 + "[9]"),
    ('<span title=">` [8]">[9]`', ' ' * len('<span title=">` [8]">') + '[9]`'),
    ("<span title='>` [8]'>[9]`", ' ' * len("<span title='>` [8]'>") + '[9]`'),
    ('< <em title=">` [8]">[9]`', '< ' + ' ' * len('<em title=">` [8]">') + '[9]`'),
    ("`< limit [9] >`", " " * len("`< limit [9] >`")),
    ("``< limit ` [9] >``", " " * len("``< limit ` [9] >``")),
    ("`< limit `> [9]", " " * len("`< limit `") + "> [9]"),
    ("< `code [8]` [9] >", "< " + " " * len("`code [8]`") + " [9] >"),
    (r"\`< limit [9] >`", r"\`< limit [9] >`"),
    (r"\\`< limit [9] >`", "\\\\" + " " * len("`< limit [9] >`")),
    ("[archive](< limit `[8] >)[9]`", " archive" + " " * len("](< limit `[8] >)") + "[9]`"),
    ("<https://example.invalid/[8]> [9]", " " * len("<https://example.invalid/[8]>") + " [9]"),
    ("< https://example.invalid/[8]> [9]", "< " + " " * len("https://example.invalid/[8]") + "> [9]"),
])
def test_literal_angles_preserve_other_inline_boundaries(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


def test_angle_prose_remains_available_to_lexical_checks():
    answer = "Archive < limit [1] > manual"
    report = check(adapt({"answer": answer, "sources": [{"idx": 1, "text": "Archive limit manual"}]}))
    assert report["findings"] == []
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [1] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_html_opener_cli_original_offsets_and_quoted_tag_controls():
    result = cli("check", ROOT / "examples/cite-check/html_openers.jsonl", "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
    prose, double, single, code, url = output["records"]
    for record, start, end in [(prose, 12, 17), (double, 29, 33), (single, 29, 33)]:
        sentence, = record["sentences"]
        answer = sentence["text"]
        assert sentence["start"] == 0 and sentence["end"] == end
        assert sentence["citation_indices"] == [9]
        finding, = record["findings"]
        assert finding["code"] == "CC001"
        assert finding["evidence"] == {"marker": "[9]", "start": start, "end": start + 3, "kind": "dangling"}
        assert answer[start:start + 3] == "[9]"
    assert prose["sentences"][0]["text"] == "说明🙂 < limit [9] >"
    for record in [code, url]:
        assert record["findings"] == [] and record["sentences"][0]["citation_indices"] == []
