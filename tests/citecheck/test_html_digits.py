"""Immediate decimal-digit angle openers are prose, with other masks intact."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.markers import parse_markers
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("body", [
    "<3 [9] >", "x <0 [9]> z", "<9thing [9]/>", "<12 [9] >",
    "<٣ [9] >", "<３ [9] >",
    '<3 title="> [9]">', "<3 title='> [9]'>",
])
def test_digit_led_angles_keep_prose_and_original_offsets(body):
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
    ("< limit [9] >", "< limit [9] >"),
    ("<em>[9]</em>", " " * 4 + "[9]" + " " * 5),
    ("<h3>[9]</h3>", " " * 4 + "[9]" + " " * 5),
    ("<img/>[9]", " " * 6 + "[9]"),
    ('<span title=">` [8]">[9]`', " " * len('<span title=">` [8]">') + "[9]`"),
    ("<span title='>` [8]'>[9]`", " " * len("<span title='>` [8]'>") + "[9]`"),
    ('<3 <em title=">` [8]">[9]`', "<3 " + " " * len('<em title=">` [8]">') + "[9]`"),
    ("`<3 [9] >`", " " * len("`<3 [9] >`")),
    ("``<3 ` [9] >``", " " * len("``<3 ` [9] >``")),
    ("`<3 `> [9]", " " * len("`<3 `") + "> [9]"),
    ("<3 `code [8]` [9] >", "<3 " + " " * len("`code [8]`") + " [9] >"),
    (r"\`<3 [9] >`", r"\`<3 [9] >`"),
    (r"\\`<3 [9] >`", "\\\\" + " " * len("`<3 [9] >`")),
    ("[archive](<3 `[8] >)[9]`", " archive" + " " * len("](<3 `[8] >)") + "[9]`"),
    ("<3 [archive](url`)[9]` >", "<3  archive" + " " * len("](url`)") + "[9]` >"),
    ("<https://example.invalid/[8]> [9]", " " * len("<https://example.invalid/[8]>") + " [9]"),
    ("<http://example.invalid/[8]> [9]", " " * len("<http://example.invalid/[8]>") + " [9]"),
    ("<3 https://example.invalid/[8] [9] >", "<3 " + " " * len("https://example.invalid/[8]") + " [9] >"),
])
def test_digit_led_angles_preserve_tag_code_link_and_url_boundaries(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


def test_digit_led_prose_remains_available_to_lexical_and_quantity_checks():
    answer = "Archive <3 [1] > manual"
    report = check(adapt({"answer": answer, "sources": [{"idx": 1, "text": "Archive 3 manual"}]}))
    assert report["findings"] == []
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [1] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_digit_led_cli_default_gate_and_original_offsets():
    result = cli("check", ROOT / "examples/cite-check/html_digits.jsonl", "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
    digit, wide, double, single, code, url, link = output["records"]
    assert digit["sentences"][0]["text"] == "说明🙂 <3 [9] >"
    for record, start, end in [(digit, 7, 12), (wide, 7, 12), (double, 29, 33), (single, 29, 33), (link, 19, 23)]:
        sentence, = record["sentences"]
        assert (sentence["start"], sentence["end"]) == (0, end)
        assert sentence["citation_indices"] == [9]
        finding, = record["findings"]
        assert finding["code"] == "CC001"
        assert finding["evidence"] == {"marker": "[9]", "start": start, "end": start + 3, "kind": "dangling"}
        assert sentence["text"][start:start + 3] == "[9]"
    for record in [code, url]:
        assert record["findings"] == [] and record["sentences"][0]["citation_indices"] == []


def test_exact_digit_led_cli_contract():
    answer = "说明🙂 <3 [9] >"
    result = cli("check", "-", "--report", "json",
                 stdin=json.dumps({"answer": answer, "sources": []}))
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
    record, = output["records"]
    sentence, = record["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, 12, answer)
    assert sentence["citation_indices"] == [9]
    finding, = record["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"] == {"marker": "[9]", "start": 7, "end": 10, "kind": "dangling"}
