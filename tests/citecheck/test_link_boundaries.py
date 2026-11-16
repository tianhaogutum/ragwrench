"""Inline destinations are opaque; earlier code retains delimiter precedence."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.markers import parse_markers
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("opener,destination", [
    ("[", "https://example.invalid/`"),
    ("[", "https://example.invalid/``"),
    ("[", "`[8] 999 kg <span"),
    ("[", r"\`"),
    ("[", r"\\`"),
    ("![", "`"),
])
def test_link_destinations_cannot_open_code(opener, destination):
    prefix, label, tail = "说明🙂 ", "archive", "literal [9]`"
    answer = prefix + opener + label + "](" + destination + ")" + tail
    masked = mask_markdown(answer)
    assert masked == prefix + " " * len(opener) + label + " " * (len(destination) + 3) + tail
    marker, = parse_markers(masked)
    assert marker.indices == (9,) and marker.error is None
    report = check(adapt({"answer": answer, "sources": []}))
    assert fails([report], "error")
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
    assert sentence["citation_indices"] == [9]
    finding, = report["findings"]
    assert finding["code"] == "CC001" and finding["source_indices"] == (9,)
    assert finding["evidence"] == {"marker": "[9]", "start": answer.index("[9]"),
                                   "end": answer.index("[9]") + 3, "kind": "dangling"}


@pytest.mark.parametrize("body,expected", [
    ("`[archive](url) [9]`", " " * len("`[archive](url) [9]`")),
    ("``[archive](` [9])``", " " * len("``[archive](` [9])``")),
    # A link-looking destination in already-open code can supply its closer.
    ("`[archive](url`)[9]", " " * len("`[archive](url`") + ")[9]"),
    # Label content is still scanned as prose, code or HTML as appropriate.
    ("[archive `code [8]`](url`)[9]`", " archive " + " " * len("`code [8]`") + " " * len("](url`)") + "[9]`"),
    ('[archive <em title="`">](url`)[9]`', ' archive ' + ' ' * len('<em title="`">') + ' ' * len('](url`)') + '[9]`'),
    # If code starts in the label and consumes the closing bracket, no link forms.
    ("[archive `](url`)[9]", "[archive " + " " * len("`](url`") + ")[9]"),
    ("[archive](url`)`code [8]` [9]`", " archive" + " " * len("](url`)") + " " * len("`code [8]`") + " [9]`"),
    # A tag consumes its link-looking attribute before brackets are considered.
    ('<span title="[archive](url`)">[9]`', ' ' * len('<span title="[archive](url`)">') + '[9]`'),
])
def test_link_code_and_html_precedence(body, expected):
    answer = "说明🙂 " + body
    assert mask_markdown(answer) == "说明🙂 " + expected


def test_label_remains_lexical_prose_but_destination_does_not():
    answer = 'Archive manual [documents](https://example.invalid/`[8]999kg"absent") work [1]`.'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 1, "text": "Archive manual documents work."}]}))
    assert report["findings"] == []
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [1]
    assert sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_unclosed_destination_does_not_form_a_link():
    answer = "说明🙂 [archive](url [9]"
    assert mask_markdown(answer) == answer
    markers = parse_markers(answer)
    assert markers[0].error  # Unsupported syntax remains observable, not hidden.


def test_link_boundary_cli_default_gate_and_offsets():
    result = cli("check", ROOT / "examples/cite-check/link_boundaries.jsonl", "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
    broken, code = output["records"]
    sentence, = broken["sentences"]
    assert (sentence["start"], sentence["end"]) == (0, 52)
    assert sentence["text"] == "说明🙂 [archive](https://example.invalid/`)literal [9]`"
    assert sentence["citation_indices"] == [9]
    finding, = broken["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"] == {"marker": "[9]", "start": 48, "end": 51, "kind": "dangling"}
    assert code["findings"] == [] and code["sentences"][0]["citation_indices"] == []
