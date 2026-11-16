"""Bare URLs are opaque outside code, at the existing whitespace/< > boundary."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.markers import parse_markers
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("url", [
    "https://example.invalid/`",
    "http://example.invalid/``",
    "https://example.invalid/`[8]999kg\"absent\"",
    r"https://example.invalid/\`",
    r"https://example.invalid/\\`",
    "https://example.invalid/`)[8]`",  # Punctuation is part of the URL.
])
def test_bare_url_backticks_cannot_open_code(url):
    prefix, tail = "说明🙂 ", " literal [9]`"
    answer = prefix + url + tail
    assert mask_markdown(answer) == prefix + " " * len(url) + tail
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
    ("`https://example.invalid/ [9]`", " " * len("`https://example.invalid/ [9]`")),
    ("``https://example.invalid/` [9]``", " " * len("``https://example.invalid/` [9]``")),
    # URL-looking code is literal, even if its backtick supplies the closer.
    ("`https://example.invalid/`literal [9]", " " * len("`https://example.invalid/`") + "literal [9]"),
    (r"\`https://example.invalid/` literal [9]`", "\\`" + " " * len("https://example.invalid/`") + " literal [9]`"),
    (r"\\`https://example.invalid/ [9]`", "\\\\" + " " * len("`https://example.invalid/ [9]`")),
    ('<span title="https://example.invalid/`">literal [9]`', ' ' * len('<span title="https://example.invalid/`">') + 'literal [9]`'),
    ("[archive](https://example.invalid/`)literal [9]`", " archive" + " " * len("](https://example.invalid/`)") + "literal [9]`"),
])
def test_url_code_escape_tag_and_link_precedence(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


@pytest.mark.parametrize("separator", [" ", "\t", "\n", "<em>", ">"])
def test_url_keeps_existing_termination_rule(separator):
    url = "https://example.invalid/`[8]"
    answer = "说明🙂 " + url + separator + "literal [9]`"
    masked_separator = " " * len(separator) if separator == "<em>" else separator
    assert mask_markdown(answer) == "说明🙂 " + " " * len(url) + masked_separator + "literal [9]`"
    marker, = parse_markers(mask_markdown(answer))
    assert marker.indices == (9,)


def test_url_contents_do_not_contaminate_lexical_checks():
    answer = 'Archive manual https://example.invalid/`[8]999kg"absent" works [1]`.'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 1, "text": "Archive manual works."}]}))
    assert report["findings"] == []
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [1] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_bare_url_cli_default_gate_and_offsets():
    result = cli("check", ROOT / "examples/cite-check/url_boundaries.jsonl", "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
    broken, code = output["records"]
    sentence, = broken["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (
        0, 42, "说明🙂 https://example.invalid/` literal [9]`")
    assert sentence["citation_indices"] == [9]
    finding, = broken["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"] == {"marker": "[9]", "start": 38, "end": 41, "kind": "dangling"}
    assert code["findings"] == [] and code["sentences"][0]["citation_indices"] == []
