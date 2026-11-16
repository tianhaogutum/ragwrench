"""Bounded same-line destination grammar, fallback and original offset contracts."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.markers import parse_markers
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("opener,destination", [
    ("[", "part(one)[8]"),
    ("[", "part(one(two)[8])[abc]"),
    ("[", "https://example.invalid/a(b(c))[8]"),
    ("[", 'part(one)[abc]【8】 Source 8 ##7$$ [ID:7] 999kg"absent"'),
    ("[", "part(one)`[8]"),
    ("[", r"part\(literal [8]"),
    ("[", r"part\)literal [8]"),
    ("[", r"part\\(nested)[8]"),
    ("[", r"part\\\(literal [8]"),
    ("[", r"part\\\)literal [8]"),
    ("![", "part(one)[8]"),
])
def test_complete_destination_masks_only_inside_markers(opener, destination):
    prefix, label, tail = "说明🙂 ", "archive", " [9]`"
    answer = prefix + opener + label + "](" + destination + ")" + tail
    expected = prefix + " " * len(opener) + label + " " * (len(destination) + 3) + tail
    assert mask_markdown(answer) == expected
    marker, = parse_markers(mask_markdown(answer))
    assert (marker.indices, marker.error, marker.start, marker.end) == (
        (9,), None, answer.index("[9]"), answer.index("[9]") + 3)
    report = check(adapt({"answer": answer, "sources": []}))
    assert fails([report], "error")
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
    assert sentence["citation_indices"] == [9]
    finding, = report["findings"]
    assert finding["code"] == "CC001" and finding["severity"] == "error"
    assert finding["evidence"] == {"marker": "[9]", "start": answer.index("[9]"),
                                   "end": answer.index("[9]") + 3, "kind": "dangling"}


@pytest.mark.parametrize("body,expected", [
    ("[archive](part(one)[8])[9]", " archive" + " " * len("](part(one)[8])") + "[9]"),
    ("[archive]()[9]", " archive   [9]"),
    ("[archive](url)[9]", " archive" + " " * len("](url)") + "[9]"),
    (r"[archive](part\\)[9]", " archive" + " " * len(r"](part\\)") + "[9]"),
    ("``[archive](part(one)[8])``[9]", " " * len("``[archive](part(one)[8])``") + "[9]"),
    ("`[archive](part(one)`[9]", " " * len("`[archive](part(one)`") + "[9]"),
    ("[archive `code [8]`](part(one)`)[9]`", " archive " + " " * len("`code [8]`") + " " * len("](part(one)`)") + "[9]`"),
    (r"\`[archive](part(one)[8])[9]`", r"\` archive" + " " * len("](part(one)[8])") + "[9]`"),
    (r"\\`[archive](part(one)[8])[9]`", "\\\\" + " " * len("`[archive](part(one)[8])[9]`")),
    ('[archive <em title="`">](part(one)`)[9]`', " archive " + " " * len('<em title="`">') + " " * len("](part(one)`)") + "[9]`"),
    ("<https://example.invalid/a(b)[8]> [9]", " " * len("<https://example.invalid/a(b)[8]>") + " [9]"),
    ("https://example.invalid/a(b)[8] [9]", " " * len("https://example.invalid/a(b)[8]") + " [9]"),
])
def test_nested_links_preserve_code_label_url_and_escape_precedence(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


@pytest.mark.parametrize("body", [
    "[archive](part [9]",  # Existing unclosed fallback.
    "[archive](part(one)[9]",  # No partial mask at the inner close.
    "[archive](part\n(one)[9])",
    "[archive](part\r\n(one)[9])",
    "[archive] (part(one)[9])",  # No space permitted before the opener.
    "[archive][9]",  # Reference links remain unsupported.
])
def test_unsupported_candidates_stay_prose(body):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + body
    assert any(marker.error for marker in parse_markers(mask_markdown(body)))


@pytest.mark.parametrize("depth,supported", [(1, True), (64, True), (65, False)])
def test_destination_depth_boundary(depth, supported):
    suffix = "(" * (depth - 1) + "[8]" + ")" * depth
    answer = "[archive](" + suffix + " [9]"
    expected = " archive" + " " * (len(suffix) + 2) + " [9]" if supported else answer
    assert mask_markdown(answer) == expected


@pytest.mark.parametrize("scan_length,supported", [(9999, True), (10000, True), (10001, False)])
def test_destination_scan_boundary(scan_length, supported):
    # Budget includes both outer parentheses, not the preceding label bracket.
    destination = "x" * (scan_length - 5) + "[8]"
    answer = "[archive](" + destination + ") [9]"
    expected = " archive" + " " * (len(destination) + 3) + " [9]" if supported else answer
    assert mask_markdown(answer) == expected


def test_rejected_candidate_keeps_later_scanning_rules():
    answer = "[archive](part(one) `code [8]` [9]"
    assert mask_markdown(answer) == "[archive](part(one) " + " " * len("`code [8]`") + " [9]"


def test_nested_destination_does_not_contaminate_existing_source_checks():
    answer = 'Archive manual [documents](part(one)[8]999kg"absent"`) work [4]`.'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 4, "id": "manual", "text": "Archive manual documents work."}]}))
    assert report["findings"] == []
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [4] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_nested_destination_cli_offsets_and_lexical_checks():
    fixture = ROOT / "examples/cite-check/nested_destinations.jsonl"
    answers = [json.loads(line)["answer"] for line in fixture.read_text().splitlines()]
    result = cli("check", fixture, "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
    broken, lexical, code = output["records"]
    sentence, = broken["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, 31, answers[0])
    assert sentence["citation_indices"] == [9]
    finding, = broken["findings"]
    assert finding["code"] == "CC001" and finding["severity"] == "error"
    assert finding["evidence"] == {"marker": "[9]", "start": 28, "end": 31, "kind": "dangling"}
    assert lexical["findings"] == [] and lexical["sentences"][0]["citation_indices"] == [4]
    assert lexical["sentences"][0]["cited_coverage"] == 1
    assert code["findings"] == [] and code["sentences"][0]["citation_indices"] == []
