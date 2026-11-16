"""Whitespace immediately after </ starts prose, with inline precedence intact."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("whitespace", [
    " ", "\t", "\r", "\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x1f",
    "\x85", "\u00a0", "\u1680", *map(chr, range(0x2000, 0x200b)),
    "\u2028", "\u2029", "\u202f", "\u205f", "\u3000",
])
def test_whitespace_after_closing_slash_cannot_open_tag_mask(whitespace):
    answer = f"说明🙂 </{whitespace}[9] >"
    assert mask_markdown(answer) == answer


@pytest.mark.parametrize("body", [
    "</ [9] >", "</  [9] >", "</\t[9] >", "</\u00a0[9] >", "</\u2003[9] >",
    '</ title="> [9]">', "</ title='> [9]'>",
])
def test_closing_whitespace_keeps_dangling_citation_and_original_offsets(body):
    answer = "说明🙂 " + body
    report = check(adapt({"answer": answer, "sources": []}))
    assert fails([report], "error") and not fails([report], "none")
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
    assert sentence["citation_indices"] == [9]
    finding, = report["findings"]
    assert finding["code"] == "CC001" and finding["severity"] == "error"
    start = answer.index("[9]")
    assert finding["evidence"] == {"marker": "[9]", "start": start, "end": start + 3, "kind": "dangling"}


@pytest.mark.parametrize("tag", [
    "</em>", "</h3>", "</élément>", "</标签>", "<em>", "<h3>", "<img/>",
    '<span title=">` [8]">', "<span title='>` [8]'>",
    '</em title=">` [8]">', "</em title='>` [8]'>",
    "<! [8]>", "<!-- ` [8] -->", '<? note=">` [8]"?>',
    # Supported word prefixes and later tag-body punctuation remain permissive.
    "</² [8]>", "</em+ [8]>",
])
def test_valid_tags_reserved_openers_and_other_bodies_keep_existing_mask(tag):
    answer = "说明🙂 " + tag + "[9]`"
    assert mask_markdown(answer) == "说明🙂 " + " " * len(tag) + "[9]`"


@pytest.mark.parametrize("body,expected", [
    ("< limit [9] >", "< limit [9] >"),
    ("<３ [9] >", "<３ [9] >"),
    ("<+ [9] >", "<+ [9] >"),
    ("`</ [9] >`", " " * len("`</ [9] >`")),
    ("``</ ` [9] >``", " " * len("``</ ` [9] >``")),
    ('`</ title="`> [9]', " " * len('`</ title="`') + "> [9]"),
    ("</ `code [8]` [9] >", "</ " + " " * len("`code [8]`") + " [9] >"),
    (r"\`</ [9] >`", r"\`</ [9] >`"),
    (r"\\`</ [9] >`", "\\\\" + " " * len("`</ [9] >`")),
    ('</ <em title=">` [8]">[9]`', "</ " + " " * len('<em title=">` [8]">') + "[9]`"),
    ("[archive](</ part(one)`[8] >)[9]`", " archive" + " " * len("](</ part(one)`[8] >)") + "[9]`"),
    ("</ [archive](part(one)`[8])[9]` >", "</  archive" + " " * len("](part(one)`[8])") + "[9]` >"),
    ("<https://example.invalid/[8]> [9]", " " * len("<https://example.invalid/[8]>") + " [9]"),
    ("<http://example.invalid/[8]> [9]", " " * len("<http://example.invalid/[8]>") + " [9]"),
    ("</ https://example.invalid/[8] [9] >", "</ " + " " * len("https://example.invalid/[8]") + " [9] >"),
])
def test_closing_whitespace_prose_keeps_code_link_tag_and_url_precedence(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


def test_closing_whitespace_keeps_sparse_source_matching_and_all_gates():
    answer = 'Archive </ "stable signal" [4] > manual'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 4, "text": "Archive stable signal manual"}]}))
    assert report["findings"] == []
    assert all(not fails([report], level) for level in ["info", "hint", "warning", "error", "none"])
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [4] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_exact_closing_whitespace_cli_contract():
    answer = "说明🙂 </ [9] >"
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


def test_closing_whitespace_cli_fixture():
    fixture = ROOT / "examples/cite-check/html_closing_whitespace.jsonl"
    answers = [json.loads(line)["answer"] for line in fixture.read_text().splitlines()]
    result = cli("check", fixture, "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["record_count"] == len(answers) and output["gate_failed"]
    for answer, record in zip(answers[:4], output["records"][:4], strict=True):
        sentence, = record["sentences"]
        assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
        assert sentence["citation_indices"] == [9]
        finding, = record["findings"]
        assert finding["code"] == "CC001"
        start = answer.index("[9]")
        assert finding["evidence"] == {"marker": "[9]", "start": start, "end": start + 3, "kind": "dangling"}
    for record in output["records"][4:]:
        assert record["findings"] == []
