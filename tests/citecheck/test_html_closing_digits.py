"""Immediate Unicode decimal digits after </ are prose, preserving other masks."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("digit", [*"0123456789", "٣", "３", "𝟛", "९"])
def test_decimal_after_closing_slash_keeps_citation_and_original_offsets(digit):
    answer = f"说明🙂 </{digit} [9] >"
    assert mask_markdown(answer) == answer
    report = check(adapt({"answer": answer, "sources": []}))
    assert fails([report], "error") and not fails([report], "none")
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, 13, answer)
    assert sentence["citation_indices"] == [9]
    finding, = report["findings"]
    assert finding["code"] == "CC001" and finding["severity"] == "error"
    assert finding["evidence"] == {"marker": "[9]", "start": 8, "end": 11, "kind": "dangling"}


@pytest.mark.parametrize("body", [
    "</12 [9] >", "</3thing [9]/>", "</٣em [9]>",
    '</3 title="> [9]">', "</３ title='> [9]'>",
])
def test_only_immediate_digit_is_rejected_regardless_of_later_body(body):
    assert mask_markdown(body) == body


@pytest.mark.parametrize("tag", [
    "</em>", "</h3>", "</élément>", "</标签>", "<em>", "<h3>", "<img/>",
    '<span title=">` [8]">', "<span title='>` [8]'>",
    '</em title=">` [8]">', "</em title='>` [8]'>",
    "<!3 [8]>", "<!-- ` [8] -->", '<?3 note=">` [8]"?>',
    # These word prefixes and tag bodies remain permissive; decimal != all numerals.
    "</² [8]>", "</Ⅲ [8]>", "</em3 [8]>",
])
def test_supported_tags_reserved_prefixes_and_nondecimal_controls_keep_mask(tag):
    answer = "说明🙂 " + tag + "[9]`"
    assert mask_markdown(answer) == "说明🙂 " + " " * len(tag) + "[9]`"
    report = check(adapt({"answer": answer, "sources": []}))
    finding, = report["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"] == {"marker": "[9]", "start": 4 + len(tag),
                                   "end": 7 + len(tag), "kind": "dangling"}


@pytest.mark.parametrize("body,expected", [
    ("</ [9] >", "</ [9] >"),
    ("</\u2003[9] >", "</\u2003[9] >"),
    ("<３ [9] >", "<３ [9] >"),
    ("<+ [9] >", "<+ [9] >"),
    ("`</3 [9] >`", " " * len("`</3 [9] >`")),
    ("``</3 ` [9] >``", " " * len("``</3 ` [9] >``")),
    ('`</3 title="`> [9]', " " * len('`</3 title="`') + "> [9]"),
    ("</3 `code [8]` [9] >", "</3 " + " " * len("`code [8]`") + " [9] >"),
    (r"\`</3 [9] >`", r"\`</3 [9] >`"),
    (r"\\`</3 [9] >`", "\\\\" + " " * len("`</3 [9] >`")),
    ('</3 <em title=">` [8]">[9]`', "</3 " + " " * len('<em title=">` [8]">') + "[9]`"),
    ("[archive](</3 part(one)`[8] >)[9]`", " archive" + " " * len("](</3 part(one)`[8] >)") + "[9]`"),
    ("</3 [archive](part(one)`[8])[9]` >", "</3  archive" + " " * len("](part(one)`[8])") + "[9]` >"),
    ("<https://example.invalid/[8]> [9]", " " * len("<https://example.invalid/[8]>") + " [9]"),
    ("<http://example.invalid/[8]> [9]", " " * len("<http://example.invalid/[8]>") + " [9]"),
    ("</3 https://example.invalid/[8] [9] >", "</3 " + " " * len("https://example.invalid/[8]") + " [9] >"),
])
def test_decimal_closing_prose_keeps_code_link_tag_and_url_precedence(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


def test_decimal_closing_prose_keeps_sparse_source_matching_and_all_gates():
    answer = 'Archive </3 "stable signal" [4] > manual'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 4, "text": "Archive 3 stable signal manual"}]}))
    assert report["findings"] == []
    assert all(not fails([report], level) for level in ["info", "hint", "warning", "error", "none"])
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [4] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_exact_decimal_closing_cli_contract():
    answer = "说明🙂 </3 [9] >"
    result = cli("check", "-", "--report", "json",
                 stdin=json.dumps({"answer": answer, "sources": []}))
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
    record, = output["records"]
    sentence, = record["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, 13, answer)
    assert sentence["citation_indices"] == [9]
    finding, = record["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"] == {"marker": "[9]", "start": 8, "end": 11, "kind": "dangling"}


def test_decimal_closing_cli_fixture():
    fixture = ROOT / "examples/cite-check/html_closing_digits.jsonl"
    answers = [json.loads(line)["answer"] for line in fixture.read_text().splitlines()]
    result = cli("check", fixture, "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["record_count"] == len(answers) and output["gate_failed"]
    for answer, record in zip(answers[:5], output["records"][:5], strict=True):
        sentence, = record["sentences"]
        assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
        assert sentence["citation_indices"] == [9]
        finding, = record["findings"]
        assert finding["code"] == "CC001"
        start = answer.index("[9]")
        assert finding["evidence"] == {"marker": "[9]", "start": start, "end": start + 3, "kind": "dangling"}
    for record in output["records"][5:]:
        assert record["findings"] == []
