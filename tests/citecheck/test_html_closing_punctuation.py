"""Punctuation/symbol closing prefixes are prose; supported names stay opaque."""
import json
import string

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("prefix", ["+", "-", "=", "_", "@", "%", "&", "*", "|", "~", "^", "\\", "，", "＋", "🙂"])
def test_punctuation_after_closing_slash_keeps_citation_and_original_offsets(prefix):
    answer = f"说明🙂 </{prefix} [9] >"
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
    "</+ [9] >", "</+thing [9]/>", "</_em [9]>",
    '</+ title="> [9]">', "</🙂 title='> [9]'>",
])
def test_only_immediate_prefix_is_rejected_regardless_of_later_body(body):
    assert mask_markdown(body) == body


@pytest.mark.parametrize("prefix", sorted(set(string.punctuation) - set("<>`")))
def test_all_ascii_punctuation_after_slash_is_prose(prefix):
    # Quotes are balanced, but cannot give a punctuation-led closing name a mask.
    answer = f"说明🙂 </{prefix} [9] {prefix}>"
    assert mask_markdown(answer) == answer


@pytest.mark.parametrize("body", ["</>", "</ [9]>", "</3 [9]>", "</٣ [9]>", "</\u0301 [9]>"])
def test_unsupported_closing_prefix_controls_stay_prose(body):
    assert mask_markdown(body) == body


@pytest.mark.parametrize("tag", [
    "</em>", "</h3>", "</élément>", "</标签>", "<em>", "<h3>", "<img/>",
    '<span title=">` [8]">', "<span title='>` [8]'>",
    '</em title=">` [8]">', "</em title='>` [8]'>",
    "<!3 [8]>", "<!-- ` [8] -->", '<?3 note=">` [8]"?>',
    # These prefixes and tag bodies remain permissive; nondecimal word numerals remain supported.
    "</² [8]>", "</Ⅲ [8]>", "</em3 [8]>", "</em+ [8]>",
])
def test_supported_tags_reserved_prefixes_and_nonpunctuation_controls_keep_mask(tag):
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
    ("`</+ [9] >`", " " * len("`</+ [9] >`")),
    ("``</+ ` [9] >``", " " * len("``</+ ` [9] >``")),
    ('`</+ title="`> [9]', " " * len('`</+ title="`') + "> [9]"),
    ("</+ `code [8]` [9] >", "</+ " + " " * len("`code [8]`") + " [9] >"),
    (r"\`</+ [9] >`", r"\`</+ [9] >`"),
    (r"\\`</+ [9] >`", "\\\\" + " " * len("`</+ [9] >`")),
    ('</+ <em title=">` [8]">[9]`', "</+ " + " " * len('<em title=">` [8]">') + "[9]`"),
    ("[archive](</+ part(one)`[8] >)[9]`", " archive" + " " * len("](</+ part(one)`[8] >)") + "[9]`"),
    ("</+ [archive](part(one)`[8])[9]` >", "</+  archive" + " " * len("](part(one)`[8])") + "[9]` >"),
    ("<https://example.invalid/[8]> [9]", " " * len("<https://example.invalid/[8]>") + " [9]"),
    ("<http://example.invalid/[8]> [9]", " " * len("<http://example.invalid/[8]>") + " [9]"),
    ("</+ https://example.invalid/[8] [9] >", "</+ " + " " * len("https://example.invalid/[8]") + " [9] >"),
])
def test_punctuation_closing_prose_keeps_code_link_tag_and_url_precedence(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


def test_punctuation_closing_prose_keeps_sparse_source_matching_and_all_gates():
    answer = 'Archive </+ "stable signal" [4] > manual'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 4, "text": "Archive stable signal manual"}]}))
    assert report["findings"] == []
    assert all(not fails([report], level) for level in ["info", "hint", "warning", "error", "none"])
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [4] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_exact_punctuation_closing_cli_contract():
    answer = "说明🙂 </+ [9] >"
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


def test_punctuation_closing_cli_fixture():
    fixture = ROOT / "examples/cite-check/html_closing_punctuation.jsonl"
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
