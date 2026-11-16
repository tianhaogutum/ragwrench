"""Punctuation-led angles are prose except the supported / ! ? openers."""
import json
import string

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("opener", [
    "+", "-", "=", "_", "@", "%", "&", "*", "|", "~", "^", "\\", "，", "＋", "🙂",
])
def test_punctuation_led_angles_keep_citations_and_original_offsets(opener):
    answer = f"说明🙂 <{opener} [9] >"
    assert mask_markdown(answer) == answer
    report = check(adapt({"answer": answer, "sources": []}))
    assert fails([report], "error")
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, 12, answer)
    assert sentence["citation_indices"] == [9]
    finding, = report["findings"]
    assert finding["code"] == "CC001" and finding["severity"] == "error"
    assert finding["evidence"] == {"marker": "[9]", "start": 7, "end": 10, "kind": "dangling"}


@pytest.mark.parametrize("opener", sorted(set(string.punctuation) - set("/!?<>`")))
def test_other_ascii_punctuation_cannot_open_tag_mask(opener):
    # No paired code, links or URLs in this input; even balanced quote-led
    # angles must remain prose. Empty/angle/backtick boundaries have controls below.
    answer = f"说明🙂 <{opener} [9] {opener}>"
    assert mask_markdown(answer) == answer


@pytest.mark.parametrize("tag", [
    "<em>", "</em>", "<img/>", "<h3>", "<élément>", "<标签>",
    '<span title=">` [8]">', "<span title='>` [8]'>",
    "<!DOCTYPE html [8]>", "<!-- ` [8] -->", "<![CDATA[ [8] ]]>",
    '<?xml note=">` [8]"?>', "<?instruction [8]?>",
])
def test_supported_tags_and_punctuation_controls_stay_opaque(tag):
    answer = "说明🙂 " + tag + "[9]`"
    assert mask_markdown(answer) == "说明🙂 " + " " * len(tag) + "[9]`"
    report = check(adapt({"answer": answer, "sources": []}))
    finding, = report["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"] == {"marker": "[9]", "start": 4 + len(tag),
                                   "end": 7 + len(tag), "kind": "dangling"}


@pytest.mark.parametrize("body,expected", [
    ("< limit [9] >", "< limit [9] >"),
    ("<３ [9] >", "<３ [9] >"),
    ("<>[9]", "<>[9]"),
    ("<<em>[9]", "<" + " " * len("<em>") + "[9]"),
    ('<+ <em title=">` [8]">[9]`', "<+ " + " " * len('<em title=">` [8]">') + "[9]`"),
    ("`<+ [9] >`", " " * len("`<+ [9] >`")),
    ("``<+ ` [9] >``", " " * len("``<+ ` [9] >``")),
    ('`<em title="`> [9]', " " * len('`<em title="`') + "> [9]"),
    ("<+ `code [8]` [9] >", "<+ " + " " * len("`code [8]`") + " [9] >"),
    ("<`code [8]` [9]>", "<" + " " * len("`code [8]`") + " [9]>"),
    (r"\`<+ [9] >`", r"\`<+ [9] >`"),
    (r"\\`<+ [9] >`", "\\\\" + " " * len("`<+ [9] >`")),
    ("[archive](<+ part(one)`[8] >)[9]`", " archive" + " " * len("](<+ part(one)`[8] >)") + "[9]`"),
    ("<+ [archive](part(one)`[8])[9]` >", "<+  archive" + " " * len("](part(one)`[8])") + "[9]` >"),
    ("<https://example.invalid/[8]> [9]", " " * len("<https://example.invalid/[8]>") + " [9]"),
    ("<http://example.invalid/[8]> [9]", " " * len("<http://example.invalid/[8]>") + " [9]"),
    ("<+ https://example.invalid/[8] [9] >", "<+ " + " " * len("https://example.invalid/[8]") + " [9] >"),
])
def test_punctuation_prose_keeps_other_inline_boundaries(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


def test_punctuation_prose_preserves_source_matching_and_gates():
    answer = 'Archive <+ "stable signal" [4] > manual'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 4, "text": "Archive stable signal manual"}]}))
    assert report["findings"] == [] and not fails([report], "info")
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [4] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_exact_punctuation_cli_default_gate_and_original_offsets():
    answer = "说明🙂 <+ [9] >"
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


def test_punctuation_cli_fixture():
    fixture = ROOT / "examples/cite-check/html_punctuation.jsonl"
    answers = [json.loads(line)["answer"] for line in fixture.read_text().splitlines()]
    result = cli("check", fixture, "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
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
