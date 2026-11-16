"""Complete same-line comments are opaque; unsupported candidates use tag fallback."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("comment", [
    "<!-- > [9] -->", "<!-- [9] -->", "<!---->", "<!--> [9] -->",
    '<!-- > "unclosed [9] -->', "<!-- > 'unclosed [9] -->",
    '<!-- > "quoted -->', "<!-- > ` [9] -->", "<!-- > `` [9] -->",
    "<!-- > [9] -- > more -->", "<!-- > <!-- [9] -->",
    "<!-- > [bad] 999kg \"absent quote\" -->",
])
def test_complete_comments_hide_the_whole_body(comment):
    answer = "说明🙂 " + comment
    assert mask_markdown(answer) == "说明🙂 " + " " * len(comment)
    report = check(adapt({"answer": answer, "sources": []}))
    assert report["findings"] == []
    assert all(not fails([report], level) for level in ["info", "hint", "warning", "error", "none"])
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
    assert sentence["citation_indices"] == []


@pytest.mark.parametrize("comment", [
    "<!---->", "<!-- > [8] -->", '<!-- > "` [8] -->', "<!-- > `` [8] -->",
    "<!-- <!-- > [8] -->", "<!-- > [8] --> <!-- > [7] -->",
])
def test_comment_backticks_do_not_hide_outside_citations(comment):
    answer = "说明🙂 " + comment + "[9]`"
    assert mask_markdown(answer) == "说明🙂 " + " " * len(comment) + "[9]`"
    report = check(adapt({"answer": answer, "sources": []}))
    sentence, = report["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
    assert sentence["citation_indices"] == [9]
    finding, = report["findings"]
    assert finding["code"] == "CC001" and fails([report], "error")
    assert finding["evidence"] == {"marker": "[9]", "start": 4 + len(comment),
                                   "end": 7 + len(comment), "kind": "dangling"}


@pytest.mark.parametrize("body,expected", [
    # Empty is complete. No nesting or HTML-body validation; the first --> wins.
    ("<!---->[9]", " " * len("<!---->") + "[9]"),
    ('<!-- " -->[9]" -->', " " * len('<!-- " -->') + '[9]" -->'),
    ("<!-- <!-- -->[9] -->", " " * len("<!-- <!-- -->") + "[9] -->"),
    # Incomplete candidates retain the preexisting generic !-tag fallback.
    ("<!-- [9]", "<!-- [9]"),
    ("<!-- [8] > [9]", " " * len("<!-- [8] >") + " [9]"),
    ("<!--> [9]", " " * len("<!-->") + " [9]"),
    ('<!-- " > [9]', '<!-- " > [9]'),
    # No special comment token crosses CR or LF. Generic tag fallback is unchanged.
    ("<!-- >\n[9] -->", " " * len("<!-- >") + "\n[9] -->"),
    ("<!--\n[9] -->", "<!--\n[9] -->"),
    ("<!-- >\r\n[9] -->", " " * len("<!-- >") + "\r\n[9] -->"),
    ("<!-- >\r[9] -->", " " * len("<!-- >") + "\r[9] -->"),
    # Legacy generic tags exclude LF only; this remains a generic-tag mask.
    ("<!--\r[9] -->", " " * len("<!--\r[9] -->")),
    ("<!-- >\n说明🙂 <!-- > [8] -->[9]",
     " " * len("<!-- >") + "\n说明🙂 " + " " * len("<!-- > [8] -->") + "[9]"),
    ("<!-- > [8] -->`code -->`<!-- > [7] -->[9]",
     " " * len("<!-- > [8] -->`code -->`<!-- > [7] -->") + "[9]"),
    ("<!-- >" * 1000 + " [9]", " " * (len("<!-- >") * 1000) + " [9]"),
    ("<! > [9] -->", " " * len("<! >") + " [9] -->"),
    ("<? > [9]", " " * len("<? >") + " [9]"),
    ("<span note='> [8]'>[9]", " " * len("<span note='> [8]'>") + "[9]"),
])
def test_terminator_and_fallback_contracts(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


@pytest.mark.parametrize("body,expected", [
    ("`<!-- > [8] -->`[9]", " " * len("`<!-- > [8] -->`") + "[9]"),
    ("``<!-- > ` [8] -->``[9]", " " * len("``<!-- > ` [8] -->``") + "[9]"),
    # Open code can close inside comment-looking text, before its terminator.
    ("`<!-- > ` [9] -->", " " * len("`<!-- > `") + " [9] -->"),
    (r"\`<!-- > ` [8] -->[9]`", "\\`" + " " * len("<!-- > ` [8] -->") + "[9]`"),
    (r"\\`<!-- > [8] -->`[9]", "\\\\" + " " * len("`<!-- > [8] -->`") + "[9]"),
    ("<!-- > ` [8] -->`[9]`", " " * len("<!-- > ` [8] -->`[9]`")),
    ("[archive <!-- > ` [8] -->](part(one))[9]`",
     " archive " + " " * len("<!-- > ` [8] -->](part(one))") + "[9]`"),
    ("[archive](part(<!-- > ` [8] -->))[9]`",
     " archive" + " " * len("](part(<!-- > ` [8] -->))") + "[9]`"),
    ("https://example.invalid/`<!-- > ` [8] -->[9]`",
     " " * len("https://example.invalid/`<!-- > ` [8] -->") + "[9]`"),
    ("<https://example.invalid/[8]> <!-- > ` [7] -->[9]`",
     " " * len("<https://example.invalid/[8]> <!-- > ` [7] -->") + "[9]`"),
])
def test_comments_keep_source_order_code_link_url_precedence(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


def test_comment_contents_do_not_contaminate_source_matching_or_gates():
    answer = 'Archive "stable signal" <!-- > [9] 999kg "absent quotation" ` --> [4] manual'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 4, "text": "Archive stable signal manual"}]}))
    assert report["findings"] == []
    assert all(not fails([report], level) for level in ["info", "hint", "warning", "error", "none"])
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [4] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_exact_comment_cli_contract():
    answer = "说明🙂 <!-- > [9] -->"
    result = cli("check", "-", "--report", "json", stdin=json.dumps({"answer": answer, "sources": []}))
    assert result.returncode == 0 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and not output["gate_failed"]
    record, = output["records"]
    assert record["findings"] == []
    sentence, = record["sentences"]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, 18, answer)
    assert sentence["citation_indices"] == []


def test_comment_cli_fixture():
    fixture = ROOT / "examples/cite-check/html_comments.jsonl"
    answers = [json.loads(line)["answer"] for line in fixture.read_text().splitlines()]
    result = cli("check", fixture, "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["record_count"] == len(answers) and output["gate_failed"]
    for i, (answer, record) in enumerate(zip(answers, output["records"], strict=True)):
        if i in [1, 2, 3, 4, 5]:
            finding, = record["findings"]
            assert finding["code"] == "CC001"
            start = answer.index("[9]")
            assert finding["evidence"] == {"marker": "[9]", "start": start, "end": start + 3, "kind": "dangling"}
        else:
            assert record["findings"] == []
