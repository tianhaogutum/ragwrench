"""Single-line tags hide quoted > and backticks; open code keeps precedence."""
import json

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails
from ragwrench.citecheck.markers import parse_markers
from ragwrench.citecheck.prose import mask_markdown
from .test_cli_io import ROOT, cli


@pytest.mark.parametrize("tag", [
    '<span title=">`">',
    "<span title='>`'>",
    '<span title="a > ` [8] 999kg" data-note=\'another > "\'>',
    "<span title='a > ` [8] 999kg' data-note=\"another > '\">",
    '<img title=">``"/>',
    '<span title="\\>`">',  # Backslashes do not escape HTML quotes.
])
def test_quoted_tag_backticks_cannot_open_code(tag):
    prefix, tail = "说明🙂 ", "literal [9]`"
    answer = prefix + tag + tail
    assert mask_markdown(answer) == prefix + " " * len(tag) + tail
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
    ('`<span title="> [9]">`', ' ' * len('`<span title="> [9]">`')),
    ('``<span title=">` [9]">``', ' ' * len('``<span title=">` [9]">``')),
    # Code can close inside tag-looking text, even before the quoted >.
    ('`<span title="`>">literal [9]', ' ' * len('`<span title="`') + '>">literal [9]'),
    ('<span title=">`">`code [8]` [9]', ' ' * len('<span title=">`">`code [8]`') + ' [9]'),
    ('[archive <em title=">`">](url`)[9]`', ' archive ' + ' ' * len('<em title=">`">](url`)') + '[9]`'),
    ('https://example.invalid/`<em title=">`">[9]`', ' ' * len('https://example.invalid/`<em title=">`">') + '[9]`'),
    (r'\`<span title=">`">literal [9]`', '\\`' + ' ' * len('<span title=">`">') + 'literal [9]`'),
    (r'\\`<span title="> [9]">`', '\\\\' + ' ' * len('`<span title="> [9]">`')),
])
def test_quoted_tags_keep_code_link_url_and_escape_precedence(body, expected):
    assert mask_markdown("说明🙂 " + body) == "说明🙂 " + expected


@pytest.mark.parametrize("body", [
    '<span title="> [9]',
    "<span title='> [9]",
    '<span title=">\'> [9]',
    '<span title="no close quote> [9]',
    '<span title=">\nvalue"> [9]',
    "<span title='>\nvalue'> [9]",
    '<span title="balanced" [9]',  # Still needs an unquoted closing >.
])
def test_unclosed_quotes_and_multiline_tags_remain_prose(body):
    answer = "说明🙂 " + body
    assert mask_markdown(answer) == answer
    marker, = parse_markers(mask_markdown(answer))
    assert marker.indices == (9,)


def test_quoted_tag_contents_do_not_contaminate_lexical_checks():
    answer = 'Archive manual <span title=">` [8] 999kg absent" data-note=\'> "fake quote"\'>works [1]`.'
    report = check(adapt({"answer": answer, "sources": [
        {"idx": 1, "text": "Archive manual works."}]}))
    assert report["findings"] == []
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [1] and sentence["cited_coverage"] == 1
    assert sentence["text"] == answer


def test_quoted_tag_cli_default_gate_and_original_offsets():
    result = cli("check", ROOT / "examples/cite-check/html_quotes.jsonl", "--jsonl", "--report", "json")
    assert result.returncode == 1 and not result.stderr
    output = json.loads(result.stdout)
    assert output["fail_on"] == "error" and output["gate_failed"]
    double, single, code = output["records"]
    for quote, record in [("\"", double), ("'", single)]:
        sentence, = record["sentences"]
        answer = f"说明🙂 <span title={quote}>`{quote}>literal [9]`"
        assert (sentence["start"], sentence["end"], sentence["text"]) == (0, 33, answer)
        assert sentence["citation_indices"] == [9]
        finding, = record["findings"]
        assert finding["code"] == "CC001"
        assert finding["evidence"] == {"marker": "[9]", "start": 29, "end": 32, "kind": "dangling"}
    assert code["findings"] == [] and code["sentences"][0]["citation_indices"] == []
