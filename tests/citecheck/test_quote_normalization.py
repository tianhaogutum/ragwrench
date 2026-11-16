"""Local synthetic quote fixtures: exact, normalization-only, and unsupported."""
import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails, markdown_report
from ragwrench.citecheck.model import Config, InputError


def report(answer, sources, strict=False):
    return check(adapt({"answer": answer, "sources": [
        {"idx": i, "text": text} for i, text in enumerate(sources, 1)
    ]}), Config(strict_quotes=strict))


@pytest.mark.parametrize("quote,source", [
    ("信号：稳定", "信号:稳定"),
    ("信号:稳定", "信号：稳定"),
    ("信号。稳定、继续", "信号.稳定,继续"),
    ("ＡＢＣ（稳定）！", "ABC(稳定)!"),
    ("Signal   stable", "signal stable"),
    ("café stable", "cafe\u0301 stable"),
    ("他说‘稳定’", "他说'稳定'"),
])
@pytest.mark.parametrize("strict", [False, True])
def test_quote_normalization_preserves_original_and_distinguishes_strict(quote, source, strict):
    answer = f"# Heading🙂\n  手册说「{quote}」【1】。"
    result = report(answer, [f"手册说{source}。"], strict)
    finding, = result["findings"]
    assert finding["code"] == "CC008"
    assert finding["severity"] == ("error" if strict else "info")
    assert finding["source_indices"] == (1,)
    evidence = finding["evidence"]
    assert evidence["quote"] == quote
    assert answer[evidence["start"]:evidence["end"]] == quote
    assert evidence["match_kind"] == "normalized"
    assert evidence["normalized_quote"]
    assert result["config"]["strict_quotes"] is strict
    assert fails([result], "error") is strict
    assert fails([result], "info")
    assert not fails([result], "none")
    assert "CC008" in markdown_report([result])


@pytest.mark.parametrize("opening,closing", [('"', '"'), ('“', '”'), ('「', '」'),
                                                    ('『', '』'), ('＂', '＂'), ('｢', '｣')])
def test_quote_delimiters_keep_sentence_and_postfix_citation(opening, closing):
    quote = "Signal! Stable."
    answer = f"The manual says {opening}{quote}{closing} [1]."
    result = report(answer, [f"The manual says {quote}"], True)
    assert result["findings"] == []
    assert result["analyzed_sentence_count"] == 1
    assert result["sentences"][0]["text"] == answer
    changed = report(answer, ["The manual says signal! stable."], True)
    assert [f["code"] for f in changed["findings"]] == ["CC008"]


@pytest.mark.parametrize("quote,source", [
    ("信号：中断", "信号:稳定"),
    ("信号：稳定", "信号稳定"),  # punctuation is mapped, never deleted
    ("Signal — stable", "Signal stable"),  # no arbitrary punctuation folding
])
@pytest.mark.parametrize("strict", [False, True])
def test_unsupported_quote_stays_cc006(quote, source, strict):
    result = report(f"The manual says 「{quote}」 [1].", [f"The manual says {source}."], strict)
    finding, = [f for f in result["findings"] if f["code"] in {"CC006", "CC008"}]
    assert finding["code"] == "CC006" and finding["severity"] == "error"
    assert finding["evidence"]["quote"] == quote
    assert finding["evidence"]["match_kind"] == "unsupported"
    assert fails([result], "error")


def test_exact_quote_in_any_cited_source_wins_and_uncited_does_not():
    answer = 'The manual says "Signal stable" [1,2].'
    assert report(answer, ["The manual says signal stable.", "The manual says Signal stable."], True)["findings"] == []
    result = report(answer.replace("[1,2]", "[1]"), ["The manual says signal stable.", "The manual says Signal stable."], True)
    quote_finding, = [f for f in result["findings"] if f["code"] == "CC008"]
    assert quote_finding["source_indices"] == (1,)
    unsupported = report(answer.replace("[1,2]", "[1]"), ["The manual says signal steady.", "The manual says Signal stable."], True)
    assert "CC006" in {f["code"] for f in unsupported["findings"]}
    assert "CC008" not in {f["code"] for f in unsupported["findings"]}


def test_normalization_does_not_hide_quantity_error_or_masked_prose():
    result = report('The manual says "１２ months" [1].', ["The manual says 12 years."])
    assert {f["code"] for f in result["findings"]} >= {"CC004", "CC006"}
    result = report('The manual says `"ＡＢＣ：稳定"` [1].', ["The manual says ABC:稳定."], True)
    assert result["findings"] == []


def test_strict_report_distinguishes_normalization_and_unsupported_together():
    answer = 'The manual says "Signal stable" and "signal steady" [1].'
    result = report(answer, ["The manual says signal stable."], True)
    findings = [f for f in result["findings"] if f["code"] in {"CC006", "CC008"}]
    assert [(f["code"], f["severity"], f["evidence"]["quote"]) for f in findings] == [
        ("CC008", "error", "Signal stable"), ("CC006", "error", "signal steady")]


def test_normalization_cannot_join_separate_sources_or_use_empty_key():
    result = report('The manual says "Signal stable" [1,2].', ["The manual says Signal", "The manual says stable"], True)
    assert "CC006" in {f["code"] for f in result["findings"]}
    result = report('The manual says "\u2003\u2003" [1].', ["The manual says stable."])
    assert "CC006" in {f["code"] for f in result["findings"]}


@pytest.mark.parametrize("value", [1, "true", None])
def test_strict_quotes_api_requires_boolean(value):
    with pytest.raises(InputError, match="strict_quotes"):
        Config(strict_quotes=value)
