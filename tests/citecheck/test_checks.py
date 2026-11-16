import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check, fails, markdown_report
from ragwrench.citecheck.lexical import coverage, quantities, tokens
from ragwrench.citecheck.model import Config, InputError


def run(answer, sources=(), config=None):
    return check(adapt({"answer": answer, "sources": [{"idx": i, "text": text} for i, text in enumerate(sources, 1)]}), config)


def codes(report):
    return {f["code"] for f in report["findings"]}


@pytest.mark.parametrize("answer,sources,expected", [
    ("The rover uses solar panels [9].", [], {"CC001"}),
    ("The rover uses solar panels.", [], {"CC002"}),
    ("The rover uses solar panels [1].", ["The ocean contains salt water."], {"CC003"}),
    ("The warranty lasts 24 months [1].", ["The warranty lasts 12 months."], {"CC004"}),
    ("The rover uses solar panels [1].", ["The ocean contains salt water.", "The rover uses solar panels."], {"CC003", "CC005", "CC007"}),
    ('The manual says "signal stable" [1].', ["The manual says signal steady."], {"CC006"}),
    ("The rover uses solar panels [1].", ["The rover uses solar panels.", "An unrelated source."], {"CC007"}),
    ("The rover uses solar panels [1].", ["The rover uses solar panels."], set()),
    ("The warranty lasts 12 months [1].", ["The warranty lasts 12 months."], set()),
    ('The manual says “signal stable” [1].', ["The manual says signal stable."], set()),
    ("The rover uses solar panels [abc].", [], {"CC001", "CC002"}),
    ("What powers the rover?", [], set()),
    ("保修期为12个月。", [], {"CC002"}),
])
def test_each_check_positive_and_negative(answer, sources, expected):
    assert codes(run(answer, sources)) == expected


def test_empty_sources_and_missing_citations():
    result = run("The warranty lasts 12 months. Returns are free.")
    assert [f["code"] for f in result["findings"]] == ["CC002", "CC002"]
    assert result["source_count"] == 0


@pytest.mark.parametrize("claim,source", [
    ("1,000 records", "1000 records"), ("25%", "25 percent"), ("2026/1/2", "2026-01-02"),
    ("2026年1月2日", "2026-01-02"), ("10 kilograms", "10 kg"), ("１２个月", "12个月"),
    ("$100", "100 USD"), ("-0.00 kg", "0 kg"), ("+1.50 kg", "1.5 kg"),
])
def test_number_date_unit_normalization(claim, source):
    a, b = quantities(claim), quantities(source)
    assert [q.key for q in a] == [q.key for q in b]
    assert len(a) == 1
    result = run(f"The value is {claim} [1].", [f"The value is {source}."])
    assert "CC004" not in codes(result)


@pytest.mark.parametrize("claim,source", [("10%", "10 records"), ("12 years", "12 months"), ("10 g", "10 kg"), ("$10", "10 EUR"), ("-10", "10")])
def test_quantity_mismatches(claim, source):
    report = run(f"The value is {claim} [1].", [f"The value is {source}."])
    assert "CC004" in codes(report)
    evidence = next(f["evidence"] for f in report["findings"] if f["code"] == "CC004")
    assert evidence["claim"]["value"] and evidence["found"]


def test_thousands_normalized_for_lexical_too():
    assert tokens("stores 1,000 records") == tokens("stores 1000 records")
    assert coverage(tokens("stores 1000 records"), tokens("stores 1,000 records")) == 1


def test_quotes_multi_sentence_and_whitespace():
    report = run('The manual says "Signal stable. Start now!" [1].', ["The manual says Signal stable. Start now!"])
    assert "CC006" not in codes(report)
    assert "CC002" not in codes(report)
    assert "CC006" not in codes(run('The manual says "signal   stable" [1].', ["The manual says signal stable."]))
    assert "CC006" in codes(run('The manual says 「signal stable」 [1].', ["The manual says signal steady."]))


def test_offsets_and_number_evidence():
    answer = "The warranty lasts 24 months. [1] Returns are free [3]."
    report = run(answer, ["The warranty lasts 12 months.", "Returns are free."])
    mismatch = next(f for f in report["findings"] if f["code"] == "CC004")
    assert mismatch["sentence"] == 1
    assert mismatch["evidence"]["claim"]["value"] == "24"
    dangling = next(f for f in report["findings"] if f["code"] == "CC001")
    assert dangling["sentence"] == 2
    assert answer[dangling["evidence"]["start"]:dangling["evidence"]["end"]] == "[3]"
    assert next(f for f in report["findings"] if f["code"] == "CC005")["source_indices"] == (2,)


def test_thresholds_actually_change_decisions():
    answer, sources = "The rover uses solar panels [1].", ["The rover uses batteries."]
    assert "CC003" in codes(run(answer, sources, Config(support_threshold=0.9)))
    assert "CC003" not in codes(run(answer, sources, Config(support_threshold=0.2)))
    assert not fails([run(answer, sources)], "error")
    assert fails([run(answer, sources, Config(support_threshold=0.9))], "warning")


@pytest.mark.parametrize("kwargs", [{"support_threshold": float("nan")}, {"best_source_threshold": 1.1}, {"misattribution_margin": -1}, {"min_claim_tokens": 0}])
def test_bad_thresholds(kwargs):
    with pytest.raises(InputError):
        Config(**kwargs)


def test_code_and_heading_only_are_explicitly_not_analyzed():
    report = run("# Title [99]\n```\necho '[88]'\n```\n")
    assert report["analyzed_sentence_count"] == 0
    assert "No prose" in " ".join(report["notes"])


@pytest.mark.parametrize("code", [
    "``use `x` [9]``", '``use `x` [9] 999 kg "invented quote"``',
    "``use `x`\n[9]``", "``use `x`\r\n[9]``",
])
def test_code_contents_do_not_contaminate_citations_or_claim_checks(code):
    answer = f"Supported syntax is {code} [1]."
    report = run(answer, ["Supported syntax is documented."])
    assert report["findings"] == []
    sentence, = report["sentences"]
    assert sentence["citation_indices"] == [1]
    assert (sentence["start"], sentence["end"], sentence["text"]) == (0, len(answer), answer)
    assert sentence["cited_coverage"] == 1


def test_real_citation_error_after_code_keeps_unicode_offsets():
    answer = "说明🙂 ``use `x` [8]`` Real citation [9]."
    report = run(answer)
    finding, = report["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"] == {"marker": "[9]", "start": answer.index("[9]"),
                                   "end": answer.index("[9]") + 3, "kind": "dangling"}


def test_line_leading_inline_code_cannot_hide_a_real_error():
    report = run("```literal [9]``` The rover uses solar panels [99].")
    finding, = report["findings"]
    assert finding["code"] == "CC001"
    assert finding["source_indices"] == (99,)
    assert report["analyzed_sentence_count"] == 1


def test_markdown_evidence_escapes_html():
    report = run('The manual says "<img onerror=alert(1)>wrong | quote" [1].', ["The manual says stable signal."])
    rendered = markdown_report([report])
    assert "<img" not in rendered and "&#124;" in rendered
