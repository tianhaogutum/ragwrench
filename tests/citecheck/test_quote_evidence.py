"""CC008 source evidence contracts, using local synthetic source text only."""
import json
from pathlib import Path

import pytest

from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check
from ragwrench.citecheck.lexical import quote_key, quote_key_spans, source_text
from ragwrench.citecheck.model import Config


def quote_findings(answer, sources, strict=False):
    result = check(adapt({"answer": answer, "sources": sources}), Config(strict_quotes=strict))
    return [f for f in result["findings"] if f["code"] in {"CC006", "CC008"}]


@pytest.mark.parametrize("quote,excerpt", [
    ("ffi stable", "ﬃ stable"),                 # NFKC expansion
    ("café stable", "cafe\u0301 stable"),       # NFKC composition
    ("STRASSE stable", "Straße stable"),       # casefold expansion
    ("가 stable", "\u1100\u1161 stable"),       # Hangul composition
    ("À\u0315 stable", "a\u0315\u0300 stable"), # canonical reordering + composition
    ("signal stable", "SIGNAL\t \n stable"),   # whitespace run preserved
    ("  signal stable  ", "SIGNAL stable"),      # trim quote edges, keep source bounds
    ("FI", "ﬃ"),                              # partial expansion: covering excerpt
    ("S stable", "ß stable"),                  # match starts inside casefold expansion
])
@pytest.mark.parametrize("strict", [False, True])
def test_source_evidence_tracks_length_changes(quote, excerpt, strict):
    # Both expansion and contraction precede the match: normalized indices
    # cannot be used directly as offsets into the extracted source prose.
    prefix = "🙂 ㍍ cafe\u0301 / "
    source = prefix + excerpt + " / trailing private context"
    answer = f'The manual says 「{quote}」 [1].'
    finding, = quote_findings(answer, [{"idx": 1, "text": source}], strict)
    assert (finding["code"], finding["severity"]) == ("CC008", "error" if strict else "info")
    evidence = finding["evidence"]
    assert answer[evidence["start"]:evidence["end"]] == quote
    assert evidence["source_match_count"] == 1
    assert evidence["source_matches"] == [{
        "source_idx": 1, "excerpt": excerpt, "start": len(prefix),
        "end": len(prefix) + len(excerpt), "offset_basis": "source_prose",
    }]
    match, = evidence["source_matches"]
    assert source_text(source)[match["start"]:match["end"]] == match["excerpt"]
    assert "private context" not in json.dumps(evidence)


def test_source_evidence_uses_first_occurrence_and_citation_order_only():
    answer = 'The manual says "SIGNAL stable" [3,1,3,2].'
    finding, = quote_findings(answer, [
        {"idx": 1, "text": "🙂 signal stable / signal stable"},
        {"idx": 2, "text": "The manual says signal steady."},
        {"idx": 3, "text": "prefix Signal\tstable"},
        {"idx": 4, "text": "SIGNAL stable"},  # uncited exact match cannot suppress
    ])
    assert finding["source_indices"] == (3, 1)
    assert finding["evidence"]["source_match_count"] == 2
    assert finding["evidence"]["source_matches"] == [
        {"source_idx": 3, "excerpt": "Signal\tstable", "start": 7, "end": 20, "offset_basis": "source_prose"},
        {"source_idx": 1, "excerpt": "signal stable", "start": 2, "end": 15, "offset_basis": "source_prose"},
    ]


def test_source_evidence_is_bounded_without_losing_matched_indices():
    answer = 'The manual says "SIGNAL stable" [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51].'
    finding, = quote_findings(answer, [{"idx": i, "text": "signal stable"} for i in range(1, 52)])
    assert finding["source_indices"] == tuple(range(1, 52))
    assert finding["evidence"]["source_match_count"] == 51
    assert [m["source_idx"] for m in finding["evidence"]["source_matches"]] == list(range(1, 51))


def test_source_evidence_html_entities_and_inserted_spaces_are_prose_offsets():
    raw = '<p data-hidden="SECRET">🙂 preface</p><b>cafe&#x301;</b><i>&amp;\tSTABLE</i><p>tail</p>'
    prose = "🙂 preface cafe\u0301 &\tSTABLE tail"
    assert source_text(raw) == prose
    finding, = quote_findings('The manual says "CAFÉ & stable" [1].', [{"idx": 1, "text": raw}])
    match, = finding["evidence"]["source_matches"]
    assert match == {"source_idx": 1, "excerpt": "cafe\u0301 &\tSTABLE", "start": 10, "end": 24,
                     "offset_basis": "source_prose"}
    assert prose[match["start"]:match["end"]] == match["excerpt"]
    assert raw[match["start"]:match["end"]] != match["excerpt"]
    assert "SECRET" not in json.dumps(finding["evidence"])


@pytest.mark.parametrize("source", ["signal steady", '<p title="SIGNAL stable">unrelated</p>', "SIGNALstable", "<p>   </p>"])
def test_source_evidence_never_invents_matches_for_unsupported_quotes(source):
    finding, = quote_findings('The manual says "SIGNAL stable" [1].', [{"idx": 1, "text": source}])
    assert finding["code"] == "CC006"
    assert "source_matches" not in finding["evidence"]
    assert "source_match_count" not in finding["evidence"]


def test_source_evidence_exact_match_suppression_and_no_cross_source_join():
    assert quote_findings('The manual says "SIGNAL stable" [1,2].', [
        {"idx": 1, "text": "signal stable"}, {"idx": 2, "text": "<b>SIGNAL stable</b>"}]) == []
    finding, = quote_findings('The manual says "SIGNAL stable" [1,2].', [
        {"idx": 1, "text": "signal"}, {"idx": 2, "text": "stable"}])
    assert finding["code"] == "CC006"
    assert "source_matches" not in finding["evidence"]


def test_source_evidence_ragflow_example():
    fixture = Path(__file__).resolve().parents[2] / "examples/cite-check/source_quote_evidence.json"
    result = check(adapt(json.loads(fixture.read_text()), "ragflow"))
    finding, = result["findings"]
    assert finding["code"] == "CC008"
    assert finding["evidence"]["source_matches"] == [{
        "source_idx": 1, "excerpt": "cafe\u0301 &\tSTABLE", "start": 26, "end": 40,
        "offset_basis": "source_prose",
    }]


@pytest.mark.parametrize("text", [
    "", " \t\n", "\u0300\u0327Ａßﬃ\u2003", "á\u0327a\u0315\u0300",
    "\u1100\u1161\u11a8\u1100\u1161", "\u1f88\u0345\u0327",
    " \tCafe\u0301  ＳＴＲＡＳＳＥ ﬃ\u00a0「信号：稳定」。\n",
])
def test_source_provenance_key_agrees_with_existing_normalization(text):
    key, spans = quote_key_spans(text)
    assert key == quote_key(text)
    assert len(spans) == len(key)
    assert all(0 <= start < end <= len(text) for start, end in spans)
