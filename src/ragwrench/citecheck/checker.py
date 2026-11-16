"""Deterministic diagnostics, not a semantic truth or entailment judge."""
from dataclasses import asdict
import json

from .lexical import coverage, quantities, quantity_supported, quote_key, quote_key_spans, quote_spans, source_text, tokens
from .markers import mask_markers, parse_markers
from .model import Config, Finding, InputError, Record, validate
from .prose import mask_markdown, sentences

CODES = ("CC001", "CC002", "CC003", "CC004", "CC005", "CC006", "CC007", "CC008")
SEVERITIES = {"CC001": "error", "CC002": "warning", "CC003": "warning", "CC004": "error",
              "CC005": "hint", "CC006": "error", "CC007": "info", "CC008": "info"}


def check(record: Record, config: Config | None = None) -> dict:
    validate(record)
    config = config or Config()
    prose = mask_markdown(record.answer)
    markers = parse_markers(prose)
    units = sentences(prose, markers)
    sources = {s.idx: s for s in record.sources}
    texts = {idx: source_text(s.text) for idx, s in sources.items()}
    source_tokens = {idx: tokens(text) for idx, text in texts.items()}
    source_quantities = {}
    quantity_budget = 20_000
    for idx, text in texts.items():
        source_quantities[idx] = quantities(text, limit=quantity_budget)
        quantity_budget -= len(source_quantities[idx])
    source_quote_keys = {idx: quote_key(text) for idx, text in texts.items()}
    source_quote_spans = {}  # Build provenance lazily, only for reported matches.
    findings = []
    used = set()
    sentence_reports = []

    def add(code, sentence, message, indices=(), evidence=None, *, severity=None):
        if len(findings) >= 10_000:
            raise InputError("report exceeds 10000 findings; split the input")
        findings.append(Finding(code, severity or SEVERITIES[code], sentence, message, tuple(indices), evidence))

    for sentence in units:
        clean = mask_markers(prose[sentence.start:sentence.end], [
            type(m)(m.start - sentence.start, m.end - sentence.start, m.raw, m.indices, m.error)
            for m in sentence.markers
        ])
        claim_tokens = tokens(clean)
        refs = tuple(dict.fromkeys(n for m in sentence.markers for n in m.indices))
        cited = tuple(n for n in refs if n in sources)
        used.update(cited)
        for marker in sentence.markers:
            if marker.error:
                add("CC001", sentence.idx, marker.error,
                    evidence={"marker": marker.raw, "start": marker.start, "end": marker.end, "kind": "malformed"})
            else:
                missing = [n for n in marker.indices if n not in sources]
                if missing:
                    add("CC001", sentence.idx, "citation has no matching source", missing,
                        {"marker": marker.raw, "start": marker.start, "end": marker.end, "kind": "dangling"})
        claim_quantities = quantities(clean, limit=256)
        # A transparent factual-sentence proxy: declarative content with enough
        # lexical terms or a literal quantity. Questions and short headings skip.
        factual = bool(claim_tokens) and (len(claim_tokens) >= config.min_claim_tokens or bool(claim_quantities))
        if clean.rstrip().endswith(("?", "？")):
            factual = False
        if factual and not refs:
            add("CC002", sentence.idx, "claim-like sentence has no valid citation marker")
        scores = {idx: coverage(claim_tokens, terms) for idx, terms in source_tokens.items()}
        cited_score = max((scores[idx] for idx in cited), default=0.0)
        if cited and claim_tokens and cited_score < config.support_threshold:
            add("CC003", sentence.idx, "low lexical coverage in every cited source", cited,
                {"coverage": round(cited_score, 6), "threshold": config.support_threshold,
                 "missing_tokens": sorted(claim_tokens - frozenset().union(*(source_tokens[n] for n in cited)))[:50]})
        if cited:
            cited_quantities = [q for n in cited for q in source_quantities[n]]
            for quantity in claim_quantities:
                if not quantity_supported(quantity, cited_quantities):
                    add("CC004", sentence.idx, "literal number/date/unit not found in cited sources", cited,
                        {"claim": asdict(quantity), "found": sorted({q.value + (" " + q.unit if q.unit else "") for q in cited_quantities})[:50], "source_quantity_count": len(cited_quantities)})
            for start, end in quote_spans(clean):
                # Masked prose finds the spans, but evidence and matching use
                # original content. Never return normalized text as the quote.
                start, end = sentence.start + start, sentence.start + end
                quote = record.answer[start:end]
                if any(quote in texts[n] for n in cited):
                    continue
                key = quote_key(quote)
                matched = tuple(n for n in cited if key and key in source_quote_keys[n])
                evidence = {"quote": quote, "start": start, "end": end,
                            "normalized_quote": key,
                            "match_kind": "normalized" if matched else "unsupported"}
                if matched:
                    source_matches = []
                    for n in matched[:50]:
                        if n not in source_quote_spans:
                            _, source_quote_spans[n] = quote_key_spans(texts[n])
                        pos = source_quote_keys[n].find(key)
                        spans = source_quote_spans[n][pos:pos + len(key)]
                        source_start = min(start for start, _ in spans)
                        source_end = max(end for _, end in spans)
                        source_matches.append({"source_idx": n,
                                               "excerpt": texts[n][source_start:source_end],
                                               "start": source_start, "end": source_end,
                                               "offset_basis": "source_prose"})
                    evidence.update(source_matches=source_matches, source_match_count=len(matched))
                    add("CC008", sentence.idx, "quoted span matches only after Unicode/whitespace normalization", matched,
                        evidence, severity="error" if config.strict_quotes else "info")
                else:
                    add("CC006", sentence.idx, "quoted span absent even after normalization in cited sources", cited,
                        evidence)
        best_idx = max(scores, key=lambda idx: (scores[idx], -idx)) if scores else None
        if refs and best_idx is not None and best_idx not in cited and claim_tokens:
            if scores[best_idx] >= config.best_source_threshold and scores[best_idx] - cited_score >= config.misattribution_margin:
                add("CC005", sentence.idx, "another source has substantially better lexical support", (best_idx,),
                    {"cited_coverage": round(cited_score, 6), "best_coverage": round(scores[best_idx], 6)})
        sentence_reports.append({"idx": sentence.idx, "start": sentence.start, "end": sentence.end,
                                 "text": record.answer[sentence.start:sentence.end].strip(),
                                 "citation_indices": list(refs), "claim_like": factual,
                                 "cited_coverage": round(cited_score, 6), "best_source": best_idx})
    for idx in sources:
        if idx not in used:
            add("CC007", None, "retrieved source is never cited", (idx,))
    # A code-only or heading-only answer must be observable even with no findings.
    return {"schema_version": 1, "kind": "citecheck-record", "config": asdict(config),
            "sentences": sentence_reports, "source_count": len(sources),
            "analyzed_sentence_count": len(units),
            "notes": ["Lexical checks cannot establish entailment, truth, negation, or cross-language paraphrase.",
                      "Quantity checks use literal decimals, numeric dates and a fixed unit vocabulary; no arithmetic or unit conversion.",
                      "Unknown units are treated as unitless; evidence lists show at most 50 values/tokens."]
                     + ["Quote matching uses extracted source prose, NFKC, casefolding, whitespace collapse and listed CJK/curly punctuation equivalences; CC008 identifies normalization-only matches."]
                     + (["No prose sentences were analyzed (Markdown headings/code were skipped)."] if not units else []),
            "findings": [f.to_dict() for f in findings]}


def fails(reports: list[dict], fail_on: str) -> bool:
    rank = {"info": 0, "hint": 1, "warning": 2, "error": 3, "none": 4}
    return any(rank[f["severity"]] >= rank[fail_on] for report in reports for f in report["findings"])


def markdown_report(reports: list[dict]) -> str:
    def safe(value):
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "&#124;").replace("\n", " ").replace("`", "&#96;").replace("[", "&#91;").replace("]", "&#93;")

    lines = ["# Citation check", "", "Deterministic lexical diagnostics; no semantic entailment guarantee.", ""]
    for n, report in enumerate(reports, 1):
        lines.extend([f"## Record {n}", "", f"Analyzed {report['analyzed_sentence_count']} sentences; {report['source_count']} sources.", "",
                      "| Code | Severity | Sentence | Sources | Diagnostic / evidence |",
                      "| --- | --- | --- | --- | --- |"])
        for finding in report["findings"]:
            evidence = json.dumps(finding["evidence"], ensure_ascii=False) if finding["evidence"] else ""
            lines.append("| " + " | ".join(safe(v) for v in (finding["code"], finding["severity"], finding["sentence"] or "—",
                                                                  ", ".join(map(str, finding["source_indices"])), finding["message"] + " " + evidence)) + " |")
        if not report["findings"]:
            lines.extend(["", "No findings."])
        lines.extend(["", *[safe(note) for note in report["notes"]], ""])
    return "\n".join(lines)
