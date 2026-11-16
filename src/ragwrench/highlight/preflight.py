"""Citation-box evidence and explicit, document-bound review decisions."""
from __future__ import annotations

from collections import Counter
import json
from typing import Any

from pypdf import PdfReader

from .core import (CiteError, MAX_JSON_BYTES, Region, canonical, geometries, identifier,
                   obj, seq, validate_regions)
from .pdf import annotation_id, overlap, text_fragments, tokens
from .sidecar import COORDINATE_SPACE, document_fingerprint

FORMAT = "citehl-preflight"


class _Assessment:
    def __init__(self, reader: PdfReader):
        self.reader = reader
        self.geometry = geometries(reader)
        self.cache = {}

    def item(self, region: Region) -> dict:
        if region.page not in self.geometry:
            raise CiteError(f"page {region.page} outside PDF (1..{len(self.geometry)})")
        reasons = []
        status = "ready"
        try:
            validate_regions([region], self.geometry, strict=True)
        except CiteError:
            reasons.append("outside_visible_page")
        if region.page not in self.cache:
            try:
                self.cache[region.page] = (text_fragments(self.reader.pages[region.page - 1]), None)
            except Exception as exc:
                # Extraction is heuristic; failure must be visible and reviewable per item.
                self.cache[region.page] = ([], type(exc).__name__)
        fragments, error = self.cache[region.page]
        expected = tokens(region.text)
        under_box = [(text, rect) for text, rect in fragments if overlap(region.rect, rect)]
        actual = set().union(*(tokens(text) for text, _ in under_box))
        if error:
            reasons.append("text_extraction_failed")
        elif not expected or not actual:
            reasons.append("text_unavailable")
        elif not expected & actual:
            reasons.append("text_mismatch")
        # Repeated shared words in distinct extracted word boxes are an ambiguity
        # signal, not a claim that either occurrence is semantically correct.
        matching = {(text, rect) for text, rect in under_box if tokens(text) & expected}
        counts = Counter(word for text, _ in matching for word in tokens(text) & expected)
        if any(count > 1 for count in counts.values()):
            reasons.append("repeated_matching_text")
        if reasons:
            status = "ambiguous" if reasons == ["repeated_matching_text"] else "unverifiable"
        return {"annotation_id": annotation_id(region), "region": region.to_dict(),
                "status": status, "reasons": reasons,
                "evidence": {"shared_tokens": sorted(expected & actual),
                             "matching_fragments": len(matching), "extraction_error": error},
                "resolution": None}


def _items(reader: PdfReader, regions: list[Region]) -> tuple[_Assessment, list[dict]]:
    assessor = _Assessment(reader)
    items = [assessor.item(r) for r in regions]
    if len({item["annotation_id"] for item in items}) != len(items):
        raise CiteError("duplicate region annotation IDs")
    return assessor, items


def _report(reader: PdfReader, items: list[dict]) -> dict:
    counts = Counter(item["status"] for item in items)
    report = {"format": FORMAT, "schema_version": 1, "coordinate_space": dict(COORDINATE_SPACE),
              "document": document_fingerprint(reader),
              "summary": {"total": len(items), "ready": counts["ready"], "ambiguous": counts["ambiguous"],
                          "unverifiable": counts["unverifiable"],
                          "needs_resolution": len(items) - counts["ready"]},
              "items": items}
    if len((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")) > MAX_JSON_BYTES:
        raise CiteError("preflight report exceeds 32 MiB JSON input limit")
    return report


def preflight(reader: PdfReader, regions: list[Region]) -> dict:
    """Return a report/template without changing the PDF or input regions."""
    _, items = _items(reader, regions)
    return _report(reader, items)


def resolve_regions(reader: PdfReader, regions: list[Region], data: Any = None) -> list[Region]:
    """Return the reviewed set using the same validation as audit generation."""
    return _resolve_regions(reader, regions, data)[0]


def resolve_with_audit(reader: PdfReader, regions: list[Region], data: Any) -> tuple[list[Region], dict]:
    """Resolve once and describe every original-to-final mapping in input order."""
    data = obj(data, "resolutions")
    result, records = _resolve_regions(reader, regions, data)
    counts = Counter(record["action"] for record in records)
    return result, {
        "format": "citehl-resolution-audit", "schema_version": 1,
        "coordinate_space": dict(COORDINATE_SPACE),
        # _resolve_regions has checked this document against recomputed page data.
        "document": data["document"],
        "summary": {"original_count": len(regions), "final_count": len(result),
                    **{action: counts[action] for action in ("ready", "accept", "replace", "skip")}},
        "items": records,
    }


def _resolve_regions(reader: PdfReader, regions: list[Region], data: Any) -> tuple[list[Region], list[dict]]:
    """Require a decision for every flagged item, then return the reviewed set.

    Only the resolution fields may be edited. Recomputing evidence binds every
    decision to the current input and assessment contract; page identity ignores
    annotation arrays so applying the same decisions again is supported.
    """
    assessor, items = _items(reader, regions)
    if data is None:
        flagged = sum(item["status"] != "ready" for item in items)
        if flagged:
            raise CiteError(f"preflight: {flagged}/{len(items)} items need resolution; run ragwrench highlight preflight "
                            "and supply the edited report with --resolutions")
        return validate_regions(regions, assessor.geometry, strict=True), []
    data = obj(data, "resolutions")
    entries = seq(data.get("items"), "resolution items")
    decisions = {}
    stripped = []
    for entry in entries:
        entry = obj(entry, "resolution item")
        name = identifier(entry.get("annotation_id"), "annotation_id")
        if name in decisions:
            raise CiteError("duplicate resolution annotation_id")
        decisions[name] = entry.get("resolution")
        stripped.append(dict(entry, resolution=None))
    # Exact version/type/coordinate/evidence/set/order checks; never trust edited
    # summary/status fields as permission to omit a required decision.
    if canonical(dict(data, items=stripped)) != canonical(_report(reader, items)):
        raise CiteError("stale or modified preflight report; regenerate for this PDF and regions "
                        "and edit only resolution fields")
    result = []
    records = []
    for region, item in zip(regions, items):
        decision = decisions[item["annotation_id"]]
        action, reason = "ready", None
        if decision is None:
            if item["status"] != "ready":
                raise CiteError(f"unresolved preflight item: {item['annotation_id']}")
        else:
            decision = obj(decision, "resolution")
            action = decision.get("action")
            reason = identifier(decision.get("reason"), "resolution reason")
            allowed = {"action", "reason", "page", "rect"} if action == "replace" else {"action", "reason"}
            if set(decision) != allowed or action not in ("accept", "skip", "replace"):
                raise CiteError("resolution requires action accept/skip/replace and reason; "
                                "replace also requires page and rect")
        if action == "replace":
            region = Region.from_dict(dict(region.to_dict(), page=decision["page"], rect=decision["rect"]))
            if assessor.item(region)["status"] != "ready":
                raise CiteError("replacement still requires review; select a verifiable, unambiguous box "
                                "or explicitly accept/skip the original")
        if action != "skip":
            # Acceptance acknowledges unavailable/ambiguous text, never invalid geometry.
            validate_regions([region], assessor.geometry, strict=True)
            result.append(region)
        records.append({"annotation_id": item["annotation_id"], "action": action, "reason": reason,
                        "final_annotation_id": annotation_id(region) if action != "skip" else None,
                        "final_region": region.to_dict() if action != "skip" else None})
    if len({annotation_id(r) for r in result}) != len(result):
        raise CiteError("resolutions produce duplicate annotation IDs")
    return result, records
