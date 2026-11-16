"""Review decisions are bound to real PDF pages and replay without duplicate highlights."""
import json
from dataclasses import replace

import pytest
from pypdf import PdfWriter

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import open_pdf, read_regions, regions_jsonl
from ragwrench.highlight.pdf import verify


@pytest.fixture
def review_case(examples, tmp_path):
    pdf = examples / "review-rotated.pdf"
    regions = tmp_path / "regions.jsonl"
    assert main(["normalize", "--format", "unstructured", str(examples / "review-rotated.unstructured.json"),
                 "--pdf", str(pdf), "-o", str(regions)]) == 0
    assert all(r.rect == (80, 120, 150, 140) for r in read_regions(regions)[:4])
    review = tmp_path / "review.json"
    assert main(["preflight", str(pdf), str(regions), "-o", str(review)]) == 0
    return pdf, regions, review


def decisions(review):
    data = json.loads(review.read_text())
    for item in data["items"]:
        ref = item["region"]["source_ref"]
        if ref == "ambiguous":
            item["resolution"] = {"action": "replace", "page": 2, "rect": [220, 120, 290, 140],
                                  "reason": "Select the second occurrence after visual inspection"}
        elif ref == "blank":
            item["resolution"] = {"action": "accept", "reason": "Retain the manually checked non-text citation"}
        elif ref == "mismatch":
            item["resolution"] = {"action": "skip", "reason": "Producer text does not match this page"}
    return data


def annotation_args(pdf, regions, output, review=None):
    args = ["annotate", str(pdf), str(regions), "-o", str(output)]
    return args + (["--resolutions", str(review)] if review else [])


def test_preflight_summary_and_resolution_replay(review_case, tmp_path, capsys):
    pdf, regions, review = review_case
    report = json.loads(review.read_text())
    assert report["summary"] == {"total": 7, "ready": 4, "ambiguous": 1, "unverifiable": 2, "needs_resolution": 3}
    assert [item["reasons"] for item in report["items"][-3:]] == [
        ["repeated_matching_text"], ["text_unavailable"], ["text_mismatch"]]
    assert all(item["resolution"] is None for item in report["items"])
    output = tmp_path / "out.pdf"
    assert main(annotation_args(pdf, regions, output)) == 1
    assert "preflight" in capsys.readouterr().err
    assert not output.exists() and not (tmp_path / "citations.json").exists()
    assert main(annotation_args(pdf, regions, output, review)) == 1
    assert not output.exists()
    review.write_text(json.dumps(decisions(review)))
    assert main(annotation_args(pdf, regions, output, review)) == 0
    resolved = read_regions(regions)[:4] + [replace(read_regions(regions)[4], rect=(220.0, 120.0, 290.0, 140.0)),
                                          read_regions(regions)[5]]
    assert verify(open_pdf(output), resolved)["text_skipped"] == 1
    assert main(["verify", str(output), str(regions), "--resolutions", str(review)]) == 0
    again = tmp_path / "again.pdf"
    assert main(annotation_args(output, regions, again, review) + ["--force"]) == 0
    assert [len(p["/Annots"]) for p in open_pdf(again).pages] == [1, 3, 1, 1]
    assert verify(open_pdf(again), resolved)["annotations"] == 6
    links = json.loads((tmp_path / "citations.json").read_text())["chunks"]
    assert "mismatch" not in links
    assert links["ambiguous"]["rects"][0]["rect"] == [220, 120, 290, 140]
    # Export the actual accepted/replaced/filtered set without reconstructing JSONL.
    sidecar = tmp_path / "reviewed.json"
    assert main(["export", str(again), "-o", str(sidecar)]) == 0
    exported = json.loads(sidecar.read_text())["annotations"]
    assert {entry["annotation_id"] for entry in exported} == {
        item["annotation_id"] for chunk in links.values() for item in chunk["rects"]}
    assert {entry["region"]["source_ref"] for entry in exported} == {r.source_ref for r in resolved}
    assert next(entry["region"]["rect"] for entry in exported
                if entry["region"]["source_ref"] == "ambiguous") == [220, 120, 290, 140]


@pytest.mark.parametrize("command", ["annotate", "resolve", "resolve-audit"])
@pytest.mark.parametrize("change", ["missing", "unknown", "duplicate", "reordered", "version", "summary", "status", "id",
                                    "decision", "reason", "replacement", "replacement-ambiguous", "region", "pdf"])
def test_stale_or_invalid_resolutions_never_replace_output(review_case, tmp_path, capsys, change, command):
    pdf, regions, review = review_case
    data = decisions(review)
    if change == "missing":
        data["items"].pop()
    elif change == "unknown":
        data["items"].append(dict(data["items"][0], annotation_id="unknown"))
    elif change == "duplicate":
        data["items"].append(data["items"][0])
    elif change == "reordered":
        data["items"].reverse()
    elif change == "version":
        data["schema_version"] = True
    elif change == "summary":
        data["summary"]["needs_resolution"] = 0
    elif change == "status":
        data["items"][-1]["status"] = "ready"
    elif change == "id":
        data["items"][-1]["annotation_id"] = "unknown"
    elif change == "decision":
        data["items"][-1]["resolution"]["action"] = "guess"
    elif change == "reason":
        data["items"][-1]["resolution"]["reason"] = " "
    elif change == "replacement":
        data["items"][-3]["resolution"]["rect"] = [0, 0, 999, 999]
    elif change == "replacement-ambiguous":
        data["items"][-3]["resolution"]["rect"] = [80, 120, 290, 140]
    elif change == "region":
        changed = read_regions(regions)
        changed[0] = replace(changed[0], text="edited")
        regions.write_text(regions_jsonl(changed))
    else:
        writer = PdfWriter(clone_from=open_pdf(pdf))
        writer.pages[0].rotate(90)
        pdf = tmp_path / "changed.pdf"
        writer.write(pdf)
    review.write_text(json.dumps(data))
    output = tmp_path / "out.pdf"
    output.write_bytes(b"keep PDF")
    manifest = tmp_path / "citations.json"
    manifest.write_text("keep manifest")
    args = annotation_args(pdf, regions, output, review)
    args[0] = "resolve" if command == "resolve-audit" else command
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"keep audit")
    if command == "resolve-audit":
        args += ["--audit", str(audit)]
    assert main(args + ["--force"]) == 1
    assert "Traceback" not in capsys.readouterr().err
    assert output.read_bytes() == b"keep PDF"
    assert manifest.read_text() == "keep manifest"
    assert audit.read_bytes() == b"keep audit"


def test_resolution_input_cannot_be_an_output(review_case, tmp_path):
    pdf, regions, review = review_case
    before = review.read_bytes()
    assert main(annotation_args(pdf, regions, review, review) + ["--force"]) == 1
    assert review.read_bytes() == before


def test_ready_boxes_need_no_decisions_and_all_skips_clear_namespace(review_case, tmp_path):
    pdf, regions, review = review_case
    good_file = tmp_path / "good.jsonl"
    good_file.write_text(regions_jsonl(read_regions(regions)[:4]))
    good_review = tmp_path / "good-review.json"
    assert main(["preflight", str(pdf), str(good_file), "-o", str(good_review)]) == 0
    good = tmp_path / "good.pdf"
    assert main(annotation_args(pdf, good_file, good, good_review)) == 0
    # Explicitly skipping every item is a useful way to remove the tool namespace.
    data = json.loads(review.read_text())
    for item in data["items"]:
        item["resolution"] = {"action": "skip", "reason": "Remove this citation"}
    review.write_text(json.dumps(data))
    cleared = tmp_path / "cleared.pdf"
    assert main(annotation_args(good, regions, cleared, review) + ["--force"]) == 0
    assert verify(open_pdf(cleared), [], text_check="off")["annotations"] == 0
    assert main(["verify", str(cleared), str(regions), "--resolutions", str(review)]) == 0


def test_outside_boxes_require_replace_or_skip(examples, tmp_path, capsys):
    pdf = examples / "review-rotated.pdf"
    # The box has text evidence but extends beyond the CropBox.
    from ragwrench.highlight.core import Region
    region = Region("outside", "outside", 1, (0, 120, 150, 140), "ANCHOR")
    path, review, output = tmp_path / "regions.jsonl", tmp_path / "review.json", tmp_path / "out.pdf"
    path.write_text(regions_jsonl([region]))
    assert main(["preflight", str(pdf), str(path), "-o", str(review)]) == 0
    data = json.loads(review.read_text())
    assert data["items"][0]["reasons"] == ["outside_visible_page"]
    data["items"][0]["resolution"] = {"action": "accept", "reason": "Try to bypass geometry validation"}
    review.write_text(json.dumps(data))
    assert main(annotation_args(pdf, path, output, review)) == 1
    assert not output.exists()
    assert "visible page" in capsys.readouterr().err
    data["items"][0]["resolution"] = {"action": "replace", "reason": "Choose the visible text", "page": 1,
                                      "rect": [80, 120, 150, 140]}
    review.write_text(json.dumps(data))
    assert main(annotation_args(pdf, path, output, review)) == 0


def test_extraction_failure_is_per_item_and_explicit(examples, monkeypatch):
    from ragwrench.highlight.preflight import preflight, resolve_regions
    from ragwrench.highlight.core import CiteError
    reader = open_pdf(examples / "sample.pdf")
    regions = read_regions(examples / "regions.jsonl")

    def unavailable(page):
        raise ValueError("synthetic extraction failure")

    monkeypatch.setattr("ragwrench.highlight.preflight.text_fragments", unavailable)
    report = preflight(reader, regions)
    assert report["summary"]["unverifiable"] == 2
    assert all(item["reasons"] == ["text_extraction_failed"] for item in report["items"])
    assert report["items"][0]["evidence"]["extraction_error"] == "ValueError"
    with pytest.raises(CiteError, match="unresolved"):
        resolve_regions(reader, regions, report)
    for item in report["items"]:
        item["resolution"] = {"action": "accept", "reason": "Visually confirmed"}
    assert resolve_regions(reader, regions, report) == regions


def test_empty_citation_text_is_unverifiable(examples):
    from ragwrench.highlight.preflight import preflight
    region = replace(read_regions(examples / "regions.jsonl")[0], text="")
    report = preflight(open_pdf(examples / "sample.pdf"), [region])
    assert report["items"][0]["reasons"] == ["text_unavailable"]


def test_preflight_output_is_protected(review_case):
    pdf, regions, review = review_case
    before = review.read_bytes()
    assert main(["preflight", str(pdf), str(regions), "-o", str(review)]) == 1
    assert review.read_bytes() == before
    before_regions = regions.read_bytes()
    assert main(["preflight", str(pdf), str(regions), "-o", str(regions), "--force"]) == 1
    assert regions.read_bytes() == before_regions


def test_resolve_materializes_reviewed_set_for_manifest_and_export(review_case, tmp_path):
    pdf, regions, review = review_case
    original = read_regions(regions)
    # Merged provenance and Unicode must survive review materialization.
    original[0] = replace(original[0], source_refs=(original[0].source_ref, "来源/α"))
    regions.write_text(regions_jsonl(original))
    assert main(["preflight", str(pdf), str(regions), "-o", str(review), "--force"]) == 0
    review.write_text(json.dumps(decisions(review)))
    final = tmp_path / "final.jsonl"
    assert main(["resolve", str(pdf), str(regions), "--resolutions", str(review), "-o", str(final)]) == 0
    expected = original[:4] + [replace(original[4], rect=(220.0, 120.0, 290.0, 140.0)), original[5]]
    assert read_regions(final) == expected
    assert final.read_text() == regions_jsonl(expected)
    assert not (tmp_path / "citations.json").exists()
    output = tmp_path / "reviewed.pdf"
    assert main(annotation_args(pdf, regions, output, review)) == 0
    assert main(["verify", str(output), str(final)]) == 0
    standalone = tmp_path / "standalone.json"
    assert main(["manifest", str(output), str(final), "-o", str(standalone)]) == 0
    assert standalone.read_bytes() == (tmp_path / "citations.json").read_bytes()
    sidecar = tmp_path / "sidecar.json"
    assert main(["export", str(output), str(final), "-o", str(sidecar)]) == 0
    assert [e["region"] for e in json.loads(sidecar.read_text())["annotations"]] == [r.to_dict() for r in expected]
    # Annotation arrays do not invalidate review replay; bytes stay deterministic.
    assert main(["resolve", str(output), str(regions), "--resolutions", str(review),
                 "-o", str(final), "--force"]) == 0
    assert final.read_text() == regions_jsonl(expected)


def test_resolve_all_skips_empty_downstream_contract(review_case, tmp_path, capsys):
    pdf, regions, review = review_case
    data = json.loads(review.read_text())
    for item in data["items"]:
        item["resolution"] = {"action": "skip", "reason": "Remove citation"}
    review.write_text(json.dumps(data))
    final, output = tmp_path / "empty.jsonl", tmp_path / "cleared.pdf"
    assert main(["resolve", str(pdf), str(regions), "--resolutions", str(review), "-o", str(final)]) == 0
    assert final.read_bytes() == b""
    assert main(annotation_args(pdf, regions, output, review)) == 0
    for command in ("verify", "manifest", "export"):
        target = tmp_path / f"{command}.json"
        args = [command, str(output), str(final)]
        if command != "verify":
            target.write_bytes(b"sentinel")
            args += ["-o", str(target), "--force"]
        assert main(args) == 1
        if command != "verify":
            assert target.read_bytes() == b"sentinel"
        assert main(args + ["--allow-empty"]) == 0
    assert json.loads((tmp_path / "manifest.json").read_text())["chunks"] == {}
    explicit = json.loads((tmp_path / "export.json").read_text())
    assert explicit["annotations"] == []
    recovered = tmp_path / "pdf-only.json"
    assert main(["export", str(output), "-o", str(recovered)]) == 0
    assert json.loads(recovered.read_text()) == explicit
    # Empty JSONL is never valid as a new original review input.
    for command in ("preflight", "resolve", "annotate"):
        args = [command, str(pdf), str(final), "-o", str(tmp_path / "refused")]
        if command == "resolve":
            args += ["--resolutions", str(review)]
        assert main(args) == 1
        assert not (tmp_path / "refused").exists()
    assert "OK 0/0" in capsys.readouterr().out


def test_duplicate_resolution_action_cannot_silently_skip_every_item(review_case, tmp_path, capsys):
    pdf, regions, review = review_case
    data = json.loads(review.read_text())
    for item in data["items"]:
        item["resolution"] = {"action": "skip", "reason": "Remove citation"}
    review.write_text(json.dumps(data).replace('"action": "skip"',
                                             '"action": "accept", "action": "skip"'))
    output = tmp_path / "final.jsonl"
    output.write_bytes(b"sentinel")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"keep audit")
    assert main(["resolve", str(pdf), str(regions), "--resolutions", str(review),
                 "-o", str(output), "--audit", str(audit), "--force"]) == 1
    assert "duplicate JSON key" in capsys.readouterr().err
    assert output.read_bytes() == b"sentinel"
    assert audit.read_bytes() == b"keep audit"


@pytest.mark.parametrize("target", ["pdf", "regions", "review", "symlink", "existing"])
def test_resolve_protects_inputs_and_existing_output(review_case, tmp_path, target):
    pdf, regions, review = review_case
    # Copy PDF locally so a regression can never damage a committed fixture.
    local = tmp_path / "source.pdf"
    local.write_bytes(pdf.read_bytes())
    pdf = local
    review.write_text(json.dumps(decisions(review)))
    existing = tmp_path / "existing"
    existing.write_bytes(b"keep")
    alias = tmp_path / "alias"
    alias.symlink_to(review)
    output = {"pdf": pdf, "regions": regions, "review": review, "symlink": alias, "existing": existing}[target]
    before = output.read_bytes()
    args = ["resolve", str(pdf), str(regions), "--resolutions", str(review), "-o", str(output)]
    assert main(args + ([] if target == "existing" else ["--force"])) == 1
    assert output.read_bytes() == before
    if target == "existing":
        assert main(args + ["--force"]) == 0
        assert len(read_regions(output)) == 6


def test_resolve_requires_report_and_preserves_output_on_unresolved_or_oversize(review_case, tmp_path, monkeypatch):
    pdf, regions, review = review_case
    output = tmp_path / "final.jsonl"
    args = ["resolve", str(pdf), str(regions), "-o", str(output)]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    output.write_bytes(b"sentinel")
    assert main(args + ["--resolutions", str(review), "--force"]) == 1
    assert output.read_bytes() == b"sentinel"
    review.write_text(json.dumps(decisions(review)))
    # Cap only final JSONL serialization; the report and source still fit their limits.
    monkeypatch.setattr("ragwrench.highlight.cli.MAX_JSON_BYTES", 10)
    assert main(args + ["--resolutions", str(review), "--force"]) == 1
    assert output.read_bytes() == b"sentinel"


@pytest.mark.parametrize("report_content", [None, "null", "[]", "{bad json}"])
def test_resolve_ready_set_requires_valid_report(examples, tmp_path, report_content):
    pdf, regions = examples / "sample.pdf", examples / "regions.jsonl"
    report, output = tmp_path / "review.json", tmp_path / "final.jsonl"
    assert main(["preflight", str(pdf), str(regions), "-o", str(report)]) == 0
    if report_content is not None:
        report.write_text(report_content)
        output.write_bytes(b"sentinel")
    result = main(["resolve", str(pdf), str(regions), "--resolutions", str(report),
                   "-o", str(output), "--force"])
    if report_content is None:
        assert result == 0
        assert read_regions(output) == read_regions(regions)
    else:
        assert result == 1
        assert output.read_bytes() == b"sentinel"


@pytest.mark.parametrize("change", ["empty", "malformed", "duplicate", "page"])
def test_resolve_invalid_original_preserves_output(review_case, tmp_path, change):
    pdf, regions, review = review_case
    data = read_regions(regions)
    if change == "empty":
        regions.write_bytes(b"")
    elif change == "malformed":
        regions.write_text("null\n")
    elif change == "duplicate":
        regions.write_text(regions_jsonl(data) * 2)
    else:
        data[0] = replace(data[0], page=999)
        regions.write_text(regions_jsonl(data))
    output = tmp_path / "final.jsonl"
    output.write_bytes(b"sentinel")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"keep audit")
    assert main(["resolve", str(pdf), str(regions), "--resolutions", str(review),
                 "-o", str(output), "--audit", str(audit), "--force"]) == 1
    assert output.read_bytes() == b"sentinel"
    assert audit.read_bytes() == b"keep audit"


def test_resolve_rejects_decisions_collapsing_to_duplicate_ids(examples, tmp_path, capsys):
    pdf = examples / "sample.pdf"
    first = read_regions(examples / "regions.jsonl")[0]
    regions, review, output = tmp_path / "regions.jsonl", tmp_path / "review.json", tmp_path / "final.jsonl"
    regions.write_text(regions_jsonl([first, replace(first, rect=(72, 680, 300, 700))]))
    assert main(["preflight", str(pdf), str(regions), "-o", str(review)]) == 0
    data = json.loads(review.read_text())
    data["items"][1]["resolution"] = {"action": "replace", "reason": "Same box", "page": 1,
                                       "rect": list(first.rect)}
    review.write_text(json.dumps(data))
    output.write_bytes(b"sentinel")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"keep audit")
    assert main(["resolve", str(pdf), str(regions), "--resolutions", str(review),
                 "-o", str(output), "--audit", str(audit), "--force"]) == 1
    assert "duplicate annotation IDs" in capsys.readouterr().err
    assert output.read_bytes() == b"sentinel"
    assert audit.read_bytes() == b"keep audit"


def test_resolve_audit_maps_every_original_to_materialized_ids(review_case, tmp_path):
    from ragwrench.highlight.pdf import annotation_id
    pdf, regions, review = review_case
    local_pdf = tmp_path / "source.pdf"
    local_pdf.write_bytes(pdf.read_bytes())
    pdf = local_pdf
    report = decisions(review)
    # Preserve supplied Unicode reasons exactly, including surrounding whitespace.
    report["items"][-2]["resolution"]["reason"] = "  已核对 α  "
    review.write_text(json.dumps(report))
    final, audit = tmp_path / "final.jsonl", tmp_path / "audit.json"
    args = ["resolve", str(pdf), str(regions), "--resolutions", str(review)]
    assert main(args + ["-o", str(final), "--audit", str(audit)]) == 0
    data = json.loads(audit.read_text())
    assert set(data) == {"format", "schema_version", "coordinate_space", "document", "summary", "items"}
    assert data["format"] == "citehl-resolution-audit"
    assert type(data["schema_version"]) is int and data["schema_version"] == 1
    assert data["coordinate_space"] == {
        "name": "unrotated-pdf-user-space", "units": "points", "origin": "bottom-left",
        "rect": "[x0,y0,x1,y1]", "page_index_base": 1}
    assert data["document"] == report["document"]
    assert [p["page"] for p in data["document"]["pages"]] == [1, 2, 3, 4]
    assert data["summary"] == {"original_count": 7, "final_count": 6, "ready": 4,
                               "accept": 1, "replace": 1, "skip": 1}
    assert all(type(count) is int for count in data["summary"].values())
    assert [item["annotation_id"] for item in data["items"]] == [i["annotation_id"] for i in report["items"]]
    assert len({item["annotation_id"] for item in data["items"]}) == 7
    assert [item["action"] for item in data["items"]] == ["ready"] * 4 + ["replace", "accept", "skip"]
    final_regions = iter(read_regions(final))
    for original, item in zip(report["items"], data["items"]):
        assert set(item) == {"annotation_id", "action", "reason", "final_annotation_id", "final_region"}
        assert item["reason"] == (original["resolution"]["reason"] if original["resolution"] else None)
        if item["action"] == "skip":
            assert item["final_annotation_id"] is None and item["final_region"] is None
        else:
            region = next(final_regions)
            assert item["final_region"] == region.to_dict()
            assert item["final_annotation_id"] == annotation_id(region)
            assert (item["final_annotation_id"] == item["annotation_id"]) == (item["action"] != "replace")
    without_audit = tmp_path / "without.jsonl"
    assert main(args + ["-o", str(without_audit)]) == 0
    assert without_audit.read_bytes() == final.read_bytes()
    manifest = tmp_path / "standalone.json"
    assert main(["manifest", str(pdf), str(final), "-o", str(manifest)]) == 0
    manifest_ids = {r["annotation_id"] for c in json.loads(manifest.read_text())["chunks"].values() for r in c["rects"]}
    assert manifest_ids == {i["final_annotation_id"] for i in data["items"] if i["action"] != "skip"}
    before = audit.read_bytes()
    assert main(args + ["-o", str(final), "--audit", str(audit), "--force"]) == 0
    assert audit.read_bytes() == before


def test_resolve_audit_all_skip_and_explicit_ready_actions(review_case, tmp_path):
    pdf, regions, review = review_case
    report = json.loads(review.read_text())
    for item in report["items"]:
        item["resolution"] = {"action": "skip", "reason": "Remove citation"}
    review.write_text(json.dumps(report))
    final, audit = tmp_path / "final.jsonl", tmp_path / "audit.json"
    args = ["resolve", str(pdf), str(regions), "--resolutions", str(review),
            "-o", str(final), "--audit", str(audit)]
    assert main(args) == 0
    assert final.read_bytes() == b""
    data = json.loads(audit.read_text())
    assert data["summary"] == {"original_count": 7, "final_count": 0, "ready": 0,
                               "accept": 0, "replace": 0, "skip": 7}
    assert len(data["items"]) == 7
    assert all(i["action"] == "skip" and i["reason"] == "Remove citation" and
               i["final_annotation_id"] is None and i["final_region"] is None for i in data["items"])
    # An explicit decision on an already-ready item must not become implicit ready.
    report["items"][0]["resolution"] = {"action": "accept", "reason": "Keep ready box"}
    report["items"][1]["resolution"] = {"action": "replace", "reason": "Confirm same geometry",
                                        "page": 2, "rect": report["items"][1]["region"]["rect"]}
    review.write_text(json.dumps(report))
    assert main(args + ["--force"]) == 0
    data = json.loads(audit.read_text())
    assert data["summary"] == {"original_count": 7, "final_count": 2, "ready": 0,
                               "accept": 1, "replace": 1, "skip": 5}
    assert [i["action"] for i in data["items"][:2]] == ["accept", "replace"]
    assert all(i["annotation_id"] == i["final_annotation_id"] for i in data["items"][:2])


@pytest.mark.parametrize("destination", ["audit", "output"])
@pytest.mark.parametrize("source", ["pdf", "regions", "review", "other-output"])
@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_resolve_audit_rejects_all_output_aliases(review_case, tmp_path, destination, source, alias_kind):
    pdf, regions, review = review_case
    local_pdf = tmp_path / "source.pdf"
    local_pdf.write_bytes(pdf.read_bytes())
    pdf = local_pdf
    review.write_text(json.dumps(decisions(review)))
    paths = {"output": tmp_path / "final.jsonl", "audit": tmp_path / "audit.json"}
    for path in paths.values():
        path.write_bytes(b"sentinel")
    other = paths["audit" if destination == "output" else "output"]
    target = {"pdf": pdf, "regions": regions, "review": review, "other-output": other}[source]
    if alias_kind == "direct":
        paths[destination] = target
    else:
        alias = tmp_path / "alias"
        if alias_kind == "symlink":
            alias.symlink_to(target)
        else:
            alias.hardlink_to(target)
        paths[destination] = alias
    protected = {path: path.read_bytes() for path in (pdf, regions, review, *paths.values())}
    assert main(["resolve", str(pdf), str(regions), "--resolutions", str(review),
                 "-o", str(paths["output"]), "--audit", str(paths["audit"]), "--force"]) == 1
    assert all(path.read_bytes() == before for path, before in protected.items())


@pytest.mark.parametrize("destination", ["audit", "output"])
@pytest.mark.parametrize("problem", ["exists", "directory", "parent"])
def test_resolve_audit_prechecks_both_destinations(review_case, tmp_path, destination, problem):
    pdf, regions, review = review_case
    review.write_text(json.dumps(decisions(review)))
    paths = {"output": tmp_path / "final.jsonl", "audit": tmp_path / "audit.json"}
    if problem == "exists":
        paths[destination].write_bytes(b"sentinel")
    elif problem == "directory":
        paths[destination].mkdir()
    else:
        paths[destination] = tmp_path / "missing" / "file"
    assert main(["resolve", str(pdf), str(regions), "--resolutions", str(review),
                 "-o", str(paths["output"]), "--audit", str(paths["audit"])]
                + ([] if problem == "exists" else ["--force"])) == 1
    other = paths["audit" if destination == "output" else "output"]
    assert not other.exists()
    if problem == "exists":
        assert paths[destination].read_bytes() == b"sentinel"


def test_resolve_audit_size_and_later_io_failure_contract(review_case, tmp_path, monkeypatch, capsys):
    import ragwrench.highlight.cli as cli
    pdf, regions, review = review_case
    review.write_text(json.dumps(decisions(review)))
    final, audit = tmp_path / "final.jsonl", tmp_path / "audit.json"
    final.write_bytes(b"keep JSONL")
    audit.write_bytes(b"keep audit")
    args = ["resolve", str(pdf), str(regions), "--resolutions", str(review),
            "-o", str(final), "--audit", str(audit), "--force"]
    # JSONL would fit, but the audit must be serialized/size-checked before either write.
    with monkeypatch.context() as patch:
        patch.setattr(cli, "MAX_JSON_BYTES", len(regions_jsonl(read_regions(regions)).encode()) + 1)
        assert main(args) == 1
    assert "audit exceeds" in capsys.readouterr().err
    assert final.read_bytes() == b"keep JSONL" and audit.read_bytes() == b"keep audit"
    real_write = cli.write_text

    def fail_audit(path, content):
        if path == audit:
            raise OSError("synthetic audit write failure")
        real_write(path, content)

    monkeypatch.setattr(cli, "write_text", fail_audit)
    assert main(args) == 1
    assert "synthetic audit write failure" in capsys.readouterr().err
    assert len(read_regions(final)) == 6  # Each output is atomic; the pair is not a transaction.
    assert audit.read_bytes() == b"keep audit"


def test_resolve_audit_missing_path_is_usage_error(review_case, tmp_path):
    pdf, regions, review = review_case
    output = tmp_path / "final.jsonl"
    with pytest.raises(SystemExit) as exc:
        main(["resolve", str(pdf), str(regions), "--resolutions", str(review), "-o", str(output), "--audit"])
    assert exc.value.code == 2
    assert not output.exists()


def test_check_audit_recomputes_mixed_artifacts_read_only(review_case, tmp_path, capsys):
    pdf, regions, review = review_case
    report = decisions(review)
    report['items'][-2]['resolution']['reason'] = '  已核对 α  '
    review.write_text(json.dumps(report))
    final, audit = tmp_path / 'final.jsonl', tmp_path / 'audit.json'
    assert main(['resolve', str(pdf), str(regions), '--resolutions', str(review),
                 '-o', str(final), '--audit', str(audit)]) == 0
    paths = [pdf, regions, review, audit, final]
    before = {p: p.read_bytes() for p in paths}
    entries = sorted(tmp_path.iterdir())
    capsys.readouterr()
    assert main(['check-audit', str(pdf), str(regions), '--resolutions', str(review),
                 '--audit', str(audit), '--final-regions', str(final)]) == 0
    result = capsys.readouterr()
    assert result.err == ''
    assert result.out == ('OK check-audit {"original_count": 7, "final_count": 6, "ready": 4, '
                          '"accept": 1, "replace": 1, "skip": 1}\n')
    assert {p: p.read_bytes() for p in paths} == before
    assert sorted(tmp_path.iterdir()) == entries


@pytest.fixture
def audit_case(review_case, tmp_path):
    pdf, regions, review = review_case
    review.write_text(json.dumps(decisions(review)))
    final, audit = tmp_path / 'final.jsonl', tmp_path / 'audit.json'
    assert main(['resolve', str(pdf), str(regions), '--resolutions', str(review),
                 '-o', str(final), '--audit', str(audit)]) == 0
    paths = dict(pdf=pdf, original=regions, review=review, audit=audit, final=final)
    args = ['check-audit', str(pdf), str(regions), '--resolutions', str(review),
            '--audit', str(audit), '--final-regions', str(final)]
    return paths, args


def test_check_audit_json_success(audit_case, capsys):
    paths, args = audit_case
    before = {p: p.read_bytes() for p in paths.values()}
    entries = sorted(paths['review'].parent.iterdir())
    capsys.readouterr()
    assert main(args + ['--json']) == 0
    captured = capsys.readouterr()
    assert captured.err == ''
    assert captured.out.count('\n') == 1
    assert json.loads(captured.out) == {
        'format': 'citehl-check-audit-result', 'schema_version': 1, 'ok': True,
        'summary': {'original_count': 7, 'final_count': 6, 'ready': 4,
                    'accept': 1, 'replace': 1, 'skip': 1}, 'error': None,
    }
    assert {p: p.read_bytes() for p in paths.values()} == before
    assert sorted(paths['review'].parent.iterdir()) == entries


# Strict type checks include Python's bool == 1 and float == int traps.
@pytest.mark.parametrize('path,value', [
    ((), None), ((), []),
    (('format',), 'citehl-preflight'),
    (('schema_version',), 2), (('schema_version',), True), (('schema_version',), 1.0),
    (('coordinate_space', 'page_index_base'), True),
    (('coordinate_space', 'origin'), 'top-left'),
    (('document', 'algorithm'), 'unknown'),
    (('document', 'fingerprint'), '0' * 64),
    (('document', 'pages', 0, 'page'), 1.0),
    (('document', 'pages', 0, 'fingerprint'), '0' * 64),
    (('summary', 'original_count'), 6), (('summary', 'final_count'), 0),
    (('summary', 'ready'), 4.0), (('summary', 'accept'), True),
    (('summary', 'replace'), -1), (('summary', 'skip'), '1'),
    (('items',), []), (('items', 0), None),
    (('items', 0, 'annotation_id'), 'changed'),
    (('items', 4, 'action'), 'accept'),
    (('items', 5, 'reason'), 'Different reason'),
    (('items', 0, 'reason'), ''),
    (('items', 4, 'final_annotation_id'), 'changed'),
    (('items', 6, 'final_annotation_id'), 'changed'),
    (('items', 6, 'final_region'), {}),
    (('items', 0, 'final_region'), None),
    (('items', 0, 'final_region', 'page'), True),
    (('items', 0, 'final_region', 'rect', 0), 81),
    (('items', 0, 'final_region', 'rect', 0), '80'),
    (('items', 0, 'final_region', 'text'), 'edited'),
    (('items', 0, 'final_region', 'source_ref'), 'edited'),
    (('items', 0, 'final_region', 'chunk_id'), 'edited'),
])
def test_check_audit_rejects_modified_audit(audit_case, capsys, path, value):
    paths, args = audit_case
    data = json.loads(paths['audit'].read_text())
    if path:
        parent = data
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
    else:
        data = value
    paths['audit'].write_text(json.dumps(data))
    before = {p: p.read_bytes() for p in paths.values()}
    capsys.readouterr()
    assert main(args) == 1
    captured = capsys.readouterr()
    assert 'audit' in captured.err and 'regenerate' in captured.err
    assert captured.out == '' and 'Traceback' not in captured.err
    assert {p: p.read_bytes() for p in paths.values()} == before


@pytest.mark.parametrize('change', ['missing-field', 'extra-field', 'nested-extra', 'item-extra',
                                    'reordered-items', 'reordered-pages', 'extra-item', 'missing-item'])
def test_check_audit_rejects_audit_shape_and_order(audit_case, capsys, change):
    paths, args = audit_case
    data = json.loads(paths['audit'].read_text())
    if change == 'missing-field':
        del data['summary']
    elif change == 'extra-field':
        data['extra'] = None
    elif change == 'nested-extra':
        data['items'][0]['final_region']['extra'] = None
    elif change == 'item-extra':
        data['items'][0]['extra'] = None
    elif change == 'reordered-items':
        data['items'].reverse()
    elif change == 'reordered-pages':
        data['document']['pages'].reverse()
    elif change == 'extra-item':
        data['items'].append(data['items'][0])
    else:
        data['items'].pop()
    paths['audit'].write_text(json.dumps(data))
    assert main(args) == 1
    assert 'audit' in capsys.readouterr().err


@pytest.mark.parametrize('change', ['empty', 'blank', 'extra', 'missing', 'duplicate', 'order', 'rect',
                                    'page-type', 'bool-coordinate', 'text', 'chunk', 'source', 'refs',
                                    'extra-field', 'missing-text', 'null', 'array'])
def test_check_audit_rejects_modified_final_jsonl(audit_case, capsys, change):
    paths, args = audit_case
    rows = [json.loads(line) for line in paths['final'].read_text().splitlines()]
    if change in ('empty', 'blank'):
        rows = []
    elif change in ('extra', 'duplicate'):
        rows.append(rows[0] if change == 'duplicate' else dict(rows[0], chunk_id='extra'))
    elif change == 'missing':
        rows.pop()
    elif change == 'order':
        rows.reverse()
    elif change == 'rect':
        rows[0]['rect'][0] += 1e-7  # no PDF serialization tolerance applies to saved JSON
    elif change == 'page-type':
        rows[0]['page'] = 1.0
    elif change == 'bool-coordinate':
        rows[0]['rect'][0] = True
    elif change in ('text', 'chunk', 'source'):
        rows[0][{'text': 'text', 'chunk': 'chunk_id', 'source': 'source_ref'}[change]] = 'edited'
    elif change == 'refs':
        rows[0]['source_refs'] = [rows[0]['source_ref'], 'added']
    elif change == 'extra-field':
        rows[0]['extra'] = None
    elif change == 'missing-text':
        del rows[0]['text']
    else:
        rows[0] = None if change == 'null' else []
    paths['final'].write_text('\n'.join(json.dumps(r) for r in rows) + ('\n\t\n' if change == 'blank' else ''))
    assert main(args) == 1
    assert 'final regions' in capsys.readouterr().err


@pytest.mark.parametrize('change', ['unresolved', 'report-region', 'report-order', 'report-count', 'report-version',
                                    'report-null', 'reason', 'empty-original', 'original-text', 'pdf', 'pair'])
def test_check_audit_requires_original_authorization(audit_case, tmp_path, capsys, change):
    paths, args = audit_case
    data = json.loads(paths['review'].read_text())
    if change == 'unresolved':
        data['items'][-2]['resolution'] = None
    elif change == 'report-region':
        data['items'][0]['region']['text'] = 'edited'
    elif change == 'report-order':
        data['items'].reverse()
    elif change == 'report-count':
        data['summary']['total'] = 0
    elif change == 'report-version':
        data['schema_version'] = True
    elif change == 'report-null':
        data = None
    elif change == 'reason':
        data['items'][-2]['resolution']['reason'] = 'New authorization needs new artifacts'
    elif change == 'empty-original':
        paths['original'].write_bytes(b'')
    elif change == 'original-text':
        rows = read_regions(paths['original'])
        rows[0] = replace(rows[0], text='edited')
        paths['original'].write_text(regions_jsonl(rows))
    elif change == 'pdf':
        writer = PdfWriter(clone_from=open_pdf(paths['pdf']))
        writer.pages[0].rotate(90)
        altered = tmp_path / 'altered.pdf'
        writer.write(altered)
        args[1] = str(altered)
    else:
        # Internal agreement between the pair cannot override original decisions.
        audit = json.loads(paths['audit'].read_text())
        final = [json.loads(line) for line in paths['final'].read_text().splitlines()]
        audit['items'][0]['final_region']['text'] = final[0]['text'] = 'Joint tampering'
        paths['audit'].write_text(json.dumps(audit))
        paths['final'].write_text('\n'.join(json.dumps(row) for row in final))
    paths['review'].write_text(json.dumps(data))
    assert main(args) == 1
    assert 'Traceback' not in capsys.readouterr().err


@pytest.mark.parametrize('target', ['audit', 'review', 'original', 'final'])
def test_check_audit_rejects_duplicate_json_keys(audit_case, capsys, target):
    paths, args = audit_case
    content = paths[target].read_text()
    if target in ('audit', 'review'):
        content = content.replace('"schema_version": 1', '"schema_version": 9, "schema_version": 1', 1)
    else:
        content = content.replace('"page":1', '"page":9,"p\\u0061ge":1', 1)
    paths[target].write_text(content)
    assert main(args) == 1
    assert 'duplicate JSON key' in capsys.readouterr().err


@pytest.mark.parametrize('target', ['audit', 'final'])
@pytest.mark.parametrize('change', ['missing', 'malformed', 'utf8', 'oversized'])
def test_check_audit_artifact_input_errors(audit_case, capsys, monkeypatch, target, change):
    paths, args = audit_case
    if change == 'missing':
        paths[target].unlink()
    elif change == 'malformed':
        paths[target].write_bytes(b'{bad}\n')
    elif change == 'utf8':
        paths[target].write_bytes(b'\xff')
    else:
        # Shared bounded reader still applies to each new input path.
        monkeypatch.setattr('ragwrench.highlight.core.MAX_JSON_BYTES', 16384)
        monkeypatch.setattr('ragwrench.highlight.audit.MAX_JSON_BYTES', 16384)
        paths[target].write_bytes(b' ' * 16385)
    assert main(args) == 1
    assert 'Traceback' not in capsys.readouterr().err


def test_check_audit_all_skip_requires_matching_empty_set(audit_case, capsys):
    paths, args = audit_case
    data = json.loads(paths['review'].read_text())
    for item in data['items']:
        item['resolution'] = {'action': 'skip', 'reason': 'Remove citation'}
    paths['review'].write_text(json.dumps(data))
    assert main(['resolve', str(paths['pdf']), str(paths['original']), '--resolutions', str(paths['review']),
                 '--audit', str(paths['audit']), '-o', str(paths['final']), '--force']) == 0
    assert paths['final'].read_bytes() == b''
    assert main(args) == 0
    assert '"final_count": 0' in capsys.readouterr().out
    paths['final'].write_text(' \n\t\n')
    assert main(args) == 0
    paths['final'].write_text(regions_jsonl(read_regions(paths['original'])[:1]))
    assert main(args) == 1
    assert 'expected 0 ordered entries' in capsys.readouterr().err


def test_check_audit_formatting_provenance_and_annotation_only_identity(audit_case, tmp_path):
    paths, args = audit_case
    rows = read_regions(paths['original'])
    rows[0] = replace(rows[0], source_refs=(rows[0].source_ref, '参考 α'))
    paths['original'].write_text(regions_jsonl(rows))
    assert main(['preflight', str(paths['pdf']), str(paths['original']), '-o', str(paths['review']), '--force']) == 0
    data = decisions(paths['review'])
    data['items'][0]['resolution'] = {'action': 'accept', 'reason': 'Explicit ready decision'}
    paths['review'].write_text(json.dumps(data))
    assert main(['resolve', str(paths['pdf']), str(paths['original']), '--resolutions', str(paths['review']),
                 '--audit', str(paths['audit']), '-o', str(paths['final']), '--force']) == 0
    audit = json.loads(paths['audit'].read_text())
    audit['items'][0]['final_region']['rect'] = [int(n) for n in audit['items'][0]['final_region']['rect']]
    paths['audit'].write_text(json.dumps(audit, sort_keys=True, ensure_ascii=False))
    final = [json.loads(line) for line in paths['final'].read_text().splitlines()]
    final[0]['rect'] = [int(n) for n in final[0]['rect']]
    paths['final'].write_text('\n\n'.join(json.dumps(row) for row in final))
    assert main(args) == 0
    output = tmp_path / 'annotated.pdf'
    assert main(annotation_args(paths['pdf'], paths['original'], output, paths['review'])) == 0
    args[1] = str(output)
    assert main(args) == 0


@pytest.mark.parametrize('missing', ['--resolutions', '--audit', '--final-regions'])
def test_check_audit_required_inputs(audit_case, missing):
    _, args = audit_case
    i = args.index(missing)
    del args[i:i + 2]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_check_audit_has_no_write_options(audit_case, monkeypatch):
    import ragwrench.highlight.core as core
    paths, args = audit_case
    before = {p: p.read_bytes() for p in paths.values()}

    def fail_write(*args, **kwargs):
        pytest.fail('read-only checker attempted a write')

    monkeypatch.setattr(core, 'atomic_write', fail_write)
    assert main(args) == 0
    assert {p: p.read_bytes() for p in paths.values()} == before
    for extra in (['--force'], ['-o', str(paths['audit'])], ['--allow-empty']):
        with pytest.raises(SystemExit) as exc:
            main(args + extra)
        assert exc.value.code == 2


@pytest.mark.parametrize('change,category,path,message', [
    ('reason', 'artifact_mismatch', 'audit.items[5].reason', 'regenerate'),
    ('version', 'artifact_mismatch', 'audit.schema_version', 'regenerate'),
    ('counts', 'artifact_mismatch', 'audit.summary.final_count', 'regenerate'),
    ('order', 'artifact_mismatch', 'audit.items[0].annotation_id', 'regenerate'),
    ('coordinate', 'artifact_mismatch', 'audit.items[0].final_region.rect[0]', 'regenerate'),
    ('final-text', 'artifact_mismatch', 'final regions[0].text', 'regenerate'),
    ('final-empty', 'artifact_mismatch', 'final regions', 'regenerate'),
    ('final-extra', 'artifact_mismatch', 'final regions', 'regenerate'),
    ('final-missing', 'artifact_mismatch', 'final regions', 'regenerate'),
    ('duplicate-key', 'input_error', None, 'duplicate JSON key'),
    ('missing', 'input_error', None, 'regular file'),
    ('malformed', 'input_error', None, 'invalid JSON'),
    ('empty-original', 'input_error', None, 'expected 1..'),
    ('bad-pdf', 'input_error', None, 'PDF'),
    ('stale', 'replay_error', None, 'stale or modified'),
    ('unresolved', 'replay_error', None, 'unresolved'),
    ('decision', 'replay_error', None, 'resolution requires'),
    ('geometry', 'replay_error', None, 'replacement'),
])
def test_check_audit_json_failures(audit_case, capsys, change, category, path, message):
    paths, args = audit_case
    if change in ('reason', 'version', 'counts', 'order', 'coordinate'):
        data = json.loads(paths['audit'].read_text())
        if change == 'reason':
            data['items'][5]['reason'] = 'Changed reason'
        elif change == 'version':
            data['schema_version'] = True
        elif change == 'counts':
            data['summary']['final_count'] = 0
        elif change == 'order':
            data['items'].reverse()
        else:
            data['items'][0]['final_region']['rect'][0] += 1e-7
        paths['audit'].write_text(json.dumps(data))
    elif change.startswith('final-'):
        rows = [json.loads(line) for line in paths['final'].read_text().splitlines()]
        if change == 'final-text':
            rows[0]['text'] = 'Modified final text'
        elif change == 'final-empty':
            rows = []
        elif change == 'final-extra':
            rows.append(rows[0])
        else:
            rows.pop()
        paths['final'].write_text('\n'.join(json.dumps(row) for row in rows))
    elif change == 'duplicate-key':
        paths['audit'].write_text('{"schema_version":9,"schema_version":1}')
    elif change == 'missing':
        paths['final'].unlink()
    elif change == 'malformed':
        paths['review'].write_text('{bad}')
    elif change == 'empty-original':
        paths['original'].write_bytes(b'')
    elif change == 'bad-pdf':
        # Never alter the repository fixture.
        bad_pdf = paths['review'].parent / 'bad.pdf'
        bad_pdf.write_bytes(b'not a PDF')
        args[1] = str(bad_pdf)
        paths['pdf'] = bad_pdf
    else:
        data = json.loads(paths['review'].read_text())
        if change == 'stale':
            data['summary']['total'] = 0
        elif change == 'unresolved':
            data['items'][5]['resolution'] = None
        elif change == 'decision':
            data['items'][5]['resolution']['action'] = 'unknown'
        else:
            data['items'][4]['resolution']['rect'] = [1, 1, 2, 2]
        paths['review'].write_text(json.dumps(data))
    before = {p: p.read_bytes() for p in paths.values() if p.exists()}
    entries = sorted(paths['review'].parent.iterdir())
    capsys.readouterr()
    assert main(args + ['--json']) == 1
    captured = capsys.readouterr()
    assert captured.err == ''
    assert captured.out.count('\n') == 1
    result = json.loads(captured.out)
    assert result == {
        'format': 'citehl-check-audit-result', 'schema_version': 1,
        'ok': False, 'summary': None,
        'error': {'category': category, 'path': path, 'message': result['error']['message']},
    }
    assert message in result['error']['message']
    assert 'Traceback' not in result['error']['message']
    assert {p: p.read_bytes() for p in before} == before
    assert sorted(paths['review'].parent.iterdir()) == entries


def test_check_audit_json_all_skip(audit_case, capsys, monkeypatch):
    paths, args = audit_case
    data = json.loads(paths['review'].read_text())
    for item in data['items']:
        item['resolution'] = {'action': 'skip', 'reason': 'Reviewed omission'}
    paths['review'].write_text(json.dumps(data))
    assert main(['resolve', str(paths['pdf']), str(paths['original']), '--resolutions', str(paths['review']),
                 '--audit', str(paths['audit']), '-o', str(paths['final']), '--force']) == 0
    assert paths['final'].read_bytes() == b''

    def fail_write(*args, **kwargs):
        pytest.fail('JSON mode attempted a write')

    monkeypatch.setattr('ragwrench.highlight.core.atomic_write', fail_write)
    capsys.readouterr()
    assert main(args + ['--json']) == 0
    captured = capsys.readouterr()
    assert captured.err == ''
    assert json.loads(captured.out) == {
        'format': 'citehl-check-audit-result', 'schema_version': 1, 'ok': True,
        'summary': {'original_count': 7, 'final_count': 0, 'ready': 0,
                    'accept': 0, 'replace': 0, 'skip': 7}, 'error': None,
    }
    paths['final'].write_text(' \n\t\n')
    assert main(args + ['--json']) == 0
    capsys.readouterr()
    paths['final'].write_bytes(paths['original'].read_bytes())
    assert main(args + ['--json']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['summary'] is None
    assert result['error']['category'] == 'artifact_mismatch'
    assert result['error']['path'] == 'final regions'


@pytest.mark.parametrize('change', ['missing', 'unknown', 'output', 'force', 'other-command'])
def test_check_audit_json_usage_errors_remain_argparse(audit_case, capsys, change):
    _, args = audit_case
    if change == 'missing':
        i = args.index('--audit')
        del args[i:i + 2]
    elif change == 'unknown':
        args += ['--unknown']
    elif change == 'output':
        args += ['-o', 'unwritten.json']
    elif change == 'force':
        args += ['--force']
    else:
        args = ['manifest', args[1], args[2], '-o', 'unwritten.json']
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(args + ['--json'])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'error:' in captured.err
