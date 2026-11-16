"""Versioned verification results use independently checked counts, never partial ones."""
import json
from pathlib import Path

import pytest

from ragwrench.highlight.cli import main


@pytest.fixture
def verified_case(examples, tmp_path):
    pdf, regions = tmp_path / "annotated.pdf", examples / "regions.jsonl"
    assert main(["annotate", str(examples / "sample.pdf"), str(regions), "-o", str(pdf)]) == 0
    return pdf, regions


def test_verify_json_success(verified_case, capsys):
    pdf, regions = verified_case
    before = {p: p.read_bytes() for p in (pdf, regions)}
    entries = sorted(pdf.parent.iterdir())
    capsys.readouterr()
    assert main(["verify", str(pdf), str(regions), "--text-check", "required", "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == {
        "format": "citehl-verify-result", "schema_version": 1, "ok": True,
        "summary": {"annotations": 2, "text_checked": 2, "text_skipped": 0,
                    "text_check": "required", "notes": []}, "error": None,
    }
    assert {p: p.read_bytes() for p in before} == before
    assert sorted(pdf.parent.iterdir()) == entries


def json_result(capsys, args, code, category=None, message=None):
    """Assert the exact envelope and streams, including absence of partial counts."""
    paths = [p for arg in args for p in [Path(arg)] if p.is_file()]
    before = {p: p.read_bytes() for p in paths}
    capsys.readouterr()
    assert main(args + ["--json"]) == code
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    result = json.loads(captured.out)
    assert type(result["schema_version"]) is int
    assert type(result["ok"]) is bool
    assert set(result) == {"format", "schema_version", "ok", "summary", "error"}
    assert result["format"] == "citehl-verify-result" and result["schema_version"] == 1
    assert result["ok"] is (code == 0)
    if code:
        assert result["summary"] is None
        assert result["error"] == {"category": category, "message": result["error"]["message"]}
        assert isinstance(result["error"]["message"], str)
        assert message in result["error"]["message"]
        assert "Traceback" not in result["error"]["message"]
    else:
        assert result["error"] is None
        summary = result["summary"]
        assert set(summary) == {"annotations", "text_checked", "text_skipped", "text_check", "notes"}
        assert all(type(summary[k]) is int and summary[k] >= 0
                   for k in ("annotations", "text_checked", "text_skipped"))
        assert isinstance(summary["notes"], list)
        assert all(isinstance(note, str) for note in summary["notes"])
    assert {p: p.read_bytes() for p in before} == before
    return result


@pytest.mark.parametrize("change,message", [
    ("missing", "missing annotation"), ("extra", "unexpected citehl annotation"),
    ("duplicate", "duplicate annotation"), ("rect", "/Rect mismatch"),
    ("quad", "/QuadPoints mismatch"), ("contents", "/Contents mismatch"),
    ("subtype", "wrong page or subtype"), ("moved", "wrong page or subtype"),
    ("hidden", "hidden/invisible"), ("opacity", "hidden/invisible"),
    ("metadata", "invalid citehl region metadata"), ("renamed", "/NM mismatch"),
])
def test_verify_json_annotation_failures(verified_case, tmp_path, capsys, change, message):
    from dataclasses import replace
    from pypdf import PdfWriter
    from pypdf.generic import ArrayObject, FloatObject, NameObject, NumberObject, TextStringObject
    from ragwrench.highlight.core import open_pdf, read_regions
    from ragwrench.highlight.pdf import add_highlight

    pdf, regions = verified_case
    writer = PdfWriter(clone_from=open_pdf(pdf))
    page = writer.pages[0]
    annotation = page["/Annots"][0].get_object()
    if change == "missing":
        page[NameObject("/Annots")] = ArrayObject()
    elif change == "extra":
        add_highlight(writer, replace(read_regions(regions)[0], chunk_id="extra"))
    elif change == "duplicate":
        page["/Annots"].append(page["/Annots"][0])
    elif change == "moved":
        writer.pages[1]["/Annots"].append(page["/Annots"].pop())
    else:
        key, value = {
            "rect": ("/Rect", ArrayObject([FloatObject(v) for v in (72, 686, 299, 700)])),
            "quad": ("/QuadPoints", ArrayObject([NumberObject(1)] * 8)),
            "contents": ("/Contents", TextStringObject("modified\n中文")),
            "subtype": ("/Subtype", NameObject("/Square")),
            "hidden": ("/F", NumberObject(2)), "opacity": ("/CA", FloatObject(0)),
            "metadata": ("/CiteHL", TextStringObject("{}")),
            "renamed": ("/NM", TextStringObject("foreign")),
        }[change]
        annotation[NameObject(key)] = value
    pdf = tmp_path / "tampered.pdf"
    writer.write(pdf)
    entries = sorted(tmp_path.iterdir())
    json_result(capsys, ["verify", str(pdf), str(regions)], 1, "verification_error", message)
    assert sorted(tmp_path.iterdir()) == entries


@pytest.mark.parametrize("change,message", [
    ("missing-pdf", "input must be a regular file"), ("invalid-pdf", "PDF"),
    ("missing-regions", "input must be a regular file"), ("invalid-regions", "invalid JSON"),
    ("duplicate-key", "duplicate JSON key"), ("duplicate-region", "duplicate"),
    ("empty", "expected"), ("bounds", "visible page"), ("page", "outside PDF"),
    ("missing-report", "input must be a regular file"), ("invalid-report", "invalid JSON"),
    ("duplicate-report-key", "duplicate JSON key"),
])
def test_verify_json_input_failures(verified_case, tmp_path, capsys, change, message):
    from dataclasses import replace
    from ragwrench.highlight.core import read_regions, regions_jsonl

    pdf, original = verified_case
    regions = tmp_path / "input.jsonl"
    regions.write_bytes(original.read_bytes())
    args = ["verify", str(pdf), str(regions)]
    if change == "missing-pdf":
        args[1] = str(tmp_path / "absent.pdf")
    elif change == "invalid-pdf":
        pdf.write_bytes(b"not a PDF")
    elif change == "missing-regions":
        regions.unlink()
    elif change == "invalid-regions":
        regions.write_text("{")
    elif change == "duplicate-key":
        regions.write_text('{"page":1,"page":2}\n')
    elif change == "duplicate-region":
        regions.write_bytes(original.read_bytes() * 2)
    elif change == "empty":
        regions.write_bytes(b"")
    elif change in ("bounds", "page"):
        r = read_regions(original)[0]
        r = replace(r, rect=(0, 0, 1000, 1000)) if change == "bounds" else replace(r, page=9)
        regions.write_text(regions_jsonl([r]))
    else:
        report = tmp_path / "report.json"
        args += ["--resolutions", str(report)]
        if change == "invalid-report":
            report.write_text("{")
        elif change == "duplicate-report-key":
            report.write_text('{"items":[],"items":[]}')
    json_result(capsys, args, 1, "input_error", message)


@pytest.fixture
def reviewed_case(examples, tmp_path):
    source = examples / "review-rotated.pdf"
    regions, review, pdf = (tmp_path / name for name in ("original.jsonl", "review.json", "reviewed.pdf"))
    assert main(["normalize", "--format", "unstructured", str(examples / "review-rotated.unstructured.json"),
                 "--pdf", str(source), "-o", str(regions)]) == 0
    assert main(["preflight", str(source), str(regions), "-o", str(review)]) == 0
    data = json.loads(review.read_text())
    for item in data["items"]:
        ref = item["region"]["source_ref"]
        if ref == "ambiguous":
            item["resolution"] = {"action": "replace", "page": 2, "rect": [220, 120, 290, 140],
                                  "reason": "Select the second occurrence"}
        elif ref == "blank":
            item["resolution"] = {"action": "accept", "reason": "Manual review — 非文字"}
        elif ref == "mismatch":
            item["resolution"] = {"action": "skip", "reason": "Incorrect text"}
    review.write_text(json.dumps(data))
    assert main(["annotate", str(source), str(regions), "--resolutions", str(review), "-o", str(pdf)]) == 0
    return pdf, regions, review


@pytest.mark.parametrize("mode,code,checked,skipped", [
    ("best-effort", 0, 5, 1), ("off", 0, 0, 0), ("required", 1, None, None),
])
def test_verify_json_reviewed_text_modes(reviewed_case, capsys, mode, code, checked, skipped):
    pdf, regions, review = reviewed_case
    result = json_result(capsys, ["verify", str(pdf), str(regions), "--resolutions", str(review),
                                 "--text-check", mode], code, "verification_error", "text unavailable")
    if not code:
        assert result["summary"] == {"annotations": 6, "text_checked": checked,
                                     "text_skipped": skipped, "text_check": mode, "notes": []}


@pytest.mark.parametrize("change,message", [
    ("unresolved", "unresolved preflight"), ("action", "action accept/skip/replace"),
    ("reason", "resolution reason"), ("replacement", "replacement still requires review"),
    ("version", "stale or modified"), ("duplicate", "duplicate resolution"),
    ("missing", "stale or modified"), ("stale-pdf", "stale or modified"),
    ("stale-regions", "stale or modified"), ("null", "unresolved preflight"),
])
def test_verify_json_replay_failures(reviewed_case, capsys, change, message):
    from pypdf import PdfWriter
    from ragwrench.highlight.core import open_pdf

    pdf, regions, review = reviewed_case
    data = json.loads(review.read_text())
    if change == "unresolved":
        data["items"][-2]["resolution"] = None
    elif change == "action":
        data["items"][-1]["resolution"]["action"] = "guess"
    elif change == "reason":
        data["items"][-1]["resolution"]["reason"] = " "
    elif change == "replacement":
        data["items"][-3]["resolution"]["rect"] = [80, 120, 290, 140]
    elif change == "version":
        data["schema_version"] = True
    elif change == "duplicate":
        data["items"].append(data["items"][0])
    elif change == "missing":
        data["items"].pop()
    elif change == "stale-pdf":
        writer = PdfWriter(clone_from=open_pdf(pdf))
        writer.pages[0].rotate(90)
        writer.write(pdf)
    elif change == "stale-regions":
        rows = [json.loads(line) for line in regions.read_text().splitlines()]
        rows[0]["text"] = "Changed"
        regions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif change == "null":
        data = None
        message = "preflight:"  # Existing null-report semantics do not authorize flagged entries.
    review.write_text(json.dumps(data))
    json_result(capsys, ["verify", str(pdf), str(regions), "--resolutions", str(review)],
                1, "replay_error", message)


@pytest.mark.parametrize("mode", ["best-effort", "required", "off"])
def test_verify_json_empty_sets(reviewed_case, tmp_path, capsys, monkeypatch, mode):
    pdf, regions, review = reviewed_case
    data = json.loads(review.read_text())
    for item in data["items"]:
        item["resolution"] = {"action": "skip", "reason": "Reviewed omission"}
    review.write_text(json.dumps(data))
    cleared, empty = tmp_path / "cleared.pdf", tmp_path / "empty.jsonl"
    assert main(["annotate", str(pdf), str(regions), "--resolutions", str(review),
                 "-o", str(cleared), "--force"]) == 0
    empty.write_bytes(b"")
    entries = sorted(tmp_path.iterdir())

    def fail_write(*args, **kwargs):
        pytest.fail("verification attempted a write")

    monkeypatch.setattr("ragwrench.highlight.core.atomic_write", fail_write)
    bases = [["verify", str(cleared), str(regions), "--resolutions", str(review)],
             ["verify", str(cleared), str(empty), "--allow-empty"]]
    for args in bases:
        result = json_result(capsys, args + ["--text-check", mode], 0)
        assert result["summary"] == {"annotations": 0, "text_checked": 0, "text_skipped": 0,
                                     "text_check": mode, "notes": []}
    empty.write_text(" \n\t\n")
    json_result(capsys, bases[1], 0)
    json_result(capsys, ["verify", str(pdf), str(empty), "--allow-empty"],
                1, "verification_error", "unexpected citehl annotation")
    json_result(capsys, ["verify", str(cleared), str(empty)], 1, "input_error", "expected")
    json_result(capsys, ["verify", str(cleared), str(empty), "--allow-empty", "--resolutions", str(review)],
                1, "input_error", "expected")
    assert sorted(tmp_path.iterdir()) == entries


def test_verify_json_extraction_unavailable(verified_case, capsys, monkeypatch):
    def unavailable(page):
        raise ValueError("unsupported text")

    monkeypatch.setattr("ragwrench.highlight.pdf.text_fragments", unavailable)
    pdf, regions = verified_case
    result = json_result(capsys, ["verify", str(pdf), str(regions)], 0)
    assert result["summary"] == {
        "annotations": 2, "text_checked": 0, "text_skipped": 2, "text_check": "best-effort",
        "notes": [f"page {p}: text extraction unavailable (ValueError)" for p in (1, 2)],
    }
    json_result(capsys, ["verify", str(pdf), str(regions), "--text-check", "required"],
                1, "verification_error", "text unavailable")
    result = json_result(capsys, ["verify", str(pdf), str(regions), "--text-check", "off"], 0)
    assert result["summary"] == {"annotations": 2, "text_checked": 0, "text_skipped": 0,
                                 "text_check": "off", "notes": []}


@pytest.mark.parametrize("mode,code", [("required", 1), ("best-effort", 1), ("off", 0)])
def test_verify_json_text_mismatch(examples, tmp_path, capsys, mode, code):
    from dataclasses import replace
    from ragwrench.highlight.core import open_pdf, read_regions, regions_jsonl
    from ragwrench.highlight.pdf import annotate

    regions = [replace(read_regions(examples / "regions.jsonl")[0], text="elephants sunshine")]
    pdf, path = tmp_path / "mismatch.pdf", tmp_path / "mismatch.jsonl"
    annotate(open_pdf(examples / "sample.pdf"), regions, pdf)
    path.write_text(regions_jsonl(regions))
    result = json_result(capsys, ["verify", str(pdf), str(path), "--text-check", mode],
                         code, "verification_error", "text mismatch")
    if not code:
        assert result["summary"]["annotations"] == 1
        assert result["summary"]["text_checked"] == result["summary"]["text_skipped"] == 0


def test_verify_json_tolerance_and_default_output(verified_case, capsys):
    from pypdf import PdfWriter
    from pypdf.generic import FloatObject
    from ragwrench.highlight.core import open_pdf

    pdf, regions = verified_case
    capsys.readouterr()
    args = ["verify", str(pdf), str(regions)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == 'OK 2/2 {"annotations": 2, "text_checked": 2, "text_skipped": 0, "text_check": "best-effort", "notes": []}\n'
    writer = PdfWriter(clone_from=open_pdf(pdf))
    rect = writer.pages[0]["/Annots"][0].get_object()["/Rect"]
    rect[0] = FloatObject(float(rect[0]) + 0.005)
    writer.write(pdf)
    json_result(capsys, args, 0)
    result = json_result(capsys, args + ["--tolerance", "0.001"], 1, "verification_error", "/Rect mismatch")
    assert main(args + ["--tolerance", "0.001"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ragwrench highlight: " + result["error"]["message"] + "\n"


@pytest.mark.parametrize("arguments", [
    ["verify"], ["verify", "a"], ["verify", "a", "b", "--unknown"],
    ["verify", "a", "b", "-o", "unwritten.json"], ["verify", "a", "b", "--force"],
    ["verify", "a", "b", "--resolutions"], ["verify", "a", "b", "--text-check", "guess"],
    ["verify", "a", "b", "--tolerance", "nan"], ["verify", "a", "b", "--tolerance", "-1"],
    ["manifest", "a", "b", "-o", "unwritten.json"],
])
def test_verify_json_usage_errors(capsys, arguments):
    with pytest.raises(SystemExit) as exc:
        main(arguments + ["--json"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
