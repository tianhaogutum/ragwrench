"""Saved manifests are derived artifacts; neither geometry nor review authorization."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import ArrayObject, FloatObject, NameObject, NumberObject, TextStringObject

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import open_pdf, read_regions, regions_jsonl
from ragwrench.highlight.pdf import add_highlight


@pytest.fixture
def manifest_case(examples, tmp_path):
    original = read_regions(examples / "regions.jsonl")
    regions = [replace(original[1], chunk_id="summary/中文"),
               replace(original[0], chunk_id="summary/中文", source_refs=("#/texts/0", "merged/中文")),
               replace(original[0], chunk_id="other")]
    path = tmp_path / "regions.jsonl"
    path.write_text(regions_jsonl(regions), encoding="utf-8")
    pdf = tmp_path / "readable % #四.pdf"
    assert main(["annotate", str(examples / "sample.pdf"), str(path), "-o", str(pdf)]) == 0
    return ["check-manifest", str(pdf), str(path), "--manifest", str(tmp_path / "citations.json")]


def check(capsys, args, code=0, message=None, annotations=3, chunks=2):
    directory = Path(args[1]).parent
    files = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    entries = sorted(directory.rglob("*"))
    capsys.readouterr()
    assert main(args) == code
    captured = capsys.readouterr()
    if code == 0:
        assert captured.err == ""
        assert captured.out == f'OK check-manifest {{"annotations": {annotations}, "chunks": {chunks}}}\n'
    else:
        assert captured.out == ""  # No success or partial counts on any failure.
        assert message in captured.err and "Traceback" not in captured.err
    assert sorted(directory.rglob("*")) == entries
    assert {p: p.read_bytes() for p in files} == files


def test_check_manifest_success(manifest_case, capsys):
    check(capsys, manifest_case + ["--text-check", "required"])


def test_writer_contract_and_equivalent_json(manifest_case, capsys):
    path = Path(manifest_case[4])
    data = json.loads(path.read_text())
    assert set(data) == {"schema_version", "coordinate_space", "pdf", "chunks"}
    assert type(data["schema_version"]) is int and data["schema_version"] == 1
    assert data["coordinate_space"] == "unrotated-pdf-user-space"
    assert data["pdf"] == "readable % #四.pdf"
    summary = data["chunks"]["summary/中文"]
    assert set(summary) == {"pages", "rects", "links"}
    assert summary["pages"] == [1, 2]
    assert summary["links"] == ["readable%20%25%20%23%E5%9B%9B.pdf#page=1",
                                 "readable%20%25%20%23%E5%9B%9B.pdf#page=2"]
    assert [r["page"] for r in summary["rects"]] == [2, 1]  # Input order, not page order.
    assert summary["rects"][1]["source_refs"] == ["#/texts/0", "merged/中文"]
    assert summary["rects"][0]["source_refs"] == [summary["rects"][0]["source_ref"]]
    for chunk in data["chunks"].values():
        for rect in chunk["rects"]:
            assert set(rect) == {"page", "rect", "source_ref", "source_refs", "annotation_id"}
            assert rect["annotation_id"].startswith("citehl:v1:") and len(rect["annotation_id"]) == 74
            rect["rect"] = [int(v) for v in rect["rect"]]
    # JSON whitespace/object order and equivalent coordinate number spellings may vary.
    data = {k: data[k] for k in reversed(data)}
    data["chunks"] = {k: data["chunks"][k] for k in reversed(data["chunks"])}
    path.write_text(json.dumps(data, ensure_ascii=True, separators=(",", ":")))
    check(capsys, manifest_case)


@pytest.mark.parametrize("path,value", [
    (["schema_version"], True), (["schema_version"], 1.0), (["schema_version"], 2),
    (["coordinate_space"], "display"), (["pdf"], "other.pdf"),
    (["pdf"], "readable%20%25%20%23%E5%9B%9B.pdf"),
    (["chunks", "summary/中文", "pages"], [2, 1]),
    (["chunks", "summary/中文", "pages"], [1, 1, 2]),
    (["chunks", "summary/中文", "pages"], [1.0, 2]),
    (["chunks", "summary/中文", "pages"], [True, 2]),
    (["chunks", "summary/中文", "rects", 0, "page"], 1),
    (["chunks", "summary/中文", "rects", 0, "page"], 2.0),
    (["chunks", "summary/中文", "rects", 0, "rect"], [72, 614, 299.999, 628]),
    (["chunks", "summary/中文", "rects", 0, "rect"], [True, 0, 1, 2]),
    (["chunks", "summary/中文", "rects", 0, "rect"], ["72", 0, 1, 2]),
    (["chunks", "summary/中文", "rects", 0, "rect"], [1e300, 0, 1, 2]),
    (["chunks", "summary/中文", "rects", 0, "rect"], [1, 2, 3]),
    (["chunks", "summary/中文", "rects", 0, "source_ref"], "forged"),
    (["chunks", "summary/中文", "rects", 1, "source_refs"], ["#/texts/0"]),
    (["chunks", "summary/中文", "rects", 1, "source_refs"], ["merged/中文", "#/texts/0"]),
    (["chunks", "summary/中文", "rects", 0, "annotation_id"], "citehl:v1:forged"),
    (["chunks", "summary/中文", "links"], ["readable % #四.pdf#page=1", "readable % #四.pdf#page=2"]),
    (["chunks", "summary/中文", "links"], ["elsewhere.pdf#page=1", "elsewhere.pdf#page=2"]),
    (["chunks", "summary/中文", "links"], ["readable%20%25%20%23%E5%9B%9B.pdf#page=2",
                                          "readable%20%25%20%23%E5%9B%9B.pdf#page=1"]),
])
def test_manifest_values(manifest_case, capsys, path, value):
    artifact = Path(manifest_case[4])
    data = json.loads(artifact.read_text())
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    artifact.write_text(json.dumps(data))
    check(capsys, manifest_case, 1, "manifest[")


@pytest.mark.parametrize("change", ["root-array", "chunks-array", "missing-top", "extra-top",
    "missing-chunk", "extra-chunk", "renamed-chunk", "missing-chunk-field", "extra-chunk-field",
    "missing-rect-field", "extra-rect-field", "missing-rect", "extra-rect", "rect-order", "missing-link"])
def test_manifest_shape_and_order(manifest_case, capsys, change):
    artifact = Path(manifest_case[4])
    data = json.loads(artifact.read_text())
    chunk = data["chunks"]["summary/中文"]
    if change == "root-array": data = []
    elif change == "chunks-array": data["chunks"] = []
    elif change == "missing-top": del data["pdf"]
    elif change == "extra-top": data["annotations"] = 3
    elif change == "missing-chunk": del data["chunks"]["other"]
    elif change == "extra-chunk": data["chunks"]["forged"] = chunk
    elif change == "renamed-chunk": data["chunks"]["forged"] = data["chunks"].pop("other")
    elif change == "missing-chunk-field": del chunk["links"]
    elif change == "extra-chunk-field": chunk["count"] = 2
    elif change == "missing-rect-field": del chunk["rects"][0]["source_refs"]
    elif change == "extra-rect-field": chunk["rects"][0]["text"] = "forged"
    elif change == "missing-rect": chunk["rects"].pop()
    elif change == "extra-rect": chunk["rects"].append(chunk["rects"][0])
    elif change == "rect-order": chunk["rects"].reverse()
    elif change == "missing-link": chunk["links"].pop()
    artifact.write_text(json.dumps(data))
    check(capsys, manifest_case, 1, "regenerate with ragwrench highlight manifest")


@pytest.mark.parametrize("target", ["manifest", "regions", "pdf"])
@pytest.mark.parametrize("change", ["missing", "malformed", "oversized"])
def test_input_errors(manifest_case, capsys, monkeypatch, target, change):
    index = {"manifest": 4, "regions": 2, "pdf": 1}[target]
    path = Path(manifest_case[index])
    if change == "missing": path.unlink()
    elif change == "malformed": path.write_bytes(b'\xffbroken')
    else:
        monkeypatch.setattr("ragwrench.highlight.core.MAX_PDF_BYTES" if target == "pdf" else "ragwrench.highlight.core.MAX_JSON_BYTES", 10)
    check(capsys, manifest_case, 1, "ragwrench highlight:")


@pytest.mark.parametrize("payload", [
    '{"schema_version":0,"schema_versio\\u006e":1}',
    '{"chunks":{"same":{},"same":{}}}',
    '{"chunks":{"x":{"rects":[{"page":2,"page":1}]}}}',
    '{"chunks":NaN}', '{"chunks":Infinity}',
])
def test_duplicate_keys_and_constants(manifest_case, capsys, payload):
    Path(manifest_case[4]).write_text(payload)
    check(capsys, manifest_case, 1, "invalid JSON")


@pytest.mark.parametrize("change,message", [
    ("missing", "missing annotation"), ("extra", "unexpected citehl annotation"),
    ("duplicate", "duplicate annotation"), ("rect", "/Rect mismatch"),
    ("contents", "/Contents mismatch"), ("hidden", "hidden/invisible"),
    ("metadata", "invalid citehl region metadata"),
])
def test_pdf_must_independently_verify(manifest_case, capsys, change, message):
    path = Path(manifest_case[1])
    writer = PdfWriter(clone_from=open_pdf(path))
    page = writer.pages[0]
    if change == "missing": page[NameObject("/Annots")] = ArrayObject()
    elif change == "extra": add_highlight(writer, replace(read_regions(Path(manifest_case[2]))[0], chunk_id="extra"))
    elif change == "duplicate": page["/Annots"].append(page["/Annots"][0])
    else:
        key, value = {"rect": ("/Rect", ArrayObject([FloatObject(v) for v in (72, 686, 299, 700)])),
                      "contents": ("/Contents", TextStringObject("forged")),
                      "hidden": ("/F", NumberObject(2)),
                      "metadata": ("/CiteHL", TextStringObject("{}"))}[change]
        page["/Annots"][0].get_object()[NameObject(key)] = value
    writer.write(path)
    check(capsys, manifest_case, 1, message)


@pytest.mark.parametrize("change", ["geometry", "text", "provenance", "subset"])
def test_jointly_edited_regions_and_manifest_do_not_authorize_pdf(manifest_case, capsys, change):
    path, saved = Path(manifest_case[2]), Path(manifest_case[4])
    regions = read_regions(path)
    if change == "geometry": regions[0] = replace(regions[0], rect=(1., 1., 10., 10.))
    elif change == "text": regions[0] = replace(regions[0], text="modified")
    elif change == "provenance": regions[0] = replace(regions[0], source_ref="forged", source_refs=())
    else: regions.pop()
    path.write_text(regions_jsonl(regions))
    assert main(["manifest", manifest_case[1], str(path), "-o", str(saved), "--force"]) == 0
    check(capsys, manifest_case, 1, "verification failed")


@pytest.fixture
def reviewed_case(examples, tmp_path):
    pdf, regions, review = examples / "review-rotated.pdf", tmp_path / "original.jsonl", tmp_path / "review.json"
    assert main(["normalize", str(examples / "review-rotated.unstructured.json"), "--format", "unstructured",
                 "--pdf", str(pdf), "-o", str(regions)]) == 0
    assert main(["preflight", str(pdf), str(regions), "-o", str(review)]) == 0
    data = json.loads(review.read_text())
    for item in data["items"]:
        ref = item["region"]["source_ref"]
        if ref == "ambiguous":
            item["resolution"] = {"action": "replace", "page": 2, "rect": [220, 120, 290, 140], "reason": "chosen occurrence"}
        elif ref == "blank": item["resolution"] = {"action": "accept", "reason": "checked picture"}
        elif ref == "mismatch": item["resolution"] = {"action": "skip", "reason": "wrong text"}
    review.write_text(json.dumps(data))
    output = tmp_path / "reviewed.pdf"
    assert main(["annotate", str(pdf), str(regions), "--resolutions", str(review), "-o", str(output)]) == 0
    return ["check-manifest", str(output), str(regions), "--manifest", str(tmp_path / "citations.json"), "--resolutions", str(review)]


def test_reviewed_and_materialized_sets(reviewed_case, tmp_path, capsys):
    check(capsys, reviewed_case, annotations=6, chunks=6)
    check(capsys, reviewed_case + ["--text-check", "off"], annotations=6, chunks=6)
    check(capsys, reviewed_case + ["--text-check", "required"], 1, "text unavailable")
    final = tmp_path / "final.jsonl"
    assert main(["resolve", reviewed_case[1], reviewed_case[2], "--resolutions", reviewed_case[6], "-o", str(final)]) == 0
    args = reviewed_case[:5]
    args[2] = str(final)
    check(capsys, args, annotations=6, chunks=6)


@pytest.mark.parametrize("change,message", [
    ("unresolved", "unresolved"), ("stale", "stale or modified"), ("evidence", "modified"),
    ("version", "stale or modified"), ("decision", "action"), ("duplicate", "duplicate JSON key"),
    ("no-report", "verification failed"), ("empty-original", "expected 1.."),
])
def test_review_cannot_be_authorized_by_manifest(reviewed_case, capsys, change, message):
    args = reviewed_case.copy()
    path = Path(args[6])
    data = json.loads(path.read_text())
    if change == "unresolved":
        next(i for i in data["items"] if i["region"]["source_ref"] == "blank")["resolution"] = None
    elif change == "stale": data["document"]["sha256"] = "stale"
    elif change == "evidence": data["summary"]["ready"] = 0
    elif change == "version": data["schema_version"] = 2
    elif change == "decision": data["items"][0]["resolution"] = {"action": "bogus", "reason": "forged"}
    elif change == "no-report": args = args[:5]
    elif change == "empty-original":
        Path(args[2]).write_text("")
        args.append("--allow-empty")
    if change == "duplicate": path.write_text('{"items":[],"items":[]}')
    else: path.write_text(json.dumps(data))
    check(capsys, args, 1, message)


def test_all_skip_and_explicit_empty_contract(reviewed_case, tmp_path, capsys):
    review = Path(reviewed_case[6])
    data = json.loads(review.read_text())
    for item in data["items"]: item["resolution"] = {"action": "skip", "reason": "omit"}
    review.write_text(json.dumps(data))
    assert main(["annotate", reviewed_case[1], reviewed_case[2], "--resolutions", str(review),
                 "-o", reviewed_case[1], "--force"]) == 1  # Existing input protection.
    cleared = tmp_path / "cleared.pdf"
    assert main(["annotate", reviewed_case[1], reviewed_case[2], "--resolutions", str(review),
                 "-o", str(cleared), "--force"]) == 0
    args = reviewed_case.copy()
    args[1] = str(cleared)
    check(capsys, args, annotations=0, chunks=0)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n \n")
    args = args[:5]
    args[2] = str(empty)
    check(capsys, args, 1, "expected 1..")
    check(capsys, args + ["--allow-empty"], annotations=0, chunks=0)
    # Empty expected manifest cannot conceal nonempty PDF annotations.
    args[1] = reviewed_case[1]
    check(capsys, args + ["--allow-empty"], 1, "unexpected citehl annotation")
    # Empty PDF/regions cannot conceal a nonempty saved manifest.
    args[1] = str(cleared)
    data = json.loads(Path(args[4]).read_text())
    data["chunks"]["extra"] = {"pages": [], "rects": [], "links": []}
    Path(args[4]).write_text(json.dumps(data))
    check(capsys, args + ["--allow-empty"], 1, "object fields differ")


def test_pdf_tolerance_does_not_relax_manifest(manifest_case, capsys):
    pdf = Path(manifest_case[1])
    writer = PdfWriter(clone_from=open_pdf(pdf))
    rect = writer.pages[0]["/Annots"][0].get_object()["/Rect"]
    rect[0] = FloatObject(float(rect[0]) + .005)
    writer.write(pdf)
    check(capsys, manifest_case)
    check(capsys, manifest_case + ["--tolerance", ".001"], 1, "/Rect mismatch")
    data = json.loads(Path(manifest_case[4]).read_text())
    data["chunks"]["other"]["rects"][0]["rect"][0] += .005
    Path(manifest_case[4]).write_text(json.dumps(data))
    check(capsys, manifest_case + ["--tolerance", "100"], 1, "numeric value differs")


def test_foreign_annotations_and_no_writes(manifest_case, capsys, monkeypatch):
    from pypdf.generic import DictionaryObject
    pdf = Path(manifest_case[1])
    writer = PdfWriter(clone_from=open_pdf(pdf))
    writer.add_annotation(0, DictionaryObject({NameObject("/Subtype"): NameObject("/Text"),
        NameObject("/Rect"): ArrayObject([NumberObject(n) for n in (1, 1, 10, 10)]),
        NameObject("/NM"): TextStringObject("foreign")}))
    writer.write(pdf)
    def forbidden(*args, **kwargs): raise AssertionError("checker must not write")
    monkeypatch.setattr("ragwrench.highlight.cli.write_text", forbidden)
    monkeypatch.setattr("ragwrench.highlight.pdf.atomic_write", forbidden)
    monkeypatch.setattr("ragwrench.highlight.core.atomic_write", forbidden)
    check(capsys, manifest_case)


def test_manifest_location_contract(manifest_case, capsys):
    saved = Path(manifest_case[4])
    nested = saved.parent / "nested"
    nested.mkdir()
    moved = nested / saved.name
    moved.write_bytes(saved.read_bytes())
    args = manifest_case.copy()
    args[4] = str(moved)
    check(capsys, args, 1, "share a directory")
    # Resolved directory aliases follow the writer's existing contract.
    alias = saved.parent / "alias"
    alias.symlink_to(saved.parent, target_is_directory=True)
    args[4] = str(alias / saved.name)
    check(capsys, args)


@pytest.mark.parametrize("change", ["pdf", "regions", "manifest", "manifest-path", "output", "force", "unknown", "nan", "negative", "text-mode"])
def test_usage_errors(manifest_case, capsys, change):
    args = manifest_case.copy()
    if change == "pdf": del args[1]
    elif change == "regions": del args[2]
    elif change == "manifest": args = args[:3]
    elif change == "manifest-path": args.pop()
    else: args += {"output": ["-o", "never"], "force": ["--force"], "unknown": ["--unknown"],
                   "nan": ["--tolerance", "nan"], "negative": ["--tolerance", "-1"],
                   "text-mode": ["--text-check", "unknown"]}[change]
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc: main(args)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "error:" in captured.err
