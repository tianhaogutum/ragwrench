"""Saved sidecars must match an independently verified current PDF export."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import open_pdf, read_regions, regions_jsonl
from ragwrench.highlight.pdf import annotate
from ragwrench.highlight.sidecar import export_sidecar


@pytest.fixture
def sidecar_case(examples, tmp_path):
    original = read_regions(examples / "regions.jsonl")
    # Deliberately put page 2 first; PDF traversal and JSONL orders differ.
    regions = [original[1], replace(original[0], source_refs=("#/texts/0", "merged/中文")),
               replace(original[0], chunk_id="other")]
    region_file = tmp_path / "final.jsonl"
    region_file.write_text(regions_jsonl(regions), encoding="utf-8")
    pdf = tmp_path / "readable % #四.pdf"
    annotate(open_pdf(examples / "sample.pdf"), regions, pdf)
    sidecar = tmp_path / "notes.json"
    sidecar.write_text(json.dumps(export_sidecar(open_pdf(pdf))), encoding="utf-8")
    return ["check-sidecar", str(pdf), "--sidecar", str(sidecar)]


def check(capsys, args, code=0, message=None, annotations=3, pages=2):
    directory = Path(args[1]).parent
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    entries = sorted(directory.rglob("*"))
    capsys.readouterr()
    assert main(args) == code
    captured = capsys.readouterr()
    if code == 0:
        assert captured.err == ""
        assert captured.out == f'OK check-sidecar {{"annotations": {annotations}, "pages": {pages}}}\n'
    else:
        assert captured.out == ""
        assert message in captured.err and "Traceback" not in captured.err
    assert sorted(directory.rglob("*")) == entries
    assert {p: p.read_bytes() for p in before} == before


def test_check_sidecar_success(sidecar_case, capsys):
    check(capsys, sidecar_case)


def saved_data(args):
    path = Path(args[3])
    return path, json.loads(path.read_text())


def write_saved(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_schema_order_and_equivalent_json(sidecar_case, capsys):
    path, data = saved_data(sidecar_case)
    assert set(data) == {"format", "schema_version", "coordinate_space", "document", "annotations"}
    assert data["format"] == "citehl-annotation-sidecar"
    assert type(data["schema_version"]) is int and data["schema_version"] == 1
    assert data["coordinate_space"] == {
        "name": "unrotated-pdf-user-space", "units": "points", "origin": "bottom-left",
        "rect": "[x0,y0,x1,y1]", "page_index_base": 1,
    }
    assert set(data["document"]) == {"algorithm", "fingerprint", "pages"}
    assert data["document"]["algorithm"] == "citehl-page-sha256-v1"
    assert [p["page"] for p in data["document"]["pages"]] == [1, 2]
    for page in data["document"]["pages"]:
        assert set(page) == {"page", "fingerprint"} and len(page["fingerprint"]) == 64
    assert [a["region"]["page"] for a in data["annotations"]] == [1, 1, 2]
    for entry in data["annotations"]:
        assert set(entry) == {"annotation_id", "region"}
        assert entry["annotation_id"].startswith("citehl:v1:")
        assert set(entry["region"]) == {"chunk_id", "source_ref", "page", "rect", "text"} | (
            {"source_refs"} if entry is data["annotations"][0] else set())
        entry["region"]["rect"] = [int(v) for v in entry["region"]["rect"]]
    assert data["annotations"][0]["region"]["source_refs"] == ["#/texts/0", "merged/中文"]
    # Key order/whitespace/Unicode escapes may differ; numeric equality only for rects.
    def reorder(value):
        if isinstance(value, dict): return {k: reorder(value[k]) for k in reversed(value)}
        if isinstance(value, list): return [reorder(v) for v in value]
        return value
    path.write_text(json.dumps(reorder(data), ensure_ascii=True, indent=3))
    check(capsys, sidecar_case)


@pytest.mark.parametrize("path,value", [
    ([], []), (["format"], "other"), (["schema_version"], True),
    (["schema_version"], 1.0), (["schema_version"], 2),
    (["coordinate_space", "name"], "display"), (["coordinate_space", "units"], "pixels"),
    (["coordinate_space", "origin"], "top-left"), (["coordinate_space", "rect"], "xywh"),
    (["coordinate_space", "page_index_base"], True), (["coordinate_space", "page_index_base"], 1.0),
    (["document", "algorithm"], "other"), (["document", "fingerprint"], "0" * 64),
    (["document", "pages", 0, "fingerprint"], "0" * 64),
    (["document", "pages", 0, "page"], True), (["document", "pages", 0, "page"], 1.0),
    (["document", "pages", 0, "page"], 2),
    (["annotations"], []), (["annotations"], None),
    (["annotations", 0, "annotation_id"], "citehl:v1:forged"),
    (["annotations", 0, "region", "chunk_id"], "forged"),
    (["annotations", 0, "region", "source_ref"], "forged"),
    (["annotations", 0, "region", "source_refs"], ["merged/中文", "#/texts/0"]),
    (["annotations", 0, "region", "source_refs"], ["#/texts/0"]),
    (["annotations", 0, "region", "text"], "forged"),
    (["annotations", 0, "region", "page"], True),
    (["annotations", 0, "region", "page"], 1.0),
    (["annotations", 0, "region", "page"], 2),
    (["annotations", 0, "region", "rect"], [1, 2, 3]),
    (["annotations", 0, "region", "rect", 0], True),
    (["annotations", 0, "region", "rect", 0], "72"),
    (["annotations", 0, "region", "rect", 0], 72.001),
])
def test_changed_values(sidecar_case, capsys, path, value):
    artifact, data = saved_data(sidecar_case)
    node = data
    for key in path[:-1]: node = node[key]
    if path: node[path[-1]] = value
    else: data = value
    write_saved(artifact, data)
    check(capsys, sidecar_case, 1, "sidecar" + "".join(f"[{k!r}]" for k in path))


@pytest.mark.parametrize("path", [[], ["coordinate_space"], ["document"], ["document", "pages", 0],
                                  ["annotations", 0], ["annotations", 0, "region"]])
@pytest.mark.parametrize("mode", ["missing", "extra", "array"])
def test_exact_object_fields(sidecar_case, capsys, path, mode):
    artifact, data = saved_data(sidecar_case)
    node = data
    for key in path: node = node[key]
    if mode == "missing": node.pop(next(iter(node)))
    elif mode == "extra": node["ignored"] = "must fail"
    elif path:
        parent = data
        for key in path[:-1]: parent = parent[key]
        parent[path[-1]] = []
    else: data = []
    write_saved(artifact, data)
    check(capsys, sidecar_case, 1, "object fields differ; regenerate with ragwrench highlight export")


@pytest.mark.parametrize("path", [["document", "pages"], ["annotations"],
                                  ["annotations", 0, "region", "source_refs"]])
@pytest.mark.parametrize("mode", ["reorder", "missing", "extra", "object"])
def test_ordered_arrays(sidecar_case, capsys, path, mode):
    artifact, data = saved_data(sidecar_case)
    node = data
    for key in path[:-1]: node = node[key]
    value = node[path[-1]]
    if mode == "reorder": value.reverse()
    elif mode == "missing": value.pop()
    elif mode == "extra": value.append(value[0])
    else: node[path[-1]] = {}
    write_saved(artifact, data)
    check(capsys, sidecar_case, 1, "sidecar[")


def test_consistently_forged_region_is_not_an_expectation(sidecar_case, capsys):
    from ragwrench.highlight.core import Region
    from ragwrench.highlight.pdf import annotation_id
    path, data = saved_data(sidecar_case)
    entry = data["annotations"][0]
    entry["region"]["rect"][0] += 1
    entry["annotation_id"] = annotation_id(Region.from_dict(entry["region"]))
    write_saved(path, data)
    check(capsys, sidecar_case, 1, "sidecar['annotations'][0]['annotation_id']")


@pytest.mark.parametrize("payload", [
    '{"annotations":[{}],"annotat\\u0069ons":[]}',
    '{"document":{"pages":[{"fingerprint":"x","fingerprint":"x"}]}}',
    '{"annotations":[{"region":{"page":1,"page":1}}]}',
    '{"annotations":NaN}', '{"annotations":Infinity}', '{"annotations":1e999}', '{broken',
])
def test_malformed_json_and_duplicates(sidecar_case, capsys, payload):
    Path(sidecar_case[3]).write_text(payload)
    check(capsys, sidecar_case, 1, "ragwrench highlight:")


@pytest.mark.parametrize("target", ["pdf", "sidecar", "regions"])
@pytest.mark.parametrize("mode", ["missing", "invalid", "oversized"])
def test_input_errors(sidecar_case, capsys, monkeypatch, target, mode):
    path = Path(sidecar_case[{"pdf": 1, "sidecar": 3}.get(target, 1)])
    args = sidecar_case
    if target == "regions":
        path = path.with_name("final.jsonl")
        args = args + ["--regions", str(path)]
    if mode == "missing": path.unlink()
    elif mode == "invalid": path.write_bytes(b'\xffbroken')
    else:
        monkeypatch.setattr("ragwrench.highlight.core.MAX_PDF_BYTES" if target == "pdf" else "ragwrench.highlight.core.MAX_JSON_BYTES", 10)
    check(capsys, args, 1, "ragwrench highlight:")


def edit_pdf(args, edit):
    from pypdf import PdfWriter
    writer = PdfWriter(clone_from=open_pdf(Path(args[1])))
    edit(writer)
    writer.write(args[1])


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "moved", "contents", "rect", "hidden",
                                    "metadata-json", "metadata-type", "metadata-extra", "metadata-id",
                                    "metadata-duplicate", "metadata-version", "nm", "nm-missing"])
def test_pdf_integrity(sidecar_case, capsys, fallback, change):
    from pypdf.generic import NameObject, NumberObject, RectangleObject, TextStringObject
    from ragwrench.highlight.pdf import add_highlight
    def edit(writer):
        annots = writer.pages[0]["/Annots"]
        annotation = annots[0].get_object()
        if change == "missing": annots.pop(0)
        elif change == "extra":
            region = read_regions(Path(sidecar_case[1]).with_name("final.jsonl"))[0]
            add_highlight(writer, replace(region, chunk_id="unexpected"))
        elif change == "duplicate": annots.append(annots[0])
        elif change == "moved": writer.pages[1]["/Annots"].append(annots.pop(0))
        elif change == "contents": annotation[NameObject("/Contents")] = TextStringObject("edited")
        elif change == "rect": annotation[NameObject("/Rect")] = RectangleObject((1, 1, 10, 10))
        elif change == "hidden": annotation[NameObject("/F")] = NumberObject(2)
        elif change == "metadata-json": annotation[NameObject("/CiteHL")] = TextStringObject("{broken")
        elif change == "metadata-type": annotation[NameObject("/CiteHL")] = NumberObject(1)
        elif change == "metadata-duplicate":
            raw = str(annotation["/CiteHL"])
            annotation[NameObject("/CiteHL")] = TextStringObject(raw[:-1] + ',"schema_version":1}')
        elif change.startswith("metadata-"):
            data = json.loads(annotation["/CiteHL"])
            if change == "metadata-extra": data["ignored"] = True
            elif change == "metadata-id": data["annotation_id"] = "citehl:v1:forged"
            else: data["schema_version"] = True
            annotation[NameObject("/CiteHL")] = TextStringObject(json.dumps(data))
        elif change == "nm": annotation[NameObject("/NM")] = TextStringObject("foreign")
        else: annotation.pop("/NM")
    edit_pdf(sidecar_case, edit)
    args = sidecar_case
    if fallback:
        # Match the saved page order, so failures must be integrity, not fallback order.
        original = read_regions(Path(args[1]).with_name("final.jsonl"))
        final = Path(args[1]).with_name("page-order.jsonl")
        final.write_text(regions_jsonl([original[1], original[2], original[0]]))
        args = args + ["--regions", str(final)]
    check(capsys, args, 1, "ragwrench highlight:")


@pytest.mark.parametrize("change", ["content", "resource", "crop", "rotation", "page-order", "page-count"])
def test_current_document_is_independently_fingerprinted(sidecar_case, capsys, change):
    from pypdf.generic import NameObject, RectangleObject
    def edit(writer):
        page = writer.pages[0]
        if change == "content":
            stream = page["/Contents"].get_object()
            stream.set_data(stream.get_data().replace(b"Revenue", b"Changed"))
        elif change == "resource": page["/Resources"]["/Font"]["/F1"][NameObject("/BaseFont")] = NameObject("/Courier")
        elif change == "crop": page.cropbox = RectangleObject((1, 1, 610, 790))
        elif change == "rotation": page.rotate(90)
        elif change == "page-count": writer.add_blank_page(width=612, height=792)
        else:
            # Strip annotations on both pages so integrity succeeds and identity must fail.
            for page in writer.pages: page.pop("/Annots", None)
            writer.add_page(writer.pages[0])
            writer.remove_page(0)
    if change == "page-order":
        # Expected empty set: a page swap must still invalidate the document fingerprint.
        edit_pdf(sidecar_case, lambda w: [p.pop("/Annots", None) for p in w.pages])
        write_saved(Path(sidecar_case[3]), export_sidecar(open_pdf(Path(sidecar_case[1]))))
    edit_pdf(sidecar_case, edit)
    check(capsys, sidecar_case, 1, "sidecar['document']")


def test_foreign_annotations_metadata_compression_and_name_are_excluded(sidecar_case, capsys):
    from pypdf.generic import DictionaryObject, NameObject, RectangleObject, TextStringObject
    def edit(writer):
        writer.add_metadata({"/Title": "renamed"})
        for page in writer.pages: page.compress_content_streams()
        writer.add_annotation(0, DictionaryObject({
            NameObject("/Subtype"): NameObject("/Text"), NameObject("/Rect"): RectangleObject((1, 1, 10, 10)),
            NameObject("/NM"): TextStringObject("foreign"), NameObject("/Contents"): TextStringObject("keep"),
        }))
    edit_pdf(sidecar_case, edit)
    pdf = Path(sidecar_case[1])
    renamed = pdf.with_name("renamed.pdf")
    pdf.rename(renamed)
    sidecar_case[1] = str(renamed)
    check(capsys, sidecar_case)


def test_legacy_fallback_and_explicit_order(sidecar_case, capsys):
    final = Path(sidecar_case[1]).with_name("final.jsonl")
    # Both modes are valid, but match the order of the export used to save the sidecar.
    check(capsys, sidecar_case + ["--regions", str(final)], 1, "sidecar['annotations'][0]")
    write_saved(Path(sidecar_case[3]), export_sidecar(open_pdf(Path(sidecar_case[1])), read_regions(final)))
    check(capsys, sidecar_case + ["--regions", str(final)])
    check(capsys, sidecar_case, 1, "sidecar['annotations'][0]")
    edit_pdf(sidecar_case, lambda w: [ref.get_object().pop("/CiteHL") for p in w.pages for ref in p["/Annots"]])
    check(capsys, sidecar_case, 1, "missing region metadata; supply the original regions JSONL")
    check(capsys, sidecar_case + ["--regions", str(final)])


@pytest.mark.parametrize("fallback", [False, True])
def test_empty_contract(sidecar_case, capsys, fallback):
    edit_pdf(sidecar_case, lambda w: [p.pop("/Annots", None) for p in w.pages])
    path, _ = saved_data(sidecar_case)
    write_saved(path, export_sidecar(open_pdf(Path(sidecar_case[1]))))
    args = sidecar_case
    if fallback:
        empty = Path(args[1]).with_name("empty.jsonl")
        empty.write_text(" \n\n")
        args = args + ["--regions", str(empty)]
        check(capsys, args, 1, "expected 1..")
        args = args + ["--allow-empty"]
    check(capsys, args, annotations=0)
    from ragwrench.highlight.pdf import add_highlight
    final = Path(args[1]).with_name("final.jsonl")
    edit_pdf(sidecar_case, lambda w: add_highlight(w, read_regions(final)[0]))
    check(capsys, args, 1, "unexpected citehl annotation" if fallback else "expected 1 ordered entries")


@pytest.mark.parametrize("extra", [["-o", "out.json"], ["--force"], ["--json"], ["--allow-empty"],
                                  ["--tolerance", "1"], ["--text-check", "off"], ["extra.jsonl"]])
def test_usage_errors_are_read_only(sidecar_case, capsys, extra):
    directory = Path(sidecar_case[1]).parent
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc: main(sidecar_case + extra)
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert output.out == "" and "error:" in output.err
    assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before


def test_missing_sidecar_is_usage_error(capsys):
    with pytest.raises(SystemExit) as exc: main(["check-sidecar", "source.pdf"])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert output.out == "" and "--sidecar" in output.err
