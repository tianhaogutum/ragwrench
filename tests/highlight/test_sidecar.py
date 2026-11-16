"""Portable annotation sidecars exercised against real, synthetic PDF objects."""
import json
from dataclasses import replace

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, RectangleObject, TextStringObject

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import open_pdf, read_regions, regions_jsonl
from ragwrench.highlight.pdf import annotate, annotation_id, verify


@pytest.fixture
def exported(examples, tmp_path):
    regions = read_regions(examples / "regions.jsonl")
    highlighted = tmp_path / "highlighted.pdf"
    annotate(open_pdf(examples / "sample.pdf"), regions, highlighted)
    sidecar = tmp_path / "notes.json"
    assert main(["export", str(highlighted), str(examples / "regions.jsonl"),
                 "-o", str(sidecar)]) == 0
    return regions, highlighted, sidecar


def import_args(pdf, sidecar, output):
    return ["import", str(pdf), str(sidecar), "-o", str(output)]


def test_empty_sidecar_is_additive_noop_with_foreign_and_citehl_annotations(exported, examples, tmp_path, capsys):
    regions, highlighted, _ = exported
    empty = tmp_path / "empty.json"
    assert main(["export", str(examples / "sample.pdf"), "-o", str(empty)]) == 0
    assert json.loads(empty.read_text())["annotations"] == []
    writer = PdfWriter(clone_from=open_pdf(highlighted))
    writer.add_annotation(0, DictionaryObject({
        NameObject("/Subtype"): NameObject("/Text"),
        NameObject("/Rect"): RectangleObject((10, 10, 30, 30)),
        NameObject("/Contents"): TextStringObject("keep foreign"),
    }))
    existing = tmp_path / "existing.pdf"
    writer.write(existing)
    before = existing.read_bytes()
    for index in range(2):
        output = tmp_path / f"noop-{index}.pdf"
        assert main(import_args(existing, empty, output)) == 0
        assert existing.read_bytes() == before
        assert verify(open_pdf(output), regions, text_check="required")["annotations"] == 2
        comment = open_pdf(output).pages[0]["/Annots"][-1].get_object()
        assert comment["/Contents"] == "keep foreign"
        existing, before = output, output.read_bytes()
    assert capsys.readouterr().out.count("OK imported 0 annotations; already present 0") == 2


@pytest.mark.parametrize("change", ["missing", "null", "object", "string", "fingerprint"])
def test_empty_sidecar_invalid_structure_or_identity_preserves_output(exported, examples, tmp_path, capsys, change):
    _, _, sidecar = exported
    data = json.loads(sidecar.read_text())
    data["annotations"] = []
    if change == "missing":
        data.pop("annotations")
    elif change == "null":
        data["annotations"] = None
    elif change == "object":
        data["annotations"] = {}
    elif change == "string":
        data["annotations"] = ""
    else:
        data["document"]["pages"][0]["fingerprint"] = "0" * 64
    sidecar.write_text(json.dumps(data))
    output = tmp_path / "out.pdf"
    output.write_bytes(b"sentinel")
    assert main(import_args(examples / "sample.pdf", sidecar, output) + ["--force"]) == 1
    assert output.read_bytes() == b"sentinel"
    assert ("fingerprint" if change == "fingerprint" else "annotations must be a") in capsys.readouterr().err


@pytest.mark.parametrize("existing_output", [False, True])
@pytest.mark.parametrize("change", ["nonempty-empty", "empty-nonempty", "null-empty", "empty-null",
                                   "empty-empty", "escaped-key", "nested-fingerprint"])
def test_duplicate_sidecar_fields_never_select_an_empty_set(
        exported, examples, tmp_path, capsys, change, existing_output):
    _, _, sidecar = exported
    data = json.loads(sidecar.read_text())
    annotations = data.pop("annotations")
    values = {
        "nonempty-empty": (annotations, []), "empty-nonempty": ([], annotations),
        "null-empty": (None, []), "empty-null": ([], None),
        "empty-empty": ([], []), "escaped-key": (annotations, []),
    }
    if change == "nested-fingerprint":
        data["annotations"] = []
        page = json.dumps(data["document"]["pages"][0])
        duplicate = page.replace('"fingerprint":', '"fingerprint": "wrong", "fingerprint":')
        payload = json.dumps(data).replace(page, duplicate, 1)
    else:
        first, second = values[change]
        # Escape decoding must happen before checking uniqueness of key names.
        second_key = '"annotat\\u0069ons"' if change == "escaped-key" else '"annotations"'
        payload = (json.dumps(data)[:-1] + ', "annotations": ' + json.dumps(first)
                   + ', ' + second_key + ': ' + json.dumps(second) + '}')
    sidecar.write_text(payload)
    output = tmp_path / "out.pdf"
    if existing_output:
        output.write_bytes(b"sentinel")
    source = examples / "sample.pdf"
    before = source.read_bytes()
    assert main(import_args(source, sidecar, output) + ["--force"]) == 1
    assert "duplicate JSON key" in capsys.readouterr().err
    assert source.read_bytes() == before
    if existing_output:
        assert output.read_bytes() == b"sentinel"
    else:
        assert not output.exists()


@pytest.mark.parametrize("command", ["verify", "export"])
def test_empty_jsonl_does_not_hide_unexpected_annotations(exported, tmp_path, command):
    _, pdf, _ = exported
    empty = tmp_path / "empty.jsonl"
    empty.write_text(" \n\n")
    args = [command, str(pdf), str(empty), "--allow-empty"]
    output = tmp_path / "out.json"
    if command == "export":
        output.write_bytes(b"sentinel")
        args += ["-o", str(output), "--force"]
    assert main(args) == 1
    if command == "export":
        assert output.read_bytes() == b"sentinel"


def test_sidecar_roundtrip_preserves_annotations_and_is_idempotent(exported, examples, tmp_path):
    regions, _, sidecar = exported
    payload = json.loads(sidecar.read_text())
    assert payload["coordinate_space"] == {
        "name": "unrotated-pdf-user-space", "units": "points", "origin": "bottom-left",
        "rect": "[x0,y0,x1,y1]", "page_index_base": 1,
    }
    assert [entry["region"] for entry in payload["annotations"]] == [r.to_dict() for r in regions]
    assert len(payload["document"]["pages"]) == 2
    # Both a foreign comment and a separate citehl highlight must survive import.
    extra = replace(regions[0], chunk_id="existing", text="existing note")
    existing = tmp_path / "existing.pdf"
    annotate(open_pdf(examples / "sample.pdf"), [extra], existing)
    writer = PdfWriter(clone_from=open_pdf(existing))
    writer.add_annotation(0, DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject("/Text"),
        NameObject("/Rect"): RectangleObject((10, 10, 30, 30)),
        NameObject("/NM"): TextStringObject("foreign"),
        NameObject("/Contents"): TextStringObject("keep me"),
    }))
    writer.write(existing)
    before = existing.read_bytes()
    restored = tmp_path / "restored.pdf"
    assert main(import_args(existing, sidecar, restored)) == 0
    assert existing.read_bytes() == before
    assert verify(open_pdf(restored), regions + [extra], text_check="off")["annotations"] == 3
    comment = open_pdf(restored).pages[0]["/Annots"][1].get_object()
    assert comment["/Contents"] == "keep me"
    again = tmp_path / "again.pdf"
    assert main(import_args(restored, sidecar, again)) == 0
    assert [len(p["/Annots"]) for p in open_pdf(again).pages] == [3, 1]
    assert verify(open_pdf(again), regions + [extra], text_check="off")["annotations"] == 3


@pytest.mark.parametrize("change", ["content", "font", "crop", "rotation", "reorder", "page-count"])
def test_rejects_changed_document_before_writing(exported, examples, tmp_path, capsys, change):
    _, _, sidecar = exported
    writer = PdfWriter(clone_from=open_pdf(examples / "sample.pdf"))
    if change == "content":
        stream = writer.pages[0]["/Contents"].get_object()
        stream.set_data(stream.get_data().replace(b"Revenue", b"Changed"))
    elif change == "font":
        writer.pages[0]["/Resources"]["/Font"]["/F1"][NameObject("/BaseFont")] = NameObject("/Courier")
    elif change == "crop":
        writer.pages[0].cropbox = RectangleObject((1, 1, 610, 790))
    elif change == "rotation":
        writer.pages[0].rotate(90)
    elif change == "reorder":
        writer = PdfWriter()
        reader = open_pdf(examples / "sample.pdf")
        writer.add_page(reader.pages[1])
        writer.add_page(reader.pages[0])
    else:
        writer.add_blank_page(width=612, height=792)
    wrong = tmp_path / "wrong.pdf"
    writer.write(wrong)
    output = tmp_path / "out.pdf"
    output.write_bytes(b"keep existing output")
    assert main(import_args(wrong, sidecar, output) + ["--force"]) == 1
    assert "document/page fingerprint mismatch" in capsys.readouterr().err
    assert output.read_bytes() == b"keep existing output"


def test_matches_reserialized_compressed_pdf_with_new_metadata(exported, examples, tmp_path):
    regions, _, sidecar = exported
    writer = PdfWriter(clone_from=open_pdf(examples / "sample.pdf"))
    writer.add_metadata({"/Title": "A renamed copy"})
    for page in writer.pages:
        page.compress_content_streams()
    target = tmp_path / "renamed.pdf"
    writer.write(target)
    output = tmp_path / "out.pdf"
    assert main(import_args(target, sidecar, output)) == 0
    assert verify(open_pdf(output), regions, text_check="required")["text_checked"] == 2


@pytest.mark.parametrize("change", ["version", "space", "fingerprint", "page", "rect", "id", "duplicate"])
def test_malformed_sidecar_refused(exported, examples, tmp_path, change):
    _, _, sidecar = exported
    data = json.loads(sidecar.read_text())
    if change == "version":
        data["schema_version"] = True
    elif change == "space":
        data["coordinate_space"]["origin"] = "top-left"
    elif change == "fingerprint":
        data["document"]["pages"][0]["fingerprint"] = "0" * 64
    elif change == "page":
        data["annotations"][0]["region"]["page"] = 3
        data["annotations"][0]["annotation_id"] = annotation_id(
            replace(read_regions(examples / "regions.jsonl")[0], page=3))
    elif change == "rect":
        data["annotations"][0]["region"]["rect"] = [0, 0, 900, 900]
        data["annotations"][0]["annotation_id"] = annotation_id(
            replace(read_regions(examples / "regions.jsonl")[0], rect=(0.0, 0.0, 900.0, 900.0)))
    elif change == "id":
        data["annotations"][0]["annotation_id"] = annotation_id(read_regions(examples / "regions.jsonl")[1])
    else:
        data["annotations"].append(data["annotations"][0])
    sidecar.write_text(json.dumps(data))
    output = tmp_path / "out.pdf"
    assert main(import_args(examples / "sample.pdf", sidecar, output)) == 1
    assert not output.exists()


@pytest.mark.parametrize("change", ["modified", "moved", "duplicate"])
def test_existing_id_conflict_refused_without_replacing(exported, tmp_path, change):
    _, highlighted, sidecar = exported
    writer = PdfWriter(clone_from=open_pdf(highlighted))
    annotation = writer.pages[0]["/Annots"][0].get_object()
    if change == "modified":
        annotation[NameObject("/Contents")] = TextStringObject("edited in viewer")
    elif change == "moved":
        writer.pages[1]["/Annots"].append(writer.pages[0]["/Annots"].pop())
    else:
        writer.pages[0]["/Annots"].append(writer.pages[0]["/Annots"][0])
    target = tmp_path / "target.pdf"
    writer.write(target)
    output = tmp_path / "out.pdf"
    assert main(import_args(target, sidecar, output)) == 1
    assert not output.exists()


def test_export_rejects_unverified_annotations(examples, tmp_path):
    output = tmp_path / "notes.json"
    assert main(["export", str(examples / "sample.pdf"), str(examples / "regions.jsonl"),
                 "-o", str(output)]) == 1
    assert not output.exists()


def test_sidecar_output_protections(exported, examples, tmp_path):
    _, highlighted, sidecar = exported
    before = sidecar.read_bytes()
    assert main(["export", str(highlighted), str(examples / "regions.jsonl"), "-o", str(sidecar)]) == 1
    assert sidecar.read_bytes() == before
    alias = tmp_path / "alias.pdf"
    alias.symlink_to(highlighted)
    assert main(import_args(highlighted, sidecar, alias) + ["--force"]) == 1
    assert alias.is_symlink()


def test_indirect_annotation_arrays(exported, examples, tmp_path):
    regions, highlighted, sidecar = exported
    writer = PdfWriter(clone_from=open_pdf(highlighted))
    for page in writer.pages:
        page[NameObject("/Annots")] = writer._add_object(page["/Annots"])
    target = tmp_path / "indirect.pdf"
    writer.write(target)
    assert main(["export", str(target), str(examples / "regions.jsonl"),
                 "-o", str(tmp_path / "export.json")]) == 0
    output = tmp_path / "out.pdf"
    assert main(import_args(target, sidecar, output)) == 0
    assert verify(open_pdf(output), regions, text_check="required")["annotations"] == 2


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_offset_boxes_and_rotation_use_native_coordinates(examples, tmp_path, rotation):
    writer = PdfWriter(clone_from=open_pdf(examples / "sample.pdf"))
    page = writer.pages[0]
    page.mediabox = RectangleObject((-20, -30, 380, 270))
    page.cropbox = RectangleObject((30, 40, 330, 240))
    page.rotate(rotation)
    source = tmp_path / "source.pdf"
    writer.write(source)
    regions = [replace(read_regions(examples / "regions.jsonl")[0], rect=(80, 120, 150, 140), text="")]
    region_file = tmp_path / "regions.jsonl"
    region_file.write_text(regions_jsonl(regions))
    highlighted = tmp_path / "highlighted.pdf"
    annotate(open_pdf(source), read_regions(region_file), highlighted)
    sidecar = tmp_path / "sidecar.json"
    assert main(["export", str(highlighted), str(region_file), "-o", str(sidecar)]) == 0
    output = tmp_path / "out.pdf"
    assert main(import_args(source, sidecar, output)) == 0
    restored = open_pdf(output)
    assert verify(restored, read_regions(region_file), text_check="off")["annotations"] == 1
    assert restored.pages[0].rotation == rotation
    assert list(restored.pages[0]["/Annots"][0].get_object()["/Rect"]) == [80, 120, 150, 140]
