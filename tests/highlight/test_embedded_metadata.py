"""PDF-only export and fail-closed embedded provenance contracts."""
import json
from dataclasses import replace

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, NumberObject, RectangleObject, TextStringObject

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import canonical, open_pdf, read_regions
from ragwrench.highlight.pdf import annotate, annotation_id, verify


@pytest.fixture
def annotated(examples, tmp_path):
    regions = read_regions(examples / "regions.jsonl")
    # Preserve merged provenance and non-ASCII text independently of /Contents.
    regions[0] = replace(regions[0], source_refs=(regions[0].source_ref, "另一个来源"))
    pdf = tmp_path / "highlighted.pdf"
    annotate(open_pdf(examples / "sample.pdf"), regions, pdf)
    return regions, pdf


def test_new_annotations_embed_versioned_regions(annotated):
    regions, pdf = annotated
    for page, region in zip(open_pdf(pdf).pages, regions):
        annotation = page["/Annots"][0].get_object()
        data = json.loads(annotation["/CiteHL"])
        assert data == {"format": "citehl-region", "schema_version": 1,
                        "annotation_id": annotation_id(region), "region": region.to_dict()}
        assert str(annotation["/CiteHL"]) == canonical(data)


def test_pdf_only_export_import_reexport(annotated, examples, tmp_path):
    regions, pdf = annotated
    # An unrelated viewer annotation has no embedded payload and is ignored.
    writer = PdfWriter(clone_from=open_pdf(pdf))
    writer.add_annotation(0, DictionaryObject({
        NameObject("/Subtype"): NameObject("/Text"), NameObject("/Rect"): RectangleObject((1, 1, 10, 10)),
        NameObject("/NM"): TextStringObject("foreign"), NameObject("/Contents"): TextStringObject("keep"),
    }))
    writer.write(pdf)
    sidecar = tmp_path / "notes.json"
    assert main(["export", str(pdf), "-o", str(sidecar)]) == 0
    data = json.loads(sidecar.read_text())
    assert [entry["region"] for entry in data["annotations"]] == [r.to_dict() for r in regions]
    restored = tmp_path / "restored.pdf"
    assert main(["import", str(examples / "sample.pdf"), str(sidecar), "-o", str(restored)]) == 0
    assert verify(open_pdf(restored), regions, text_check="required")["annotations"] == 2
    reexport = tmp_path / "reexport.json"
    assert main(["export", str(restored), "-o", str(reexport)]) == 0
    assert reexport.read_bytes() == sidecar.read_bytes()


@pytest.mark.parametrize("mode", ["pdf-only", "jsonl", "verify", "import"])
@pytest.mark.parametrize("change", ["json", "type", "version", "future-version", "format", "unknown-field", "region",
                                    "region-id", "payload-id", "nm", "nm-foreign", "nm-missing"])
def test_inconsistent_metadata_is_rejected(annotated, examples, tmp_path, capsys, mode, change):
    regions, pdf = annotated
    # Write a valid sidecar before corrupting the PDF, using the legacy invocation.
    sidecar = tmp_path / "valid.json"
    region_file = tmp_path / "regions.jsonl"
    region_file.write_text("\n".join(canonical(r.to_dict()) for r in regions))
    assert main(["export", str(pdf), str(region_file), "-o", str(sidecar)]) == 0
    writer = PdfWriter(clone_from=open_pdf(pdf))
    annotation = writer.pages[0]["/Annots"][0].get_object()
    data = json.loads(annotation["/CiteHL"])
    if change == "json":
        annotation[NameObject("/CiteHL")] = TextStringObject("{broken")
    elif change == "type":
        annotation[NameObject("/CiteHL")] = NumberObject(1)
    elif change == "nm":
        annotation[NameObject("/NM")] = TextStringObject(annotation_id(regions[1]))
    elif change == "nm-foreign":
        annotation[NameObject("/NM")] = TextStringObject("foreign")
    elif change == "nm-missing":
        annotation.pop("/NM")
    else:
        if change == "version":
            data["schema_version"] = True
        elif change == "future-version":
            data["schema_version"] = 2
        elif change == "format":
            data["format"] = "unknown"
        elif change == "unknown-field":
            data["ignored"] = "must not silently discard"
        elif change == "region":
            data["region"]["rect"] = [1, 2, 0, 0]
        elif change == "region-id":
            data["region"]["source_ref"] = "different"
            data["region"].pop("source_refs")
        else:
            data["annotation_id"] = annotation_id(regions[1])
        annotation[NameObject("/CiteHL")] = TextStringObject(canonical(data))
    writer.write(pdf)
    output = tmp_path / "out"
    output.write_bytes(b"sentinel")
    if mode == "verify":
        args = ["verify", str(pdf), str(region_file), "--text-check", "off"]
    elif mode == "import":
        args = ["import", str(pdf), str(sidecar), "-o", str(output), "--force"]
    else:
        args = ["export", str(pdf)] + ([str(region_file)] if mode == "jsonl" else [])
        args += ["-o", str(output), "--force"]
    assert main(args) == 1
    assert "metadata" in capsys.readouterr().err
    assert output.read_bytes() == b"sentinel"


@pytest.mark.parametrize("change", ["legacy", "mixed", "duplicate", "moved", "rect", "contents", "hidden"])
def test_pdf_only_export_refuses_incomplete_or_conflicting_annotations(annotated, tmp_path, capsys, change):
    regions, pdf = annotated
    writer = PdfWriter(clone_from=open_pdf(pdf))
    annotation = writer.pages[0]["/Annots"][0].get_object()
    if change in ("legacy", "mixed"):
        annotation.pop("/CiteHL")
        if change == "legacy":
            writer.pages[1]["/Annots"][0].get_object().pop("/CiteHL")
    elif change == "duplicate":
        writer.pages[0]["/Annots"].append(writer.pages[0]["/Annots"][0])
    elif change == "moved":
        writer.pages[1]["/Annots"].append(writer.pages[0]["/Annots"].pop())
    elif change == "rect":
        annotation[NameObject("/Rect")] = RectangleObject((1, 1, 10, 10))
    elif change == "contents":
        annotation[NameObject("/Contents")] = TextStringObject("edited")
    elif change == "hidden":
        annotation[NameObject("/F")] = NumberObject(2)
    writer.write(pdf)
    output = tmp_path / "out.json"
    output.write_bytes(b"sentinel")
    assert main(["export", str(pdf), "-o", str(output), "--force"]) == 1
    assert output.read_bytes() == b"sentinel"
    error = capsys.readouterr().err
    if change in ("legacy", "mixed"):
        assert "regions" in error and "metadata" in error
        # Compatibility mode still verifies old annotations using external regions.
        region_file = tmp_path / "regions.jsonl"
        region_file.write_text("\n".join(canonical(r.to_dict()) for r in regions))
        assert main(["export", str(pdf), str(region_file), "-o", str(output), "--force"]) == 0


def test_embedded_numeric_identity_roundtrip(examples, tmp_path):
    # Public Python writers also accept validated integer-valued coordinates.
    region = replace(read_regions(examples / "regions.jsonl")[0], rect=(72, 686, 300, 700))
    pdf = tmp_path / "integer.pdf"
    annotate(open_pdf(examples / "sample.pdf"), [region], pdf)
    sidecar = tmp_path / "integer.json"
    assert main(["export", str(pdf), "-o", str(sidecar)]) == 0
    assert json.loads(sidecar.read_text())["annotations"][0]["annotation_id"] == annotation_id(region)
    restored = tmp_path / "restored.pdf"
    assert main(["import", str(examples / "sample.pdf"), str(sidecar), "-o", str(restored)]) == 0
    assert verify(open_pdf(restored), [region], text_check="off")["annotations"] == 1


@pytest.mark.parametrize("change", ["page", "outside", "nonfinite", "source-refs"])
def test_writer_refuses_invalid_metadata_without_replacing(examples, tmp_path, change):
    from ragwrench.highlight.core import CiteError
    region = read_regions(examples / "regions.jsonl")[0]
    if change == "page":
        region = replace(region, page=0)
    elif change == "outside":
        region = replace(region, rect=(0, 0, 900, 900))
    elif change == "nonfinite":
        region = replace(region, rect=(0, 0, float("inf"), 900))
    else:
        region = replace(region, source_refs=("wrong",))
    output = tmp_path / "out.pdf"
    output.write_bytes(b"sentinel")
    with pytest.raises(CiteError):
        annotate(open_pdf(examples / "sample.pdf"), [region], output)
    assert output.read_bytes() == b"sentinel"
