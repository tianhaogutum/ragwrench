from dataclasses import replace

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, FloatObject, NameObject, NumberObject, TextStringObject

from ragwrench.highlight.core import CiteError, geometries, open_pdf, read_regions, validate_regions
from ragwrench.highlight.pdf import PREFIX, annotate, annotation_id, manifest, quad_points, verify


def test_end_to_end_and_idempotence(examples, tmp_path):
    regions = read_regions(examples / "regions.jsonl")
    output = tmp_path / "out.pdf"
    source = open_pdf(examples / "sample.pdf")
    annotate(source, regions, output)
    first = open_pdf(output)
    assert len(first.pages) == 2
    for page, region in zip(first.pages, regions):
        assert page.extract_text().strip() == region.text
        assert len(page["/Annots"]) == 1
        annotation = page["/Annots"][0].get_object()
        assert annotation["/Subtype"] == "/Highlight"
        assert list(annotation["/Rect"]) == list(region.rect)
        assert list(annotation["/QuadPoints"]) == quad_points(region.rect)
        assert annotation["/NM"] == annotation_id(region)
        assert annotation["/Contents"] == f"{region.chunk_id}\n{region.text}"
        assert annotation["/C"] == [1, 1, 0]
    report = verify(first, regions, text_check="required")
    assert report == {"annotations": 2, "text_checked": 2, "text_skipped": 0, "text_check": "required", "notes": []}
    again = tmp_path / "again.pdf"
    annotate(first, regions, again)
    assert verify(open_pdf(again), regions)["annotations"] == 2
    assert [len(p["/Annots"]) for p in open_pdf(again).pages] == [1, 1]
    assert (examples / "sample.pdf").read_bytes() != output.read_bytes()
    assert all("/Annots" not in p for p in source.pages)


def test_preserves_foreign_annotation_and_replaces_stale(examples, tmp_path):
    writer = PdfWriter(clone_from=open_pdf(examples / "sample.pdf"))
    writer.add_annotation(0, DictionaryObject({NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Text"), NameObject("/Rect"): ArrayObject([NumberObject(0)] * 4),
        NameObject("/NM"): TextStringObject("someone-else"), NameObject("/Contents"): TextStringObject("keep me")}))
    source = tmp_path / "source.pdf"
    writer.write(source)
    regions = read_regions(examples / "regions.jsonl")
    out = tmp_path / "out.pdf"
    annotate(open_pdf(source), regions, out)
    changed = [replace(regions[0], rect=(73, 686, 300, 700))]
    final = tmp_path / "final.pdf"
    annotate(open_pdf(out), changed, final)
    reader = open_pdf(final)
    assert len(reader.pages[0]["/Annots"]) == 2
    assert len(reader.pages[1].get("/Annots", [])) == 0
    assert reader.pages[0]["/Annots"][0].get_object()["/Contents"] == "keep me"
    assert verify(reader, changed)["annotations"] == 1


def test_tampered_regions_fail(examples, tmp_path):
    regions = read_regions(examples / "regions.jsonl")
    output = tmp_path / "out.pdf"
    annotate(open_pdf(examples / "sample.pdf"), regions, output)
    for changed in [replace(regions[0], rect=(73, 686, 300, 700)), replace(regions[0], text="tampered"),
                    replace(regions[0], page=2), replace(regions[0], chunk_id="different")]:
        with pytest.raises(CiteError, match="verification failed"):
            verify(open_pdf(output), [changed, regions[1]])


@pytest.mark.parametrize("field,value", [
    ("/Rect", ArrayObject([FloatObject(v) for v in (72, 686, 299, 700)])),
    ("/QuadPoints", ArrayObject([NumberObject(1)] * 8)),
    ("/Contents", TextStringObject("tampered")),
    ("/Subtype", NameObject("/Square")),
    ("/F", NumberObject(2)), ("/CA", FloatObject(0)),
])
def test_tampered_annotation_fields(examples, tmp_path, field, value):
    regions = read_regions(examples / "regions.jsonl")
    output = tmp_path / "out.pdf"
    annotate(open_pdf(examples / "sample.pdf"), regions, output)
    writer = PdfWriter(clone_from=open_pdf(output))
    writer.pages[0]["/Annots"][0].get_object()[NameObject(field)] = value
    writer.write(tmp_path / "bad.pdf")
    with pytest.raises(CiteError, match="verification failed"):
        verify(open_pdf(tmp_path / "bad.pdf"), regions)


def test_unexpected_and_duplicate_annotations(examples, tmp_path):
    regions = read_regions(examples / "regions.jsonl")
    output = tmp_path / "out.pdf"
    annotate(open_pdf(examples / "sample.pdf"), regions, output)
    with pytest.raises(CiteError, match="unexpected"):
        verify(open_pdf(output), regions[:1])
    writer = PdfWriter(clone_from=open_pdf(output))
    writer.pages[0]["/Annots"].append(writer.pages[0]["/Annots"][0])
    writer.write(tmp_path / "bad.pdf")
    with pytest.raises(CiteError, match="duplicate"):
        verify(open_pdf(tmp_path / "bad.pdf"), regions)


def test_numeric_tolerance(examples, tmp_path):
    regions = read_regions(examples / "regions.jsonl")
    output = tmp_path / "out.pdf"
    annotate(open_pdf(examples / "sample.pdf"), regions, output)
    writer = PdfWriter(clone_from=open_pdf(output))
    rect = writer.pages[0]["/Annots"][0].get_object()["/Rect"]
    rect[0] = FloatObject(float(rect[0]) + 0.005)
    writer.write(tmp_path / "rounded.pdf")
    assert verify(open_pdf(tmp_path / "rounded.pdf"), regions)["annotations"] == 2
    with pytest.raises(CiteError, match="/Rect mismatch"):
        verify(open_pdf(tmp_path / "rounded.pdf"), regions, tolerance=0.001)


def test_text_mismatch_and_unavailable(examples, tmp_path):
    regions = read_regions(examples / "regions.jsonl")
    wrong = [replace(regions[0], text="elephants sunshine")]
    output = tmp_path / "out.pdf"
    annotate(open_pdf(examples / "sample.pdf"), wrong, output)
    with pytest.raises(CiteError, match="text mismatch"):
        verify(open_pdf(output), wrong)
    assert verify(open_pdf(output), wrong, text_check="off")["annotations"] == 1
    empty = [replace(regions[0], rect=(10, 10, 40, 40))]
    annotate(open_pdf(examples / "sample.pdf"), empty, output)
    assert verify(open_pdf(output), empty)["text_skipped"] == 1
    with pytest.raises(CiteError, match="text unavailable"):
        verify(open_pdf(output), empty, text_check="required")


def test_manifest_links_and_multipage(examples):
    regions = read_regions(examples / "regions.jsonl")
    regions = [replace(r, chunk_id="summary") for r in regions]
    result = manifest(regions, "report #1.pdf")["chunks"]["summary"]
    assert result["pages"] == [1, 2]
    assert result["links"] == ["report%20%231.pdf#page=1", "report%20%231.pdf#page=2"]
    assert [r["page"] for r in result["rects"]] == [1, 2]


def test_encrypted_pdf_rejected(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("")
    writer.write(tmp_path / "encrypted.pdf")
    with pytest.raises(CiteError, match="encrypted"):
        open_pdf(tmp_path / "encrypted.pdf")


def test_invalid_pdf(tmp_path):
    path = tmp_path / "not.pdf"
    path.write_bytes(b"hello, not a PDF")
    with pytest.raises(CiteError, match="invalid PDF"):
        open_pdf(path)


def test_geometry_and_region_failure_cases(examples):
    source = open_pdf(examples / "sample.pdf")
    regions = read_regions(examples / "regions.jsonl")
    with pytest.raises(CiteError, match="outside PDF"):
        validate_regions([replace(regions[0], page=3)], geometries(source), strict=True)
    with pytest.raises(CiteError, match="wholly outside"):
        validate_regions([replace(regions[0], rect=(900, 900, 950, 950))], geometries(source), strict=False)
    source.pages[0][NameObject("/Rotate")] = NumberObject(45)
    with pytest.raises(CiteError, match="multiples of 90"):
        geometries(source)
    source.pages[0][NameObject("/Rotate")] = NumberObject(0)
    source.pages[0][NameObject("/UserUnit")] = NumberObject(2)
    with pytest.raises(CiteError, match="UserUnit"):
        geometries(source)


def test_rotated_crop_pdf_readback(examples, tmp_path):
    from ragwrench.highlight.normalize import normalize_unstructured
    from pypdf.generic import RectangleObject
    writer = PdfWriter(clone_from=open_pdf(examples / "sample.pdf"))
    writer.pages[0].cropbox = RectangleObject((20, 30, 220, 330))
    writer.pages[0].rotate(90)
    writer.write(tmp_path / "crop.pdf")
    source = open_pdf(tmp_path / "crop.pdf")
    data = [{"element_id": "e", "text": "", "metadata": {"page_number": 1, "coordinates": {
        "system": "PixelSpace", "layout_width": 300, "layout_height": 200,
        "points": [[10, 20], [70, 40]]}}}]
    regions = normalize_unstructured(data, geometries(source))
    assert regions[0].rect == (40, 40, 60, 100)
    annotate(source, regions, tmp_path / "out.pdf")
    output = open_pdf(tmp_path / "out.pdf")
    assert output.pages[0].rotation == 90
    assert list(output.pages[0].cropbox) == [20, 30, 220, 330]
    assert verify(output, regions)["annotations"] == 1
