import base64
import gzip
import json

import pytest

from ragwrench.highlight.core import CiteError, Geometry, geometries, open_pdf, read_json
from ragwrench.highlight.normalize import normalize_docling, normalize_unstructured, orig_elements


def doc(origin="BOTTOMLEFT", bbox=None):
    return {"texts": [{"self_ref": "#/texts/0", "text": "Revenue grew 12%.",
                       "prov": [{"page_no": 1, "bbox": bbox or {
                           "l": 72, "t": 700 if origin == "BOTTOMLEFT" else 92,
                           "r": 300, "b": 686 if origin == "BOTTOMLEFT" else 106,
                           "coord_origin": origin}}]}],
            "pages": {"1": {"size": {"width": 612, "height": 792}}}}


def element(system="PixelSpace", width=1700, height=2200, points=None):
    return {"element_id": "e1", "text": "Revenue grew 12%.", "metadata": {
        "page_number": 1, "coordinates": {"system": system, "layout_width": width,
                                          "layout_height": height,
                                          "points": points or [[200, 200], [400, 400]]}}}


@pytest.mark.parametrize("origin", ["TOPLEFT", "BOTTOMLEFT"])
def test_docling_origins(origin):
    assert normalize_docling(doc(origin))[0].rect == (72, 686, 300, 700)


def test_docling_default_origin():
    data = doc("TOPLEFT")
    del data["texts"][0]["prov"][0]["bbox"]["coord_origin"]
    assert normalize_docling(data)[0].rect == (72, 686, 300, 700)


def test_pixel_scaling():
    geometry = {1: Geometry((0, 0, 612, 792), (0, 0, 612, 792))}
    assert normalize_unstructured([element()], geometry)[0].rect == pytest.approx((72, 648, 144, 720))


def test_point_space_does_not_flip():
    assert normalize_unstructured([element("PointSpace", 612, 792, [[72, 686], [300, 700]])])[0].rect == (72, 686, 300, 700)


@pytest.mark.parametrize("rotation,expected", [
    (0, (30, 290, 90, 310)), (90, (40, 40, 60, 100)),
    (180, (150, 50, 210, 70)), (270, (180, 260, 200, 320)),
])
def test_display_rotation_and_crop_offset(rotation, expected):
    geometry = Geometry((0, 0, 400, 500), (20, 30, 220, 330), rotation)
    width, height = geometry.dimensions("crop", "display")
    e = element(width=width, height=height, points=[[10, 20], [70, 40]])
    assert normalize_unstructured([e], {1: geometry})[0].rect == pytest.approx(expected)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_docling_native_axes_not_double_rotated(rotation):
    geometry = Geometry((0, 0, 612, 792), (20, 30, 600, 780), rotation)
    assert normalize_docling(doc(), {1: geometry})[0].rect == (72, 686, 300, 700)


def test_media_box_offset():
    geometry = Geometry((10, 20, 622, 812), (10, 20, 622, 812))
    assert normalize_docling(doc(), {1: geometry})[0].rect == (82, 706, 310, 720)


def test_missing_page_size_fallback(examples):
    data = doc()
    del data["pages"]
    with pytest.warns(UserWarning, match="MediaBox"):
        actual = normalize_docling(data, geometries(open_pdf(examples / "sample.pdf")))
    assert actual[0].rect == (72, 686, 300, 700)
    with pytest.raises(CiteError, match="provide --pdf"):
        normalize_docling(data)


def test_clamp_and_strict():
    data = doc(bbox={"l": -10, "t": 800, "r": 300, "b": 686, "coord_origin": "BOTTOMLEFT"})
    with pytest.warns(UserWarning, match="clamped"):
        assert normalize_docling(data)[0].rect == (0, 686, 300, 792)
    with pytest.raises(CiteError, match="outside"):
        normalize_docling(data, strict=True)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["texts"][0].pop("prov"), "prov"),
    (lambda d: d["texts"][0]["prov"][0].update(page_no=0), "page"),
    (lambda d: d["texts"][0]["prov"][0]["bbox"].update(coord_origin="SIDEWAYS"), "unknown"),
    (lambda d: d["texts"][0]["prov"][0]["bbox"].update(l=500), "inverted"),
    (lambda d: d["texts"][0]["prov"][0]["bbox"].update(l=float("nan")), "finite"),
    (lambda d: d["pages"]["1"]["size"].update(width=0), "positive"),
    (lambda d: d["texts"][0]["prov"][0].update(charspan=[1, 999]), "charspan"),
])
def test_bad_docling(mutate, match):
    data = doc()
    mutate(data)
    with pytest.raises(CiteError, match=match):
        normalize_docling(data)


def test_page_outside_pdf():
    data = doc()
    data["texts"][0]["prov"][0]["page_no"] = 2
    with pytest.raises(CiteError, match="outside PDF"):
        normalize_docling(data, {1: Geometry((0, 0, 612, 792), (0, 0, 612, 792))})


@pytest.mark.parametrize("system", ["RelativeCoordinateSystem", "CoordinateSystem", None])
def test_unknown_system(system):
    with pytest.raises(CiteError, match="unknown"):
        normalize_unstructured([element(system)])


def test_pixels_need_pdf():
    with pytest.raises(CiteError, match="requires --pdf"):
        normalize_unstructured([element()])


def test_charspan_and_tables():
    data = doc()
    data["texts"][0]["prov"][0]["charspan"] = [0, 7]
    assert normalize_docling(data)[0].text == "Revenue"
    item = data.pop("texts")[0]
    item.pop("text")
    item["prov"][0].pop("charspan")
    item["data"] = {"table_cells": [{"text": "Quarter"}, {"text": "Revenue"}]}
    data["tables"] = [item]
    assert normalize_docling(data)[0].text == "Quarter Revenue"


def test_inline_compressed_chunks(examples):
    chunks = read_json(examples / "chunks.unstructured.json")
    result = normalize_unstructured(chunks, geometries(open_pdf(examples / "sample.pdf")))
    assert [r.chunk_id for r in result] == ["quarterly-summary"] * 2
    assert [r.page for r in result] == [1, 2]
    assert result[0].rect == pytest.approx((72, 686, 300, 700))


def test_orig_elements_bounds_and_errors(monkeypatch):
    with pytest.raises(CiteError, match="gzip JSON"):
        orig_elements("this is not base64")
    with pytest.raises(CiteError, match="gzip JSON"):
        orig_elements(base64.b64encode(b"not gzip").decode())
    monkeypatch.setattr("ragwrench.highlight.normalize.MAX_JSON_BYTES", 100)
    encoded = base64.b64encode(gzip.compress(b" " * 101)).decode()
    with pytest.raises(CiteError, match="decompression"):
        orig_elements(encoded)


def test_nested_orig_elements_rejected():
    source = element("PointSpace")
    source["metadata"]["orig_elements"] = [element("PointSpace")]
    chunk = {"element_id": "chunk", "metadata": {"orig_elements": [source]}}
    with pytest.raises(CiteError, match="nested"):
        normalize_unstructured([chunk])


@pytest.mark.parametrize("compressor", [gzip.compress, __import__("zlib").compress])
def test_both_upstream_compression_headers(compressor):
    data = [{"element_id": "a"}]
    assert orig_elements(base64.b64encode(compressor(json.dumps(data).encode())).decode()) == data


def test_compressed_stream_integrity():
    import zlib
    compressed = zlib.compress(b"[]")
    for data in (compressed[:-2], compressed + b"trailing"):
        with pytest.raises(CiteError, match="truncated or has trailing"):
            orig_elements(base64.b64encode(data).decode())


def test_coordinate_conversion_overflow():
    e = element("PointSpace", width=1e-300, height=792, points=[[0, 0], [1e300, 700]])
    with pytest.raises(CiteError, match="overflow"):
        normalize_unstructured([e], {1: Geometry((0, 0, 612, 792), (0, 0, 612, 792))})


def test_docling_display_override_matches_pixel():
    geometry = Geometry((0, 0, 400, 500), (20, 30, 220, 330), 90)
    data = {"texts": [{"self_ref": "a", "text": "", "prov": [{"page_no": 1, "bbox": {
        "l": 10, "t": 20, "r": 70, "b": 40, "coord_origin": "TOPLEFT"}}]}],
        "pages": {"1": {"size": {"width": 300, "height": 200}}}}
    assert normalize_docling(data, {1: geometry}, page_box="crop", coordinate_frame="display")[0].rect == (40, 40, 60, 100)
