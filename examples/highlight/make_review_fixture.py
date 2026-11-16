"""Regenerate the synthetic rotated/cropped preflight fixture; no renderer needed."""
import json
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, RectangleObject


def main():
    root = Path(__file__).resolve().parent
    writer = PdfWriter()
    for rotation in (0, 90, 180, 270):
        page = writer.add_blank_page(width=400, height=300)
        page.mediabox = RectangleObject((-20, -30, 380, 270))
        page.cropbox = RectangleObject((30, 40, 330, 240))
        page.rotate(rotation)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 1 0 0 1 83 123 Tm (ANCHOR) Tj ET\n"
                        b"BT /F1 12 Tf 1 0 0 1 223 123 Tm (ANCHOR) Tj ET\n"
                        b"BT /F1 12 Tf 1 0 0 1 83 183 Tm (ALPHA) Tj ET\n")
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(root / "review-rotated.pdf")
    # Hand-derived top-left displayed rectangles, at 2 pixels per point.
    display = [(50, 100, 120, 120), (80, 50, 100, 120),
               (180, 80, 250, 100), (100, 180, 120, 250)]
    elements = []
    for index, (rotation, rect) in enumerate(zip((0, 90, 180, 270), display), 1):
        width, height = (300, 200) if rotation in (0, 180) else (200, 300)
        elements.append(element(f"good-{rotation}", index, "ANCHOR", rect, width, height))
    elements.extend([
        element("ambiguous", 2, "ANCHOR", (80, 50, 100, 260), 200, 300),
        element("blank", 2, "MISSING", (20, 50, 40, 120), 200, 300),
        element("mismatch", 2, "WRONG", (140, 50, 160, 120), 200, 300),
    ])
    (root / "review-rotated.unstructured.json").write_text(json.dumps(elements, indent=2) + "\n")


def element(ref, page, text, rect, width, height):
    return {"element_id": ref, "text": text, "metadata": {"page_number": page, "coordinates": {
        "system": "PixelSpace", "layout_width": width * 2, "layout_height": height * 2,
        "points": [[rect[0] * 2, rect[1] * 2], [rect[2] * 2, rect[3] * 2]]}}}


if __name__ == "__main__":
    main()
