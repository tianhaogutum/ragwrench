"""Regenerate the tiny, synthetic, redistributable fixtures (run from repo root)."""
import base64
import json
import zlib
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject


def make_pdf(path, lines=("Revenue grew 12%.", "Costs fell 3%.")):
    writer = PdfWriter()
    for line in lines:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 1 0 0 1 72 688 Tm ({escaped}) Tj ET\n".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with Path(path).open("wb") as output:
        writer.write(output)


def main():
    root = Path(__file__).resolve().parent
    make_pdf(root / "sample.pdf")
    texts = ["Revenue grew 12%.", "Costs fell 3%."]
    docling = {"schema_name": "DoclingDocument", "version": "1.0.0", "name": "Synthetic quarterly report",
               "texts": [{"self_ref": f"#/texts/{i}", "label": "text", "text": text,
                          "prov": [{"page_no": i + 1,
                                    "bbox": {"l": 72, "t": 700, "r": 300, "b": 686,
                                             "coord_origin": "BOTTOMLEFT"},
                                    "charspan": [0, len(text)]}]} for i, text in enumerate(texts)],
               "pages": {str(p): {"page_no": p, "size": {"width": 612, "height": 792}} for p in (1, 2)}}
    elements = [{"type": "NarrativeText", "element_id": f"element-{i}", "text": text,
                 "metadata": {"page_number": i + 1, "coordinates": {
                     "system": "PixelSpace", "layout_width": 1700, "layout_height": 2200,
                     "points": [[200, 255.55555555555554], [200, 294.44444444444446],
                                [833.3333333333334, 294.44444444444446], [833.3333333333334, 255.55555555555554]]}}}
                for i, text in enumerate(texts)]
    compressed = base64.b64encode(zlib.compress(json.dumps(elements).encode())).decode()
    files = {"sample.docling.json": docling, "sample.unstructured.json": elements,
             "chunks.docling.json": [{"chunk_id": "quarterly-summary", "text": " ".join(texts),
                                      "meta": {"doc_items": [{"self_ref": f"#/texts/{i}"} for i in (0, 1)]}}],
             "chunks.unstructured.json": [{"element_id": "quarterly-summary", "text": " ".join(texts),
                                           "metadata": {"orig_elements": compressed}}]}
    for name, data in files.items():
        (root / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    regions = [{"chunk_id": f"#/texts/{i}", "source_ref": f"#/texts/{i}", "page": i + 1,
                "rect": [72.0, 686.0, 300.0, 700.0], "text": text} for i, text in enumerate(texts)]
    (root / "regions.jsonl").write_text("\n".join(json.dumps(r) for r in regions) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
