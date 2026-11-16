"""Independent Poppler rendering catches incorrect native/display coordinate mapping."""
import shutil
import subprocess

import pytest

from ragwrench.highlight.cli import main
from ragwrench.highlight.core import read_regions, regions_jsonl


def read_ppm(path):
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"P6"
        width, height = map(int, stream.readline().split())
        assert stream.readline().strip() == b"255"
        pixels = stream.read()
    assert len(pixels) == width * height * 3
    return width, height, pixels


def test_rendered_highlights_cover_ink_at_all_rotations(examples, tmp_path):
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        pytest.skip("Poppler pdftoppm is required for the independent rendered fixture check")
    pdf = examples / "review-rotated.pdf"
    regions = tmp_path / "regions.jsonl"
    assert main(["normalize", "--format", "unstructured", str(examples / "review-rotated.unstructured.json"),
                 "--pdf", str(pdf), "-o", str(regions)]) == 0
    regions.write_text(regions_jsonl(read_regions(regions)[:4]))
    output, again = tmp_path / "highlighted.pdf", tmp_path / "again.pdf"
    assert main(["annotate", str(pdf), str(regions), "-o", str(output)]) == 0
    assert main(["annotate", str(output), str(regions), "-o", str(again), "--force"]) == 0
    for source, prefix in ((pdf, "source"), (again, "annotated")):
        subprocess.run([renderer, "-r", "72", "-cropbox", str(source), str(tmp_path / prefix)],
                       check=True, capture_output=True, timeout=30)
    # Hand-derived top-left screen boxes. No production geometry functions are
    # used for these expectations; MediaBox and CropBox origins are nonzero.
    boxes = [(50, 100, 120, 120), (80, 50, 100, 120),
             (180, 80, 250, 100), (100, 180, 120, 250)]
    for page, (left, top, right, bottom) in enumerate(boxes, 1):
        width, height, source = read_ppm(tmp_path / f"source-{page}.ppm")
        out_width, out_height, rendered = read_ppm(tmp_path / f"annotated-{page}.ppm")
        assert (width, height) == (out_width, out_height)
        assert (width, height) == ((300, 200) if page in (1, 3) else (200, 300))
        yellow, ink = [], []
        for y in range(height):
            for x in range(width):
                offset = (y * width + x) * 3
                r, g, b = rendered[offset:offset + 3]
                if r > 180 and g > 180 and r - b > 25 and g - b > 25:
                    yellow.append((x, y))
                if left <= x < right and top <= y < bottom and max(source[offset:offset + 3]) < 100:
                    ink.append((x, y))
        assert len(ink) > 30  # Actual rendered ANCHOR glyphs, not just a blank rectangle.
        assert len(yellow) > 500
        extent = (min(x for x, _ in yellow), min(y for _, y in yellow),
                  max(x for x, _ in yellow) + 1, max(y for _, y in yellow) + 1)
        assert all(abs(actual - expected) <= 5 for actual, expected in zip(extent, (left, top, right, bottom)))
        # The blue channel of yellow-highlighted black glyphs drops relative to
        # the source; dark glyph pixels remain inside the actual yellow extent.
        assert all(extent[0] <= x < extent[2] and extent[1] <= y < extent[3] for x, y in ink)
        assert sum(rendered[(y * width + x) * 3 + 2] < source[(y * width + x) * 3 + 2]
                   for y in range(top, bottom) for x in range(left, right)) > 500
