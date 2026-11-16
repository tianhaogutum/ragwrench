"""Regenerate chunks.jsonl, a deliberately broken export of handbook.md.

Run from any directory: python examples/audit/generate.py

The chunks imitate what a fixed-size splitter plus a flaky loader produce: a code
block and two tables cut in half, an orphan heading, a list and a sentence split
across chunks, a dropped section and a re-ingested duplicate. All content is
fictional; nothing is downloaded.
"""
import json
from pathlib import Path

# Each chunk starts at this text; ``None`` marks a section the "loader" dropped.
CUTS = [
    "# Lanternfish",
    "## Install",
    "lanternfish queues list",
    "## Rate limits",
    "| Business",
    "## Rotating API keys",
    "1. Create",
    "4. Revoke",
    "fails validation",
    "| LF-429",
    None,  # "## Webhooks" is lost: source coverage drops
    "## Support",
]
DROPPED = "## Webhooks"


def generate():
    directory = Path(__file__).resolve().parent
    source = (directory / "handbook.md").read_text(encoding="utf-8")
    starts = [source.index(DROPPED if cut is None else cut) for cut in CUTS] + [len(source)]
    records = []
    for number, (cut, start, end) in enumerate(zip(CUTS, starts, starts[1:]), 1):
        if cut is None:
            continue
        records.append({"id": f"lf-{number:02d}", "page_content": source[start:end],
                        "metadata": {"source": "handbook.md", "splitter": "fixed-size-demo"}})
    records.append({**records[-1], "id": "lf-12-reingested"})
    (directory / "chunks.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


if __name__ == "__main__":
    generate()
