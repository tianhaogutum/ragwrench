from pathlib import Path
import pytest
from ragwrench.tablechunk import split_document, count_tokens, TableChunkError
from ragwrench.tablechunk.markdown import parse_markdown
from ragwrench.tablechunk.models import Table


def document(n=37):
    return "# Catalog\n\n## Pricing\n\nAnnual rates.\n\n| Plan | Seats | Price |\n|---|---|---|\n" + "\n".join(f"| ID{i:03d} | {i + 1} | ${i * 5} |" for i in range(n))


@pytest.mark.parametrize("mode,budget", [("markdown", 65), ("kv", 65), ("html", 160)])
def test_row_invariants_and_budget(mode, budget):
    chunks = split_document(document(), max_tokens=budget, mode=mode)
    tables = [c for c in chunks if c.metadata["kind"] == "table"]
    assert len(tables) > 1
    row_indices = []
    for chunk in tables:
        m = chunk.metadata
        assert all(col in chunk.text for col in ("Plan", "Seats", "Price"))
        assert "Catalog" in chunk.text and "Pricing" in chunk.text and "Annual rates." in chunk.text
        assert m["header_path"] == ["Catalog", "Pricing"]
        assert m["token_count"] == count_tokens(chunk.text) <= budget
        assert not m.get("fragment")
        row_indices.extend(range(m["row_start"], m["row_end"]))
        assert m["line_start"] == 9 + m["row_start"]
        assert m["line_end"] == 8 + m["row_end"]
    assert row_indices == list(range(37))
    for index in range(37):
        assert sum(c.text.count(f"ID{index:03d}") for c in tables) == 1


def test_markdown_content_roundtrip_for_plain_values():
    chunks = split_document(document(12), max_tokens=65)
    rows = []
    for chunk in chunks:
        if chunk.metadata["kind"] == "table":
            table = next(b for b in parse_markdown(chunk.text) if isinstance(b, Table))
            rows.extend(table.rows)
    original = next(b for b in parse_markdown(document(12)) if isinstance(b, Table))
    assert rows == original.rows


def test_oversized_row_cell_fragments_are_lossless():
    values = ["alpha beta gamma " * 30, "other value " * 25]
    text = "| A | B |\n|---|---|\n| " + " | ".join(values) + " |"
    chunks = split_document(text, max_tokens=40)
    assert len(chunks) > 4
    recovered = {0: [], 1: []}
    for chunk in chunks:
        meta = chunk.metadata
        assert meta["warnings"] == ["oversized_row_cell_split"]
        assert (meta["row_start"], meta["row_end"]) == (0, 1)
        fragment = meta["fragment"]
        col = fragment["cell_index"]
        recovered[col].append((fragment["char_start"], fragment["char_end"]))
        assert count_tokens(chunk.text) <= 40
    for col, spans in recovered.items():
        assert spans[0][0] == 0
        assert spans[-1][1] == len(values[col].strip())
        assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
        # Fragments encode precisely these source substrings, modulo cell-edge whitespace.
        original = values[col].strip()
        fragments = [parse_markdown(c.text)[0].rows[0][col] for c in chunks if c.metadata["fragment"]["cell_index"] == col]
        assert fragments == [original[a:b].strip() for a, b in spans]


def test_kv_and_adapters():
    chunk = split_document("| Name | Value |\n|---|---|\n| x | 42 |", mode="kv")[0]
    assert "Name: x\nValue: 42" in chunk.text
    assert chunk.as_dict("langchain") == {"id": chunk.id, "page_content": chunk.text, "metadata": chunk.metadata}
    assert chunk.as_dict("llamaindex") == {"id_": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
    assert chunk.as_dict() == {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
    with pytest.raises(TableChunkError):
        chunk.as_dict("unknown")


def test_prose_packing_headings_and_long_paragraphs():
    text = "# Root\n\nFirst paragraph.\n\nSecond paragraph.\n\n## Child\n\n" + "Sentence words. " * 100 + "\n\n## Sibling\n\nTail."
    chunks = split_document(text, max_tokens=30)
    assert chunks[0].text == "Root\nFirst paragraph.\n\nSecond paragraph."
    assert chunks[-1].metadata["header_path"] == ["Root", "Sibling"]
    assert all(c.metadata["token_count"] <= 30 for c in chunks)
    long_content = "".join(c.text.removeprefix("Root > Child\n") for c in chunks if c.metadata["header_path"] == ["Root", "Child"])
    assert long_content == "Sentence words. " * 100


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": True},
                                    {"mode": "invalid"}, {"input_format": "pdf"},
                                    {"counter": lambda _: -1}, {"counter": lambda _: 1.5}])
def test_invalid_arguments(kwargs):
    with pytest.raises(TableChunkError):
        split_document("some text", **kwargs)


def test_impossible_context_and_headers():
    with pytest.raises(TableChunkError, match="headers exceed"):
        split_document("| Many words in header | Another long heading |\n|---|---|\n| x | y |", max_tokens=5)
    with pytest.raises(TableChunkError, match="budget too small"):
        split_document("# Many heading words\n\nBody", max_tokens=1)


def test_empty_and_header_only_table():
    assert split_document("") == []
    assert split_document("# Heading")[0].text == "Heading"
    assert split_document("| A | B |\n|---|---|")[0].metadata["row_end"] == 0


def test_empty_sibling_sections_and_trailing_headings_preserved():
    chunks = split_document("# Root\n## Empty\n## Content\n\nBody\n\n## Trailing")
    assert [c.text for c in chunks] == ["Root > Empty", "Root > Content\nBody", "Root > Trailing"]


def test_cjk_counter_and_custom_counter():
    assert count_tokens("中文 abc 123!") == 5
    assert count_tokens("你好世界") == 4
    chunks = split_document("| A |\n|---|\n| " + "汉字" * 80 + " |", max_tokens=30)
    assert all(c.metadata["token_count"] <= 30 for c in chunks)
    chunks = split_document("One two three four five", counter=len, max_tokens=10)
    assert "".join(c.text for c in chunks) == "One two three four five"
    assert all(c.metadata["token_count"] <= 10 for c in chunks)


def test_example_e2e():
    text = (Path(__file__).parents[2] / "examples" / "table-chunk" / "report.md").read_text()
    chunks = split_document(text, max_tokens=120)
    tables = [c for c in chunks if c.metadata["kind"] == "table"]
    assert len(tables) > 1
    assert sum(c.metadata["row_end"] - c.metadata["row_start"] for c in tables) == 12
    assert any(r"Analytics \| audit" in c.text for c in tables)


def test_chunk_limit_is_an_explicit_error(monkeypatch):
    import ragwrench.tablechunk.splitter as splitter
    monkeypatch.setattr(splitter, "MAX_CHUNKS", 2)
    with pytest.raises(TableChunkError, match="output exceeds"):
        split_document(document(10), max_tokens=42)


def test_fenced_code_split_warns_and_preserves_content():
    text = "```\n" + "some words\n" * 30 + "```"
    chunks = split_document(text, max_tokens=20)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == text
    assert all("code_block_split" in c.metadata["warnings"] for c in chunks)


def test_seeded_mixed_rows_preserved_with_multiple_budgets():
    import random
    rng = random.Random(24)
    rows = [[f"R{i:03d}", " ".join(["value"] * rng.randint(1, 6)), str(rng.randint(1, 999))] for i in range(50)]
    text = "# Test\n\n| ID | Notes | Value |\n|---|---|---|\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    for budget in (40, 67, 120):
        tables = [c for c in split_document(text, max_tokens=budget) if c.metadata["kind"] == "table"]
        parsed = [row for c in tables for row in next(b for b in parse_markdown(c.text) if isinstance(b, Table)).rows]
        assert parsed == rows
        assert all(count_tokens(c.text) <= budget for c in tables)
