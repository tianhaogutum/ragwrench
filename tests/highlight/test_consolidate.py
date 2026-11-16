import pytest

from ragwrench.highlight.consolidate import assign_chunks, consolidate
from ragwrench.highlight.core import CiteError, Region, read_json, read_regions


def region(ref, rect, page=1, chunk="chunk"):
    return Region(chunk, ref, page, rect, ref)


def test_same_line_merge_no_cross_page_or_column_or_chunk():
    regions = [region("a", (0, 10, 30, 20)), region("b", (34, 10, 60, 20)),
               region("c", (100, 10, 130, 20)), region("d", (0, 30, 30, 40)),
               region("e", (0, 10, 30, 20), page=2), region("f", (0, 10, 30, 20), chunk="other")]
    result = consolidate(regions)
    assert len(result) == 5
    merged = next(r for r in result if len(r.source_refs) == 2)
    assert merged.rect == (0, 10, 60, 20)
    assert merged.source_refs == ("a", "b")
    assert merged.text == "a b"
    assert consolidate(result) == result


def test_union_per_page():
    result = consolidate([region("a", (0, 0, 10, 10)), region("b", (20, 20, 40, 40)),
                          region("c", (0, 0, 10, 10), page=2)], per_page=True)
    assert len(result) == 2
    assert result[0].rect == (0, 0, 40, 40)
    assert result[1].page == 2


@pytest.mark.parametrize("data", [
    {"summary": ["a", "b"]},
    [{"chunk_id": "summary", "meta": {"doc_items": [{"self_ref": "a"}, {"self_ref": "b"}]}}],
    [{"element_id": "summary", "element_ids": ["a", "b"]}],
    [{"element_id": "summary", "metadata": {"element_ids": ["a", "b"]}}],
    [{"element_id": "summary", "metadata": {"orig_elements": [{"element_id": "a"}, {"element_id": "b"}]}}],
])
def test_chunk_mapping_dialects(data):
    result = assign_chunks([region("a", (0, 0, 10, 10)), region("b", (20, 0, 30, 10), page=2)], data)
    assert len(result) == 2
    assert [r.chunk_id for r in result] == ["summary", "summary"]
    assert [r.page for r in result] == [1, 2]


def test_shared_source_multiple_chunks():
    result = assign_chunks([region("a", (0, 0, 10, 10))], {"c1": ["a"], "c2": ["a"]})
    assert [r.chunk_id for r in result] == ["c1", "c2"]


def test_docling_real_fixture(examples):
    result = assign_chunks(read_regions(examples / "regions.jsonl"), read_json(examples / "chunks.docling.json"))
    assert len(consolidate(result, per_page=True)) == 2
    assert result[0].text == "Revenue grew 12%. Costs fell 3%."


@pytest.mark.parametrize("data,match", [
    ({"c": ["missing"]}, "missing element"), ({}, "nonempty"),
    ([{"chunk_id": "c"}], "has no"),
    ([{"chunk_id": "c", "element_ids": ["a"]}] * 2, "unique"),
])
def test_bad_mapping(data, match):
    with pytest.raises(CiteError, match=match):
        assign_chunks([region("a", (0, 0, 10, 10))], data)


def test_native_docling_chunk_without_id_stable():
    source = [region("a", (0, 0, 10, 10))]
    chunks = [{"text": "a", "meta": {"doc_items": [{"self_ref": "a"}]}}]
    first = assign_chunks(source, chunks)
    assert first[0].chunk_id.startswith("docling:")
    assert assign_chunks(source, chunks) == first


def test_merge_resource_cap():
    regions = [region(str(i), (0, i * 20, 10, i * 20 + 10)) for i in range(2001)]
    with pytest.raises(CiteError, match="at most 2000"):
        consolidate(regions)
    assert len(consolidate(regions, per_page=True)) == 1


def test_mapping_resource_cap(monkeypatch):
    monkeypatch.setattr("ragwrench.highlight.consolidate.MAX_REGIONS", 1)
    with pytest.raises(CiteError, match="maximum region"):
        assign_chunks([region("a", (0, 0, 10, 10))], {"c1": ["a"], "c2": ["a"]})


def test_partial_remap_merged_box_rejected():
    merged = consolidate([region("a", (0, 0, 10, 10)), region("b", (12, 0, 20, 10))])
    with pytest.raises(CiteError, match="partially remap"):
        assign_chunks(merged, {"new": ["a"]})
    assert len(assign_chunks(merged, {"new": ["a", "b"]})) == 1
