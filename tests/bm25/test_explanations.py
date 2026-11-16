"""Real local-Qdrant explanations checked against hand-derived BM25 values."""

import json
import math
from contextlib import closing

import pytest

from ragwrench.bm25 import store
from ragwrench.bm25.cli import main
from ragwrench.bm25.encoder import token_hash


def test_hand_explanation_and_typed_id_ranking(client):
    docs = [
        {"id": 7, "text": "北京 北京 atlas", "metadata": {"kind": "integer"}},
        {"id": "7", "text": "北京 atlas atlas", "metadata": {"kind": "string"}},
        {"id": "other", "text": "atlas"},
    ]
    store.build_collection(client, "kb", docs)
    query = "ＢＥＩＪＩＮＧ 北京 北京"
    plain = store.search_collection(client, "kb", query)
    report = store.explain_collection(client, "kb", query)
    assert report["query"]["tokens"] == ["beijing", "北京", "北京"]
    assert report["query"]["term_frequencies"] == {"beijing": 1, "北京": 2}
    assert report["corpus"]["document_count"] == 3
    assert report["config"]["avgdl"] == 7 / 3
    dimensions = {d["index"]: d for d in report["query"]["dimensions"]}
    assert dimensions[token_hash("北京")] == {
        "index": token_hash("北京"),
        "tokens": ["北京"],
        "query_weight": 1.0,
        "document_frequency": 2,
        "idf": pytest.approx(math.log(1.6)),
    }
    assert dimensions[token_hash("beijing")]["document_frequency"] == 0
    assert [h["id"] for h in report["results"]] == [7, "7"]
    for hit, original, frequency in zip(report["results"], plain, [2, 1]):
        assert {k: v for k, v in hit.items() if k != "explanation"} == original
        explanation = hit["explanation"]
        assert explanation["document_length"] == 3
        assert explanation["term_frequencies"]["北京"] == frequency
        term = next(d for d in explanation["dimensions"] if d["index"] == token_hash("北京"))
        tf = frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * 3 / (7 / 3)))
        assert term["document_terms"] == [
            {"token": "北京", "frequency": frequency, "tf_weight": pytest.approx(tf)}
        ]
        assert term["tf_weight"] == pytest.approx(tf)
        assert term["contribution"] == pytest.approx(tf * math.log(1.6))
        assert explanation["score_sum"] == pytest.approx(hit["score"], abs=1e-6)
        assert explanation["score_delta"] == explanation["score_sum"] - hit["score"]
        assert abs(explanation["score_delta"]) < 1e-6


def test_real_collision_explains_false_match_and_merged_query(client):
    # Two real SHA-256-prefix collisions; no monkeypatched hash/adapter.
    a, b = "collision402", "collision169677"
    docs = [{"id": 7, "text": f"{a} {a}"}, {"id": "7", "text": f"{a} {b}"}]
    with pytest.warns(UserWarning, match="hash collision"):
        store.build_collection(client, "kb", docs, tokenizer="whitespace", b=0)
    report = store.explain_collection(client, "kb", b)
    assert report["query"]["dimensions"][0]["document_frequency"] == 2
    assert [h["id"] for h in report["results"]] == ["7", 7]
    false_match = report["results"][1]["explanation"]
    assert b not in false_match["term_frequencies"]
    assert false_match["dimensions"][0]["document_terms"][0]["token"] == a
    for hit in report["results"]:
        assert hit["explanation"]["score_sum"] == pytest.approx(hit["score"], abs=1e-6)
    merged = report["results"][0]["explanation"]["dimensions"][0]
    assert merged["tf_weight"] == 2.0
    assert {t["token"] for t in merged["document_terms"]} == {a, b}
    with pytest.warns(UserWarning, match="hash collision"):
        both = store.explain_collection(client, "kb", f"{a} {b} {b}")
    assert len(both["query"]["dimensions"]) == 1
    assert both["query"]["dimensions"][0]["tokens"] == sorted([a, b])
    assert both["query"]["dimensions"][0]["query_weight"] == 1.0
    assert both["results"] == report["results"]


@pytest.mark.parametrize("params", [{}, {"avgdl": 12, "b": 0}, {"k1": 2, "b": 1}])
@pytest.mark.parametrize("tokenizer", ["whitespace", "unicode", "cjk-bigram"])
def test_updated_explanations_match_rebuild(client, tokenizer, params):
    docs = [{"id": 7, "text": "数据库 数据库 skip"}, {"id": "7", "text": "数据库 备份"}]
    addition = {"id": "new", "text": "数据库 快照 恢复 备份"}
    options = {"tokenizer": tokenizer, "stopwords": ("ＳＫＩＰ",), **params}
    store.build_collection(client, "kb", docs, **options)
    store.update_collection(client, "kb", additions=[addition], delete_ids=[7])
    store.build_collection(client, "rebuilt", [docs[1], addition], **options)
    query = "数据库 数据库 备份 ＳＫＩＰ"
    report = store.explain_collection(client, "kb", query)
    assert report == store.explain_collection(client, "rebuilt", query)
    assert "skip" not in report["query"]["tokens"]
    for hit in report["results"]:
        explanation = hit["explanation"]
        assert explanation["score_sum"] == pytest.approx(hit["score"], abs=1e-6)
        assert sum(d["contribution"] for d in explanation["dimensions"]) == pytest.approx(
            explanation["score_sum"]
        )


def test_empty_no_match_and_failure_behavior(client):
    store.build_collection(client, "kb", [{"id": 7, "text": "北京 skip"}], stopwords=("skip",))
    for query in ("", "ＳＫＩＰ"):
        with pytest.warns(UserWarning, match="empty"):
            report = store.explain_collection(client, "kb", query)
        assert report["query"]["tokens"] == []
        assert report["query"]["dimensions"] == []
        assert report["results"] == []
    assert store.explain_collection(client, "kb", "missing")["results"] == []
    with pytest.raises(ValueError, match="differs"):
        store.explain_collection(client, "kb", "北京", tokenizer="unicode")
    with pytest.raises(ValueError, match="top-k"):
        store.explain_collection(client, "kb", "北京", top_k=0)
    store.update_collection(client, "kb", delete_ids=[7])
    assert store.explain_collection(client, "kb", "北京")["corpus"]["document_count"] == 0
    metadata = client.get_collection("kb").config.metadata
    metadata["qbm25"]["state"] = "updating"
    client.update_collection("kb", metadata=metadata)
    with pytest.raises(ValueError, match="incomplete"):
        store.explain_collection(client, "kb", "北京")


def test_pagination_and_top_k_do_not_change_corpus_idf(client):
    docs = [{"id": i, "text": "北京" if i < 129 else "東京"} for i in range(130)]
    store.build_collection(client, "kb", docs)
    report = store.explain_collection(client, "kb", "北京", top_k=1)
    assert len(report["results"]) == 1
    assert report["corpus"]["document_count"] == 130
    dimension = report["query"]["dimensions"][0]
    assert dimension["document_frequency"] == 129
    assert dimension["idf"] == pytest.approx(math.log(1 + 1.5 / 129.5))
    assert report["results"][0]["explanation"]["score_sum"] == pytest.approx(
        report["results"][0]["score"]
    )


def test_cli_explained_json_and_persistence(tmp_path, capsys):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(
            client, "kb", [{"id": 7, "text": "北京 北京"}, {"id": "7", "text": "北京"}]
        )
    args = ["search", "--path", str(path), "--collection", "kb", "北京 北京"]
    assert main(args) == 0
    plain = json.loads(capsys.readouterr().out)
    assert main([*args, "--explain"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [{k: v for k, v in h.items() if k != "explanation"} for h in report["results"]] == plain
    assert main([*args, "--explain", "--tokenizer", "unicode"]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "differs" in output.err


def test_missing_stored_vector_fails_explanation(client):
    store.build_collection(client, "kb", [{"id": "a", "text": "北京"}])
    client.delete_vectors("kb", ["bm25"], [store.point_id("a")])
    with pytest.raises(ValueError, match="sparse vector"):
        store.explain_collection(client, "kb", "北京")


@pytest.mark.parametrize(
    "query,tokens",
    [("𠮷野家", ["𠮷野", "野家"]), ("カタカナ", ["カタ", "タカ", "カナ"]), ("서울", ["서울"])],
)
def test_multilingual_normalized_tokens(client, query, tokens):
    store.build_collection(client, "kb", [{"id": 7, "text": "𠮷野家 ｶﾀｶﾅ 서울 서울"}])
    report = store.explain_collection(client, "kb", query)
    assert report["query"]["tokens"] == tokens
    hit = report["results"][0]
    assert hit["id"] == 7
    assert hit["explanation"]["score_sum"] == pytest.approx(hit["score"], abs=1e-6)


def test_jieba_explanation(client):
    pytest.importorskip("jieba")
    store.build_collection(
        client,
        "kb",
        [{"id": 7, "text": "我来到北京清华大学。"}],
        tokenizer="jieba",
        stopwords=("我",),
    )
    report = store.explain_collection(client, "kb", "清华大学 清华大学 我")
    assert report["config"]["jieba_version"] == "0.42.1"
    assert report["query"]["tokens"] == ["清华大学", "清华大学"]
    hit = report["results"][0]
    assert hit["explanation"]["tokens"] == ["来到", "北京", "清华大学"]
    assert hit["explanation"]["score_sum"] == pytest.approx(hit["score"], abs=1e-6)


def test_explanation_is_read_only_and_supports_legacy_metadata(client, monkeypatch):
    store.build_collection(client, "kb", [{"id": 7, "text": "北京"}])
    metadata = client.get_collection("kb").config.metadata
    del metadata["qbm25"]["corpus"]
    client.update_collection("kb", metadata={"qbm25": None})
    client.update_collection("kb", metadata=metadata)

    def unexpected(*args, **kwargs):
        pytest.fail("explanation must not write")

    for method in ("upsert", "delete", "update_collection", "set_payload", "create_collection"):
        monkeypatch.setattr(client, method, unexpected)
    assert store.explain_collection(client, "kb", "北京")["results"][0]["id"] == 7
