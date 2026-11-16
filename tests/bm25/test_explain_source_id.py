"""Selected source documents retain typed identity and full-corpus scoring."""

import json
import math
from contextlib import closing

import pytest

from ragwrench.bm25 import store
from ragwrench.bm25.cli import main


DOCS = [
    {"id": 7, "text": "北京 北京", "metadata": {"kind": "integer"}},
    {"id": "7", "text": "北京", "metadata": {"kind": "string"}},
    {"id": "other", "text": "東京"},
]


def test_selected_below_top_k_preserves_typed_id_and_global_idf(client):
    store.build_collection(client, "kb", DOCS)
    original = store.explain_collection(client, "kb", "北京", top_k=1)
    report = store.explain_collection(client, "kb", "北京", top_k=1, source_id="7")
    assert {k: v for k, v in report.items() if k != "selected_document"} == original
    assert [hit["id"] for hit in report["results"]] == [7]
    selected = report["selected_document"]
    assert selected["id"] == "7" and type(selected["id"]) is str
    assert selected["text"] == "北京" and selected["metadata"] == {"kind": "string"}
    assert selected == store.explain_collection(client, "kb", "北京")["results"][1]
    tf = 2.2 / (1 + 1.2 * (0.25 + 0.75 / (4 / 3)))
    assert selected["score"] == pytest.approx(tf * math.log(1.6), abs=1e-6)
    assert 0 < selected["score"] < report["results"][0]["score"]
    integer = store.explain_collection(client, "kb", "北京", source_id=7)["selected_document"]
    assert integer == report["results"][0]


def test_nonmatch_empty_and_stopword_query_are_explained_at_zero(client):
    store.build_collection(client, "kb", DOCS, stopwords=("skip",))
    for query in ("北京", "unseen", "", "ＳＫＩＰ"):
        if query in ("", "ＳＫＩＰ"):
            with pytest.warns(UserWarning, match="empty"):
                report = store.explain_collection(client, "kb", query, source_id="other")
        else:
            report = store.explain_collection(client, "kb", query, source_id="other")
        selected = report["selected_document"]
        assert selected["id"] == "other"
        assert selected["score"] == 0
        explanation = selected["explanation"]
        assert explanation["tokens"] == ["東京"]
        assert explanation["document_length"] == 1
        assert explanation["score_sum"] == explanation["score_delta"] == 0
        for dimension in explanation["dimensions"]:
            assert dimension["document_terms"] == []
            assert dimension["tf_weight"] == dimension["contribution"] == 0


@pytest.mark.parametrize("avgdl", [None, 12])
def test_selected_updates_replacements_and_deletions_match_rebuild(client, avgdl):
    store.build_collection(client, "kb", DOCS, avgdl=avgdl, stopwords=("skip",))
    replacement = {"id": "7", "text": "東京 東京 skip", "metadata": {"replaced": True}}
    store.update_collection(client, "kb", additions=[replacement], delete_ids=["7", 7])
    store.build_collection(
        client, "rebuilt", [replacement, DOCS[2]], avgdl=avgdl, stopwords=("skip",)
    )
    for query in ("北京", "東京"):
        report = store.explain_collection(client, "kb", query, top_k=1, source_id="7")
        assert report == store.explain_collection(client, "rebuilt", query, top_k=1, source_id="7")
    with pytest.raises(ValueError, match="source ID does not exist"):
        store.explain_collection(client, "kb", "北京", source_id=7)


def test_selected_pagination_and_real_hash_collision(client):
    a, b = "collision402", "collision169677"
    docs = [{"id": i, "text": "北京"} for i in range(129)]
    docs += [{"id": 2**64 - 1, "text": a}, {"id": "７/東京", "text": b}]
    with pytest.warns(UserWarning, match="hash collision"):
        store.build_collection(client, "kb", docs, tokenizer="whitespace")
    report = store.explain_collection(client, "kb", b, top_k=1, source_id=2**64 - 1)
    selected = report["selected_document"]
    assert selected["id"] == 2**64 - 1
    assert report["corpus"]["document_count"] == 131
    assert report["query"]["dimensions"][0]["document_frequency"] == 2
    assert selected["explanation"]["dimensions"][0]["document_terms"][0]["token"] == a
    assert selected["score"] == pytest.approx(math.log(1 + 129.5 / 2.5), abs=1e-6)
    assert (
        store.explain_collection(client, "kb", "北京", source_id="７/東京")["selected_document"][
            "score"
        ]
        == 0
    )


@pytest.mark.parametrize("source_id", [True, 7.0, -1, 2**64, "", "x" * 257, [], {}])
def test_invalid_source_id_rejected(client, source_id):
    store.build_collection(client, "kb", DOCS)
    with pytest.raises(ValueError, match="id must"):
        store.explain_collection(client, "kb", "北京", source_id=source_id)


def test_selected_read_only_and_failures(client, monkeypatch):
    store.build_collection(client, "kb", DOCS)
    before = client.get_collection("kb").config.metadata

    def unexpected(*args, **kwargs):
        pytest.fail("selected explanation must not write")

    for method in ("upsert", "delete", "update_collection", "set_payload", "create_collection"):
        monkeypatch.setattr(client, method, unexpected)
    assert (
        store.explain_collection(client, "kb", "北京", source_id="other")["selected_document"][
            "score"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="source ID does not exist"):
        store.explain_collection(client, "kb", "北京", source_id="unknown")
    with pytest.raises(ValueError, match="differs"):
        store.explain_collection(client, "kb", "北京", source_id=7, tokenizer="unicode")
    with pytest.raises(ValueError, match="top-k"):
        store.explain_collection(client, "kb", "北京", source_id=7, top_k=0)
    assert client.get_collection("kb").config.metadata == before


def test_selected_missing_vector_and_incomplete_collection_fail(client):
    store.build_collection(client, "kb", DOCS)
    client.delete_vectors("kb", ["bm25"], [store.point_id("other")])
    with pytest.raises(ValueError, match="sparse vector"):
        store.explain_collection(client, "kb", "北京", source_id="other")
    metadata = client.get_collection("kb").config.metadata
    metadata["qbm25"]["state"] = "updating"
    client.update_collection("kb", metadata=metadata)
    with pytest.raises(ValueError, match="incomplete"):
        store.explain_collection(client, "kb", "北京", source_id="other")


def test_cli_selected_json_ids_and_invalid_inputs(tmp_path, capsys):
    path = tmp_path / "qd"
    with closing(store.open_local(path)) as client:
        store.build_collection(client, "kb", DOCS)
    args = ["search", "--path", str(path), "--collection", "kb", "北京", "--top-k", "1"]
    for source_id in (7, "7", "other"):
        assert main([*args, "--explain-id", json.dumps(source_id)]) == 0
        selected = json.loads(capsys.readouterr().out)["selected_document"]
        assert selected["id"] == source_id and type(selected["id"]) is type(source_id)
    assert main([*args, "--explain", "--explain-id", '"7"']) == 0
    assert json.loads(capsys.readouterr().out)["selected_document"]["id"] == "7"
    for raw in (
        "7x",
        "null",
        "true",
        "7.0",
        "-1",
        str(2**64),
        '""',
        '"unknown"',
        "[]",
        "{}",
        "NaN",
        "Infinity",
        json.dumps("x" * 257),
    ):
        assert main([*args, "--explain-id=" + raw]) == 2
        output = capsys.readouterr()
        assert output.out == "" and "error:" in output.err
    assert main(args) == 0
    assert [hit["id"] for hit in json.loads(capsys.readouterr().out)] == [7]


def test_zero_id_skipped_source_and_empty_collection(client):
    with pytest.warns(UserWarning, match="skipped"):
        store.build_collection(
            client,
            "kb",
            [{"id": 0, "text": "北京 atlas"}, {"id": "0", "text": "skip"}],
            stopwords=("skip",),
        )
    report = store.explain_collection(client, "kb", "北京 atlas", source_id=0)
    assert report["selected_document"] == report["results"][0]
    assert report["selected_document"]["id"] == 0
    assert len(report["selected_document"]["explanation"]["dimensions"]) == 2
    assert report["selected_document"]["score"] == pytest.approx(2 * math.log(4 / 3), abs=1e-6)
    with pytest.raises(ValueError, match="source ID does not exist"):
        store.explain_collection(client, "kb", "北京", source_id="0")
    store.update_collection(client, "kb", delete_ids=[0])
    with pytest.raises(ValueError, match="source ID does not exist"):
        store.explain_collection(client, "kb", "北京", source_id=0)
