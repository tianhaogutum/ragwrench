import json
import sys

import pytest
from qdrant_client import models

from ragwrench.sync.core import SyncError, build_snapshot, make_plan
from ragwrench.sync.embedders import CommandEmbedder, HashEmbedder, PrecomputedEmbedder
from ragwrench.sync.review import review_plan
from ragwrench.sync.store import apply_plan, local_client, verify_snapshot


def snapshot(*texts, **kwargs):
    return build_snapshot([{"doc_id": "a", "text": t} for t in texts], **kwargs)


def initialize(client, old, embedder):
    return apply_plan(client, "kb", make_plan(build_snapshot([]), old), embedder)


def test_persistent_roundtrip_move_payload_delete_and_idempotence(tmp_path):
    old = build_snapshot([{"doc_id": "old", "text": "repeat"}, {"doc_id": "old", "text": "repeat"},
                          {"doc_id": "old", "text": "remove"}, {"doc_id": "same", "text": "payload", "metadata": {"x": 1}}])
    new = build_snapshot([{"doc_id": "renamed", "text": "repeat"}, {"doc_id": "renamed", "text": "repeat"},
                          {"doc_id": "new", "text": "added"}, {"doc_id": "same", "text": "payload", "metadata": {"x": 2}}])
    plan = make_plan(old, new)
    with local_client(str(tmp_path / "qd")) as client:
        embedder = HashEmbedder()
        initialize(client, old, embedder)
        before = embedder.calls
        assert apply_plan(client, "kb", plan, embedder)["embedded"] == 1
        assert embedder.calls == before + 1
        assert client.count("kb").count == 4
    with local_client(str(tmp_path / "qd")) as client:
        embedder = HashEmbedder()
        assert apply_plan(client, "kb", plan, embedder)["status"] == "already_applied"
        assert embedder.calls == 0
        assert verify_snapshot(client, "kb", new, embedder)["vectors_recomputed"] == 4


def test_payload_only_never_embeds():
    old = build_snapshot([{"doc_id": "a", "text": "a", "metadata": {"x": 1}}])
    new = build_snapshot([{"doc_id": "a", "text": "a", "metadata": {"x": 2}}])
    with local_client(":memory:") as client:
        embedder = HashEmbedder()
        initialize(client, old, embedder)
        embedder.embed = lambda *args: pytest.fail("payload change called embedder")
        assert apply_plan(client, "kb", make_plan(old, new), embedder)["embedded"] == 0
        assert client.retrieve("kb", [new["chunks"][0]["id"]])[0].payload["metadata"] == {"x": 2}


def test_model_change_apply():
    old, new = snapshot("a", "b"), snapshot("a", "b", embed_model="changed")
    with local_client(":memory:") as client:
        embedder = HashEmbedder()
        initialize(client, old, embedder)
        plan = make_plan(old, new)
        review = review_plan(plan, overrides=["embed_model_changed"], reviewer="test", reason="intentional model migration")
        assert apply_plan(client, "kb", plan, embedder, review=review)["embedded"] == 2
        verify_snapshot(client, "kb", new, embedder)


def test_empty_collection_is_managed_and_verifiable():
    empty = build_snapshot([])
    with local_client(":memory:") as client:
        embedder = HashEmbedder()
        apply_plan(client, "kb", make_plan(empty, empty), embedder)
        assert verify_snapshot(client, "kb", empty, embedder)["points"] == 0


def test_resume_actual_command_failure(tmp_path):
    # A real child process fails exactly once on its second input, across
    # Qdrant close/reopen. Its argv identity stays the same on resume.
    script = tmp_path / "embed.py"
    counter = tmp_path / "counter"
    script.write_text('''import json, sys
from pathlib import Path
from ragwrench.sync.embedders import HashEmbedder
r = json.loads(sys.stdin.readline())
p = Path(sys.argv[1])
n = int(p.read_text()) + 1 if p.exists() else 1
p.write_text(str(n))
if n == 3:
    print("sensitive token should not be echoed", file=sys.stderr)
    sys.exit(23)
v = HashEmbedder(8).embed(r, r["embed_model"])
print(json.dumps({"id": r["id"], "vector": v}))
''')
    command = [sys.executable, str(script), str(counter)]
    old, new = snapshot("old"), snapshot("new1", "new2", "new3")
    plan = make_plan(old, new)
    with local_client(str(tmp_path / "db")) as client:
        embedder = CommandEmbedder(command, 8)
        initialize(client, old, embedder)  # command call 1
        with pytest.raises(SyncError, match="code 23") as err:
            apply_plan(client, "kb", plan, embedder)
        assert "sensitive" not in str(err.value)
        assert client.count("kb").count == 2  # old point not deleted
        with pytest.raises(SyncError, match="different plan"):
            apply_plan(client, "kb", make_plan(old, snapshot("other")), embedder)
    with local_client(str(tmp_path / "db")) as client:
        embedder = CommandEmbedder(command, 8)
        result = apply_plan(client, "kb", plan, embedder)
        assert result["embedded"] == 2  # successful new1 did not run again
        assert result["points"] == 3
        verify_snapshot(client, "kb", new, embedder)


@pytest.mark.parametrize("tamper", ["payload", "vector", "extra", "missing"])
def test_refuse_external_collection_changes(tamper):
    old, new = snapshot("a"), snapshot("a", "b")
    with local_client(":memory:") as client:
        embedder = HashEmbedder()
        initialize(client, old, embedder)
        pid = old["chunks"][0]["id"]
        if tamper == "payload":
            client.set_payload("kb", payload={"extra": 1}, points=[pid])
        elif tamper == "vector":
            point = client.retrieve("kb", [pid])[0]
            client.upsert("kb", points=[models.PointStruct(id=pid, vector=[0.] * 32, payload=point.payload)])
        elif tamper == "extra":
            client.upsert("kb", points=[models.PointStruct(id=123, vector=[0.] * 32)])
        else:
            client.delete("kb", points_selector=[pid])
        with pytest.raises(SyncError, match="collection"):
            apply_plan(client, "kb", make_plan(old, new), embedder)
        with pytest.raises(SyncError):
            verify_snapshot(client, "kb", old, embedder)


def test_wrong_base_dimension_and_embedder():
    old = snapshot("a")
    with local_client(":memory:") as client:
        embedder = HashEmbedder()
        initialize(client, old, embedder)
        with pytest.raises(SyncError, match="different snapshot"):
            apply_plan(client, "kb", make_plan(snapshot("wrong"), snapshot("new")), embedder)
        with pytest.raises(SyncError, match="dimension"):
            apply_plan(client, "kb", make_plan(old, snapshot("new")), HashEmbedder(8))
        other = HashEmbedder()
        other.signature = "different"
        with pytest.raises(SyncError, match="embedder fingerprint"):
            apply_plan(client, "kb", make_plan(old, snapshot("new")), other)


def test_missing_and_unmanaged_collection():
    with local_client(":memory:") as client:
        embedder = HashEmbedder()
        with pytest.raises(SyncError, match="collection missing"):
            apply_plan(client, "kb", make_plan(snapshot("a"), snapshot("b")), embedder)
        client.create_collection("kb", vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT))
        client.upsert("kb", points=[models.PointStruct(id=1, vector=[0.] * 32)])
        with pytest.raises(SyncError, match="unmanaged"):
            initialize(client, snapshot("a"), embedder)


def test_independent_verify_catches_bad_embedder():
    old = snapshot("a")
    with local_client(":memory:") as client:
        embedder = HashEmbedder()
        initialize(client, old, embedder)
        embedder.embed = lambda *args: [0.] * 32
        with pytest.raises(SyncError, match="from-scratch"):
            verify_snapshot(client, "kb", old, embedder)


@pytest.mark.parametrize("name", ["../escape", "/tmp/out", ".", "a/b", "", "a\\b"])
def test_collection_path_traversal_refused(name):
    with local_client(":memory:") as client:
        with pytest.raises(SyncError, match="collection must"):
            apply_plan(client, name, make_plan(build_snapshot([]), snapshot("a")), HashEmbedder())


def test_store_symlink_refused(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(SyncError, match="symlink"):
        with local_client(str(tmp_path / "link")):
            pass


def test_precomputed_roundtrip_and_refusals(tmp_path):
    new = snapshot("a", "b")
    vectors = tmp_path / "vectors.jsonl"
    rows = [{"id": c["id"], "vector": [0.1, 0.2]} for c in new["chunks"]]
    vectors.write_text("\n".join(json.dumps(r) for r in rows))
    with local_client(":memory:") as client:
        embedder = PrecomputedEmbedder(vectors, 2)
        initialize(client, new, embedder)
        verify_snapshot(client, "kb", new, embedder)
    vectors.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]))
    with pytest.raises(SyncError, match="duplicate"):
        PrecomputedEmbedder(vectors, 2)
    vectors.write_text(json.dumps(rows[0]))
    with pytest.raises(SyncError, match="dimension"):
        PrecomputedEmbedder(vectors, 3)
    with pytest.raises(SyncError, match="missing"):
        PrecomputedEmbedder(vectors, 2).embed(new["chunks"][1], "hash-v1")


@pytest.mark.parametrize("program, message", [
    ('print("not-json")', "invalid JSONL"),
    ('print("{}")', "matching id"),
    ('import json,sys; r=json.load(sys.stdin); print(json.dumps({"id":r["id"],"vector":[1]}))', "dimension"),
    ('print("{}\\n{}")', "exactly one"),
    ('import time; time.sleep(2)', "timeout"),
])
def test_command_protocol_failure(program, message):
    embedder = CommandEmbedder([sys.executable, "-c", program], 2, timeout=0.15 if message == "timeout" else 5)
    with pytest.raises(SyncError, match=message):
        embedder.embed(snapshot("a")["chunks"][0], "hash-v1")


def test_resume_after_delete_crash_and_rename_upsert(tmp_path):
    old = build_snapshot([{"doc_id": "old", "text": "same"}, {"doc_id": "deleted", "text": "gone"}])
    new = build_snapshot([{"doc_id": "new", "text": "same"}])
    plan = make_plan(old, new)
    with local_client(str(tmp_path / "db")) as client:
        embedder = HashEmbedder()
        initialize(client, old, embedder)
        original_delete = client.delete

        def crash_after_delete(*args, **kwargs):
            original_delete(*args, **kwargs)
            raise RuntimeError("injected crash after delete persisted")

        client.delete = crash_after_delete
        with pytest.raises(RuntimeError, match="injected"):
            apply_plan(client, "kb", plan, embedder)
        # Retirement is journaled per point. The first delete persisted, while
        # the move's source remains available until its own retirement step.
        remaining, _ = client.scroll("kb")
        assert {str(p.id) for p in remaining} == {new["chunks"][0]["id"], plan["moved"][0]["before"]["id"]}
    with local_client(str(tmp_path / "db")) as client:
        embedder = HashEmbedder()
        assert apply_plan(client, "kb", plan, embedder)["embedded"] == 0
        verify_snapshot(client, "kb", new, embedder)


def test_resume_after_upsert_before_checkpoint(tmp_path):
    old, new = snapshot("old"), snapshot("new1", "new2")
    plan = make_plan(old, new)
    with local_client(str(tmp_path / "db")) as client:
        embedder = HashEmbedder()
        initialize(client, old, embedder)
        original_upsert = client.upsert

        def crash_after_upsert(*args, **kwargs):
            original_upsert(*args, **kwargs)
            raise RuntimeError("injected crash after upsert persisted")

        client.upsert = crash_after_upsert
        with pytest.raises(RuntimeError, match="injected"):
            apply_plan(client, "kb", plan, embedder)
    with local_client(str(tmp_path / "db")) as client:
        embedder = HashEmbedder()
        assert apply_plan(client, "kb", plan, embedder)["embedded"] == 1
        verify_snapshot(client, "kb", new, embedder)
