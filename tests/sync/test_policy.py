"""Persistent collection policy contracts, across separate apply invocations."""
from copy import deepcopy

import pytest

from ragwrench.sync.core import SyncError, build_snapshot, make_plan
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.policy import policy_override, validate_policy_override
from ragwrench.sync.review import review_plan
from ragwrench.sync.store import apply_plan, local_client, verify_snapshot


def snapshot(*texts):
    return build_snapshot([{"doc_id": "a", "text": text} for text in texts])


def test_collection_policy_cannot_be_silently_weakened(tmp_path):
    old = snapshot("a", "b", "c", "d")
    bootstrap = make_plan(snapshot(), old)
    with local_client(str(tmp_path / "db")) as client:
        apply_plan(client, "kb", bootstrap, HashEmbedder(),
                   review=review_plan(bootstrap, max_deletion_fraction=0.5))
    with local_client(str(tmp_path / "db")) as client:
        plan = make_plan(old, snapshot("a"))
        before = deepcopy(client.get_collection("kb").config.metadata)
        embedder = HashEmbedder()
        with pytest.raises(SyncError, match="collection review policy"):
            apply_plan(client, "kb", plan, embedder)
        assert embedder.calls == 0
        assert client.get_collection("kb").config.metadata == before
        assert client.count("kb").count == 4


def bootstrap(client, old, limit=0.5):
    plan = make_plan(snapshot(), old)
    return apply_plan(client, "kb", plan, HashEmbedder(), review=review_plan(plan, max_deletion_fraction=limit))


def record(plan, before=0.5, after=0.75, **kw):
    return policy_override(plan, "kb", from_max_deletion_fraction=before,
                           max_deletion_fraction=after, reviewer=kw.get("reviewer", "ops"),
                           reason=kw.get("reason", "approved retirement DOC-42"))


def forbid_writes(guard, client, embedder):
    for name in ("upsert", "delete", "overwrite_payload", "update_collection", "create_collection"):
        guard.setattr(client, name, lambda *a, **kw: pytest.fail("refusal must precede mutation"))
    guard.setattr(embedder, "embed", lambda *a: pytest.fail("refusal must precede embedding"))


@pytest.mark.parametrize("case", ["zero_deletions", "explicit_limit", "gate_override", "completed_replay"])
def test_weakening_is_refused_regardless_of_actual_deletions_or_gate_override(tmp_path, monkeypatch, case):
    old = snapshot("a", "b", "c", "d")
    with local_client(str(tmp_path / "db")) as client:
        result = bootstrap(client, old)
        assert result["review_policy"] == {"max_deletion_fraction": 0.5}
        target = old if case == "completed_replay" else snapshot("a", "b", "c", "d", "e") if case == "zero_deletions" else snapshot("a")
        plan = make_plan(snapshot() if case == "completed_replay" else old, target)
        review = review_plan(plan, max_deletion_fraction=0.6, overrides=["max_deletion_fraction"],
                             reviewer="ops", reason="retire") if case == "gate_override" else review_plan(plan)
        if case == "explicit_limit":
            review = review_plan(plan, max_deletion_fraction=0.75)
        embedder = HashEmbedder()
        forbid_writes(monkeypatch, client, embedder)
        with pytest.raises(SyncError, match="collection review policy"):
            apply_plan(client, "kb", plan, embedder, review=review)


def test_gate_override_keeps_strict_collection_policy(tmp_path):
    old = snapshot("a", "b", "c", "d")
    plan = make_plan(old, snapshot("a"))
    review = review_plan(plan, max_deletion_fraction=0.5, overrides=["max_deletion_fraction"],
                         reviewer="ops", reason="one-time deletion")
    with local_client(str(tmp_path / "db")) as client:
        bootstrap(client, old)
        result = apply_plan(client, "kb", plan, HashEmbedder(), review=review)
        assert result["review_policy"] == {"max_deletion_fraction": 0.5}
        assert result["policy_override"] is None


def test_explicit_policy_override_persists_and_replays(tmp_path):
    old, new = snapshot("a", "b", "c", "d"), snapshot("a")
    plan = make_plan(old, new)
    receipt, override = review_plan(plan, max_deletion_fraction=0.75), record(plan)
    with local_client(str(tmp_path / "db")) as client:
        bootstrap(client, old)
        result = apply_plan(client, "kb", plan, HashEmbedder(), review=receipt, policy_override=override)
        assert result["review_policy"] == {"max_deletion_fraction": 0.75}
        assert result["policy_override"] == override
        state = client.get_collection("kb").config.metadata["chunksync"]
        assert state["last_policy_override"] == override and state["last_review"] == receipt
    with local_client(str(tmp_path / "db")) as client:
        assert verify_snapshot(client, "kb", new, HashEmbedder())["points"] == 1
        replay = apply_plan(client, "kb", plan, HashEmbedder(), review=receipt, policy_override=override)
        assert replay["status"] == "already_applied" and replay["policy_override"] == override
        # A fresh exception cannot be recorded by replaying a completed transaction.
        with pytest.raises(SyncError, match="policy override"):
            apply_plan(client, "kb", plan, HashEmbedder(), review=receipt,
                       policy_override=record(plan, reason="new reason"))


def test_tightening_admitted_for_later_plan_and_retains_last_policy_exception(tmp_path):
    old, middle, new = snapshot("a", "b", "c", "d"), snapshot("a", "b"), snapshot("a", "b", "e")
    first, second = make_plan(old, middle), make_plan(middle, new)
    override = record(first)
    with local_client(str(tmp_path / "db")) as client:
        bootstrap(client, old)
        apply_plan(client, "kb", first, HashEmbedder(), review=review_plan(first, max_deletion_fraction=0.75), policy_override=override)
        result = apply_plan(client, "kb", second, HashEmbedder(), review=review_plan(second, max_deletion_fraction=0.25))
        assert result["review_policy"] == {"max_deletion_fraction": 0.25}
        assert result["policy_override"] == override
    with local_client(str(tmp_path / "db")) as client:
        with pytest.raises(SyncError, match="collection review policy"):
            apply_plan(client, "kb", make_plan(new, snapshot("a", "b", "e", "f")), HashEmbedder(),
                       review=review_plan(make_plan(new, snapshot("a", "b", "e", "f")), max_deletion_fraction=0.5))


@pytest.mark.parametrize("change", ["missing", "reason", "from", "to", "collection", "plan", "hash", "extra", "malformed"])
def test_bad_or_stale_policy_override_refused_without_mutation(tmp_path, monkeypatch, change):
    old, new = snapshot("a", "b", "c", "d"), snapshot("a")
    plan = make_plan(old, new)
    override = record(plan)
    if change == "missing":
        override = None
    elif change == "reason":
        override["provenance"]["reason"] = "tampered"
    elif change == "from":
        override = record(plan, before=0.25)
    elif change == "to":
        override = record(plan, after=1.0)
    elif change == "collection":
        override = policy_override(plan, "other", from_max_deletion_fraction=0.5, max_deletion_fraction=0.75, reviewer="ops", reason="x")
    elif change == "plan":
        override = record(make_plan(old, snapshot("b")))
    elif change == "hash":
        override["fingerprint"] = "0" * 64
    elif change == "extra":
        override["extra"] = True
    else:
        override = []
    with local_client(str(tmp_path / "db")) as client:
        bootstrap(client, old)
        embedder = HashEmbedder()
        forbid_writes(monkeypatch, client, embedder)
        with pytest.raises(SyncError, match="policy override"):
            apply_plan(client, "kb", plan, embedder, review=review_plan(plan, max_deletion_fraction=0.75), policy_override=override)


@pytest.mark.parametrize("field, value", [
    ("from_max_deletion_fraction", True), ("from_max_deletion_fraction", -0.1),
    ("max_deletion_fraction", 1.1), ("max_deletion_fraction", float("nan")),
    ("max_deletion_fraction", float("inf")), ("max_deletion_fraction", 10**400),
    ("max_deletion_fraction", "0.75"), ("max_deletion_fraction", 0.5),
    ("max_deletion_fraction", 0.25), ("reviewer", " "), ("reason", ""),
    ("reviewer", "x" * 257), ("reason", "x" * 4097),
])
def test_policy_exception_input_contract(field, value):
    options = dict(from_max_deletion_fraction=0.5, max_deletion_fraction=0.75, reviewer="ops", reason="approved")
    options[field] = value
    with pytest.raises(SyncError):
        policy_override(make_plan(snapshot("a"), snapshot()), "kb", **options)


@pytest.mark.parametrize("change", ["missing", "reason", "review"])
def test_policy_override_is_bound_before_persisted_failure_and_resume(tmp_path, monkeypatch, change):
    old, new = snapshot("a", "b", "c", "d"), snapshot("a")
    plan = make_plan(old, new)
    review, override = review_plan(plan, max_deletion_fraction=0.75), record(plan)
    with local_client(str(tmp_path / "db")) as client:
        bootstrap(client, old)
        original = client.delete
        with monkeypatch.context() as failure:
            def persist_then_fail(*a, **kw):
                original(*a, **kw)
                raise RuntimeError("persisted delete")
            failure.setattr(client, "delete", persist_then_fail)
            with pytest.raises(RuntimeError, match="persisted delete"):
                apply_plan(client, "kb", plan, HashEmbedder(), review=review, policy_override=override)
        state = client.get_collection("kb").config.metadata["chunksync"]
        assert state["review_policy"] == review["policy"] and state["last_policy_override"] == override
        assert state["pending"]["policy_override_fingerprint"] == override["fingerprint"]
    with local_client(str(tmp_path / "db")) as client:
        changed = None if change == "missing" else record(plan, reason="changed") if change == "reason" else override
        changed_review = review_plan(plan, max_deletion_fraction=1) if change == "review" else review
        embedder = HashEmbedder()
        with monkeypatch.context() as guard:
            forbid_writes(guard, client, embedder)
            with pytest.raises(SyncError, match="(binding|policy override).*mismatch"):
                apply_plan(client, "kb", plan, embedder, review=changed_review, policy_override=changed)
        calls = []
        original = client.delete
        def count_delete(*a, **kw):
            calls.extend(kw["points_selector"].points)
            return original(*a, **kw)
        monkeypatch.setattr(client, "delete", count_delete)
        resumed = apply_plan(client, "kb", plan, HashEmbedder(), review=review, policy_override=override)
        assert resumed["embedded"] == 0 and len(calls) == 2
        assert verify_snapshot(client, "kb", new, HashEmbedder())["points"] == 1


@pytest.mark.parametrize("legacy", ["last_review", "no_review", "invalid"])
def test_legacy_policy_migration_and_corrupt_policy_fail_closed(tmp_path, monkeypatch, legacy):
    old, new = snapshot("a", "b"), snapshot("a", "b", "c")
    plan = make_plan(old, new)
    with local_client(str(tmp_path / "db")) as client:
        bootstrap(client, old)
        state = deepcopy(client.get_collection("kb").config.metadata["chunksync"])
        state.pop("review_policy")
        if legacy == "no_review":
            state.pop("last_review")
        elif legacy == "invalid":
            state["review_policy"] = {"max_deletion_fraction": True}
        client.update_collection("kb", metadata={"chunksync": state})
    with local_client(str(tmp_path / "db")) as client:
        if legacy == "no_review":
            result = apply_plan(client, "kb", plan, HashEmbedder())
            assert result["review_policy"] == {"max_deletion_fraction": 1.0}
        else:
            embedder = HashEmbedder()
            with monkeypatch.context() as guard:
                forbid_writes(guard, client, embedder)
                with pytest.raises(SyncError, match="collection review policy"):
                    apply_plan(client, "kb", plan, embedder)
            if legacy == "last_review":
                assert apply_plan(client, "kb", plan, HashEmbedder(), review=review_plan(plan, max_deletion_fraction=0.5))["review_policy"] == {"max_deletion_fraction": 0.5}


def test_unused_policy_override_and_plan_gate_still_refused(tmp_path):
    old = snapshot("a", "b", "c", "d")
    plan = make_plan(old, snapshot("a"))
    with local_client(str(tmp_path / "db")) as client:
        bootstrap(client, old, limit=1)
        with pytest.raises(SyncError, match="current collection review policy"):
            apply_plan(client, "kb", plan, HashEmbedder(), review=review_plan(plan, max_deletion_fraction=0.75), policy_override=record(plan))
    with local_client(str(tmp_path / "other")) as client:
        bootstrap(client, old)
        # A policy override alone does not override the plan's deletion violation.
        with pytest.raises(SyncError, match="review blocked"):
            apply_plan(client, "kb", plan, HashEmbedder(), review=review_plan(plan, max_deletion_fraction=0.6), policy_override=record(plan, after=0.6))


def test_cli_policy_record_receipt_admission_and_offline_dry_run(tmp_path):
    import json
    import subprocess
    import sys
    from ragwrench.sync.core import write_json

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "ragwrench.sync.cli", *map(str, args)], capture_output=True, text=True)

    old = snapshot("a", "b", "c", "d")
    initial, plan_path, receipt, override_path = [tmp_path / f"{name}.json" for name in ("initial", "plan", "receipt", "override")]
    plan = make_plan(old, snapshot("a"))
    write_json(initial, make_plan(snapshot(), old))
    write_json(plan_path, plan)
    db = tmp_path / "db"
    assert cli("apply", initial, "--qdrant-path", db, "--max-deletion-fraction", "0.5").returncode == 0
    assert cli("review", plan_path, "--max-deletion-fraction", "0.75", "-o", receipt).returncode == 0
    before = (db / "meta.json").read_bytes()
    denied = cli("apply", plan_path, "--qdrant-path", db, "--review", receipt)
    assert denied.returncode == 2 and "collection review policy" in denied.stderr
    assert (db / "meta.json").read_bytes() == before
    generated = cli("policy-override", plan_path, "--from-max-deletion-fraction", "0.5", "--max-deletion-fraction", "0.75",
                    "--reviewer", "policy-owner", "--reason", "DOC-42", "-o", override_path)
    assert generated.returncode == 0, generated.stderr
    override = json.loads(generated.stdout)
    assert validate_policy_override(plan, "kb", override, {"max_deletion_fraction": 0.75}) == override
    absent = tmp_path / "absent"
    dry = cli("apply", plan_path, "--qdrant-path", absent, "--review", receipt, "--policy-override", override_path, "--dry-run")
    assert dry.returncode == 0 and json.loads(dry.stdout)["collection_checked"] is False and not absent.exists()
    bad = deepcopy(override)
    bad["provenance"]["reason"] = "tampered"
    write_json(override_path, bad)
    invalid = cli("apply", plan_path, "--qdrant-path", absent, "--review", receipt, "--policy-override", override_path, "--dry-run")
    assert invalid.returncode == 2 and "policy override" in invalid.stderr and not absent.exists()
    write_json(override_path, override)
    args = ("apply", plan_path, "--qdrant-path", db, "--review", receipt, "--policy-override", override_path)
    applied = cli(*args, "--verify")
    assert applied.returncode == 0, applied.stderr
    assert json.loads(applied.stdout)["policy_override"] == override
    replay = cli(*args)
    assert replay.returncode == 0 and json.loads(replay.stdout)["status"] == "already_applied"
    invalid = cli("policy-override", plan_path, "--from-max-deletion-fraction", "0.75", "--max-deletion-fraction", "0.5",
                  "--reviewer", "ops", "--reason", "wrong direction", "-o", tmp_path / "invalid.json")
    assert invalid.returncode == 2 and not (tmp_path / "invalid.json").exists()


def test_policy_is_recorded_before_embedding_and_survives_torn_metadata(tmp_path, monkeypatch):
    old, new = snapshot("a", "b"), snapshot("a", "b", "c")
    plan = make_plan(old, new)
    review, override = review_plan(plan, max_deletion_fraction=0.75), record(plan)
    db = tmp_path / "db"
    with local_client(str(db)) as client:
        bootstrap(client, old)
        embedder = HashEmbedder()
        def fail_embedding(*args):
            state = client.get_collection("kb").config.metadata["chunksync"]
            assert state["review_policy"] == review["policy"]
            assert state["last_policy_override"] == override
            assert state["pending"]["policy_override_fingerprint"] == override["fingerprint"]
            raise RuntimeError("embedding interrupted")
        monkeypatch.setattr(embedder, "embed", fail_embedding)
        with pytest.raises(RuntimeError, match="embedding interrupted"):
            apply_plan(client, "kb", plan, embedder, review=review, policy_override=override)
    (db / "meta.json").write_text('{"collections":')
    with local_client(str(db)) as client:
        state = client.get_collection("kb").config.metadata["chunksync"]
        assert state["review_policy"] == review["policy"] and state["last_policy_override"] == override
        with pytest.raises(SyncError, match="policy override binding mismatch"):
            apply_plan(client, "kb", plan, HashEmbedder(), review=review)
        result = apply_plan(client, "kb", plan, HashEmbedder(), review=review, policy_override=override)
        assert result["embedded"] == 1 and result["policy_override"] == override
        assert verify_snapshot(client, "kb", new, HashEmbedder())["points"] == 3


def test_completed_replay_does_not_silently_change_policy(tmp_path):
    old = snapshot("a")
    plan = make_plan(snapshot(), old)
    with local_client(str(tmp_path / "db")) as client:
        original = bootstrap(client, old)
        replay = apply_plan(client, "kb", plan, HashEmbedder(), review=review_plan(plan, max_deletion_fraction=0.25))
        assert replay["status"] == "already_applied" and replay["review"] == original["review"]
        assert replay["review_policy"] == original["review_policy"] == {"max_deletion_fraction": 0.5}


def test_explicit_null_policy_override_file_is_invalid_even_offline(tmp_path):
    import subprocess
    import sys
    from ragwrench.sync.core import write_json
    plan_path, override_path = tmp_path / "plan.json", tmp_path / "null.json"
    write_json(plan_path, make_plan(snapshot(), snapshot("a")))
    write_json(override_path, None)
    result = subprocess.run([sys.executable, "-m", "ragwrench.sync.cli", "apply", str(plan_path),
                             "--qdrant-path", str(tmp_path / "absent"), "--policy-override", str(override_path),
                             "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 2 and "invalid policy override" in result.stderr
    assert not (tmp_path / "absent").exists()


def test_policy_override_cannot_initialize_unmanaged_collection(tmp_path, monkeypatch):
    from qdrant_client import models
    plan = make_plan(snapshot(), snapshot("a"))
    with local_client(str(tmp_path / "db")) as client:
        client.create_collection("kb", vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT))
        embedder = HashEmbedder()
        forbid_writes(monkeypatch, client, embedder)
        with pytest.raises(SyncError, match="policy override requires an existing collection review policy"):
            apply_plan(client, "kb", plan, embedder, review=review_plan(plan, max_deletion_fraction=0.75), policy_override=record(plan))
