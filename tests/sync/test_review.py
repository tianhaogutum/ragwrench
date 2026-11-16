"""Synthetic plans and real local Qdrant review admission/recovery."""
import json
from copy import deepcopy
import subprocess
import sys

import pytest

from ragwrench.sync.core import SyncError, build_snapshot, make_plan, write_json
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.store import apply_plan, local_client, verify_snapshot
from ragwrench.sync.review import compare_plans, require_allowed, review_plan, validate_review
from .conftest import linux_durable


def snapshot(*texts, model="hash-v1", doc="a"):
    return build_snapshot([{"doc_id": doc, "text": text} for text in texts], embed_model=model)


def cli(*args):
    return subprocess.run([sys.executable, "-m", "ragwrench.sync.cli", *map(str, args)],
                          capture_output=True, text=True)


@linux_durable
def test_default_model_change_refused_before_embedding_or_writes(tmp_path, monkeypatch):
    old, new = snapshot("same"), snapshot("same", model="hash-v2")
    with local_client(str(tmp_path / "db")) as client:
        apply_plan(client, "kb", make_plan(snapshot(), old), HashEmbedder())
        before = client.get_collection("kb").config.metadata
        embedder = HashEmbedder()
        for name in ("upsert", "delete", "overwrite_payload", "update_collection"):
            monkeypatch.setattr(client, name, lambda *a, **kw: pytest.fail("mutation before review"))
        with pytest.raises(SyncError, match="review.*embed_model_changed"):
            apply_plan(client, "kb", make_plan(old, new), embedder)
        assert embedder.calls == 0
        assert client.get_collection("kb").config.metadata == before


def test_cli_deletion_gate_refuses_before_store_open(tmp_path):
    plan = tmp_path / "plan.json"
    write_json(plan, make_plan(snapshot("a", "b", "c", "d"), snapshot("a")))
    db = tmp_path / "must-not-exist"
    for extra in ([], ["--dry-run"]):
        result = cli("apply", plan, "--qdrant-path", db, "--max-deletion-fraction", "0.5", *extra)
        assert result.returncode == 2
        assert "review blocked: max_deletion_fraction" in result.stderr
        assert not db.exists()


def test_cli_compare_shared_source(tmp_path):
    old = snapshot("a", "b", "c", "d")
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    write_json(baseline, make_plan(old, snapshot("a", "b", "c")))
    write_json(candidate, make_plan(old, snapshot("a")))
    result = cli("compare", baseline, candidate)
    assert result.returncode == 10, result.stderr
    comparison = json.loads(result.stdout)
    assert comparison["summary_delta"]["delete"] == 2
    assert len(comparison["operations"]["delete"]["added"]) == 2
    assert comparison["operations"]["delete"]["removed"] == []
    assert comparison["deletion_fraction"] == {"baseline": 0.25, "candidate": 0.75}


def test_comparison_reports_changed_payload_with_equal_counts_and_ids():
    old = build_snapshot([{"doc_id": "a", "text": "same", "metadata": {"v": 0}}])
    plans = [make_plan(old, build_snapshot([{"doc_id": "a", "text": "same", "metadata": {"v": v}}]))
             for v in (1, 2)]
    comparison = compare_plans(*plans)
    assert set(comparison["summary_delta"].values()) == {0}
    assert not comparison["identical"]
    assert comparison["operations"]["payload_only"] == {
        "added": [], "removed": [], "changed": [old["chunks"][0]["id"]]}
    assert compare_plans(plans[0], plans[0])["identical"]
    with pytest.raises(SyncError, match="same source"):
        compare_plans(plans[0], make_plan(snapshot("other"), snapshot("new")))
    corrupt = deepcopy(plans[1])
    corrupt["delete"] = []
    corrupt["summary"]["payload_only"] = 0
    with pytest.raises(SyncError, match="operations mismatch"):
        compare_plans(plans[0], corrupt)


def test_review_fraction_boundaries_empty_source_and_rename():
    old = snapshot("a", "b", "c", "d")
    plan = make_plan(old, snapshot("a"))
    assert review_plan(plan, max_deletion_fraction=0.75)["allowed"]
    receipt = review_plan(plan, max_deletion_fraction=0.5)
    assert not receipt["allowed"]
    assert receipt["observed"]["deletions"] == 3
    assert receipt["observed"]["deletion_fraction"] == 0.75
    assert receipt["blocked"] == ["max_deletion_fraction"]
    with pytest.raises(SyncError, match="review blocked"):
        require_allowed(receipt)
    for before, after in ((snapshot(), old), (snapshot(), snapshot()), (old, snapshot("a", "b", "c", "d", doc="renamed"))):
        receipt = review_plan(make_plan(before, after), max_deletion_fraction=0)
        assert receipt["allowed"] and receipt["observed"]["deletion_fraction"] == 0
    assert review_plan(make_plan(old, snapshot()))["allowed"]  # compatibility default: 1.0


@pytest.mark.parametrize("limit", [-0.1, 1.01, float("nan"), float("inf"), 10**400, True, "0.5", None])
def test_invalid_review_policy(limit):
    with pytest.raises(SyncError, match="max-deletion-fraction"):
        review_plan(make_plan(snapshot("a"), snapshot()), max_deletion_fraction=limit)


@pytest.mark.parametrize("options, message", [
    ({"overrides": ["unknown"]}, "supported review gates"),
    ({"overrides": ["embed_model_changed", "embed_model_changed"]}, "unique"),
    ({"overrides": ["max_deletion_fraction"], "reviewer": "ops", "reason": "planned"}, "violated"),
    ({"overrides": ["embed_model_changed"]}, "reviewer and reason"),
    ({"overrides": ["embed_model_changed"], "reviewer": " ", "reason": "planned"}, "reviewer and reason"),
    ({"overrides": ["embed_model_changed"], "reviewer": "ops", "reason": " "}, "reviewer and reason"),
    ({"reviewer": "ops", "reason": "planned"}, "explicit override"),
])
def test_override_requires_exact_gate_and_provenance(options, message):
    plan = make_plan(snapshot("a"), snapshot("a", model="v2"))
    with pytest.raises(SyncError, match=message):
        review_plan(plan, **options)


def test_model_and_deletion_gates_require_independent_overrides():
    plan = make_plan(snapshot("a", "b"), snapshot("a", model="v2"))
    options = dict(max_deletion_fraction=0.5, reviewer="ops", reason="approved rebuild")
    partial = review_plan(plan, overrides=["embed_model_changed"], **options)
    assert partial["blocked"] == ["max_deletion_fraction"]
    full = review_plan(plan, overrides=["embed_model_changed", "max_deletion_fraction"], **options)
    assert full["allowed"]
    assert full["observed"]["deletion_fraction"] == 1.0
    assert full["provenance"] == {"reviewer": "ops", "reason": "approved rebuild"}
    assert validate_review(plan, full) == full


@pytest.mark.parametrize("field", ["plan", "allowed", "fraction", "reason", "fingerprint", "extra", "malformed"])
def test_receipt_tampering_and_other_plans_refused_before_store_open(tmp_path, field):
    plan = make_plan(snapshot("a", "b"), snapshot("a"))
    receipt = review_plan(plan, max_deletion_fraction=0, overrides=["max_deletion_fraction"],
                          reviewer="ops", reason="planned deletion")
    if field == "plan":
        plan = make_plan(snapshot("a", "b"), snapshot("b"))
    elif field == "allowed":
        receipt["allowed"] = False
    elif field == "fraction":
        receipt["observed"]["deletion_fraction"] = 0
    elif field == "reason":
        receipt["provenance"]["reason"] = "changed after review"
    elif field == "fingerprint":
        receipt["fingerprint"] = "0" * 64
    elif field == "extra":
        receipt["extra"] = True
    else:
        receipt = []
    plan_path, review_path = tmp_path / "plan.json", tmp_path / "review.json"
    write_json(plan_path, plan)
    write_json(review_path, receipt)
    result = cli("apply", plan_path, "--review", review_path, "--qdrant-path", tmp_path / "absent")
    assert result.returncode == 2
    assert "review receipt" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "absent").exists()


@linux_durable
def test_cli_review_receipt_apply_and_comparison_exit_codes(tmp_path):
    old, new = snapshot("a", "b", "c", "d"), snapshot("a")
    plan, bootstrap, review = [tmp_path / f"{name}.json" for name in ("plan", "bootstrap", "review")]
    write_json(plan, make_plan(old, new))
    write_json(bootstrap, make_plan(snapshot(), old))
    db = tmp_path / "db"
    blocked = cli("review", plan, "--max-deletion-fraction", "0.5", "-o", review)
    assert blocked.returncode == 3
    assert json.loads(review.read_text())["blocked"] == ["max_deletion_fraction"]
    assert cli("apply", plan, "--review", review, "--qdrant-path", db).returncode == 2
    assert not db.exists()
    approved = cli("review", plan, "--max-deletion-fraction", "0.5", "--override-gate", "max_deletion_fraction",
                   "--reviewer", "team@example.test", "--reason", "remove retired documents", "-o", review)
    assert approved.returncode == 0, approved.stderr
    receipt = json.loads(approved.stdout)
    assert cli("apply", bootstrap, "--qdrant-path", db).returncode == 0
    applied = cli("apply", plan, "--review", review, "--qdrant-path", db, "--verify")
    assert applied.returncode == 0, applied.stderr
    assert json.loads(applied.stdout)["review"] == receipt
    replay = cli("apply", plan, "--review", review, "--qdrant-path", db)
    assert json.loads(replay.stdout)["status"] == "already_applied"
    assert json.loads(replay.stdout)["review"] == receipt
    assert cli("compare", plan, plan).returncode == 0
    assert cli("apply", plan, "--review", review, "--max-deletion-fraction", "1", "--qdrant-path", db).returncode == 2


@pytest.mark.parametrize("change", ["policy", "reviewer", "reason"])
@linux_durable
def test_override_provenance_bound_across_persisted_failure(tmp_path, monkeypatch, change):
    old, new = snapshot("a", "b", "c", "d"), snapshot("a")
    plan = make_plan(old, new)
    options = dict(max_deletion_fraction=0.5, overrides=["max_deletion_fraction"],
                   reviewer="ops", reason="retired documents")
    receipt = review_plan(plan, **options)
    db = str(tmp_path / "db")
    with local_client(db) as client:
        apply_plan(client, "kb", make_plan(snapshot(), old), HashEmbedder())
        original = client.delete

        def fail_after(*a, **kw):
            original(*a, **kw)
            raise RuntimeError("persisted deletion")

        monkeypatch.setattr(client, "delete", fail_after)
        with pytest.raises(RuntimeError, match="persisted deletion"):
            apply_plan(client, "kb", plan, HashEmbedder(), review=receipt)
        state = client.get_collection("kb").config.metadata["chunksync"]
        assert state["last_review"] == receipt
        assert state["pending"]["review_fingerprint"] == receipt["fingerprint"]
    changed = dict(options)
    changed[change if change != "policy" else "max_deletion_fraction"] = 0.6 if change == "policy" else "different"
    with local_client(db) as client:
        with monkeypatch.context() as guard:
            for name in ("delete", "upsert", "overwrite_payload", "update_collection"):
                guard.setattr(client, name, lambda *a, **kw: pytest.fail("changed provenance mutated state"))
            with pytest.raises(SyncError, match="review binding mismatch"):
                apply_plan(client, "kb", plan, HashEmbedder(), review=review_plan(plan, **changed))
        assert apply_plan(client, "kb", plan, HashEmbedder(), review=receipt)["embedded"] == 0
        assert client.get_collection("kb").config.metadata["chunksync"]["last_review"] == receipt
        assert verify_snapshot(client, "kb", new, HashEmbedder())["points"] == 1


@linux_durable
def test_old_journal_without_review_binding_remains_resumable(tmp_path, monkeypatch):
    old, new = snapshot("old"), snapshot("new")
    plan = make_plan(old, new)
    with local_client(str(tmp_path / "db")) as client:
        apply_plan(client, "kb", make_plan(snapshot(), old), HashEmbedder())
        with monkeypatch.context() as failure:
            failure.setattr(client, "upsert", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("injected")))
            with pytest.raises(RuntimeError):
                apply_plan(client, "kb", plan, HashEmbedder())
        # Exact round-2 pending format: no review key in state or journal.
        state = deepcopy(client.get_collection("kb").config.metadata["chunksync"])
        state.pop("last_review")
        state["pending"].pop("review_fingerprint")
        state["pending"].pop("policy_override_fingerprint")
        state.pop("review_policy")
        client.update_collection("kb", metadata={"chunksync": state})
    with local_client(str(tmp_path / "db")) as client:
        apply_plan(client, "kb", plan, HashEmbedder())
        assert verify_snapshot(client, "kb", new, HashEmbedder())["points"] == 1
