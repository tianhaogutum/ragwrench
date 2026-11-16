from copy import deepcopy
import json

import pytest

from ragwrench.port.bundle import load_bundle, write_bundle
from ragwrench.port.common import PortError, atomic_json, content_hash, read_json
from ragwrench.port.fakes import FakeState, running_fake
from ragwrench.port.operations import export_bundle, import_bundle, make_plan, verify_bundle


def count_remote(state):
    docs = [d for ds in state.datasets.values() for d in ds["documents"].values()]
    return len(state.datasets), len(docs), sum(len(d["chunks"]) for d in docs)


@pytest.mark.parametrize("source,target", [("dify", "ragflow"), ("ragflow", "dify"), ("dify", "dify"), ("ragflow", "ragflow")])
def test_roundtrip_and_idempotent(source, target, example, tmp_path, api_factory):
    src, dst = FakeState(source), FakeState(target)
    dataset = src.seed(example)
    out, mapping = tmp_path / "out", tmp_path / "mapping.json"
    with running_fake(src) as source_url, running_fake(dst) as target_url:
        export_bundle(api_factory(source, source_url), dataset, out)
        api = api_factory(target, target_url)
        result = import_bundle(api, out, mapping, dataset_name="imported")
        assert result["chunks"] == 15 and result["skipped"] == 0
        assert count_remote(dst) == (1, 3, 15)
        before = len([r for r in dst.requests if r["method"] != "GET"])
        import_bundle(api, out, mapping, dataset_name="imported")
        assert len([r for r in dst.requests if r["method"] != "GET"]) == before
        assert verify_bundle(api, out, state_path=mapping)["verified"]
        assert verify_bundle(api, out, dataset_id=result["dataset_id"])["verified"]
        if target == "dify":
            deletes = [r for r in dst.requests if r["method"] == "DELETE"]
            assert len(deletes) == 3  # bootstrap segments really existed and were removed
            requests = [r for r in dst.requests if r["path"].endswith("create-by-text")]
            assert all(r["body"]["process_rule"]["mode"] == "custom" for r in requests)
        blobs = b"".join(p.read_bytes() for p in out.iterdir() if p.is_file()) + mapping.read_bytes()
        assert b"fake-key" not in blobs
        # Content, metadata fields, disabled flags and multiplicity are compared by verification.
        doc = next(iter(dst.datasets[result["dataset_id"]]["documents"].values()))
        chunk = next(iter(doc["chunks"].values()))
        chunk["content"] = "tampered"
        with pytest.raises(PortError, match="Verification failed"):
            verify_bundle(api, out, state_path=mapping)


@pytest.mark.parametrize("target", ["dify", "ragflow"])
@pytest.mark.parametrize("stage", ["dataset", "document", "chunk", "flag"])
def test_resume_after_committed_write_response_lost(target, stage, example, tmp_path, api_factory):
    dst = FakeState(target)
    prefix = "/v1" if target == "dify" else "/api/v1"
    suffix = {"dataset": prefix + "/datasets", "document": "create-by-text" if target == "dify" else "/documents",
              "chunk": "/segments" if target == "dify" else "/chunks"}
    mapping = tmp_path / "state.json"
    with running_fake(dst) as url:
        api = api_factory(target, url)
        if stage == "flag":
            dst.faults = [{"method": "POST" if target == "dify" else "PATCH", "phase": "after", "path_suffix": "never-match"}]
            # Inject specifically in the adapter after the server applied the flag.
            original = api.set_enabled
            def lost(*args):
                original(*args)
                api.set_enabled = original
                raise PortError("simulated lost flag response")
            api.set_enabled = lost
        else:
            dst.faults = [{"method": "POST", "path_suffix": suffix[stage], "phase": "after", "status": 503}]
        with pytest.raises(PortError):
            import_bundle(api, example, mapping, dataset_name="resumable")
        result = import_bundle(api, example, mapping, dataset_name="resumable")
        assert count_remote(dst) == (1, 3, 15)
        assert result["chunks"] == 15
        assert verify_bundle(api, example, state_path=mapping)["verified"]


@pytest.mark.parametrize("chunk_retries", [0, 2])
def test_missing_pending_write_requires_explicit_retry(example, tmp_path, api_factory, chunk_retries):
    dst = FakeState("ragflow")
    dst.faults = [{"method": "POST", "path_suffix": "/chunks", "status": 503}]
    state = tmp_path / "state.json"
    with running_fake(dst) as url:
        api = api_factory("ragflow", url)
        with pytest.raises(PortError):
            import_bundle(api, example, state, dataset_name="interrupted", chunk_retries=chunk_retries)
        assert sum(r["method"] == "POST" and r["path"].endswith("/chunks") for r in dst.requests) == 1
        before = count_remote(dst)
        with pytest.raises(PortError, match="no visible match"):
            import_bundle(api, example, state, dataset_name="interrupted", chunk_retries=chunk_retries)
        assert count_remote(dst) == before
        import_bundle(api, example, state, dataset_name="interrupted", retry_pending=True)
        assert count_remote(dst) == (1, 3, 15)


@pytest.mark.parametrize("target", ["dify", "ragflow"])
def test_chunk_recovery_after_interruption(target, example, tmp_path, api_factory):
    dst = FakeState(target)
    state = tmp_path / "state.json"
    resource = "/segments" if target == "dify" else "/chunks"
    with running_fake(dst) as url:
        api = api_factory(target, url)
        original = api.set_enabled
        calls = 0
        def interrupt(*args):
            nonlocal calls
            original(*args)
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt
        api.set_enabled = interrupt
        with pytest.raises(KeyboardInterrupt):
            import_bundle(api, example, state, dataset_name="interrupted")
        checkpoint = read_json(state)
        doc = next(iter(checkpoint["documents"].values()))
        assert len(doc["chunks"]) == 2 and doc["pending"]["target"]
        completed = set(doc["chunks"].values())
        before = len(dst.requests)
        api.set_enabled = original
        # The next new chunk commits, but its HTTP response reports failure.
        dst.faults = [{"method": "POST", "path_suffix": resource, "phase": "after", "status": 503}]
        result = import_bundle(api, example, state, dataset_name="interrupted", chunk_retries=1)
        assert result["chunks"] == 15 and count_remote(dst) == (1, 3, 15)
        assert not any(r["method"] != "GET" and r["path"].rsplit("/", 1)[-1] in completed
                       for r in dst.requests[before:])
        # No create POST was replayed, including Dify's non-content-derived IDs.
        assert sum(r["method"] == "POST" and r["path"].endswith(resource) for r in dst.requests) == 15
        assert verify_bundle(api, example, state_path=state)["verified"]
        before = len(dst.requests)
        import_bundle(api, example, state, dataset_name="interrupted", chunk_retries=1)
        assert all(r["method"] == "GET" for r in dst.requests[before:])


@pytest.mark.parametrize("target", ["dify", "ragflow"])
@pytest.mark.parametrize("failures", [2, 3])
def test_chunk_flag_retry_budget(target, failures, example, tmp_path, api_factory):
    class FlagFailures(FakeState):
        flag_attempts = 0
        def dispatch(self, method, path, query, body):
            result = super().dispatch(method, path, query, body)
            if body and ("segment" in body or "available" in body):
                self.flag_attempts += 1
                if self.flag_attempts <= failures:
                    return self.error(503)  # the flag committed before the response failed
            return result
    dst = FlagFailures(target)
    state = tmp_path / "state.json"
    waits = []
    with running_fake(dst) as url:
        api = api_factory(target, url)
        api.wait_before_retry = lambda attempt, retry_after=0: waits.append(attempt)
        if failures == 3:
            with pytest.raises(PortError, match="503"):
                import_bundle(api, example, state, dataset_name="flags", chunk_retries=2)
            checkpoint = next(iter(read_json(state)["documents"].values()))
            assert checkpoint["pending"]["target"] and not checkpoint["chunks"]
            assert count_remote(dst) == (1, 1, 1)
            assert dst.flag_attempts == 3
        else:
            import_bundle(api, example, state, dataset_name="flags", chunk_retries=2)
            assert count_remote(dst) == (1, 3, 15)
        assert waits == [0, 1]
        import_bundle(api, example, state, dataset_name="flags", chunk_retries=0)
        assert verify_bundle(api, example, state_path=state)["verified"]
        resource = "/segments" if target == "dify" else "/chunks"
        assert sum(r["method"] == "POST" and r["path"].endswith(resource) for r in dst.requests) == 15


@pytest.mark.parametrize("status,body", [(401, {}), (403, {}), (400, {}), (200, {"code": 109})])
def test_chunk_permanent_errors_do_not_recover(status, body, example, tmp_path, api_factory):
    dst = FakeState("ragflow")
    dst.faults = [{"method": "POST", "path_suffix": "/chunks", "status": status, "body": body}]
    with running_fake(dst) as url:
        api = api_factory("ragflow", url)
        waits = []
        api.wait_before_retry = lambda *args: waits.append(args)
        with pytest.raises(PortError):
            import_bundle(api, example, tmp_path / "state.json", dataset_name="rejected", chunk_retries=10)
    assert not waits
    assert sum(r["method"] == "POST" and r["path"].endswith("/chunks") for r in dst.requests) == 1


def test_retry_pending_does_not_authorize_automatic_replays(example, tmp_path, api_factory):
    dst = FakeState("dify")  # repeated creates really would duplicate segments
    state = tmp_path / "state.json"
    with running_fake(dst) as url:
        api = api_factory("dify", url)
        dst.faults = [{"method": "POST", "path_suffix": "/segments", "status": 503}]
        with pytest.raises(PortError):
            import_bundle(api, example, state, dataset_name="once")
        dst.faults = [{"method": "POST", "path_suffix": "/segments", "status": 503}]
        before = len(dst.requests)
        with pytest.raises(PortError, match="no visible match"):
            import_bundle(api, example, state, dataset_name="once", retry_pending=True, chunk_retries=2)
        assert sum(r["method"] == "POST" and r["path"].endswith("/segments") for r in dst.requests[before:]) == 1
        assert count_remote(dst) == (1, 1, 0)


@pytest.mark.parametrize("budget", [-1, 11, 1.5, True])
def test_invalid_chunk_retry_budget_is_offline(budget, example, tmp_path, api_factory):
    state = tmp_path / "absent" / "state.json"
    api = api_factory("ragflow", "http://127.0.0.1:1")
    with pytest.raises(PortError, match="chunk-retries"):
        import_bundle(api, example, state, dataset_name="invalid", chunk_retries=budget, dry_run=True)
    assert not state.parent.exists()


def test_chunk_recovery_polls_visibility_and_stops_on_ambiguity(example, tmp_path, api_factory):
    dst = FakeState("dify")
    state = tmp_path / "state.json"
    with running_fake(dst) as url:
        api = api_factory("dify", url)
        dst.faults = [{"method": "POST", "path_suffix": "/segments", "phase": "after", "status": 503}]
        with pytest.raises(PortError):
            import_bundle(api, example, state, dataset_name="visibility")
        original_pages = api.pages
        hidden = 0
        def delayed(kind, *args, **kwargs):
            nonlocal hidden
            rows = original_pages(kind, *args, **kwargs)
            if kind == "chunks" and hidden < 2:
                hidden += 1
                return []
            return rows
        api.pages = delayed
        import_bundle(api, example, state, dataset_name="visibility", chunk_retries=2)
        assert hidden == 2 and count_remote(dst) == (1, 3, 15)
        assert sum(r["method"] == "POST" and r["path"].endswith("/segments") for r in dst.requests) == 15

    dst = FakeState("dify")
    state = tmp_path / "ambiguous.json"
    with running_fake(dst) as url:
        api = api_factory("dify", url)
        dst.faults = [{"method": "POST", "path_suffix": "/segments", "phase": "after", "status": 503}]
        with pytest.raises(PortError):
            import_bundle(api, example, state, dataset_name="ambiguity")
        doc = next(iter(next(iter(dst.datasets.values()))["documents"].values()))
        chunk = deepcopy(next(iter(doc["chunks"].values())))
        chunk["id"] = dst.new_id()
        doc["chunks"][chunk["id"]] = chunk
        waits = []
        api.wait_before_retry = lambda *args: waits.append(args)
        before = len(dst.requests)
        with pytest.raises(PortError, match="Ambiguous pending chunk"):
            import_bundle(api, example, state, dataset_name="ambiguity", chunk_retries=10)
        assert not waits and all(r["method"] == "GET" for r in dst.requests[before:])


def test_duplicate_dataset_and_explicit_reuse(example, tmp_path, api_factory):
    dst = FakeState("ragflow")
    ident = dst.seed(example)
    with running_fake(dst) as url:
        api = api_factory("ragflow", url)
        with pytest.raises(PortError, match="Duplicate target dataset"):
            import_bundle(api, example, tmp_path / "state.json", dataset_name="Synthetic handbook")
        assert count_remote(dst) == (1, 3, 15)
        result = import_bundle(api, example, tmp_path / "reused.json", dataset_id=ident)
        assert result["dataset_id"] == ident
        assert count_remote(dst) == (1, 6, 30)
        assert verify_bundle(api, example, state_path=tmp_path / "reused.json")["verified"]


def test_dry_run_is_offline_and_has_no_writes(example, tmp_path, api_factory):
    api = api_factory("ragflow", "http://127.0.0.1:1")
    state = tmp_path / "absent" / "state.json"
    result = import_bundle(api, example, state, dataset_name="planned", dry_run=True)
    assert result["dry_run"] and result["chunks"] == 15
    assert not state.parent.exists()


def test_preflight_chunk_limit_abort_and_skip(example, tmp_path, api_factory):
    dst = FakeState("ragflow")
    with running_fake(dst) as url:
        api = api_factory("ragflow", url)
        with pytest.raises(PortError, match="byte limit"):
            import_bundle(api, example, tmp_path / "abort.json", dataset_name="limit", max_chunk_bytes=10)
        assert not dst.requests and not (tmp_path / "abort.json").exists()
        result = import_bundle(api, example, tmp_path / "skip.json", dataset_name="limit", max_chunk_bytes=10, on_error="skip")
        assert result["skipped"] == 15 and result["chunks"] == 0
        assert count_remote(dst) == (1, 3, 0)
        with pytest.raises(PortError, match="byte limit"):
            verify_bundle(api, example, state_path=tmp_path / "skip.json", max_chunk_bytes=10)


def test_projection_mapping_and_rejection(example):
    m, docs, chunks = load_bundle(example)
    chunks[0]["questions"] = ["How?"]
    with pytest.raises(PortError, match="embedding questions"):
        make_plan(m, docs, chunks, "dify")
    plan, mapped, _ = make_plan(m, docs, chunks, "dify", allow_lossy=True)
    first = next(iter(mapped.values()))
    assert first["questions"] == [] and plan["warnings"]
    plan, mapped, _ = make_plan(m, docs, chunks, "ragflow")
    assert next(iter(mapped.values()))["questions"] == ["How?"]
    chunks[0]["answer"] = "Through APIs."
    with pytest.raises(PortError, match="QA structure"):
        make_plan(m, docs, chunks, "ragflow")
    plan, mapped, _ = make_plan(m, docs, chunks, "ragflow", allow_lossy=True)
    assert "\nAnswer: Through APIs." in next(iter(mapped.values()))["content"]
    assert next(iter(mapped.values()))["answer"] is None


def test_native_qa_same_system(example, tmp_path, api_factory):
    m, docs, chunks = load_bundle(example)
    docs = docs[:1]
    docs[0]["doc_form"] = "qa_model"
    chunks = [c for c in chunks if c["doc_id"] == docs[0]["id"]]
    for c in chunks:
        c["answer"] = "Synthetic answer."
    bundle = tmp_path / "qa"
    write_bundle(bundle, m["source"], {}, docs, chunks)
    dst = FakeState("dify")
    with running_fake(dst) as url:
        api = api_factory("dify", url)
        result = import_bundle(api, bundle, tmp_path / "state.json", dataset_name="QA", indexing="economy")
        assert result["chunks"] == 5
        assert verify_bundle(api, bundle, state_path=tmp_path / "state.json")["verified"]
        created = next(iter(dst.datasets[result["dataset_id"]]["documents"].values()))
        assert created["doc_form"] == "qa_model"
        assert all(c["answer"] == "Synthetic answer." for c in created["chunks"].values())


def test_questions_survive_ragflow(example, tmp_path, api_factory):
    m, docs, chunks = load_bundle(example)
    chunks[0]["questions"] = ["How to restore?"]
    bundle = tmp_path / "questions"
    write_bundle(bundle, m["source"], {}, docs, chunks)
    dst = FakeState("ragflow")
    with running_fake(dst) as url:
        api = api_factory("ragflow", url)
        result = import_bundle(api, bundle, tmp_path / "state.json", dataset_name="questions")
        verify_bundle(api, bundle, state_path=tmp_path / "state.json")
        export_bundle(api, result["dataset_id"], tmp_path / "exported")
    assert any(c["questions"] == ["How to restore?"] for c in load_bundle(tmp_path / "exported")[2])


def test_repeated_ragflow_content_rejected(example):
    m, docs, chunks = load_bundle(example)
    chunks[1]["content"] = chunks[0]["content"]
    with pytest.raises(PortError, match="Repeated content"):
        make_plan(m, docs, chunks, "ragflow")


def test_state_mismatch_blocks_writes(example, tmp_path, api_factory):
    state = tmp_path / "state.json"
    dst = FakeState("ragflow")
    with running_fake(dst) as url:
        api = api_factory("ragflow", url)
        import_bundle(api, example, state, dataset_name="one")
        count = len(dst.requests)
        with pytest.raises(PortError, match="different bundle/target/options"):
            import_bundle(api, example, state, dataset_name="two")
        assert len(dst.requests) == count


def test_dify_poll_timeout_and_resume(example, tmp_path, api_factory):
    dst = FakeState("dify")
    dst.indexing_polls = 100
    state = tmp_path / "state.json"
    with running_fake(dst) as url:
        api = api_factory("dify", url, poll_timeout=0)
        with pytest.raises(PortError, match="timed out"):
            import_bundle(api, example, state, dataset_name="slow")
        assert count_remote(dst) == (1, 1, 1)
        dst.indexing_polls = 0
        import_bundle(api, example, state, dataset_name="slow")
        assert count_remote(dst) == (1, 3, 15)


def test_verification_reused_dataset_ignores_unrelated_pending_docs(example, tmp_path, api_factory):
    dst = FakeState("dify")
    ident = dst.seed(example)
    with running_fake(dst) as url:
        api = api_factory("dify", url)
        import_bundle(api, example, tmp_path / "state.json", dataset_id=ident)
        unrelated = next(iter(dst.datasets[ident]["documents"].values()))
        unrelated["enabled"] = False
        unrelated["indexing_status"] = "indexing"
        assert verify_bundle(api, example, state_path=tmp_path / "state.json")["verified"]


def test_resume_clears_remaining_bootstrap_segments(example, tmp_path, api_factory):
    dst = FakeState("dify")
    state = tmp_path / "state.json"
    dst.faults = [{"method": "DELETE", "phase": "after", "status": 503}]
    with running_fake(dst) as url:
        api = api_factory("dify", url)
        with pytest.raises(PortError):
            import_bundle(api, example, state, dataset_name="clearing")
        assert count_remote(dst) == (1, 1, 0)
        import_bundle(api, example, state, dataset_name="clearing")
        assert count_remote(dst) == (1, 3, 15)
        assert verify_bundle(api, example, state_path=state)["verified"]


def test_document_metadata_is_not_silently_dropped(example):
    m, docs, chunks = load_bundle(example)
    docs[0]["metadata"] = {"topic": "backup"}
    with pytest.raises(PortError, match="Document metadata"):
        make_plan(m, docs, chunks, "ragflow")
    plan, _, _ = make_plan(m, docs, chunks, "ragflow", allow_lossy=True)
    assert "Document metadata preserved in bundle only" in plan["warnings"]


def test_same_target_verification_detects_enabled_and_keyword_changes(example, tmp_path, api_factory):
    dst = FakeState("ragflow")
    state = tmp_path / "state.json"
    with running_fake(dst) as url:
        api = api_factory("ragflow", url)
        result = import_bundle(api, example, state, dataset_name="fields")
        doc = next(iter(dst.datasets[result["dataset_id"]]["documents"].values()))
        chunk = next(iter(doc["chunks"].values()))
        chunk["enabled"] = not chunk["enabled"]
        with pytest.raises(PortError, match="Verification failed"):
            verify_bundle(api, example, state_path=state)
        chunk["enabled"] = not chunk["enabled"]
        chunk["keywords"] = ["changed"]
        with pytest.raises(PortError, match="Verification failed"):
            verify_bundle(api, example, state_path=state)
