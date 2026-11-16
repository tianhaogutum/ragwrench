"""Pinned RAGFlow dataset contracts; all records and KB text are synthetic."""
import json
from pathlib import Path

import httpx
import pytest

from ragwrench.port.bundle import load_bundle, write_bundle
from ragwrench.port.common import PortError, atomic_json, read_json
from ragwrench.port.fakes import FakeState, running_fake
from ragwrench.port.operations import export_bundle, import_bundle, verify_bundle

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("profile,expected_body", [
    ("ragflow-0.27.3", {"name": "new", "chunk_method": "naive"}),
    ("ragflow-1.0", {"name": "new", "parse_type": 1, "parser_id": "general"}),
])
def test_negotiate_native_contract_and_create_payload(profile, expected_body, api_factory):
    envelope = json.loads((FIXTURES / f"{profile}-datasets.json").read_text())
    calls = []
    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=envelope)
        assert json.loads(request.content) == expected_body
        return httpx.Response(200, json={"code": 0, "data": {"id": "new", "name": "new"}})
    api = api_factory("ragflow", "http://offline.invalid", api_profile="auto")
    api.http.close()
    api.http = httpx.Client(transport=httpx.MockTransport(handler))
    assert api.negotiate() == profile
    assert all(r.method == "GET" for r in calls)
    assert api.create_dataset("new") == "new"


@pytest.mark.parametrize("records", [[], [{"id": "d", "name": "unknown"}],
    [{"id": "d", "name": "both", "chunk_method": "naive", "parser_id": "general"}],
    [{"id": "a", "name": "legacy", "chunk_method": "naive"},
     {"id": "b", "name": "modern", "parser_id": "general"}],
])
def test_auto_ambiguous_contract_stops_without_state_or_writes(records, example, tmp_path, api_factory):
    api = api_factory("ragflow", "http://offline.invalid", api_profile="auto", page_size=100)
    api.http.close()
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"code": 0, "data": records})
    api.http = httpx.Client(transport=httpx.MockTransport(handler))
    mapping = tmp_path / "absent" / "state.json"
    with pytest.raises(PortError, match="explicit --api-profile"):
        import_bundle(api, example, mapping, dataset_name="new")
    assert all(r.method == "GET" for r in calls)
    assert not mapping.parent.exists()


@pytest.mark.parametrize("profile", ["ragflow-0.27.3", "ragflow-1.0"])
def test_profile_negotiated_roundtrip_and_state_binding(profile, example, tmp_path, api_factory):
    target = FakeState("ragflow", api_profile=profile)
    target.seed(example)  # supplies a native dataset signature for discovery
    mapping = tmp_path / "state.json"
    with running_fake(target) as url:
        api = api_factory("ragflow", url, api_profile="auto")
        result = import_bundle(api, example, mapping, dataset_name="negotiated")
        assert result["api_profile"] == profile and result["chunks"] == 15
        assert read_json(mapping)["identity"]["api_profile"] == profile
        assert verify_bundle(api, example, state_path=mapping)["verified"]
        before = len(target.requests)
        import_bundle(api, example, mapping, dataset_name="negotiated")
        assert all(r["method"] == "GET" for r in target.requests[before:])
        out = tmp_path / "exported"
        assert export_bundle(api_factory("ragflow", url, api_profile="auto"), result["dataset_id"], out)["api_profile"] == profile
        assert len(load_bundle(out)[2]) == 15
        wrong = "ragflow-1.0" if profile == "ragflow-0.27.3" else "ragflow-0.27.3"
        # Changing profiles cannot resume or verify an existing mapping.
        wrong_api = api_factory("ragflow", url, api_profile=wrong)
        before = len(target.requests)
        with pytest.raises(PortError, match="profile|contract"):
            import_bundle(wrong_api, example, mapping, dataset_name="negotiated")
        with pytest.raises(PortError, match="profile|contract"):
            verify_bundle(wrong_api, example, state_path=mapping)
        assert all(r["method"] == "GET" for r in target.requests[before:])


@pytest.mark.parametrize("profile", ["ragflow-0.27.3", "ragflow-1.0"])
def test_unsupported_fields_reported_before_discovery(profile, example, tmp_path, api_factory):
    manifest, docs, chunks = load_bundle(example)
    # Both RAGFlow contracts lack a separate QA answer field on chunk creation.
    docs = docs[:1]
    chunks = [c for c in chunks if c["doc_id"] == docs[0]["id"]]
    chunks[0].update(json.loads((FIXTURES / "unsupported-chunk-fields.json").read_text())["ragflow"])
    bundle = tmp_path / "qa"
    write_bundle(bundle, manifest["source"], {}, docs, chunks)
    api = api_factory("ragflow", "http://127.0.0.1:1", api_profile=profile)
    mapping = tmp_path / "absent" / "state.json"
    with pytest.raises(PortError, match="Unsupported fields:.*metadata.*QA structure"):
        import_bundle(api, bundle, mapping, dataset_name="qa")
    assert not mapping.parent.exists()
    plan = import_bundle(api, bundle, mapping, dataset_name="qa", allow_lossy=True, dry_run=True)
    assert plan["api_profile"] == profile
    assert plan["warnings"] == ["QA structure flattened to Question/Answer text",
                                "chunk metadata (including image/layout/tags)"]
    assert not mapping.parent.exists()


@pytest.mark.parametrize("system,profile", [("dify", "dify-f8fc371a"), ("ragflow", "ragflow-1.0")])
def test_pre_profile_state_and_checkpoint_remain_compatible(system, profile, example, tmp_path, api_factory):
    target = FakeState(system)
    mapping = tmp_path / "state.json"
    output = tmp_path / "export"
    with running_fake(target) as url:
        api = api_factory(system, url)
        imported = import_bundle(api, example, mapping, dataset_name="original")
        state = read_json(mapping)
        del state["identity"]["api_profile"]
        atomic_json(mapping, state)
        before = len(target.requests)
        import_bundle(api, example, mapping, dataset_name="original")
        assert verify_bundle(api, example, state_path=mapping)["verified"]
        assert all(r["method"] == "GET" for r in target.requests[before:])
        export_bundle(api, imported["dataset_id"], output)
        checkpoint = read_json(output / ".export-state.json")
        del checkpoint["identity"]["api_profile"]
        atomic_json(output / ".export-state.json", checkpoint)
        before = len(target.requests)
        assert export_bundle(api, imported["dataset_id"], output, resume=True)["api_profile"] == profile
        assert not target.requests[before:]  # completed cached pages remain reusable


@pytest.mark.parametrize("profile", ["ragflow-0.27.3", "ragflow-1.0"])
def test_explicit_profile_bootstraps_empty_server(profile, example, tmp_path, api_factory):
    target = FakeState("ragflow", api_profile=profile)
    with running_fake(target) as url:
        api = api_factory("ragflow", url, api_profile=profile)
        result = import_bundle(api, example, tmp_path / "state.json", dataset_name="empty")
        assert result["api_profile"] == profile and result["chunks"] == 15
        assert verify_bundle(api, example, state_path=tmp_path / "state.json")["verified"]


def test_dify_unsupported_questions_fixture_is_reported(example, tmp_path, api_factory):
    manifest, docs, chunks = load_bundle(example)
    chunks[0].update(json.loads((FIXTURES / "unsupported-chunk-fields.json").read_text())["dify"])
    bundle = tmp_path / "questions"
    write_bundle(bundle, manifest["source"], {}, docs, chunks)
    api = api_factory("dify", "http://127.0.0.1:1", api_profile="dify-f8fc371a")
    with pytest.raises(PortError, match="Unsupported fields: embedding questions"):
        import_bundle(api, bundle, tmp_path / "state.json", dataset_name="questions", dry_run=True)
    plan = import_bundle(api, bundle, tmp_path / "state.json", dataset_name="questions", dry_run=True, allow_lossy=True)
    assert plan["warnings"] == ["embedding questions unsupported by Dify"]


@pytest.mark.parametrize("profile,fields", [("ragflow-1.0", {"chunk_method": "naive"}),
                                           ("ragflow-0.27.3", {"parser_id": "general"})])
def test_conflicting_explicit_profile_never_mutates(profile, fields, example, tmp_path, api_factory):
    api = api_factory("ragflow", "http://offline.invalid", api_profile=profile)
    api.http.close()
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"code": 0, "data": [{"id": "d", "name": "existing", **fields}]})
    api.http = httpx.Client(transport=httpx.MockTransport(handler))
    mapping = tmp_path / "state.json"
    with pytest.raises(PortError, match="conflicts"):
        import_bundle(api, example, mapping, dataset_name="new")
    assert not mapping.exists() and all(r.method == "GET" for r in calls)
