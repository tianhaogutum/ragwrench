import json
from pathlib import Path

import httpx
import pytest

from ragwrench.port.api import API
from ragwrench.port.bundle import load_bundle
from ragwrench.port.common import PortError, TransientAPIError, read_json
from ragwrench.port.fakes import FakeState, running_fake
from ragwrench.port.operations import export_bundle


@pytest.mark.parametrize("system", ["dify", "ragflow"])
def test_paginated_export(system, example, tmp_path, api_factory):
    state = FakeState(system)
    dataset = state.seed(example)
    with running_fake(state) as url:
        result = export_bundle(api_factory(system, url, page_size=1), dataset, tmp_path / "out")
    assert result["documents"] == 3 and result["chunks"] == 15
    assert any(r["query"].get("page") == ["3"] and r["path"].endswith("documents") for r in state.requests)
    chunk_calls = [r for r in state.requests if r["path"].endswith(("chunks", "segments"))]
    assert len(chunk_calls) == 15
    _, _, chunks = load_bundle(tmp_path / "out")
    assert chunks[3]["enabled"] is False
    assert chunks[0]["keywords"] == ["backup", "手册"]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retry_get(status, example, tmp_path, api_factory):
    state = FakeState("dify")
    dataset = state.seed(example)
    path = f"/v1/datasets/{dataset}/documents"
    state.faults = [{"path": path, "page": "2", "status": status}]
    with running_fake(state) as url:
        export_bundle(api_factory("dify", url, page_size=1, retries=1), dataset, tmp_path / "out")
    assert sum(r["path"] == path and r["query"].get("page") == ["2"] for r in state.requests) == 2


@pytest.mark.parametrize("system", ["dify", "ragflow"])
def test_export_checkpoint_resumes_completed_pages(system, example, tmp_path, api_factory):
    state = FakeState(system)
    dataset = state.seed(example)
    doc_id = next(iter(state.datasets[dataset]["documents"]))
    prefix = "/v1" if system == "dify" else "/api/v1"
    path = f"{prefix}/datasets/{dataset}/documents/{doc_id}/" + ("segments" if system == "dify" else "chunks")
    state.faults = [{"path": path, "page": "2", "status": 503}]
    out = tmp_path / "out"
    with running_fake(state) as url:
        api = api_factory(system, url)
        with pytest.raises(PortError, match="503"):
            export_bundle(api, dataset, out)
        assert not (out / "manifest.json").exists()
        checkpoint = read_json(out / ".export-state.json")
        assert checkpoint["pages"][path.removeprefix(prefix)]["next"] == 2
        before = len(state.requests)
        export_bundle(api, dataset, out, resume=True)
        assert not any(r["path"] == path and r["query"].get("page") == ["1"] for r in state.requests[before:])
        with pytest.raises(PortError, match="exists"):
            export_bundle(api, dataset, out)
        with pytest.raises(PortError, match="different"):
            export_bundle(api_factory(system, url, page_size=1), dataset, out, resume=True)
    assert load_bundle(out)[0]["counts"]["chunks"] == 15


@pytest.mark.parametrize("system", ["dify", "ragflow"])
def test_truncated_page_aborts(system, example, tmp_path, api_factory):
    state = FakeState(system)
    dataset = state.seed(example)
    state.truncate = True
    with running_fake(state) as url:
        with pytest.raises(PortError, match="Truncated|not found"):
            export_bundle(api_factory(system, url), dataset, tmp_path / "out")
    assert not (tmp_path / "out" / "manifest.json").exists()


@pytest.mark.parametrize("status", [401, 403])
def test_auth_does_not_retry_or_leak(status, api_factory):
    state = FakeState("dify")
    state.faults = [{"status": status, "body": {"message": "NEVER-PRINT-THIS-KEY"}}]
    with running_fake(state) as url:
        with pytest.raises(PortError) as err:
            api_factory("dify", url, retries=3).datasets()
    assert str(status) in str(err.value) and "NEVER-PRINT" not in str(err.value)
    assert len(state.requests) == 1


def test_wrong_key(api_factory):
    state = FakeState("ragflow")
    with running_fake(state) as url:
        with pytest.raises(PortError, match="401"):
            api_factory("ragflow", url, key="bad").datasets()


def test_ragflow_error_envelope(api_factory):
    state = FakeState("ragflow")
    state.faults = [{"status": 200, "body": {"code": 109, "message": "secret"}}]
    with running_fake(state) as url:
        with pytest.raises(PortError, match="code 109"):
            api_factory("ragflow", url).datasets()


@pytest.mark.parametrize("system,fixture", [("dify", "dify-segments.json"), ("ragflow", "ragflow-chunks.json")])
def test_native_response_fixtures(system, fixture, api_factory):
    response = json.loads((Path(__file__).parent / "fixtures" / fixture).read_text())
    api = api_factory(system, "http://offline.invalid", page_size=100)
    api.http.close()
    api.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)))
    chunks = api.pages("chunks", "dataset", "document")
    assert chunks[0]["enabled"] is False
    assert chunks[0]["keywords"] == ["backup"]
    if system == "dify":
        assert chunks[0]["answer"] == "通过公开 API 导出。"
    else:
        assert chunks[0]["questions"] == ["How to back up?"]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "https://user:password@host/v1", "http://host/v1?key=secret", "http://host/v1#secret", "http://host/wrong"])
def test_unsafe_base_urls(url):
    with pytest.raises(PortError, match="Base URL"):
        API("dify", url, key="key")


def test_metadata_list_mapping(api_factory):
    api = api_factory("dify", "http://offline.invalid")
    fields = [{"id": "field", "name": "topic", "type": "string", "value": "backup"}]
    doc = api.normalize("documents", {"id": "doc", "name": "doc.txt", "doc_metadata": fields})
    assert doc["metadata"] == {"dify_fields": fields}


def test_path_id_injection(api_factory):
    api = api_factory("dify", "http://offline.invalid")
    with pytest.raises(PortError, match="identifier"):
        api.pages("chunks", "../../credentials", "doc")


def test_missing_total_rejected(api_factory):
    api = api_factory("ragflow", "http://offline.invalid")
    api.http.close()
    api.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"code": 0, "data": {"docs": []}})))
    with pytest.raises(PortError, match="omitted total"):
        api.pages("documents", "dataset")


def test_pagination_total_changes_or_duplicate_ids_are_errors(api_factory):
    for case in ("total", "duplicate", "has_more"):
        def handler(request):
            page = int(request.url.params.get("page", "1"))
            item = {"id": "same" if case == "duplicate" else f"doc{page}", "name": f"document {page}"}
            total = 3 if case == "total" and page == 2 else 2
            return httpx.Response(200, json={"data": [item], "total": total,
                                            "has_more": False if case == "has_more" else page < total})
        api = api_factory("dify", "http://offline.invalid", page_size=1)
        api.http.close()
        api.http = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(PortError, match="total changed|Duplicate IDs|has_more"):
            api.pages("documents", "dataset")


def test_redirect_refused_without_forwarding_auth(api_factory):
    api = api_factory("dify", "http://offline.invalid")
    api.http.close()
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://other-host.invalid/steal"})
    api.http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(PortError, match="Redirect refused"):
        api.datasets()
    assert len(calls) == 1


def test_backoff_and_retry_after_are_honored(api_factory, monkeypatch):
    api = api_factory("dify", "http://offline.invalid", retries=2)
    api.backoff = 0.25
    api.http.close()
    attempts = []
    delays = []
    def handler(request):
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"Retry-After": "2" if len(attempts) == 2 else "0"})
        return httpx.Response(200, json={"data": [], "total": 0, "has_more": False})
    api.http = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("ragwrench.port.api.time.sleep", delays.append)
    assert api.datasets() == []
    assert delays == [0.25, 2]


def test_mutation_is_not_retried_even_with_get_retry_budget(api_factory):
    state = FakeState("ragflow")
    state.faults = [{"method": "POST", "phase": "after", "status": 503}]
    with running_fake(state) as url:
        api = api_factory("ragflow", url, retries=3)
        with pytest.raises(PortError, match="503"):
            api.create_dataset("only-once")
    assert len(state.datasets) == 1
    assert sum(r["method"] == "POST" for r in state.requests) == 1


@pytest.mark.parametrize("failure", ["transport", "429", "503"])
def test_transient_write_classification_and_bounded_wait(failure, api_factory, monkeypatch):
    api = api_factory("ragflow", "http://offline.invalid", retries=10)
    api.http.close()
    calls, waits = [], []
    def handler(request):
        calls.append(request)
        if failure == "transport":
            raise httpx.RemoteProtocolError("synthetic response loss")
        return httpx.Response(int(failure), headers={"Retry-After": "999"})
    api.http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(TransientAPIError) as error:
        api.request("POST", "/datasets")
    assert len(calls) == 1
    monkeypatch.setattr("ragwrench.port.api.time.sleep", waits.append)
    api.backoff = 2
    api.wait_before_retry(0, error.value.retry_after)
    api.wait_before_retry(9)
    assert waits == ([2, 30] if failure == "transport" else [30, 30])


@pytest.mark.parametrize("option,value", [("poll_timeout", float("nan")), ("timeout", float("inf")), ("backoff", float("nan"))])
def test_nonfinite_timeout_rejected(option, value):
    with pytest.raises(PortError, match="Invalid"):
        API("dify", "http://offline.invalid", key="key", **{option: value})


def test_resource_record_limit(api_factory):
    api = api_factory("dify", "http://offline.invalid")
    api.http.close()
    api.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [], "total": 100001, "has_more": True})))
    with pytest.raises(PortError, match="100000"):
        api.pages("documents", "dataset")


@pytest.mark.parametrize("system,fields", [("dify", {"enabled": False}), ("dify", {"archived": True}),
                                           ("dify", {"indexing_status": "indexing"}), ("dify", {"doc_form": "hierarchical_model"}),
                                           ("ragflow", {"status": "0"})])
def test_unsupported_document_state_is_explicit(system, fields, api_factory):
    api = api_factory(system, "http://offline.invalid")
    with pytest.raises(PortError):
        api.normalize("documents", {"id": "doc", "name": "source.txt", **fields})
