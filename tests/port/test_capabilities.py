"""Read-only contract inspection before an operator selects any bundle."""
import json
from pathlib import Path

import httpx
import pytest

from ragwrench.port.bundle import load_bundle
from ragwrench.port.cli import main
from ragwrench.port.common import PortError
from ragwrench.port.operations import make_plan


@pytest.mark.parametrize("system,profile", [
    ("dify", "dify-f8fc371a"), ("ragflow", "ragflow-1.0"),
    ("ragflow", "ragflow-0.27.3"),
])
def test_offline_capabilities_contract(system, profile, monkeypatch, capsys, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline capabilities must not construct an API/client, read input or write files")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIFY_KB_API_KEY", raising=False)
    monkeypatch.delenv("RAGFLOW_API_KEY", raising=False)
    monkeypatch.setattr("ragwrench.port.cli.API", forbidden)
    monkeypatch.setattr("ragwrench.port.cli.load_bundle", forbidden)
    monkeypatch.setattr("ragwrench.port.cli.import_status", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert main(["capabilities", system, "--api-profile", profile]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    result = json.loads(captured.out)
    assert result["system"] == system and result["api_profile"] == profile
    assert result["scope"] == "import" and result["bundle_format"] == "kbbundle/1"
    assert result["profile_resolution"] == {
        "method": "explicit", "server_checked": False, "exact_server_version_verified": False}
    assert set(result["fields"]["documents"]) == {"id", "name", "doc_form", "metadata"}
    chunks = result["fields"]["chunks"]
    assert set(chunks) == {"id", "doc_id", "position", "content", "keywords", "questions",
                           "answer", "enabled", "metadata", "content_sha256"}
    for field in ("content", "keywords", "enabled"):
        assert chunks[field]["import"] == "preserved"
    assert chunks["answer"]["import"] == ("conditional" if system == "dify" else "requires_allow_lossy")
    assert chunks["questions"]["import"] == ("requires_allow_lossy" if system == "dify" else "preserved")
    assert chunks["metadata"]["import"] == result["fields"]["documents"]["metadata"]["import"] == "requires_allow_lossy"
    assert chunks["position"]["import"] == "not_restored"
    assert result["limits"]["default_max_chunk_bytes"] == 16000
    assert result["limits"]["max_chunk_bytes_ceiling"] == 1048576
    if system == "ragflow":
        assert result["dataset_creation"] == ({"chunk_method": "naive"} if profile.endswith("0.27.3")
                                               else {"parse_type": 1, "parser_id": "general"})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("system,profile", [("dify", "dify-f8fc371a"), ("ragflow", "ragflow-1.0")])
def test_default_profile_has_no_server_claim(system, profile, capsys):
    assert main(["capabilities", system]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["api_profile"] == profile
    assert result["profile_resolution"] == {
        "method": "default", "server_checked": False, "exact_server_version_verified": False}


@pytest.mark.parametrize("args,expected", [
    (["dify", "--api-profile", "auto"], "Unsupported API profile"),
    (["dify", "--api-profile", "ragflow-1.0"], "Unsupported API profile"),
    (["ragflow", "--api-profile", "dify-f8fc371a"], "Unsupported API profile"),
    (["ragflow", "--api-profile", "auto"], "requires --base-url"),
    (["ragflow", "--base-url", "http://localhost"], "only with --api-profile auto"),
    (["dify", "--key-env", "SECRET"], "only with --api-profile auto"),
    (["ragflow", "--page-size", "2"], "only with --api-profile auto"),
    (["ragflow", "--retries", "0"], "only with --api-profile auto"),
    (["ragflow", "--timeout", "1"], "only with --api-profile auto"),
    (["ragflow", "--backoff", "0"], "only with --api-profile auto"),
])
def test_invalid_selection_fails_before_client(args, expected, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid selection must fail before client construction")
    monkeypatch.setattr("ragwrench.port.cli.API", forbidden)
    assert main(["capabilities", *args]) == 1
    captured = capsys.readouterr()
    assert not captured.out and expected in captured.err


@pytest.mark.parametrize("args", [[], ["other"], ["dify", "--api-profile", "unknown"],
    ["ragflow", "some-bundle"], ["dify", "--state", "state.json"], ["dify", "--allow-lossy"]])
def test_capabilities_argument_errors(args, capsys):
    with pytest.raises(SystemExit) as error:
        main(["capabilities", *args])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out and "error:" in captured.err


@pytest.fixture
def mock_discovery(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAPABILITIES_TEST_KEY", "private-test-token")
    original_client = httpx.Client
    calls = []
    def install(handler):
        def recorded(request):
            assert request.method == "GET" and request.url.path == "/api/v1/datasets"
            assert request.headers["Authorization"] == "Bearer private-test-token"
            calls.append(request)
            return handler(request)
        monkeypatch.setattr(httpx, "Client", lambda **kw: original_client(
            transport=httpx.MockTransport(recorded), **kw))
        return calls
    return install


def auto_args(*extra):
    return ["capabilities", "ragflow", "--api-profile", "auto", "--base-url", "http://fixture.invalid",
            "--key-env", "CAPABILITIES_TEST_KEY", "--retries", "0", *extra]


@pytest.mark.parametrize("profile,signature", [("ragflow-1.0", {"parser_id": "general"}),
                                               ("ragflow-0.27.3", {"chunk_method": "naive"})])
def test_auto_capabilities_paginate_read_only(profile, signature, mock_discovery, capsys, tmp_path):
    def handler(request):
        page = int(request.url.params["page"])
        records = [{"id": f"private-id-{page}", "name": "private-dataset-name", **signature}] if page <= 2 else []
        return httpx.Response(200, json={"code": 0, "data": records})
    calls = mock_discovery(handler)
    assert main(auto_args("--page-size", "1")) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert not captured.err and result["api_profile"] == profile
    assert result["profile_resolution"] == {
        "method": "dataset_signatures", "server_checked": True, "exact_server_version_verified": False}
    assert [r.url.params["page"] for r in calls] == ["1", "2", "3"]
    assert "private-" not in captured.out and "fixture.invalid" not in captured.out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("records", [[], [{"id": "d"}],
    [{"id": "d", "parser_id": "general", "chunk_method": "naive"}],
    [{"id": "a", "parser_id": "general"}, {"id": "b", "chunk_method": "naive"}],
    [{"id": "a", "parser_id": "general"}, {"id": "b"}],
])
def test_auto_capabilities_refuse_ambiguous_records(records, mock_discovery, capsys, tmp_path):
    calls = mock_discovery(lambda r: httpx.Response(200, json={"code": 0, "data": records}))
    assert main(auto_args()) == 1
    captured = capsys.readouterr()
    assert not captured.out and "explicit --api-profile" in captured.err
    assert len(calls) == 1 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status,body,expected", [
    (401, {"private": "private-remote-body"}, "HTTP 401"),
    (503, {"private": "private-remote-body"}, "HTTP 503"),
    (302, {"private": "private-remote-body"}, "Redirect refused"),
    (200, {"code": 42, "data": "private-remote-body"}, "error envelope"),
    (200, {"code": 0, "data": "private-remote-body"}, "pagination envelope"),
])
def test_auto_capabilities_api_errors_are_safe(status, body, expected, mock_discovery, capsys, tmp_path):
    calls = mock_discovery(lambda r: httpx.Response(status, json=body))
    assert main(auto_args()) == 1
    captured = capsys.readouterr()
    assert not captured.out and expected in captured.err
    assert "private-" not in captured.err and len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_auto_capabilities_transport_error(mock_discovery, capsys):
    def handler(request):
        raise httpx.ConnectError("private-transport-message", request=request)
    calls = mock_discovery(handler)
    assert main(auto_args()) == 1
    captured = capsys.readouterr()
    assert not captured.out and "transport failure" in captured.err and "private-" not in captured.err
    assert len(calls) == 1


def test_auto_capabilities_missing_key(monkeypatch, capsys):
    monkeypatch.delenv("CAPABILITIES_TEST_KEY", raising=False)
    assert main(auto_args()) == 1
    captured = capsys.readouterr()
    assert not captured.out and "API key" in captured.err


@pytest.mark.parametrize("option,value", [("--page-size", "0"), ("--page-size", "101"),
    ("--retries", "11"), ("--timeout", "0"), ("--backoff", "-1"), ("--timeout", "nan")])
def test_auto_capabilities_reject_invalid_http_limits(option, value, mock_discovery, capsys):
    calls = mock_discovery(lambda r: pytest.fail("Invalid limits must fail before HTTP"))
    assert main(auto_args(option, value)) == 1
    captured = capsys.readouterr()
    assert not captured.out and "Invalid page size, retry, backoff or timeout" in captured.err
    assert not calls


def test_auto_capabilities_invalid_json(mock_discovery, capsys):
    calls = mock_discovery(lambda r: httpx.Response(200, content=b"private-invalid-json"))
    assert main(auto_args()) == 1
    captured = capsys.readouterr()
    assert not captured.out and "invalid JSON" in captured.err and "private-" not in captured.err
    assert len(calls) == 1


@pytest.mark.parametrize("system,profile", [("dify", "dify-f8fc371a"),
    ("ragflow", "ragflow-1.0"), ("ragflow", "ragflow-0.27.3")])
def test_reported_loss_boundaries_match_import_preflight(system, profile, example, capsys):
    assert main(["capabilities", system, "--api-profile", profile]) == 0
    report = json.loads(capsys.readouterr().out)
    manifest, docs, chunks = load_bundle(example)
    docs, chunks = docs[:1], chunks[:1]
    chunk = chunks[0]
    # Independent operator scenario: QA plus embedding questions and descriptive metadata.
    docs[0].update(doc_form="qa_model", metadata={"topic": "synthetic"})
    chunk.update(answer="Seven days.", questions=["How long?"], metadata={"image_id": "synthetic"})
    with pytest.raises(PortError, match="Document metadata cannot be restored"):
        make_plan(manifest, docs, chunks, system)
    assert report["fields"]["documents"]["metadata"]["import"] == "requires_allow_lossy"
    docs[0]["metadata"] = {}
    with pytest.raises(PortError, match="Unsupported fields"):
        make_plan(manifest, docs, chunks, system)
    plan, projected, forms = make_plan(manifest, docs, chunks, system, allow_lossy=True)
    actual = next(iter(projected.values()))
    assert plan["skipped"] == 0 and actual["metadata"] == {}
    assert report["fields"]["chunks"]["metadata"]["import"] == "requires_allow_lossy"
    if system == "dify":
        assert actual["content"] == chunk["content"] and actual["answer"] == "Seven days."
        assert actual["questions"] == [] and forms[docs[0]["id"]] == "qa_model"
        docs[0]["doc_form"] = "text_model"
        with pytest.raises(PortError, match="text documents require no answer"):
            make_plan(manifest, docs, chunks, system, allow_lossy=True)
        assert report["fields"]["chunks"]["answer"]["import"] == "conditional"
    else:
        assert actual["content"] == f"Question: {chunk['content']}\nAnswer: Seven days."
        assert actual["answer"] is None and actual["questions"] == ["How long?"]
        assert forms[docs[0]["id"]] == "text_model"
        assert report["fields"]["chunks"]["answer"]["import"] == "requires_allow_lossy"
    for key in ("keywords", "enabled"):
        assert actual[key] == chunk[key] and report["fields"]["chunks"][key]["import"] == "preserved"
