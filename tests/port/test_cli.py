import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ragwrench.port.bundle import load_bundle, write_bundle
from ragwrench.port.cli import main
from ragwrench.port.common import atomic_json, read_json
from ragwrench.port.fakes import FakeState, running_fake


def run(*args, key=None):
    env = {k: v for k, v in os.environ.items() if k not in ("DIFY_KB_API_KEY", "RAGFLOW_API_KEY")}
    if key:
        env.update(DIFY_KB_API_KEY=key, RAGFLOW_API_KEY=key)
    return subprocess.run([sys.executable, "-m", "ragwrench.port", *map(str, args)],
                          env=env, capture_output=True, text=True, timeout=15)


def test_validate_cli_success_and_failure(bundle):
    good = run("validate", bundle)
    assert good.returncode == 0 and json.loads(good.stdout)["chunks"] == 15
    (bundle / "chunks.jsonl").write_text("broken")
    bad = run("validate", bundle)
    assert bad.returncode == 1 and "SHA256" in bad.stderr and "Traceback" not in bad.stderr


def test_offline_dry_run_without_credentials_or_state_writes(example, tmp_path):
    out = tmp_path / "absent" / "state.json"
    result = run("import", "dify", example, "--base-url", "http://127.0.0.1:1/v1",
                 "--dataset-name", "offline", "--state", out, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["dry_run"]
    assert not out.parent.exists()


def test_missing_credentials_is_clean_error(example, tmp_path):
    result = run("import", "ragflow", example, "--base-url", "http://127.0.0.1:1",
                 "--dataset-name", "no-key", "--state", tmp_path / "state.json")
    assert result.returncode == 1 and "API key" in result.stderr and "Traceback" not in result.stderr


def test_bad_key_exit_and_no_secret_leak(example, tmp_path):
    state = FakeState("ragflow")
    key = "never-log-this-private-token"
    with running_fake(state) as url:
        out = tmp_path / "bundle"
        result = run("export", "ragflow", "--base-url", url, "--dataset-id", "dataset", "-o", out, key=key)
    assert result.returncode == 1 and "401" in result.stderr
    assert key not in result.stderr + result.stdout
    for file in out.iterdir():
        assert key.encode() not in file.read_bytes()


def test_partial_skip_exit_code(example, tmp_path):
    state = FakeState("ragflow")
    with running_fake(state) as url:
        result = run("import", "ragflow", example, "--base-url", url, "--dataset-name", "partial",
                     "--state", tmp_path / "state.json", "--max-chunk-bytes", "1", "--on-error", "skip", key="fake-key")
    assert result.returncode == 4, result.stderr
    assert json.loads(result.stdout)["skipped"] == 15


def test_state_cannot_overwrite_bundle(example):
    before = (example / "manifest.json").read_bytes()
    result = run("import", "dify", example, "--base-url", "http://127.0.0.1:1", "--dataset-name", "x",
                 "--state", example / "manifest.json", "--dry-run")
    assert result.returncode == 1 and "overwrite" in result.stderr
    assert (example / "manifest.json").read_bytes() == before


def test_remote_document_names_are_not_paths(example, tmp_path):
    m, docs, chunks = load_bundle(example)
    docs[0]["name"] = "../../outside-token.txt"
    bundle = tmp_path / "input"
    write_bundle(bundle, m["source"], {}, docs, chunks)
    state = FakeState("ragflow")
    with running_fake(state) as url:
        result = run("import", "ragflow", bundle, "--base-url", url, "--dataset-name", "paths",
                     "--state", tmp_path / "state.json", key="fake-key")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "outside-token.txt").exists()
    assert any(d["name"].startswith("../../") for ds in state.datasets.values() for d in ds["documents"].values())


def test_invalid_api_json_is_clean_error(example, tmp_path):
    state = FakeState("dify")
    state.faults = [{"status": 200, "body": {"data": "private-malformed-payload"}}]
    with running_fake(state) as url:
        result = run("export", "dify", "--base-url", url, "--dataset-id", "dataset", "-o", tmp_path / "out", key="fake-key")
    assert result.returncode == 1 and "private-malformed" not in result.stderr and "Traceback" not in result.stderr


def test_main_parser_error():
    with pytest.raises(SystemExit) as error:
        main(["unknown-command"])
    assert error.value.code == 2


def test_cli_bounded_chunk_recovery(example, tmp_path):
    target = FakeState("ragflow")
    target.faults = [{"method": "POST", "path_suffix": "/chunks", "phase": "after", "status": 503}]
    with running_fake(target) as url:
        args = ("import", "ragflow", example, "--base-url", url, "--dataset-name", "recovery",
                "--state", tmp_path / "mapping.json", "--chunk-retries", "1", "--backoff", "0")
        result = run(*args, key="fake-key")
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["chunks"] == 15
        assert sum(r["method"] == "POST" and r["path"].endswith("/chunks") for r in target.requests) == 15
        before = len(target.requests)
        assert run(*args, key="fake-key").returncode == 0
        assert all(r["method"] == "GET" for r in target.requests[before:])


@pytest.mark.parametrize("budget", [-1, 11])
def test_invalid_chunk_retries_cli(example, tmp_path, budget):
    state = tmp_path / "absent" / "mapping.json"
    result = run("import", "ragflow", example, "--base-url", "http://127.0.0.1:1",
                 "--dataset-name", "invalid", "--state", state, "--chunk-retries", budget, "--dry-run")
    assert result.returncode == 1 and "chunk-retries must be 0..10" in result.stderr
    assert not state.parent.exists()


def test_cli_auto_profile_and_offline_failure(example, tmp_path):
    mapping = tmp_path / "absent" / "mapping.json"
    result = run("import", "ragflow", example, "--base-url", "http://127.0.0.1:1",
                 "--api-profile", "auto", "--dataset-name", "offline", "--state", mapping, "--dry-run")
    assert result.returncode == 1 and "explicit --api-profile" in result.stderr
    assert not mapping.parent.exists()
    target = FakeState("ragflow", api_profile="ragflow-0.27.3")
    target.seed(example)
    with running_fake(target) as url:
        args = ("import", "ragflow", example, "--base-url", url, "--api-profile", "auto",
                "--dataset-name", "legacy", "--state", mapping)
        result = run(*args, key="fake-key")
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["api_profile"] == "ragflow-0.27.3"
        result = run("verify", "ragflow", example, "--base-url", url, "--api-profile", "auto",
                     "--state", mapping, key="fake-key")
        assert result.returncode == 0 and json.loads(result.stdout)["verified"], result.stderr


@pytest.mark.parametrize("system,profile", [("dify", "auto"), ("dify", "ragflow-0.27.3"), ("ragflow", "dify-f8fc371a")])
def test_cli_rejects_cross_system_profiles(system, profile, example, tmp_path):
    mapping = tmp_path / "absent" / "state.json"
    result = run("import", system, example, "--base-url", "http://127.0.0.1:1",
                 "--api-profile", profile, "--dataset-name", "invalid", "--state", mapping, "--dry-run")
    assert result.returncode == 1 and "Unsupported API profile" in result.stderr
    assert not mapping.parent.exists()
