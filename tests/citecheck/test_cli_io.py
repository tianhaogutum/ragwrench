import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ragwrench.citecheck.cli import main
from ragwrench.citecheck.io import loads, read_bytes, write_output
from ragwrench.citecheck.model import InputError, MAX_INPUT_BYTES, MAX_ANSWER, MAX_SOURCE_TEXT
from ragwrench.citecheck.adapters import adapt
from ragwrench.citecheck.checker import check

ROOT = Path(__file__).resolve().parents[2]
GOOD = ROOT / "examples/cite-check/dify_response.json"
BAD = ROOT / "examples/cite-check/dify_errors.json"


def cli(*args, cwd=None, stdin=None):
    return subprocess.run([sys.executable, "-m", "ragwrench.citecheck", *map(str, args)], input=stdin,
                          capture_output=True, text=True, cwd=cwd, timeout=15)


def test_installed_console_script():
    result = subprocess.run([str(Path(sys.executable).parent / "ragwrench"), "cite-check", "--version"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and result.stdout.strip() == "0.1.0"


def test_clean_end_to_end_stdout_json():
    result = cli("check", "--format", "dify", GOOD, "--report", "json")
    assert result.returncode == 0 and not result.stderr
    report = json.loads(result.stdout)
    assert report["records"][0]["findings"] == []
    assert not report["gate_failed"]
    assert report["format"] == "dify"


def test_broken_end_to_end_still_writes_complete_report(tmp_path):
    result = cli("check", "--format", "dify", BAD, "--report", "json", "-o", "report.json", cwd=tmp_path)
    assert result.returncode == 1 and not result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    findings = report["records"][0]["findings"]
    assert {f["code"] for f in findings} == {"CC001", "CC004", "CC005", "CC007"}
    assert report["gate_failed"]


@pytest.mark.parametrize("level,expected", [("error", 0), ("warning", 1), ("hint", 1), ("info", 1), ("none", 0)])
def test_gate_thresholds(level, expected):
    result = cli("check", "-", "--fail-on", level, stdin=json.dumps({"answer": "The rover uses solar panels.", "sources": []}))
    assert result.returncode == expected
    assert "CC002" in result.stdout


def test_stdin_jsonl():
    result = cli("check", "-", "--jsonl", "--report", "json", stdin=(ROOT / "examples/cite-check/generic.jsonl").read_text())
    assert result.returncode == 0
    assert json.loads(result.stdout)["record_count"] == 2


def test_backtick_code_jsonl_passes_strict_gate():
    result = cli("check", ROOT / "examples/cite-check/backtick_code.jsonl", "--jsonl", "--report", "json", "--fail-on", "info")
    assert result.returncode == 0 and not result.stderr
    report = json.loads(result.stdout)
    assert report["record_count"] == 3 and not report["gate_failed"]
    for record in report["records"]:
        assert record["findings"] == []
        sentence, = record["sentences"]
        assert sentence["citation_indices"] == [1]


@pytest.mark.parametrize("strict,exit_code", [(False, 0), (True, 1)])
def test_unicode_quote_normalization_cli(strict, exit_code):
    result = cli("check", "-", "--report", "json", *(["--strict-quotes"] if strict else []),
                 stdin=json.dumps({"answer": "手册说「信号。稳定」【1】。", "sources": [
                     {"idx": 1, "text": "手册说信号.稳定。"}]}))
    assert result.returncode == exit_code and not result.stderr
    output = json.loads(result.stdout)
    finding, = output["records"][0]["findings"]
    assert finding["code"] == "CC008"
    assert finding["evidence"]["quote"] == "信号。稳定"
    assert output["gate_failed"] is strict


def test_escaped_backtick_citation_error_keeps_unicode_offsets():
    answer = r"说明🙂 \`literal [9]\`"
    result = cli("check", "-", "--report", "json",
                 stdin=json.dumps({"answer": answer, "sources": []}))
    assert result.returncode == 1 and not result.stderr
    finding, = json.loads(result.stdout)["records"][0]["findings"]
    assert finding["code"] == "CC001"
    assert finding["evidence"]["start"] == answer.index("[9]")
    assert finding["evidence"]["end"] == answer.index("[9]") + 3


@pytest.mark.parametrize("payload,message", [("{bad", "invalid JSON"), ('{"answer":"secret", "answer":"other"}', "duplicate JSON"),
                                            ('{"x": NaN}', "non-finite"), ("[]", "object"), ("{}", "missing")])
def test_bad_input_has_no_traceback_or_payload_leak(payload, message):
    result = cli("check", "-", stdin=payload)
    assert result.returncode == 2 and message in result.stderr
    assert not result.stdout and "Traceback" not in result.stderr and "secret" not in result.stderr


def test_wrong_adapter():
    result = cli("check", "--format", "llamaindex", GOOD)
    assert result.returncode == 2 and "response" in result.stderr and "--format" in result.stderr


def test_invalid_utf8(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_bytes(b"\xff\xfe")
    result = cli("check", path)
    assert result.returncode == 2 and "UTF-8" in result.stderr


def test_jsonl_failure_is_atomic(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"answer":"The rover uses solar panels [1].","sources":[]}\n{}\n')
    result = cli("check", path, "--jsonl", "-o", "result.md", cwd=tmp_path)
    assert result.returncode == 2 and "record 2" in result.stderr
    assert not (tmp_path / "result.md").exists()


@pytest.mark.parametrize("extra", [("--support-threshold", "nan"), ("--misattribution-margin", "2"), ("--min-claim-tokens", "0")])
def test_bad_cli_config(extra):
    result = cli("check", GOOD, "--format", "dify", *extra)
    assert result.returncode == 2 and "ragwrench cite-check:" in result.stderr


def test_output_path_and_clobber_protection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "escape-report.md"
    with pytest.raises(InputError, match="within"):
        write_output("../escape-report.md", "unsafe")
    assert not outside.exists()
    original = tmp_path / "original.json"
    original.write_text("input")
    with pytest.raises(InputError, match="input"):
        write_output("original.json", "bad", force=True, inputs=(str(original),))
    assert original.read_text() == "input"
    write_output("report.json", "first")
    with pytest.raises(InputError, match="already exists"):
        write_output("report.json", "second")
    write_output("report.json", "second", force=True)
    assert (tmp_path / "report.json").read_text() == "second"
    assert not list(tmp_path.glob("tmp*"))


def test_output_symlink_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    (tmp_path / "alias-file").symlink_to(tmp_path / "real/target")
    for path in ("alias/report.json", "alias-file"):
        with pytest.raises(InputError, match="symlink"):
            write_output(path, "bad", force=True)
    assert not list((tmp_path / "real").iterdir())


def test_fifo_refused_without_blocking(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    result = cli("check", fifo)
    assert result.returncode == 2 and "regular file" in result.stderr


def test_input_byte_cap(tmp_path):
    path = tmp_path / "huge.json"
    with path.open("wb") as stream:
        stream.truncate(MAX_INPUT_BYTES + 1)
    with pytest.raises(InputError, match="bytes"):
        read_bytes(str(path))


@pytest.mark.parametrize("record,match", [
    ({"answer": "x" * (MAX_ANSWER + 1), "sources": []}, "answer exceeds"),
    ({"answer": "x", "sources": [{"idx": 1, "text": "x" * (MAX_SOURCE_TEXT + 1)}]}, "source text exceeds"),
    ({"answer": "x", "sources": [{"idx": n, "text": "a"} for n in range(1, 258)]}, "256 sources"),
    ({"answer": "x", "sources": [{"idx": n, "text": "a" * 100_000} for n in range(1, 22)]}, "combined source"),
])
def test_record_bounds(record, match):
    with pytest.raises(InputError, match=match):
        adapt(record)


@pytest.mark.parametrize("answer,match", [("a" * 10_001, "sentence exceeds"), ("Fact.\n" * 1001, "1000 sentences")])
def test_prose_bounds(answer, match):
    with pytest.raises(InputError, match=match):
        check(adapt({"answer": answer, "sources": []}))


def test_no_execution_or_url_fetch_from_source_metadata(tmp_path):
    sentinel = tmp_path / "executed"
    payload = {"answer": "The rover uses solar panels [1].", "sources": [{"idx": 1, "text": "The rover uses solar panels.",
               "id": f"$(touch {sentinel})", "title": f"file://{sentinel}", "url": "http://127.0.0.1:1", "path": "../../escape"}]}
    result = cli("check", "-", "--report", "json", stdin=json.dumps(payload))
    assert result.returncode == 0
    assert not sentinel.exists() and "$(touch" not in result.stdout


def test_json_nesting_fails_cleanly():
    with pytest.raises(InputError, match="nesting"):
        loads("[" * 5000 + "0" + "]" * 5000)


def test_escaped_brackets_in_json_string_do_not_count_as_nesting():
    text = '["' + '[' * 100 + '\\"' + ']' * 100 + '"]'
    assert len(loads(text)[0]) == 201


def test_literal_quantity_bounds():
    from ragwrench.citecheck.lexical import quantities
    with pytest.raises(InputError, match="256 literal"):
        quantities("1 " * 257, limit=256)
    with pytest.raises(InputError, match="128 characters"):
        quantities("1" * 129)


def test_many_spaces_without_quantities_terminate():
    from ragwrench.citecheck.lexical import quantities
    # Exercises a past quadratic regex shape without a fragile timing assertion.
    assert quantities("ordinary prose " + " " * 90_000) == []


def test_json_unpaired_surrogate_rejected_cleanly():
    result = cli("check", "-", stdin='{"answer":"Fact \\ud800.","sources":[]}')
    assert result.returncode == 2 and "surrogate" in result.stderr
    assert "Traceback" not in result.stderr


def test_unknown_markdown_quote_links_cannot_render_remote_images():
    from ragwrench.citecheck.checker import markdown_report
    report = {"analyzed_sentence_count": 1, "source_count": 1, "notes": [], "findings": [
        {"code": "CC006", "severity": "error", "sentence": 1, "source_indices": [1],
         "message": "quoted span missing", "evidence": {"quote": "![secret](https://example.invalid/image)"}}]}
    rendered = markdown_report([report])
    assert "![secret]" not in rendered and "&#91;secret&#93;" in rendered
