"""`ragwrench audit`: categories, measured auto-fix counts, fix commands, formats and exit codes."""
import json
from pathlib import Path

import pytest

from ragwrench import cli
from ragwrench.audit import SCHEMA
from ragwrench.audit.cli import main

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/audit/chunks.jsonl"
SOURCE = ROOT / "examples/audit/handbook.md"


def run(capsys, *args):
    code = main([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_chunks(path, texts):
    path.write_text("".join(json.dumps({"id": f"c{i}", "text": t}) + "\n" for i, t in enumerate(texts, 1)))
    return path


def test_example_reports_every_rule_with_source(capsys):
    code, out, _ = run(capsys, EXAMPLE, "--source", SOURCE, "--format", "json")
    assert code == 1
    doc = json.loads(out)
    assert doc["schema"] == SCHEMA
    found = {c["rule"]: c["found"] for c in doc["summary"]["categories"]}
    assert found == {"CL001": 2, "CL002": 2, "CL003": 1, "CL004": 1, "CL005": 1, "CL006": 2, "CL007": 1, "CL008": 1}
    assert doc["summary"]["findings"] == 11
    assert doc["summary"]["auto_fixable"] == 8
    gap = doc["files"][0]["gaps"][0]
    assert (gap["line_start"], gap["line_end"]) == (48, 50)
    assert gap["preview"].startswith("## Webhooks")


def test_auto_fix_count_matches_real_lint_fix(capsys, tmp_path):
    from ragwrench.lint.cli import main as lint_main
    code, out, _ = run(capsys, EXAMPLE, "--source", SOURCE, "--format", "json")
    doc = json.loads(out)
    repair = doc["top_fixes"][0]
    assert repair["kind"] == "lint-fix" and repair["resolves"] == 8
    assert repair["commands"][0].startswith("ragwrench lint fix ")
    output = tmp_path / "fixed.jsonl"
    assert lint_main(["fix", str(EXAMPLE), "--source", str(SOURCE), "-o", str(output)]) == 1
    capsys.readouterr()
    code, out, _ = run(capsys, output, "--source", SOURCE, "--format", "json")
    assert json.loads(out)["summary"]["findings"] == 11 - 8


def test_text_report_lists_fix_commands_and_is_plain_without_tty(capsys):
    code, out, _ = run(capsys, EXAMPLE, "--source", SOURCE)
    assert code == 1
    assert "\x1b[" not in out
    assert "Broken code fences" in out and "Worst chunks" in out
    assert "ragwrench lint fix " in out and "ragwrench table-chunk split " in out
    assert "--fail-on error: FAIL (exit 1)" in out


def test_without_source_coverage_is_not_checked(capsys):
    code, out, _ = run(capsys, EXAMPLE)
    assert "needs --source" in out
    assert "SOURCE.md" in out  # table-chunk needs the original document


def test_clean_export_exits_0(capsys, tmp_path):
    path = write_chunks(tmp_path / "clean.jsonl", ["A complete first paragraph about the topic.",
                                                   "Another complete paragraph with different words."])
    code, out, _ = run(capsys, path)
    assert code == 0
    assert "Nothing to fix." in out and "pass (exit 0)" in out


@pytest.mark.parametrize("threshold,expected", [("error", 0), ("warning", 1), ("none", 0)])
def test_fail_on_threshold(capsys, tmp_path, threshold, expected):
    path = write_chunks(tmp_path / "warn.jsonl", ["## Heading only\n", "Body text that follows the heading."])
    code, _, _ = run(capsys, path, "--fail-on", threshold)
    assert code == expected


def test_directory_skips_unreadable_files_and_sarif_has_one_run_per_file(capsys, tmp_path):
    write_chunks(tmp_path / "a.jsonl", ["## Heading only\n", "Body text that follows the heading."])
    write_chunks(tmp_path / "b.jsonl", ["A complete first paragraph about the topic."])
    (tmp_path / "config.json").write_text('{"not": "chunks"}')
    (tmp_path / "a.fixed.jsonl.changes.json").write_text("{}")
    code, out, _ = run(capsys, tmp_path, "--format", "json")
    doc = json.loads(out)
    assert [Path(f["path"]).name for f in doc["files"]] == ["a.jsonl", "b.jsonl"]
    assert [Path(s["path"]).name for s in doc["skipped"]] == ["config.json"]
    code, out, _ = run(capsys, tmp_path, "--format", "sarif")
    assert len(json.loads(out)["runs"]) == 2


def test_usage_and_input_errors_exit_2(capsys, tmp_path):
    assert run(capsys, tmp_path / "missing.jsonl")[0] == 2
    assert run(capsys, tmp_path)[0] == 2  # empty directory
    write_chunks(tmp_path / "a.jsonl", ["text one is here", "text two is here"])
    assert run(capsys, tmp_path, "--source", SOURCE)[0] == 2
    with pytest.raises(SystemExit) as exc:
        main(["--format", "xml", str(EXAMPLE)])
    assert exc.value.code == 2


def test_registered_first_in_first_group():
    assert cli.COMMANDS[0].name == "audit"
    assert cli.COMMANDS[0].group == cli.GROUPS[0]
    assert cli.main(["audit", str(EXAMPLE), "--fail-on", "none", "--format", "json"]) == 0
