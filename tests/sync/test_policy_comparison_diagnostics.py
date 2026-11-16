"""Health-sensitive offline comparisons retain the complete drift report."""
import json

import pytest

from ragwrench.sync.cli import main
from .test_policy_comparison import bad, ok, report


def test_fail_on_diagnostics_for_equal_archives(tmp_path, capsys):
    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path in paths:
        path.write_text(json.dumps(report([ok("first"), bad("last")])), encoding="utf-8")
    argv = ["compare-policies", *map(str, paths)]
    assert main(argv) == 0
    default = capsys.readouterr()
    assert main([*argv, "--fail-on-diagnostics"]) == 3
    strict = capsys.readouterr()
    assert strict == default and not strict.err
    assert json.loads(strict.out)["identical"]


@pytest.mark.parametrize("archive_mode", ["none", "new", "existing"])
@pytest.mark.parametrize("status", ["unmanaged", "unsupported", "invalid", "unavailable"])
@pytest.mark.parametrize("case", ["equal", "unchanged-with-drift", "before-changed", "after-changed", "removed", "added"])
def test_each_diagnostic_status_and_side_keeps_exact_report(tmp_path, capsys, monkeypatch,
                                                          archive_mode, status, case):
    import io
    import sqlite3
    from pathlib import Path
    import ragwrench.sync.cli as cli
    import ragwrench.sync.inventory as inventory
    import ragwrench.sync.recovery as recovery
    import ragwrench.sync.store as store
    from ragwrench.sync.embedders import HashEmbedder
    from ragwrench.sync.policy_comparison import compare_policies, comparison_json

    # The diagnostic is last; a different healthy row can cause all the drift.
    diagnostic = bad("z-last", status, message="Überprüfung 合成" if status in ("invalid", "unavailable") else None)
    before, after = report([ok("a-first"), diagnostic]), report([ok("a-first"), diagnostic], "recovery")
    if case == "unchanged-with-drift": after["collections"] = [ok("a-first", limit=0.25), diagnostic]
    elif case == "before-changed": after["collections"] = [ok("a-first"), ok("z-last")]
    elif case == "after-changed": before["collections"] = [ok("a-first"), ok("z-last")]
    elif case == "removed": after["collections"] = [ok("a-first")]
    elif case == "added": before["collections"] = [ok("a-first")]
    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path, value in zip(paths, (before, after)):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    saved = [p.read_bytes() for p in paths]
    expected = comparison_json(compare_policies(before, after))
    output = tmp_path / "diff.json"
    if archive_mode == "existing": output.write_bytes(b"previous archive\n")
    argv = ["compare-policies", *map(str, paths)]
    if archive_mode != "none": argv += ["-o", str(output)]
    reads, real_open = [], io.open
    def only_inputs(path, mode="r", *a, **kw):
        # tempfile uses io.open(parent, "wb", opener=...) for the archive.
        if mode == "wb":
            assert archive_mode != "none" and path == tmp_path and callable(kw.get("opener"))
            return real_open(path, mode, *a, **kw)
        assert path in paths and mode == "rb"
        reads.append(path)
        return real_open(path, mode, *a, **kw)
    def forbidden(*a, **kw): pytest.fail("diagnostic exit must only read the two archives")
    with monkeypatch.context() as guard:
        guard.setattr(io, "open", only_inputs)
        for obj, name in ((cli, "local_client"), (cli, "list_policies"), (cli, "make_embedder"),
                          (inventory, "load_json"), (recovery, "recovery_catalog"),
                          (store, "QdrantClient"), (sqlite3, "connect"), (HashEmbedder, "embed")):
            guard.setattr(obj, name, forbidden)
        assert main(argv) == (0 if case == "equal" else 10)
        default = capsys.readouterr()
        assert main([*argv, "--fail-on-diagnostics"]) == 3
        strict = capsys.readouterr()
    assert reads == paths * 2
    assert strict == default and not strict.err and strict.out == expected
    assert strict.out.endswith("\n") and not strict.out.endswith("\n\n")
    result = json.loads(strict.out)
    assert result["identical"] == (case == "equal")
    if case == "removed": assert result["removed"] == [diagnostic]
    if case == "added": assert result["added"] == [diagnostic]
    if case == "before-changed": assert result["changed"][0]["before"] == diagnostic
    if case == "after-changed": assert result["changed"][0]["after"] == diagnostic
    if archive_mode != "none": assert output.read_bytes() == strict.out.encode("utf-8")
    else: assert not output.exists()
    assert [p.read_bytes() for p in paths] == saved
    assert not list(tmp_path.glob(".chunksync-*.tmp"))


@pytest.mark.parametrize("archive_mode", ["none", "new", "existing"])
@pytest.mark.parametrize("case", ["empty", "healthy-equal", "healthy-drift", "healthy-added", "healthy-removed"])
def test_healthy_comparisons_keep_default_exits(tmp_path, capsys, archive_mode, case):
    before = report() if case == "empty" else report([ok()])
    after = report(before["collections"], "recovery")
    if case == "healthy-drift": after["collections"] = [ok(limit=0.25)]
    elif case == "healthy-added": after["collections"].append(ok("new"))
    elif case == "healthy-removed": after["collections"] = []
    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path, value in zip(paths, (before, after)): path.write_text(json.dumps(value))
    saved = [p.read_bytes() for p in paths]
    output = tmp_path / "diff.json"
    if archive_mode == "existing": output.write_bytes(b"old")
    argv = ["compare-policies", *map(str, paths)]
    if archive_mode != "none": argv += ["--output", str(output)]
    expected = 0 if case in ("empty", "healthy-equal") else 10
    assert main(argv) == expected
    default = capsys.readouterr()
    assert main([*argv, "--fail-on-diagnostics"]) == expected
    strict = capsys.readouterr()
    assert strict == default and not strict.err
    assert json.loads(strict.out)["identical"] == (expected == 0)
    if archive_mode != "none": assert output.read_bytes() == strict.out.encode("utf-8")
    else: assert not output.exists()
    assert [p.read_bytes() for p in paths] == saved


def test_mixed_statuses_preserve_full_sorted_rows(tmp_path, capsys):
    rows = [bad(f"c{i}", status) for i, status in enumerate(("unmanaged", "unsupported", "invalid", "unavailable"))]
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(report([ok("keep"), *reversed(rows)])))
    changed = ok("keep", limit=0.25)
    after.write_text(json.dumps(report([changed, rows[-1]], "recovery")))
    argv = ["compare-policies", str(before), str(after)]
    assert main(argv) == 10
    default = capsys.readouterr()
    assert main([*argv, "--fail-on-diagnostics"]) == 3
    strict = capsys.readouterr()
    assert strict == default and not strict.err
    diff = json.loads(strict.out)
    assert diff["removed"] == rows[:-1] and not diff["added"]
    assert diff["changed"] == [{"collection": "keep", "before": ok("keep"), "after": changed}]
