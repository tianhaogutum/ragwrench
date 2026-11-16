"""Offline weakening gates change only the exit status of a complete comparison."""
import json

import pytest

from ragwrench.sync.cli import main
from .test_policy_comparison import bad, ok, override, report


def test_fail_on_policy_weakening_for_increased_limit(tmp_path, capsys):
    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path, limit in zip(paths, (0.25, 0.5)):
        path.write_text(json.dumps(report([ok(limit=limit)])), encoding="utf-8")
    argv = ["compare-policies", *map(str, paths)]
    assert main(argv) == 10
    default = capsys.readouterr()
    assert main([*argv, "--fail-on-policy-weakening"]) == 4
    strict = capsys.readouterr()
    assert strict == default and not strict.err
    assert json.loads(strict.out)["changed"][0]["collection"] == "kb"


def check_gates(tmp_path, capsys, before, after, archive_mode, weakens):
    from ragwrench.sync.policy_comparison import compare_policies, comparison_json
    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path, value in zip(paths, (before, after)):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    saved = [p.read_bytes() for p in paths]
    output = tmp_path / "diff.json"
    if archive_mode == "existing": output.write_bytes(b"previous diff\n")
    argv = ["compare-policies", *map(str, paths)]
    if archive_mode != "none": argv += ["--output", str(output)]
    expected = comparison_json(compare_policies(before, after))
    default_exit = 0 if json.loads(expected)["identical"] else 10
    diagnostics = any(row["status"] != "ok" for archive in (before, after) for row in archive["collections"])
    baseline = None
    for flags in ([], ["--fail-on-policy-weakening"], ["--fail-on-diagnostics"],
                  ["--fail-on-policy-weakening", "--fail-on-diagnostics"]):
        code = (3 if diagnostics and "--fail-on-diagnostics" in flags else
                4 if weakens and "--fail-on-policy-weakening" in flags else default_exit)
        assert main([*argv, *flags]) == code
        captured = capsys.readouterr()
        assert captured.out == expected and not captured.err
        if baseline is None: baseline = captured
        assert captured == baseline
        if archive_mode != "none": assert output.read_bytes() == expected.encode("utf-8")
        else: assert not output.exists()
        assert [p.read_bytes() for p in paths] == saved
    assert not list(tmp_path.glob(".chunksync-*.tmp"))
    return json.loads(expected)


@pytest.mark.parametrize("archive_mode", ["none", "new", "existing"])
@pytest.mark.parametrize("before_source,after_source,direction", [
    (old, new, direction)
    for old in ("explicit", "legacy-review", "default")
    for new in ("explicit", "legacy-review", "default")
    for direction in ("increase", "equal", "decrease")
    if not (direction == "increase" and old == "default" or direction == "decrease" and new == "default")
])
def test_effective_limits_across_sources(tmp_path, capsys, archive_mode, before_source, after_source, direction):
    old, new = ((0.25, 1) if direction == "increase" else
                (1, 0.25) if direction == "decrease" else (1, 1.0))
    before = report([ok(limit=old, source=before_source)])
    after = report([ok(limit=new, source=after_source)], "recovery")
    check_gates(tmp_path, capsys, before, after, archive_mode, direction == "increase")


@pytest.mark.parametrize("old,new,weakens", [
    (0, 1, True), (0, 5e-324, True), (0.5, 0.5000000000000001, True),
    (1, 0, False), (1, 1.0, False), (-0.0, 0, False), (0.5, 0.5, False),
])
def test_strict_numeric_boundaries(tmp_path, capsys, old, new, weakens):
    check_gates(tmp_path, capsys, report([ok(limit=old)]), report([ok(limit=new)]), "existing", weakens)


@pytest.mark.parametrize("archive_mode", ["none", "new", "existing"])
@pytest.mark.parametrize("old_status", ["ok", "unmanaged", "unsupported", "invalid", "unavailable"])
@pytest.mark.parametrize("new_status", ["ok", "unmanaged", "unsupported", "invalid", "unavailable"])
def test_no_inference_for_non_ok_rows(tmp_path, capsys, archive_mode, old_status, new_status):
    old = ok(limit=0.25) if old_status == "ok" else bad(status=old_status)
    new = ok(limit=1) if new_status == "ok" else bad(status=new_status)
    check_gates(tmp_path, capsys, report([old]), report([new]), archive_mode,
                old_status == new_status == "ok")


@pytest.mark.parametrize("archive_mode", ["none", "new", "existing"])
@pytest.mark.parametrize("case", ["empty", "added", "removed", "renamed", "pending", "override-only", "mixed", "retained-override"])
def test_exact_identity_and_unrelated_changes(tmp_path, capsys, archive_mode, case):
    before, after = [ok(limit=0.25)], [ok(limit=0.25)]
    if case == "empty": before, after = [], []
    elif case == "added": after.append(ok("new", limit=1))
    elif case == "removed": before.append(ok("old", limit=0))
    elif case == "renamed": after = [ok("KB", limit=1)]
    elif case == "pending": after = [ok(limit=0.25, pending="b" * 64)]
    elif case == "override-only": after = [ok(limit=0.25, override=override())]
    elif case == "retained-override":
        # Historical exception need not equal current limits or pending plan.
        before = [ok(limit=0.25, override=override(), pending="b" * 64)]
        after = [ok(limit=0.5, override=override(), pending="c" * 64)]
    else:
        before = [ok("z-rise", limit=0.25), ok("a-fall", limit=1), bad("removed"), bad("unchanged"), ok("same")]
        after = [ok("added", limit=1), ok("same"), bad("unchanged"), ok("a-fall", limit=0), ok("z-rise", limit=0.5)]
    result = check_gates(tmp_path, capsys, report(before), report(after, "recovery"), archive_mode,
                         case in ("mixed", "retained-override"))
    if case == "mixed":
        assert [r["collection"] for r in result["changed"]] == ["a-fall", "z-rise"]
        assert result["removed"] == [bad("removed")] and result["added"] == [ok("added", limit=1)]


def test_weakening_only_reads_two_archives(tmp_path, capsys, monkeypatch):
    import io
    import sqlite3
    import ragwrench.sync.cli as cli
    import ragwrench.sync.inventory as inventory
    import ragwrench.sync.recovery as recovery
    import ragwrench.sync.store as store
    from ragwrench.sync.embedders import HashEmbedder

    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path, limit in zip(paths, (0.25, 0.5)):
        path.write_text(json.dumps(report([bad("a-diagnostic"), ok("z-rise", limit=limit)])))
    saved = [p.read_bytes() for p in paths]
    output = tmp_path / "diff.json"
    reads, real_open = [], io.open
    def only_inputs(path, mode="r", *a, **kw):
        if mode == "wb":
            assert path == tmp_path and callable(kw.get("opener"))
        else:
            assert path in paths and mode == "rb"
            reads.append(path)
        return real_open(path, mode, *a, **kw)
    def forbidden(*a, **kw): pytest.fail("offline gate must not access stores or embeddings")
    baseline = None
    with monkeypatch.context() as guard:
        guard.setattr(io, "open", only_inputs)
        for obj, name in ((cli, "local_client"), (cli, "list_policies"), (cli, "make_embedder"),
                          (inventory, "load_json"), (recovery, "recovery_catalog"),
                          (store, "QdrantClient"), (sqlite3, "connect"), (HashEmbedder, "embed")):
            guard.setattr(obj, name, forbidden)
        argv = ["compare-policies", *map(str, paths), "-o", str(output)]
        for flags, expected in (([], 10), (["--fail-on-policy-weakening"], 4),
                                (["--fail-on-policy-weakening", "--fail-on-diagnostics"], 3)):
            assert main([*argv, *flags]) == expected
            captured = capsys.readouterr()
            assert not captured.err
            if baseline is None: baseline = captured
            assert captured == baseline
    assert reads == paths * 3
    assert output.read_bytes() == baseline.out.encode("utf-8")
    assert [p.read_bytes() for p in paths] == saved


def test_compare_policies_help_documents_all_exit_codes(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    assert "exit 10 for drift, 2 for invalid input, opt-in 3 or 4 gates" in " ".join(capsys.readouterr().out.split())
