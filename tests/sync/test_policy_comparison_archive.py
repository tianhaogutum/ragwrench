"""Atomic comparison archives must never replace either input inventory."""
import json
import os
from pathlib import Path

import pytest

from ragwrench.sync.cli import main
from .test_policy_comparison import bad, ok, report


def test_output_option_archives_equal_comparison(tmp_path, capsys):
    before, after, output = [tmp_path / name for name in ("before.json", "after.json", "diff.json")]
    for path in (before, after):
        path.write_text(json.dumps(report()), encoding="utf-8")
    inputs = [path.read_bytes() for path in (before, after)]
    assert main(["compare-policies", str(before), str(after), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert not captured.err and json.loads(captured.out)["identical"]
    assert output.read_bytes() == captured.out.encode("utf-8")
    assert inputs == [path.read_bytes() for path in (before, after)]


@pytest.fixture
def inputs(tmp_path):
    paths = [tmp_path / name for name in ("before.json", "after.json")]
    for path in paths:
        path.write_text(json.dumps(report([ok()])), encoding="utf-8")
    return paths


@pytest.fixture(params=["default", "diagnostics", "weakening", "both"])
def diagnostic_flag(request, inputs):
    flags = []
    if request.param in ("diagnostics", "both"):
        flags.append("--fail-on-diagnostics")
    if request.param in ("weakening", "both"):
        flags.append("--fail-on-policy-weakening")
    # Neither a known diagnostic nor known weakening may mask a global failure.
    for index, path in enumerate(inputs):
        rows = [bad("first")] if "--fail-on-diagnostics" in flags else []
        rows.append(ok("last", limit=0.25 if index == 0 and "--fail-on-policy-weakening" in flags else 0.5))
        path.write_text(json.dumps(report(rows)))
    return flags


def invoke(inputs, output, capsys, expected, flags=()):
    saved = [p.read_bytes() if p.is_file() else None for p in inputs]
    previous = output.read_bytes() if isinstance(output, Path) and output.is_file() else None
    code = main(["compare-policies", *map(str, inputs), "--output", str(output), *flags])
    captured = capsys.readouterr()
    assert code == expected
    assert saved == [p.read_bytes() if p.is_file() else None for p in inputs]
    if expected == 2:
        assert not captured.out and captured.err.startswith("ragwrench sync: ")
        assert "Traceback" not in captured.err
        if isinstance(output, Path) and output.is_file():
            assert output.read_bytes() == previous
    else:
        assert not captured.err
        assert output.read_bytes() == captured.out.encode("utf-8")
        assert captured.out.endswith("\n") and not captured.out.endswith("\n\n")
    return captured


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("case", ["equal", "source-only", "drift", "mixed", "unicode"])
def test_exact_archive_stdout_and_no_backend_reads(inputs, tmp_path, capsys, monkeypatch, existing, case):
    import sqlite3
    import ragwrench.sync.cli as cli
    import ragwrench.sync.inventory as inventory
    import ragwrench.sync.store as store
    from ragwrench.sync.policy_comparison import compare_policies, comparison_json
    before, after = report([ok()]), report([ok()])
    if case == "source-only":
        after["catalog_source"] = "recovery"
    elif case == "drift":
        after = report([ok(limit=0.25)])
    elif case == "mixed":
        before = report([bad("z", "unavailable"), ok("keep"), ok("d"), bad("a", "unmanaged"), ok("q")])
        after = report([ok("z"), ok("keep"), bad("d"), ok("c"), bad("b", "unsupported")], "recovery")
    elif case == "unicode":
        before, after = report([bad(message="vorher Überprüfung")]), report([bad(message="nachher 合成")])
    for path, value in zip(inputs, (before, after)):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "diff.json"
    if existing:
        output.write_bytes(b"previous archive\n")
    saved = [p.read_bytes() for p in inputs]
    real_open, reads = Path.open, []
    def only_inputs(path, mode="r", *a, **kw):
        assert path in inputs and mode == "rb"
        reads.append(path)
        return real_open(path, mode, *a, **kw)
    def forbidden(*a, **kw):
        pytest.fail("comparison archive must not read stores/backends or embed")
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", only_inputs)
        for obj, name in ((cli, "local_client"), (cli, "list_policies"), (cli, "make_embedder"),
                          (inventory, "load_json"), (store, "QdrantClient"), (sqlite3, "connect")):
            guard.setattr(obj, name, forbidden)
        code = main(["compare-policies", *map(str, inputs), "-o", str(output)])
    captured = capsys.readouterr()
    assert code == (0 if case in ("equal", "source-only") else 10)
    assert not captured.err and reads == inputs
    assert output.read_bytes() == captured.out.encode("utf-8")
    assert captured.out == comparison_json(compare_policies(before, after))
    assert [p.read_bytes() for p in inputs] == saved
    if case == "mixed":
        result = json.loads(captured.out)
        assert [r["collection"] for r in result["added"]] == ["b", "c"]
        assert [r["collection"] for r in result["removed"]] == ["a", "q"]
        assert [r["collection"] for r in result["changed"]] == ["d", "z"]


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("case", ["direct", "dotdot", "relative", "hardlink", "input-symlink", "input-ancestor"])
def test_refuse_alias_of_either_input_before_read(inputs, tmp_path, capsys, monkeypatch, side, case, diagnostic_flag):
    import ragwrench.sync.cli as cli
    source = inputs[side]
    output = source
    if case == "dotdot":
        (tmp_path / "sub").mkdir()
        output = tmp_path / "sub" / ".." / source.name
    elif case == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path(source.name)
    elif case == "hardlink":
        output = tmp_path / "alias.json"
        os.link(source, output)
    elif case == "input-symlink":
        alias = tmp_path / "input-alias.json"
        alias.symlink_to(source)
        inputs[side] = alias
    elif case == "input-ancestor":
        alias = tmp_path / "input-dir"
        alias.symlink_to(tmp_path, target_is_directory=True)
        inputs[side] = alias / source.name
    def forbidden(*a, **kw):
        pytest.fail("destination validation must precede input reads")
    monkeypatch.setattr(cli, "load_inventory", forbidden)
    saved = source.read_bytes()
    assert "distinct from both" in invoke(inputs, output, capsys, 2, diagnostic_flag).err
    assert source.read_bytes() == saved


@pytest.mark.parametrize("case", ["file", "input", "dangling", "directory", "ancestor-dotdot"])
def test_refuse_destination_symlink_traversal(inputs, tmp_path, capsys, monkeypatch, case, diagnostic_flag):
    import ragwrench.sync.cli as cli
    old = tmp_path / "old.json"
    old.write_bytes(b"previous archive")
    link = tmp_path / "link"
    if case in ("file", "input", "dangling"):
        link.symlink_to(old if case == "file" else inputs[0] if case == "input" else tmp_path / "missing")
        output = link
    else:
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "diff.json" if case == "directory" else link / ".." / tmp_path.name / "old.json"
    def forbidden(*a, **kw):
        pytest.fail("unsafe destination must refuse before reads")
    monkeypatch.setattr(cli, "load_inventory", forbidden)
    assert "symlinks" in invoke(inputs, output, capsys, 2, diagnostic_flag).err
    assert old.read_bytes() == b"previous archive" and link.is_symlink()


@pytest.mark.parametrize("case", ["missing-parent", "file-parent", "directory", "fifo", "null", "empty"])
def test_invalid_destination_no_mkdir(inputs, tmp_path, capsys, monkeypatch, case, diagnostic_flag):
    import ragwrench.sync.cli as cli
    output = tmp_path / "diff.json"
    if case == "missing-parent": output = tmp_path / "absent" / "diff.json"
    elif case == "file-parent":
        output.write_bytes(b"old")
        output = output / "diff.json"
    elif case == "directory": output.mkdir()
    elif case == "fifo": os.mkfifo(output)
    elif case == "null": output = str(output) + "\0"
    else: output = ""
    def forbidden(*a, **kw): pytest.fail("invalid output must refuse before reads")
    monkeypatch.setattr(cli, "load_inventory", forbidden)
    previous = set(tmp_path.iterdir())
    invoke(inputs, output, capsys, 2, diagnostic_flag)
    assert set(tmp_path.iterdir()) == previous


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("case", ["malformed", "utf8", "schema-last", "duplicate", "count", "bytes", "missing", "directory", "unreadable"])
def test_input_failures_preserve_previous_diff(inputs, tmp_path, capsys, monkeypatch, existing, side, case, diagnostic_flag):
    import ragwrench.sync.inventory as inventory
    import ragwrench.sync.policy_comparison as comparison
    source = inputs[side]
    if case == "malformed": source.write_bytes(b'{')
    elif case == "utf8": source.write_bytes(b'\xff')
    elif case == "schema-last":
        value = json.loads(source.read_text())
        value["collections"].append({})
        source.write_text(json.dumps(value))
    elif case == "duplicate": source.write_bytes(b'{"version":1,"version":1}')
    elif case == "count": source.write_text(json.dumps(report([ok(f"c{i}") for i in range(1001)])))
    elif case == "bytes":
        monkeypatch.setattr(comparison, "MAX_JSON_BYTES", 2048)
        source.write_bytes(b' ' * 2049)
    elif case == "missing": source.unlink()
    elif case == "directory": source.unlink(); source.mkdir()
    else:
        real_open = Path.open
        def denied(path, *a, **kw):
            if path == source: raise PermissionError("synthetic unreadable inventory")
            return real_open(path, *a, **kw)
        # Refusal during main only; preservation snapshots remain readable.
    output = tmp_path / "diff.json"
    if existing: output.write_bytes(b"previous diff")
    saved = [p.read_bytes() if p.is_file() else None for p in inputs]
    previous = output.read_bytes() if existing else None
    def forbidden(*a, **kw): pytest.fail("invalid input must refuse before temp creation")
    with monkeypatch.context() as guard:
        guard.setattr(inventory.tempfile, "NamedTemporaryFile", forbidden)
        if case == "unreadable": guard.setattr(Path, "open", denied)
        assert main(["compare-policies", *map(str, inputs), "--output", str(output), *diagnostic_flag]) == 2
    captured = capsys.readouterr()
    assert not captured.out and captured.err.startswith("ragwrench sync: ") and "Traceback" not in captured.err
    assert saved == [p.read_bytes() if p.is_file() else None for p in inputs]
    assert output.read_bytes() == previous if existing else not output.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_result_byte_limit_preserves_diff(inputs, tmp_path, capsys, monkeypatch, existing, diagnostic_flag):
    import ragwrench.sync.inventory as inventory
    import ragwrench.sync.policy_comparison as comparison
    values = [report([ok("rise", limit=limit), bad(message=letter * 800)])
              for limit, letter in ((0.25, "a"), (0.5, "b"))]
    for path, value in zip(inputs, values): path.write_text(json.dumps(value))
    monkeypatch.setattr(comparison, "MAX_JSON_BYTES", max(len(comparison.comparison_json(v).encode()) for v in values))
    output = tmp_path / "diff.json"
    if existing: output.write_bytes(b"previous diff")
    def forbidden(*a, **kw): pytest.fail("oversized output must refuse before temp creation")
    monkeypatch.setattr(inventory.tempfile, "NamedTemporaryFile", forbidden)
    invoke(inputs, output, capsys, 2, diagnostic_flag)
    assert output.exists() == existing


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("drift", [False, True])
@pytest.mark.parametrize("phase", ["create", "write", "flush", "fsync", "close", "replace"])
def test_publication_faults_preserve_archive_and_inputs(inputs, tmp_path, capsys, monkeypatch, existing, drift, phase, diagnostic_flag):
    import ragwrench.sync.inventory as inventory
    if drift:
        value = json.loads(inputs[1].read_text())
        value["collections"].append(bad("new-diagnostic"))
        inputs[1].write_text(json.dumps(value))
    output = tmp_path / "diff.json"
    if existing: output.write_bytes(b"previous diff")
    real_temp, real_sync = inventory.tempfile.NamedTemporaryFile, os.fsync
    def fail(): raise OSError("synthetic comparison " + phase)
    class FaultyFile:
        def __init__(self, stream): self.stream, self.name = stream, stream.name
        def __enter__(self): return self
        def __exit__(self, *args):
            self.stream.__exit__(*args)
            if phase == "close": fail()
        def write(self, data):
            if phase == "write":
                self.stream.write(data[:7])
                fail()
            return self.stream.write(data)
        def flush(self):
            self.stream.flush()
            if phase == "flush": fail()
        def fileno(self): return self.stream.fileno()
    def create(*a, **kw):
        if phase == "create": fail()
        return FaultyFile(real_temp(*a, **kw))
    def sync(fd):
        real_sync(fd)
        if phase == "fsync": fail()
    def replace(src, dst):
        assert Path(src).parent == output.parent and Path(dst) == output
        assert json.loads(Path(src).read_bytes())["kind"] == "chunksync.policy_comparison"
        assert output.read_bytes() == b"previous diff" if existing else not output.exists()
        fail()
    monkeypatch.setattr(inventory.tempfile, "NamedTemporaryFile", create)
    monkeypatch.setattr(inventory.os, "fsync", sync)
    if phase == "replace": monkeypatch.setattr(inventory.os, "replace", replace)
    assert "synthetic comparison " + phase in invoke(inputs, output, capsys, 2, diagnostic_flag).err
    assert output.exists() == existing
    assert not list(tmp_path.glob(".chunksync-*.tmp"))


def test_publish_only_after_full_serialization_sync_and_close(inputs, tmp_path, capsys, monkeypatch, diagnostic_flag):
    import ragwrench.sync.cli as cli
    import ragwrench.sync.inventory as inventory
    output = tmp_path / "diff.json"
    output.write_bytes(b"previous diff")
    real_compare, real_json = cli.compare_policies, cli.comparison_json
    real_temp, real_sync, real_replace = inventory.tempfile.NamedTemporaryFile, os.fsync, os.replace
    events, streams = [], []
    def compare(*a):
        result = real_compare(*a)
        events.append("validated")
        return result
    def encode(value):
        result = real_json(value)
        events.append("serialized")
        return result
    def create(*a, **kw):
        assert events == ["validated", "serialized"]
        stream = real_temp(*a, **kw)
        streams.append(stream)
        return stream
    def sync(fd):
        payload = Path(streams[0].name).read_bytes()
        assert json.loads(payload)["identical"] == ("--fail-on-policy-weakening" not in diagnostic_flag)
        real_sync(fd)
        events.append("sync")
    def replace(src, dst):
        assert events == ["validated", "serialized", "sync"] and streams[0].closed
        assert Path(src).parent == output.parent and Path(dst) == output
        assert output.read_bytes() == b"previous diff"
        assert not capsys.readouterr().out  # stdout must follow publication
        real_replace(src, dst)
        events.append("replace")
    monkeypatch.setattr(cli, "compare_policies", compare)
    monkeypatch.setattr(cli, "comparison_json", encode)
    monkeypatch.setattr(inventory.tempfile, "NamedTemporaryFile", create)
    monkeypatch.setattr(inventory.os, "fsync", sync)
    monkeypatch.setattr(inventory.os, "replace", replace)
    expected = (3 if "--fail-on-diagnostics" in diagnostic_flag else
                4 if "--fail-on-policy-weakening" in diagnostic_flag else 0)
    invoke(inputs, output, capsys, expected, diagnostic_flag)
    assert events == ["validated", "serialized", "sync", "replace"]
    assert not list(tmp_path.glob(".chunksync-*.tmp"))
