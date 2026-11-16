import json
from pathlib import Path
import subprocess
import sys

from ragwrench.sync.bench import run_bench
from ragwrench.sync.core import build_snapshot, make_plan, write_json
from .conftest import linux_durable


def cli(*args):
    return subprocess.run([sys.executable, "-m", "ragwrench.sync.cli", *map(str, args)], capture_output=True, text=True)


@linux_durable
def test_cli_e2e(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    empty, old, new, bootstrap, plan = [tmp_path / f"{x}.json" for x in ("empty", "old", "new", "bootstrap", "plan")]
    db = tmp_path / "qdrant"
    assert cli("snapshot", "--empty", "-o", empty).returncode == 0
    for directory, out in (("docs_v1", old), ("docs_v2", new)):
        assert cli("snapshot", "--dir", fixtures / directory, "-o", out).returncode == 0
    assert cli("plan", empty, old, "-o", bootstrap).returncode == 10
    assert cli("apply", bootstrap, "--qdrant-path", db).returncode == 0
    changed = cli("plan", old, new, "-o", plan, "--format", "json")
    assert changed.returncode == 10
    assert json.loads(changed.stdout)["summary"]["upsert"] == 1
    result = cli("apply", plan, "--qdrant-path", db, "--verify")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["embedded"] == 1
    verified = cli("verify", new, "--qdrant-path", db)
    assert verified.returncode == 0, verified.stderr
    assert cli("plan", new, new).returncode == 0
    assert cli("verify", new, "--qdrant-path", ":memory:").returncode == 2


def test_dry_run_no_db_or_external_execution(tmp_path):
    empty = build_snapshot([])
    new = build_snapshot([{"doc_id": "a", "text": "test"}])
    plan = tmp_path / "plan.json"
    write_json(plan, make_plan(empty, new))
    side_effect = tmp_path / "must-not-exist"
    command = [sys.executable, "-c", f"open({str(side_effect)!r}, 'w').write('bad')"]
    db = tmp_path / "not-created"
    result = cli("apply", plan, "--qdrant-path", db, "--dry-run", "--embedder", "command", "--embed-command", json.dumps(command))
    assert result.returncode == 0
    assert json.loads(result.stdout)["collection_checked"] is False
    assert not side_effect.exists()
    assert not db.exists()


def test_bad_cli_inputs(tmp_path):
    corrupt = tmp_path / "bad.json"
    corrupt.write_text('{"kind":"chunksync.plan","version":0}')
    result = cli("apply", corrupt, "--qdrant-path", tmp_path / "db")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "db").exists()
    result = cli("plan", corrupt, corrupt)
    assert result.returncode == 2


def test_bench_determinism_and_boundary_comparison(tmp_path):
    a, b = run_bench(), run_bench()
    assert a == b
    rows = {r["scenario"]: r["strategies"] for r in a["results"]}
    insert = rows["insert_at_top"]
    assert 0 < insert["chunk_cdc"]["embedded"] < insert["chunk_fixed"]["embedded"]
    assert insert["document_hash"]["embedded"] > insert["chunk_cdc"]["embedded"]
    assert rows["rename"]["chunk_cdc"]["embedded"] == 0
    assert rows["delete"]["chunk_cdc"]["deleted"] > 0
    output = tmp_path / "bench.json"
    result = cli("bench", "-o", output)
    assert result.returncode == 0
    assert json.loads(output.read_text()) == a
