"""Offline failure/restart demonstration using real stores and the installed CLI.

The injected error occurs after the second backend write has succeeded. Progress
still acknowledges only batch one, so the fresh CLI process must replay batch two.
No production failure-injection flag is added to ragwrench migrate.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import ragwrench.migrate as vecparity
from ragwrench.migrate.cli import main as cli_main
from ragwrench.migrate.stores import Store


def fingerprint(path: Path) -> dict:
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/migrate/resume_failure.py NEW_OUTPUT_DIR")
    root = Path(sys.argv[1]).expanduser().absolute()
    fixture_script = Path(__file__).with_name("make_fixture.py")
    subprocess.run([sys.executable, str(fixture_script), str(root)], check=True)
    cli = Path(sys.executable).parent / "ragwrench"
    results = {"module": vecparity.__file__, "directions": []}

    def run(args):
        proc = subprocess.run([str(cli), "migrate", *args, "--json"], capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"CLI failed with exit {proc.returncode}: {proc.stderr}")
        return json.loads(proc.stdout)

    for source_kind, source_dir, target_kind, target_dir in [
        ("chroma", "chroma", "qdrant", "qdrant"),
        ("qdrant", "qdrant", "chroma", "restored"),
    ]:
        source = f"{source_kind}:{root / source_dir}#docs"
        target = f"{target_kind}:{root / target_dir}#docs"
        checkpoint = root / f"{source_kind}-to-{target_kind}.json"
        before = fingerprint(root / source_dir)
        writes = 0
        original_write = Store.write

        def fail_second_write(store, batch):
            nonlocal writes
            writes += 1
            original_write(store, batch)
            if writes == 2:
                raise OSError("injected one write failure after the backend accepted the batch")

        out, err = io.StringIO(), io.StringIO()
        args = ["copy", source, target, "--batch-size", "2", "--checkpoint", str(checkpoint), "--top-k", "2"]
        with patch.object(Store, "write", fail_second_write), redirect_stdout(out), redirect_stderr(err):
            failure_exit = cli_main(args + ["--json"])
        assert failure_exit == 2 and out.getvalue() == ""
        assert "injected one write failure" in json.loads(err.getvalue())["error"]
        saved = json.loads(checkpoint.read_text())
        assert len(saved["completed_ids"]) == 2 and saved["parity"] is None
        assert fingerprint(root / source_dir) == before

        # A separate installed CLI process, with a different batch size.
        args[args.index("--batch-size") + 1] = "1"
        resumed = run(args + ["--resume"])
        assert resumed["copied"] == 3 and resumed["skipped"] == 2
        assert resumed["parity"]["ok"] and resumed["parity"]["top_k"]["overlap_mean"] == 1.0
        assert resumed["parity"]["mismatch_counts"]["id_mapping"] == 0
        assert json.loads(checkpoint.read_text())["manifest"] == saved["manifest"]
        repeated = run(args + ["--resume"])
        assert repeated["copied"] == 0 and repeated["skipped"] == 5 and repeated["parity"]["ok"]
        assert run(["verify", source, target, "--top-k", "2"])["ok"]
        assert fingerprint(root / source_dir) == before
        results["directions"].append({"source": source, "target": target,
            "failureExit": failure_exit, "acknowledgedAfterFailure": len(saved["completed_ids"]),
            "resume": resumed, "repeatCopied": repeated["copied"], "sourceUnchanged": True})

    # Plain copy and verify without a checkpoint still work with the installed CLI.
    plain = f"qdrant:{root / 'plain'}#docs"
    source = f"chroma:{root / 'chroma'}#docs"
    assert run(["copy", source, plain])["copied"] == 5
    assert run(["verify", source, plain])["ok"]
    assert run(["verify", source, f"chroma:{root / 'restored'}#docs", "--top-k", "2"])["ok"]
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Both directions: injected exit 2; resumed copied=3 skipped=2; PARITY OK; source unchanged. Results: {root / 'results.json'}")


if __name__ == "__main__":
    main()
