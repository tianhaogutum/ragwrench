"""Offline no-repair inspection example; also runs with an installed wheel."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from ragwrench.sync.core import build_snapshot, make_plan, write_json


def main():
    with TemporaryDirectory(prefix="chunksync-inspection-") as tmp:
        root = Path(tmp)
        db = root / "db"
        plan = make_plan(build_snapshot([]), build_snapshot([{"doc_id": "guide", "text": "a"}]))
        write_json(root / "plan.json", plan)
        executable = str(Path(sys.executable).with_name("ragwrench"))

        def cli(*args, expected=0):
            argv = [executable, "sync", *map(str, args)]
            result = subprocess.run(argv, capture_output=True, text=True)
            print(json.dumps({"command": argv, "exitCode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}), flush=True)
            assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)
            return json.loads(result.stdout) if result.stdout.strip() else result.stderr

        cli("apply", root / "plan.json", "--qdrant-path", db, "--max-deletion-fraction", "0.5")
        args = ("inspect-policy", "--qdrant-path", db, "--collection", "kb")
        expected = cli(*args)
        assert expected == {"kind": "chunksync.policy_inspection", "version": 1, "collection": "kb",
                            "max_deletion_fraction": 0.5, "policy_source": "explicit",
                            "pending_plan_fingerprint": None, "policy_override": None}

        def files():
            return {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()}

        def unchanged(expected_exit=0):
            before = files()
            value = cli(*args, "--no-repair", expected=expected_exit)
            assert files() == before
            return value

        assert unchanged() == expected
        primary, sidecar = db / "meta.json", db / "chunksync-recovery.json"
        saved = sidecar.read_bytes()
        for damage in ("missing", "truncated", "invalid-utf8"):
            if damage == "missing":
                primary.unlink()
            else:
                primary.write_bytes(b'{"collections":' if damage == "truncated" else b"\xff")
            assert "primary metadata repair required" in unchanged(2)
            assert cli(*args) == expected
            assert json.loads(primary.read_text()) == json.loads(saved)["metadata"]
            assert sidecar.read_bytes() == saved

        # A broken fallback cannot turn refusal into success or overwrite the primary.
        primary.write_bytes(b'{"collections":')
        sidecar.write_bytes(b"broken sidecar")
        assert "primary metadata repair required" in unchanged(2)
        before = files()
        assert "cannot read JSON" in cli(*args, expected=2)
        assert files() == before
        sidecar.write_bytes(saved)
        assert cli(*args) == expected
        print(json.dumps({"healthy_no_repair_exit": 0, "damaged_no_repair_exit": 2,
                          "damage_cases": ["missing", "truncated", "invalid-utf8"],
                          "no_repair_preserved_files": True, "default_repair_exit": 0,
                          "bad_sidecar_default_exit": 2}))


if __name__ == "__main__":
    main()
