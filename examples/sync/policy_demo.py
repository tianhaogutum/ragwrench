"""Offline collection policy example, also runnable with an installed wheel."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from ragwrench.sync.core import build_snapshot, make_plan, write_json
from ragwrench.sync.store import local_client


def snapshot(*texts):
    return build_snapshot([{"doc_id": "guide", "text": text} for text in texts])


def main():
    old, desired = snapshot("a", "b", "c", "d"), snapshot("a")
    with TemporaryDirectory(prefix="chunksync-policy-") as tmp:
        root = Path(tmp)
        db, override_file = root / "db", root / "policy-override.json"
        for name, value in (("bootstrap", make_plan(snapshot(), old)),
                            ("candidate", make_plan(old, desired)), ("desired", desired)):
            write_json(root / f"{name}.json", value)
        executable = str(Path(sys.executable).with_name("ragwrench"))

        def cli(*args, expected=0):
            argv = [executable, "sync", *map(str, args)]
            result = subprocess.run(argv, capture_output=True, text=True)
            print(json.dumps({"command": argv, "exitCode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}), flush=True)
            assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)
            return json.loads(result.stdout) if result.stdout.strip() else result.stderr

        def inspect_unchanged(collection="kb", expected=0):
            before = {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()}
            result = cli("inspect-policy", "--qdrant-path", db, "--collection", collection, expected=expected)
            assert {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()} == before
            return result

        bootstrap = cli("apply", root / "bootstrap.json", "--qdrant-path", db, "--max-deletion-fraction", "0.5")
        assert bootstrap["review_policy"] == {"max_deletion_fraction": 0.5}
        inspected = inspect_unchanged()
        assert inspected == {"kind": "chunksync.policy_inspection", "version": 1, "collection": "kb",
                             "max_deletion_fraction": 0.5, "policy_source": "explicit",
                             "pending_plan_fingerprint": None, "policy_override": None}
        candidate = root / "candidate.json"
        refused = cli("apply", candidate, "--qdrant-path", db, expected=2)
        assert "collection review policy" in refused
        # The plan is allowed at 0.75, but changing collection policy requires its own record.
        refused = cli("apply", candidate, "--qdrant-path", db, "--max-deletion-fraction", "0.75", expected=2)
        assert "policy override" in refused
        override = cli("policy-override", candidate, "--collection", "kb", "--from-max-deletion-fraction", "0.5",
                       "--max-deletion-fraction", "0.75", "--reviewer", "policy-owner",
                       "--reason", "DOC-42 approves a higher ongoing limit", "-o", override_file)
        args = ("apply", candidate, "--qdrant-path", db, "--max-deletion-fraction", "0.75", "--policy-override", override_file)
        applied = cli(*args, "--verify")
        assert applied["review_policy"] == {"max_deletion_fraction": 0.75}
        assert applied["policy_override"] == override and applied["points"] == 1
        inspected = inspect_unchanged()
        assert inspected["max_deletion_fraction"] == 0.75 and inspected["policy_override"] == override
        assert inspected["pending_plan_fingerprint"] is None
        assert "does not exist" in inspect_unchanged("missing", expected=2)
        with local_client(str(db)) as client:
            state = client.get_collection("kb").config.metadata["chunksync"]
            assert state["review_policy"] == applied["review_policy"]
            assert state["last_policy_override"] == override and state["pending"] is None
        verified = cli("verify", root / "desired.json", "--qdrant-path", db)
        replay = cli(*args)
        assert replay["status"] == "already_applied" and replay["policy_override"] == override
        print(json.dumps({"initial_policy": bootstrap["review_policy"], "refused_apply_exit": 2,
                          "persisted_policy": applied["review_policy"], "policy_override": override["provenance"],
                          "policy_override_fingerprint": override["fingerprint"],
                          "inspection_policy_source": inspected["policy_source"],
                          "inspection_preserved_store_bytes": True, "missing_inspection_exit": 2,
                          "final_verified_points": verified["points"], "replay_status": replay["status"]}))


if __name__ == "__main__":
    main()
