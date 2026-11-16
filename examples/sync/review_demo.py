"""Offline synthetic comparison/review example, including the installed CLI."""
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
    old, baseline, desired = snapshot("a", "b", "c", "d"), snapshot("a", "b", "c"), snapshot("a")
    with TemporaryDirectory(prefix="chunksync-review-") as tmp:
        root = Path(tmp)
        db, receipt_path = root / "db", root / "review.json"
        for name, value in (("bootstrap", make_plan(snapshot(), old)), ("baseline", make_plan(old, baseline)),
                            ("candidate", make_plan(old, desired)), ("desired", desired)):
            write_json(root / f"{name}.json", value)
        executable = str(Path(sys.executable).with_name("ragwrench"))

        def cli(*args, expected=0):
            result = subprocess.run([executable, "sync", *map(str, args)], capture_output=True, text=True)
            assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)
            return json.loads(result.stdout) if result.stdout.strip() else result.stderr

        candidate = root / "candidate.json"
        comparison = cli("compare", root / "baseline.json", candidate, expected=10)
        assert comparison["summary_delta"]["delete"] == 2
        blocked = cli("review", candidate, "--max-deletion-fraction", "0.5", "-o", receipt_path, expected=3)
        assert blocked["observed"]["deletion_fraction"] == 0.75
        refusal = cli("apply", candidate, "--review", receipt_path, "--qdrant-path", db, expected=2)
        assert "review blocked" in refusal and not db.exists()
        approved = cli("review", candidate, "--max-deletion-fraction", "0.5", "--override-gate", "max_deletion_fraction",
                       "--reviewer", "example-operator", "--reason", "retire three obsolete chunks", "-o", receipt_path)
        cli("apply", root / "bootstrap.json", "--qdrant-path", db)
        applied = cli("apply", candidate, "--review", receipt_path, "--qdrant-path", db, "--verify")
        assert applied["review"] == approved and applied["points"] == 1 and applied["embedded"] == 0
        with local_client(str(db)) as client:
            state = client.get_collection("kb").config.metadata["chunksync"]
            assert state["last_review"] == approved and state["pending"] is None
        verified = cli("verify", root / "desired.json", "--qdrant-path", db)
        replay = cli("apply", candidate, "--review", receipt_path, "--qdrant-path", db)
        assert replay["status"] == "already_applied" and replay["review"] == approved
        print(json.dumps({"extra_deletions_vs_baseline": comparison["summary_delta"]["delete"],
                          "deletion_fraction": blocked["observed"]["deletion_fraction"],
                          "blocked_review_exit": 3, "blocked_apply_exit": 2,
                          "override": approved["provenance"], "persisted_review_fingerprint": approved["fingerprint"],
                          "final_verified_points": verified["points"], "replay_status": replay["status"]}))


if __name__ == "__main__":
    main()
