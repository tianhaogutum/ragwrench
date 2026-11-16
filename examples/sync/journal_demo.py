"""Synthetic persistent-Qdrant failure/resume; also works with an installed wheel.

Run with the Python from the environment containing the ragwrench console script.
No server, credentials, or external embedding process is used.
"""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from ragwrench.sync.core import build_snapshot, make_plan, write_json
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.store import apply_plan, local_client


def main():
    old = build_snapshot([{"doc_id": "a", "text": "old1"}, {"doc_id": "a", "text": "old2"}])
    new = build_snapshot([{"doc_id": "a", "text": "new1"}, {"doc_id": "a", "text": "new2"}])
    plan = make_plan(old, new)
    successful = Counter()
    with TemporaryDirectory(prefix="chunksync-journal-") as tmp:
        root = Path(tmp)
        db = str(root / "db")
        for name, value in (("bootstrap", make_plan(build_snapshot([]), old)), ("plan", plan), ("new", new)):
            write_json(root / f"{name}.json", value)
        executable = str(Path(sys.executable).with_name("ragwrench"))

        def cli(*args):
            result = subprocess.run([executable, "sync", *map(str, args), "--qdrant-path", db],
                                    capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        cli("apply", root / "bootstrap.json")
        with local_client(db) as client:
            original = client.delete

            def fail_after_delete(*args, **kwargs):
                result = original(*args, **kwargs)
                successful.update(str(pid) for pid in kwargs["points_selector"].points)
                raise RuntimeError("synthetic failure after first persisted delete")

            client.delete = fail_after_delete
            try:
                apply_plan(client, "kb", plan, HashEmbedder())
            except RuntimeError as exc:
                assert "synthetic failure" in str(exc)
            else:
                raise AssertionError("expected injected vector-store failure")
            partial = client.count("kb").count
            journal = client.get_collection("kb").config.metadata["chunksync"]["pending"]
            assert partial == 3 and journal["next_operation"] == 2
        with local_client(db) as client:
            original = client.delete

            def record_delete(*args, **kwargs):
                result = original(*args, **kwargs)
                successful.update(str(pid) for pid in kwargs["points_selector"].points)
                return result

            client.delete = record_delete
            resumed = apply_plan(client, "kb", plan, HashEmbedder())
        assert successful == Counter({c["id"]: 1 for c in plan["delete"]})
        assert resumed["embedded"] == 0 and resumed["points"] == 2
        verification = cli("verify", root / "new.json")
        assert cli("apply", root / "plan.json")["status"] == "already_applied"
        print(json.dumps({"failure": "after first persisted delete", "points_after_failure": partial,
                          "resume": resumed, "successful_deletes_per_id": sorted(successful.values()),
                          "installed_cli_verification": verification}))


if __name__ == "__main__":
    main()
