"""Synthetic catalog inventory through the installed ragwrench console script."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from qdrant_client import models

from ragwrench.sync.core import build_snapshot, make_plan
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.review import review_plan
from ragwrench.sync.store import apply_plan, local_client


def main():
    with TemporaryDirectory(prefix="chunksync-inventory-") as tmp:
        db = Path(tmp) / "db"
        plan = make_plan(build_snapshot([]), build_snapshot([{"doc_id": "guide", "text": "synthetic private text"}]))
        with local_client(str(db)) as client:
            for name, limit in (("zeta", 0.75), ("alpha", 0.25)):
                apply_plan(client, name, plan, HashEmbedder(),
                           review=review_plan(plan, max_deletion_fraction=limit))
            client.create_collection("middle", vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT))
        (db / ".lock").unlink()
        executable = str(Path(sys.executable).with_name("ragwrench"))

        archive = Path(tmp) / "report.json"

        def cli(*options, expected=0):
            argv = [executable, "sync", "list-policies", "--qdrant-path", str(db), *options]
            before = {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()}
            output = Path(options[options.index("--output") + 1]) if "--output" in options else None
            previous = output.read_bytes() if output is not None and output.is_file() else None
            result = subprocess.run(argv, capture_output=True, text=True)
            if output is not None:
                if expected in (0, 3):
                    assert output.read_bytes() == result.stdout.encode("utf-8")
                else:
                    assert (output.read_bytes() if output.is_file() else None) == previous
            print(json.dumps({"command": argv, "exitCode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr,
                              "archive_exact_stdout": output is not None and expected in (0, 3),
                              "archive_preserved": output is not None and expected == 2}), flush=True)
            assert result.returncode == expected, result.stderr
            assert {p.relative_to(db): p.read_bytes() for p in db.rglob("*") if p.is_file()} == before
            assert not (db / ".lock").exists()
            if expected in (0, 3):
                assert not result.stderr
                assert "synthetic private text" not in result.stdout
                return json.loads(result.stdout)
            assert not result.stdout
            return result.stderr

        report = cli()
        assert report == cli("--no-repair", "--output", str(archive))
        assert report == cli("--output", str(archive))
        assert report == cli("--fail-on-diagnostics", expected=3)
        assert report == cli("--fail-on-diagnostics", "--no-repair", "--output", str(archive), expected=3)
        assert "outside the Qdrant store" in cli("--output", str(db / "meta.json"), expected=2)
        assert report["kind"] == "chunksync.policy_inventory" and report["version"] == 1
        assert [r["collection"] for r in report["collections"]] == ["alpha", "middle", "zeta"]
        assert [r["management_state"] for r in report["collections"]] == ["managed", "unmanaged", "managed"]
        assert [r["policy"]["max_deletion_fraction"] for r in report["collections"] if r["policy"]] == [0.25, 0.75]
        assert report["collections"][1]["policy"] is None

        primary = db / "meta.json"
        metadata = json.loads(primary.read_text())
        state = metadata["collections"]["alpha"]["metadata"]["chunksync"]
        state.pop("review_policy")
        primary.write_text(json.dumps(metadata))
        legacy = cli("--output", str(archive))
        assert legacy["collections"][0]["policy"]["policy_source"] == "legacy-review"
        state["review_policy"] = {"max_deletion_fraction": True}
        primary.write_text(json.dumps(metadata))
        invalid = cli("--output", str(archive))
        assert invalid["collections"][0]["status"] == "invalid"
        assert invalid["collections"][0]["policy"] is None
        assert invalid["collections"][2] == report["collections"][2]
        assert invalid == cli("--fail-on-diagnostics", "--output", str(archive), expected=3)

        state.clear()
        state["version"] = 2
        primary.write_text(json.dumps(metadata))
        unsupported = cli("--fail-on-diagnostics", "--output", str(archive), expected=3)
        assert unsupported["collections"][0]["status"] == "unsupported"
        metadata["collections"]["alpha"] = []
        primary.write_text(json.dumps(metadata))
        unavailable = cli("--fail-on-diagnostics", "--output", str(archive), expected=3)
        assert unavailable["collections"][0]["status"] == "unavailable"

        primary.write_text('{"collections":')
        assert "primary metadata repair required" in cli("--no-repair", "--output", str(archive), expected=2)
        assert cli("--output", str(archive)) == {**report, "catalog_source": "recovery"}
        assert cli("--fail-on-diagnostics", "--output", str(archive), expected=3) == {**report, "catalog_source": "recovery"}
        cli("--fail-on-diagnostics", "--no-repair", "--output", str(archive), expected=2)
        assert primary.read_text() == '{"collections":'  # no disk repair
        primary.write_text("[]")
        assert "unsupported Qdrant metadata" in cli("--output", str(archive), expected=2)
        cli("--fail-on-diagnostics", "--output", str(archive), expected=2)

        healthy = json.loads((db / "chunksync-recovery.json").read_text())["metadata"]
        del healthy["collections"]["middle"]
        primary.write_text(json.dumps(healthy))
        assert all(r["status"] == "ok" for r in cli("--fail-on-diagnostics", "--output", str(archive))["collections"])
        healthy["collections"] = {}
        primary.write_text(json.dumps(healthy))
        assert cli("--fail-on-diagnostics", "--no-repair", "--output", str(archive))["collections"] == []
        print(json.dumps({"collections": 3, "managed": 2, "unmanaged": 1,
                          "invalid_row_isolated": True, "legacy_not_adopted": True,
                          "recovery": "in-memory catalog only", "store_files_unchanged": True,
                          "archive": "exact stdout; atomic replacement; refusals preserve previous bytes"}))


if __name__ == "__main__":
    main()
