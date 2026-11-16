"""Tear Qdrant metadata, recover once-only writes, then verify via installed CLI.

Run with the Python from the environment containing the ragwrench console script.
All input and failures are synthetic; temporary data is removed on exit.
"""
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qdrant_client.local import qdrant_local

from ragwrench.sync.core import SyncError, build_snapshot, make_plan, write_json
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.store import apply_plan, local_client


def main(failure="torn_metadata"):
    old = build_snapshot([{"doc_id": "a", "text": t} for t in ("old1", "old2")])
    new = build_snapshot([{"doc_id": "a", "text": t} for t in ("new1", "new2")])
    plan = make_plan(old, new)
    successful = Counter()
    with TemporaryDirectory(prefix="chunksync-recovery-") as tmp:
        root = Path(tmp)
        db = root / "db"
        for name, value in (("bootstrap", make_plan(build_snapshot([]), old)), ("plan", plan), ("new", new)):
            write_json(root / f"{name}.json", value)
        executable = str(Path(sys.executable).with_name("ragwrench"))

        def cli(*args):
            command = [executable, "sync", *map(str, args), "--qdrant-path", str(db)]
            result = subprocess.run(command, capture_output=True, text=True)
            print(json.dumps({"command": command, "exitCode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}))
            result.check_returncode()
            return json.loads(result.stdout)

        def record(client):
            for name in ("upsert", "delete"):
                original = getattr(client, name)

                def write(*args, _name=name, _original=original, **kwargs):
                    result = _original(*args, **kwargs)
                    ids = ([str(p.id) for p in kwargs["points"]] if _name == "upsert"
                           else [str(pid) for pid in kwargs["points_selector"].points])
                    successful.update((_name, pid) for pid in ids)
                    return result

                setattr(client, name, write)

        cli("apply", root / "bootstrap.json")
        with local_client(str(db)) as client:
            record(client)

            @contextmanager
            def torn_open(path, mode="r", *args, **kwargs):
                with open(path, mode, *args, **kwargs) as out:
                    pending = client.get_collection("kb").config.metadata["chunksync"]["pending"]
                    # qdrant-client 1.16 writes meta.json in place; 1.20+ writes meta.json.tmp.
                    if (Path(path) in (db / "meta.json", db / "meta.json.tmp") and mode == "w"
                            and isinstance(pending, dict)
                            and pending["next_operation"] == 2 and pending["in_flight"] is None):
                        out.write('{"collections":')
                        out.flush()
                        raise OSError("synthetic torn checkpoint metadata")
                    yield out

            original_fsync = os.fsync

            def fail_directory_sync(fd):
                # Perform the real sync, then lose its acknowledgement. The
                # complete sidecar is already published; primary stays old.
                result = original_fsync(fd)
                pending = client.get_collection("kb").config.metadata["chunksync"]["pending"]
                if (stat.S_ISDIR(os.fstat(fd).st_mode) and isinstance(pending, dict)
                        and pending["next_operation"] == 2 and pending["in_flight"] is None):
                    raise OSError("synthetic directory fsync acknowledgement failure")
                return result

            injection = (patch.object(qdrant_local, "open", torn_open, create=True)
                         if failure == "torn_metadata" else patch.object(os, "fsync", fail_directory_sync))
            with injection:
                try:
                    apply_plan(client, "kb", plan, HashEmbedder())
                except OSError as exc:
                    assert "synthetic" in str(exc)
                else:
                    raise AssertionError("expected injected metadata failure")
            partial = client.count("kb").count
            assert partial == 4
            # Removing the fault does not make live metadata authoritative.
            # The fence must precede embeddings and all management/point writes.
            retry_embedder = HashEmbedder()

            def forbidden(*args, **kwargs):
                raise AssertionError("mutation on failed live client")

            with patch.multiple(client, **{name: forbidden for name in
                                         ("create_collection", "update_collection", "upsert",
                                          "overwrite_payload", "delete")}):
                try:
                    apply_plan(client, "kb", plan, retry_embedder)
                except SyncError as exc:
                    assert "close and reopen the client" in str(exc)
                    retry_error = str(exc)
                else:
                    raise AssertionError("failed live client was not fenced")
            assert retry_embedder.calls == 0
        if failure == "torn_metadata":
            if (db / "meta.json.tmp").exists():
                # Atomic upstream save: the tear stayed in the unsynced temp
                # file. Model a crash after its rename, leaving meta.json torn.
                (db / "meta.json").write_text('{"collections":')
            assert (db / "meta.json").read_text() == '{"collections":'
        else:
            # Directory sync failure must prevent the primary writer entirely.
            pending = json.loads((db / "meta.json").read_text())["collections"]["kb"]["metadata"]["chunksync"]["pending"]
            assert pending["next_operation"] == 1 and pending["in_flight"] is not None
        with local_client(str(db)) as client:
            record(client)
            resumed = apply_plan(client, "kb", plan, HashEmbedder())
        expected = Counter({("upsert", c["id"]): 1 for c in plan["upsert"]})
        expected.update({("delete", c["id"]): 1 for c in plan["delete"]})
        assert successful == expected and resumed["embedded"] == 0 and resumed["points"] == 2
        # Exercise fallback through the console script too, from committed state.
        (db / "meta.json").write_text('{"collections":')
        verification = cli("verify", root / "new.json")
        assert cli("apply", root / "plan.json")["status"] == "already_applied"
        print(json.dumps({"failure": failure, "points_after_failure": partial,
                          "same_client_retry_error": retry_error, "retry_embeddings": retry_embedder.calls,
                          "resume": resumed, "successful_mutations_per_id": sorted(successful.values()),
                          "installed_cli_recovery_verification": verification}))


if __name__ == "__main__":
    for failure in ("torn_metadata", "directory_fsync_acknowledgement"):
        main(failure)
