"""Offline Markdown -> snapshot -> plan -> local apply -> from-scratch verify."""
from pathlib import Path
from tempfile import TemporaryDirectory

from ragwrench.sync.core import build_snapshot, make_plan, snapshot_dir
from ragwrench.sync.embedders import HashEmbedder
from ragwrench.sync.store import apply_plan, local_client, verify_snapshot

root = Path(__file__).resolve().parents[2]
old = snapshot_dir(root / "tests/sync/fixtures/docs_v1")
new = snapshot_dir(root / "tests/sync/fixtures/docs_v2")
plan = make_plan(old, new)
with TemporaryDirectory(prefix="chunksync-demo-", dir=root) as temp:
    with local_client(str(Path(temp) / "qdrant")) as client:
        embedder = HashEmbedder()
        print("Initial:", apply_plan(client, "kb", make_plan(build_snapshot([]), old), embedder))
        print("Plan:", plan["summary"])
        print("Update:", apply_plan(client, "kb", plan, embedder))
        print("Verify:", verify_snapshot(client, "kb", new, embedder))
