"""Run a real CLI subprocess round trip through two loopback HTTP APIs, offline."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from ragwrench.port.fakes import FakeState, running_fake

ROOT = Path(__file__).resolve().parents[2]


def main():
    source, target = FakeState("dify"), FakeState("ragflow")
    dataset = source.seed(ROOT / "examples/port/bundle")
    env = {**os.environ, "DIFY_KB_API_KEY": "fake-key", "RAGFLOW_API_KEY": "fake-key"}
    def cli(*args):
        result = subprocess.run([sys.executable, "-m", "ragwrench.port", *map(str, args)], env=env,
                                text=True, capture_output=True, check=True)
        print(result.stdout.strip())
        return json.loads(result.stdout)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="demo-", dir=ROOT / ".cache") as tmp, running_fake(source) as source_url, running_fake(target) as target_url:
        bundle, state = Path(tmp) / "bundle", Path(tmp) / "mapping.json"
        cli("export", "dify", "--base-url", source_url, "--dataset-id", dataset, "--page-size", "2", "-o", bundle)
        cli("validate", bundle)
        cli("import", "ragflow", bundle, "--base-url", target_url, "--dataset-name", "portable-handbook", "--state", state, "--dry-run")
        cli("import", "ragflow", bundle, "--base-url", target_url, "--dataset-name", "portable-handbook", "--state", state)
        before = sum(r["method"] != "GET" for r in target.requests)
        cli("import", "ragflow", bundle, "--base-url", target_url, "--dataset-name", "portable-handbook", "--state", state)
        assert sum(r["method"] != "GET" for r in target.requests) == before, "re-import made writes"
        cli("verify", "ragflow", bundle, "--base-url", target_url, "--state", state)
        print("PASS: 3 documents / 15 chunks; disabled flags and keywords verified; re-import wrote nothing.")


if __name__ == "__main__":
    main()
