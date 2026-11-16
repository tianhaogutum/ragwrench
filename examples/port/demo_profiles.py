"""Exercise installed CLI profile negotiation against pinned synthetic HTTP contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from ragwrench.port.bundle import load_bundle, write_bundle
from ragwrench.port.fakes import FakeState, running_fake

ROOT = Path(__file__).resolve().parents[2]


def main():
    cli_path = Path(sys.executable).with_name("ragwrench")
    env = {**os.environ, "RAGFLOW_API_KEY": "fake-key"}
    def cli(*args, expected=0):
        result = subprocess.run([str(cli_path), "port", *map(str, args)], env=env,
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)
        if expected:
            return result.stderr
        value = json.loads(result.stdout)
        print(result.stdout.strip())
        return value
    (ROOT / ".cache").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="profiles-", dir=ROOT / ".cache") as tmp:
        tmp = Path(tmp)
        manifest, docs, chunks = load_bundle(ROOT / "examples/port/bundle")
        chunks[0].update(answer="Seven synthetic days.", metadata={"image_id": "synthetic-image"})
        unsupported = tmp / "unsupported"
        write_bundle(unsupported, manifest["source"], {}, docs, chunks)
        for profile in ("ragflow-0.27.3", "ragflow-1.0"):
            target = FakeState("ragflow", api_profile=profile)
            mapping = tmp / profile / "mapping.json"
            with running_fake(target) as url:
                args = ("import", "ragflow", ROOT / "examples/port/bundle", "--base-url", url,
                        "--api-profile", "auto", "--dataset-name", "negotiated", "--state", mapping)
                assert "explicit --api-profile" in cli(*args, expected=1)
                assert not mapping.parent.exists() and not target.datasets
                assert all(r["method"] == "GET" for r in target.requests)
                target.seed(ROOT / "examples/port/bundle")
                imported = cli(*args)
                assert imported["api_profile"] == profile and imported["chunks"] == 15
                before = len(target.requests)
                cli(*args)
                assert all(r["method"] == "GET" for r in target.requests[before:])
                verified = cli("verify", "ragflow", ROOT / "examples/port/bundle", "--base-url", url,
                               "--api-profile", "auto", "--state", mapping)
                assert verified["verified"] and verified["api_profile"] == profile
                before = len(target.requests)
                rejected_args = ("import", "ragflow", unsupported, "--base-url", url,
                                 "--api-profile", profile, "--dataset-name", "unsupported",
                                 "--state", tmp / "unused" / "mapping.json", "--dry-run")
                error = cli(*rejected_args, expected=1)
                assert "Unsupported fields:" in error and "metadata" in error and "QA structure" in error
                plan = cli(*rejected_args, "--allow-lossy")
                assert len(plan["warnings"]) == 2 and plan["api_profile"] == profile
                assert not target.requests[before:] and not (tmp / "unused").exists()
        print("PASS: both profiles negotiated; 15 chunks verified per import; zero repeat writes; unsupported fields reported.")


if __name__ == "__main__":
    main()
