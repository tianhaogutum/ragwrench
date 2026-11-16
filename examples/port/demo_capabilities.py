"""Run the installed CLI before choosing any bundle; synthetic loopback GETs only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import ragwrench.port as kbport
from ragwrench.port.fakes import FakeState, running_fake


def main():
    cli_path = Path(sys.executable).with_name("ragwrench")
    print(f"Installed package: {kbport.__file__}")
    clean_env = {k: v for k, v in os.environ.items() if k not in ("DIFY_KB_API_KEY", "RAGFLOW_API_KEY")}
    with TemporaryDirectory(prefix="kbport-capabilities-") as tmp:
        def cli(*args, expected=0, online=False):
            env = {**clean_env, **({"RAGFLOW_API_KEY": "fake-key"} if online else {})}
            result = subprocess.run([str(cli_path), "port", "capabilities", *args], env=env, cwd=tmp,
                                    text=True, capture_output=True, timeout=30)
            print(f"capabilities {' '.join(args)} -> exit {result.returncode}")
            print(result.stdout.strip() or result.stderr.strip())
            assert result.returncode == expected, (result.stdout, result.stderr)
            assert list(Path(tmp).iterdir()) == []  # no bundle, state, checkpoint or locks
            if expected:
                assert not result.stdout and "Traceback" not in result.stderr
                return result.stderr
            assert not result.stderr
            return json.loads(result.stdout)

        dify = cli("dify")
        assert dify["api_profile"] == "dify-f8fc371a" and not dify["profile_resolution"]["server_checked"]
        assert dify["fields"]["chunks"]["questions"]["import"] == "requires_allow_lossy"
        ragflow = cli("ragflow", "--api-profile", "ragflow-0.27.3")
        assert ragflow["dataset_creation"] == {"chunk_method": "naive"}
        assert ragflow["fields"]["chunks"]["answer"]["import"] == "requires_allow_lossy"
        assert "requires --base-url" in cli("ragflow", "--api-profile", "auto", expected=1)
        assert "Unsupported API profile" in cli("dify", "--api-profile", "ragflow-1.0", expected=1)
        for profile in ("ragflow-0.27.3", "ragflow-1.0"):
            target = FakeState("ragflow", api_profile=profile)
            with running_fake(target) as url:
                args = ("ragflow", "--api-profile", "auto", "--base-url", url, "--page-size", "1", "--retries", "0")
                assert "API key" in cli(*args, expected=1)
                assert not target.requests
                assert "explicit --api-profile" in cli(*args, expected=1, online=True)
                for number in range(2):
                    ident = f"synthetic-{number}"
                    target.datasets[ident] = {"id": ident, "name": "Private synthetic name", "documents": {}}
                before = deepcopy(target.datasets)
                start = len(target.requests)
                report = cli(*args, online=True)
                assert report["api_profile"] == profile and report["profile_resolution"]["server_checked"]
                assert not report["profile_resolution"]["exact_server_version_verified"]
                assert "Private synthetic name" not in json.dumps(report)
                assert len(target.requests[start:]) == 3  # two full pages, then empty page
                assert target.datasets == before
                assert all(r["method"] == "GET" and r["path"] == "/api/v1/datasets" for r in target.requests)
        print("PASS: credential-free offline reports; both auto profiles paginated via GET only; negative cases fail without output or files.")


if __name__ == "__main__":
    main()
