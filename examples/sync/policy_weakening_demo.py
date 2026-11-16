"""Installed offline CLI: independent weakening/diagnostic exits, unchanged reports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def healthy(name="kb", limit=0.25, source="explicit", override=None):
    return {"collection": name, "management_state": "managed", "status": "ok",
            "policy": {"kind": "chunksync.policy_inspection", "version": 1,
                       "collection": name, "max_deletion_fraction": limit,
                       "policy_source": source, "pending_plan_fingerprint": None,
                       "policy_override": override}, "diagnostic": None}


def inventory(rows, source="primary"):
    return {"kind": "chunksync.policy_inventory", "version": 1,
            "catalog_source": source, "collections": rows}


def main():
    executable = str(Path(sys.executable).with_name("ragwrench"))
    weak, diagnostics = "--fail-on-policy-weakening", "--fail-on-diagnostics"
    with TemporaryDirectory(prefix="chunksync-policy-weakening-") as temporary:
        root = Path(temporary)
        before, after, output = [root / name for name in ("before.json", "after.json", "diff.json")]

        def cli(flags, expected, *, archive=True, target=None):
            destination = output if target is None else target
            saved = [path.read_bytes() for path in (before, after)]
            previous = destination.read_bytes() if destination.is_file() else None
            argv = [executable, "sync", "compare-policies", str(before), str(after), *flags]
            if archive: argv += ["--output", str(destination)]
            result = subprocess.run(argv, capture_output=True)
            print(json.dumps({"command": argv, "cwd": os.getcwd(), "exitCode": result.returncode,
                              "stdout": result.stdout.decode("utf-8"), "stderr": result.stderr.decode("utf-8")}), flush=True)
            assert result.returncode == expected, result.stderr
            assert saved == [path.read_bytes() for path in (before, after)]
            if expected == 2:
                assert not result.stdout and result.stderr.startswith(b"ragwrench sync: ")
                assert destination.read_bytes() == previous if previous is not None else not destination.exists()
            else:
                assert not result.stderr
                assert result.stdout.endswith(b"\n")
                if archive: assert destination.read_bytes() == result.stdout
            return result.stdout

        def scenario(old, new, default_exit, weakening_exit, both_exit, *, archive=True):
            before.write_text(json.dumps(inventory(old), ensure_ascii=False), encoding="utf-8")
            after.write_text(json.dumps(inventory(new, "recovery"), ensure_ascii=False), encoding="utf-8")
            baseline = cli([], default_exit, archive=archive)
            assert cli([weak], weakening_exit, archive=archive) == baseline
            assert cli([weak, diagnostics], both_exit, archive=archive) == baseline
            return json.loads(baseline)

        # Same exact complete report at 10/4 with no archive, a new archive and replacement.
        for archive in (False, True, True):
            result = scenario([healthy()], [healthy(limit=0.5)], 10, 4, 4, archive=archive)
            assert result["changed"][0]["after"] == healthy(limit=0.5)
        scenario([healthy(limit=0)], [healthy(limit=5e-324)], 10, 4, 4)
        scenario([healthy(source="legacy-review")], [healthy(limit=1, source="default")], 10, 4, 4)
        scenario([healthy(limit=1)], [healthy(limit=1.0)], 0, 0, 0)
        scenario([healthy(limit=1)], [healthy(limit=1, source="default")], 10, 10, 10)
        scenario([healthy(limit=1, source="default")], [healthy()], 10, 10, 10)
        scenario([], [], 0, 0, 0)
        scenario([healthy()], [healthy("KB", 1)], 10, 10, 10)  # Exact-name rename, not an increase.
        scenario([], [healthy(limit=1)], 10, 10, 10)
        scenario([healthy(limit=0)], [], 10, 10, 10)

        record = {"kind": "chunksync.policy_override", "version": 1, "collection": "kb",
                  "plan_fingerprint": "a" * 64, "from_policy": {"max_deletion_fraction": 0.2},
                  "to_policy": {"max_deletion_fraction": 0.8},
                  "provenance": {"reviewer": "Ada", "reason": "historical Überprüfung 合成"}}
        record["fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                                         separators=(",", ":")).encode()).hexdigest()
        scenario([healthy(override=record)], [healthy(limit=0.5, override=record)], 10, 4, 4)
        for status, management, code, message in (
            ("unmanaged", "unmanaged", "unmanaged_collection", "collection has no management state"),
            ("unsupported", "unsupported", "unsupported_management_state", "unsupported collection management state"),
            ("invalid", "managed", "invalid_policy_metadata", "invalid review policy"),
            ("unavailable", "unknown", "unavailable_collection_metadata", "unavailable metadata"),
        ):
            row = {"collection": "kb", "management_state": management, "status": status,
                   "policy": None, "diagnostic": {"code": code, "message": message}}
            scenario([healthy()], [row], 10, 10, 3)
            scenario([row], [healthy(limit=1)], 10, 10, 3)
            row["collection"] = "external"
            scenario([healthy(), row], [healthy(limit=0.5), row], 10, 4, 3)

        # An early matched increase must not bypass later invalid data or publication.
        scenario([healthy()], [healthy(limit=0.5)], 10, 4, 4)
        good = after.read_bytes()
        for invalid in (inventory([healthy(limit=0.5), {}]),
                        inventory([healthy(limit=0.5), healthy(limit=0.5)]),
                        inventory([healthy(f"c{i}") for i in range(1001)])):
            after.write_text(json.dumps(invalid))
            cli([weak, diagnostics], 2)
        after.write_bytes(good)
        for source in (before, after):
            cli([weak, diagnostics], 2, target=source)
            link = root / "hardlink.json"
            os.link(source, link)
            cli([weak, diagnostics], 2, target=link)
            link.unlink()
        link = root / "symlink.json"
        link.symlink_to(output)
        cli([weak, diagnostics], 2, target=link)
        link.unlink()
        cli([weak, diagnostics], 2, target=root / "missing" / "diff.json")
        assert not (root / "missing").exists()
        with after.open("wb") as stream: stream.truncate(64 * 1024 * 1024 + 1)
        cli([weak, diagnostics], 2)
        print("Verified byte-identical default/weakening/both stdout and archives; exits 0/10/4/3/2, unchanged inputs, no source stores.")


if __name__ == "__main__":
    main()
