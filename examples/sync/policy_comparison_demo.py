"""Installed CLI example using synthetic archived inventories; no store exists."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main():
    executable = str(Path(sys.executable).with_name("ragwrench"))
    with TemporaryDirectory(prefix="chunksync-policy-comparison-") as temporary:
        root = Path(temporary)
        before, after = root / "before.json", root / "after.json"
        healthy = {"collection": "kb", "management_state": "managed", "status": "ok",
                   "policy": {"kind": "chunksync.policy_inspection", "version": 1,
                              "collection": "kb", "max_deletion_fraction": 0.5,
                              "policy_source": "explicit", "pending_plan_fingerprint": None,
                              "policy_override": None}, "diagnostic": None}
        diagnostic = {"collection": "external", "management_state": "unmanaged", "status": "unmanaged",
                      "policy": None, "diagnostic": {"code": "unmanaged_collection",
                                                      "message": "collection has no management state"}}
        original = {"kind": "chunksync.policy_inventory", "version": 1,
                    "catalog_source": "primary", "collections": [healthy, diagnostic]}
        before.write_text(json.dumps(original), encoding="utf-8")

        output = root / "diff.json"

        last_stdout = None

        def cli(value, expected, *, archive=True, target=None, fail_on_diagnostics=False):
            nonlocal last_stdout
            after.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            saved = [p.read_bytes() for p in (before, after)]
            destination = output if target is None else target
            previous = destination.read_bytes() if destination.is_file() else None
            argv = [executable, "sync", "compare-policies", str(before), str(after)]
            if archive:
                argv += ["--output", str(destination)]
            if fail_on_diagnostics:
                argv.append("--fail-on-diagnostics")
            result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8")
            last_stdout = result.stdout
            print(json.dumps({"command": argv, "exitCode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}), flush=True)
            assert result.returncode == expected, result.stderr
            assert saved == [p.read_bytes() for p in (before, after)]
            if expected == 2:
                assert not result.stdout and result.stderr.startswith("ragwrench sync: ")
                assert destination.read_bytes() == previous if previous is not None else not destination.exists()
                return None
            assert not result.stderr
            if archive:
                assert destination.read_bytes() == result.stdout.encode("utf-8")
            return json.loads(result.stdout)

        assert cli(original, 0, archive=False)["identical"]
        assert not output.exists()
        assert cli(original, 0)["identical"]
        reordered = {**original, "catalog_source": "recovery", "collections": [diagnostic, healthy]}
        equal = cli(reordered, 0)
        assert equal["identical"] and equal["catalog_source"] == {"before": "primary", "after": "recovery"}
        changed = {**healthy, "policy": {**healthy["policy"], "max_deletion_fraction": 0.25,
                                         "pending_plan_fingerprint": "b" * 64}}
        new_row = {**diagnostic, "collection": "added"}
        drift = cli({**original, "collections": [changed, new_row]}, 10)
        assert drift["changed"] == [{"collection": "kb", "before": healthy, "after": changed}]
        assert drift["removed"] == [diagnostic] and drift["added"] == [new_row]
        transition = cli({**original, "collections": [{**diagnostic, "collection": "kb"}]}, 10)
        assert transition["changed"] == [{"collection": "kb", "before": healthy,
                                           "after": {**diagnostic, "collection": "kb"}}]
        cli({**original, "collections": [healthy, healthy]}, 2)
        cli({**original, "version": 2}, 2)
        cli({**original, "collections": [healthy, {**diagnostic, "policy": healthy["policy"]}]}, 2)
        cli({**original, "collections": [{**diagnostic, "collection": f"c{i}"} for i in range(1001)]}, 2)
        # Each status counts on either side, including removed and unchanged rows.
        def paired(value, default_exit, strict_exit, **kwargs):
            result = cli(value, default_exit, **kwargs)
            baseline = last_stdout
            assert cli(value, strict_exit, fail_on_diagnostics=True, **kwargs) == result
            assert last_stdout == baseline
            return result

        for status, management, code, message in (
            ("unmanaged", "unmanaged", "unmanaged_collection", "collection has no management state"),
            ("unsupported", "unsupported", "unsupported_management_state", "unsupported collection management state"),
            ("invalid", "managed", "invalid_policy_metadata", "invalid review policy Überprüfung"),
            ("unavailable", "unknown", "unavailable_collection_metadata", "unsupported configuration 合成"),
        ):
            row = {**diagnostic, "status": status, "management_state": management,
                   "diagnostic": {"code": code, "message": message}}
            baseline = {**original, "collections": [healthy, row]}
            before.write_text(json.dumps(baseline), encoding="utf-8")
            output.unlink(missing_ok=True)
            assert paired(baseline, 0, 3, archive=False)["identical"]
            assert not output.exists()
            assert paired({**baseline, "catalog_source": "recovery"}, 0, 3)["identical"]
            # Removed diagnostic, changed diagnostic-to-healthy, unchanged with drift.
            assert paired({**baseline, "collections": [healthy]}, 10, 3)["removed"] == [row]
            external_ok = {**healthy, "collection": "external",
                           "policy": {**healthy["policy"], "collection": "external"}}
            paired({**baseline, "collections": [healthy, external_ok]}, 10, 3)
            paired({**baseline, "collections": [changed, row]}, 10, 3)
            before.write_text(json.dumps({**baseline, "collections": [healthy]}), encoding="utf-8")
            assert paired(baseline, 10, 3)["added"] == [row]
            before.write_text(json.dumps({**baseline, "collections": [healthy, external_ok]}), encoding="utf-8")
            paired(baseline, 10, 3)

        for rows, other, expected in (([], [], 0), ([healthy], [healthy], 0),
                                      ([healthy], [changed], 10), ([healthy], [], 10)):
            before.write_text(json.dumps({**original, "collections": rows}), encoding="utf-8")
            paired({**original, "collections": other, "catalog_source": "recovery"}, expected, expected)
        before.write_text(json.dumps(original), encoding="utf-8")
        for invalid in ({**original, "version": 2},
                        {**original, "collections": [diagnostic, {}]},
                        {**original, "collections": [diagnostic, diagnostic]},
                        {**original, "collections": [diagnostic] + [{**healthy, "collection": f"c{i}"}
                                                                     for i in range(1000)]}):
            paired(invalid, 2, 2)
        # Both inputs are protected, even if an input is reached via an alias.
        for source in (before, after):
            paired(original, 2, 2, target=source)
            alias = root / "hardlink.json"
            os.link(source, alias)
            paired(original, 2, 2, target=alias)
            alias.unlink()
        link = root / "symlink.json"
        link.symlink_to(output)
        paired(original, 2, 2, target=link)
        link.unlink()
        paired(original, 2, 2, target=root / "missing-parent" / "diff.json")
        assert not (root / "missing-parent").exists()
        # Oversized archives are refused before JSON parsing, without any store.
        with after.open("wb") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
        saved = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (before, after, output)]
        argv = [executable, "sync", "compare-policies", str(before), str(after), "--output", str(output), "--fail-on-diagnostics"]
        result = subprocess.run(argv, capture_output=True, text=True)
        print(json.dumps({"command": argv, "exitCode": result.returncode,
                          "stdout": result.stdout, "stderr": result.stderr}), flush=True)
        assert result.returncode == 2 and not result.stdout and "64 MiB" in result.stderr
        assert saved == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (before, after, output)]
        print("Verified default/opt-in exact stdout/archives, every diagnostic status on either side at exit 3, healthy 0/10, global-error priority 2 and both-input preservation without source stores.")


if __name__ == "__main__":
    main()
