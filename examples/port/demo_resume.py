"""Interrupt an installed CLI, then resume through a failed chunk response, offline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

from ragwrench.port.bundle import fingerprint, load_bundle, write_bundle
from ragwrench.port.common import read_json
from ragwrench.port.fakes import FakeState, running_fake

ROOT = Path(__file__).resolve().parents[2]


class InterruptingTarget(FakeState):
    client_pid = None
    flag_writes = 0

    def dispatch(self, method, path, query, body):
        result = super().dispatch(method, path, query, body)
        if method == "PATCH" and body and "available" in body:
            self.flag_writes += 1
            if self.flag_writes == 3:
                os.kill(self.client_pid, signal.SIGINT)
        return result


def main():
    # Use the console script belonging to this interpreter's installed package.
    cli = Path(sys.executable).with_name("ragwrench")
    env = {**os.environ, "RAGFLOW_API_KEY": "fake-key"}
    target = InterruptingTarget("ragflow")
    with TemporaryDirectory(prefix="kbport-resume-") as tmp, running_fake(target) as url:
        state = Path(tmp) / "mapping.json"
        command = [str(cli), "port", "import", "ragflow", str(ROOT / "examples/port/bundle"),
                   "--base-url", url, "--dataset-name", "resumable-handbook", "--state", str(state),
                   "--backoff", "0", "--chunk-retries", "1"]
        with subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            target.client_pid = process.pid
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 130, (process.returncode, stdout, stderr)
        checkpoint = next(iter(read_json(state)["documents"].values()))
        assert len(checkpoint["chunks"]) == 2 and checkpoint["pending"]["target"]
        print(json.dumps({"interrupted_exit": 130, "completed_chunks": 2, "pending_chunks": 1}))

        # Inspect the saved intent without credentials, HTTP calls or checkpoint changes.
        offline_env = {k: v for k, v in os.environ.items() if k not in ("DIFY_KB_API_KEY", "RAGFLOW_API_KEY")}
        def check_bundle_chunks(command, full):
            before = len(target.requests), {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                           for p in Path(tmp).rglob("*") if p.is_file()}
            expected = []
            for doc in full["bundle_progress"]["documents"]:
                result = subprocess.run(command + ["--document", doc["source_id"], "--list-chunks"],
                                        env=offline_env, text=True, capture_output=True, timeout=10)
                assert result.returncode == 0 and not result.stderr
                expected.extend(json.loads(result.stdout)["bundle_progress"]["chunks"])
            assert len(expected) == (full["completed_chunks"] + full["bundle_progress"]["remaining_chunks"] +
                                     full["bundle_progress"]["skipped_chunks"])
            def read(extra):
                result = subprocess.run(command + ["--list-bundle-chunks"] + extra, env=offline_env,
                                        text=True, capture_output=True, timeout=10)
                assert result.returncode == 0 and not result.stderr, result.stderr
                print(json.dumps({"bundle_chunk_command": result.args, "exitCode": result.returncode,
                                  "stdout": json.loads(result.stdout)}))
                return json.loads(result.stdout)
            for status in (None, "completed", "pending", "unvisited", "skipped"):
                extra = ["--bundle-chunk-status", status] if status is not None else []
                rows = [row for row in expected if status is None or row["status"] == status]
                result = read(extra)
                assert result["bundle_progress"].pop("chunk_details") == rows and result == full
                token, offset, seen = None, 0, []
                while True:
                    size = 4 if offset == 0 else 6
                    page_extra = extra + ["--bundle-chunk-page-size", str(size)]
                    if token is not None: page_extra += ["--bundle-chunk-cursor", token]
                    result = read(page_extra)
                    page = result["bundle_progress"].pop("chunk_details")
                    token = result["bundle_progress"].pop("next_bundle_chunk_cursor")
                    assert page == rows[offset:offset+size] and result == full
                    seen.extend(page); offset += len(page)
                    if offset == len(rows):
                        assert token is None
                        break
                    assert token
                    stale_path = Path(tmp) / "bundle-chunk-stale.json"
                    stale_path.write_bytes(Path(command[command.index("--state") + 1]).read_bytes() + b" ")
                    stale_command = command.copy()
                    stale_command[stale_command.index("--state") + 1] = str(stale_path)
                    stale = subprocess.run(stale_command + ["--list-bundle-chunks"] + extra +
                                           ["--bundle-chunk-page-size", "1", "--bundle-chunk-cursor", token],
                                           env=offline_env, text=True, capture_output=True, timeout=10)
                    assert stale.returncode == 1 and not stale.stdout and "snapshot mismatch" in stale.stderr
                    print(json.dumps({"bundle_chunk_stale_command": stale.args, "exitCode": stale.returncode,
                                      "stderr": stale.stderr}))
                    stale_path.unlink()
                assert seen == rows
            assert (len(target.requests), {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                          for p in Path(tmp).rglob("*") if p.is_file()}) == before

        def check_skips(command, full):
            check_bundle_chunks(command, full)
            expected = [{"document_source_id": doc["source_id"], **item}
                        for doc in full["bundle_progress"]["documents"] for item in doc["skipped_chunks"]]
            before = len(target.requests), {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                           for p in Path(tmp).rglob("*") if p.is_file()}
            def read(extra):
                result = subprocess.run(command + ["--list-skipped-chunks"] + extra, env=offline_env,
                                        text=True, capture_output=True, timeout=10)
                assert result.returncode == 0 and not result.stderr, result.stderr
                print(json.dumps({"skip_command": command + ["--list-skipped-chunks"] + extra,
                                  "exitCode": result.returncode, "stdout": json.loads(result.stdout)}))
                return json.loads(result.stdout)
            listed = read([])
            assert listed["bundle_progress"].pop("skipped_chunk_details") == expected
            assert listed == full
            token, offset, seen = None, 0, []
            while True:
                size = 1 if offset == 0 else 2
                extra = ["--skip-page-size", str(size)]
                if token is not None: extra += ["--skip-cursor", token]
                result = read(extra)
                rows = result["bundle_progress"].pop("skipped_chunk_details")
                token = result["bundle_progress"].pop("next_skip_cursor")
                assert rows == expected[offset:offset+size] and result == full
                seen.extend(rows); offset += len(rows)
                if offset == len(expected):
                    assert token is None
                    break
                assert token
                stale_path = Path(tmp) / "skip-stale.json"
                stale_path.write_bytes(Path(command[command.index("--state") + 1]).read_bytes() + b" ")
                stale_command = command.copy()
                stale_command[stale_command.index("--state") + 1] = str(stale_path)
                stale = subprocess.run(stale_command + ["--list-skipped-chunks", "--skip-page-size", "1",
                                                       "--skip-cursor", token], env=offline_env,
                                       text=True, capture_output=True, timeout=10)
                assert stale.returncode == 1 and not stale.stdout and "snapshot mismatch" in stale.stderr
                print(json.dumps({"skip_stale_command": stale.args, "exitCode": stale.returncode, "stderr": stale.stderr}))
                stale_path.unlink()
            assert seen == expected
            assert (len(target.requests), {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                          for p in Path(tmp).rglob("*") if p.is_file()}) == before

        def check_summary(command, full):
            before = len(target.requests), {
                p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
            summary = subprocess.run(command + ["--summary-only"], env=offline_env,
                                     text=True, capture_output=True, timeout=10)
            assert summary.returncode == 0 and not summary.stderr, summary.stderr
            expected = {key: value for key, value in full.items() if key != "documents"}
            if "bundle_progress" in full:
                expected["bundle_progress"] = {key: value for key, value in full["bundle_progress"].items()
                                               if key != "documents"}
            assert json.loads(summary.stdout) == expected
            counts = {"dataset_creation": int(full["dataset_pending"])}
            for stage in ("document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag"):
                counts[stage] = sum(row["pending_stage"] == stage for row in full["documents"])
            counted = subprocess.run(command + ["--summary-only", "--pending-counts"], env=offline_env,
                                     text=True, capture_output=True, timeout=10)
            assert counted.returncode == 0 and not counted.stderr, counted.stderr
            assert json.loads(counted.stdout) == {**expected, "pending_counts": counts}
            assert all(type(value) is int for value in json.loads(counted.stdout)["pending_counts"].values())
            # Drill into exactly the same saved document intents, preserving all totals.
            for stage in ("document_creation", "bootstrap_cleanup", "chunk_creation", "enabled_flag"):
                filtered = subprocess.run(command + ["--pending-stage", stage], env=offline_env,
                                          text=True, capture_output=True, timeout=10)
                assert filtered.returncode == 0 and not filtered.stderr, filtered.stderr
                matching = [row for row in full["documents"] if row["pending_stage"] == stage]
                ids = {row["source_id"] for row in matching}
                wanted = {**full, "documents": matching}
                if "bundle_progress" in full:
                    wanted["bundle_progress"] = {**full["bundle_progress"], "documents": [
                        row for row in full["bundle_progress"]["documents"] if row["source_id"] in ids]}
                assert json.loads(filtered.stdout) == wanted
                assert len(matching) == counts[stage]
                offset, cursor, seen = 0, None, []
                while True:
                    extra = ["--pending-stage", stage, "--document-page-size", "1"]
                    if cursor is not None:
                        extra += ["--document-cursor", cursor]
                    paged = subprocess.run(command + extra, env=offline_env, text=True,
                                           capture_output=True, timeout=10)
                    assert paged.returncode == 0 and not paged.stderr, paged.stderr
                    page = json.loads(paged.stdout)
                    cursor = page.pop("next_document_cursor")
                    ids = {row["source_id"] for row in matching[offset:offset+1]}
                    expected_page = {**full, "documents": matching[offset:offset+1]}
                    if "bundle_progress" in full:
                        expected_page["bundle_progress"] = {**full["bundle_progress"], "documents": [
                            row for row in full["bundle_progress"]["documents"] if row["source_id"] in ids]}
                    assert page == expected_page
                    seen.extend(row["source_id"] for row in page["documents"])
                    offset += 1
                    print(json.dumps({"document_page": json.loads(paged.stdout), "exit": paged.returncode}))
                    if offset >= len(matching):
                        assert cursor is None
                        break
                    assert cursor
                assert seen == [row["source_id"] for row in matching]
                print(json.dumps({"pending_stage": stage, "exit": filtered.returncode,
                                  "snapshot": json.loads(filtered.stdout)}))
            print(counted.stdout.strip())
            assert (len(target.requests), {
                p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}) == before
            print(summary.stdout.strip())

        before_status = len(target.requests), state.read_bytes()
        status = subprocess.run([str(cli), "port", "import-status", "--state", str(state)],
                                env=offline_env, text=True, capture_output=True, timeout=10)
        assert status.returncode == 0, status.stderr
        snapshot = json.loads(status.stdout)
        assert snapshot["mapped_documents"] == 1 and snapshot["completed_chunks"] == 2
        assert snapshot["documents"][0]["pending_stage"] == "enabled_flag"
        assert (len(target.requests), state.read_bytes()) == before_status
        check_summary([str(cli), "port", "import-status", "--state", str(state)], snapshot)
        print(status.stdout.strip())
        missing_counts = Path(tmp) / "missing-counts" / "state.json"
        rejected = subprocess.run([str(cli), "port", "import-status", "--state", str(missing_counts), "--pending-counts"],
                                  env=offline_env, text=True, capture_output=True, timeout=10)
        assert rejected.returncode == 2 and not rejected.stdout
        assert "--pending-counts requires --summary-only" in rejected.stderr
        assert not missing_counts.parent.exists()
        for extra in (["--summary-only"], ["--pending-counts"], ["--document", ""],
                      ["--chunk", ""], ["--list-chunks"], ["--chunk-status", "pending"],
                      ["--page-size", "1"], ["--cursor", ""]):
            rejected = subprocess.run([str(cli), "port", "import-status", "--state", str(missing_counts),
                                       "--pending-stage", "enabled_flag"] + extra,
                                      env=offline_env, text=True, capture_output=True, timeout=10)
            assert rejected.returncode == 2 and not rejected.stdout
            assert "--pending-stage cannot be combined" in rejected.stderr
        for extra in (["--document-page-size", "0", "--pending-stage", "enabled_flag"],
                      ["--document-page-size", "1"], ["--document-cursor", ""],
                      ["--document-cursor", "token", "--pending-stage", "enabled_flag"],
                      ["--document-page-size", "1", "--pending-stage", "enabled_flag", "--summary-only"]):
            rejected = subprocess.run([str(cli), "port", "import-status", "--state", str(missing_counts)] + extra,
                                      env=offline_env, text=True, capture_output=True, timeout=10)
            assert rejected.returncode == 2 and not rejected.stdout
        for stage in ("dataset_creation", "", "invalid"):
            rejected = subprocess.run([str(cli), "port", "import-status", "--state", str(missing_counts),
                                       "--pending-stage", stage], env=offline_env,
                                      text=True, capture_output=True, timeout=10)
            assert rejected.returncode == 2 and not rejected.stdout and "invalid choice" in rejected.stderr
        assert not missing_counts.parent.exists()

        bundle = ROOT / "examples/port/bundle"
        bundle_command = [str(cli), "port", "import-status", "--state", str(state), "--bundle", str(bundle)]
        before_files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
        status = subprocess.run(bundle_command, env=offline_env, text=True, capture_output=True, timeout=10)
        assert status.returncode == 0 and not status.stderr, status.stderr
        bundle_snapshot = json.loads(status.stdout)

        # Derive a mixed-intent snapshot from the real interrupted checkpoint so
        # installed continuations traverse more than one matching document.
        derived = read_json(state)
        _, source_docs, source_chunks = load_bundle(bundle)
        second, third = [doc["id"] for doc in source_docs[1:]]
        derived["documents"][second] = {
            "id": "derived-doc", "phase": "chunks", "chunks": {},
            "pending": {"source": next(c["id"] for c in source_chunks if c["doc_id"] == second),
                        "target": "derived-chunk"}}
        derived["documents"][third] = {"id": None, "phase": "creating", "chunks": {}, "pending": None}
        derived_path = Path(tmp) / "document-pages.json"
        derived_path.write_text(json.dumps(derived))
        for scoped in (False, True):
            derived_command = [str(cli), "port", "import-status", "--state", str(derived_path)]
            if scoped:
                derived_command += ["--bundle", str(bundle)]
            result = subprocess.run(derived_command, env=offline_env, text=True, capture_output=True, timeout=10)
            assert result.returncode == 0 and not result.stderr, result.stderr
            check_summary(derived_command, json.loads(result.stdout))
            page_args = ["--pending-stage", "enabled_flag", "--document-page-size", "1"]
            first = subprocess.run(derived_command + page_args, env=offline_env,
                                   text=True, capture_output=True, timeout=10)
            assert first.returncode == 0 and not first.stderr
            token = json.loads(first.stdout)["next_document_cursor"]
            assert token
            for stage, scope_args in (("chunk_creation", []),
                                      ("enabled_flag", [] if scoped else ["--bundle", str(bundle)])):
                # Switching/removing bundle scope must invalidate continuation.
                scoped_command = [str(cli), "port", "import-status", "--state", str(derived_path)] + scope_args
                rejected = subprocess.run(scoped_command + ["--pending-stage", stage, "--document-page-size", "1",
                                                            "--document-cursor", token], env=offline_env,
                                          text=True, capture_output=True, timeout=10)
                assert rejected.returncode == 1 and not rejected.stdout
                assert "Invalid document continuation" in rejected.stderr
            before_derived = derived_path.read_bytes()
            derived_path.write_bytes(before_derived + b" ")
            rejected = subprocess.run(derived_command + page_args + ["--document-cursor", token],
                                      env=offline_env, text=True, capture_output=True, timeout=10)
            assert rejected.returncode == 1 and not rejected.stdout and "Invalid document continuation" in rejected.stderr
            derived_path.write_bytes(before_derived)
        derived_path.unlink()  # Remove only this synthetic example fixture.
        assert state.read_bytes() == before_status[1]
        print(json.dumps({"document_pages": "mixed interrupted-derived state/bundle continuations; stage/scope/byte changes rejected"}))
        progress = bundle_snapshot["bundle_progress"]
        check_skips(bundle_command, bundle_snapshot)
        check_summary(bundle_command, bundle_snapshot)
        _, source_docs, source_chunks = load_bundle(bundle)

        def check_chunk(command, snapshot, doc_index, chunk_index, local_status,
                        target_id=None, pending_stage=None, reason=None):
            source, chunk = source_docs[doc_index]["id"], source_chunks[chunk_index]["id"]
            detail = {"document_source_id": source, "source_id": chunk, "status": local_status,
                      "target_id": target_id, "pending_stage": pending_stage, "reason": reason}
            selected = subprocess.run(command + ["--document", source, "--chunk", chunk],
                                      env=offline_env, text=True, capture_output=True, timeout=10)
            assert selected.returncode == 0 and not selected.stderr, selected.stderr
            p = snapshot["bundle_progress"]
            assert json.loads(selected.stdout) == {**snapshot, "bundle_progress": {
                **p, "documents": [p["documents"][doc_index]], "chunk": detail}}
            print(selected.stdout.strip())

        filtered_cursors = []

        def check_list(command, snapshot, doc_index):
            source = source_docs[doc_index]["id"]
            selected_command = command + ["--document", source]
            listed = subprocess.run(selected_command + ["--list-chunks"], env=offline_env,
                                    text=True, capture_output=True, timeout=10)
            assert listed.returncode == 0 and not listed.stderr, listed.stderr
            result = json.loads(listed.stdout)
            rows = result["bundle_progress"].pop("chunks")
            p = snapshot["bundle_progress"]
            assert result == {**snapshot, "bundle_progress": {**p, "documents": [p["documents"][doc_index]]}}
            assert [r["source_id"] for r in rows] == sorted(c["id"] for c in source_chunks if c["doc_id"] == source)
            for row in rows:
                single = subprocess.run(selected_command + ["--chunk", row["source_id"]], env=offline_env,
                                        text=True, capture_output=True, timeout=10)
                assert single.returncode == 0 and not single.stderr, single.stderr
                assert row == json.loads(single.stdout)["bundle_progress"]["chunk"]
            first_cursor = None
            for local_status in (None, "completed", "pending", "unvisited", "skipped"):
                matching = [row for row in rows if local_status is None or row["status"] == local_status]
                filter_args = ["--chunk-status", local_status] if local_status else []
                if local_status:
                    filtered = subprocess.run(selected_command + ["--list-chunks"] + filter_args,
                                              env=offline_env, text=True, capture_output=True, timeout=10)
                    assert filtered.returncode == 0 and not filtered.stderr, filtered.stderr
                    value = json.loads(filtered.stdout)
                    assert value["bundle_progress"].pop("chunks") == matching
                    assert value == result
                    print(filtered.stdout.strip())
                size = 1 if local_status else 2
                offset, cursor = 0, None
                while True:
                    extra = ["--list-chunks", "--page-size", str(size)] + filter_args
                    if cursor is not None:
                        extra += ["--cursor", cursor]
                    paged = subprocess.run(selected_command + extra, env=offline_env,
                                           text=True, capture_output=True, timeout=10)
                    assert paged.returncode == 0 and not paged.stderr, paged.stderr
                    page = json.loads(paged.stdout)
                    actual = page["bundle_progress"].pop("chunks")
                    assert actual == matching[offset:offset + size]
                    cursor = page["bundle_progress"].pop("next_cursor")
                    assert page == result
                    offset += len(actual)
                    print(paged.stdout.strip())
                    if offset == len(matching):
                        assert cursor is None
                        break
                    assert cursor
                    if local_status is None:
                        first_cursor = first_cursor or cursor
                    else:
                        filtered_cursors.append((selected_command, cursor, local_status))
                        # A token cannot be reused with an omitted or different filter.
                        for other in ([], ["--chunk-status", "pending"]):
                            rejected = subprocess.run(selected_command + ["--list-chunks", "--page-size", "1",
                                                                          "--cursor", cursor] + other,
                                                      env=offline_env, text=True, capture_output=True, timeout=10)
                            assert rejected.returncode == 1 and not rejected.stdout
                            assert "snapshot mismatch" in rejected.stderr
            print(listed.stdout.strip())
            return first_cursor

        completed_target = checkpoint["chunks"][source_chunks[0]["id"]]
        check_chunk(bundle_command, bundle_snapshot, 0, 0, "completed", completed_target)
        check_chunk(bundle_command, bundle_snapshot, 0, 2, "pending", checkpoint["pending"]["target"], "enabled_flag")
        check_chunk(bundle_command, bundle_snapshot, 0, 4, "unvisited")
        check_chunk(bundle_command, bundle_snapshot, 1, 5, "unvisited")
        initial_cursor = check_list(bundle_command, bundle_snapshot, 0)
        check_list(bundle_command, bundle_snapshot, 1)
        assert (progress["planned_documents"], progress["planned_chunks"], progress["skipped_chunks"]) == (3, 15, 0)
        assert (progress["remaining_documents"], progress["remaining_chunks"]) == (3, 13)
        assert (progress["unvisited_documents"], progress["unvisited_chunks"], progress["pending_chunks"]) == (2, 12, 1)
        rows = progress["documents"]
        assert [r["source_id"] for r in rows] == sorted(r["source_id"] for r in rows)
        assert [r["completed_chunks"] for r in rows] == [2, 0, 0]
        assert [r["pending_chunks"] for r in rows] == [1, 0, 0]
        assert [r["remaining_chunks"] for r in rows] == [3, 5, 5]
        assert [r["unvisited_chunks"] for r in rows] == [2, 5, 5]
        assert [r["visited"] for r in rows] == [True, False, False]
        assert all(not r["complete"] and not r["skipped_chunks"] for r in rows)
        for row in rows[:2]:
            selected = subprocess.run(bundle_command + ["--document", row["source_id"]], env=offline_env,
                                      text=True, capture_output=True, timeout=10)
            assert selected.returncode == 0 and not selected.stderr, selected.stderr
            assert json.loads(selected.stdout) == {
                **bundle_snapshot, "bundle_progress": {**progress, "documents": [row]}}
            print(selected.stdout.strip())
        unknown = subprocess.run(bundle_command + ["--document", "private-unknown"], env=offline_env,
                                 text=True, capture_output=True, timeout=10)
        assert unknown.returncode == 1 and not unknown.stdout and "private-unknown" not in unknown.stderr
        missing_bundle = subprocess.run([str(cli), "port", "import-status", "--state", str(state),
                                         "--document", rows[0]["source_id"]], env=offline_env,
                                        text=True, capture_output=True, timeout=10)
        assert missing_bundle.returncode == 2 and not missing_bundle.stdout
        for extra in (["--chunk-status", "pending"],
                      ["--chunk-status", "pending", "--list-chunks"],
                      ["--bundle", str(bundle), "--document", rows[0]["source_id"], "--chunk-status", "pending"],
                      ["--chunk", "private-chunk"],
                      ["--bundle", str(bundle), "--chunk", "private-chunk"],
                      ["--document", rows[0]["source_id"], "--chunk", "private-chunk"],
                      ["--list-chunks"], ["--bundle", str(bundle), "--list-chunks"],
                      ["--document", rows[0]["source_id"], "--list-chunks"],
                      ["--bundle", str(bundle), "--document", rows[0]["source_id"],
                       "--chunk", "private-chunk", "--list-chunks"],
                      ["--bundle", str(bundle), "--document", rows[0]["source_id"],
                       "--list-chunks", "--page-size", "0"],
                      ["--bundle", str(bundle), "--document", rows[0]["source_id"],
                       "--list-chunks", "--cursor", "private-cursor"]):
            failed = subprocess.run([str(cli), "port", "import-status", "--state", str(state)] + extra,
                                    env=offline_env, text=True, capture_output=True, timeout=10)
            assert failed.returncode == 2 and not failed.stdout and "private" not in failed.stderr
        for extra in (["--document", rows[0]["source_id"]], ["--chunk", source_chunks[0]["id"]],
                      ["--list-chunks"], ["--chunk-status", "completed"],
                      ["--page-size", "1"], ["--cursor", "private-token"]):
            rejected = subprocess.run(bundle_command + ["--summary-only", "--pending-counts"] + extra, env=offline_env,
                                      text=True, capture_output=True, timeout=10)
            assert rejected.returncode == 2 and not rejected.stdout
            assert "--summary-only cannot be combined" in rejected.stderr
        unknown_list = subprocess.run(bundle_command + ["--document", "private-document", "--list-chunks"],
                                      env=offline_env, text=True, capture_output=True, timeout=10)
        assert unknown_list.returncode == 1 and not unknown_list.stdout and "private" not in unknown_list.stderr
        for doc, chunk in ((rows[0]["source_id"], "private-chunk"),
                           (rows[0]["source_id"], source_chunks[5]["id"]),
                           ("private-document", source_chunks[0]["id"])):
            failed = subprocess.run(bundle_command + ["--document", doc, "--chunk", chunk],
                                    env=offline_env, text=True, capture_output=True, timeout=10)
            assert failed.returncode == 1 and not failed.stdout and "private" not in failed.stderr
        print(json.dumps({"rejected_selector": "unknown", "exit": unknown.returncode,
                          "without_bundle_exit": missing_bundle.returncode}))
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()} == before_files
        assert (len(target.requests), state.read_bytes()) == before_status
        print(status.stdout.strip())

        # A cursor also binds exact checkpoint bytes and document, without writes.
        changed_path = Path(tmp) / "whitespace-state.json"
        changed_path.write_bytes(state.read_bytes() + b" ")
        for path, source in ((state, source_docs[1]["id"]), (changed_path, source_docs[0]["id"])):
            before = len(target.requests), path.read_bytes(), path.stat().st_mtime_ns
            rejected = subprocess.run([str(cli), "port", "import-status", "--state", str(path), "--bundle", str(bundle),
                                       "--document", source, "--list-chunks", "--page-size", "2",
                                       "--cursor", initial_cursor], env=offline_env,
                                      text=True, capture_output=True, timeout=10)
            assert rejected.returncode == 1 and not rejected.stdout and "snapshot mismatch" in rejected.stderr
            assert (len(target.requests), path.read_bytes(), path.stat().st_mtime_ns) == before
        print(json.dumps({"cursor_rejected": ["different document", "state whitespace"], "exit": 1}))

        # A valid but different bundle and damaged bytes both fail before output.
        manifest, documents, chunks = load_bundle(bundle)
        # Derive a separate offline legacy checkpoint to illustrate saved skip preflight.
        # Chunk 5 is still unvisited, so excluding it leaves all saved mappings in scope.
        skipped_bundle = Path(tmp) / "skip-bundle"
        chunks[4]["metadata"] = {"unsupported": True}
        skipped_manifest = write_bundle(skipped_bundle, manifest["source"], manifest["settings"], documents, chunks)
        skipped_state = read_json(state)
        skipped_state["identity"].update(bundle=fingerprint(skipped_manifest), on_error="skip")
        del skipped_state["identity"]["api_profile"]
        skipped_path = Path(tmp) / "skip-state.json"
        skipped_path.write_text(json.dumps(skipped_state))
        before_files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
        skipped_status = subprocess.run([str(cli), "port", "import-status", "--state", str(skipped_path),
                                         "--bundle", str(skipped_bundle)], env=offline_env,
                                        text=True, capture_output=True, timeout=10)
        assert skipped_status.returncode == 0 and not skipped_status.stderr, skipped_status.stderr
        skipped_snapshot = json.loads(skipped_status.stdout)
        p = skipped_snapshot["bundle_progress"]
        first = p["documents"][0]
        assert skipped_snapshot["completed_chunks"] == 2 and skipped_snapshot["api_profile"] == "ragflow-1.0"
        assert (p["planned_chunks"], p["skipped_chunks"], p["remaining_chunks"], p["unvisited_chunks"]) == (14, 1, 12, 11)
        assert (first["planned_chunks"], first["completed_chunks"], first["remaining_chunks"],
                first["pending_chunks"], first["unvisited_chunks"]) == (4, 2, 2, 1, 1)
        assert first["skipped_chunks"] == [{"source_id": chunks[4]["id"], "reason":
            "Unsupported fields: chunk metadata (including image/layout/tags); use --allow-lossy to acknowledge"}]
        selected = subprocess.run([str(cli), "port", "import-status", "--state", str(skipped_path),
                                   "--bundle", str(skipped_bundle), "--document", first["source_id"]],
                                  env=offline_env, text=True, capture_output=True, timeout=10)
        assert selected.returncode == 0 and not selected.stderr, selected.stderr
        assert json.loads(selected.stdout) == {
            **skipped_snapshot, "bundle_progress": {**p, "documents": [first]}}
        print(selected.stdout.strip())
        skip_command = [str(cli), "port", "import-status", "--state", str(skipped_path), "--bundle", str(skipped_bundle)]
        check_skips(skip_command, skipped_snapshot)
        check_summary(skip_command, skipped_snapshot)
        check_chunk(skip_command, skipped_snapshot, 0, 4, "skipped", reason=first["skipped_chunks"][0]["reason"])
        check_list(skip_command, skipped_snapshot, 0)
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()} == before_files
        assert (len(target.requests), state.read_bytes()) == before_status
        print(skipped_status.stdout.strip())
        # A second derived skip snapshot exercises continuation across documents.
        chunks[9]["metadata"] = {"layout": "descriptive"}
        chunks[10]["metadata"] = {"layout": "descriptive"}
        multi_bundle = Path(tmp) / "multi-skip-bundle"
        multi_manifest = write_bundle(multi_bundle, manifest["source"], manifest["settings"], documents, chunks)
        multi_state = read_json(skipped_path)
        multi_state["identity"]["bundle"] = fingerprint(multi_manifest)
        multi_path = Path(tmp) / "multi-skip-state.json"
        multi_path.write_text(json.dumps(multi_state))
        multi_command = [str(cli), "port", "import-status", "--state", str(multi_path), "--bundle", str(multi_bundle)]
        multi = subprocess.run(multi_command, env=offline_env, text=True, capture_output=True, timeout=10)
        assert multi.returncode == 0 and not multi.stderr
        multi_snapshot = json.loads(multi.stdout)
        assert multi_snapshot["bundle_progress"]["skipped_chunks"] == 3
        check_skips(multi_command, multi_snapshot)
        # Derive an offline pre-create intent; inspecting it does not retry a write.
        creating_state = read_json(skipped_path)
        creating_state["documents"][first["source_id"]]["pending"]["target"] = None
        creating_path = Path(tmp) / "creating-state.json"
        creating_path.write_text(json.dumps(creating_state))
        before_files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
        creating_command = [str(cli), "port", "import-status", "--state", str(creating_path), "--bundle", str(skipped_bundle)]
        creating_snapshot = json.loads(skipped_status.stdout)
        creating_snapshot["state"] = str(creating_path)
        creating_snapshot["documents"][0].update(pending_stage="chunk_creation", pending_chunk={
            "source_id": source_chunks[2]["id"], "target_id": None})
        check_bundle_chunks(creating_command, creating_snapshot)
        check_summary(creating_command, creating_snapshot)
        check_chunk(creating_command, creating_snapshot, 0, 2, "pending", pending_stage="chunk_creation")
        check_list(creating_command, creating_snapshot, 0)
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()} == before_files
        assert (len(target.requests), state.read_bytes()) == before_status
        # Selecting another doc must still reject an out-of-scope pending source in doc 1.
        skipped_state["documents"][first["source_id"]]["pending"]["source"] = "private-unknown-chunk"
        invalid_path = Path(tmp) / "invalid-state.json"
        invalid_path.write_text(json.dumps(skipped_state))
        before_files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
        invalid = subprocess.run([str(cli), "port", "import-status", "--state", str(invalid_path),
                                  "--bundle", str(skipped_bundle), "--document", rows[1]["source_id"],
                                  "--list-chunks", "--chunk-status", "completed", "--page-size", "1"],
                                 env=offline_env, text=True, capture_output=True, timeout=10)
        assert invalid.returncode == 1 and not invalid.stdout and "private" not in invalid.stderr
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()} == before_files
        assert (len(target.requests), state.read_bytes()) == before_status
        print(json.dumps({"rejected_selector": "invalid state outside selected doc", "exit": invalid.returncode}))
        invalid_summary = subprocess.run([str(cli), "port", "import-status", "--state", str(invalid_path),
                                          "--bundle", str(skipped_bundle), "--summary-only", "--pending-counts"],
                                         env=offline_env, text=True, capture_output=True, timeout=10)
        invalid_filtered = subprocess.run([str(cli), "port", "import-status", "--state", str(invalid_path),
                                           "--bundle", str(skipped_bundle), "--pending-stage", "document_creation"],
                                          env=offline_env, text=True, capture_output=True, timeout=10)
        assert invalid_filtered.returncode == 1 and not invalid_filtered.stdout
        assert "Invalid import state" in invalid_filtered.stderr and "private" not in invalid_filtered.stderr
        assert invalid_summary.returncode == 1 and not invalid_summary.stdout
        assert "Invalid import state" in invalid_summary.stderr and "private" not in invalid_summary.stderr
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()} == before_files
        assert (len(target.requests), state.read_bytes()) == before_status
        # Restore the original records for the mismatch/hash checks below.
        manifest, documents, chunks = load_bundle(bundle)
        other = Path(tmp) / "other-bundle"
        write_bundle(other, manifest["source"], {"different": True}, documents, chunks)
        for corrupt in (False, True):
            if corrupt:
                with (other / "chunks.jsonl").open("a") as stream:
                    stream.write("\n")
            before_files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()}
            failed = subprocess.run(bundle_command[:-1] + [str(other), "--document", rows[1]["source_id"],
                                                          "--list-chunks", "--chunk-status", "completed", "--page-size", "1"], env=offline_env,
                                    text=True, capture_output=True, timeout=10)
            assert failed.returncode == 1 and not failed.stdout and failed.stderr
            assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(tmp).rglob("*") if p.is_file()} == before_files
            assert (len(target.requests), state.read_bytes()) == before_status
            print(json.dumps({"rejected_bundle": "hash" if corrupt else "fingerprint", "exit": failed.returncode}))

        # Commit the next chunk but replace its successful response with HTTP 503.
        target.faults = [{"method": "POST", "path_suffix": "/chunks", "phase": "after", "status": 503}]
        resumed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=20)
        assert resumed.returncode == 0, resumed.stderr
        print(resumed.stdout.strip())
        docs = [d for ds in target.datasets.values() for d in ds["documents"].values()]
        assert len(target.datasets) == 1 and len(docs) == 3 and sum(len(d["chunks"]) for d in docs) == 15
        creates = sum(r["method"] == "POST" and r["path"].endswith("/chunks") for r in target.requests)
        assert creates == 15, "a chunk create was replayed"
        before = len(target.requests)
        repeated = subprocess.run(command, env=env, text=True, capture_output=True, timeout=20)
        assert repeated.returncode == 0, repeated.stderr
        assert all(r["method"] == "GET" for r in target.requests[before:])
        verified = subprocess.run([str(cli), "port", "verify", "ragflow", str(ROOT / "examples/port/bundle"),
                                   "--base-url", url, "--state", str(state)],
                                  env=env, text=True, capture_output=True, timeout=20)
        assert verified.returncode == 0, verified.stderr
        print(verified.stdout.strip())
        status = subprocess.run([str(cli), "port", "import-status", "--state", str(state)],
                                env=offline_env, text=True, capture_output=True, timeout=10)
        assert status.returncode == 0, status.stderr
        snapshot = json.loads(status.stdout)
        assert snapshot["mapped_documents"] == 3 and snapshot["completed_chunks"] == 15
        assert all(d["pending_stage"] is None for d in snapshot["documents"])
        print(status.stdout.strip())
        before_status = len(target.requests), state.read_bytes()
        status = subprocess.run(bundle_command, env=offline_env, text=True, capture_output=True, timeout=10)
        assert status.returncode == 0 and not status.stderr, status.stderr
        progress = json.loads(status.stdout)["bundle_progress"]
        assert progress["completed_documents"] == 3 and progress["planned_chunks"] == 15
        assert all(progress[key] == 0 for key in ("remaining_documents", "remaining_chunks", "unvisited_documents",
                                                 "unvisited_chunks", "pending_chunks", "skipped_chunks"))
        assert all(r["complete"] and r["completed_chunks"] == 5 and r["remaining_chunks"] == 0
                   and r["skipped_chunks"] == [] for r in progress["documents"])
        check_skips(bundle_command, json.loads(status.stdout))
        check_summary(bundle_command, json.loads(status.stdout))
        check_summary([str(cli), "port", "import-status", "--state", str(state)], snapshot)
        check_list(bundle_command, json.loads(status.stdout), 0)
        stale = subprocess.run(bundle_command + ["--document", source_docs[0]["id"], "--list-chunks",
                                                 "--page-size", "2", "--cursor", initial_cursor],
                               env=offline_env, text=True, capture_output=True, timeout=10)
        assert stale.returncode == 1 and not stale.stdout and "snapshot mismatch" in stale.stderr
        filtered_command, filtered_cursor, local_status = filtered_cursors[0]
        stale_filtered = subprocess.run(filtered_command + ["--list-chunks", "--chunk-status", local_status,
                                                            "--page-size", "1", "--cursor", filtered_cursor],
                                        env=offline_env, text=True, capture_output=True, timeout=10)
        assert stale_filtered.returncode == 1 and not stale_filtered.stdout and "snapshot mismatch" in stale_filtered.stderr
        print(json.dumps({"cursor_rejected": "checkpoint changed by resume", "exit": stale.returncode}))
        assert (len(target.requests), state.read_bytes()) == before_status
        print(status.stdout.strip())
        missing = Path(tmp) / "absent" / "mapping.json"
        failed = subprocess.run([str(cli), "port", "import-status", "--state", str(missing)],
                                env=offline_env, text=True, capture_output=True, timeout=10)
        assert failed.returncode == 1 and not failed.stdout and not missing.parent.exists()
        print("PASS: bundle-wide chunk details and all status filters match document-scoped rows and page slices, "
              "including null/no matches, exact-byte stale rejection and both pending stages; "
              "bundle-wide skip listing/paging preserves full snapshot and saved reasons, "
              "including interrupted/no-skip/legacy/multi-document exclusions, changed page sizes, null end and stale exact-byte rejection; "
              "pending-stage filters both document arrays by saved intent with global totals unchanged, "
              "including no matches, state-only/bundle/legacy skip/pending-create/complete checkpoints; "
              "invalid stages/conflicts exit2 before missing inputs and nonmatching invalid details exit1; "
              "pending-counts matches saved stage intents without arrays; rejects missing summary, "
              "conflicts and invalid omitted details; summary-only equals full aggregates with both document arrays omitted, "
              "including state-only/interrupted/legacy skips/pending-create/complete; "
              "conflicts exit 2 and invalid omitted detail exits 1 without HTTP or writes; "
              "offline status 1 document / 2 chunks / pending flag; bundle plan 3 / 15, remaining 3 / 13, "
              "unvisited 2 / 12; selected visited/unvisited docs retain global totals and legacy skip reasons; "
              "selected completed/pending flag/pending create/unvisited/skipped chunks retain existing JSON; "
              "per-document chunk lists match independent single-chunk details in source-ID order, "
              "including interrupted, unvisited, legacy skip, pending-create and complete checkpoints; "
              "all four status filters equal full-list subsets before slicing, including no matches; "
              "pages equal matching slices, null at end; v1/v2 cursors reject filter/document/state changes and resume; "
              "invalid page/list scope, combined chunk/list and unknown list document rejected; "
              "unknown/wrong-document chunks and missing chunk scope rejected; "
              "unknown/missing-bundle selectors and invalid state outside selected doc rejected; resumed to 3 documents / 15 chunks; 15 create POSTs; repeat wrote nothing; "
              "mismatched/corrupt bundle and missing state exit 1 without writes or HTTP.")


if __name__ == "__main__":
    main()
