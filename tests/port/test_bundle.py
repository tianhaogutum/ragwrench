import json
from pathlib import Path

import pytest

from ragwrench.port.bundle import load_bundle, write_bundle
from ragwrench.port.common import PortError, atomic_json, content_hash, digest, locked, read_json


def rewrite(bundle, file, mutate):
    rows = [json.loads(line) for line in (bundle / file).read_text().splitlines()]
    mutate(rows)
    blob = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    (bundle / file).write_bytes(blob)
    manifest = read_json(bundle / "manifest.json")
    manifest["sha256"][file] = digest(blob)
    atomic_json(bundle / "manifest.json", manifest)


def test_good_bundle(example):
    m, docs, chunks = load_bundle(example)
    assert m["counts"] == {"documents": 3, "chunks": 15}
    assert len(docs) == 3 and len(chunks) == 15
    assert chunks[3]["enabled"] is False


def test_deterministic_bytes(example, tmp_path):
    m, docs, chunks = load_bundle(example)
    out = tmp_path / "out"
    write_bundle(out, m["source"], m["settings"], list(reversed(docs)), list(reversed(chunks)))
    for file in ("manifest.json", "documents.jsonl", "chunks.jsonl"):
        assert (out / file).read_bytes() == (example / file).read_bytes()


def test_corrupt_file(bundle):
    with (bundle / "chunks.jsonl").open("ab") as out:
        out.write(b"{}\n")
    with pytest.raises(PortError, match="SHA256"):
        load_bundle(bundle)


@pytest.mark.parametrize("case", ["hash", "unknown_document", "duplicate_id", "duplicate_position", "whitespace", "schema", "order"])
def test_semantic_bundle_failures(bundle, case):
    def mutate(rows):
        if case == "hash":
            rows[0]["content_sha256"] = "0" * 64
        elif case == "unknown_document":
            rows[0]["doc_id"] = "missing"
        elif case == "duplicate_id":
            rows[1]["id"] = rows[0]["id"]
        elif case == "duplicate_position":
            rows[1]["position"] = rows[0]["position"]
        elif case == "whitespace":
            rows[0]["content"] = " \n"
            rows[0]["content_sha256"] = content_hash(" \n")
        elif case == "schema":
            rows[0]["enabled"] = "false"
        else:
            rows.reverse()
    rewrite(bundle, "chunks.jsonl", mutate)
    with pytest.raises(PortError):
        load_bundle(bundle)


@pytest.mark.parametrize("case", ["version", "count", "path", "hash"])
def test_manifest_failures(bundle, case):
    m = read_json(bundle / "manifest.json")
    if case == "version":
        m["format"] = "kbbundle/2"
    elif case == "count":
        m["counts"]["chunks"] += 1
    elif case == "path":
        m["sha256"]["../../secret"] = "0" * 64
    else:
        m["sha256"]["chunks.jsonl"] = "bad"
    atomic_json(bundle / "manifest.json", m)
    with pytest.raises(PortError):
        load_bundle(bundle)


def test_symlink_refused(bundle, tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes((bundle / "chunks.jsonl").read_bytes())
    (bundle / "chunks.jsonl").unlink()
    (bundle / "chunks.jsonl").symlink_to(outside)
    with pytest.raises(PortError, match="symlink"):
        load_bundle(bundle)
    with pytest.raises(PortError, match="symlink"):
        atomic_json(bundle / "chunks.jsonl", {})


def test_lock_prevents_concurrent_writer(tmp_path):
    with locked(tmp_path / "state.lock"):
        with pytest.raises(PortError, match="Another"):
            with locked(tmp_path / "state.lock"):
                pass


def test_invalid_jsonl_and_nonfinite(bundle):
    for blob in (b'{"broken":\n', b'{"value":NaN}\n', b'\n'):
        (bundle / "chunks.jsonl").write_bytes(blob)
        m = read_json(bundle / "manifest.json")
        m["sha256"]["chunks.jsonl"] = digest(blob)
        atomic_json(bundle / "manifest.json", m)
        with pytest.raises(PortError):
            load_bundle(bundle)


def test_writer_refuses_unreadable_multibyte_line(example, tmp_path):
    m, docs, chunks = load_bundle(example)
    chunks[0]["content"] = "界" * 800000
    chunks[0]["content_sha256"] = content_hash(chunks[0]["content"])
    with pytest.raises(PortError, match="byte limits"):
        write_bundle(tmp_path / "oversized", m["source"], {}, docs, chunks)
    assert not (tmp_path / "oversized" / "manifest.json").exists()
