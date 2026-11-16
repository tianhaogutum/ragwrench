from __future__ import annotations

import math
import uuid
from dataclasses import replace
from pathlib import Path, PurePath

import pytest

from ragwrench.migrate.model import (DISTANCES, ENVELOPE, ID_NAMESPACE, ParityError, Record,
    StoreURI, decode_metadata, dumps, encode_metadata, physical_id, validate_record)


@pytest.mark.parametrize("rid,expected", [
    ("0", 0), ("42", 42), (str(2**64-1), 2**64-1),
    ("550E8400-E29B-41D4-A716-446655440000", "550e8400-e29b-41d4-a716-446655440000"),
])
def test_kept_ids(rid, expected):
    assert physical_id(rid) == expected


@pytest.mark.parametrize("rid", ["guide.md::12", "笔记", "0007", "-1", str(2**64), "../doc"])
def test_uuid5_ids(rid):
    assert physical_id(rid) == str(uuid.uuid5(ID_NAMESPACE, rid))
    assert physical_id(rid) == physical_id(rid)


@pytest.mark.parametrize("kind", ["chroma", "qdrant"])
@pytest.mark.parametrize("metadata", [None, {}, {"scalar": 1, "bool": True, "float": 1.25, "empty": ""},
    {"nested": {"path": [1, None, {"x": "é"}]}, "null": None, "list": [True, 2], "big": 2**80},
    {"document": "metadata", "__id__": "user", ENVELOPE: "user envelope", "vp_m_646f63756d656e74": 1, "": []}])
def test_metadata_codec(kind, metadata):
    r = Record("doc", [1., 2.], metadata, "text", 42)
    payload = encode_metadata(r, kind)
    restored, qid = decode_metadata(payload, kind)
    assert dumps(restored) == dumps(metadata)
    assert qid == 42
    if kind == "chroma":
        assert all(type(v) in {str, int, bool, float} for v in payload.values())
    else:
        assert payload["__id__"] == "doc"
        assert payload["document"] == "text"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"x": {1, 2}}, {"x": (1, 2)}, {1: "bad"}])
def test_unsupported_metadata(value):
    with pytest.raises(ParityError):
        validate_record(Record("x", [1.], {"v": value}), 1, "l2")


@pytest.mark.parametrize("vector,distance", [([], "l2"), ([math.nan], "l2"), ([math.inf], "ip"), ([0.], "cosine"), ([True], "l2")])
def test_bad_vectors(vector, distance):
    with pytest.raises(ParityError):
        validate_record(Record("x", vector, None), 1, distance)


@pytest.mark.parametrize("value", ["http://host#docs", "qdrant:/tmp/x#../x", "chroma:/tmp/x#foo/bar", "qdrant:/tmp/x#foo..bar",
    "chroma:#docs", "chroma:/tmp/x#docs#x", "qdrant:https://host#docs", "chroma:/tmp/x#a"])
def test_uri_boundary(value):
    with pytest.raises(ParityError):
        StoreURI.parse(value)


def test_local_uri(tmp_path):
    uri = StoreURI.parse(f"chroma:{tmp_path}/db#my-docs")
    assert uri.path == tmp_path / "db"
    assert uri.collection == "my-docs"


@pytest.mark.parametrize("kind", ["chroma", "qdrant"])
@pytest.mark.parametrize("name", ["a0Z", "a.b_c-9", "A" + "x" * 126 + "9"])
def test_direct_and_parsed_uri_accept_same_collection_names(tmp_path, kind, name):
    direct = StoreURI(kind, tmp_path / "missing", name)
    assert direct == StoreURI.parse(f"{kind}:{direct.path}#{name}")
    assert not direct.path.exists()


@pytest.mark.parametrize("kind", ["chroma", "qdrant"])
@pytest.mark.parametrize("name", [
    "", "ab", "a" * 129, ".docs", "docs-", "foo..bar", "../outside",
    "foo/bar", "foo\\bar", "/tmp/outside", "a b", "döcs", "docs\n", "docs#x", "ab\x00c",
])
def test_direct_and_parsed_uri_reject_same_collection_names(tmp_path, kind, name):
    with pytest.raises(ParityError, match="Collection name"):
        StoreURI(kind, tmp_path / "missing", name)
    with pytest.raises(ParityError):
        StoreURI.parse(f"{kind}:{tmp_path / 'missing'}#{name}")
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("name", [None, 123, b"docs", ["docs"]])
def test_direct_uri_requires_string_collection_name(tmp_path, name):
    with pytest.raises(ParityError, match="Collection name"):
        StoreURI("qdrant", tmp_path / "missing", name)


@pytest.mark.parametrize("kind", ["", "sqlite", "Qdrant", None, ["qdrant"]])
def test_direct_uri_requires_supported_backend(tmp_path, kind):
    with pytest.raises(ParityError, match="backend"):
        StoreURI(kind, tmp_path / "missing", "docs")


@pytest.mark.parametrize("kind", ["chroma", "qdrant"])
@pytest.mark.parametrize("construction", ["direct", "replace"])
@pytest.mark.parametrize("path", [Path("bad\x00path"), Path("/bad\x00parent/store"), Path("bad\x00parent/../store")])
def test_nul_path_rejected_at_construction(kind, construction, path):
    original = StoreURI(kind, Path("valid-local-store"), "docs")
    with pytest.raises(ParityError, match="local path.*NUL"):
        if construction == "direct":
            StoreURI(kind, path, "docs")
        else:
            replace(original, path=path)


@pytest.mark.parametrize("payload", [{ENVELOPE: "not json"}, {ENVELOPE: {"version": 5}},
    {ENVELOPE: {"version": 1, "keys": {"a": "a"}, "json": [], "metadata_none": False}},
    {ENVELOPE: {"version": 1, "keys": {}, "json": [], "metadata_none": False}, "unmapped": "loss"}])
def test_invalid_envelope(payload):
    with pytest.raises(ParityError):
        decode_metadata(payload, "qdrant")


def test_distance_contract():
    assert DISTANCES == {"l2": "Euclid", "cosine": "Cosine", "ip": "Dot"}


def test_namespace_and_long_id():
    # Frozen mapping example, independent of the production namespace constant.
    assert physical_id("doc") == "a4345041-1b57-581a-875e-2b79d5cac285"
    assert uuid.UUID(physical_id("9" * 5000)).version == 5


def test_metadata_depth_limit():
    value = {}
    for _ in range(34):
        value = {"nested": value}
    with pytest.raises(ParityError, match="32"):
        validate_record(Record("x", [1.], value), 1, "l2")


@pytest.mark.parametrize("record", [Record("\ud800", [1.], None), Record("x", [1.], {"\ud800": "v"}),
                                     Record("x", [1.], {"x": "\ud800"}), Record("x", [1.], None, "\ud800")])
def test_invalid_unicode(record):
    with pytest.raises(ParityError, match="Unicode"):
        validate_record(record, 1, "l2")


class UnsupportedPathLike:
    def __fspath__(self):
        pytest.fail("Unsupported path input was coerced via __fspath__")

    def __str__(self):
        pytest.fail("Unsupported path input was coerced via __str__")


@pytest.mark.parametrize("kind", ["chroma", "qdrant"])
@pytest.mark.parametrize("construction", ["direct", "replace"])
@pytest.mark.parametrize("path", [
    "./store", b"./store", None, 123, ["store"],
    pytest.param(PurePath("store"), id="pure-path"),
    pytest.param(UnsupportedPathLike(), id="custom-pathlike"),
])
def test_non_path_input_rejected_without_coercion(kind, construction, path):
    original = StoreURI(kind, Path("valid-local-store"), "docs")
    with pytest.raises(ParityError, match=r"^Store path must be a local pathlib\.Path object$"):
        if construction == "direct":
            StoreURI(kind, path, "docs")
        else:
            replace(original, path=path)
