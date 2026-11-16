"""Versioned, snapshot-bound plans for the existing conservative repair engine."""
from dataclasses import asdict
from difflib import SequenceMatcher
import hashlib
import json

from . import __version__
from .fix import fix
from .io import _decode, parse_chunks, read_text
from .model import Config, InputError
from .plan_validation import validate_structure

ADAPTERS = ("auto", "generic", "langchain", "llamaindex")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def encode_records(records):
    """Canonical generic JSONL: sorted keys, compact JSON, literal Unicode, LF."""
    return "".join(_json(record) + "\n" for record in records).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _snapshot(raw):
    return {"sha256": _sha(raw), "bytes": len(raw)}


def create_plan(input_raw, source_raw=None, adapter="auto", config=None):
    if adapter not in ADAPTERS:
        raise InputError("unsupported plan adapter")
    config = (config or Config()).validate()
    chunks = parse_chunks(input_raw.decode("utf-8-sig"), adapter)
    source = source_raw.decode("utf-8-sig") if source_raw is not None else None
    result = fix(chunks, source, config)
    before = [chunk.record() for chunk in chunks]
    after = [chunk.record() for chunk in result.chunks]
    left, right = [_json(record) for record in before], [_json(record) for record in after]
    offsets = [0]
    for line in left:
        offsets.append(offsets[-1] + len(line.encode("utf-8")) + 1)
    edits = []
    # IDs are unique within each side, so equal records are unambiguous anchors.
    for tag, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if tag != "equal":
            edits.append({"start_chunk": a, "end_chunk": b,
                          "start_byte": offsets[a], "end_byte": offsets[b],
                          "before_sha256": _sha(encode_records(before[a:b])),
                          "replacement": after[c:d]})
    plan = {"format": "chunklint.fix-plan", "version": 1, "tool_version": __version__,
            "coordinates": "canonical-generic-jsonl-utf8-v1",
            "input": {**_snapshot(input_raw), "adapter": adapter},
            "source": _snapshot(source_raw) if source_raw is not None else None,
            "config": asdict(config), "edits": edits,
            "output_sha256": _sha(encode_records(after))}
    return plan, result


def load_plan(path):
    try:
        plan = _decode(read_text(path))
    except (ValueError, RecursionError) as exc:
        raise InputError(f"{path}: invalid fix plan: {exc}") from exc
    return validate_structure(plan)


def validate_plan(plan, input_raw, source_raw=None):
    """Reject the entire plan before any output; return the exact byte-spliced result.

    Recompute the deterministic repair to validate all edits and conservation guards,
    rather than trusting hand-edited replacement records or stored invariant claims.
    Ambient config is deliberately ignored: the plan captures the full configuration.
    """
    baseline = plan.get("input")
    if not isinstance(baseline, dict) or {k: baseline.get(k) for k in ("sha256", "bytes")} != _snapshot(input_raw):
        raise InputError("input snapshot differs from fix plan; regenerate the plan")
    source_snapshot = _snapshot(source_raw) if source_raw is not None else None
    if plan.get("source") != source_snapshot:
        raise InputError("source snapshot differs from fix plan; supply the original source or regenerate the plan")
    settings = plan.get("config")
    if not isinstance(settings, dict) or set(settings) != set(Config.__dataclass_fields__):
        raise InputError("invalid fix plan configuration")
    config = Config(**settings).validate()
    expected, result = create_plan(input_raw, source_raw, baseline.get("adapter"), config)
    # Canonical JSON comparison also rejects booleans/floats in integer coordinates.
    if _json(plan) != _json(expected):
        raise InputError("fix plan edits do not match the deterministic repair; regenerate the plan")
    chunks = parse_chunks(input_raw.decode("utf-8-sig"), baseline["adapter"])
    original = encode_records([chunk.record() for chunk in chunks])
    parts, cursor = [], 0
    for edit in plan["edits"]:
        start, end = edit["start_byte"], edit["end_byte"]
        parts.extend((original[cursor:start], encode_records(edit["replacement"])))
        cursor = end
    parts.append(original[cursor:])
    output = b"".join(parts)
    if _sha(output) != plan["output_sha256"]:
        raise InputError("fix plan output hash mismatch; no files written")
    return output.decode("utf-8"), result


def preview_text(plan, input_raw):
    chunks = parse_chunks(input_raw.decode("utf-8-sig"), plan["input"]["adapter"])
    records = [chunk.record() for chunk in chunks]
    lines = [f"Input SHA256: {plan['input']['sha256']}",
             f"Source SHA256: {plan['source']['sha256'] if plan['source'] else 'not supplied'}",
             f"{len(plan['edits'])} edits; coordinates: {plan['coordinates']}"]
    for edit in plan["edits"]:
        lines.append(f"chunks [{edit['start_chunk']}, {edit['end_chunk']}), "
                     f"UTF-8 bytes [{edit['start_byte']}, {edit['end_byte']}):")
        lines.extend("- " + _json(record) for record in records[edit["start_chunk"]:edit["end_chunk"]])
        lines.extend("+ " + _json(record) for record in edit["replacement"])
    return "\n".join(lines) + "\n"
