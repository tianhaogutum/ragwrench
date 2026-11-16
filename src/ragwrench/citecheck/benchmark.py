"""Replay independent case-level labels against checker and marker-exists baseline."""
from dataclasses import asdict
import hashlib
from importlib.resources import files
import time
import re

from .adapters import adapt
from .checker import CODES, check
from .io import decode, loads, read_bytes
from .markers import parse_markers
from .model import Config, InputError
from .prose import mask_markdown


def marker_exists_baseline(record):
    # Shared syntax/Markdown handling gives this baseline the SAME structural
    # information as the checker, rather than handicapping its marker parser.
    ids = {s.idx for s in record.sources}
    markers = parse_markers(mask_markdown(record.answer))
    return {"CC001"} if any(m.error or any(n not in ids for n in m.indices) for m in markers) else set()


def metrics(expected, predicted):
    result = {}
    for code in CODES:
        tp = sum(code in e and code in p for e, p in zip(expected, predicted))
        fp = sum(code not in e and code in p for e, p in zip(expected, predicted))
        fn = sum(code in e and code not in p for e, p in zip(expected, predicted))
        tn = len(expected) - tp - fp - fn
        result[code] = _metric(tp, fp, fn, tn)
    sums = {k: sum(m[k] for m in result.values()) for k in ("tp", "fp", "fn", "tn")}
    result["micro"] = _metric(**sums)
    return result


def _metric(tp, fp, fn, tn):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 6) if tp + fp else None,
            "recall": round(tp / (tp + fn), 6) if tp + fn else None,
            "f1": round(2 * tp / (2 * tp + fp + fn), 6) if 2 * tp + fp + fn else None}


def run_benchmark(path=None, config=None):
    config = config or Config()
    data = read_bytes(path) if path else files("ragwrench.citecheck").joinpath("data/benchmark.jsonl").read_bytes()
    cases = []
    seen = set()
    for n, line in enumerate(decode(data).splitlines(), 1):
        if not line.strip():
            continue
        raw = loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"]:
            raise InputError(f"benchmark line {n} needs a non-empty string id")
        if re.search(r"[\ud800-\udfff]", raw["id"]):
            raise InputError(f"benchmark line {n}: id contains an unpaired Unicode surrogate")
        if len(raw["id"]) > 128 or raw["id"] in seen:
            raise InputError(f"benchmark line {n}: duplicate or overlong id")
        seen.add(raw["id"])
        labels = raw.get("expected")
        if not isinstance(labels, list) or any(not isinstance(c, str) or c not in CODES for c in labels) or len(labels) != len(set(labels)):
            raise InputError(f"benchmark line {n} needs unique expected CC001–CC008 labels")
        record = adapt(raw.get("record"), raw.get("format", "generic"))
        cases.append((raw["id"], record, set(labels)))
        if len(cases) > 1000:
            raise InputError("benchmark supports at most 1000 cases")
    if not cases:
        raise InputError("benchmark must contain labelled cases")
    expected, predicted, baseline, results = [], [], [], []
    start = time.perf_counter()
    for case_id, record, labels in cases:
        report = check(record, config)
        actual = {f["code"] for f in report["findings"]}
        predicted.append(actual)
        expected.append(labels)
        results.append({"id": case_id, "expected": sorted(labels), "predicted": sorted(actual),
                        "false_positives": sorted(actual - labels), "false_negatives": sorted(labels - actual)})
    checker_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for _, record, _ in cases:
        baseline.append(marker_exists_baseline(record))
    baseline_ms = (time.perf_counter() - start) * 1000
    for result, base in zip(results, baseline):
        result["baseline_predicted"] = sorted(base)
    return {"schema_version": 1, "kind": "citecheck-benchmark", "case_count": len(cases),
            "dataset_sha256": hashlib.sha256(data).hexdigest(), "config": asdict(config),
            "evaluation_unit": "presence of each diagnostic code per case (includes informational CC007/CC008)",
            "metrics": metrics(expected, predicted), "baseline_metrics": metrics(expected, baseline),
            "elapsed_ms": {"checker": round(checker_ms, 3), "marker_exists_baseline": round(baseline_ms, 3)},
            "notes": ["Synthetic labels are regression targets, not evidence of real-world accuracy.",
                      "Precision is null when no positives are predicted; recall is null when no positives are labelled.",
                      "Timings are actual single-process measurements and may vary between runs.",
                      "Baseline shares marker parsing and Markdown handling, but only predicts CC001."],
            "cases": results}
