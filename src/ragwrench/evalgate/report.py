"""Deterministic reports; JUnit failures correspond to actual gate violations."""
import csv
import html
import io
import json
import re
import xml.etree.ElementTree as ET

from .formats import InputError


def render(report, fmt):
    if report.get("mode") == "preflight-only" and fmt != "json":
        raise InputError("preflight-only supports only JSON")
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if fmt == "junit":
        return _junit(report)
    if report["command"] in {"compare-many", "compare-many-saved"}:
        return _many(report, fmt)
    markdown = fmt == "markdown"
    if report["command"] == "eval":
        rows = [f"Queries: {report['query_count']} (missing: {len(report['missing_queries'])}, ignored: {len(report['ignored_queries'])})"]
        if markdown:
            rows += ["", "| Metric | Mean |", "|---|---:|"]
        for key, value in report["means"].items():
            rows.append(f"| {key} | {value:.6f} |" if markdown else f"{key:18} {value:.6f}")
        return "\n".join(rows) + "\n"
    rows = [f"{'PASS' if report['passed'] else 'FAIL'}: {report['query_count']} paired queries; {report['gain']} gain"]
    rows.append(f"Missing baseline/candidate: {len(report['baseline_missing_queries'])}/{len(report['candidate_missing_queries'])}; "
                f"ignored baseline/candidate: {len(report['baseline_ignored_queries'])}/{len(report['candidate_ignored_queries'])}")
    if markdown:
        rows += ["", "| Metric | Baseline | Candidate | Delta | p (two-sided) | CI | W/T/L |", "|---|---:|---:|---:|---:|---|---|"]
    else:
        rows += ["", "metric             baseline candidate delta      p-value    CI                  W/T/L"]
    for key, row in report["metrics"].items():
        lo, hi = row["ci"]
        wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
        if markdown:
            rows.append(f"| {key} | {row['baseline']:.6f} | {row['candidate']:.6f} | {row['delta']:+.6f} | {row['p_value']:.6g} | [{lo:.6f}, {hi:.6f}] | {wtl} |")
        else:
            rows.append(f"{key:18} {row['baseline']:.6f} {row['candidate']:.6f} {row['delta']:+.6f} {row['p_value']:.6g} [{lo:.6f}, {hi:.6f}] {wtl}")
    rows.append("")
    for key, row in report["metrics"].items():
        rows.append(f"  {key}: test={row['test']}; {row['confidence']:.1%} CI: {row['ci_method']}; seed={row['seed']}; samples={row['samples']}")
        if row["warning"]:
            rows.append(f"  Warning: {row['warning']}")
        if "adjusted_p_value" in row:
            rows.append(f"  {key}: adjusted p={row['adjusted_p_value']:.6g}; significant={row['significant']}")
    for key, row in report["metrics"].items():
        if row["top_regressed"]:
            rows += ["", f"Top regressions ({key}):"]
            for item in row["top_regressed"]:
                qid = re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", html.escape(item["query_id"])) if markdown else item["query_id"]
                rows.append(f"  {qid}: {item['baseline']:.6f} -> {item['candidate']:.6f} ({item['delta']:+.6f})")
    for failure in report["violations"]:
        rows.append(f"FAIL: {failure['metric']} dropped by {failure['drop']:.6g} (> {failure['max_drop']:.6g})")
    return "\n".join(rows) + "\n"


def _many(report, fmt):
    if fmt == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["candidate_name", "query_id", "metric", "baseline", "candidate", "delta", "outcome"])
        for name, candidate in report["candidates"].items():
            for qid, metrics in candidate["per_query"].items():
                for metric, row in metrics.items():
                    writer.writerow([name, qid, metric, row["baseline"], row["candidate"], row["delta"], row["outcome"]])
        return stream.getvalue()
    policy = report["policy"]
    rows = [f"{'PASS' if report['passed'] else 'FAIL'}: {len(report['candidates'])} candidates against one fixed baseline",
            f"Correction: {policy['correction']}; family: {policy['family_size']} candidate-metric pairs; alpha={policy['alpha']:g}",
            "CI adjustment: none; absolute mean-drop thresholds; significance does not affect gates."]
    for name, candidate in report["candidates"].items():
        rows.extend(["", f"Candidate: {name}", render(candidate, fmt).rstrip("\n")])
    return "\n".join(rows) + "\n"


def _junit(report):
    if report["command"] in {"compare-many", "compare-many-saved"}:
        suite = ET.Element("testsuite", name=f"retgate.{report['command']}",
                           tests=str(report["policy"]["family_size"]),
                           failures=str(len(report["violations"])), errors="0")
        for name, candidate in report["candidates"].items():
            for metric, row in candidate["metrics"].items():
                case = ET.SubElement(suite, "testcase", classname=f"retgate.{report['command']}", name=f"{name}/{metric}")
                _case_result(case, metric, row, {"policy": report["policy"], **_comparison_details(candidate, row)})
        return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"
    comparing = report["command"] in {"compare", "compare-saved"}
    rows = report["metrics"] if comparing else report["means"]
    failures = len(report["violations"]) if comparing else 0
    suite = ET.Element("testsuite", name="retgate", tests=str(len(rows)), failures=str(failures), errors="0")
    for metric, row in rows.items():
        case = ET.SubElement(suite, "testcase", classname="retgate.compare" if comparing else "retgate.eval", name=metric)
        if comparing:
            _case_result(case, metric, row, _comparison_details(report, row))
        else:
            ET.SubElement(case, "system-out").text = json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def _comparison_details(report, row):
    if report["command"] != "compare-saved":
        return row
    return {**row, "evaluation_manifests": report["evaluation_manifests"],
            **{k: report[k] for k in ("query_count", "baseline_missing_queries", "candidate_missing_queries",
                                     "baseline_ignored_queries", "candidate_ignored_queries")}}


def _case_result(case, metric, row, details):
    if not row["passed"]:
        message = f"{metric} dropped by {-row['delta']:.12g} (> {row['max_drop']:.12g})"
        ET.SubElement(case, "failure", message=message, type="RetrievalRegression").text = message
    ET.SubElement(case, "system-out").text = json.dumps(details, ensure_ascii=False, allow_nan=False, sort_keys=True)
