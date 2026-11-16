"""Audit one chunk export (or a directory of them) with the ``ragwrench lint`` engine.

Rules come from :func:`ragwrench.lint.lint.lint`; the repair preview is
:func:`ragwrench.lint.fix.fix` run in memory, so "auto-fixable" counts are measured,
not guessed. This module only adds categories, ranking and fix commands.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
import shlex

from ..lint.fix import fix
from ..lint.io import load_chunks, read_text
from ..lint.lint import lint
from ..lint.model import RULES, InputError
from ..lint.report import failed

# rule -> (category label, why it hurts retrieval)
CATEGORIES = {
    "CL001": ("Broken code fences", "half a code block is retrieved"),
    "CL002": ("Table rows without a header", "cells are retrieved without column names"),
    "CL003": ("Orphan headings", "a heading is cut off from its text"),
    "CL004": ("Mid-sentence cuts", "a sentence is split across chunks"),
    "CL005": ("Size outliers", "too small to mean anything, or too big to embed"),
    "CL006": ("Duplicates and heavy overlap", "the same text crowds the top-k"),
    "CL007": ("Source coverage loss", "source text that no chunk contains"),
    "CL008": ("Split lists", "list items are cut across chunks"),
}
POINTS = {"error": 3, "warning": 1, "note": 0}
OPERATIONS = {  # lint fix change log operation -> (singular, plural)
    "rejoin_fence": ("{} code block rejoined", "{} code blocks rejoined"),
    "rejoin_table": ("{} table rejoined", "{} tables rejoined"),
    "merge_heading_forward": ("{} heading merged", "{} headings merged"),
    "drop_duplicate": ("{} duplicate dropped", "{} duplicates dropped"),
    "split_structure": ("{} oversized chunk split", "{} oversized chunks split"),
}
BOUNDARY_RULES = ("CL004", "CL008", "CL005", "CL006", "CL003", "CL001")
SKIP_SUFFIXES = (".changes.json",)  # ragwrench lint fix change logs, not chunk exports


@dataclass
class FileAudit:
    path: Path
    source: Path | None
    chunks: list
    report: object
    fixed_by_rule: dict | None  # findings left per rule after an in-memory `lint fix`
    changes: list
    notes: list = field(default_factory=list)
    fix_error: str | None = None
    gaps: list = field(default_factory=list)  # source spans no chunk covers, with line numbers

    @property
    def by_rule(self) -> dict:
        return dict(Counter(issue.rule for issue in self.report.issues))

    def fixable(self) -> dict:
        if self.fixed_by_rule is None:
            return {}
        return {rule: n - self.fixed_by_rule.get(rule, 0) for rule, n in self.by_rule.items()
                if n > self.fixed_by_rule.get(rule, 0)}


def find_inputs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise InputError(f"{target}: no such file or directory")
    found = [path for path in sorted(target.rglob("*")) if path.is_file()
             and path.suffix in (".jsonl", ".json") and not path.name.endswith(SKIP_SUFFIXES)
             and not any(part.startswith(".") for part in path.relative_to(target).parts)]
    if not found:
        raise InputError(f"{target}: no .jsonl or .json chunk files found")
    return found


def _pair_budget(chunks, config):
    """Skip CL006 (pairwise) instead of failing when an export is too large for its budget."""
    if "CL006" in config.disable:
        return config, []
    pairs = len(chunks) * (len(chunks) - 1) // 2
    reason = None
    if pairs > config.max_pair_checks:
        reason = f"{len(chunks):,} chunks need {pairs:,} pair checks (max_pair_checks={config.max_pair_checks:,})"
    elif any(len(c.text) > 200000 for c in chunks) or sum(len(c.text) for c in chunks) > 1000000:
        reason = "the export exceeds the duplicate check's size limits"
    if reason is None:
        return config, []
    return replace(config, disable=sorted({*config.disable, "CL006"})), [
        f"CL006 duplicate check skipped: {reason}; raise max_pair_checks in chunklint.toml to run it"]


def audit_file(path: Path, source: Path | None, config, adapter: str = "auto") -> FileAudit:
    chunks = load_chunks(path, adapter)
    text = read_text(source) if source is not None else None
    config, notes = _pair_budget(chunks, config)
    report = lint(chunks, text, config)
    fixed_by_rule, changes, fix_error = None, [], None
    if report.issues:
        try:
            result = fix(chunks, text, config)
            fixed_by_rule, changes = dict(Counter(i.rule for i in result.after.issues)), result.changes
        except InputError as exc:
            fix_error = str(exc)
    else:
        fixed_by_rule = {}
    gaps = [_gap(text, gap) for gap in report.stats["source"]["gaps"]] if text is not None else []
    return FileAudit(path, source, chunks, report, fixed_by_rule, changes, notes, fix_error, gaps)


def _gap(text: str, gap: dict) -> dict:
    start, end = gap["start_char"], gap["end_char"]
    segment = text[start:end]
    first, last = start + len(segment) - len(segment.lstrip()), start + len(segment.rstrip())
    return {**gap, "line_start": text.count("\n", 0, first) + 1, "line_end": text.count("\n", 0, last) + 1,
            "preview": preview(text[first:last])}


def preview(text: str, width: int = 56) -> str:
    flat = "".join("↵" if ch == "\n" else (" " if ch.isspace() or not ch.isprintable() else ch)
                   for ch in text.strip())
    return flat if len(flat) <= width else flat[:width - 1].rstrip() + "…"


def worst_chunks(audits: list[FileAudit], limit: int) -> list[dict]:
    rows = []
    for order, audit in enumerate(audits):
        texts = {c.id: c.text for c in audit.chunks}
        position = {c.id: i for i, c in enumerate(audit.chunks)}
        grouped: dict[str, list] = {}
        for issue in audit.report.issues:
            if issue.chunk_id is not None:
                grouped.setdefault(issue.chunk_id, []).append(issue)
        for chunk_id, issues in grouped.items():
            rules = sorted({i.rule for i in issues})
            rows.append({"file": str(audit.path), "chunk_id": chunk_id, "rules": rules,
                         "findings": len(issues), "score": sum(POINTS[i.severity] for i in issues),
                         "preview": preview(texts[chunk_id]), "_key": (order, position[chunk_id])})
    rows.sort(key=lambda row: (-row["score"], -row["findings"], row["_key"]))
    return [{k: v for k, v in row.items() if k != "_key"} for row in rows[:limit]]


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def fixes(audit: FileAudit, flags: list[str]) -> list[dict]:
    """Ranked, concrete next steps for one file. Each has a command that exists today."""
    q = shlex.quote
    src = ["--source", str(audit.source)] if audit.source else []
    by_rule, total, out = audit.by_rule, len(audit.report.issues), []
    if audit.changes:
        ops = Counter(change["operation"] for change in audit.changes)
        resolved = sum(audit.fixable().values())
        target = audit.path.with_name(audit.path.stem + ".fixed.jsonl")
        out.append({
            "kind": "lint-fix", "resolves": resolved, "path": str(audit.path),
            "title": "Repair in place (" + _join([OPERATIONS[op][ops[op] != 1].format(ops[op])
                                                  for op in OPERATIONS if ops.get(op)]) + ")",
            "detail": f"resolves {resolved} of {total} findings (measured by a dry run)"
                      + ("; source coverage preserved" if audit.source else ""),
            "commands": [" ".join(q(a) for a in ["ragwrench", "lint", "fix", str(audit.path), *src, *flags,
                                                 "-o", str(target)])]})
    elif audit.fix_error:
        out.append({"kind": "lint-fix", "resolves": 0, "path": str(audit.path),
                    "title": "Automatic repair is not safe for this export",
                    "detail": audit.fix_error, "commands": []})
    left = audit.fixed_by_rule if audit.fixed_by_rule is not None else by_rule
    check = " ".join(q(a) for a in ["ragwrench", "lint", "check", str(audit.path), *src, *flags])
    if left.get("CL007"):
        cov = audit.report.stats["source"]
        out.append({
            "kind": "coverage", "resolves": left["CL007"], "path": str(audit.path),
            "title": f"Re-ingest the source text no chunk contains ({cov['missing_chars']:,} of "
                     f"{cov['total_chars']:,} characters, {_plural(len(audit.gaps), 'gap')})",
            "detail": "a loader or filter dropped it; gaps are listed above",
            "commands": [check + " --format json"]})
    if by_rule.get("CL002"):
        source = str(audit.source) if audit.source else "SOURCE.md"
        out.append({
            "kind": "table-chunk", "resolves": left.get("CL002", 0), "path": str(audit.path),
            "title": "Stop splitting tables: re-chunk so every table chunk repeats its header row "
                     f"({_plural(by_rule['CL002'], 'fragment')} without one)",
            "detail": "table-aware chunks carry the heading path, caption and header row",
            "commands": [" ".join(q(a) for a in ["ragwrench", "table-chunk", "split", source,
                                                 "--max-tokens", "256", "-o",
                                                 str(audit.path.with_name(audit.path.stem + ".tables.jsonl"))])]})
    boundary = {rule: left[rule] for rule in BOUNDARY_RULES if left.get(rule)}
    if boundary:
        names = [CATEGORIES[rule][0].lower() for rule in boundary]
        out.append({
            "kind": "splitter", "resolves": sum(boundary.values()), "path": str(audit.path),
            "title": f"{_plural(sum(boundary.values()), 'finding')} need a splitter change ({_join(names)})",
            "detail": "cut at paragraph and list boundaries; inspect each one by line",
            "commands": [check]})
    return out


def summarize(audits: list[FileAudit], fail_on: str) -> dict:
    issues = [issue for audit in audits for issue in audit.report.issues]
    by_rule = Counter(issue.rule for issue in issues)
    fixable = Counter()
    for audit in audits:
        fixable.update(audit.fixable())
    return {
        "files": len(audits),
        "chunks": sum(len(a.chunks) for a in audits),
        "characters": sum(a.report.stats["characters"] for a in audits),
        "chunks_with_findings": sum(len({i.chunk_id for i in a.report.issues if i.chunk_id}) for a in audits),
        "findings": len(issues),
        "errors": sum(i.severity == "error" for i in issues),
        "warnings": sum(i.severity == "warning" for i in issues),
        "auto_fixable": sum(fixable.values()),
        "categories": [{"rule": rule, "name": CATEGORIES[rule][0], "severity": RULES[rule][0],
                        "why": CATEGORIES[rule][1], "found": by_rule.get(rule, 0),
                        "auto_fixable": fixable.get(rule, 0)} for rule in RULES],
        "fail_on": fail_on,
        "failed": any(failed(a.report, fail_on) for a in audits),
    }
