from collections import Counter
from dataclasses import dataclass

from .align import align_chunks, coverage, normalized
from .markdown import HEADING, LIST, cells, scan, table_runs, trailing_heading, valid_table
from .model import Config, InputError, Issue


@dataclass
class Report:
    issues: list
    alignments: list
    stats: dict

    def record(self):
        return {"version": "0.1.0", "issues": [i.record() for i in self.issues],
                "alignments": [a.record() for a in self.alignments], "stats": self.stats}


def suffix_overlap(left, right):
    """Linear-time longest suffix/prefix (KMP); sentinel cannot collide with text."""
    sequence = list(right) + [None] + list(left[-len(right):])
    pi = [0] * len(sequence)
    for i in range(1, len(sequence)):
        j = pi[i - 1]
        while j and sequence[i] != sequence[j]:
            j = pi[j - 1]
        if sequence[i] == sequence[j]:
            j += 1
        pi[i] = j
    return pi[-1] if pi else 0


def _prose(line):
    return bool(line.strip()) and not (HEADING.match(line) or LIST.match(line)
                                          or cells(line) or line.startswith(("    ", "\t", ">")))


def lint(chunks, source=None, config=None):
    config = (config or Config()).validate()
    if not chunks:
        raise InputError("empty chunk set")
    issues, structures = [], [scan(c.text) for c in chunks]

    def add(rule, severity, chunk, message, line=1, related=None, details=None):
        if rule not in config.disable:
            issues.append(Issue(rule, severity, chunk.id if chunk else None, message,
                                line, related or [], details or {}))

    for i, (chunk, structure) in enumerate(zip(chunks, structures)):
        if structure.unclosed:
            fence = structure.unclosed
            add("CL001", "error", chunk,
                f"{fence.length}-character {fence.char!r} fence is not closed in this chunk",
                fence.line, [chunks[i + 1].id] if i + 1 < len(chunks) else [])
        for start, end in table_runs(structure):
            if not valid_table(structure.lines[start:end]):
                add("CL002", "warning", chunk, "pipe-table fragment has no header + delimiter row", start + 1)
        heading = trailing_heading(chunk.text)
        if heading:
            add("CL003", "warning", chunk, "chunk ends with a heading without following content", heading,
                [chunks[i + 1].id] if i + 1 < len(chunks) else [])
        size = len(chunk.text)
        if size < config.min_chars or size > config.max_chars:
            add("CL005", "warning", chunk,
                f"size {size} characters is outside [{config.min_chars}, {config.max_chars}]",
                details={"chars": size})
        if i + 1 < len(chunks):
            following = structures[i + 1]
            left_nonempty = [j for j, line in enumerate(structure.lines) if line.strip()]
            right_nonempty = [j for j, line in enumerate(following.lines) if line.strip()]
            if not left_nonempty or not right_nonempty:
                continue
            li, ri = left_nonempty[-1], right_nonempty[0]
            left, right = structure.lines[li], following.lines[ri]
            if structure.outside[li] and following.outside[ri] and _prose(left) and _prose(right):
                # Closing quotes/brackets follow sentence punctuation in both scripts.
                tail = left.rstrip().rstrip('"\'”’」』）)]}')
                first = right.lstrip()[0]
                if (tail and tail[-1] not in ".!?。！？…:：;；" and
                        (first.islower() or "\u3400" <= first <= "\u9fff")):
                    add("CL004", "warning", chunk, "sentence appears to continue in the next chunk", li + 1,
                        [chunks[i + 1].id])
            if structure.outside[li] and following.outside[ri]:
                list_at_end = bool(LIST.match(left))
                # Also catch a multiline final list item with indented continuation.
                if not list_at_end and left.startswith(("  ", "\t")):
                    list_at_end = any(LIST.match(s) and out for s, out in zip(structure.lines, structure.outside))
                if list_at_end and (LIST.match(right) or right.startswith(("  ", "\t"))):
                    add("CL008", "warning", chunk, "list continues across the chunk boundary", li + 1,
                        [chunks[i + 1].id])

    if "CL006" not in config.disable:
        count = len(chunks) * (len(chunks) - 1) // 2
        if count > config.max_pair_checks:
            raise InputError(f"CL006 needs {count} pairs, exceeds max_pair_checks={config.max_pair_checks}; "
                             "raise the explicit budget or disable CL006")
        if any(len(c.text) > 200000 for c in chunks):
            raise InputError("CL006 supports at most 200000 characters per chunk; disable CL006 for larger chunks")
        if sum(len(c.text) for c in chunks) > 1000000:
            raise InputError("CL006 supports at most 1000000 total characters; disable CL006 for larger sets")
        normals = [normalized(c.text) for c in chunks]
        shingles = [set(t[j:j + 5] for j in range(len(t) - 4)) for t in normals]
        for i, chunk in enumerate(chunks):
            for j in range(i):
                prior = chunks[j]
                if chunk.text == prior.text:
                    add("CL006", "warning", chunk, "exact duplicate chunk", related=[prior.id],
                        details={"kind": "exact", "similarity": 1.0})
                elif min(len(normals[i]), len(normals[j])) >= 20:
                    a, b = shingles[i], shingles[j]
                    similarity = len(a & b) / len(a | b) if a or b else 0
                    if similarity >= config.near_duplicate:
                        add("CL006", "warning", chunk, f"near duplicate (5-gram Jaccard {similarity:.3f})",
                            related=[prior.id], details={"kind": "near", "similarity": similarity})
            if i:
                overlap = suffix_overlap(normals[i - 1], normals[i])
                fraction = overlap / min(len(normals[i - 1]), len(normals[i]))
                if overlap >= 20 and fraction >= config.max_overlap:
                    add("CL006", "warning", chunk, f"excessive boundary overlap ({fraction:.1%})",
                        related=[chunks[i - 1].id], details={"kind": "overlap", "chars": overlap,
                                                            "fraction": fraction})

    alignments = align_chunks(chunks, source) if source is not None else []
    stats = {"chunks": len(chunks), "characters": sum(len(c.text) for c in chunks)}
    if source is not None:
        cov = coverage(source, alignments)
        failed = sum(a.status in ("unaligned", "partial") for a in alignments)
        stats.update({"source": cov, "alignment_failure_rate": failed / len(chunks),
                      "ambiguous_chunks": sum(a.candidates > 1 for a in alignments),
                      "fragmented_chunks": sum(a.status == "fragmented" for a in alignments)})
        for chunk, alignment in zip(chunks, alignments):
            if alignment.status in ("unaligned", "partial"):
                add("CL007", "error", chunk,
                    f"source alignment {alignment.status}: {alignment.matched_chars}/{alignment.total_chars} "
                    "non-whitespace characters matched", details={"kind": "alignment"})
        if cov["missing_chars"]:
            add("CL007", "error", None,
                f"source coverage loss: {cov['missing_chars']}/{cov['total_chars']} non-whitespace characters missing",
                details={"kind": "coverage", **cov})
    stats["defects"] = len(issues)
    stats["by_rule"] = dict(sorted(Counter(i.rule for i in issues).items()))
    return Report(issues, alignments, stats)
