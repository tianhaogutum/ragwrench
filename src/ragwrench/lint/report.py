from pathlib import Path

from .model import RULES


def failed(report, threshold):
    levels = {"note": 0, "warning": 1, "error": 2, "none": 3}
    return any(levels[i.severity] >= levels[threshold] for i in report.issues)


def text_report(report):
    lines = [f"{i.rule} {i.severity} {i.chunk_id or '<source>'}:{i.line}: {i.message}" for i in report.issues]
    lines.append(f"{report.stats['chunks']} chunks; {len(report.issues)} defects")
    if "source" in report.stats:
        lines.append(f"Source coverage: {report.stats['source']['coverage']:.2%}; "
                     f"alignment failure rate: {report.stats['alignment_failure_rate']:.2%}; "
                     f"ambiguous chunks: {report.stats['ambiguous_chunks']}")
    return "\n".join(lines) + "\n"


def sarif(report, input_path, source_path=None):
    artifacts = [{"location": {"uri": Path(input_path).resolve().as_uri()}}]
    if source_path is not None:
        artifacts.append({"location": {"uri": Path(source_path).resolve().as_uri()}})
    alignments = {a.chunk_id: a for a in report.alignments}
    results = []
    for issue in report.issues:
        result = {"ruleId": issue.rule, "level": issue.severity,
                  "message": {"text": issue.message},
                  "properties": {"chunkId": issue.chunk_id, "chunkLine": issue.line,
                                 "relatedChunks": issue.related, **issue.details}}
        alignment = alignments.get(issue.chunk_id)
        if source_path is not None and alignment and alignment.spans:
            # A recovered span's first source line is reliable even when the chunk
            # stripped newlines. Chunk-relative location stays in properties.
            location = {"physicalLocation": {"artifactLocation": {"index": 1,
                        "uri": artifacts[1]["location"]["uri"]},
                        "region": {"startLine": alignment.spans[0].start_line}}}
            result["locations"] = [location]
        elif source_path is not None and issue.chunk_id is None:
            result["locations"] = [{"physicalLocation": {"artifactLocation": {
                "index": 1, "uri": artifacts[1]["location"]["uri"]}}}]
        else:
            result["locations"] = [{"physicalLocation": {"artifactLocation": {
                "index": 0, "uri": artifacts[0]["location"]["uri"]}}}]
        if issue.chunk_id:
            result["locations"][0]["logicalLocations"] = [{"name": issue.chunk_id, "kind": "element"}]
        results.append(result)
    return {"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
            "runs": [{"tool": {"driver": {"name": "chunklint", "version": "0.1.0", "rules": [
                {"id": rule, "shortDescription": {"text": title},
                 "defaultConfiguration": {"level": level}} for rule, (level, title) in RULES.items()]}},
                      "artifacts": artifacts, "results": results,
                      "properties": {"stats": report.stats}}]}
