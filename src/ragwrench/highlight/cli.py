"""The `ragwrench highlight` CLI. No network calls, shell execution, or metadata-derived writes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pypdf.errors import PyPdfError

from . import __version__
from .audit import AuditMismatch, check_audit, read_final_regions
from .consolidate import assign_chunks, consolidate
from .core import (CiteError, MAX_JSON_BYTES, canonical, geometries, number, obj, open_pdf,
                   preflight_outputs, read_json, read_regions, regions_jsonl, validate_regions, write_text)
from .normalize import normalize_docling, normalize_unstructured
from .manifest_check import ManifestMismatch, check_manifest
from .pdf import annotate, manifest, verify
from .preflight import preflight, resolve_regions, resolve_with_audit
from .inventory import annotation_inventory
from .removal import remove_annotations
from .sidecar import check_sidecar, export_sidecar, import_sidecar


CLI_ERRORS = (CiteError, OSError, PyPdfError, RecursionError, UnicodeError,
              KeyError, TypeError, ValueError, IndexError, OverflowError)


def error_message(exc: Exception) -> str:
    if isinstance(exc, (CiteError, OSError, PyPdfError, RecursionError, UnicodeError)):
        return str(exc)
    return f"invalid PDF or provenance structure: {exc}"


def nonnegative(value: str) -> float:
    try:
        result = number(float(value), "tolerance")
        if result < 0:
            raise CiteError("tolerance must be nonnegative")
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragwrench highlight", description="Offline provenance-to-PDF highlights and citations")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    norm = sub.add_parser("normalize", help="Docling/Unstructured JSON to PDF-space regions.jsonl")
    norm.add_argument("input", type=Path)
    norm.add_argument("--format", choices=("docling", "unstructured"), required=True)
    norm.add_argument("--pdf", type=Path, help="source PDF; required for PixelSpace and missing Docling page sizes")
    norm.add_argument("--page-box", choices=("media", "crop"), help="source coordinate extent (defaults depend on dialect)")
    norm.add_argument("--coordinate-frame", choices=("unrotated", "display"), help="source axes before/after /Rotate")
    norm.add_argument("--chunks", type=Path, help="optional chunk membership JSON")
    norm.add_argument("--strict", action="store_true", help="error instead of warning/clamping out-of-bounds boxes")
    con = sub.add_parser("consolidate", help="map chunks and merge adjacent same-line regions")
    con.add_argument("input", type=Path)
    con.add_argument("--chunks", type=Path)
    con.add_argument("--per-page", action="store_true", help="union all boxes for each (chunk,page)")
    con.add_argument("--line-tolerance", type=nonnegative, default=2)
    con.add_argument("--max-gap", type=nonnegative, default=8)
    ann = sub.add_parser("annotate", help="write standard PDF highlights and citations.json")
    ann.add_argument("pdf", type=Path)
    ann.add_argument("regions", type=Path)
    ann.add_argument("--manifest", type=Path, help="default: citations.json beside output PDF")
    ann.add_argument("--strict", action="store_true",
                     help="compatibility option; annotation now always requires in-bounds reviewed boxes")
    ann.add_argument("--resolutions", type=Path, help="preflight report with per-item review decisions")
    pre = sub.add_parser("preflight", help="summarize citation evidence and create a resolution template")
    pre.add_argument("pdf", type=Path)
    pre.add_argument("regions", type=Path)
    res = sub.add_parser("resolve", help="materialize reviewed regions as normalized JSONL (all skips: empty file)")
    res.add_argument("pdf", type=Path)
    res.add_argument("regions", type=Path, help="original nonempty normalized regions JSONL")
    res.add_argument("--resolutions", type=Path, required=True, help="preflight report with per-item review decisions")
    res.add_argument("--audit", type=Path, help="optional resolution audit JSON; one original-to-final record per item")
    check = sub.add_parser("check-audit", help="read-only replay and comparison of saved resolution artifacts")
    check.add_argument("pdf", type=Path)
    check.add_argument("regions", type=Path, help="original nonempty normalized regions JSONL")
    check.add_argument("--resolutions", type=Path, required=True, help="edited preflight report authorizing decisions")
    check.add_argument("--audit", type=Path, required=True, help="saved resolve --audit JSON")
    check.add_argument("--final-regions", type=Path, required=True, help="saved resolved JSONL (may be empty)")
    check.add_argument("--json", action="store_true", help="emit a versioned JSON result on stdout (usage errors remain stderr)")
    man = sub.add_parser("manifest", help="write per-chunk rects and #page links")
    man.add_argument("pdf", type=Path)
    man.add_argument("regions", type=Path)
    ver = sub.add_parser("verify", help="reopen and compare annotations with regions (never clamps)")
    ver.add_argument("pdf", type=Path)
    ver.add_argument("regions", type=Path)
    ver.add_argument("--tolerance", type=nonnegative, default=0.01, help="numeric PDF serialization tolerance in points")
    ver.add_argument("--text-check", choices=("off", "best-effort", "required"), default="best-effort")
    ver.add_argument("--resolutions", type=Path, help="replay decisions against the original regions")
    ver.add_argument("--json", action="store_true", help="emit a versioned JSON result on stdout (usage errors remain stderr)")
    check_man = sub.add_parser("check-manifest", help="read-only verification of PDF annotations and saved citations")
    check_man.add_argument("pdf", type=Path)
    check_man.add_argument("regions", type=Path)
    check_man.add_argument("--manifest", type=Path, required=True, help="saved citation manifest beside the PDF")
    check_man.add_argument("--resolutions", type=Path, help="replay decisions against the original nonempty regions")
    check_man.add_argument("--tolerance", type=nonnegative, default=0.01, help="PDF numeric tolerance; manifest comparison is exact")
    check_man.add_argument("--text-check", choices=("off", "best-effort", "required"), default="best-effort")
    check_man.add_argument("--json", action="store_true", help="emit a versioned JSON result on stdout (usage errors remain stderr)")
    exp = sub.add_parser("export", help="export verified citehl annotations to a fingerprinted JSON sidecar")
    exp.add_argument("pdf", type=Path)
    exp.add_argument("regions", type=Path, nargs="?",
                     help="legacy PDF fallback; otherwise recover regions from embedded annotation metadata")
    check_side = sub.add_parser("check-sidecar", help="read-only verification of a saved annotation sidecar")
    check_side.add_argument("pdf", type=Path)
    check_side.add_argument("--sidecar", type=Path, required=True, help="saved citehl export JSON")
    check_side.add_argument("--regions", type=Path,
                            help="legacy fallback; final JSONL in the original export order")
    imp = sub.add_parser("import", help="add sidecar highlights to a matching PDF, preserving existing annotations")
    imp.add_argument("pdf", type=Path)
    imp.add_argument("sidecar", type=Path)
    inventory = sub.add_parser("list-annotations", help="read-only JSON inventory of persisted citehl IDs and provenance")
    inventory.add_argument("pdf", type=Path)
    rem = sub.add_parser("remove", help="remove explicitly selected citehl IDs into a separate PDF copy")
    rem.add_argument("pdf", type=Path)
    rem.add_argument("selection", type=Path, help='strict JSON: {"version":1,"annotation_ids":["exact-id"]}; [] selects nothing')
    for command in (man, ver, exp, check_man, check_side):
        command.add_argument("--allow-empty", action="store_true",
                             help="explicitly allow an empty regions JSONL file")
    for command in (norm, con, ann, pre, res, man, exp, imp, rem):
        command.add_argument("-o", "--output", type=Path, required=True)
        command.add_argument("--force", action="store_true", help="atomically replace existing output files")
    return p


def run_check_audit(args: argparse.Namespace) -> int:
    result = {"format": "citehl-check-audit-result", "schema_version": 1,
              "ok": False, "summary": None, "error": None}
    category = "input_error"
    try:
        reader, original = open_pdf(args.pdf), read_regions(args.regions)
        resolutions, audit = read_json(args.resolutions), read_json(args.audit)
        final = read_final_regions(args.final_regions)
        category = "replay_error"
        summary = check_audit(reader, original, resolutions, audit, final)
    except CLI_ERRORS as exc:
        if not args.json:
            raise
        result["error"] = {
            "category": "artifact_mismatch" if isinstance(exc, AuditMismatch) else category,
            "path": exc.path if isinstance(exc, AuditMismatch) else None,
            "message": error_message(exc),
        }
        print(json.dumps(result, ensure_ascii=True))
        return 1
    if args.json:
        result.update(ok=True, summary=summary)
        print(json.dumps(result, ensure_ascii=True))
    else:
        print("OK check-audit " + json.dumps(summary))
    return 0


def run_verify(args: argparse.Namespace) -> int:
    result = {"format": "citehl-verify-result", "schema_version": 1,
              "ok": False, "summary": None, "error": None}
    category = "input_error"
    try:
        reader = open_pdf(args.pdf)
        original = read_regions(args.regions, allow_empty=args.allow_empty and not args.resolutions)
        if args.resolutions:
            resolutions = read_json(args.resolutions)
            category = "replay_error"
            regions = resolve_regions(reader, original, resolutions)
        else:
            regions = validate_regions(original, geometries(reader), strict=True)
        category = "verification_error"
        report = verify(reader, regions, tolerance=args.tolerance, text_check=args.text_check)
    except CLI_ERRORS as exc:
        if not args.json:
            raise
        result["error"] = {"category": category, "message": error_message(exc)}
        print(json.dumps(result, ensure_ascii=True))
        return 1
    if args.json:
        result.update(ok=True, summary=report)
        print(json.dumps(result, ensure_ascii=True))
    else:
        print(f"OK {report['annotations']}/{len(regions)} " + json.dumps(report, ensure_ascii=False))
    return 0


def run_check_manifest(args: argparse.Namespace) -> int:
    result = {"format": "citehl-check-manifest-result", "schema_version": 1,
              "ok": False, "summary": None, "error": None}
    category = "input_error"
    try:
        if args.manifest.parent.resolve() != args.pdf.parent.resolve():
            raise CiteError("manifest and PDF must share a directory so relative links remain valid")
        reader = open_pdf(args.pdf)
        original = read_regions(args.regions, allow_empty=args.allow_empty and not args.resolutions)
        if args.resolutions:
            resolutions = read_json(args.resolutions)
            category = "replay_error"
            regions = resolve_regions(reader, original, resolutions)
        else:
            regions = validate_regions(original, geometries(reader), strict=True)
        category = "input_error"
        saved = read_json(args.manifest)
        category = "verification_error"
        summary = check_manifest(reader, regions, saved, args.pdf.name,
                                 tolerance=args.tolerance, text_check=args.text_check)
    except CLI_ERRORS as exc:
        if not args.json:
            raise
        result["error"] = {
            "category": "manifest_mismatch" if isinstance(exc, ManifestMismatch) else category,
            "path": exc.path if isinstance(exc, ManifestMismatch) else None,
            "message": error_message(exc),
        }
        print(json.dumps(result, ensure_ascii=True))
        return 1
    if args.json:
        result.update(ok=True, summary=summary)
        print(json.dumps(result, ensure_ascii=True))
    else:
        print("OK check-manifest " + json.dumps(summary))
    return 0


def run(args: argparse.Namespace) -> int | None:
    if args.command == "normalize":
        inputs = [args.input] + ([args.pdf] if args.pdf else []) + ([args.chunks] if args.chunks else [])
        preflight_outputs([args.output], inputs, args.force)
        geometry = geometries(open_pdf(args.pdf)) if args.pdf else None
        function = normalize_docling if args.format == "docling" else normalize_unstructured
        regions = function(read_json(args.input), geometry, strict=args.strict,
                           page_box=args.page_box, coordinate_frame=args.coordinate_frame)
        if args.chunks:
            regions = assign_chunks(regions, read_json(args.chunks))
        regions = list(dict.fromkeys(regions))
        write_text(args.output, regions_jsonl(regions))
        print(f"OK normalized {len(regions)} regions -> {args.output}")
    elif args.command == "consolidate":
        preflight_outputs([args.output], [args.input] + ([args.chunks] if args.chunks else []), args.force)
        regions = read_regions(args.input)
        before = len(regions)
        if args.chunks:
            regions = assign_chunks(regions, read_json(args.chunks))
        regions = consolidate(regions, per_page=args.per_page,
                              line_tolerance=args.line_tolerance, max_gap=args.max_gap)
        write_text(args.output, regions_jsonl(regions))
        print(f"OK consolidated {before} -> {len(regions)} regions")
    elif args.command == "preflight":
        preflight_outputs([args.output], [args.pdf, args.regions], args.force)
        report = preflight(open_pdf(args.pdf), read_regions(args.regions))
        write_text(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("OK preflight " + json.dumps(report["summary"]) + f" -> {args.output}")
    elif args.command == "resolve":
        outputs = [args.output] + ([args.audit] if args.audit else [])
        preflight_outputs(outputs, [args.pdf, args.regions, args.resolutions], args.force)
        reader, original = open_pdf(args.pdf), read_regions(args.regions)
        report = obj(read_json(args.resolutions), "resolutions")
        audit_data = None
        if args.audit:
            regions, audit = resolve_with_audit(reader, original, report)
            audit_data = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
            if len(audit_data.encode("utf-8")) > MAX_JSON_BYTES:
                raise CiteError("resolution audit exceeds 32 MiB JSON limit")
        else:
            regions = resolve_regions(reader, original, report)
        data = regions_jsonl(regions, allow_empty=True)
        if len(data.encode("utf-8")) > MAX_JSON_BYTES:
            raise CiteError("resolved regions exceed 32 MiB JSONL input limit")
        write_text(args.output, data)
        if args.audit:
            write_text(args.audit, audit_data)
        print(f"OK resolved {len(regions)} regions -> {args.output}"
              + (f"; audit -> {args.audit}" if args.audit else ""))
    elif args.command == "check-audit":
        return run_check_audit(args)
    elif args.command == "verify":
        return run_verify(args)
    elif args.command == "check-manifest":
        return run_check_manifest(args)
    elif args.command == "check-sidecar":
        reader = open_pdf(args.pdf)
        regions = read_regions(args.regions, allow_empty=args.allow_empty) if args.regions else None
        saved = read_json(args.sidecar)
        report = check_sidecar(reader, saved, regions)
        print("OK check-sidecar " + json.dumps(report))
    elif args.command == "export":
        preflight_outputs([args.output], [args.pdf] + ([args.regions] if args.regions else []), args.force)
        data = export_sidecar(open_pdf(args.pdf),
                              read_regions(args.regions, allow_empty=args.allow_empty) if args.regions else None)
        # Compact canonical JSON keeps exported files within the import size limit.
        write_text(args.output, canonical(data))
        print(f"OK exported {len(data['annotations'])} annotations -> {args.output}")
    elif args.command == "import":
        preflight_outputs([args.output], [args.pdf, args.sidecar], args.force)
        report = import_sidecar(open_pdf(args.pdf), read_json(args.sidecar), args.output)
        print(f"OK imported {report['added']} annotations; already present {report['already_present']} -> {args.output}")
    elif args.command == "list-annotations":
        print(canonical(annotation_inventory(open_pdf(args.pdf))))
    elif args.command == "remove":
        preflight_outputs([args.output], [args.pdf, args.selection], args.force)
        if (args.pdf.resolve() == args.selection.resolve()
                or (args.pdf.exists() and args.selection.exists() and args.pdf.samefile(args.selection))):
            raise CiteError("PDF and selection inputs must be distinct")
        report = remove_annotations(open_pdf(args.pdf), read_json(args.selection), args.output)
        print(f"OK removed {report['removed']} annotations; preserved {report['preserved']} -> {args.output}")
    else:
        outputs = [args.output]
        if args.command == "annotate":
            manifest_path = args.manifest or args.output.with_name("citations.json")
            outputs.append(manifest_path)
        resolutions_path = getattr(args, "resolutions", None)
        inputs = [args.pdf, args.regions] + ([resolutions_path] if resolutions_path else [])
        preflight_outputs(outputs, inputs, args.force)
        reader = open_pdf(args.pdf)
        if args.command == "annotate":
            regions = resolve_regions(reader, read_regions(args.regions),
                                      read_json(resolutions_path) if resolutions_path else None)
        else:
            regions = validate_regions(read_regions(args.regions, allow_empty=args.allow_empty),
                                       geometries(reader), strict=True)
        pdf_name = args.output.name if args.command == "annotate" else args.pdf.name
        if args.command == "annotate" and manifest_path.parent.resolve() != args.output.parent.resolve():
            raise CiteError("manifest and annotated PDF must share a directory so relative links remain valid")
        data = json.dumps(manifest(regions, pdf_name), ensure_ascii=False, indent=2) + "\n"
        if args.command == "annotate":
            annotate(reader, regions, args.output)
            write_text(manifest_path, data)
            print(f"OK annotated {len(regions)} regions -> {args.output}; manifest -> {manifest_path}")
        else:
            if args.output.parent.resolve() != args.pdf.parent.resolve():
                raise CiteError("manifest and PDF must share a directory so relative links remain valid")
            write_text(args.output, data)
            print(f"OK manifest {len(regions)} regions -> {args.output}")


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.command == "check-sidecar" and args.allow_empty and args.regions is None:
        p.error("check-sidecar --allow-empty requires --regions")
    try:
        return run(args) or 0
    except CLI_ERRORS as exc:
        print(f"ragwrench highlight: {error_message(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
