"""The `ragwrench eval` command line interface. Exit 0 success, 1 gate failure, 2 error."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

from . import __version__
from .compare import (PreflightMismatch, compare, compare_many, compare_many_saved, compare_saved, parse_candidate_thresholds,
                      parse_thresholds, validate_candidate_names)
from .formats import InputError, decode, read_qrels, read_run, write_qrels, write_run
from .metrics import parse_metrics
from .preflight import diff_preflight, validate_preflight
from .report import render
from .saved import read_evaluation, saved_evaluation


HELP = {
    "run": "score one retrieval run against qrels and write a self-contained eval artifact",
    "compare": "paired comparison of a candidate run against a baseline; exit 1 if a --max-drop gate fails",
    "compare-many": "compare several named candidates against one baseline with multiple-testing correction",
}
# Hidden, backward-compatible spellings of subcommands (``ragwrench eval eval`` -> ``run``).
ALIASES = {"eval": "run"}


def parser():
    root = argparse.ArgumentParser(prog="ragwrench eval", description="Offline deterministic retrieval evaluation and CI regression gate",
                                   epilog="typical flow: 'run' scores one run, 'compare' gates a candidate against a baseline")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("run", "compare", "compare-many"):
        sub = commands.add_parser(name, help=HELP[name])
        sub.add_argument("--qrels", required=True, type=Path)
        sub.add_argument("--qrels-format", choices=("auto", "trec", "beir", "llamaindex"), default="auto")
        sub.add_argument("--run-format", choices=("auto", "trec", "jsonl"), default="auto")
        sub.add_argument("--metrics", default="ndcg@10,mrr@10,recall@100")
        sub.add_argument("--gain", choices=("linear", "exponential"), default="linear")
        formats = ("text", "json", "markdown", "junit", "csv") if name == "compare-many" else ("text", "json", "markdown", "junit")
        sub.add_argument("--format", choices=formats, default="json" if name == "run" else None)
        _output_options(sub)
        if name == "run":
            sub.add_argument("--run", required=True, type=Path)
        else:
            sub.add_argument("--baseline", required=True, type=Path)
            if name == "compare-many":
                _family_options(sub)
            else:
                sub.add_argument("--candidate", required=True, type=Path)
            _comparison_options(sub)
    for name in ("compare-saved", "compare-many-saved"):
        sub = commands.add_parser(name, help="compare eval JSON artifacts with matching judged datasets and scoring parameters")
        sub.add_argument("--baseline", required=True, type=Path)
        if name == "compare-many-saved":
            _family_options(sub)
        else:
            sub.add_argument("--candidate", required=True, type=Path)
        formats = ("text", "json", "markdown", "junit", "csv") if name == "compare-many-saved" else ("text", "json", "markdown", "junit")
        sub.add_argument("--format", choices=formats)
        _comparison_options(sub)
        _output_options(sub)
    sub = commands.add_parser("validate-preflight", help="validate one complete bound family audit without retrieval inputs")
    sub.add_argument("--input", required=True, type=Path)
    _output_options(sub)
    sub = commands.add_parser("validate-evaluation", help="validate one complete eval artifact and its checksum without retrieval inputs")
    sub.add_argument("--input", required=True, type=Path)
    _output_options(sub)
    sub = commands.add_parser("diff-preflight", help="diff two complete bound family audits without retrieval inputs")
    sub.add_argument("--expected", required=True, type=Path)
    sub.add_argument("--actual", required=True, type=Path)
    _output_options(sub)
    sub = commands.add_parser("convert", help="convert qrels or runs between TREC, BEIR, LlamaIndex and JSONL formats")
    sub.add_argument("--input", required=True, type=Path)
    sub.add_argument("--from", dest="source", required=True, choices=("trec-qrels", "beir-qrels", "llamaindex", "trec-run", "jsonl-run"))
    sub.add_argument("--to", dest="target", required=True, choices=("trec-qrels", "beir-qrels", "trec-run", "jsonl-run"))
    _output_options(sub)
    return root


def _family_options(sub):
    sub.add_argument("--expected-preflight", type=Path, metavar="PATH",
                     help="require a matching complete bound preflight JSON audit before inference")
    sub.add_argument("--preflight-diff-output", type=Path, metavar="PATH",
                     help="with --expected-preflight, atomically write a JSON diff only on valid audit drift; --force permits replacement")
    sub.add_argument("--preflight-only", action="store_true",
                     help="validate family and emit a JSON audit summary without inference or gate decisions")
    sub.add_argument("--candidate", required=True, action="append", metavar="NAME=PATH",
                     help="repeatable named candidate; all candidates use the same baseline")
    sub.add_argument("--correction", required=True, choices=("none", "bonferroni", "holm"),
                     help="p-value correction across all candidate-metric pairs; does not affect gates")
    sub.add_argument("--alpha", type=float, default=0.05, help="adjusted p-value significance cutoff")
    sub.add_argument("--candidate-max-drop", action="append", default=[], metavar="NAME:METRIC=VALUE|off",
                     help="repeatable candidate-specific maximum drop; overrides shared --max-drop; off disables that gate")
    sub.add_argument("--max-family-resamples", type=int, metavar="LIMIT",
                     help="optional positive total-family work limit: queries * samples * candidate-metric pairs; ttest uses zero")
    sub.add_argument("--max-family-pairs", type=int, metavar="LIMIT",
                     help="optional positive candidate-metric pair limit, including ttest and ungated pairs")


def _comparison_options(sub):
    sub.add_argument("--test", choices=("bootstrap", "randomization", "ttest"), default="randomization")
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--samples", type=int, default=10000)
    sub.add_argument("--confidence", type=float, default=0.95)
    sub.add_argument("--max-drop", action="append", default=[], metavar="METRIC=VALUE", help="repeatable maximum absolute mean drop; significance does not affect gate")
    sub.add_argument("--top", type=int, default=10, help="top regressed queries per metric")


def _output_options(sub):
    sub.add_argument("--output", type=Path, help="write atomically to this file instead of stdout; parent must exist")
    sub.add_argument("--force", action="store_true", help="replace existing output; never an input file")


def parse_candidates(values):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not path:
            raise InputError("candidate must be NAME=PATH with a nonempty path")
        validate_candidate_names([name])
        if name in result:
            raise InputError(f"duplicate candidate name: {name!r}")
        result[name] = Path(path)
    return result


def emit(text, output, force, inputs):
    if output is None:
        sys.stdout.write(text)
        return
    if output.is_symlink() or any(output.resolve() == path.resolve() for path in inputs):
        raise InputError("output must not be a symlink or overwrite an input file")
    # Temporary file + hard link ensures no clobber by default, including racing writers.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".retgate-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_named_evaluation(path, label):
    try:
        return read_evaluation(path)
    except (InputError, OSError, UnicodeError, RecursionError) as exc:
        raise InputError(f"{label}: {exc}") from None


def _read_audit(path, label):
    try:
        return decode(path.read_text(encoding="utf-8"), path)
    except (InputError, OSError, UnicodeError, RecursionError) as exc:
        raise InputError(f"{label} audit: {exc}") from None


def _aliases(left, right):
    return (left.resolve() == right.resolve() or
            (left.exists() and right.exists() and left.samefile(right)))


def _validate_guard_destinations(args, expected_path, diagnostic):
    """Reserve both outputs before inference, even when the guard will match."""
    if expected_path is None:
        raise InputError("--preflight-diff-output requires --expected-preflight")
    paths = parse_candidates(args.candidate)
    inputs = [expected_path, args.baseline, *paths.values()]
    if args.command == "compare-many":
        inputs.append(args.qrels)
    destinations = [diagnostic] + ([args.output] if args.output is not None else [])
    if len(destinations) == 2 and _aliases(*destinations):
        raise InputError("preflight diff output must not alias normal report output")
    for output in destinations:
        if output.is_symlink() or any(_aliases(output, path) for path in inputs):
            raise InputError("output must not be a symlink or overwrite an input file")
        if not output.parent.is_dir():
            raise InputError(f"output parent must be an existing directory: {output.parent}")
        if output.exists() and (not output.is_file() or not args.force):
            raise InputError(f"output must be a regular file and requires --force to replace: {output}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ALIASES:
        argv[0] = ALIASES[argv[0]]
    args = parser().parse_args(argv)
    try:
        preflight_only = getattr(args, "preflight_only", False)
        expected_path = getattr(args, "expected_preflight", None)
        diagnostic = getattr(args, "preflight_diff_output", None)
        if diagnostic is not None:
            _validate_guard_destinations(args, expected_path, diagnostic)
        expected_preflight = None
        if expected_path is not None:
            # This audit is also an input. Refuse aliases before any inference,
            # including a matching audit with --force and an existing output.
            if args.output is not None and (args.output.is_symlink() or args.output.resolve() == expected_path.resolve()):
                raise InputError("output must not be a symlink or overwrite an input file (expected preflight)")
            expected_preflight = decode(expected_path.read_text(encoding="utf-8"), expected_path)
            if expected_preflight is None:
                raise InputError("expected preflight must be a complete bound audit object")
        if preflight_only and args.format not in (None, "json"):
            raise InputError("preflight-only supports only JSON; omit --format or use --format json")
        if hasattr(args, "format"):
            args.format = args.format or ("json" if preflight_only else "text")
        if args.command in {"validate-preflight", "validate-evaluation"}:
            inputs = [args.input]
            try:
                aliases_input = args.output is not None and (args.output.is_symlink() or _aliases(args.output, args.input))
            except RuntimeError as exc:  # pathlib reports symlink loops this way on Python 3.10–3.12.
                raise InputError(f"cannot resolve validation input/output path: {exc}") from None
            if aliases_input:
                raise InputError("output must not be a symlink or overwrite an input file")
            if args.command == "validate-evaluation":
                _read_named_evaluation(args.input, "input evaluation")
            else:
                validate_preflight(_read_audit(args.input, "input"))
            text = render({"schema_version": 1, "command": args.command, "valid": True}, "json")
            status = 0
        elif args.command == "diff-preflight":
            inputs = [args.expected, args.actual]
            report = diff_preflight(_read_audit(args.expected, "expected"), _read_audit(args.actual, "actual"))
            text = render(report, "json")
            status = 0 if report["matches"] else 1
        elif args.command == "convert":
            inputs = [args.input]
            if args.source in {"trec-qrels", "beir-qrels", "llamaindex"}:
                if args.target not in {"trec-qrels", "beir-qrels"}:
                    raise InputError("cannot convert qrels into a run")
                data = read_qrels(args.input, args.source.removesuffix("-qrels"))
                text = write_qrels(data, args.target.removesuffix("-qrels"))
            else:
                if args.target not in {"trec-run", "jsonl-run"}:
                    raise InputError("cannot convert a run into qrels")
                data = read_run(args.input, args.source.removesuffix("-run"))
                text = write_run(data, args.target.removesuffix("-run"))
            status = 0
        elif args.command in {"compare-saved", "compare-many-saved"}:
            inputs = [args.baseline]
            baseline = _read_named_evaluation(args.baseline, "baseline")
            metrics = parse_metrics(",".join(baseline["manifest"]["parameters"]["metrics"]))
            options = dict(method=args.test, seed=args.seed, samples=args.samples, confidence=args.confidence,
                           thresholds=parse_thresholds(args.max_drop, metrics), top=args.top)
            if args.command == "compare-many-saved":
                paths = parse_candidates(args.candidate)
                inputs += paths.values()
                candidate_thresholds = parse_candidate_thresholds(args.candidate_max_drop, paths, metrics)
                candidates = {name: _read_named_evaluation(path, f"candidate {name!r}")
                              for name, path in sorted(paths.items())}
                report = compare_many_saved(baseline, candidates, correction=args.correction, alpha=args.alpha,
                                            candidate_thresholds=candidate_thresholds,
                                            max_family_resamples=args.max_family_resamples,
                                            max_family_pairs=args.max_family_pairs, preflight_only=preflight_only,
                                          expected_preflight=expected_preflight, **options)
            else:
                inputs.append(args.candidate)
                report = compare_saved(baseline, read_evaluation(args.candidate), **options)
            status = 0 if preflight_only or report["passed"] else 1
            text = render(report, args.format)
        else:
            metrics = parse_metrics(args.metrics)
            qrels = read_qrels(args.qrels, args.qrels_format)
            inputs = [args.qrels]
            if args.command == "run":
                inputs.append(args.run)
                report = saved_evaluation(qrels, read_run(args.run, args.run_format), metrics, args.gain)
                status = 0
            else:
                thresholds = parse_thresholds(args.max_drop, metrics)
                options = dict(gain=args.gain, method=args.test, seed=args.seed, samples=args.samples,
                               confidence=args.confidence, thresholds=thresholds, top=args.top)
                if args.command == "compare-many":
                    paths = parse_candidates(args.candidate)
                    candidate_thresholds = parse_candidate_thresholds(args.candidate_max_drop, paths, metrics)
                    inputs += [args.baseline, *paths.values()]
                    baseline = read_run(args.baseline, args.run_format)
                    candidates = {name: read_run(path, args.run_format) for name, path in sorted(paths.items())}
                    report = compare_many(qrels, baseline, candidates, metrics,
                                          correction=args.correction, alpha=args.alpha,
                                          candidate_thresholds=candidate_thresholds,
                                          max_family_resamples=args.max_family_resamples,
                                          max_family_pairs=args.max_family_pairs, preflight_only=preflight_only,
                                          expected_preflight=expected_preflight, **options)
                else:
                    inputs += [args.baseline, args.candidate]
                    report = compare(qrels, read_run(args.baseline, args.run_format),
                                     read_run(args.candidate, args.run_format), metrics, **options)
                status = 0 if preflight_only or report["passed"] else 1
            text = render(report, args.format)
        if expected_path is not None:
            inputs.append(expected_path)
        emit(text, args.output, args.force, inputs)
        return status
    except BrokenPipeError:
        return 2
    except (InputError, OSError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, PreflightMismatch) and diagnostic is not None:
            try:
                inputs = [expected_path, args.baseline, *parse_candidates(args.candidate).values()]
                if args.command == "compare-many":
                    inputs.append(args.qrels)
                if args.output is not None:
                    inputs.append(args.output)
                emit(render(exc.diff, "json"), diagnostic, args.force, inputs)
            except (InputError, OSError, UnicodeError, RecursionError) as write_error:
                print(f"ragwrench eval: error: preflight diff output: {write_error}", file=sys.stderr)
                return 2
        print(f"ragwrench eval: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
