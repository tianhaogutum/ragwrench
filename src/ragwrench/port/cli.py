import argparse
import json
from pathlib import Path
import sys

from .api import API, API_PROFILES, select_profile
from .bundle import load_bundle
from .capabilities import capabilities
from .common import PortError
from .operations import export_bundle, import_bundle, verify_bundle
from .status import CHUNK_STATUSES, PENDING_STAGES, import_status


def _positive_page_size(value):
    try:
        number = int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("page size must be a positive integer")


def parser():
    root = argparse.ArgumentParser(prog="ragwrench port", description="Portable, hash-verified knowledge base chunks")
    root.add_argument("--version", action="version", version="ragwrench port 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Offline bundle schema, relationship and hash validation")
    validate.add_argument("bundle", type=Path)
    status = commands.add_parser("import-status", help="Offline snapshot of saved import mappings and pending stages")
    status.add_argument("--state", type=Path, required=True, help="Existing import state file; read without writes or credentials")
    status.add_argument("--bundle", type=Path, help="Original bundle; validate and recompute local progress using saved preflight options")
    status.add_argument("--list-bundle-chunks", action="store_true", help="With --bundle, list local chunk status across every document; global snapshot unchanged")
    status.add_argument("--bundle-chunk-status", choices=CHUNK_STATUSES, help="With --list-bundle-chunks, filter local status before pagination")
    status.add_argument("--bundle-chunk-page-size", type=_positive_page_size, help="With --list-bundle-chunks, limit chunk_details and add next_bundle_chunk_cursor")
    status.add_argument("--bundle-chunk-cursor", help="Continue --list-bundle-chunks --bundle-chunk-page-size using identical state, bundle and status filter")
    status.add_argument("--list-skipped-chunks", action="store_true", help="With --bundle, list saved-preflight skips across all documents; global snapshot unchanged")
    status.add_argument("--skip-page-size", type=_positive_page_size, help="With --list-skipped-chunks, limit skip rows and add next_skip_cursor")
    status.add_argument("--skip-cursor", help="Continue --list-skipped-chunks --skip-page-size using identical state and bundle")
    status.add_argument("--summary-only", action="store_true", help="Omit document detail arrays; full validation and aggregate local counts remain unchanged")
    status.add_argument("--pending-counts", action="store_true", help="With --summary-only, add aggregate counts of saved dataset/document/chunk pending stages")
    status.add_argument("--pending-stage", choices=PENDING_STAGES, help="Filter document detail arrays by one saved pending stage; all counts remain global")
    status.add_argument("--document-page-size", type=_positive_page_size, help="With --pending-stage, limit both document arrays and add top-level next_document_cursor")
    status.add_argument("--document-cursor", help="Continue --pending-stage --document-page-size using identical saved state and optional bundle scope")
    status.add_argument("--document", metavar="SOURCE_ID", help="With --bundle, show one document's detail row; snapshot and totals remain global")
    status.add_argument("--chunk", metavar="SOURCE_CHUNK_ID", help="With --bundle and --document, add one chunk's local checkpoint status")
    status.add_argument("--list-chunks", action="store_true", help="With --bundle and --document, list all source chunks' local status; excludes --chunk")
    status.add_argument("--chunk-status", choices=CHUNK_STATUSES, help="With --list-chunks, filter local status before pagination; totals stay global")
    status.add_argument("--page-size", type=_positive_page_size, help="With --list-chunks, return at most this many rows and a snapshot-bound next_cursor")
    status.add_argument("--cursor", help="Continue --list-chunks --page-size from next_cursor using identical saved state and bundle")
    capability = commands.add_parser("capabilities", help="Read-only pinned import fields and loss boundaries; no bundle needed")
    capability.add_argument("system", choices=("dify", "ragflow"))
    capability.add_argument("--api-profile", choices=API_PROFILES, help="Default/explicit profiles are offline; auto uses RAGFlow dataset GETs")
    capability.add_argument("--base-url", help="Required only for --api-profile auto")
    capability.add_argument("--key-env", help="API key environment variable, only for auto discovery")
    capability.add_argument("--page-size", type=int, help="Auto discovery only; default 100")
    capability.add_argument("--retries", type=int, help="Auto GET retries only; default 3")
    capability.add_argument("--backoff", type=float, help="Auto discovery only; default 0.5 seconds")
    capability.add_argument("--timeout", type=float, help="Auto discovery only; default 30 seconds")
    for command in ("export", "import", "verify"):
        sub = commands.add_parser(command, help={"export": "export a knowledge base from Dify or RAGFlow into a hash-checked bundle",
                                                 "import": "import a bundle into Dify or RAGFlow, resuming from saved mappings",
                                                 "verify": "compare a live knowledge base with a bundle"}[command])
        sub.add_argument("system", choices=("dify", "ragflow"))
        sub.add_argument("--api-profile", choices=API_PROFILES,
                         help="Pinned API contract (default: Dify f8fc371a / RAGFlow 1.0); auto uses RAGFlow dataset GETs")
        sub.add_argument("--base-url", required=True, help="Instance root or API root ending /v1 or /api/v1")
        sub.add_argument("--key-env", help="Environment variable containing API key (never a literal key)")
        sub.add_argument("--page-size", type=int, default=100)
        sub.add_argument("--retries", type=int, default=3, help="GET retries only")
        sub.add_argument("--backoff", type=float, default=0.5)
        sub.add_argument("--timeout", type=float, default=30)
        sub.add_argument("--poll-timeout", type=float, default=120)
        if command == "export":
            sub.add_argument("--dataset-id", required=True)
            sub.add_argument("-o", "--output", type=Path, required=True)
            sub.add_argument("--resume", action="store_true")
        else:
            sub.add_argument("bundle", type=Path)
            sub.add_argument("--dataset-id")
            sub.add_argument("--state", type=Path, required=command == "import", help="Persistent ID mapping file; keep for resume/verification")
            sub.add_argument("--allow-lossy", action="store_true")
            sub.add_argument("--max-chunk-bytes", type=int, default=16000)
        if command == "import":
            sub.add_argument("--dataset-name")
            sub.add_argument("--dry-run", action="store_true", help="Validate and plan offline, without credentials or writes")
            sub.add_argument("--on-error", choices=("skip", "abort"), default="abort", help="Skip or abort on per-chunk preflight errors; ambiguous writes always stop")
            sub.add_argument("--indexing", choices=("high_quality", "economy"), default="high_quality", help="Dify target indexing technique")
            sub.add_argument("--retry-pending", action="store_true", help="Explicitly retry a pending write with no visible match (inspect target first)")
            sub.add_argument("--chunk-retries", type=int, default=0, help="0..10 extra chunk recovery attempts: reconcile creates and retry enabled flags")
    return root


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    if args.command == "import-status":
        if (args.list_bundle_chunks or args.bundle_chunk_status is not None or
                args.bundle_chunk_page_size is not None or args.bundle_chunk_cursor is not None):
            if not args.list_bundle_chunks or args.bundle is None:
                root.error("Bundle chunk listing, filtering and pagination require --bundle and --list-bundle-chunks")
            if (args.summary_only or args.pending_counts or args.pending_stage is not None or
                    args.document is not None or args.chunk is not None or args.list_chunks or
                    args.chunk_status is not None or args.page_size is not None or args.cursor is not None or
                    args.document_page_size is not None or args.document_cursor is not None or
                    args.list_skipped_chunks or args.skip_page_size is not None or args.skip_cursor is not None):
                root.error("--list-bundle-chunks cannot be combined with summary, counts, pending stage, document/chunk selection, other listing, filtering or pagination")
        if args.bundle_chunk_cursor is not None and args.bundle_chunk_page_size is None:
            root.error("--bundle-chunk-cursor requires --bundle-chunk-page-size")
        if args.list_skipped_chunks or args.skip_page_size is not None or args.skip_cursor is not None:
            if not args.list_skipped_chunks or args.bundle is None:
                root.error("--list-skipped-chunks, --skip-page-size and --skip-cursor require --bundle and --list-skipped-chunks")
            if (args.summary_only or args.pending_counts or args.pending_stage is not None or
                    args.document is not None or args.chunk is not None or args.list_chunks or
                    args.chunk_status is not None or args.page_size is not None or args.cursor is not None or
                    args.document_page_size is not None or args.document_cursor is not None):
                root.error("--list-skipped-chunks cannot be combined with summary, counts, pending stage, document/chunk selection, listing, filtering or other pagination")
        if args.skip_cursor is not None and args.skip_page_size is None:
            root.error("--skip-cursor requires --skip-page-size")
    if args.command == "import-status" and args.pending_stage is not None and (
            args.summary_only or args.pending_counts or args.document is not None or args.chunk is not None or
            args.list_chunks or args.chunk_status is not None or args.page_size is not None or args.cursor is not None):
        root.error("--pending-stage cannot be combined with --summary-only, --pending-counts, --document, --chunk, --list-chunks, --chunk-status, --page-size or --cursor")
    if args.command == "import-status" and args.pending_counts and not args.summary_only:
        root.error("--pending-counts requires --summary-only")
    if args.command == "import-status" and args.summary_only and (
            args.document is not None or args.chunk is not None or args.list_chunks or
            args.chunk_status is not None or args.page_size is not None or args.cursor is not None):
        root.error("--summary-only cannot be combined with --document, --chunk, --list-chunks, --chunk-status, --page-size or --cursor")
    if args.command == "import-status" and args.list_chunks:
        if args.chunk is not None:
            root.error("--list-chunks cannot be combined with --chunk")
        if args.bundle is None or args.document is None:
            root.error("--list-chunks requires --bundle and --document")
    if args.command == "import-status" and args.chunk is not None and (args.bundle is None or args.document is None):
        root.error("--chunk requires --bundle and --document")
    if args.command == "import-status" and args.document is not None and args.bundle is None:
        root.error("--document requires --bundle")
    if args.command == "import-status":
        if (args.document_page_size is not None or args.document_cursor is not None) and args.pending_stage is None:
            root.error("--document-page-size and --document-cursor require --pending-stage")
        if args.document_cursor is not None and args.document_page_size is None:
            root.error("--document-cursor requires --document-page-size")
        if args.chunk_status is not None and not args.list_chunks:
            root.error("--chunk-status requires --list-chunks")
        if (args.page_size is not None or args.cursor is not None) and not args.list_chunks:
            root.error("--page-size and --cursor require --list-chunks")
        if args.cursor is not None and args.page_size is None:
            root.error("--cursor requires --page-size")
    api = None
    try:
        if args.command == "validate":
            manifest, _, _ = load_bundle(args.bundle)
            result = {"valid": True, **manifest["counts"]}
        elif args.command == "import-status":
            result = import_status(args.state, args.bundle, document=args.document, chunk=args.chunk,
                                   list_chunks=args.list_chunks, page_size=args.page_size, cursor=args.cursor,
                                   chunk_status=args.chunk_status, summary_only=args.summary_only,
                                   pending_counts=args.pending_counts, pending_stage=args.pending_stage,
                                   document_page_size=args.document_page_size, document_cursor=args.document_cursor,
                                   list_skipped_chunks=args.list_skipped_chunks, skip_page_size=args.skip_page_size,
                                   skip_cursor=args.skip_cursor, list_bundle_chunks=args.list_bundle_chunks,
                                   bundle_chunk_status=args.bundle_chunk_status,
                                   bundle_chunk_page_size=args.bundle_chunk_page_size,
                                   bundle_chunk_cursor=args.bundle_chunk_cursor)
        elif args.command == "capabilities":
            profile = select_profile(args.system, args.api_profile)
            network_options = {name: getattr(args, name) for name in
                               ("base_url", "key_env", "page_size", "retries", "backoff", "timeout")
                               if getattr(args, name) is not None}
            if profile == "auto":
                if not args.base_url:
                    raise PortError("Capabilities auto discovery requires --base-url")
                api = API(args.system, api_profile=profile, **network_options)
                result = capabilities(args.system, api.negotiate(), "dataset_signatures")
            else:
                if network_options:
                    raise PortError("Capabilities network options are used only with --api-profile auto; omit them for offline inspection")
                result = capabilities(args.system, profile, "explicit" if args.api_profile else "default")
        else:
            dry_run = getattr(args, "dry_run", False)
            api = API(args.system, args.base_url, key="offline-plan" if dry_run else None,
                      api_profile=args.api_profile,
                      key_env=args.key_env, page_size=args.page_size, retries=args.retries,
                      backoff=args.backoff, timeout=args.timeout, poll_timeout=args.poll_timeout)
            if args.command == "export":
                result = export_bundle(api, args.dataset_id, args.output, resume=args.resume)
            elif args.command == "import":
                protected = [(args.bundle / f).resolve() for f in ("manifest.json", "documents.jsonl", "chunks.jsonl", ".export-state.json")]
                if args.state.resolve() in protected or args.state.is_dir():
                    raise PortError("State path must not overwrite bundle files or a directory")
                result = import_bundle(api, args.bundle, args.state, dataset_name=args.dataset_name,
                                       dataset_id=args.dataset_id, dry_run=dry_run,
                                       allow_lossy=args.allow_lossy, max_chunk_bytes=args.max_chunk_bytes,
                                       on_error=args.on_error, indexing=args.indexing, retry_pending=args.retry_pending,
                                       chunk_retries=args.chunk_retries)
            else:
                result = verify_bundle(api, args.bundle, dataset_id=args.dataset_id, state_path=args.state,
                                       allow_lossy=args.allow_lossy, max_chunk_bytes=args.max_chunk_bytes)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 4 if result.get("skipped", 0) else 0
    except PortError as exc:
        print(f"ragwrench port: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        # Keep remote bodies, URLs containing sensitive data, and malformed private input out of logs.
        print("ragwrench port: malformed API/state/input or local I/O error; check permissions and the documented API contract", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ragwrench port: interrupted; retain checkpoint/state and resume", file=sys.stderr)
        return 130
    finally:
        if api:
            api.close()


if __name__ == "__main__":
    raise SystemExit(main())
