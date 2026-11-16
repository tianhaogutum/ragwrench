import argparse
import json
import os
import sys

from .check import check_endpoint, check_request_file
from .config import load_config
from .credentials import environment_key
from .errors import ConfigError
from .jsonutil import loads
from .service import Application, make_server
from .snapshot import SnapshotStatusError, snapshot_status


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            # argparse normally echoes unknown arguments, which may be secrets.
            # Subparsers inherit this class and the same invocation context.
            if argv and argv[0] in {"snapshot-status", "check"}:
                command = argv[0]
                self.exit(2, f"Configuration failure: invalid {command} arguments; use {command} --help.\n")
            super().error(message)

    parser = Parser(prog="ragwrench dify-bridge", description="Dify External Knowledge API bridge and conformance checker")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Serve configured knowledge sources")
    serve.add_argument("--config", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    check = sub.add_parser("check", help="Send read-only retrieval/auth/error probes to an endpoint")
    check.add_argument("--endpoint", required=True, help="Base URL, without /retrieval")
    check.add_argument("--knowledge-id")
    check.add_argument("--request-file", help="Complete saved UTF-8 retrieval request (max 1 MiB); excludes inline request flags")
    check.add_argument("--query", help="Baseline query; use a known-hit query unless expecting/allowing empty results")
    check.add_argument("--key-env", default="BRIDGE_API_KEY")
    check.add_argument("--timeout", type=float, default=10.0)
    check.add_argument("--allow-empty", action="store_true")
    check.add_argument("--metadata-condition", help="Dify metadata_condition JSON object (not a filename)")
    check.add_argument("--top-k", type=int, help="Baseline top_k (1..100; default 3)")
    check.add_argument("--score-threshold", type=float, help="Baseline score threshold (0..1; default 0)")
    check.add_argument("--expect-count", type=int, help="Require exactly this many baseline records, including zero")
    check.add_argument("--expect-title", action="append", default=[], help="Require a baseline title; repeat for multiple titles")
    status = sub.add_parser("snapshot-status", help="Read the active JSONL snapshot version and record count")
    status.add_argument("--endpoint", required=True, help="HTTP(S) base URL, without /snapshot/status")
    status.add_argument("--knowledge-id", required=True)
    status.add_argument("--key-env", default="BRIDGE_API_KEY", help="Read the key only from this environment variable")
    status.add_argument("--timeout", default="10", help="Request timeout in seconds (0,120]; default 10")
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot-status":
            try:
                report = snapshot_status(args.endpoint, args.knowledge_id,
                                         key_env=args.key_env, timeout=args.timeout)
            except SnapshotStatusError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            print(json.dumps(report))
            return 0
        if args.command == "check":
            key = environment_key(args.key_env, context="Checker")
            if not 0 < args.timeout <= 120:
                raise ConfigError("Checker timeout must be in (0,120]")
            inline = (args.knowledge_id, args.query, args.metadata_condition, args.top_k, args.score_threshold)
            options = dict(allow_empty=args.allow_empty, timeout=args.timeout,
                           expect_count=args.expect_count, expect_titles=args.expect_title)
            if args.request_file is not None:
                if any(value is not None for value in inline):
                    raise ConfigError("Checker request-file cannot be combined with inline request flags")
                report = check_request_file(args.endpoint, key, args.request_file, **options)
            else:
                metadata_condition = None
                if args.metadata_condition is not None:
                    try:
                        metadata_condition = loads(args.metadata_condition)
                        if not isinstance(metadata_condition, dict):
                            raise ValueError
                    except (ValueError, RecursionError):
                        raise ConfigError("Checker metadata_condition must be a valid JSON object without duplicate keys") from None
                report = check_endpoint(args.endpoint, key, args.knowledge_id, args.query,
                                        metadata_condition=metadata_condition,
                                        top_k=3 if args.top_k is None else args.top_k,
                                        score_threshold=0.0 if args.score_threshold is None else args.score_threshold,
                                        **options)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 1
        config = load_config(args.config)
        app = Application(config, os.environ.get("BRIDGE_API_KEY"))
        try:
            server = make_server(app, args.host, args.port)
            try:
                print(f"ragwrench dify-bridge listening on {server.server_address[0]}:{server.server_address[1]}", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
            finally:
                server.server_close()
        finally:
            app.close()
        return 0
    except ConfigError as exc:
        print(f"Configuration failure: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError):
        print("Configuration/startup failure: check config types, source files, environment keys and bind address.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
