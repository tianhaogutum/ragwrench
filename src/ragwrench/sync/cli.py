from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import SUBCOMMANDS
from .bench import run_bench
from .core import (NAMESPACE, SyncError, build_snapshot, has_changes, load_json, make_plan,
                   snapshot_dir, snapshot_jsonl, validate_plan, validate_snapshot, write_json)
from .embedders import CommandEmbedder, HashEmbedder, PrecomputedEmbedder
from .inventory import archive_inventory, archive_path, inventory_json, list_policies
from .policy import policy_override, validate_policy_override
from .policy_comparison import (compare_policies, comparison_archive_path, comparison_json,
                                load_inventory)
from .review import GATES, compare_plans, require_allowed, review_plan, validate_review
from .store import apply_plan, inspect_policy, local_client, validate_collection_name, verify_snapshot


def review_options(p):
    p.add_argument("--max-deletion-fraction", type=float, help="maximum unmatched old chunks / old chunks (default: 1.0)")
    p.add_argument("--override-gate", action="append", choices=GATES, default=[], help="explicitly override one violated gate; repeat for both")
    p.add_argument("--reviewer", help="caller-stated identity for overrides")
    p.add_argument("--reason", help="specific reason for overrides")


def make_review(args, plan):
    if getattr(args, "review", None):
        if (args.max_deletion_fraction is not None or args.override_gate
                or args.reviewer is not None or args.reason is not None):
            raise SyncError("--review cannot be combined with review policy/override arguments")
        return validate_review(plan, load_json(args.review))
    return review_plan(plan, max_deletion_fraction=1.0 if args.max_deletion_fraction is None else args.max_deletion_fraction,
                       overrides=args.override_gate, reviewer=args.reviewer, reason=args.reason)


def parser():
    p = argparse.ArgumentParser(prog="ragwrench sync", description="Plan and apply incremental local Qdrant indexes")
    p.add_argument("--version", action="version", version="ragwrench sync 0.1.0")
    commands = p.add_subparsers(dest="command", required=True)
    s = commands.add_parser("snapshot", help="snapshot UTF-8 documents or already split JSONL chunks")
    inputs = s.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dir")
    inputs.add_argument("--chunks", help="JSONL rows: doc_id, text, optional metadata and embed_metadata")
    inputs.add_argument("--empty", action="store_true", help="initial snapshot with no points")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--embed-model", default="hash-v1")
    s.add_argument("--namespace", default=NAMESPACE)
    s.add_argument("--chunker", choices=["fixed", "cdc"], default="cdc")
    s.add_argument("--min-size", type=int, default=128)
    s.add_argument("--target-size", type=int, default=256)
    s.add_argument("--max-size", type=int, default=512)
    s = commands.add_parser("plan", help="read-only diff; exit 10 if snapshots differ")
    s.add_argument("old")
    s.add_argument("new")
    s.add_argument("-o", "--output", help="write a self-contained apply plan")
    s.add_argument("--format", choices=["text", "json"], default="text")
    s = commands.add_parser("compare", help="compare two plans from the same source; exit 10 if different")
    s.add_argument("baseline")
    s.add_argument("candidate")
    s.add_argument("-o", "--output")
    s = commands.add_parser("compare-policies", help="compare archived inventories without store access; exit 10 for drift, 2 for invalid input, opt-in 3 or 4 gates")
    s.add_argument("before", help="version-1 policy_inventory JSON")
    s.add_argument("after", help="version-1 policy_inventory JSON")
    s.add_argument("-o", "--output", help="atomically archive exact stdout JSON; existing parent, no symlink traversal or aliases of either input")
    s.add_argument("--fail-on-diagnostics", action="store_true", help="exit 3 if either complete input has any non-ok row; still emit/archive the full comparison (global errors exit 2)")
    s.add_argument("--fail-on-policy-weakening", action="store_true", help="exit 4 if an exact matched healthy collection increases its effective deletion limit; still emit/archive the full comparison (global errors 2, requested diagnostics 3 take priority)")
    s = commands.add_parser("review", help="evaluate plan gates; exit 3 if blocked, 2 for invalid input")
    s.add_argument("input", help="plan JSON")
    s.add_argument("-o", "--output", help="write a plan-bound review receipt for apply --review")
    review_options(s)
    s = commands.add_parser("policy-override", help="record a separate plan/collection-bound policy weakening exception")
    s.add_argument("input", help="plan JSON")
    s.add_argument("--collection", default="kb")
    s.add_argument("--from-max-deletion-fraction", type=float, required=True)
    s.add_argument("--max-deletion-fraction", type=float, required=True)
    s.add_argument("--reviewer", required=True)
    s.add_argument("--reason", required=True)
    s.add_argument("-o", "--output", required=True)
    s = commands.add_parser("inspect-policy", help="read a local collection's effective policy and latest exception; exit 2 for invalid/missing state")
    s.add_argument("--qdrant-path", required=True, help="existing persistent local directory (opening may repair damaged metadata)")
    s.add_argument("--collection", default="kb")
    s.add_argument("--no-repair", action="store_true", help="refuse missing/undecodable primary metadata without repairing it; normal lock handling still applies")
    s = commands.add_parser("list-policies", help="inventory local catalog policies without opening points; per-collection diagnostics are returned as rows")
    s.add_argument("--qdrant-path", required=True, help="existing offline local store; at most 1000 collections")
    s.add_argument("--no-repair", action="store_true", help="require primary metadata; default may read validated recovery metadata in memory, never repairs files")
    s.add_argument("-o", "--output", help="atomically archive exact stdout JSON outside the store; existing parent and no symlink traversal required")
    s.add_argument("--fail-on-diagnostics", action="store_true", help="exit 3 if any collection row is non-ok; still emit/archive the complete inventory (global errors exit 2)")
    for name in ("apply", "verify"):
        s = commands.add_parser(name, help=SUBCOMMANDS[name])
        s.add_argument("input", help="plan JSON" if name == "apply" else "snapshot JSON")
        s.add_argument("--qdrant-path", required=True, help="local directory or :memory:")
        s.add_argument("--collection", default="kb")
        s.add_argument("--embedder", choices=["hash", "command", "precomputed"], default="hash")
        s.add_argument("--dimension", type=int, default=32)
        s.add_argument("--embed-command", help='JSON argv array, e.g. ["python3", "embed.py"] (never a shell string)')
        s.add_argument("--vectors", help="precomputed JSONL with id and vector")
        s.add_argument("--embed-timeout", type=float, default=30)
        if name == "apply":
            review_options(s)
            s.add_argument("--review", help="use an exact review receipt, including policy and override provenance")
            s.add_argument("--policy-override", help="separate policy weakening record JSON (not a plan-gate override)")
            s.add_argument("--dry-run", action="store_true", help="validate plan only; no DB or embedder execution")
            s.add_argument("--verify", action="store_true", help="also recompute and compare all vectors")
    s = commands.add_parser("bench", help="replay synthetic edit scenarios and write measured counts")
    s.add_argument("-o", "--output", default="bench-results.json")
    s.add_argument("--seed", type=int, default=25)
    s.add_argument("--paragraphs", type=int, default=60)
    return p


def make_embedder(args):
    if not 1 <= args.dimension <= 4096:
        raise SyncError("dimension must be between 1 and 4096")
    if args.embedder != "command" and args.embed_command:
        raise SyncError("--embed-command requires --embedder command")
    if args.embedder != "precomputed" and args.vectors:
        raise SyncError("--vectors requires --embedder precomputed")
    if args.embedder == "hash":
        return HashEmbedder(args.dimension)
    if args.embedder == "precomputed":
        if not args.vectors:
            raise SyncError("--vectors is required")
        return PrecomputedEmbedder(args.vectors, args.dimension)
    if not args.embed_command:
        raise SyncError("--embed-command JSON argv is required")
    try:
        argv = json.loads(args.embed_command)
    except ValueError as exc:
        raise SyncError("--embed-command must be a JSON argv array") from exc
    return CommandEmbedder(argv, args.dimension, args.embed_timeout)


def execute(args):
    if args.command == "snapshot":
        opts = {"embed_model": args.embed_model, "namespace": args.namespace}
        if args.empty:
            snapshot = build_snapshot([], **opts)
        elif args.chunks:
            snapshot = snapshot_jsonl(args.chunks, **opts)
        else:
            snapshot = snapshot_dir(args.dir, method=args.chunker, min_size=args.min_size,
                                    target_size=args.target_size, max_size=args.max_size, **opts)
        write_json(args.output, snapshot)
        print(json.dumps({"chunks": len(snapshot["chunks"]), "fingerprint": snapshot["fingerprint"]}))
        return 0
    if args.command == "plan":
        plan = make_plan(load_json(args.old), load_json(args.new))
        if args.output:
            write_json(args.output, plan)
        if args.format == "json":
            print(json.dumps(plan, indent=2, ensure_ascii=False))
        else:
            s = plan["summary"]
            print(f"Plan: {s['upsert']} upsert, {s['delete']} delete, {s['unchanged']} unchanged, "
                  f"{s['payload_only']} payload-only, {s['moved']} moved")
            total, avoided = s["new_chunks"], s["embeddings_avoided"]
            print(f"Embeddings avoided vs full re-index: {avoided}/{total} ({100 * avoided / total if total else 100:.1f}%)")
            if s["embed_model_changed"]:
                print("Embedding model tag changed: all surviving content requires embedding.")
        return 10 if has_changes(plan) else 0
    if args.command == "compare":
        comparison = compare_plans(load_json(args.baseline), load_json(args.candidate))
        if args.output:
            write_json(args.output, comparison)
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        return 0 if comparison["identical"] else 10
    if args.command == "compare-policies":
        target = comparison_archive_path(args.output, args.before, args.after) if args.output is not None else None
        before, after = load_inventory(args.before), load_inventory(args.after)
        result = compare_policies(before, after)
        encoded = comparison_json(result)
        if target is not None:
            archive_inventory(target, encoded)
        print(encoded, end="")
        # Unchanged diagnostics are absent from the diff but still count.
        # Both complete inputs and publication must succeed before exit 3.
        if args.fail_on_diagnostics and any(row["status"] != "ok"
                                            for archive in (before, after)
                                            for row in archive["collections"]):
            return 3
        # Changed rows already contain normalized limits from both full archives.
        # An increase must change its row; additions/removals are not comparable.
        if args.fail_on_policy_weakening and any(
                change["before"]["status"] == change["after"]["status"] == "ok"
                and change["after"]["policy"]["max_deletion_fraction"]
                > change["before"]["policy"]["max_deletion_fraction"]
                for change in result["changed"]):
            return 4
        return 0 if result["identical"] else 10
    if args.command == "review":
        receipt = make_review(args, load_json(args.input))
        if args.output:
            write_json(args.output, receipt)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 0 if receipt["allowed"] else 3
    if args.command == "policy-override":
        record = policy_override(load_json(args.input), args.collection,
                                 from_max_deletion_fraction=args.from_max_deletion_fraction,
                                 max_deletion_fraction=args.max_deletion_fraction,
                                 reviewer=args.reviewer, reason=args.reason)
        write_json(args.output, record)
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    if args.command == "bench":
        result = run_bench(args.seed, args.paragraphs)
        write_json(args.output, result)
        print("scenario             full  doc-hash  fixed  cdc (new vectors)")
        for row in result["results"]:
            counts = row["strategies"]
            print(f"{row['scenario']:20} " + "  ".join(f"{counts[k]['embedded']:5}" for k in ("full_reindex", "document_hash", "chunk_fixed", "chunk_cdc")))
        print(f"Results: {args.output}")
        return 0
    if args.command == "list-policies":
        target = archive_path(args.output, args.qdrant_path) if args.output is not None else None
        result = list_policies(args.qdrant_path, repair_metadata=not args.no_repair)
        encoded = inventory_json(result)
        if target is not None:
            archive_inventory(target, encoded)
        print(encoded, end="")
        return 3 if args.fail_on_diagnostics and any(row["status"] != "ok" for row in result["collections"]) else 0
    validate_collection_name(args.collection)
    if args.command == "inspect-policy":
        root = Path(args.qdrant_path)
        if (args.qdrant_path == ":memory:" or not root.is_dir()
                or not any((root / name).is_file() for name in ("meta.json", "chunksync-recovery.json"))):
            raise SyncError("inspect-policy requires an existing persistent Qdrant store")
        with local_client(args.qdrant_path, repair_metadata=not args.no_repair) as client:
            result = inspect_policy(client, args.collection)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    value = load_json(args.input)
    if args.command == "apply":
        validate_plan(value)
        review = make_review(args, value)
        require_allowed(review)
        override = load_json(args.policy_override) if args.policy_override else None
        if args.policy_override is not None:
            validate_policy_override(value, args.collection, override, review["policy"])
        if args.dry_run:
            print(json.dumps({"status": "dry_run", "summary": value["summary"], "collection_checked": False,
                              "review": review, "policy_override": override}))
            return 0
    else:
        validate_snapshot(value)
        if args.qdrant_path == ":memory:":
            raise SyncError("verify cannot reopen :memory:; use apply --verify or the Python API")
    embedder = make_embedder(args)
    with local_client(args.qdrant_path) as client:
        if args.command == "apply":
            result = apply_plan(client, args.collection, value, embedder, review=review, policy_override=override)
            if args.verify:
                result["verification"] = verify_snapshot(client, args.collection, value["new"], embedder)
        else:
            result = verify_snapshot(client, args.collection, value, embedder)
    print(json.dumps(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (SyncError, OSError, UnicodeError, RuntimeError) as exc:
        print(f"ragwrench sync: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
