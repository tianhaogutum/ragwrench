"""Top-level ``ragwrench`` command: dispatches to one tool per subcommand.

Each tool lives in its own subpackage exposing ``cli.main(argv) -> int``. Tools
are imported lazily, so ``ragwrench --help`` stays fast and a missing optional
dependency only affects the command that needs it.
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib
import importlib
import importlib.util
import sys

from . import __version__

PROG = "ragwrench"
REPO_URL = "https://github.com/tianhaogutum/ragwrench"


@dataclass(frozen=True)
class Command:
    name: str
    module: str
    summary: str
    group: str
    extra: str | None = None
    requires: tuple[str, ...] = ()


GROUPS = ("Chunking", "Indexing & vector stores", "Answers & evaluation", "Citations & PDFs", "Platforms")

COMMANDS: tuple[Command, ...] = (
    Command("audit", "ragwrench.audit.cli",
            "One read-only health report for a chunk export, with the command that fixes each problem",
            "Chunking"),
    Command("lint", "ragwrench.lint.cli",
            "Lint Markdown RAG chunks against their source and conservatively repair bad cuts", "Chunking"),
    Command("table-chunk", "ragwrench.tablechunk.cli",
            "Split Markdown/HTML documents into table-aware chunks with provenance", "Chunking"),
    Command("sync", "ragwrench.sync.cli",
            "Plan, review and apply incremental chunk updates to a local Qdrant index",
            "Indexing & vector stores", extra="qdrant", requires=("qdrant_client",)),
    Command("bm25", "ragwrench.bm25.cli",
            "CJK-aware BM25 sparse vectors and search in Qdrant local mode",
            "Indexing & vector stores", extra="qdrant", requires=("qdrant_client",)),
    Command("migrate", "ragwrench.migrate.cli",
            "Migrate Chroma <-> Qdrant local collections and verify parity",
            "Indexing & vector stores", extra="chroma", requires=("chromadb", "qdrant_client")),
    Command("cite-check", "ragwrench.citecheck.cli",
            "Check citations in RAG answers (Dify, RAGFlow, LlamaIndex, JSONL) against sources",
            "Answers & evaluation"),
    Command("eval", "ragwrench.evalgate.cli",
            "Score retrieval runs (TREC/JSONL) and gate regressions with paired statistics",
            "Answers & evaluation"),
    Command("highlight", "ragwrench.highlight.cli",
            "Turn Docling/Unstructured provenance into PDF highlights and citation manifests",
            "Citations & PDFs"),
    Command("port", "ragwrench.port.cli",
            "Export, import and verify hash-checked knowledge bases between Dify and RAGFlow",
            "Platforms"),
    # Only Qdrant-backed knowledge needs the extra; JSONL serving and `check` work without it,
    # so nothing is required up front (the Qdrant backend reports the install hint itself).
    Command("dify-bridge", "ragwrench.difybridge.cli",
            "Serve Qdrant/JSONL as a Dify External Knowledge API and check endpoints",
            "Platforms", extra="qdrant"),
)
BY_NAME = {command.name: command for command in COMMANDS}

# Former standalone tool names, used only to point users at the new subcommand.
LEGACY = {
    "citehl": "highlight", "kbport": "port", "chunklint": "lint", "tablechunk": "table-chunk",
    "chunksync": "sync", "qbm25": "bm25", "vecparity": "migrate", "citecheck": "cite-check",
    "retgate": "eval", "difykb-bridge": "dify-bridge",
}


def _present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _installed(command: Command) -> bool:
    return _present(command.module.rpartition(".")[0])


def _missing_requirements(command: Command) -> bool:
    return any(not _present(name) for name in command.requires)


def _install_hint(command: Command) -> str:
    return (f"{PROG} {command.name}: missing optional dependencies; "
            f"install them with: pip install '{PROG}[{command.extra}]'")


def _help_without_extra(command: Command) -> str:
    """A usage summary that needs only the command's package, not its optional extra."""
    lines = [f"usage: {PROG} {command.name} <subcommand> [args...]", "", command.summary + "."]
    try:
        subcommands = getattr(importlib.import_module(command.module.rpartition(".")[0]), "SUBCOMMANDS", {})
    except ImportError:
        subcommands = {}
    if subcommands:
        width = max(map(len, subcommands)) + 2
        lines += ["", "subcommands:", *(f"  {name.ljust(width)}{text}" for name, text in subcommands.items())]
    return "\n".join(lines) + "\n\n"


def overview() -> str:
    width = max(len(command.name) for command in COMMANDS) + 2
    lines = [
        f"{PROG} {__version__} - offline QA and plumbing tools for RAG pipelines",
        "",
        f"usage: {PROG} <command> [args...]",
        f"       {PROG} <command> --help",
    ]
    for group in GROUPS:
        lines += ["", f"{group}:"]
        for command in COMMANDS:
            if command.group != group:
                continue
            note = ""
            if not _installed(command):
                note = "  (coming soon)"
            elif command.extra and _missing_requirements(command):
                note = f"  (needs {PROG}[{command.extra}])"
            lines.append(f"  {command.name.ljust(width)}{command.summary}{note}")
    lines += [
        "",
        "options:",
        "  -h, --help     show this overview",
        "  -V, --version  show the version and exit",
        "",
        f"Docs and issues: {REPO_URL}",
    ]
    return "\n".join(lines) + "\n"


def _unknown(name: str) -> int:
    message = f"{PROG}: unknown command {name!r}"
    if name in LEGACY:
        message += f"; {name} is now '{PROG} {LEGACY[name]}'"
    else:
        match = difflib.get_close_matches(name, [*BY_NAME, *LEGACY], n=1, cutoff=0.6)
        if match:
            suggestion = LEGACY.get(match[0], match[0])
            message += f"; did you mean '{suggestion}'?"
    sys.stderr.write(message + f"\nRun '{PROG} --help' to list commands.\n")
    return 2


def run(command: Command, argv: list[str]) -> int:
    if not _installed(command):
        sys.stderr.write(f"{PROG}: command {command.name!r} is not available in "
                         f"{PROG} {__version__} yet; see {REPO_URL}\n")
        return 2
    if command.extra and _missing_requirements(command):
        if any(arg in ("-h", "--help") for arg in argv):
            sys.stdout.write(_help_without_extra(command))
            sys.stdout.write(_install_hint(command) + "\n")
            return 0
        sys.stderr.write(_install_hint(command) + "\n")
        return 2
    try:
        module = importlib.import_module(command.module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if command.extra and any(req == missing or req.startswith(missing + ".") for req in command.requires):
            sys.stderr.write(_install_hint(command) + "\n")
            return 2
        raise
    result = module.main(argv)
    return 0 if result is None else result


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(overview())
        return 0
    first, rest = args[0], args[1:]
    if first in ("-V", "--version"):
        sys.stdout.write(f"{PROG} {__version__}\n")
        return 0
    if first == "help":
        if not rest:
            sys.stdout.write(overview())
            return 0
        first, rest = rest[0], ["--help"]
    if first.startswith("-"):
        sys.stderr.write(f"{PROG}: unknown option {first!r}\nRun '{PROG} --help' for usage.\n")
        return 2
    command = BY_NAME.get(first)
    if command is None:
        return _unknown(first)
    return run(command, rest)


if __name__ == "__main__":
    raise SystemExit(main())
