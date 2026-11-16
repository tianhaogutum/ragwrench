"""Top-level `ragwrench` dispatcher: overview, version, routing, extras and error paths."""
import importlib.util
import subprocess
import sys

import pytest

import ragwrench
from ragwrench import cli

CORE = ["audit", "lint", "table-chunk", "cite-check", "eval", "highlight", "port", "dify-bridge"]
BLOCK = """
import sys
BLOCKED = {blocked!r}
class Block:
    def find_spec(self, name, path=None, target=None):
        if any(name == root or name.startswith(root + ".") for root in BLOCKED):
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
sys.meta_path.insert(0, Block())
from ragwrench.cli import main
raise SystemExit(main(sys.argv[1:]))
"""


def run(*args, module=("-m", "ragwrench")):
    return subprocess.run([sys.executable, *module, *map(str, args)], capture_output=True, text=True, timeout=60)


def blocked(*names):
    return ("-c", BLOCK.format(blocked=names))


def installed(*modules):
    return all(importlib.util.find_spec(name) is not None for name in modules)


@pytest.mark.parametrize("args", [[], ["--help"], ["-h"], ["help"]])
def test_overview_lists_every_group_and_command(args, capsys):
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"ragwrench {ragwrench.__version__} ")
    for group in cli.GROUPS:
        assert f"\n{group}:\n" in out
    for command in cli.COMMANDS:
        assert f"\n  {command.name} " in out


def test_command_table_shape():
    assert [c.name for c in cli.COMMANDS if c.group == "Chunking"] == ["audit", "lint", "table-chunk"]
    assert [c.name for c in cli.COMMANDS if c.group == "Indexing & vector stores"] == ["sync", "bm25", "migrate"]
    assert {c.name: c.module for c in cli.COMMANDS} == {
        "audit": "ragwrench.audit.cli", "lint": "ragwrench.lint.cli", "table-chunk": "ragwrench.tablechunk.cli", "sync": "ragwrench.sync.cli",
        "bm25": "ragwrench.bm25.cli", "migrate": "ragwrench.migrate.cli", "cite-check": "ragwrench.citecheck.cli",
        "eval": "ragwrench.evalgate.cli", "highlight": "ragwrench.highlight.cli", "port": "ragwrench.port.cli",
        "dify-bridge": "ragwrench.difybridge.cli",
    }
    assert {c.name: c.extra for c in cli.COMMANDS if c.extra} == {
        "sync": "qdrant", "bm25": "qdrant", "migrate": "chroma", "dify-bridge": "qdrant"}
    assert all(c.group in cli.GROUPS and c.summary for c in cli.COMMANDS)
    assert set(cli.LEGACY.values()) == set(cli.BY_NAME) - {"audit"}  # audit is new, no former name


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version(flag, capsys):
    assert cli.main([flag]) == 0
    assert capsys.readouterr().out == f"ragwrench {ragwrench.__version__}\n"


@pytest.mark.parametrize("tool", CORE)
def test_tool_help_uses_new_program_name(tool):
    result = run(tool, "--help")
    assert result.returncode == 0, result.stderr
    assert f"usage: ragwrench {tool}" in result.stdout


@pytest.mark.parametrize("tool", CORE)
def test_help_subcommand_routes_to_tool_help(tool, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["help", tool])
    assert exc.value.code == 0
    assert f"usage: ragwrench {tool}" in capsys.readouterr().out


def test_tool_exit_codes_pass_through():
    assert run("lint", "--version").stdout == f"ragwrench lint {ragwrench.__version__}\n"
    bad = run("eval", "no-such-subcommand")
    assert bad.returncode == 2 and "usage: ragwrench eval" in bad.stderr


def test_unknown_command_suggests_close_match(capsys):
    assert cli.main(["migrat"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'migrat'" in err and "did you mean 'migrate'?" in err


@pytest.mark.parametrize("old,new", sorted(cli.LEGACY.items()))
def test_unknown_legacy_name_points_to_new_command(old, new, capsys):
    assert cli.main([old]) == 2
    assert f"{old} is now 'ragwrench {new}'" in capsys.readouterr().err


def test_unknown_command_without_match(capsys):
    assert cli.main(["zzzzzz"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'zzzzzz'" in err and "did you mean" not in err


def test_unknown_option(capsys):
    assert cli.main(["--frobnicate"]) == 2
    assert "unknown option '--frobnicate'" in capsys.readouterr().err


def test_missing_module_reports_not_available(monkeypatch, capsys):
    ghost = cli.Command("ghost", "ragwrench.ghost_tool.cli", "not shipped", "Platforms")
    monkeypatch.setitem(cli.BY_NAME, "ghost", ghost)
    monkeypatch.setattr(cli, "COMMANDS", (*cli.COMMANDS, ghost))
    assert "ghost" in cli.overview() and "(coming soon)" in cli.overview()
    assert cli.main(["ghost", "--help"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ragwrench: command 'ghost' is not available") and "Traceback" not in err


@pytest.mark.parametrize("tool,extra,block,sub", [
    ("sync", "qdrant", ("qdrant_client",), "plan"),
    ("bm25", "qdrant", ("qdrant_client",), "search"),
    ("migrate", "chroma", ("chromadb",), "verify"),
])
def test_help_without_extra_prints_usage_and_hint(tool, extra, block, sub):
    for args in ([tool, "--help"], [tool, sub, "--help"], ["help", tool]):
        result = run(*args, module=blocked(*block))
        assert result.returncode == 0, (args, result.stderr)
        assert result.stdout.startswith(f"usage: ragwrench {tool} <subcommand>")
        assert f"\n  {sub} " in result.stdout
        assert result.stdout.endswith(f"install them with: pip install 'ragwrench[{extra}]'\n")


@pytest.mark.parametrize("tool,extra,block", [
    ("sync", "qdrant", ("qdrant_client",)),
    ("bm25", "qdrant", ("qdrant_client",)),
    ("migrate", "chroma", ("chromadb",)),
    ("migrate", "chroma", ("qdrant_client",)),
])
def test_command_without_extra_prints_install_hint(tool, extra, block):
    result = run(tool, "inspect", "x", module=blocked(*block))
    assert result.returncode == 2 and result.stdout == ""
    assert result.stderr == (f"ragwrench {tool}: missing optional dependencies; "
                             f"install them with: pip install 'ragwrench[{extra}]'\n")
    overview = run("--help", module=blocked(*block))
    assert f"(needs ragwrench[{extra}])" in overview.stdout


def test_dify_bridge_does_not_require_qdrant_up_front():
    result = run("dify-bridge", "--help", module=blocked("qdrant_client"))
    assert result.returncode == 0, result.stderr
    assert "usage: ragwrench dify-bridge" in result.stdout
    assert "missing optional dependencies" not in result.stdout + result.stderr
    overview = run("--help", module=blocked("qdrant_client")).stdout
    # sync/bm25 legitimately carry the hint; the dify-bridge line must not.
    (line,) = [line for line in overview.splitlines() if line.strip().startswith("dify-bridge ")]
    assert "needs ragwrench[" not in line


@pytest.mark.parametrize("module", ["ragwrench.audit", "ragwrench.lint", "ragwrench.tablechunk", "ragwrench.citecheck",
                                    "ragwrench.evalgate", "ragwrench.highlight", "ragwrench.port"])
def test_subpackage_main_routes_through_dispatcher(module):
    result = run("--help", module=("-m", module))
    assert result.returncode == 0 and "usage: ragwrench " in result.stdout


SUBCOMMANDS = {
    "audit": [],
    "lint": ["check", "fix", "plan", "preview", "apply", "validate-plan", "bench"],
    "table-chunk": ["split", "bench"],
    "sync": ["snapshot", "plan", "compare", "compare-policies", "review", "policy-override", "inspect-policy",
             "list-policies", "apply", "verify", "bench"],
    "bm25": ["encode", "index", "search", "update", "recover", "status", "verify-document", "verify-documents",
             "export-documents", "eval"],
    "migrate": ["copy", "verify", "inspect"],
    "cite-check": ["check", "bench"],
    "eval": ["run", "compare", "compare-many", "compare-saved", "compare-many-saved", "validate-preflight",
             "validate-evaluation", "diff-preflight", "convert"],
    "highlight": ["normalize", "consolidate", "annotate", "preflight", "resolve", "check-audit", "manifest",
                  "verify", "check-manifest", "export", "check-sidecar", "import", "list-annotations", "remove"],
    "port": ["validate", "import-status", "capabilities", "export", "import", "verify"],
    "dify-bridge": ["serve", "check", "snapshot-status"],
}
ALL_HELP = [(name, None) for name in SUBCOMMANDS] + [(name, sub) for name, subs in SUBCOMMANDS.items() for sub in subs]


def _main(argv, capsys):
    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_subcommand_table_covers_every_command():
    assert set(SUBCOMMANDS) == set(cli.BY_NAME)


def test_extra_subcommand_lists_match_table():
    from ragwrench import bm25, migrate, sync
    assert list(sync.SUBCOMMANDS) == SUBCOMMANDS["sync"]
    assert list(bm25.SUBCOMMANDS) == SUBCOMMANDS["bm25"]
    assert list(migrate.SUBCOMMANDS) == SUBCOMMANDS["migrate"]


@pytest.mark.parametrize("flag", ["--help", "-h"])
@pytest.mark.parametrize("command,sub", ALL_HELP, ids=lambda v: v or "-")
def test_every_command_and_subcommand_help_exits_0(command, sub, flag, capsys):
    if not installed(*cli.BY_NAME[command].requires):
        pytest.skip(f"needs ragwrench[{cli.BY_NAME[command].extra}]")
    argv = [command, *([sub] if sub else []), flag]
    code, out, err = _main(argv, capsys)
    assert code == 0, (argv, err)
    assert "usage" in out.lower(), (argv, out)
    if sub:
        assert f"ragwrench {command} {sub}" in out, (argv, out)
