# ragwrench audit

A chunk export can look fine and still be quietly broken: a code block cut in half, table
rows that lost their header, a heading stranded at the end of a chunk, a section the loader
dropped. `ragwrench audit` reads the export once and prints a health report: findings per
category, the worst chunks, any source text that is in no chunk, and the top fixes, each
with the exact command to run. It is read-only and offline.

## Quick start

```sh
pip install ragwrench
cd examples/audit
ragwrench audit chunks.jsonl                          # structure only
ragwrench audit chunks.jsonl --source handbook.md     # also checks source coverage (CL007)
ragwrench audit chunks.jsonl --format json            # ragwrench.audit/v1 document
ragwrench audit ../lint --fail-on warning             # every .jsonl/.json under a directory
```

`examples/audit/chunks.jsonl` is a deliberately broken export of `handbook.md` (fictional
content); `examples/audit/generate.py` rebuilds it. With `--source` it triggers all eight
rules: 11 findings, 8 of which `ragwrench lint fix` repairs.

## Input

The same formats as [`ragwrench lint`](lint.md): JSONL or a JSON array of chunk objects.
LangChain documents (`page_content`, or the serialized `constructor`/`kwargs` envelope),
LlamaIndex nodes (`id_`, `text`) and plain `{"id", "text", "metadata"}` records are
detected automatically; `--adapter` forces one. A directory is searched recursively for
`.jsonl` and `.json` files (hidden paths and `*.changes.json` fix logs are skipped). Files
that are not chunk exports are listed under "Skipped" and do not stop the audit.

`--source` names the original document and only works with a single chunk file.

## What it reports

| Category | Rule | Notes |
| --- | --- | --- |
| Broken code fences | CL001 | error |
| Table rows without a header | CL002 | |
| Orphan headings | CL003 | |
| Mid-sentence cuts | CL004 | boundary heuristic |
| Size outliers | CL005 | `--min-chars` / `--max-chars`, default 40..4000 |
| Duplicates and heavy overlap | CL006 | skipped with a note when the export exceeds `max_pair_checks` |
| Source coverage loss | CL007 | error; needs `--source` |
| Split lists | CL008 | boundary heuristic |

Rules, thresholds and `chunklint.toml` are shared with [`ragwrench lint`](lint.md#rules-and-repairs);
`--config`, `--disable`, `--min-chars` and `--max-chars` mean the same thing, and suggested
commands repeat them.

The **auto-fix** column is measured: `audit` runs `ragwrench lint fix` in memory with the
same configuration and counts the findings that disappear. **Worst chunks** ranks chunks
by findings (errors weigh 3, warnings 1). **Source text missing from every chunk** shows
each coverage gap with source line numbers.

**Top fixes**, in order:

1. `ragwrench lint fix ... -o <name>.fixed.jsonl` when the dry run changed anything, with
   the number of findings it resolves.
2. Re-ingesting text that no chunk contains (CL007), with a `lint check --format json`
   command that lists every gap.
3. `ragwrench table-chunk split <source> ...` when tables were split without their header
   (it needs the original document; `SOURCE.md` is a placeholder when `--source` is not given).
4. Findings that need a splitter change (cuts at sentence or list boundaries, size,
   near duplicates), with the `lint check` command that lists them by line.

## Output formats and exit status

- `--format text` (default): the report above; colored on a terminal, plain otherwise or
  with `--no-color` / `NO_COLOR`.
- `--format json`: `schema: "ragwrench.audit/v1"`, a `summary` (counts, `categories`,
  `failed`), `worst_chunks`, `top_fixes`, and per-file `stats`, `issues`, `gaps` and
  `fixes`, plus `skipped` files.
- `--format sarif`: SARIF 2.1.0 with one run per file, as produced by `lint check --format sarif`.

Exit `0` when no finding is at or above `--fail-on` (`error` by default, or `fail_on` from
`chunklint.toml`; `warning`, `note`, `none` also accepted), `1` when one is, `2` on usage or
input errors (missing file, unreadable single export, `--source` with a directory).

## Boundaries

- Nothing is written. The repair preview runs in memory; run the suggested command to apply it.
- `audit` checks structure and coverage, not answer quality or embedding quality.
- The duplicate check (CL006) compares every pair of chunks within a file; very large
  exports skip it with a note unless you raise `max_pair_checks`.
