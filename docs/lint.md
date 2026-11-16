# ragwrench lint

Text splitters silently break Markdown when they cut it into chunks: code fences lose
their closing marker, tables lose their header row, headings are left orphaned at the end
of a chunk and text goes missing between chunks. `ragwrench lint` finds these defects in
CI, whatever splitter produced the chunks, and conservatively repairs the safe cases. Use
it after chunking and before embedding, or as a gate when you change splitter settings.

_Formerly the standalone `chunk-lint` tool (`chunklint`)._

Audit Markdown RAG chunk sets in CI, independently of the splitter that made
them. Recover source spans, detect broken structures and missing text, apply
conservative deterministic repairs, and emit text, JSON or SARIF 2.1.0 reports.
Everything runs locally.

## Quick start

```bash
pip install ragwrench
ragwrench lint --help
ragwrench lint check examples/lint/chunks_langchain.jsonl --source examples/lint/doc.md
# Exit 1 is expected: this fixture deliberately has broken fences.

ragwrench lint fix examples/lint/chunks_langchain.jsonl --source examples/lint/doc.md \
  --min-chars 0 --max-chars 160 -o fixed.jsonl
ragwrench lint check fixed.jsonl --source examples/lint/doc.md \
  --min-chars 0 --max-chars 160 --fail-on warning
ragwrench lint check fixed.jsonl --source examples/lint/doc.md --format sarif > audit.sarif
```

The fix writes **generic JSONL** and `fixed.jsonl.changes.json`, containing every
change, skipped edit, before/after report and conservation checks. It does not
modify the input. To review edits without creating files:

```bash
ragwrench lint fix examples/lint/chunks_langchain.jsonl --source examples/lint/doc.md \
  --dry-run --format json
```

The installed package includes the benchmark fixture, with no repository
dependency. Tests and all CLI operations need no network, credentials, models, or
service.

## Saved repair plans

Save an exact repair for review, then preview or apply it against the original
input and source snapshots:

```bash
ragwrench lint plan examples/lint/chunks_langchain.jsonl --source examples/lint/doc.md \
  --min-chars 0 --max-chars 160 -o repairs.plan.json
ragwrench lint preview examples/lint/chunks_langchain.jsonl --source examples/lint/doc.md \
  --plan repairs.plan.json
ragwrench lint apply examples/lint/chunks_langchain.jsonl --source examples/lint/doc.md \
  --plan repairs.plan.json -o planned-fixed.jsonl --format json
```

Validate a saved plan in CI **before chunk/source files are available**:

```bash
ragwrench lint validate-plan repairs.plan.json
ragwrench lint validate-plan repairs.plan.json --format json
ragwrench lint validate-plan --schema > fix-plan-v1.schema.json
```

`validate-plan` reads only the named plan, ignores ambient configuration, runs
no repair, and writes no files. Valid plans exit **0**, regardless of remaining
lint findings or the saved failure threshold. Invalid plans exit **2**. Text
success is `Valid fix plan v1: 1 edits (snapshots not checked)`; errors go to
stderr with a field path, for example `$.edits[0].replacement[0].text`.
JSON mode writes either
`{"valid":true,"version":1,"edits":1,"snapshots_checked":false}` or
`{"valid":false,"diagnostics":[{"path":"$.input.sha256","message":"must be a lowercase 64-digit SHA256"}]}`
to stdout, with empty stderr. Only the first error is reported; JSON/file read
errors use path `$`. Argument usage errors retain argparse's stderr/exit-2
behavior. `--schema` accepts neither a filename nor `--format`.

The published [v1 JSON Schema](../src/ragwrench/lint/schemas/fix-plan-v1.schema.json)
uses Draft 2020-12 and ships in the wheel/sdist. `--schema` prints that same
schema for use with an independent validator; it needs no network resolution.
It defines required/unknown fields, types, hashes, bounds and replacement record
shape (including paired optional character hints). `validate-plan` additionally
checks configuration relationships, ordered nonoverlapping ranges in both
coordinate systems, hint ordering, unique replacement IDs and compatibility
with the installed tool version. It requires exact JSON integer types (rejects
`1.0`/`true` in integer fields); standard JSON Schema accepts integral floats.
The strict JSON reader also rejects duplicate keys, nonfinite numbers and
invalid UTF-8. The 32 MiB regular-file limit applies to plans.

Success does **not** verify hash contents, snapshot freshness, byte/chunk ranges
against the original stream, IDs against untouched chunks, or repair/conservation
correctness. Use `preview` or `apply` with original snapshots for those checks.
Preview/apply load and structurally validate the saved plan **before reading
either input or source**. A malformed plan therefore reports its field path
(for example `$.edits[0].replacement[0].text`) even when a snapshot path is
unavailable. No chunk/source reads, repair or output writes occur on that
failure. Once the plan passes preflight, snapshot reads, freshness checks and
deterministic repair validation proceed as usual. These commands report errors
on stderr and exit 2, including with `--format json`; structured JSON diagnostics
are specific to `validate-plan`. Argument parsing still precedes plan preflight.
Plan generation keeps its existing configuration/input/source validation order.

For a minimal example, input records `{"id":"h","text":"# Café"}` and
`{"id":"p","text":"Body."}` with source `# Café\nBody.` and `--min-chars 0`
produce one replacement of chunks `[0,2)`: chunk `h` has text `# Café\nBody.`,
both metadata objects survive in merge provenance, and coverage stays 100%.
Changing either input record (including metadata), input JSON formatting, or
even just source line endings makes preview/apply exit **2** before any write.
A pre-existing output remains byte-identical on rejection. Regenerate the plan
after intentional input changes.

The saved file is JSON with this explicit v1 format:

| Field | Meaning |
| --- | --- |
| `format`, `version`, `tool_version` | `"chunklint.fix-plan"`, integer `1`, generating ragwrench lint version |
| `coordinates` | `"canonical-generic-jsonl-utf8-v1"` |
| `input` | Raw-file `sha256` (lowercase hex), `bytes` (length), and selected `adapter` |
| `source` | Raw-file `sha256` and `bytes`, or `null` when no source was used |
| `config` | All effective configuration values, including rule disables, budgets and exit threshold |
| `edits` | Ordered, nonoverlapping replacements; each has `start_chunk`, `end_chunk`, `start_byte`, `end_byte`, `before_sha256`, and `replacement` (array of complete generic records; empty means deletion) |
| `output_sha256` | SHA256 of the exact canonical UTF-8 output |

Chunk and byte ranges are zero-based, half-open, and refer to the **original
canonical generic JSONL stream**, not to source Markdown, raw framework JSON,
or Unicode character offsets. Canonicalization adapts each record to
`{id,text,metadata}` with retained optional character hints, sorts JSON
keys recursively, uses compact separators (`,` and `:`), preserves literal
Unicode, and terminates every record with LF. Specifically, each record uses
Python `json.dumps(ensure_ascii=False, sort_keys=True, separators=(',', ':'),
allow_nan=False)`. UTF-8 encoding determines byte offsets. `before_sha256` hashes
the bytes in that edit's original range. Adjacent changed records may share a
single replacement; unchanged records anchor separate edits. BOM, CRLF and other
raw formatting still participate in the input/source snapshot hashes.

Preview shows both removed and replacement records plus their exact ranges;
it writes nothing. `--format json` returns the plan and before/after repair
report. Apply validates every snapshot, recomputes the deterministic repair and
its conservation guards, requires all plan fields/edits to match, then splices
the specified byte ranges and verifies the output hash. All validation finishes
before one atomic output replacement. Apply creates no change-log sidecar and
never modifies the input/source/plan. Paths must be supplied explicitly; no
paths or commands in the plan are executed. Plans and previews contain payload
text/metadata, so treat them as input-sensitive.

Preview/apply use the saved configuration, ignoring ambient `chunklint.toml`;
they accept no rule/config overrides. Supply `--source` exactly when the plan
was created with one. Plan/preview/apply use the **after** report's saved failure
threshold, so exit 1 can accompany a successfully saved plan or applied output
with remaining findings. Inconsistent edits, unknown fields/versions, malformed
JSON and stale snapshots exit 2. Plans are for deterministic ragwrench lint repairs;
custom manual replacement text requires a new input and plan. Preview/apply
rerun repair validation, and plans are tied to their generating tool version.
Saved plans are limited to 32 MiB; oversized generation fails without writing.
Snapshot checks operate on bytes read for the invocation; concurrent writers
are not locked. As with `fix`, output is generic JSONL, including for no-op plans.

## Input and output

Input must be ordered UTF-8 JSONL or a JSON array, one document per invocation.
Blank JSONL lines are ignored; errors name the JSONL line or array item.

```json
{"id":"c1","text":"## Usage\nRun ragwrench lint locally.","metadata":{"source":"guide.md"}}
```

| Adapter | Accepted content and identity |
| --- | --- |
| `generic` | `text`, optional `id`, optional object `metadata` |
| `langchain` | `page_content`, optional `id`, `metadata`; also a `type: "constructor"` data envelope with `kwargs` |
| `llamaindex` | TextNode `text`, `id_`, `metadata`, optional paired `start_char_idx` / `end_char_idx` |
| `auto` (default) | LangChain when `page_content`/`kwargs` exists, otherwise generic/TextNode |

Constructor names are **never imported or executed**. Unrecognized top-level
framework fields (embeddings, relationships, framework types) are not exported;
this is a text/metadata interchange format, not a lossless framework serializer.
Use `--adapter` to override detection. Dify/RAGFlow exports need conversion to
`{id,text,metadata}`; their native envelopes are not v1 adapters. Empty sets,
blank text, duplicate IDs/JSON keys and invalid metadata/hints are errors.
Absent IDs (and LangChain null IDs) become `c1`, `c2`, etc. IDs and metadata are inert data, never paths.

`check --format json` returns `issues`, per-chunk `alignments` and `stats`.
Spans use half-open **Unicode character offsets**, zero-based, and one-based
source lines. CRLF is preserved and counted as two characters. LlamaIndex hints
are verified against the source before use; stale hints fall back to search.
Merged chunks may include verified character hints in their output records.

Alignment first searches the entire text after removing Unicode whitespace
(including tabs/NBSP). It then tries individual nonblank lines for noncontiguous
fragments such as repeated table headers. Statuses are `aligned`, `fragmented`,
`partial`, `unaligned`. Every recovered span is matched text, not an unverified
metadata claim. Repeated occurrences prefer a verified hint, then a forward
match, then a backtracking match; `candidates: 2` means **two or more** possible
occurrences. JSON reports the ambiguity count and alignment failure rate.

Coverage measures the union of recovered source spans over **non-whitespace
source characters**. Whitespace changes are tolerated, not reported as lost
text. A 100% score is lexical coverage under the reported alignment heuristic;
repeated phrases can make provenance ambiguous. Review ambiguity counts for
repetitive documents. Without `--source`, source coverage is explicitly unchecked.

## Rules and repairs

| Rule | Default | Detection | Automatic repair |
| --- | --- | --- | --- |
| CL001 | error | Unclosed backtick/tilde fence, matching character and length, up to 3 spaces indent | Rejoin adjacent chunks until the fence closes |
| CL002 | warning | Pipe-table run without a header plus delimiter, including explicit blockquotes | Rejoin adjacent same-depth header/row fragments; repeat header when splitting oversized complete tables |
| CL003 | warning | Last nonblank line is an ATX heading | Merge forward if another chunk exists |
| CL004 | warning | Adjacent prose appears mid-sentence (English lowercase continuation / CJK, punctuation aware) | Report only |
| CL005 | warning | Character size outside configured bounds | Split oversized complete code blocks at line boundaries and tables at row boundaries |
| CL006 | warning | Exact duplicates, whitespace-normalized character 5-gram Jaccard near duplicates, adjacent suffix/prefix overlap | Drop exact duplicates only, subject to source coverage |
| CL007 | error | Missing source characters or partial/failed chunk alignment | Report only; never fabricate missing source text |
| CL008 | warning | Adjacent list items or an indented list continuation cross chunks | Report only |

Fenced code contents are not interpreted as tables, headings or lists. A shorter
backtick marker does not close a longer opener. Backticks in a backtick opener's
info string are invalid. Closing markers allow spaces/tabs, not other text.

With `--source`, repair also recovers a cut after the first or second character
of an **unquoted backtick/tilde opening marker of any length (at least three)**. Both adjacent
records must align uniquely, in order without overlap, and their exact raw
concatenation must match the source. The source must prove the cut is inside the
actual opener and a complete matching closer is present in these two records.
Info strings, opaque code payload, whitespace and line endings survive unchanged;
the left ID and ordered metadata provenance are retained in one `rejoin_fence`
edit. The synthetic [opener-cut fixture](../tests/lint/fixtures/fence_opener_cut/CONTRACT.md)
splits `# Example` and its Python block inside the opening marker:

```bash
ragwrench lint fix tests/lint/fixtures/fence_opener_cut/chunks.jsonl \
  --source tests/lint/fixtures/fence_opener_cut/source.md --min-chars 0 -o opener-fixed.jsonl
ragwrench lint check opener-fixed.jsonl \
  --source tests/lint/fixtures/fence_opener_cut/source.md --min-chars 0 --fail-on warning
```

The [four-character fixture](../tests/lint/fixtures/four_fence_opener_cut/CONTRACT.md)
splits the same preamble and Python block after two of four opening backticks.
It likewise becomes one exact-source record with left ID `a`, ordered
provenance, one edit, 100% coverage and zero second-fix edits. A three-character
marker inside that block stays literal code, including table- and
heading-looking lines on both sides of it. The
[five-character fixture](../tests/lint/fixtures/long_fence_opener_cut/CONTRACT.md) extends
this to longer markers: lengths 5/6/8/37 are tested for both marker types,
including shorter interior markers. Its initial `check` exits 0 because the
right-hand suffix independently looks like a complete shorter fence; `fix`
still reconstructs the source-backed opener. An installed wheel's Python can run
`tests/lint/installed_fence_opener_smoke.py` with a durable evidence directory to
verify check/fix/check/second-fix, LF/CRLF/CR, opaque payload, ordered prior
provenance and refused boundaries.

This path refuses absent source, missing bytes/closers, and ambiguous
matches even with verified hints, and list containers. Because lists are
outside the scanner, it conservatively refuses any preceding source list marker
outside code (including quoted lists for quoted recovery), and indented/nested
preambles. Unquoted recovery also refuses quoted preambles. Cuts in info
strings or payload, or after three or more marker characters, do not use this
path; existing complete-opener joins retain their behavior. Reconstruction is
bounded to two adjacent records; multi-record opener tokens remain unsupported. Disabling CL001 disables this repair too.

Single-level **quoted opener-marker cuts** also recover with source. The fixture
splits a `> # Example` preamble and quoted Python opener after its second
backtick. Its two records become one left-ID `a` chunk exactly equal to their
concatenation, with one `rejoin_fence` edit, ordered provenance, no findings,
100% coverage and zero second-fix edits. The same source/uniqueness,
actual-opener and complete-in-pair closer guards apply. Every raw line must use
one identical prefix: zero to three spaces followed by `> `; the opener may
have up to three further indentation spaces. Quoted preamble lines retain their
exact bytes. Additional `>` inside active code remains literal payload.

Both marker types and lengths 3/4/5/8, shorter interior markers, Unicode and
LF/CRLF/CR are covered by the [quoted opener fixture](../tests/lint/fixtures/quoted_fence_opener_cut/CONTRACT.md)
and tests. Run an installed wheel's Python with
`tests/lint/installed_quoted_fence_opener_smoke.py <evidence-directory>` to retain
check/fix/check/second-fix commands, raw outputs and exact-byte/provenance checks.
Nested containers, mixed/missing prefixes, tab/no-space prefixes, list preambles,
prefix-token cuts and multi-record opener reconstruction remain unsupported by
this fallback. Existing complete-opener joins have their separate contract.

CL001 and code-block repairs also handle explicit blockquote prefixes, including
nested quotes such as `> > `. Fence contents stay opaque at the opener's quote
depth: `> > > ``` ` is literal code inside a fence opened with `> > ```python`.
Rejoining requires a matching closer before the quote container ends. A missing
prefix (including a bare blank line) ends the container; repair will not join
across it or use a closer from another depth. An implicit container end within
one chunk is valid Markdown and does not require an explicit closer. Oversized
explicitly closed blocks split at code-line boundaries, repeating the original
raw quoted opener/closer and preserving every payload line and line ending.

The fixture in [tests/lint/fixtures/quoted_fence](../tests/lint/fixtures/quoted_fence) was
generated by **langchain-text-splitters 0.3.11 / langchain-core 0.3.79**, using
`RecursiveCharacterTextSplitter` with 65-character chunks, no overlap and
preserved line endings. Its three chunks split a nested quoted four-backtick
block containing Unicode, a shorter fence and table-like code. This command
rejoins them into one chunk whose text is exactly `source.md`, retains all three
metadata objects and reaches 100% source coverage:

```bash
ragwrench lint fix tests/lint/fixtures/quoted_fence/chunks.jsonl \
  --source tests/lint/fixtures/quoted_fence/source.md --min-chars 0 -o quoted-fixed.jsonl
ragwrench lint check quoted-fixed.jsonl \
  --source tests/lint/fixtures/quoted_fence/source.md --min-chars 0 --fail-on warning
```

An unfinished quoted fence with only a shallower/deeper closer remains unchanged
and reports CL001 (exit 1 at the default threshold). Regenerate this original
synthetic fixture with its `generate.py` in a separate environment containing
the two pinned LangChain packages; normal tests and CLI runs need neither package.

CL002 also recognizes simple GFM pipe tables inside explicit blockquotes. Header,
delimiter and body rows must stay at the same quote depth; raw prefix spellings
such as `> > ` and `>> ` can differ. Rejoining needs a matching header/delimiter
cell count and adjacent whole rows. In both unquoted and explicitly quoted
tables, each delimiter cell accepts **one or more hyphens**, optionally preceded
and/or followed by a colon: `-`, `:-`, `-:`, `:-:`, `--`, `:--`, `--:`, `:--:`
and longer forms. Padding around a cell is allowed; empty or colon-only cells,
internal spaces and other characters are rejected. Escaped pipes retain their
existing cell boundaries. A depth change, bare/quoted blank line,
intervening prose/fence or an already complete right-hand table prevents an
ordinary whole-row join.
A source-restored blank between unquoted or quoted fragments also prevents a
join: original records, metadata and hints stay unchanged, and a headerless row
keeps its CL002 finding. Unquoted source-backed cuts inside a body row can still
recover that exact row without inventing a newline. With `--source`, a cut
inside a delimiter row can also recover when the left fragment has one complete
header plus the partial delimiter, both fragments align in order, and their
combined source is one contiguous valid unquoted table. Both sides must touch
the same source delimiter row. A cut inside the header also recovers when the
left record contains that partial header, optionally after whole unquoted prose
lines and blanks, and both fragments touch the same unquoted source header row. This includes a right fragment that independently
looks like a complete table. Both cases require ordered, nonoverlapping aligned
source spans and exact combined normalized content; blanks, malformed cells,
mismatched header counts and quote-depth changes prevent recovery. A header-cut
preamble must retain its exact raw whitespace and contain no pipe-table runs,
quoted lines or fences. The recovered table must reach the end of the pair.
Delimiter cuts after a preamble, quoted prefix cuts and later quoted body-row
token cuts remain unsupported. Missing
headers remain CL002; the fixer never invents a header or quote prefix. CL005
splits oversized recognized tables at row boundaries, repeating the exact raw
header/delimiter and retaining every raw body row and line ending.

The [quoted-table fixture](../tests/lint/fixtures/quoted_table/CONTRACT.md) uses the same
two pinned LangChain packages, with 30-character chunks and no overlap. Its five
records separate a nested quoted header, delimiter and three Unicode/escaped-pipe
rows. This rejoins them into exactly `source.md`, preserves all five metadata
objects, and passes the warning threshold with 100% lexical source coverage:

```bash
ragwrench lint fix tests/lint/fixtures/quoted_table/chunks.jsonl \
  --source tests/lint/fixtures/quoted_table/source.md --min-chars 0 -o quoted-table-fixed.jsonl
ragwrench lint check quoted-table-fixed.jsonl \
  --source tests/lint/fixtures/quoted_table/source.md --min-chars 0 --fail-on warning
```

Regenerate with `tests/lint/fixtures/quoted_table/generate.py` in the pinned splitter
environment. Runtime and normal tests need no LangChain. An installed-wheel CLI
smoke is reproducible with `python tests/lint/installed_quoted_table_smoke.py` using
that installation's Python; it runs outside the repository and also checks
idempotence, saved-plan apply and a rejected depth-changing join.

For example, `> | k | v |\n> | - | :-: |\n> | a | b |\n` is a complete
table with no CL002. Splitting those three raw lines into adjacent records and
fixing with `--min-chars 0` rejoins the exact original text, preserves ordered
metadata and keeps 100% source coverage with `--source`. A second fix makes no
changes. `python tests/lint/installed_short_delimiter_smoke.py` checks this example,
unquoted and nested two-hyphen tables, raw CRLF/escaped pipes and idempotence
using an installed wheel outside the repository; its transcript includes every
CLI command, exit status and raw output.

For a rejected join, take chunks `| k | v |\n| - | :-: |\n` and
`| a | b |\n` with source `| k | v |\n| - | :-: |\n\n| a | b |\n`.
The source blank separates the table from the final row. With `--min-chars 0`,
fix preserves both records and 100% source coverage, logs a skipped table join
and makes no changes on a second fix. Check/fix/check with `--fail-on warning`
each exit 1 because CL002 remains on the second record at line 1. Reproduce
this boundary and successful adjacent-row/token-cut recovery using an installed
wheel's Python with `python tests/lint/installed_table_boundary_smoke.py`.

For delimiter recovery, chunks `| k | v |\n| -` and
`- | -- |\n| a | b |\n` with source `| k | v |\n| -- | -- |\n| a | b |\n`
rejoin to that exact source with one `rejoin_table` edit and ordered metadata
provenance. With `--min-chars 0 --fail-on warning`, check exits 1 before fixing,
fix/check exit 0 afterward, coverage stays 100%, and a second fix makes no
changes. Without source this cut remains unchanged and reports CL002. The same
installed smoke checks LF/CRLF/CR delimiter recovery alongside the rejected
blank boundaries above.

For header recovery, chunks `| ke` and
`y | v |\n| -- | -- |\n| a | b |\n` with source
`| key | v |\n| -- | -- |\n| a | b |\n` rejoin to that exact source. The left
ID and both metadata objects survive in ordered provenance, CL002 clears and
coverage stays 100%. With `--min-chars 0 --fail-on warning`, check exits 1 before
fixing and fix/check exit 0 afterward; a second fix makes no changes. Without
source, the records stay unchanged. The installed boundary smoke also verifies
header cuts with LF/CRLF/CR, Unicode, escaped pipes and optional outer pipes.

A header cut after a whole prose preamble also recovers: chunks
`Introduction.\n\n| ke` and `y | v |\n| -- | -- |\n| a | b |\n`
rejoin to `Introduction.\n\n| key | v |\n| -- | -- |\n| a | b |\n`.
The preamble remains byte-for-byte intact, the left ID and ordered metadata
provenance are retained, and one `rejoin_table` edit yields no CL002, 100% source
coverage and zero second-fix edits. Without source the records stay unchanged.
The installed boundary smoke covers this case and CRLF/CR prose preambles.

Single-level quoted header cuts also recover with source. Chunks `> | ke` and
`y | v |\n> | -- | -- |\n> | a | b |\n` become the exact contiguous quoted
source table, preserving all prefix/whitespace bytes, the left ID and existing
ordered provenance. One `rejoin_table` edit clears CL002, retains 100% coverage
and makes zero changes on a second fix. Each line must share the identical raw
prefix: zero to three ASCII spaces followed by `> ` (one ASCII space). The cut
must lie inside the header payload, with no in-record preamble. Both chunks
need unique, aligned, ordered nonoverlapping source spans and an exact raw
source slice outside fenced code. Nested/mixed depths, missing/changed prefixes,
list/fence boundaries, blank-separated tables, invalid delimiters and ambiguous
source occurrences (even with hints) stay unchanged. No-source behavior is
unchanged. See the [fixture contract](../tests/lint/fixtures/quoted_header_cut/CONTRACT.md).

```bash
ragwrench lint fix tests/lint/fixtures/quoted_header_cut/chunks.jsonl \
  --source tests/lint/fixtures/quoted_header_cut/source.md --min-chars 0 -o quoted-header-fixed.jsonl
ragwrench lint check quoted-header-fixed.jsonl \
  --source tests/lint/fixtures/quoted_header_cut/source.md --min-chars 0 --fail-on warning
```

An installed wheel's Python can run `tests/lint/installed_quoted_header_smoke.py`
outside the repository, checking LF/CRLF/CR, escaped pipes, Unicode, prior
provenance, refused boundaries, exact bytes, coverage and second-fix idempotence.

Source-backed cuts inside a consistently quoted **delimiter** row also recover:
`> | k | v |\n> | -` plus `- | -- |\n> | a | b |\n` become one
record containing the exact source table, with one `rejoin_table` edit. The left
record must contain one complete header plus a partial delimiter; both aligned
sides must touch that same delimiter row after its prefix. The identical raw
single-level prefix, exact-source, uniqueness and refusal requirements above
also apply here, including with verified hints. Short/colon delimiter cells,
Unicode, escaped pipes and LF/CRLF/CR retain every raw byte. Existing ordered
provenance stays in input order; coverage remains 100% and a second fix makes
zero edits. See the [delimiter fixture contract](../tests/lint/fixtures/quoted_delimiter_cut/CONTRACT.md).

```bash
ragwrench lint fix tests/lint/fixtures/quoted_delimiter_cut/chunks.jsonl \
  --source tests/lint/fixtures/quoted_delimiter_cut/source.md --min-chars 0 -o quoted-delimiter-fixed.jsonl
ragwrench lint check quoted-delimiter-fixed.jsonl \
  --source tests/lint/fixtures/quoted_delimiter_cut/source.md --min-chars 0 --fail-on warning
```

`tests/lint/installed_quoted_delimiter_smoke.py` checks the same recovery and refusal
contracts through an installed wheel's check/fix/check/second-fix commands.

Source-backed cuts inside the **first quoted body row** now recover too:
`> | k | v |\n> | -- | -- |\n> | al` plus `pha | b |\n` become one h
record containing the exact quoted source table. The left record must contain
one complete header and delimiter plus a partial first body row; both unique,
aligned, ordered nonoverlapping spans must touch that same body row after its
prefix. The identical raw single-level prefix and all exact-source/refusal
requirements above apply. Preserve every payload, prefix and whitespace byte,
the left ID and ordered prior provenance. One `rejoin_table` edit clears CL002,
coverage stays 100% and a second fix makes zero edits. Prefix-only cuts
remain unsupported. See the [body fixture contract](../tests/lint/fixtures/quoted_body_cut/CONTRACT.md).

```bash
ragwrench lint fix tests/lint/fixtures/quoted_body_cut/chunks.jsonl \
  --source tests/lint/fixtures/quoted_body_cut/source.md --min-chars 0 -o quoted-body-fixed.jsonl
ragwrench lint check quoted-body-fixed.jsonl \
  --source tests/lint/fixtures/quoted_body_cut/source.md --min-chars 0 --fail-on warning
```

`tests/lint/installed_quoted_body_smoke.py` verifies check/fix/check/second-fix with an
installed wheel, including Unicode, escaped pipes, short/colon delimiters,
LF/CRLF/CR, prior provenance, exact bytes, conservation and refused boundaries.

Source-backed cuts in **later quoted body rows** also recover after any number
of complete preceding rows, with no fixed row-count allowlist. For example,
`> | k | v |\n> | -- | -- |\n> | first | row |\n> | al` plus
`pha | b |\n` become one left-ID record equal to the complete quoted source
table. The same raw prefix, payload-row cut, exact concatenation, unique ordered
nonoverlapping alignment, provenance and conservation requirements apply.
Later-body recovery also requires the local table to be the complete matching
run in full source, outside fences; it refuses partial final rows, omitted
source rows and any preceding outside-code source list marker. This last guard
is conservative because list containers are not modeled. Blank/list/container
boundaries, nested/mixed prefixes, ambiguity even with hints and prefix-only
cuts remain refused. No quoted preambles are accepted. Body cell counts follow
the existing simple pipe-table dialect. See the
[later-body fixture contract](../tests/lint/fixtures/quoted_later_body_cut/CONTRACT.md).

```bash
ragwrench lint fix tests/lint/fixtures/quoted_later_body_cut/chunks.jsonl \
  --source tests/lint/fixtures/quoted_later_body_cut/source.md --min-chars 0 -o quoted-later-body-fixed.jsonl
ragwrench lint check quoted-later-body-fixed.jsonl \
  --source tests/lint/fixtures/quoted_later_body_cut/source.md --min-chars 0 --fail-on warning
```

`tests/lint/installed_quoted_later_body_smoke.py <evidence-directory>` validates exact
bytes, ordered prior provenance, coverage and idempotence through an installed
wheel, preserving inputs, outputs and raw CLI logs in that directory.

Fixes preserve input order, retain original payload slices, and add only
structural context/newlines. When fully verified adjacent source spans cover
the same normalized text, source whitespace recovers a boundary inside a code
token or table row. Metadata conflicts survive in ordered
`metadata.chunklint.inputs`; split chunks keep `parent_metadata`.
`metadata.chunklint` is the tool's provenance namespace. Generated IDs use
`~2`, `~3`, etc., skipping collisions.

Before writing, repair checks the multiset of input non-whitespace characters
against output (excluding logged exact duplicate removals), and, with a source,
checks that **every previously covered source position remains covered**. Unsafe
edits are skipped or the repair aborts with exit 2 and no repaired output.
A real repeated source passage is retained if removing it reduces coverage.
Repeated payload pieces created by structural splitting are also retained,
even if byte-identical. Near duplicates and overlaps are never deleted.
Re-running the same repair/configuration is tested for idempotence.

## CI and configuration

Exit status: **0** no finding at the failure threshold, **1** findings at/above
that threshold, **2** invalid input/configuration, resource limit, I/O failure,
or failed repair invariant. `fix` applies the threshold to its **after** report;
remaining defects are still listed. The default threshold is `error`.

```bash
ragwrench lint check chunks.jsonl --source guide.md --fail-on warning
ragwrench lint check chunks.jsonl --disable CL004,CL005 --format json
```

A `chunklint.toml` in the current directory is loaded automatically, or use
`--config path.toml`. CLI overrides take precedence. Unknown keys/rules fail.

```toml
[chunklint]
min_chars = 40
max_chars = 4000
near_duplicate = 0.90
max_overlap = 0.50
fail_on = "error" # warning, note, none also accepted
max_pair_checks = 100000
disable = []
```

Disabling a rule suppresses its findings and its associated edits (CL005 controls
structural re-splitting); source conservation remains enforced during repair.
SARIF includes rule definitions, levels, artifact URIs, chunk IDs, related chunks
and recovered source locations. The primary location is the recovered first
source line; chunk-relative rule line numbers stay in result properties because
normalization can change line counts. Reports do not embed input text/metadata;
fixed JSONL intentionally retains metadata, so treat that file as input-sensitive.

## Reproduce the experiment and tests

```bash
pip install -e '.[test]'   # from a clone of the repository
python -m pytest tests/lint -q
ragwrench lint check examples/lint/chunks_langchain.jsonl --source examples/lint/doc.md
# Expected exit 1: the fixture is deliberately defective.
ragwrench lint bench
ragwrench lint bench --write docs/lint/benchmark.json
```

The benchmark runs the public engine on committed synthetic LangChain Document
records with a shorter nested fence, broken table, orphan heading and duplicate.
It uses the same rules and configuration before and after; no upstream command,
LLM score, timer, network, or invented performance estimate is involved.
The actual run in [docs/lint/benchmark.json](lint/benchmark.json) records **6 → 0
defects**, **100% → 100% source coverage**, and identical results on a second fix.
This demonstrates structural repair, not a measured retrieval accuracy gain.
The corresponding TextNode JSON array is also committed and tested.

The test suite in `tests/lint/` exercises every rule and repair on original, synthetic and
redistributable fixtures.

## Limits and boundaries

- This is a structural linter, not a complete CommonMark/GFM renderer. It scans
  top-level fences, ATX headings, simple pipe tables and top-level lists, plus
  fences and simple pipe tables inside explicit (possibly nested) blockquotes. Quoted headings,
  lazy quote continuation, list containers (including fences nested in lists),
  Setext headings, HTML blocks, indented code,
  inline-code pipes and advanced table dialects are not modeled. Pipe-table,
  list and sentence rules are heuristics; disable them for other dialects.
  Quoted fence rejoining requires full explicit quote prefixes on continuation
  lines; cuts inside a quote prefix or payload token may remain unrepaired.
  Quoted table rejoining generally requires whole rows with full explicit prefixes;
  source-backed single-level header, delimiter and body-row cuts have the
  stricter contracts above. Prefix-only token cuts remain unsupported. Short delimiter cells
  follow GFM's one-or-more-hyphen rule; the scanner remains a subset of GFM,
  not a complete GFM table parser. Unquoted source-backed delimiter-cut recovery
  requires a complete header in the left fragment and one table across the two
  records. Source-backed header cuts also allow a whole raw prose preamble in
  the left record; delimiter cuts with a preamble and
  preambles containing tables, quotes or fences remain unsupported.
- Missing table headers, an EOF orphan heading, unclosed EOF fence, oversized
  prose, and a single huge code line/table row remain visible findings. The
  fixer never splits a line/row arbitrarily or guesses missing schema/content.
  Without a matching later fence closer, it leaves subsequent chunks separate.
  Reopening fences/repeated headers/context can make an indivisible part exceed
  `max_chars`; CL005 remains. Fenced blocks/tables must be complete to re-split.
- Whitespace-insensitive coverage cannot detect missing formatting whitespace or
  establish semantic provenance for repeated text. Fragment matching can recover
  noncontiguous lines; review its spans and ambiguity instead of treating it as
  exact original layout. No semantic/LLM, PDF or HTML layout checks are provided.
- Inputs/config/source are regular local files, maximum 32 MiB each; maximum
  10,000 records. CL006 has an explicit 100,000-pair budget (447 chunks at the
  default) and 200,000-character per-chunk / 1,000,000-character total limits. Exceeding a limit is exit 2;
  raise `max_pair_checks` or disable CL006 explicitly. Pair analysis is quadratic;
  source search and conservative fix validation can also cost multiple passes.
- Only explicit output filenames are written, never paths from records. Output
  parents must already exist; symlink outputs and overwriting input/source/config
  are refused. Files are replaced atomically individually; the change log and
  JSONL are two separate writes, not a multi-file transaction. I/O failure may
  leave a change log without a corresponding repaired output. No external
  commands, template evaluation, deserialization code, or network calls run.

Original code: MIT. Upstream projects are format references, not dependencies or
endorsers.
