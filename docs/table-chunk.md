# ragwrench table-chunk

Generic text splitters cut tables mid-row and drop their headers, so a retrieved chunk
contains numbers with no column names or row labels. `ragwrench table-chunk` splits
Markdown and HTML documents so that every table chunk carries its own headers, caption
and heading path. Use it when your corpus contains tables (reports, price lists, exports
from Docling, Unstructured or MarkItDown) and chunks should answer cell lookups on their own.

_Formerly the standalone `table-chunker` tool (`tablechunk`)._

`ragwrench table-chunk` turns Markdown and HTML table exports into self-contained RAG chunks.
Every table chunk repeats its heading breadcrumb, caption (or preceding paragraph's
last sentence), and headers. Normal rows stay intact; row ranges and source line
spans trace retrieved results back to the document.

Standard-library runtime, MIT. No models, credentials, framework
imports, services or background processes.

## Quick start

```sh
pip install ragwrench
ragwrench table-chunk split examples/table-chunk/report.md --max-tokens 120 -o /tmp/tc.jsonl
ragwrench table-chunk split tests/tablechunk/fixtures/export.html --mode kv --adapter langchain
ragwrench table-chunk split - --format markdown --mode html < examples/table-chunk/report.md
```

Install `'ragwrench[tiktoken]'` for the optional tiktoken counter. Splitting, benchmarks
and tests run offline by default.

Input is UTF-8, or stdin `-`. `.html`/`.htm` selects HTML; other filenames and
stdin default to Markdown. Override with `--format html`. Output is JSONL, one
chunk per line, to stdout by default. Errors exit 2; warnings go to stderr and
affected chunks' metadata. Output files are replaced atomically after successful
parsing and splitting.

Example input:

```markdown
## Pricing

Plans as of 2026.

| Plan | Seats | Price |
|---|---|---|
| Free | 3 | $0 |
| Team | 25 | $99 |
```

The table output (metadata abbreviated):

```json
{"id":"t0-r0-2","text":"Pricing\nPlans as of 2026.\n| Plan | Seats | Price |\n| --- | --- | --- |\n| Free | 3 | $0 |\n| Team | 25 | $99 |","metadata":{"table_index":0,"row_start":0,"row_end":2,"columns":["Plan","Seats","Price"],"header_path":["Pricing"]}}
```

Longer tables become consecutive row groups. The lead-in also stays in its prose
chunk, so prose is independently retrievable. Adjacent prose paragraphs pack
together with breadcrumbs. Row indices are **zero-based, end-exclusive**; source
lines are **one-based, inclusive**. IDs are deterministic within a document;
namespace them by source when combining documents.

## Python API and integrations

```python
from ragwrench.tablechunk import split_document, count_tokens

with open("examples/table-chunk/report.md", encoding="utf-8") as stream:
    chunks = split_document(stream.read(), max_tokens=120, mode="kv", source="report.md")
for chunk in chunks:
    assert count_tokens(chunk.text) <= 120
    print(chunk.as_dict())              # id, text, metadata
    print(chunk.as_dict("langchain"))   # id, page_content, metadata
    print(chunk.as_dict("llamaindex"))  # id_, text, metadata
```

These are constructor-shaped dictionaries, not framework persistence formats.
Use `Document(**chunk.as_dict("langchain"))` or
`TextNode(**chunk.as_dict("llamaindex"))` where those frameworks are installed.
Tests verify field mappings without installing the frameworks; specific upstream
framework versions are not integration-tested.

For Unstructured, extract `element["metadata"]["text_as_html"]` and pass that
string with `input_format="html"`. For Docling/MarkItDown, pass the exported
Markdown string. No converter is invoked; this is an independent export consumer.

Table modes:

- `markdown`: GFM header; hierarchical HTML header labels joined by ` / `.
- `kv`: `Columns: ...` then `Header: value` lines per row.
- `html`: expanded `<thead>` rows and `<tbody>`, escaped cell/context text.

The mode controls tables. Prose retains original text plus a breadcrumb; treat
source Markdown/prose as untrusted input when rendering it.

Metadata includes `kind`, `source`, `token_count`, `warnings`, `table_index`,
`row_start`, `row_end`, `columns`, `headers`, `header_path`, `caption`, `lead_in`,
`mode`, `line_start`, `line_end`, `table_line_start`, `table_line_end`, and
`row_line_spans`. HTML rowspan provenance includes the originating source row
of carried cells. Prose fragments retain the full paragraph line span, rather
than claiming character-exact line provenance.

### HTML normalization accounting

HTML input (including HTML tables embedded in Markdown) adds
`metadata.normalization`, schema version 1, in every table output mode:

- `header_cell_ids` repeats the expanded header grid in every chunk.
  `row_cell_ids` covers the chunk's `row_start:row_end` range. Repeated IDs
  identify copies introduced by rowspan/colspan; `null` identifies a slot with
  **no source cell**, unlike an explicit empty `<td></td>`, which has an ID.
- `source_cells` contains the source records referenced by those grids and by
  explicit header associations, including cells outside the chunk's rows.
  IDs such as `r2c0` identify the
  zero-based outer source row and physical cell within that row, counting header
  rows. IDs are scoped to a source/table, and exclude nested table rows/cells.
  Each record has `row_index`, `cell_index`, `tag`, `rowspan`, `resolved_rowspan`, `colspan`,
  `line_start`, `column_start`, `line_end`, and `column_end`.
  `rowspan` retains the declared integer (including zero; omitted means 1).
  `resolved_rowspan` records the actual number of source rows covered.
- Cell positions identify the **starts of the opening and closing tags**. Lines
  are one-based document lines; columns are zero-based Unicode character offsets
  in those lines (tabs count as one character), not byte or visual offsets.
- `diagnostics` explains missing grid slots (`missing_cell`), generated labels
  (`synthetic_headers`), flattened nested tables (`nested_table_flattened`), and
  ambiguous equal column paths (`duplicate_column_path`). Diagnostics refer to
  source cell IDs, normalized grid coordinates, or column indices. Grid diagnostic
  row indices are table-wide within `headers` or `rows`, not chunk-local.
  Cell/row diagnostics are filtered to the chunk's references; table-wide warning
  codes remain in `metadata.warnings` and are printed by the CLI.

Distinct header cells retain separate hierarchy levels even when their text is
equal. A carried copy of one rowspan header contributes only one level. For example:

```html
<table>
<tr><th rowspan="2">Region</th><th>Revenue</th></tr>
<tr><th>Revenue</th></tr>
<tr><td rowspan="2">West</td><td></td></tr>
<tr></tr>
</table>
```

Expected columns are `["Region", "Revenue / Revenue"]`; normalized data rows are
`[["West", ""], ["West", ""]]`. Their source IDs are
`[["r2c0", "r2c1"], ["r2c0", null]]`: the first empty value was explicit, while
the second gets a `missing_cell` diagnostic at data row 1, column 1. The `West`
source record always points to source row 2, cell 0, line 4, column 4, even in a
chunk containing only the second data row. See
`tests/tablechunk/fixtures/spans-accounting.html` for a larger runnable fixture:

```sh
ragwrench table-chunk split tests/tablechunk/fixtures/spans-accounting.html --mode kv --max-tokens 40
```

Oversized-cell fragments retain the complete source-row ID grid; only the cell
selected by `fragment.cell_index` contributes text. Fragment character offsets
refer to normalized cell text. Accounting records preserve normalized text only
for explicitly referenced headers (below), not raw HTML: entities are decoded, whitespace normalized, inline markup
discarded, and nested structure flattened with a diagnostic. Every accepted
outer source cell is mapped, including explicit empty cells. Existing strict
failures remain: overlapping/out-of-table spans raise `TableChunkError` with the
offending cell's source position and ID; CLI errors exit 2 without replacing an
existing output file.

### Explicit header associations

An outer `<td headers="...">` adds an ordered `headers` list to that cell's
`source_cells` record. Only explicitly named outer `<th id="...">` cells in the
same table resolve; leading column labels, `scope`, layout and browser DOM repairs
do not supply associations. For example:

```html
<table><tr><th id="地域"> Région </th><th id="tax">Tax &amp; fees</th></tr>
<tr><td headers="tax 地域" colspan="2">12</td></tr></table>
```

The `r1c0` source record includes:

```json
{"headers":[{"id":"tax","cell_id":"r0c1","text":"Tax & fees"},{"id":"地域","cell_id":"r0c0","text":"Région"}]}
```

Reference order follows the declaration, including multiple headers and Unicode
IDs/text. IDs match exactly and case-sensitively, without Unicode normalization;
only ASCII spaces, tabs, line feeds, form feeds and carriage returns separate
references. Text uses the existing entity/whitespace/inline normalization.
Body and forward header references are supported. Referenced header source
records remain in each affected chunk even when outside its grid; carried and
fragmented data cells retain the same complete association list and source IDs.
Text, columns, grids, chunk IDs and token budgets are unchanged.

Absent `headers` attributes add no field; empty/valueless attributes add `[]`.
Valid header IDs alone add no output fields. Outer header IDs must be nonempty
and contain no ASCII whitespace, and cannot repeat within a table. Repeated
references, missing/non-header targets, repeated `id` attributes on outer `th`,
and repeated `headers` attributes on outer `td` are rejected at the offending
source cell's opening position. For example,
`<table><tr><td headers="missing">x</td></tr></table>` fails with
`line 1, column 11: source cell r0c0: headers target 'missing' is not an outer <th>`.
Buffered/streamed HTML and embedded HTML in buffered Markdown share this contract.
CLI errors exit 2 with empty stdout and preserve existing output, even for a late
invalid table. IDs reset between tables. Flattened nested tables contribute no
resolvable IDs; their attributes remain uninterpreted. Associations declared on
`th` itself, non-cell IDs, `scope` inference and transitive header chains are
outside this explicit data-cell contract. Streaming association records have the
existing 1 MiB working bound; metadata remains outside text-token budgets.

Runnable multi-header, Unicode and span fixture:

```sh
ragwrench table-chunk split tests/tablechunk/fixtures/explicit-headers.html --mode kv --max-tokens 100
```

### Row-group spans

`rowspan="0"` repeats a cell through the last row of its owning `thead`,
`tbody` or `tfoot` element. Adjacent elements with the same tag are separate
groups. Direct `<tr>` children of a table form an implicit group, ending at the
next explicit section; rows after that section start a new implicit group.
Sections must be direct children of the outer table. Nested tables still flatten
into their outer cell and do not change the outer row groups.

`thead`/`tbody`/`tfoot` openings also require a table when none is open,
including standalone, wrapped, empty and self-closing groups. For example,
`<tbody>body</tbody>` fails with
`line 1, column 0: row group must be a direct child of table`;
`<table><tr><td>earlier</td></tr></table>\n<tbody>late</tbody>` fails with
`line 2, column 0: row group must be a direct child of table` (the `\n` denotes
a physical newline). Buffered and streamed HTML report the group's opening
position before processing its content or descendants. CLI failures exit 2 with
`ragwrench table-chunk: ` followed by that exact diagnostic on stderr, empty stdout,
preserved existing output bytes and no partial replacement files, even after a
valid table. In buffered Markdown this applies only inside parsed HTML table
fragments, including the closing physical line; tags outside those blocks remain
literal prose. Escaped tags, comments, script/style content and flattened nested
tables retain their existing behavior.

Outer `<tr>` tags must be direct children of `table`, `thead`, `tbody` or `tfoot`;
outer `<td>`/`<th>` tags must be direct children of `tr`. Wrappers around those
structural tags are rejected; inline/block content **inside** a cell is still
accepted. For example,
`<table><tbody><div><tr><td>x</td></tr></div></tbody></table>` fails with
`line 1, column 19: <tr> must be a direct child of table/thead/tbody/tfoot`.
`<table><tr><div><td>x</td></div></tr></table>` fails with
`line 1, column 16: <td> must be a direct child of tr` (likewise for `th`).
This strict contract applies to buffered and streamed HTML, and HTML tables in
buffered Markdown. CLI failures exit 2, emit no stdout and preserve existing
output files. Nested-table rows/cells remain flattened into their outer cell;
their ownership is not interpreted as outer structure.

Ownership also applies when no table is open: standalone or wrapped `tr`, `td`
and `th` openings reject before their content can become prose or their span
attributes are interpreted. For example, `<tr><td rowspan="bogus">x</td></tr>`
fails with `line 1, column 0: <tr> must be a direct child of table/thead/tbody/tfoot`;
`<td colspan="0">x</td>` fails with
`line 1, column 0: <td> must be a direct child of tr` (likewise for `th`).
Buffered and streamed HTML report the opening tag's document line and Unicode
column. CLI errors exit 2 with `ragwrench table-chunk: ` followed by the diagnostic on
stderr, empty stdout, preserved output bytes and no partial replacement files,
including orphan cells after a valid table. In buffered Markdown this check
applies only to fragments sent to the HTML table parser, including tags on the
table's closing line. Ordinary Markdown inline tags and tags on later lines
outside that block retain their literal prose behavior. Escaped tags, comments,
script/style contents and flattened nested-table structure keep their existing
handling.

Outer `<caption>` tags must also be direct children of `table`. A caption
wrapped in `div`, inside a row/cell, or inside another caption is rejected.
This also rejects openings when no table is open, including standalone,
wrapped, empty, self-closing and uppercase captions, before their contents can
become prose or disappear. `<caption>label</caption>` and `<caption/>` fail with
`line 1, column 0: <caption> must be a direct child of table`;
`<div><caption>label</caption></div>` fails at column 5. A caption after a closed
table, such as `<table><tr><td>earlier</td></tr></table>\n<caption>late</caption>`,
fails with `line 2, column 0: <caption> must be a direct child of table`
(the `\n` denotes a physical newline). Buffered and streamed HTML report the
opening tag's document line and Unicode column. CLI errors exit 2 with
`ragwrench table-chunk: ` followed by the exact diagnostic on stderr, empty stdout,
preserved existing output bytes and no partial replacement files. Buffered
Markdown checks parsed HTML table fragments, including the closing physical
line; tags outside those blocks remain literal prose. Escaped tags, comments,
script/style contents and flattened nested tables keep their existing handling.
An orphan closing `</caption>` also fails when no table is open, at the start
of the closing tag: `</caption>` reports
`line 1, column 0: unexpected closing caption`, and
`<div></CAPTION></div>` reports column 5. After a closed table,
`<table><tr><td>earlier</td></tr></table>\n</caption>` reports
`line 2, column 0: unexpected closing caption` (a physical newline).
Buffered and streamed HTML share this diagnostic and the CLI atomicity contract
above. Buffered Markdown checks complete closing tags within consumed HTML
table fragments, including the closing physical line; outside those fragments
the tags remain literal prose. Normal caption closures, nested-table flattening
and mismatched-tag diagnostics inside tables retain their existing behavior.
Orphan outer row-group closing tags also fail when no table is open:
`</thead>` reports `line 1, column 0: unexpected closing row group`,
`<div></TBODY></div>` reports `line 1, column 5: unexpected closing row group`,
and `<p>北</p>\n é\t</tfoot>` reports
`line 2, column 3: unexpected closing row group` (a physical newline and tab).
After a valid table, `<table><tr><td>earlier</td></tr></table>\n</tbody>`
reports `line 2, column 0: unexpected closing row group`.
Positions identify closing-tag starts in Unicode characters, including uppercase
and multiline tags. Buffered/streamed HTML, complete tags in consumed buffered
Markdown table fragments (including the closing physical line), and CLI
atomicity follow the same contracts above. Tags outside those Markdown fragments
remain literal prose; streamed embedded HTML remains unsupported. Valid group
closures, in-table mismatched-tag errors, ignored script/style/comment content,
escaped tags and nested-table flattening retain their existing behavior.

Orphan outer row/cell closing tags also fail when no table is open:
`</tr>` reports `line 1, column 0: unexpected closing row`,
`<div></TD></div>` reports `line 1, column 5: unexpected closing cell`, and
`<p>北</p>\n é\t</th>` reports `line 2, column 3: unexpected closing cell`
(physical newline and tab). After a valid table,
`<table><tr><td>earlier</td></tr></table>\n</tr>` reports
`line 2, column 0: unexpected closing row`. Positions identify closing-tag
starts in Unicode characters. These errors share the buffered/streamed HTML,
buffered-Markdown fragment and CLI atomicity contracts above, including when
an earlier table was valid. Valid closures and in-table mismatched-tag errors
retain their existing behavior.
For example,
`<table><div><caption>label</caption></div><tr><td>x</td></tr></table>` fails with
`line 1, column 12: <caption> must be a direct child of table`, with the same
buffered/streamed error and CLI atomicity contract above. Direct captions keep
inline text, entities and line breaks: `<caption><em>Label</em> &amp; details</caption>`
becomes `Label & details`. Captions inside nested tables remain flattened into
their outer cell, including nested markup whose ownership is not validated.

Each outer table accepts at most one direct caption, even if that caption is
empty or self-closing. A second caption is rejected at its opening tag:
`<table><caption>First</caption><caption>Second</caption><tr><td>x</td></tr></table>`
fails with `line 1, column 31: duplicate outer <caption>`, rather than silently
joining the labels. The CLI exits 2, emits no stdout and preserves existing
output files. Caption presence resets for each outer table. Nested-table
captions still flatten into their outer cell and do not count as outer captions.
An outer caption must precede all outer rows and row groups, including empty or
self-closing `thead`/`tbody`/`tfoot` groups. For example,
`<table><tr><td>x</td></tr><caption>Late</caption></table>` fails at the caption
opening with `line 1, column 26: <caption> must precede rows and row groups`.
Buffered HTML, streamed HTML and embedded HTML in buffered Markdown share this
rule and the CLI atomicity contract above. The ordering state resets for each
outer table; nested-table captions still flatten into their outer source cell,
including captions after nested rows. Existing parent/duplicate-caption errors
take precedence when those rules are also violated. Other direct children, such
as `colgroup`, do not constrain caption placement under this subset.

For example:

```html
<table>
<thead><tr><th>Region</th><th>Sales</th></tr></thead>
<tbody>
<tr><td rowspan="0">West</td><td>10</td></tr>
<tr><td>20</td></tr>
</tbody>
<tbody><tr><td>East</td><td>30</td></tr></tbody>
</table>
```

This produces columns `["Region", "Sales"]` and rows
`[["West", "10"], ["West", "20"], ["East", "30"]]`. The `West` source cell
`r1c0` retains `rowspan: 0` and `resolved_rowspan: 2`, including in chunks and
cell fragments containing its carried copy. Ordinary positive spans have equal
declared and resolved extents. A positive `rowspan="3"` on that cell fails with
`line 4, column 4: source cell r1c0: rowspan crosses row group boundary`;
the CLI exits 2, emits no stdout, and preserves an existing output file.

Declared positive rowspans and colspans remain limited to 256. A zero rowspan
may resolve to more than 256 rows, subject to the 100,000-cell expanded grid
limit. Each outer `td`/`th` accepts at most one `rowspan` and one `colspan`
attribute, with case-insensitive names. Repeated declarations are rejected before
span values are validated, even when equal or when an earlier invalid value would
otherwise be overwritten. For example,
`<table><tr><td rowspan="bogus" rowspan="1">x</td></tr></table>` fails with
`line 1, column 11: duplicate rowspan attribute`;
`<table><tr><td colspan="0" colspan="1">x</td></tr></table>` fails with
`line 1, column 11: duplicate colspan attribute`. Errors point to the outer cell's
opening tag in buffered/streamed HTML and embedded HTML in buffered Markdown.
The CLI exits 2 with `ragwrench table-chunk: ` followed by that diagnostic on stderr, empty
stdout, preserved existing output bytes and no partial replacement files, even
for a late invalid table. Existing structural ownership errors take precedence.
Nested-table attributes remain flattened into their outer cell without span
validation; unrelated attributes retain their existing behavior.

Zero colspan, overlaps and spans beyond the table remain errors. Full
header/body/footer and adjacent-body examples are runnable with:

```sh
ragwrench table-chunk split tests/tablechunk/fixtures/row-groups.html --mode kv --max-tokens 40
```

## Streaming large Markdown and HTML tables

```sh
ragwrench table-chunk split large.md --stream --max-tokens 256 -o chunks.jsonl
ragwrench table-chunk split - --stream --mode kv < large.md > chunks.jsonl
ragwrench table-chunk split tests/tablechunk/fixtures/spans-accounting.html --stream --mode kv --max-tokens 26
```

`--stream` accepts UTF-8 **Markdown or HTML input**, with all three table output
modes and all adapters. `.html`/`.htm` selects HTML; use `--format html` for stdin
or `input_format="html"` in `stream_document`. It removes the buffered mode's 10 MiB input and 10,000-chunk
limits. JSONL records, IDs, greedy row grouping, fragment offsets, warnings and
complete provenance match buffered mode on inputs accepted by both. For example:

```markdown
| A |
|---|
| x |
| y |
```

At `--max-tokens 12`, both modes emit `t0-r0-1` containing the header and `x`,
then `t0-r1-2` containing the header and `y`. Their row ranges are `[0,1)` and
`[1,2)`, source row lines are 3 and 4, and `table_line_end` is 4 in both records.
Late wider rows still cause earlier rows to be padded, with the same generated
column labels and table-wide warnings as buffered mode.

```python
from ragwrench.tablechunk import stream_document

with open("large.md", "rb") as stream:
    for chunk in stream_document(stream, max_tokens=256, source="large.md"):
        print(chunk.as_dict())
```

Input reads are at most 64 KiB. Each Markdown source line, paragraph and fenced code block
is limited to 1 MiB UTF-8; rendered working candidates are limited to 1,048,576
Unicode characters, independently of the token counter. Exceeding a limit raises
`TableChunkError` (CLI exit 2); shorten the block or use buffered mode where its
limits permit. These fixed bounds keep working memory independent of total rows,
paragraphs and chunks. They are buffer limits, not a promised 1 MiB process RSS.

Markdown tables are scanned into temporary disk files before emitting their first chunk:
this discovers final widths, ragged warnings and table end lines without retaining
rows in RAM. Pending prose also uses disk. CLI output is serialized incrementally
into a temporary file, then atomically replaces the destination after success.
Stdout is copied only after successful validation, so late parse/budget/UTF-8
failures leave stdout empty and existing output files intact. Disk space scales
with the current table/pending prose plus total JSONL output. Temporary files use
the system temp directory (`TMPDIR`); an output replacement file uses the output's
directory. Storage errors exit 2 and remove the partial replacement file.

The Python iterator yields before the complete document is validated, so a later
error can follow earlier chunks. Consume it fully or call `.close()` when stopping
early to release its temporary files; it never closes the caller's input stream.
Avoid collecting chunks into a list if bounded memory is required. Custom counters
must also avoid retaining every string. Embedded HTML table blocks in Markdown
streaming remain unsupported; fenced HTML remains ordinary code. Select HTML input
for full HTML documents.

HTML streaming uses the same strict parser and span normalizer as buffered HTML.
Outer tables must close explicitly before any of their chunks can be emitted.
Source rows, expanded grids, row-group ends, cell records and diagnostics use a
temporary SQLite database; completed tables release their records for reuse.
Heading/prose context, captions, nested-table flattening, zero/positive rowspans,
missing slots, source coordinates and cell fragments retain full buffered JSONL
parity, demonstrated by `tests/tablechunk/fixtures/spans-accounting.html` in every mode and
adapter. For example, `<table><tr><th>A</th></tr><tr><td>x</td></tr></table>`
at `--max-tokens 12` emits `t0-r0-1` with text `| A |\n| --- |\n| x |`.

HTML retains the **100,000-cell expanded grid** cap per table and existing span
limits. It removes the total input/chunk caps, allowing multiple tables or larger
cell text within the working limits. HTML prose blocks, captions, row working
blocks, JSON-encoded disk records, aggregate headers and unfinished parser tokens
are limited to 1 MiB UTF-8; table nesting is limited to 1,024 open tags. Rendered
candidates retain the 1,048,576-character limit. These are storage/working bounds,
not process RSS guarantees. SQLite cache and parser read buffers add overhead;
normalization metadata is outside the rendered-text token budget. Temporary disk
use can include several representations of the table. No HTML timing or memory
benchmark is claimed.

An unclosed outer table raises `TableChunkError("unclosed HTML table")`; invalid
spans retain source-cell errors. CLI errors exit 2, leave stdout empty, preserve an
existing output file and remove temporary files, including on late UTF-8 or
storage errors. The Python iterator may already have yielded earlier tables.

### Measured content parity and memory

```sh
python benchmarks/table-chunk/streaming.py -o /tmp/streaming-results.json
```

The stored Linux/Python 3.12 run in `benchmarks/table-chunk/streaming-results.json` launches a
fresh CLI process per mode, using the **same input file**, regex counter,
256-token budget and adapter for each pair. SHA256 hashes match for every JSONL
byte (including all metadata): 296 chunks at 5,000 rows and 2,943 at 50,000 rows.
The 50,000-row input is 3,073,853 bytes. Buffered CLI peak RSS was 72,100 KiB;
streaming was 20,708 KiB. At 5,000 rows they were 25,972 and 20,728 KiB. This
exposes the buffered source/rows/chunks/serialized-output accumulation and the
streaming mode's disk-backed alternative. The larger run took 1.633 seconds
buffered and 1.772 seconds streaming: this is a memory tradeoff, with no speedup
claim. RSS includes the interpreter/imports; timings and RSS are single local
measurements and depend on platform, allocator, data width and budget. Temporary
source paths appear in provenance, so output hashes are compared within a run.

## Budgets and explicit fallbacks

The default `regex-v1` counter counts CJK ideographs individually, runs of other
letters/digits as one token, and non-word punctuation individually. Whitespace
and underscores do not count. This is a deterministic approximation, **not a
model tokenizer**: a long Latin identifier is one token. Repeated context and
serialisation punctuation count toward the budget.

Pass a nonnegative prefix-monotonic `counter: Callable[[str], int]` to the API
(e.g. `counter=len` for character budgets). Optional model tokenizer:

```sh
pip install 'ragwrench[tiktoken]'
ragwrench table-chunk split examples/table-chunk/report.md --counter tiktoken --encoding cl100k_base
```

Tiktoken can download its vocabulary on first use. Defaults and tests do not use
it. Budgets apply to rendered `text`, not JSON syntax or metadata. The document
is validated before output is written.

- Fitting rows are never split. By default rows are not duplicated and groups
  greedily fill budgets; optional whole-row overlap is described below.
- Oversized rows fall back to **one nonempty cell at a time**, using fitting
  character prefixes with whitespace boundaries preferred. Headers/context stay.
  Warning: `oversized_row_cell_split`; metadata `fragment` contains `cell_index`,
  per-row `index`, and end-exclusive `char_start`/`char_end` in parsed cell text.
  Row ranges repeat for fragments; reconstruct using fragment metadata. Empty
  cells need no separate fragment. Other cells' values (including row identifiers)
  are not repeated in each fragment; join by source/table/row for full-row context.
- Context/headers that exceed the budget, or a content character that cannot fit,
  cause an error. Increase the budget or shorten the context.
- Ragged rows pad to the widest row with `ragged_rows_padded`; extra columns get
  stable `column_N` labels. Headerless HTML uses `synthetic_headers`.
- Nested HTML tables flatten into their parent cell with `nested_table_flattened`;
  inner row provenance is not retained independently.
- Fenced code is never parsed as a table. Oversized code follows prose splitting
  with `code_block_split`; resulting fragments need not be valid code fences.

## Whole-row overlap

```sh
ragwrench table-chunk split report.md --max-tokens 120 --overlap-rows 1
ragwrench table-chunk split report.html --stream --max-tokens 120 --overlap-rows 1
```

`--overlap-rows` and the `overlap_rows` keyword on `split_document` and
`stream_document` accept integers in `0..100` (default `0`; API booleans are
rejected). Zero preserves the existing JSONL bytes, including metadata. All input
formats, output modes and adapters use the same policy. For example, this table
has five rows, each containing 30 words:

```python
from ragwrench.tablechunk import split_document

report = "| A |\n|---|\n" + "".join(
    "| " + " ".join([f"r{i}"] + ["word"] * 29) + " |\n" for i in range(5)
)
chunks = split_document(report, max_tokens=120, overlap_rows=1)
assert [(c.metadata["row_start"], c.metadata["row_end"]) for c in chunks] == [(0, 3), (2, 5)]
```

Without overlap the ranges are `[0,3)` and `[3,5)`. With overlap the IDs are
`t0-r0-3` and `t0-r2-5`. Opted-in **table** metadata records `overlap_rows`
(the actual repeated row count: `0`, then `1`) and `requested_overlap_rows`
(the policy: `1`). Prose metadata is unchanged. Row indices and line spans refer
to the original table; HTML normalization keeps the original source-cell IDs,
span extents and explicit header associations for repeated rows.

Each new chunk repeats at most the requested suffix of complete logical rows
from the preceding chunk of that table, in source order. Headers, heading and
caption/lead-in context, and repeated rows all count toward the same text budget.
Oldest overlap rows are dropped until at least one new whole row fits. One-row
chunks therefore have zero overlap; the last chunk never contains overlap alone.
Overlap resets at each table and cannot cross prose. Oversized cell-fragment
rows are barriers: their text, IDs, offsets and provenance retain the existing
behavior, with actual overlap `0`, and the next whole row starts a fresh group.
Header-only tables also record actual overlap `0`; context-too-large errors and
CLI staging/atomic failure behavior remain unchanged. The streaming iterator
still may raise after earlier yields, and existing working-buffer limits apply.

## Reproduce the retrieval experiment

```sh
ragwrench table-chunk bench
ragwrench table-chunk bench --corpus benchmarks/table-chunk/fixture/corpus.md \
  --queries benchmarks/table-chunk/fixture/queries.json -o /tmp/tc-bench.json
# New fixture directory must not exist:
ragwrench table-chunk bench --rows 20 --tables 2 --fixture-dir /tmp/tc-fixture
```

Default: four 60-row synthetic subscription tables, 720 cell-lookup queries.
Query text has entity and column, **never the answer**. The baseline slices the
same source into non-overlapping 120-token windows. Each method builds its own
pure-Python BM25 index (`k1=1.5`, `b=0.75`, positive Robertson IDF, stable document
order ties) and ranks actual queries. All use the same counter and text budget.

The stored actual run in `benchmarks/table-chunk/results.json` produced:

| Method | Chunks | Evidence hit@3 | MRR | Total chunk tokens |
| --- | ---: | ---: | ---: | ---: |
| Fixed-token baseline | 20 | 0.15417 | 0.15417 | 2304 |
| Table Markdown | 32 | 1.00000 | 1.00000 | 3136 |
| Table KV | 40 | 1.00000 | 1.00000 | 3996 |

Evidence relevance requires entity, all column-label terms and answer tokens
in one chunk: **self-contained cell retrieval**, not answer generation or semantic
proof. `entity_hit@k` separately measures entity retrieval; `answerable_fraction`
shows queries with any evidence chunk. MRR uses the full ranking. Repeated
headers increase index size; total tokens expose that cost. This retrieval experiment makes no timing/memory
performance claim.

This table-heavy synthetic corpus isolates header loss. Perfect table-aware
scores do not predict production accuracy. Token co-occurrence can misattribute
answers in ambiguous real tables. Replay accepts your Markdown and a JSON query
array (`query`, `entity`, `column`, `answer` strings). Corpus/query hashes and
configuration accompany results; no clock, random inputs, downloads or models.

## Scope and boundaries

- GFM requires a delimiter row (at least three dashes per column). Optional outer
  pipes, alignment, escaped pipes, matched backticks, empty/ragged cells and fences
  are supported. No delimiter means prose. This is not full CommonMark: setext
  headings, indented-code recognition, multiline physical rows, blockquote tables
  and reference/link-aware pipe escaping are outside scope. Use a single-row
  `<br>` or HTML for multiline cells.
- HTML supports thead/tbody/tfoot, th/td, spans, captions, inline tags, br and
  heading/paragraph context. Leading consecutive all-th rows or thead rows become
  headers. Expanded spans repeat the shared value per logical row/column. CSS
  and layout are ignored.
- Table tags and their inline tags must close explicitly and nest correctly.
  Omitted td/tr/section endings, overlapping/cross-group/out-of-table spans and
  text outside cells/captions are rejected. Script/style content is ignored,
  never evaluated; no DTD fetching or network I/O. In Markdown, embedded HTML
  tables must start on their own line. Use `--format html` for full HTML files.
- Buffered mode: 10 MiB UTF-8 input, 10,000 output chunks; `split_document`
  returns a list. HTML positive spans: 1..256; zero rowspans resolve to the group
  end. Expanded HTML grids: 100,000 cells.
  Markdown/HTML streaming has separate working-buffer limits described above. Benchmark
  options bound generated fixtures and allow at most 30,000 replay queries.
- No PDF/image extraction, OCR, XLSX/formulas, LLM summaries, SQL, template
  execution, workflow plugins, archive extraction or implicit subprocesses.
  Only explicit output paths are written; there are no service/dry-run actions.
