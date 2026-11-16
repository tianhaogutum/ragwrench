# ragwrench highlight

Document parsers such as Docling and Unstructured record where every chunk sits on the PDF
page, but that provenance usually ends up in JSON that no reader ever sees.
`ragwrench highlight` turns it into standard PDF highlight annotations and `#page=`
citation links, so a reader can open a citation and see the highlighted passage. Use it
to build checkable citations for RAG answers over PDFs, or to audit how a parser mapped
chunks onto pages.

_Formerly the standalone `cite-highlight-pdf` tool (`citehl`)._

Turn Docling or Unstructured provenance into actual PDF `/Highlight` annotations,
page-local chunk boxes, and `#page=` citation links. Runs entirely offline after
installation; no Docling/Unstructured runtime, OCR, model, server, or API key.

## Quick start

Run these from a clone of the repository (the examples live in `examples/highlight/`):

```bash
pip install ragwrench
ragwrench highlight normalize --format docling examples/highlight/sample.docling.json \
  --pdf examples/highlight/sample.pdf -o examples/highlight/normalized.jsonl
ragwrench highlight annotate examples/highlight/sample.pdf examples/highlight/normalized.jsonl \
  -o examples/highlight/highlighted.pdf
ragwrench highlight verify examples/highlight/highlighted.pdf examples/highlight/normalized.jsonl
```

The last command reports `OK 2/2`, including `text_checked: 2`. Open
`examples/highlight/highlighted.pdf` in a PDF viewer to see the yellow highlights.
`examples/highlight/citations.json` contains links to both pages. These generated files are
not required fixtures; delete them or pass `--force` on subsequent runs.

`python -m ragwrench.highlight` is equivalent to `ragwrench highlight`.

## Normalize provenance

```bash
ragwrench highlight normalize --format unstructured examples/highlight/sample.unstructured.json \
  --pdf examples/highlight/sample.pdf -o examples/highlight/pixel-regions.jsonl
```

Docling accepts a document object with `texts`, `tables`, and `pictures`. Each
item needs `self_ref` and nonempty `prov` with `page_no` and `bbox`. Page sizes
come from `pages["N"].size`; without them, `--pdf` supplies MediaBox dimensions
and a warning. `charspan` selects the corresponding slice of item text. Table
text falls back to `data.table_cells[].text`; pictures can have empty text.
Unstructured accepts an element array with `element_id`, `text`, and
`metadata.page_number` plus `metadata.coordinates` (`points`, `system`,
`layout_width`, `layout_height`). A chunk containing `metadata.orig_elements`
is expanded automatically. This can be an array or base64-encoded zlib/gzip JSON.

Normalized output is UTF-8 JSONL, one rectangular region per provenance entry:

```json
{"chunk_id":"#/texts/0","source_ref":"#/texts/0","page":1,"rect":[72.0,686.0,300.0,700.0],"text":"Revenue grew 12%."}
```

All output rectangles use **absolute, unrotated PDF user-space points**, bottom-left
origin, with ordered `[x0,y0,x1,y1]`; pages are 1-based. Identical provenance
entries are deduplicated. `source_ref` is an opaque identifier, never a file path.

### Coordinate contracts

Producer pipelines can emit different page extents and axes. Defaults are:

| Input dialect | Source origin | Source page extent | Source axes |
| --- | --- | --- | --- |
| Docling bbox | `coord_origin`: TOPLEFT (default) or BOTTOMLEFT | MediaBox | unrotated |
| Unstructured PointSpace | bottom-left | MediaBox | unrotated |
| Unstructured PixelSpace | top-left | CropBox | displayed, after `/Rotate` |

Dimensions in the JSON scale to the chosen PDF box. PixelSpace always requires
`--pdf`; without a PDF, PointSpace and Docling assume an origin-zero page of the
specified point dimensions. Supply the PDF whenever it has a crop, offset, or
rotation. No automatic inference from aspect ratio is attempted.

Override defaults to match your producer, for example a Docling pipeline that
measured a rendered, cropped, rotated page:

```bash
ragwrench highlight normalize --format docling input.json --pdf source.pdf \
  --page-box crop --coordinate-frame display -o regions.jsonl
```

`display` coordinates are inverse-mapped through `/Rotate` 0/90/180/270, including
nonzero box origins. `unrotated` coordinates already use native PDF axes and are
not rotated twice. Boxes are clipped to the intersection of CropBox and MediaBox
with a warning; `--strict` makes any partial overflow an error. A wholly invisible
or zero-area box is always an error. Invalid page numbers, unknown systems,
nonfinite numbers, unsupported `/UserUnit`, and missing provenance fail explicitly.

## Chunk membership and consolidation

A compact mapping file is simplest:

```json
{"quarterly-summary":["#/texts/0","#/texts/1"]}
```

```bash
ragwrench highlight consolidate examples/highlight/regions.jsonl \
  --chunks examples/highlight/chunks.docling.json --per-page -o examples/highlight/chunk-regions.jsonl
```

Also accepts chunk arrays carrying `chunk_id` and `meta.doc_items[].self_ref`,
Unstructured `element_id` with `element_ids` (top-level or metadata), or
`metadata.orig_elements`. Native Docling exports without IDs get a deterministic
`docling:<24-hex-content-hash>` ID. Add your own `chunk_id` to keep citation IDs
stable when editing chunk text. References must resolve; missing members fail.
A source may belong to multiple chunks. Unreferenced source elements are excluded.
`normalize --chunks FILE` applies the same membership mapping without merging.

Default consolidation merges boxes with top and bottom edges within 2 points
and a horizontal gap at most 8 points. Tune `--line-tolerance` and `--max-gap`.
It preserves distinct columns and lines. `--per-page` instead unions all boxes
for each chunk/page (including whitespace). Neither mode ever joins pages or
chunks. Merged regions retain a primary `source_ref` and sorted `source_refs`.
Remapping only some source references of an already merged box is rejected;
apply new membership mappings to the original normalized regions instead.
Line merging uses native PDF X/Y axes; for vertically oriented text, keep source
boxes separate or choose a per-page union.

## Annotate, link, and verify

```bash
ragwrench highlight annotate source.pdf regions.jsonl -o highlighted.pdf
ragwrench highlight verify highlighted.pdf regions.jsonl --text-check required
ragwrench highlight manifest highlighted.pdf regions.jsonl -o extra-citations.json
```

`annotate` clones the PDF, preserves unrelated annotations, and writes printable
yellow highlights with `/Rect`, `/QuadPoints`, `/Contents` (chunk label + text),
and a stable `/NM` derived from the region. Re-annotating replaces **all** prior
`citehl:v1:` annotations with the supplied set, so reruns neither duplicate nor
leave stale tool annotations. The original PDF is never overwritten.

The default manifest is `citations.json` beside the output PDF; choose
`--manifest another-name.json` if needed. The manifest must share the PDF directory
so percent-encoded filename links resolve locally. It contains:

```json
{"schema_version":1,"coordinate_space":"unrotated-pdf-user-space","pdf":"highlighted.pdf","chunks":{"quarterly-summary":{"pages":[1,2],"rects":[{"page":1,"rect":[72,686,300,700],"source_ref":"#/texts/0","source_refs":["#/texts/0"],"annotation_id":"citehl:v1:..."}],"links":["highlighted.pdf#page=1","highlighted.pdf#page=2"]}}}
```

(The example abbreviates `rects` and the hash.) Deep links select a page; visible
highlights come from the PDF annotations. A viewer must support those features.

Check a saved manifest against the current PDF and expected regions without
writing any files:

```bash
ragwrench highlight check-manifest highlighted.pdf regions.jsonl --manifest citations.json
ragwrench highlight check-manifest reviewed.pdf original.jsonl --manifest citations.json \
  --resolutions review.json
ragwrench highlight check-manifest cleared.pdf empty.jsonl --manifest citations.json --allow-empty
```

`check-manifest` first validates region geometry (or independently replays the
edited preflight report), runs the existing PDF verifier, then compares the saved
JSON to a freshly generated manifest. The saved manifest cannot authorize geometry
or review decisions. Success exits 0 and prints one line, for example
`OK check-manifest {"annotations": 2, "chunks": 2}`. Counts appear only after
**both** PDF verification and manifest comparison pass. Invalid input or mismatch
exits 1 with an actionable stderr diagnostic and no stdout or partial counts;
missing/unknown arguments are argparse usage exit 2. No output or force
option is accepted. Library warnings may use stderr.

For automation, append `--json` (all existing checker options apply):

```bash
ragwrench highlight check-manifest highlighted.pdf regions.jsonl --manifest citations.json --json
ragwrench highlight check-manifest reviewed.pdf original.jsonl --manifest citations.json \
  --resolutions review.json --json
ragwrench highlight check-manifest cleared.pdf empty.jsonl --manifest citations.json --allow-empty --json
```

Valid command syntax produces exactly one newline-terminated JSON object on
stdout. Success exits 0:

```json
{"format":"citehl-check-manifest-result","schema_version":1,"ok":true,"summary":{"annotations":2,"chunks":2},"error":null}
```

`schema_version` is integer 1 and `ok` is boolean. Success counts are nonnegative
integers recomputed from verified PDF annotations and regenerated chunks, never
copied from the saved manifest. Failure exits 1 with `ok: false`, `summary: null`
and an `error` object containing string `category`, string-or-null `path`, and
human-readable string `message`. No partial counts are exposed.

- `input_error`: same-directory check, loading PDF/regions/report/manifest, or
  raw region geometry validation failed; `path: null`.
- `replay_error`: the loaded review report failed independent authorization
  replay; `path: null`.
- `verification_error`: PDF annotation/text verification failed; `path: null`.
- `manifest_mismatch`: exact regenerated-manifest comparison failed. `path`
  identifies the first field or container, for example
  `manifest['chunks']['summary/中文']['rects'][0]['source_ref']`. Array indices
  are zero-based; object-shape or array-length errors name the container. Keys
  use Python string repr in brackets, preserving arbitrary chunk IDs. The path
  comes directly from the comparator, without parsing diagnostic text.

Stages retain the text checker's order: directory, PDF/regions loading,
report loading/replay or geometry validation, manifest loading, PDF verification,
then manifest comparison. Only the first failing stage is returned; a verifier
message may describe multiple annotation/text failures. Messages are not stable
machine codes. Handled failures do not print a `ragwrench highlight:` stderr diagnostic in
JSON mode; library warnings may still use stderr. Missing/unknown options,
invalid tolerance/text mode, `-o`, or `--force` remain argparse stderr-only
usage exit 2 with **no JSON**. Help remains text, exit 0. Without `--json`,
the existing text output and errors are unchanged. Both modes are read-only.

The checker uses the existing writer's v1 schema without adding fields:

- The top object contains exactly integer `schema_version: 1`, string
  `coordinate_space: "unrotated-pdf-user-space"`, `pdf` (the literal input PDF
  basename), and the `chunks` object with exactly the expected chunk IDs.
- Each chunk has exactly `pages`, `rects`, and `links`. Pages are sorted unique
  1-based integers; links follow that page order. Each link is the UTF-8
  percent-encoded PDF basename plus `#page=N` (including encoding spaces, `%`,
  `#`, and non-ASCII characters). The literal `pdf` field remains unencoded.
  Manifest and PDF must share their resolved parent directory, as for the writer.
- Rect entries retain input/resolved order **within each chunk**, including
  multiple regions on one page. Each has exactly integer `page`, four-number
  `rect`, string `source_ref`, complete ordered `source_refs` (normalized sorted
  references, or the singleton primary reference), and string `annotation_id`.
  The existing ID binds the complete region, including citation text/provenance.
  Text and counts are not additional manifest fields.

Object key order and JSON formatting may differ. Integer fields reject booleans
and floats; coordinates accept equivalent finite integer/float spellings with
**no rounding tolerance**. Missing, extra, changed or reordered array entries,
fields, chunks, IDs, provenance, pages and links fail; duplicate JSON keys fail
at any nesting depth. `--tolerance` (default 0.01 point) and `--text-check`
(`best-effort` default, `required`, or `off`) apply to PDF verification only.
Unexpected/duplicate/modified tool annotations still fail; foreign annotations
remain outside its namespace.

An empty materialized JSONL requires `--allow-empty`; its manifest must have
`chunks: {}` and the PDF must have zero tool annotations. All-skip resolution
replay also permits that result, but original regions with `--resolutions` must
remain nonempty even with `--allow-empty`. Existing 32 MiB JSON/JSONL and 128 MiB
PDF input limits apply. The manifest is derived provenance, not a signature or
document fingerprint. Keep original regions and the edited report for replay.
The standalone `manifest` writer still only generates links/rects; use this
checker to verify agreement with actual annotations.

`verify` reopens the PDF, requires an exact set of tool annotation IDs, and checks
pages, subtype, rects, quad points, contents, and visibility. Numeric tolerance is
0.01 point (`--tolerance`); changed source regions still change the ID. Missing,
extra, duplicated, moved, or modified tool annotations return nonzero. Other
applications' annotations are outside the verification namespace.

Text overlap is an additional heuristic for ordinary horizontal Type1/TrueType
text: font widths when present, otherwise approximate 0.5-em word advances.
It checks at least one shared Unicode word token inside each rect. Complex fonts,
scanned pages, missing text, and unsupported matrices may be unavailable:
`--text-check best-effort` (default) reports `text_skipped`, `required` fails on
unavailable evidence, and `off` disables this additional check. This is not a
semantic grounding score, exact glyph extraction, OCR, or tamper-proof signature.

For automation, add `--json` to `verify` (all existing verification options apply):

```bash
ragwrench highlight verify highlighted.pdf regions.jsonl --text-check required --json
ragwrench highlight verify reviewed.pdf original.jsonl --resolutions review.json --json
ragwrench highlight verify cleared.pdf empty.jsonl --allow-empty --json
```

Success exits 0 and writes exactly one newline-terminated JSON object to stdout:

```json
{"format":"citehl-verify-result","schema_version":1,"ok":true,"summary":{"annotations":2,"text_checked":2,"text_skipped":0,"text_check":"required","notes":[]},"error":null}
```

Version 1 has exactly `format`, integer `schema_version`, boolean `ok`, `summary`,
and `error`. The success summary is the existing independently verified report:
nonnegative integer `annotations`, `text_checked`, `text_skipped`, the selected
`text_check` string, and `notes` (an array of text extraction diagnostic strings).
Counts appear only after **all** annotation and requested text checks pass.
With `best-effort`, checked plus skipped equals annotations; with `required`,
all annotations must be text-checked; with `off`, both text counts are 0.
An explicit empty set (or all-skip resolution replay) has all counts 0.

Handled failures exit 1 and write the same envelope with `ok: false`,
`summary: null` (no partial counts), and an error object containing exactly
`category` and `message` strings. Categories are `input_error` for loading/parsing
the PDF, regions or report and ordinary region geometry validation; `replay_error`
for failed resolution authorization/document/evidence/decision checks after the
inputs load; and `verification_error` for annotation integrity or text evidence
failures. The existing verifier rejects missing, extra, duplicate, moved or
modified annotations; a review decision cannot bypass the selected text mode.
Messages identify failed checks and are human-readable, not stable
machine codes. No file is written. Original regions with `--resolutions` must
remain nonempty even with `--allow-empty`.

JSON mode puts handled diagnostics on stdout without a stderr error line;
PDF/library warnings may still use stderr. Missing/unknown arguments, invalid
tolerance and invalid text mode remain argparse usage exit 2 with stderr only
and **no JSON**. Help is unchanged. Without `--json`, existing text output and
diagnostics are unchanged. JSON results are available on `verify` and
`check-audit` and `check-manifest`; other commands do not accept this flag.

Normalize with `--pdf` before annotating to persist any clamped rects. `normalize`
still warns and clamps partial overflow (or rejects it with `--strict`). `annotate`
now requires visible, reviewed boxes: it never silently clips raw JSONL. Its
`--strict` flag remains accepted for compatibility; annotation geometry checks
are always strict.

## Preflight and resolve uncertain boxes

`annotate` now checks text evidence **before writing either output**. A box is
`ready` when at least one expected Unicode word overlaps it, `ambiguous` when
a shared word occurs in multiple distinct extracted word boxes inside it, and
`unverifiable` for missing/failed text extraction, a text mismatch, or geometry
outside the visible page. These are heuristics using the same approximate simple
font word boxes as `verify`, not a semantic grounding assessment. Repeated common
words can be flagged even in a legitimate paragraph. Scans, pictures, empty
citation text, and unsupported fonts need an explicit manual decision.

Create an editable JSON report/resolution template:

```bash
mkdir -p /tmp/citehl-review-demo
ragwrench highlight normalize --format unstructured examples/highlight/review-rotated.unstructured.json \
  --pdf examples/highlight/review-rotated.pdf -o /tmp/citehl-review-demo/regions.jsonl
ragwrench highlight preflight examples/highlight/review-rotated.pdf /tmp/citehl-review-demo/regions.jsonl \
  -o /tmp/citehl-review-demo/review.json
```

This redistributable fixture contains readable text on four pages with offset
MediaBox/CropBox and rotations 0/90/180/270. Expected summary: `total: 7`,
`ready: 4`, `ambiguous: 1`, `unverifiable: 2`, `needs_resolution: 3`. Preflight
exits 0 when it successfully writes a report, even if decisions are needed;
malformed input or invalid page numbers fail with exit 1. It does not annotate.

Each `items[]` entry has a stable annotation ID, original region, status, reasons,
shared-word evidence, and `resolution: null`. **Edit only `resolution` fields**.
For this example, assign these decisions by `region.source_ref`:

| Source ref | Resolution value |
| --- | --- |
| `ambiguous` | `{"action":"replace","page":2,"rect":[220,120,290,140],"reason":"Select the second occurrence after visual inspection"}` |
| `blank` | `{"action":"accept","reason":"Retain the manually checked non-text citation"}` |
| `mismatch` | `{"action":"skip","reason":"Producer text does not match this page"}` |

Leave the four ready entries null. `accept` retains the original box despite
uncertain text evidence; `skip` omits it; `replace` changes only page/rect in
absolute unrotated PDF points and must produce a ready box. Every decision needs
a nonempty reason. Acceptance never bypasses bounds checks: replace or skip an
out-of-bounds box. Ready entries may also be explicitly skipped or replaced.

After editing, apply and verify against the **original** region file:

```bash
ragwrench highlight annotate examples/highlight/review-rotated.pdf /tmp/citehl-review-demo/regions.jsonl \
  --resolutions /tmp/citehl-review-demo/review.json -o /tmp/citehl-review-demo/reviewed.pdf
ragwrench highlight verify /tmp/citehl-review-demo/reviewed.pdf /tmp/citehl-review-demo/regions.jsonl \
  --resolutions /tmp/citehl-review-demo/review.json
ragwrench highlight annotate /tmp/citehl-review-demo/reviewed.pdf /tmp/citehl-review-demo/regions.jsonl \
  --resolutions /tmp/citehl-review-demo/review.json -o /tmp/citehl-review-demo/again.pdf --force
```

Expected: 6 annotations, 5 text checks and 1 unavailable text check; the blank box
was explicitly accepted. `--text-check required` still fails for it; decisions
do not suppress verification evidence. Reapplication retains 6 annotations
(per-page counts 1/3/1/1), and the manifest reflects the replacement and omission.
Skipping all items clears the tool annotation namespace while preserving foreign
annotations. Without decisions, this input fails with exit 1 and writes nothing.

Reports use `format: "citehl-preflight"`, integer `schema_version: 1`, the explicit
native-point coordinate contract, and the same ordered page fingerprints as
sidecars below. On application, the full report/evidence is recomputed. Changed
page data, edited/reordered regions, missing/extra/duplicate IDs, modified report
fields, unresolved items, and invalid decisions fail before replacing outputs,
including with `--force`. Annotation arrays are excluded from page identity, so
the same report applies to the resulting PDF on a repeat run. A report is not a
signature; editing a decision is deliberate user authorization.

Materialize the reviewed set for standalone manifests, verification, or legacy
sidecar export without reconstructing regions by hand:

```bash
ragwrench highlight resolve examples/highlight/review-rotated.pdf /tmp/citehl-review-demo/regions.jsonl \
  --resolutions /tmp/citehl-review-demo/review.json -o /tmp/citehl-review-demo/final.jsonl \
  --audit /tmp/citehl-review-demo/resolution-audit.json
ragwrench highlight manifest /tmp/citehl-review-demo/reviewed.pdf /tmp/citehl-review-demo/final.jsonl \
  -o /tmp/citehl-review-demo/standalone-citations.json
ragwrench highlight export /tmp/citehl-review-demo/reviewed.pdf /tmp/citehl-review-demo/final.jsonl \
  -o /tmp/citehl-review-demo/reviewed.citehl.json
ragwrench highlight verify /tmp/citehl-review-demo/reviewed.pdf /tmp/citehl-review-demo/final.jsonl
```

`resolve` requires the original **nonempty** normalized JSONL and an edited
preflight report supplied with `--resolutions`. It recomputes the same page
fingerprints, evidence and decision checks as `annotate`, rejects unresolved or
stale/modified reports, and validates replacement boxes and final ID uniqueness
before writing. It writes canonical UTF-8 JSONL: original order, accepted and
unchanged ready regions retained, replacement page/rect applied, skipped entries
omitted, with text and source references preserved. The result must fit the
32 MiB JSONL input limit. Existing outputs need `--force`; invalid input never
replaces an output, even with that flag. Omitting `--resolutions` is usage exit 2;
an invalid report is exit 1. A ready-only set can use its unedited preflight report.

Optional `--audit OUTPUT_JSON` also writes a machine-readable review mapping;
omitting it creates only JSONL, with identical bytes. Audit v1 contains:

- `format: "citehl-resolution-audit"`, integer `schema_version: 1`,
  `coordinate_space` with the native-point contract below, and `document` with
  the same algorithm, document hash and ordered page fingerprints as the validated
  preflight report. Annotation-only changes preserve this identity.
- `summary`: nonnegative integer `original_count`, `final_count`, `ready`, `accept`,
  `replace`, `skip`. Action counts sum to `original_count`;
  `final_count = original_count - skip`.
- `items`: one object per original region in input order, containing exactly
  `annotation_id` (original ID), `action` (`ready`, `accept`, `replace`, `skip`),
  `reason` (the supplied string, or null for implicit ready), `final_annotation_id`
  and `final_region` (the complete normalized region). Replacement IDs match the
  materialized JSONL and its manifest. Skips have **both final fields null**.
  An explicit decision on a ready item retains that action, including a replacement
  with identical geometry and thus the same ID.

The audit is UTF-8 JSON with a final newline, capped at 32 MiB, and is deterministic
for the same inputs. It is a derived review record, not a replay input or signature.
All decisions, both serialized sizes and both output paths are checked before
writing, including with `--force`. Audit/JSONL outputs cannot alias each other or
the PDF, original JSONL or report, including via symlinks or existing hard links.
JSONL is written first, then the audit; each file is individually atomic. A later
audit I/O failure returns exit 1 and can leave completed JSONL beside an unchanged
or absent audit. The pair is not a transaction.

Check that a saved audit and materialized JSONL still agree with the source and
the edited report, without writing any files:

```bash
ragwrench highlight check-audit examples/highlight/review-rotated.pdf /tmp/citehl-review-demo/regions.jsonl \
  --resolutions /tmp/citehl-review-demo/review.json \
  --audit /tmp/citehl-review-demo/resolution-audit.json \
  --final-regions /tmp/citehl-review-demo/final.jsonl
```

All five inputs are required. `check-audit` independently replays the existing
resolution checks against the PDF, original **nonempty** regions, and edited
report. Only then does it compare the saved audit and final JSONL. The audit
cannot authorize acceptance or substitute for the report. Success exits 0 and
prints only `OK check-audit` plus the recomputed summary counts; invalid input or
a mismatch exits 1 with an error identifying the failed field or replay check.
Missing required arguments are usage exit 2. There are no output or force flags.

The v1 audit must match exactly: object fields and scalar types, ordered document
fingerprints and item mappings, all IDs, actions, exact reasons, null skip fields,
regions, and counts. Version/count/page-index fields require integers, excluding
booleans and floats. Final JSONL must contain the complete normalized regions in
materialized order; missing/extra fields or entries and changed text/provenance
fail. JSON object key order, whitespace, and equivalent finite integer/float
coordinate spellings may differ. Coordinates have no rounding tolerance.
Duplicate JSON keys fail in every JSON/JSONL input. Each JSON/JSONL file retains
the 32 MiB limit. An empty or blank-line-only final JSONL is accepted without
`--allow-empty` **only when replay produces zero regions**. Original regions must
still be nonempty. Annotation-only PDF changes preserve the report identity;
this checks resolution artifacts, not the PDF's current annotation set (use
`verify` for that). Matching artifacts are provenance, not signatures.

For automation, append `--json` to `check-audit`. For valid command syntax,
stdout contains exactly one newline-terminated JSON result, with no text prefix:

```json
{"format":"citehl-check-audit-result","schema_version":1,"ok":true,"summary":{"original_count":7,"final_count":6,"ready":4,"accept":1,"replace":1,"skip":1},"error":null}
```

The result always has exactly these five fields. `schema_version` is integer 1,
`ok` is a boolean, and success `summary` contains the six recomputed integer
counts (including `final_count: 0` for an all-skip review). Failure exits 1 with
`ok: false`, `summary: null`, and `error` containing exactly `category`, `path`,
and a nonempty human-readable `message`. Categories are:

- `input_error`: a PDF or JSON/JSONL file could not be loaded or parsed.
- `replay_error`: original regions, report, document identity, evidence or
  decisions failed revalidation.
- `artifact_mismatch`: the saved audit or final JSONL differs from replay.

Mismatch `path` identifies the first differing field, for example
`audit.items[5].reason`, `audit.summary.final_count`, or
`final regions[0].rect[0]`. An object-shape or array-length mismatch names its
containing object/array. Input and replay errors have `path: null`. For example:

```json
{"format":"citehl-check-audit-result","schema_version":1,"ok":false,"summary":null,"error":{"category":"artifact_mismatch","path":"audit.items[5].reason","message":"audit.items[5].reason: value or type differs; regenerate with ragwrench highlight resolve --audit using the original PDF, regions and edited report"}}
```

Consumers should use category/path rather than parse `message`. File loading
precedes replay, which precedes artifact comparison; the first failure is
reported without partial counts. In JSON mode, handled failures emit their
diagnostic in the result instead of stderr; PDF/library warnings can still use
stderr. Missing/unknown arguments remain argparse usage exit 2, with stderr and
no JSON result; `--help` retains its usual behavior. Success remains exit 0.
This flag writes no files; omitting it preserves the default text output.
`verify` also supports JSON results using its own schema described above.

**Empty-set contract:** an all-skipped review successfully writes a **zero-byte
JSONL file**. An optional audit still has one record per original item, all with
action `skip` and null final fields; `final_count` is 0. `manifest`, `verify`, and
`export` with a JSONL argument require
`--allow-empty` to consume an empty (or blank-line-only) file. The manifest has
`chunks: {}`, verification reports `OK 0/0` only when no tool annotations exist,
and export writes a fingerprinted sidecar with `annotations: []`. Unexpected
tool annotations still fail verification/export. For example, after resolving
and annotating an all-skip report into `cleared.pdf`:

```bash
ragwrench highlight manifest cleared.pdf empty.jsonl --allow-empty -o empty-citations.json
ragwrench highlight verify cleared.pdf empty.jsonl --allow-empty
ragwrench highlight export cleared.pdf empty.jsonl --allow-empty -o empty.citehl.json
```

`normalize`, `consolidate`, and original inputs to `preflight`, `resolve`,
`annotate`, or `verify --resolutions` retain their nonempty-input requirement.
To clear highlights, annotate the original regions with an all-skip report.
Materialized JSONL stores final geometry/provenance; the optional audit adds
reasons, page fingerprints and original-to-final IDs. Retain the original regions
and edited report to replay the review authorization.
An accepted uncertain region can still need a decision if used as fresh input
to `annotate`; use the original regions plus report to replay the authorization.

The report and its decisions must fit the 32 MiB JSON input limit. Preflight
fingerprinting inherits the sidecar resource limits below. The CLI gates new
`annotate` calls; sidecar import does not repeat text preflight. Newly written
highlights carry the final accepted/replaced regions, so PDF-only `export` below
also works after resolution decisions. Standalone `manifest` still requires the
final region set; use `resolve` to produce it from the resolution report.

## Portable annotation sidecars

Export the PDF's verified `ragwrench highlight` highlights and their provenance to one UTF-8
JSON file without the original JSONL, then restore them onto a matching copy:

```bash
mkdir -p /tmp/citehl-sidecar-demo
ragwrench highlight annotate examples/highlight/sample.pdf examples/highlight/regions.jsonl \
  -o /tmp/citehl-sidecar-demo/highlighted.pdf
ragwrench highlight export /tmp/citehl-sidecar-demo/highlighted.pdf \
  -o /tmp/citehl-sidecar-demo/highlights.citehl.json
ragwrench highlight import examples/highlight/sample.pdf /tmp/citehl-sidecar-demo/highlights.citehl.json \
  -o /tmp/citehl-sidecar-demo/restored.pdf
ragwrench highlight verify /tmp/citehl-sidecar-demo/restored.pdf examples/highlight/regions.jsonl \
  --text-check required
ragwrench highlight import /tmp/citehl-sidecar-demo/restored.pdf \
  /tmp/citehl-sidecar-demo/highlights.citehl.json -o /tmp/citehl-sidecar-demo/again.pdf
```

Expected: export reports 2 annotations; first import reports 2 added and 0 already
present; verification reports `OK 2/2`; second import reports 0 added and 2 already
present. Use fresh paths or `--force` when repeating these commands.

New highlights created by `annotate` or added by `import` store a PDF text string
under the reserved `/CiteHL` annotation key. It contains canonical UTF-8 JSON with
`format: "citehl-region"`, integer `schema_version: 1`, `annotation_id`, and the
complete normalized `region`, including text and merged `source_refs`. Version 1
uses the native-point coordinate contract below. The metadata ID must equal both
the SHA-256 region ID and the annotation's `/NM`; malformed/unknown metadata or
inconsistent IDs fail export, verify, and import. Renaming or removing `/NM` does
not hide a metadata-bearing annotation from these checks. Metadata is provenance,
not a signature or proof of text grounding.

`ragwrench highlight export PDF -o JSON` recovers regions in page/annotation order, checks bounds
and the exact annotation set, and writes the existing sidecar format. Missing
metadata on any `ragwrench highlight` highlight fails with exit 1; a mixed old/new set cannot
silently export only its new highlights. For legacy PDFs, supply the final region
set: `ragwrench highlight export PDF regions.jsonl -o JSON`. This fallback allows absent metadata
but never bypasses invalid metadata that is present. Export also rejects duplicate
IDs, moved highlights, or changed geometry/contents/visibility before replacing
output, including with `--force`. A PDF with no `ragwrench highlight` highlights exports a
valid fingerprinted empty sidecar (`annotations: []`) without needing JSONL.
Foreign annotations without `/CiteHL` are ignored.

Embedded payloads are limited to 32 MiB each; PDF-only export also limits their
combined UTF-8 size to 32 MiB and their count to 100,000. The final sidecar must
fit the same 32 MiB import limit. Existing matching legacy annotations skipped by
additive import are preserved without metadata backfill; use the JSONL fallback
for their export. Viewers may strip custom metadata, requiring that same fallback.

The sidecar has `format: "citehl-annotation-sidecar"`, integer `schema_version: 1`,
`document` fingerprints, and `annotations` containing each stable `annotation_id`
and full normalized `region` (including text and source references). Its explicit
`coordinate_space` is:

```json
{"name":"unrotated-pdf-user-space","units":"points","origin":"bottom-left","rect":"[x0,y0,x1,y1]","page_index_base":1}
```

`document.algorithm` is `citehl-page-sha256-v1`. Every page, including pages with
no highlights, has a SHA-256 fingerprint of normalized MediaBox/CropBox/rotation
and the remaining page dictionary with resolved content/resources. Object numbers,
the page-tree parent, and annotation arrays are excluded. Container children use
typed SHA-256 digests, dictionary keys are sorted, and streams hash decoded bytes
instead of compression fields (`/Length`, `/Filter`, `/DecodeParms`). An ordered
array of `{page, fingerprint}` records is itself hashed for the document identity.
Renaming, document metadata changes, annotation edits, and supported lossless
stream recompression do not change this identity.

### Check a saved sidecar without writing files

```bash
ragwrench highlight check-sidecar /tmp/citehl-sidecar-demo/highlighted.pdf \
  --sidecar /tmp/citehl-sidecar-demo/highlights.citehl.json
# Legacy annotations without embedded metadata: use the final JSONL export order.
ragwrench highlight check-sidecar legacy.pdf --sidecar legacy.citehl.json --regions final.jsonl
# Empty legacy fallback requires explicit opt-in.
ragwrench highlight check-sidecar cleared.pdf --sidecar empty.citehl.json \
  --regions empty.jsonl --allow-empty
```

Success exits 0 and prints one line, for example
`OK check-sidecar {"annotations": 2, "pages": 2}`. Counts appear only after
`export_sidecar` verifies the current PDF and the entire regenerated sidecar
matches. Invalid input, metadata, annotation integrity, or artifact mismatch
exits 1 with no stdout or partial counts; stderr identifies the failure.
Comparison reports the first differing field/container path and a regeneration
hint. Missing/unknown options are usage exit 2; help exits 0. This command accepts
no output, force, JSON-result, text-check or tolerance flags and writes no files.
`--allow-empty` without `--regions` is usage exit 2.

The expected data comes entirely from the current PDF, or from explicitly supplied
final JSONL verified against that PDF. Saved regions, IDs and fingerprints never
supply expectations. PDF-only order is ascending page order followed by each
page's annotation array order. `--regions` preserves final JSONL order, even across
pages. Use the same mode/order as the original export. Foreign annotations are
ignored. Missing metadata needs the explicit legacy fallback; present invalid
metadata fails in both modes. As for export, PDF geometry tolerance is 0.01 point
and text overlap is off; this does not establish semantic grounding or replay
review authorization.

The checker requires exactly the v1 export fields at every level:

- Root: `format`, `schema_version`, `coordinate_space`, `document`, `annotations`.
  The coordinate-space object must contain exactly the five fields/values above.
- Document: exactly `algorithm`, `fingerprint`, `pages`; ordered page entries
  have exactly integer `page` and string `fingerprint`. Both document and page
  fingerprints must equal those independently regenerated from the current PDF.
- Each annotation: exactly string `annotation_id` and `region`. Each region has
  exactly string `chunk_id`, string `source_ref`, integer `page`, four-number
  `rect`, string `text`, plus ordered string-array `source_refs` **only when the
  exporter emits it**. Missing text/provenance is not defaulted; extra fields
  are not discarded. Complete IDs, regions and reference order must match.

JSON whitespace, escaping and object key order may differ. Integer fields reject
booleans and floats; rect coordinates accept equivalent finite integer/float
spellings with no rounding tolerance. Missing/extra/reordered array entries,
changed fields/types and duplicate JSON keys at any depth fail. An empty saved
annotation array succeeds only when export independently verifies zero tool
annotations. PDF-only empty checks need no JSONL; empty fallback JSONL needs
`--allow-empty`. This exact-current-set check does not change additive import:
importing an empty sidecar still preserves all existing annotations. Existing
input, metadata and fingerprint resource limits apply. Fingerprints and embedded
provenance are not signatures or original-document authentication.

Import requires the entire ordered page fingerprint table to match. A PDF with
different content or fonts, added/reordered pages, a changed crop, or rotation
fails with `document/page fingerprint mismatch`, exit 1, before writing output.
Unknown versions/coordinate spaces, duplicate or inconsistent IDs, invalid page
numbers, and out-of-bounds regions also fail. There is no automatic page remapping;
`--force` permits replacing an output file but never bypasses these checks.

Import is additive: foreign annotations and existing `ragwrench highlight` highlights are
retained, including highlights absent from the sidecar. Matching IDs are skipped
after checking page, subtype, geometry, contents, and visibility (0.01-point
tolerance). Conflicting or duplicate existing IDs cause failure without replacing
any annotation or output. Unlike `annotate`, import does not refresh the namespace
and does not write a citation manifest. It writes only the selected output PDF.
An explicit `annotations: []` is a valid additive no-op: import reports 0 added
and 0 already present and writes a matching PDF with all existing foreign and
`ragwrench highlight` annotations preserved. It never means delete. The same format,
coordinate-space, page fingerprint and present-metadata validation still applies.
Missing, null, or non-array `annotations` is invalid.
Object keys must also be unique: duplicate `annotations` fields fail rather than
letting a later empty array silently override a nonempty set. Duplicate fields
at any depth (including page fingerprints) fail before output replacement, even
with `--force`; equivalent JSON-escaped key names count as duplicates.

Export verifies the recovered or supplied region set against the PDF. This version
transports `ragwrench highlight` region metadata and recreates its standard yellow highlight style, not arbitrary
viewer annotations, appearance streams, or viewer-edited styling. Matching copies
may have different unrelated annotations, attachments, or document metadata: the
fingerprint identifies page data, not every byte of the PDF, and is not a signature
or authentication of an untrusted sidecar. Matching is conservative; visually
equivalent PDFs rewritten into different resource/content structures can fail.
Cyclic page-resource graphs, depth over 100, over 100,000 container objects, or
over 128 MiB of decoded streams across the document fail explicitly. These limits
inherit pypdf's decoding behavior and do not sandbox hostile PDFs.

## Inventory persisted annotation IDs

`ragwrench highlight list-annotations PDF` reads a supported current PDF and prints exactly one
newline-terminated JSON object. It writes no files and accepts no output, force,
selection, JSON-mode, text-check or tolerance options. Success exits 0; invalid or
unsupported input/I/O failure exits 1 with stderr and **no stdout or partial
inventory**; missing/unknown arguments exit 2. Help exits 0.

The v1 result contains exactly:

- `format`: `"citehl-annotation-inventory"`; integer `schema_version`: `1`.
- `coordinate_space`: the five-field native-point object documented for sidecars.
- `annotations`: owned entries in ascending page order, then PDF annotation-array
  order. Each has exact string `annotation_id` from persisted `/NM`, integer
  1-based current `page`, current `/Rect` as four finite numbers `[x0,y0,x1,y1]`
  with positive width/height, boolean `metadata_available`, and `region`.
- `summary`: nonnegative integer `owned` and `foreign` counts of page annotation
  entries. `owned` equals the array length. No owned annotations is a success:
  `annotations: []`, `owned: 0`, with the accurate foreign count.

Ownership uses only the reserved `citehl:v1:` `/NM` namespace. IDs must be PDF text
strings with exactly 64 lowercase hex digits after that prefix; malformed reserved
IDs and duplicate owned IDs anywhere (including repeated references) fail closed.
Titles, text and appearance never establish ownership. Every owned entry is
included once. Legacy entries with no `/CiteHL` have `metadata_available: false`
and `region: null`; their persisted IDs are never regenerated from geometry.
Present metadata must pass the strict v1 schema, unique JSON keys, region hash and
`/NM` binding checks; malformed/conflicting metadata anywhere, including metadata
on a renamed/missing `/NM`, fails rather than being treated as legacy/foreign.
Valid metadata yields `metadata_available: true` and the complete recovered region,
including text and ordered merged source references when present.

This is an inventory, not `verify`: current page/rect can differ from recovered
provenance after viewer edits, and out-of-page rects are reported without clamping.
Styles, contents, visibility, quads, text overlap and semantic grounding are not
verified. Invalid/missing owned rects and malformed annotation arrays/dictionaries
fail; foreign rects/payloads without `/CiteHL` are not interpreted. The PDF bytes,
annotations and page content remain untouched. Supported page geometry and 128 MiB
PDF limit apply; at most 100,000 owned entries, 32 MiB per/aggregate embedded
metadata, and 32 MiB serialized inventory JSON (including its newline). PDF parsing
inherits pypdf limits and is not a hostile-PDF sandbox. Namespace ownership and
embedded provenance are conventions, not authentication or signatures.

Inventory and selective removal resolve indirect PDF `/NM` text strings. A
selected indirect ID with valid embedded metadata behaves like the same direct
text string; the metadata still binds to its exact persisted v1 ID and region
hash, even when current page/rect have changed. Neither command regenerates IDs
from current geometry.

`verify` resolves a persisted `/NM` to its PDF text string. `check-manifest`,
`export`, and `check-sidecar` build on it, so they accept a valid embedded or legacy
annotation whose exact v1 ID is stored as an indirect reference, with the same
counts and text-check behavior as direct storage. Conflicting, edited, hidden,
moved, or missing annotations still fail, as do renamed, missing, or non-text IDs
and duplicate IDs (direct or indirect, including repeated references). Identity is
never inferred from geometry. Installed-wheel checks confirmed that `annotate`
re-annotation and additive `import` into an indirect-ID PDF do not create
duplicates.

Still not supported:

- A `/NM` stored as a PDF **name** (for example `/citehl:v1:...`) is not a text
  string. `verify`, `check-manifest`, `export`, and `check-sidecar` reject it.
  Inventory and `remove` still classify such an entry as foreign, so `remove`
  cannot select it.
- Legacy annotations with no `/CiteHL` still need the original regions JSONL for
  `export` and `check-sidecar`; PDF-only recovery is unchanged for direct and
  indirect IDs alike.

## Remove explicitly selected highlights

`remove` writes a separate PDF with only the requested ragwrench highlight annotations removed:

```bash
# Inventory also supports legacy IDs and viewer-edited geometry.
# These redirects are explicit shell captures; list-annotations itself writes no files.
ragwrench highlight list-annotations highlighted.pdf > current-inventory.json
python - <<'PYSELECT'
import json
from pathlib import Path
notes = json.loads(Path("current-inventory.json").read_text())
# Explicitly choose the first owned entry; review current geometry and provenance.
selected = [notes["annotations"][0]["annotation_id"]]
Path("selection.json").write_text(json.dumps({"version": 1, "annotation_ids": selected}) + "\n")
PYSELECT
ragwrench highlight remove highlighted.pdf selection.json -o selected-removed.pdf
ragwrench highlight list-annotations selected-removed.pdf
```

The second inventory lists the exact remaining IDs and owned/foreign counts.
Legacy entries have no recovered region; inventory is not an export or verifier.

Selection JSON must have **exactly** `version` (integer `1`, excluding booleans
and floats) and `annotation_ids` (an array of unique strings). Each ID must be
an exact `citehl:v1:` followed by 64 lowercase hexadecimal digits; no trimming,
case folding, text matching or ID regeneration occurs. For example, an explicit
no-op selection is `{"version":1,"annotation_ids":[]}`. Missing/null/non-array
lists, unknown fields/versions, duplicate IDs or JSON keys, and missing or foreign
IDs fail. Selection is limited to 32 MiB and 100,000 IDs.

The persisted PDF `/NM` in the reserved `citehl:v1:` namespace establishes
ownership. `/T` titles, highlight appearance and `/Contents` text do not.
Legacy annotations with a matching exact ID and no `/CiteHL` metadata can be
removed; obtain their exact persisted IDs with `list-annotations`.
Present **selected** `/CiteHL` metadata must pass the existing strict v1 payload,
region hash and `/NM` binding checks. Renamed/missing IDs are never recovered
from metadata or guessed from geometry. Duplicate PDF IDs in the ragwrench highlight namespace
are ambiguous and fail anywhere in the document, including for an empty selection.
Unselected metadata is preserved without interpretation; removal does not assert
that remaining highlights pass `verify`/`export`. Viewer-edited styling, contents
or geometry do not change ownership and are not reverified to authorize removal.

Success exits 0 and prints one line, for example
`OK removed 1 annotations; preserved 2 -> selected-removed.pdf` for two owned
annotations plus one foreign annotation. `removed` counts selected page annotation
entries; `preserved` counts **all** remaining entries, including foreign and
unselected owned annotations. Selection order does not change retained order.
An empty list succeeds with zero removals and a new no-op PDF copy containing all
original annotations. A repeated request for an already removed ID fails as missing.

PDF, selection and output paths must all be distinct, including symlink/hardlink
aliases. Existing outputs require `--force`; directories cannot be replaced.
The complete selection and supported PDF page geometry are validated before
cloning/writing. The command writes only the chosen output, using a sibling
temporary file and atomic replacement. Invalid input or publication failure exits
1 with stderr, no success/counts, and any existing output unchanged. Missing or
unknown arguments exit 2. Inputs remain byte-for-byte unchanged.

Retained annotation dictionaries, appearance/custom objects, ordering and page
content/resources/rotation are preserved through PDF cloning. Cloning can change
file bytes and object numbers, including for a no-op copy, and invalidates digital
signatures. Removal unlinks selected entries from page annotation arrays; it is
not secure erasure/redaction of PDF bytes or content. It does not update saved
manifests/sidecars, which may now be stale. Additive `import` behavior is unchanged:
an empty sidecar still preserves every existing annotation.

## Errors, boundaries, and limits

- Exit codes: 0 success, 1 invalid/unsupported input or verification/I/O failure,
  2 command usage error. Diagnostics go to stderr; no external commands run.
- Existing outputs require `--force`. Output/input aliases (including symlinks)
  and colliding output paths are rejected. Writes use sibling temporary files
  and atomic replacement. PDF/manifest and resolved JSONL/audit are individually atomic, not a
  two-file transaction: a later manifest write failure can leave a completed PDF
  and returns nonzero. Only user-selected output paths are written.
- JSON/JSONL: 32 MiB per input; decoded `orig_elements`: 32 MiB per stream;
  PDF input: 128 MiB; at most 100,000 regions. Same-line merging is capped at
  2,000 boxes per chunk/page; larger groups must use `--per-page` or smaller chunks.
  PDF processing inherits pypdf's resource limits and is not a sandbox for hostile PDFs.
- All JSON objects require unique field names after JSON escape decoding,
  including producer JSON, region JSONL, preflight decisions, sidecars and embedded
  metadata. Duplicate fields return exit 1 rather than selecting the last value.
  Separate objects may reuse names; matching is case-sensitive.
- Encrypted PDFs (even empty-password), `/UserUnit != 1`, nested `orig_elements`,
  unknown coordinate systems, and invalid/inverted boxes are rejected. Polygons
  become axis-aligned enclosing rectangles. No arbitrary-rotation polygon or
  token-level clipping of a chunk inside a source element.
- No OCR, parser execution, viewer UI, DOCX/HTML support, downloads, workflow
  execution, or network access at runtime. Producer coordinate conventions must
  be selected correctly. No automatic PDF identity matching from provenance.
- Cloning rewrites the PDF; digital signatures are not preserved as valid.
  Existing PDF actions/attachments are preserved, not executed or sanitized.

## Reproduce validation

```bash
pip install -e '.[test]'   # from a clone of the repository
python -m pytest tests/highlight -q
ragwrench highlight annotate examples/highlight/sample.pdf examples/highlight/regions.jsonl -o /tmp/citehl-out.pdf
ragwrench highlight verify /tmp/citehl-out.pdf examples/highlight/regions.jsonl
```

Use fresh output paths or `--force` when repeating. Tests run in a single process,
disable network connections and use synthetic, redistributable PDFs and JSON.
`tests/highlight/test_rendered_rotation.py` renders real source/annotated PDF pages
with Poppler `pdftoppm` and checks highlight placement over the text at all four rotations. This one test skips if Poppler is
absent; install it to run the rendered check. The CLI does not invoke
Poppler. Regenerate committed fixtures with `python examples/highlight/make_examples.py`
and `python examples/highlight/make_review_fixture.py`.

This independent integration uses pypdf to write standard annotations. Docling's image
overlays, integrated RAG viewers, and this local artifact workflow serve different
deployment needs. MIT for original code; see [THIRD_PARTY.md](../THIRD_PARTY.md)
for upstream attribution and schema references.
