# ragwrench sync

Re-embedding a whole corpus whenever one document changes is slow and costly, and blind
upserts can silently delete or duplicate points. `ragwrench sync` snapshots your
documents, shows a reviewable plan of exactly which chunks are new, moved, updated or
deleted, and applies only that diff to a local Qdrant collection. Use it for incremental
reindexing when you want a dry run, review gates on destructive changes and verification.

_Formerly the standalone `chunk-sync-planner` tool (`chunksync`)._

`ragwrench sync` previews and applies incremental chunk indexes to **local Qdrant**.
It distinguishes new embeddings, deletions, metadata updates, and moves. A local
paragraph insertion can reuse later chunks instead of embedding the entire document.
No API key, model download, or running server is needed.

## Quick start

Needs the `qdrant` extra. Run these from a clone of the repository (the fixtures live in
`tests/sync/fixtures/`):

```sh
pip install 'ragwrench[qdrant]'
ragwrench sync snapshot --empty -o empty.json
ragwrench sync snapshot --dir tests/sync/fixtures/docs_v1 -o old.json
ragwrench sync plan empty.json old.json -o bootstrap.json   # exit 10: there are changes
ragwrench sync apply bootstrap.json --qdrant-path .qdrant --collection kb
ragwrench sync verify old.json --qdrant-path .qdrant --collection kb
```

The scripted demos cover journaling, review gates, recovery and policies:

```sh
python examples/sync/demo.py
python examples/sync/journal_demo.py
python examples/sync/review_demo.py
python examples/sync/recovery_demo.py
python examples/sync/policy_demo.py
ragwrench sync bench
```

Tests and demos run offline after installing the dependencies. They use synthetic, MIT-licensed fixtures
from this repository.

The demo reads the committed Markdown fixtures, initializes a temporary local
collection, plans the inserted introduction, applies it, and independently
recomputes every final vector. Actual fixture result: **1 new embedding, 4 reused
vectors, 5 final points**. Temporary demo data is removed on exit.

## Snapshot, plan, apply, verify

```sh
ragwrench sync snapshot --empty -o empty.json
ragwrench sync snapshot --dir tests/sync/fixtures/docs_v1 --chunker cdc -o old.json
ragwrench sync snapshot --dir tests/sync/fixtures/docs_v2 --chunker cdc -o new.json
ragwrench sync plan empty.json old.json -o bootstrap.json
# Expected exit code: 10 (changes), with bootstrap.json written.
ragwrench sync apply bootstrap.json --qdrant-path .qdrant --collection kb
ragwrench sync plan old.json new.json -o plan.json
# Expected exit code: 10 again.
ragwrench sync apply plan.json --qdrant-path .qdrant --collection kb --dry-run
ragwrench sync apply plan.json --qdrant-path .qdrant --collection kb --verify
ragwrench sync verify new.json --qdrant-path .qdrant --collection kb
```

Run these as separate commands: **`plan` returns 0 for equal snapshots, 10 for
changes, and 2 for errors**. Do not chain a changing plan with `&& apply`.
In a shell using `set -e`, accept the intentional status explicitly:

```sh
if ragwrench sync plan old.json new.json -o plan.json; then
    :
else
    code=$?
    test "$code" -eq 10 || exit "$code"
fi
```

`plan --format json` prints the full machine-readable plan. `-o` writes it
atomically. It contains both snapshots and all operations, so apply requires no
source documents. Planning never opens Qdrant or executes an embedder. Apply and
verify return 0 on success or 2 on refusal/error. `--dry-run` validates the plan,
review gates/receipt, and collection name; it does **not** create/open the store,
validate its live fingerprint or stored review policy, or execute any command. Its output says
`collection_checked: false`.

`--qdrant-path :memory:` uses embedded, temporary Qdrant. Use `apply --verify` to
verify it in the same process; a separate CLI invocation cannot reopen memory.

## Compare plans and review destructive changes

Compare two competing plans built from the **same exact old snapshot**:

```sh
ragwrench sync compare baseline-plan.json candidate-plan.json -o comparison.json
# Exit 10 if different, 0 if identical, 2 if invalid or based on different sources.
```

JSON output includes both plan hashes, count deltas (candidate minus baseline),
deletion fractions, target model tags, and operation identities added, removed,
or changed. Moves use `[before_id, after_id]` identities. Changed entries expose
payload differences even when operation counts and IDs stay equal. The original
plans contain the actual payload values. Comparison never accesses Qdrant or an
embedder, and makes no claim that either plan is safe.

Review gates apply to both the CLI and Python API:

- `max_deletion_fraction`: `delete / old_chunks`, with zero for an empty source.
  Renames/moves preserve content and are excluded; replacing content counts as
  deleting the unmatched old chunks. The gate blocks **above** the limit, so an
  exact threshold is allowed. The compatibility default is `1.0`; pass a stricter
  limit for deletion protection. Admitted limits persist per collection; later
  invocations must use the same or a stricter limit unless separately overridden.
- `embed_model_changed`: a changed snapshot model tag is blocked by default,
  including changes between empty snapshots. This is the declared embedding
  schema; changes hidden behind an unchanged tag cannot be detected. Dimension,
  distance and embedder identity mismatches remain hard refusals.

For four old chunks `a, b, c, d` and a candidate retaining only `a`, a limit of
`0.5` reports three deletions, fraction `0.75`, and blocks apply before opening
Qdrant or starting an embedding command:

```sh
ragwrench sync review candidate-plan.json --max-deletion-fraction 0.5 -o review.json
# Exit 3 (blocked); the JSON receipt is still written. Invalid inputs return 2.
ragwrench sync apply candidate-plan.json --review review.json --qdrant-path .qdrant
# Exit 2 (review blocked), including with --dry-run.
```

An intentional exception must name each violated gate, its reviewer and reason:

```sh
ragwrench sync review candidate-plan.json --max-deletion-fraction 0.5 \
  --override-gate max_deletion_fraction --reviewer alice \
  --reason 'Retire three obsolete chunks; change ticket DOC-42' -o approved.json
# Exit 0 (allowed).
ragwrench sync apply candidate-plan.json --review approved.json --qdrant-path .qdrant --verify
```

For a model migration, use `--override-gate embed_model_changed`; if both gates
are violated, repeat `--override-gate` for both. Unknown, duplicate or unviolated
override gates, and blank reviewer/reason values, are rejected. The same policy
and override flags can be passed directly to `apply` instead of using a receipt.
`--review` cannot be combined with those flags. In Python, pass
`review=ragwrench.sync.review.review_plan(plan, ...)` to `apply_plan`.

Receipts bind the exact plan hash, threshold, observations, decisions, and
caller-stated override provenance in a deterministic SHA-256 fingerprint. Apply
recomputes them and refuses changed receipts or receipts from other plans. The
admitted receipt is returned in apply JSON and persisted as collection metadata
`chunksync.last_review`, retained after commit. New journals bind its fingerprint:
resume with the same receipt or identical flags, reviewer and reason. A different
policy/provenance is refused before further embedding or mutation. Replaying a
completed plan returns its original stored review; older committed collections
may return `review: null` until a new transaction is admitted.

This is a local audit record, with no identity authentication or signature.
Collection metadata retains only the latest admitted review; archive receipt
files for longer history. Round-2 pending journals without a review binding still resume
under current admission gates, but cannot prove their original review policy.
No gate overrides store integrity checks or permits an in-place dimension change.

`python examples/sync/review_demo.py` runs this synthetic example end to end:
two extra deletions versus the baseline, blocked review/apply exits `3`/`2`, an
explicit persisted override, one independently verified final point, and an
idempotent replay. Its temporary store is removed on exit.

## Persistent collection review policy

The first admitted transaction persists its review limit in
`chunksync.review_policy`. Each later transaction can keep or tighten that limit.
Omitting `--max-deletion-fraction` still requests the compatibility default
`1.0`, so it **refuses** on a collection whose stored limit is stricter. Pass the
stored limit explicitly, or supply a receipt with that limit. Refusal happens
before embeddings, point writes, or management-state changes, after opening the
store to read its policy. It applies even when the new plan deletes nothing.

A plan's `--override-gate max_deletion_fraction` approves that plan's deletions
while retaining its strict policy. Raising the ongoing collection limit requires
a **separate policy override record**, with independent reviewer/reason:

```sh
# Collection kb already has limit 0.5. Candidate deletes 3 of 4 chunks.
ragwrench sync apply candidate-plan.json --qdrant-path .qdrant \
  --max-deletion-fraction 0.75
# Exit 2: the plan passes at 0.75, but the collection policy cannot silently weaken.
ragwrench sync policy-override candidate-plan.json --collection kb \
  --from-max-deletion-fraction 0.5 --max-deletion-fraction 0.75 \
  --reviewer policy-owner --reason 'DOC-42 approves a higher ongoing limit' \
  -o policy-override.json
ragwrench sync apply candidate-plan.json --qdrant-path .qdrant \
  --max-deletion-fraction 0.75 --policy-override policy-override.json --verify
```

`policy-override` writes deterministic version-1 JSON and returns 0, or 2 for
invalid input. It never opens Qdrant. Its fingerprint binds the exact plan,
collection name, old/new limits, reviewer and reason. Limits must be finite
numbers in `[0,1]`, the new limit must be higher, and provenance must be nonblank
(reviewer at most 256 characters, reason at most 4096). Apply rejects tampering,
wrong plan/collection/limits, stale source policy, and unnecessary exceptions.
Records are collection-name-bound, not store-path-bound; archive them with their
store context. A policy exception never overrides plan gates or integrity checks.

Apply accepts `--policy-override` alongside either direct review flags or
`--review`. In Python, use `ragwrench.sync.policy.policy_override(plan, "kb",
from_max_deletion_fraction=0.5, max_deletion_fraction=0.75,
reviewer="policy-owner", reason="DOC-42")`, then pass the record as
`policy_override=record` to `apply_plan`, alongside the requested review receipt.

The admitted policy and latest `chunksync.last_policy_override` persist with the
initial journal **before embedding or point writes**, including across failures.
New journals bind the separate override fingerprint (or its absence); resume with
the same review and override file. Completed replay returns `already_applied`,
the original review, effective `review_policy`, and latest `policy_override`.
Replay enforces the stored limit but does not change it, even if a tighter limit
was supplied; admit a new transaction to tighten policy. It cannot record a new
exception. An identical recorded exception can be replayed. Later transactions
retain the latest policy exception for audit, even when they tighten the limit.

Existing collections without `review_policy` inherit their latest review's limit,
or `1.0` if no review exists. Invalid stored policies fail closed. Existing pending
journals resume with their old binding format but cannot admit a new policy
exception. `review` and `--dry-run` remain offline: they validate supplied records
but **do not check collection policy** (`collection_checked: false`).

`python examples/sync/policy_demo.py` demonstrates persistent `0.5` admission,
refusals after reopen, an explicit `0.5 → 0.75` policy exception, one independently
verified final point, policy inspection before/after the exception, missing-collection
refusal, and replay using the installed console script. It prints each command,
exit code and captured output, and checks healthy store bytes stay unchanged by
inspection. All data is
synthetic and its temporary store is removed. Policy provenance is caller-stated
and unsigned; these checks do not authenticate an operator or protect metadata
from someone who can rewrite the store. Only the latest policy exception is kept;
archive records for history.

### Inspect policy before applying

```sh
ragwrench sync inspect-policy --qdrant-path .qdrant --collection kb
```

Exit 0 prints exactly this version-1 JSON shape (the collection defaults to `kb`):

```json
{"kind":"chunksync.policy_inspection","version":1,"collection":"kb","max_deletion_fraction":0.5,"policy_source":"explicit","pending_plan_fingerprint":null,"policy_override":null}
```

`policy_source` is `explicit` when `review_policy` is present, `legacy-review`
when its absence falls back to a non-null latest review, or `default` for `1.0`
when neither exists. A malformed explicit policy is an error, never a reason to
fall back. Inspection does not adopt a legacy policy or change the effective limit.
`pending_plan_fingerprint` reports a pending journal's plan hash or an old string
marker, without resuming it. `policy_override` contains the **full latest recorded
exception**, including its `provenance` reviewer/reason, plan/from/to limits and
`fingerprint`, or null when absent. It remains visible if a later transaction
tightened policy, so its target limit need not equal the current effective limit.

Exit 2 prints a diagnostic to stderr and no success JSON for a missing store or
collection, unmanaged/unsupported state, malformed policy, pending fingerprint
or latest exception. The path must be an existing persistent Qdrant directory;
`:memory:` and empty directories are refused without creating a store. Inspection
checks the reported fields and exception structure/hash, not the full pending
journal, historical exception plan, point integrity, or review authenticity.
Provenance remains caller-stated and unsigned.

To refuse opening-time metadata repair explicitly:

```sh
ragwrench sync inspect-policy --qdrant-path .qdrant --collection kb --no-repair
```

Healthy stores return the same JSON and exit 0, including legacy policy fallback
without adoption. If `meta.json` is missing, malformed JSON, or invalid UTF-8,
`--no-repair` returns exit 2, prints no success JSON, and emits:

```text
chunksync: primary metadata repair required; reopen without --no-repair to attempt validated recovery
```

The refusal leaves primary and sidecar bytes (or absence) unchanged, before
opening Qdrant or reading the sidecar. It does not promise that recovery is
possible: the sidecar might be missing or invalid. The existing-store check runs
first; if both primary and sidecar are absent, the usual existing-persistent-store
diagnostic applies. Semantic/configuration errors, oversized input, and I/O
errors retain their normal refusals. A healthy primary remains authoritative
even with a broken or missing sidecar; all store preflight and policy/record
checks still apply.

Inspection performs no embedding, point writes or management-state updates.
**Default inspection still repairs** missing/undecodable `meta.json` from a
validated recovery sidecar, even if the requested collection/state later fails
inspection. Both modes retain normal local-client preflight and lock handling:
a healthy open may create/use a lock file, so `--no-repair` is not a globally
write-free filesystem open. There is still one writer and no concurrent filesystem
replacement. See the recovery contract below for validation and durability limits.
Python callers can use `local_client(path, repair_metadata=False)` before
`ragwrench.sync.store.inspect_policy(client, "kb")`; this option controls opening-time
repair only, not subsequent mutations through the client. Already-open clients
cannot undo prior repairs.

`python examples/sync/inspection_demo.py` prints exact console-script commands,
exit codes, stdout and stderr. It checks healthy no-repair output, unchanged files
on missing/truncated/invalid-UTF-8 primary refusal, default validated recovery,
and bad-sidecar refusal using temporary synthetic stores. It also runs with an
installed wheel.

### Inventory policies across a store

```sh
ragwrench sync list-policies --qdrant-path .qdrant
ragwrench sync list-policies --qdrant-path .qdrant --no-repair
ragwrench sync list-policies --qdrant-path .qdrant --output reports/policies.json
ragwrench sync list-policies --qdrant-path .qdrant --fail-on-diagnostics --output reports/policies.json
python examples/sync/inventory_demo.py
```

`list-policies` requires an existing persistent catalog and no collection name.
It reads only `meta.json` or recovery catalog JSON: **no Qdrant client, SQLite,
point blobs, document payloads, vectors, or embeddings are opened**. Python:
`ragwrench.sync.inventory.list_policies(path, repair_metadata=True)`.

Success returns one JSON object with `kind: "chunksync.policy_inventory"`,
`version: 1`, `catalog_source: "primary"` or `"recovery"`, and `collections`, sorted
by collection name (Unicode code point order; supported names are ASCII). Each row has exactly `collection`, `management_state`,
`status`, `policy`, and `diagnostic`. A valid managed row looks like:

```json
{"collection":"alpha","management_state":"managed","status":"ok","policy":{"kind":"chunksync.policy_inspection","version":1,"collection":"alpha","max_deletion_fraction":0.25,"policy_source":"explicit","pending_plan_fingerprint":null,"policy_override":null},"diagnostic":null}
```

The policy object uses the same validators, precedence, pending fingerprint and
historical override fields as `inspect-policy`. `policy_source: "legacy-review"`
or `"default"` describes an effective compatibility limit; it does **not** assert
an explicitly persisted policy and never adopts one. Unmanaged collections never
receive a default policy in the report.

| Catalog entry | management_state | status | diagnostic.code |
|---|---|---|---|
| Valid version-1 policy fields | managed | ok | none |
| Missing/null management state | unmanaged | unmanaged | unmanaged_collection |
| Other management shape/version | unsupported | unsupported | unsupported_management_state |
| Invalid version-1 policy/pending/override | managed | invalid | invalid_policy_metadata |
| Invalid name or unsupported/invalid collection configuration | unknown | unavailable | unavailable_collection_metadata |

Every non-`ok` row has `policy: null` and a `{code, message}` diagnostic. One bad
collection does not hide other rows. **Default exit 0 includes per-collection diagnostics**;
consumers must inspect row statuses. An initialized empty catalog returns `[]`.
There is no directory scan for collections omitted from the catalog.

For automation, opt into `--fail-on-diagnostics`: a complete inventory returns
**exit 3 if any row has a non-`ok` status**, including `unmanaged`, `unsupported`,
`invalid` or `unavailable`. All-ok and empty catalogs return 0. Exit 3 emits the
same complete JSON and no stderr; it also publishes the same exact optional
archive. No rows or fields are removed or added by this flag. Recovered catalogs
use the same rule. Treat 3 explicitly in shells using `set -e` if the report must
be processed after diagnostics are found.

Optional `--output REPORT.json` (or `-o`) archives the **exact UTF-8 stdout JSON,
including its final newline**, after the complete inventory succeeds. Reports
with per-collection diagnostics are also successful archives, including exit 3
with `--fail-on-diagnostics`. The destination
must be outside the store, with an existing parent directory and either an
existing regular file or an absent filename. No directories are created. Store
destinations (including catalog, sidecar, SQLite and lock files), symlink targets,
and any symlink traversal in output ancestors are refused before inventory reads.
This also rejects external and dangling symlinks; use the direct external path.

Publication writes a private temporary file in the output's parent directory,
flushes/fsyncs/closes it, then atomically replaces the destination. Existing report
permissions are replaced by the private temporary file's permissions. Catalog,
size, path, temporary-file write/sync/close and rename refusals return exit 2,
stderr only, preserving the previous report (or its absence) and store state.
Handled failures clean up the temporary file when filesystem permissions allow.
This is atomic visibility on a local filesystem, **not a power-loss durability
guarantee**: no directory fsync follows replacement. Process termination can leave
a temporary file; a stdout failure after replacement cannot undo the archive.
Archive publication and stdout delivery are not a two-resource transaction.

At most **1000 catalog entries**, **64 MiB per input JSON**, and **64 MiB UTF-8
output including its newline** are accepted. There is no silent truncation,
pagination or partial stdout. Missing/nonpersistent paths, symlinked catalog
paths/ancestors, malformed catalog envelopes/aliases, I/O errors and limit errors
return exit 2 with stderr only. Invalid primary JSON constants and semantic/size/I/O
errors never trigger fallback. A healthy primary is authoritative, including when
the sidecar is absent or malformed.
Global errors, including archive publication failures, remain exit 2 even with
`--fail-on-diagnostics`; a diagnostic row encountered before a later global
failure cannot turn an incomplete or unpublished report into exit 3. Such errors
emit no report and preserve the previous archive or its absence.

For missing/undecodable primary metadata, default mode can read the existing
schema/checksum-validated recovery catalog **in memory** and reports
`catalog_source: "recovery"`. Configuration and policy validation still produce
per-collection diagnostics. `--no-repair` refuses before reading the sidecar with
the established repair-required diagnostic. Inventory **never repairs disk files**:
actual repair through `local_client`/`inspect-policy` still requires full point
preflight. A recovered inventory is no assertion that the points are safe or the
store can be opened. Checksums and provenance remain unauthenticated.

Use an **offline store with no concurrent writer or filesystem replacement**.
Inventory acquires no backend lock and creates no lock file; it cannot coordinate
with Qdrant writers. Existing `inspect-policy`/`local_client` retain their normal
preflight, lock creation/use and default disk-repair caveats. Inventory validates
reported metadata only, not point integrity or full journal progress. The installed
synthetic example prints exact commands/exits/stdout/stderr, checks two managed
collections and one unmanaged, isolated invalid metadata, legacy state and
in-memory recovery, and asserts store-file invariance.
It also compares default/opt-in output, archives complete exit-3 reports for
each diagnostic status, and checks healthy/empty exit 0 and global-error exit 2.

### Compare archived policies offline

```sh
ragwrench sync compare-policies before.json after.json
ragwrench sync compare-policies before.json after.json --output reports/diff.json
ragwrench sync compare-policies before.json after.json --fail-on-diagnostics --output reports/diff.json
ragwrench sync compare-policies before.json after.json --fail-on-policy-weakening --fail-on-diagnostics --output reports/diff.json
python examples/sync/policy_comparison_demo.py
python examples/sync/policy_weakening_demo.py
```

`compare-policies` reads only the two archived `chunksync.policy_inventory`
version-1 JSON files; neither source store needs to exist. It opens no Qdrant
client, catalog, SQLite file or points, performs no embedding, and creates no
locks. Without `--output` it writes no files. Both complete reports must validate
before any stdout is emitted, including all healthy and diagnostic rows from primary or recovered inventories.

The newline-terminated UTF-8 result has `kind: "chunksync.policy_comparison"`,
`version: 1`, `identical`, `catalog_source: {"before": ..., "after": ...}`,
and `added`, `removed`, `changed` lists. Added/removed entries are full inventory
rows. Each changed entry is `{"collection": NAME, "before": FULL_ROW,
"after": FULL_ROW}`. All three lists sort by exact collection name; JSON object
keys sort too. Collection ordering and object-key ordering do not affect equality.
Numeric limits such as `1` and `1.0` compare equally and policy numbers are
normalized through the shared inspection validators. Every row field counts,
including effective limit, policy source, pending fingerprint, retained exception
provenance and diagnostic message. A `catalog_source` change alone appears as
context and keeps `identical: true`.

Exit **0** means equal collection rows, **10** means added/removed/changed rows,
and **2** means invalid input, destination, oversized result or archive publication
failure, with stderr and no report stdout. Diagnostic rows are valid comparison
data; their presence alone is not an error or drift.

Opt-in **`--fail-on-diagnostics`** returns **3** if either complete validated input
has any non-ok row (`unmanaged`, `unsupported`, `invalid`, or `unavailable`). This
includes unchanged diagnostics omitted from the diff, removed rows present only
in BEFORE, and added rows present only in AFTER. Exit 3 takes priority over 0/10,
but leaves the full comparison JSON and optional exact archive byte-for-byte
unchanged; it adds no stderr. Empty/all-ok inputs retain 0/10, and a catalog-source
change alone still counts as equal. Both archives and the entire bounded result
must validate and any requested archive must publish before exit 3 is selected.
Global input, destination, size and publication failures retain **exit 2**, no
report stdout, and preservation of both inputs and the previous archive or absence.
Python `compare_policies` continues returning the same report; this option controls
only CLI exit selection.

Opt-in **`--fail-on-policy-weakening`** returns **4** when an exact collection
name is healthy (`status: "ok"`) on both sides and its normalized effective
`max_deletion_fraction` strictly increases. For example, `0.25` to `0.5` triggers
the gate; tightening and equal values (including `1` versus `1.0`) do not.
The effective number is compared regardless of `explicit`, `legacy-review` or
`default` policy source. A source-only change can still be ordinary drift, but
does not trigger this gate. Added/removed/renamed collections and any pair with
a non-ok row provide no weakening inference. A retained historical override
does **not** waive this offline gate; it does not authorize policy writes.

The flag leaves the full JSON/stdout/exact archive unchanged, with no extra
stderr. Both complete bounded inputs and the entire result must validate, and
any requested archive must publish, before gate exit selection. With both flags,
priority is **2 global error → 3 requested diagnostics → 4 requested weakening →
0 equal / 10 drift**. Diagnostics count anywhere in either full input, even if
unchanged or removed; weakening requires a matched healthy pair. The Python
comparison API and default CLI exits remain unchanged.

Validation requires exact report/row/inspection/diagnostic
fields, status/management/code consistency, valid policy numbers, matching
inspection identity, pending SHA-256 format and the shared stored-override
shape/hash checks. Default policy source requires limit `1`. Unavailable rows
may retain invalid collection names that inventory reports as diagnostics; other
rows require supported collection names. Collection identities must be unique.
Duplicate JSON object keys, unsupported versions, malformed/invalid UTF-8 JSON,
nonfinite numbers, missing/nonregular/unreadable inputs and excessive nesting
are refused. Each input and the complete serialized result are bounded to
64 MiB, including the output newline; each report allows at most 1000 rows.
Python `ragwrench.sync.policy_comparison.compare_policies(before, after)` validates
the same schemas, row counts and serialized byte bounds.

Optional `--output` / `-o` publishes the exact complete stdout JSON as a UTF-8
archive, including its final newline, at exits 0, 10 and opt-in 3 or 4. The parent
directory must already exist. The destination must be distinct from **both input
archives**: direct/resolved paths and existing hardlink aliases are refused,
including aliases hidden by input symlinks. All destination symlink traversal
(including ancestors, dangling/external links and symlink/`..` paths), nonregular
targets and missing/non-directory parents are refused before input reads.
Separate output files may share the inputs' directory. No directories are created.

Both full inputs and the full bounded result validate before any temporary file
is created. Publication reuses the inventory archive writer: same-directory private
temp, write, flush, file fsync, close, then atomic replacement. Input, destination,
size and pre-publication failures leave both inputs and the previous diff archive
(or its absence) intact, with exit 2 and no report stdout. Temporary-file cleanup
on handled failures is best effort. This guarantees **atomic visibility**, not
power-loss durability: no directory fsync follows replacement. Replacement uses
private temporary-file permissions instead of preserving old permissions/ownership;
process termination may leave a temporary file. Stdout is emitted after publication
and is not transactional with the archive: a later stdout/encoding/delivery failure
cannot undo replacement. Keep input files and filesystem paths stable throughout;
there is no concurrent-replacement coordination.

This compares reported metadata, without authenticating archives or provenance,
checking point integrity, or establishing that both files refer to the same
store. Exact names are identities: renames appear as removal plus addition.
A retained override can predate a tighter limit or a different pending plan.
Archives need not be healthy to compare equally. Keep the input files stable
while reading; comparison provides no coordination with concurrent replacements.
The installed example checks stdout-only compatibility, absent/existing exact diff archives at
exit 0/10, default versus opt-in byte equality, every diagnostic status on either
side at exit 3, input/count/size failure priority, and both-input path/hardlink
refusals without creating a source store.
The separate installed weakening example checks strict numeric/source boundaries,
retained overrides, uncomparable rows, default/weakening/both byte equality,
exit priority 2/3/4, and unchanged inputs using synthetic archives only.

## Inputs and identity

Directory inputs must contain only real UTF-8 `.md` and `.txt` files. Document IDs
are relative POSIX paths. Symlinks and unsupported formats are rejected, rather
than followed or silently skipped. Empty files contribute no chunks. No PDF,
HTML, Office, or Markdown syntax parsing is performed.

For an existing chunking pipeline, supply JSONL instead:

```jsonl
{"doc_id":"guide","text":"First paragraph.","metadata":{"owner":"team-a"},"embed_metadata":{"title":"Orchard"}}
{"doc_id":"guide","text":"Second paragraph.","metadata":{"owner":"team-a"},"embed_metadata":{"title":"Orchard"}}
```

```sh
ragwrench sync snapshot --chunks chunks.jsonl --embed-model my-model-v1 -o new.json
```

Only `doc_id`, `text`, `metadata`, and `embed_metadata` are accepted. Text is
required and nonempty after normalization; both metadata fields are JSON objects.
The order of rows within each document defines its chunk positions. `metadata`
is stored payload only; `embed_metadata` affects embeddings. Directory snapshots
use empty metadata objects; use JSONL when you need metadata.

Normalization applies Unicode NFC, LF line endings, trailing-space removal per
line, and outer whitespace stripping. Internal spaces remain significant.
Content hashes are SHA-256 of canonical JSON containing normalized text,
`embed_metadata`, and the **embed_model tag**. IDs are UUIDv5 over canonical
`[doc_id, content_hash, occurrence]` under the manifest namespace. Duplicate text
within a document gets occurrence numbers 0, 1, ... and distinct Qdrant-valid IDs.
Document ID separators cannot cause ambiguous concatenations.

Changing the model tag invalidates every surviving content hash and is flagged
in the plan. Change it whenever embedding weights, preprocessing, or behavior
changes; the tool cannot discover changes inside an external program or model.

`--chunker fixed` splits by character count (`--target-size`). `--chunker cdc`
uses deterministic anchors from a sliding 64-character window, preferentially at
paragraph/sentence boundaries and then word boundaries. `--min-size` (128),
`--target-size` (256), and `--max-size` (512) are Unicode character counts and must
satisfy `1 <= min <= target <= max`. CDC has a hard maximum (including long
unbroken strings), with a possible short final chunk. Its anchors are independent
of absolute offsets; min/max constraints can still cause cascades on some inputs.
This is an optimization, not a guarantee that every insertion affects one chunk.

## Plan operations and collection safety

- `upsert`: a new chunk without a reusable retired source, requiring a vector.
- `delete`: old content absent from the new snapshot.
- `unchanged`: same ID and identical stored payload.
- `payload_only`: same ID/content with changed metadata; no embedding.
- `moved`: same content at a changed position or renamed document. The old
  vector is reused. Renames match removed/new hashes one-to-one in deterministic
  order, including repeated text. New UUIDs are written before old UUIDs retire.

Categories are exclusive for each new chunk. Deletes count unmatched old chunks;
old IDs retired by a move are represented by the move, not counted again as
`delete`. A pure position change is a move; a metadata and position change on the
same ID is `payload_only`. Embeddings avoided is `new_chunks - upsert`.

Collections use one dense **DOT** vector and retain their management state in
Qdrant collection metadata; there is no extra control point in search results.
Before opening a persistent store, the adapter validates all collection names,
rejects symlinks and aliases, and preflights its SQLite point blobs with a
restricted unpickler. Qdrant's own pickle loader then reads the validated format.
Executable pickle opcodes/globals and legacy dbm migration are refused. This
preflight assumes no concurrent filesystem replacement; arbitrary third-party
Qdrant serialization is unsupported.

The state records the snapshot fingerprint, embedder identity, a float32-normalized
point/vector fingerprint, and an apply journal while a plan is unfinished. A fresh nonempty
unmanaged collection is refused. Existing dimension/distance or embedder identity
mismatches are refused. An out-of-date plan or external changes to a committed
collection's IDs, payloads, or vectors are refused before embedding.

All new vectors, reused vectors, and payload writes precede deletes. If an
embedder or vector-store call fails, rerun the **same plan, store, collection and
embedder arguments**. After a persistent metadata-save failure, further
`apply_plan` calls on that client are refused until you close it and open a new
client. The journal in Qdrant collection metadata binds the plan
SHA-256, both source snapshot hashes (including their text/content hashes), the
committed source point/vector fingerprint, and the embedder identity. It records
a completed operation prefix, a checkpoint of point/payload/vector hashes, and
one write-ahead intent with the affected point's before/after hashes. Deletes
retire one ID per call so progress can be resumed within the retirement batch.

On resume, a call that persisted and then raised is recognized by its intended
after hash and skipped, without repeating its upsert, payload write, or delete.
Completed upserts do not call the embedder again. A call that failed before
persisting is retried; its embedding may be recomputed and must match the
recorded intent. Changed journal bindings, unrelated missing/changed points, or
changed vectors are refused before further point writes. Other plans are refused
while one is pending. Apply is idempotent after completion; there is no automatic
rollback or partial-transaction cancellation.

`python examples/sync/journal_demo.py` reproduces a deterministic store failure:
`old1, old2` are replaced by `new1, new2`, and the first delete persists then
raises. Three points remain at failure. After close/reopen, recovery skips both
upserts and that delete, deletes only the remaining old point, and independently
verifies two final vectors through the CLI. Expected resume: `embedded: 0`,
`points: 2`; each retired ID is successfully deleted exactly once.

Committed version-1 collections remain compatible. An old string-only pending
marker is upgraded on resume using its original admission checks; earlier vector
history cannot be reconstructed, so journal integrity starts at that upgrade.
The extra metadata writes and per-ID deletes add overhead; no throughput
improvement is claimed.

### Recovering interrupted metadata writes

The CLI and Python `local_client(path)` protect persistent stores with
`chunksync-recovery.json` beside Qdrant's `meta.json`. Before each Qdrant metadata
save, recovery opens the destination directory, writes a temporary file in that
directory, flushes its Python buffer, calls `fsync` on the file, closes it,
atomically renames it over the sidecar, and calls `fsync` on the directory before
closing the directory descriptor. Only then does Qdrant's metadata writer start.
The sidecar contains the full intended collection catalog, management state,
apply journals, and review receipts, with a SHA-256 checksum. Point payloads and
vectors remain in SQLite. An error at any of these steps propagates and stops
the apply call before any subsequent point mutation or Qdrant metadata save.
Before rename, the prior sidecar and primary metadata stay unchanged; temporary
files are cleaned on handled failures. After rename, the complete new sidecar
may already be visible even if directory sync fails: this is an error, not an
acknowledgement of durability, and the primary metadata writer still does not
start. The first save error propagates unchanged. **Close/reopen after any save
error**: subsequent `apply_plan` calls on that client raise `SyncError` with
`persistent metadata save failed; close and reopen the client before applying again`.
This check runs before embedding, collection creation, point writes, or
management-state updates, including attempts with another plan/collection or a
completed replay. It also covers failed collection-creation saves and interrupted
saves. Reads (including policy inspection) and close remain usable, but reads can
show unsaved live metadata. There is no live-state rollback.

The fence is local to that persistent client; reopened clients, other clients,
and in-memory clients are unaffected. Validation/review refusals, embedding
errors, and point-call failures do not fence a client unless a persistent metadata
save actually failed. Direct low-level Qdrant mutations are outside this apply
guard. A write that already succeeded before its checkpoint failed is recognized
by the journal on reopen.

On reopen, missing, malformed JSON, or invalid UTF-8 `meta.json` is repaired
atomically from the sidecar with the same file/directory sync sequence **after**
checking its schema/checksum, collection
names/configurations, and SQLite point blobs. Resume the same plan and review:
the restored journal still checks bindings and point/vector hashes before further
point writes or embeddings. A persisted upsert, payload update, or delete is
recognized without repeating it. A final commit recovered from the sidecar
returns `already_applied`.

Valid primary JSON stays authoritative, even if the sidecar is stale or broken.
Semantic errors such as aliases or unsupported configurations, oversized files,
and filesystem read errors are refused rather than replaced from a backup. A
missing/damaged sidecar cannot repair a broken primary file. Existing stores
without a sidecar remain readable and gain one on their next metadata write;
clients opened directly with `QdrantClient` do not get this protection. Even
`verify` or default `inspect-policy` can repair metadata when opening a damaged
store; `inspect-policy --no-repair` refuses that repair, and `apply --dry-run` never
opens or repairs it.

`python examples/sync/recovery_demo.py` runs two synthetic failures at the
second upsert's checkpoint for `old1, old2 → new1, new2`: a torn real Qdrant
metadata write (left torn in `meta.json`, as after a crash), and an error after a real directory `fsync` but before Qdrant's writer.
Four points remain at each failure. An accidental same-client retry is refused
with zero embeddings or point/management writes. Reopening resumes with `embedded: 0`,
verifies two final vectors, and proves each of the two upserts and two deletes
succeeded exactly once. It also exercises recovery through CLI verification and idempotent replay. The
example prints each CLI command, exit code and captured output, works with an
installed wheel and removes its temporary store on exit.

Recovery persistence is supported on **local Linux filesystems implementing
file and directory `fsync`**. Unsupported platforms refuse recovery writes;
directory-open and sync errors are never silently ignored. Ordinary exported
snapshot/plan/review JSON retains portable atomic replacement without fsync;
in-memory clients are unaffected. The recovery tests observe real syscall
ordering and flushed bytes, inject errors before/after file sync, rename and
directory sync at upsert/delete intent/checkpoint and commit, then reopen using
both valid primary metadata and a torn primary requiring sidecar recovery.

This adapter relies on qdrant-client local-mode storage internals (supported:
qdrant-client 1.20 or later below 2; it was first written against 1.16.2).
qdrant-client 1.16 rewrote `meta.json` in place, while 1.20+ writes
`meta.json.tmp` and renames it over `meta.json`. Neither version fsyncs that
write, so the durable sidecar is still required: with 1.20+, an interrupted
save normally leaves the previous complete `meta.json` (valid, so it stays
authoritative and the journal recognizes already-persisted writes), and a
crash after an unsynced rename can leave a missing or torn primary that is
repaired from the sidecar. It requires one
writer and no concurrent filesystem replacement. These exception tests **do not
prove physical power-loss safety**: no power cut, kernel crash, disk/controller
cache test, or filesystem certification is performed. The adapter does not add
fsync to Qdrant's primary writer, SQLite, or the parent of a newly created store
directory; network filesystems and Windows/macOS are outside this contract.
SQLite corruption, arbitrary metadata corruption, and authenticated backups are
outside its scope. Keep the catalog,
sidecar, and collection databases together when copying a store. The checksum
detects accidental changes, not an actor able to rewrite both content and checksum.

`verify` checks exact ID sets and complete payloads, verifies the committed
collection fingerprint, and independently recomputes all vectors from the new
snapshot. Vector comparisons allow `1e-6` absolute/relative float32 tolerance.
It intentionally calls the embedder for every point. Precomputed mode compares
to the supplied authoritative vectors instead of computing an ML embedding.

## Embedding options

The default `--embedder hash --dimension 32` is a deterministic hash-based test
embedding. It has no semantic retrieval quality. Dimension is limited to 1–4096.

For an external embedder, pass an explicit **JSON argv array**:

```sh
ragwrench sync apply bootstrap.json --qdrant-path .qdrant-command --collection kb \
  --embedder command --dimension 8 \
  --embed-command '[".venv/bin/python", "examples/sync/embed_command.py", "--dimension", "8"]'
```

One process is started per needed vector, without a shell. It receives exactly
one JSONL row with `id`, `text`, `embed_metadata`, and `embed_model`. It must return
exactly one JSONL row `{"id":"<matching UUID>","vector":[0.1,0.2,...]}`. Vectors
must have the declared dimension and finite numbers of magnitude at most `1e30`.
The default per-call timeout is 30 seconds (`--embed-timeout`); stdout is limited
to a 1 MiB read per chunk. Stderr is not echoed because it can contain credentials.
The command is explicitly trusted executable code: it can perform its own side
effects and write output until it exits or times out. Use a resource-constrained
wrapper for untrusted programs. Plans and input documents cannot specify commands.

An embedder must be a deterministic function of **text, embed_metadata, and
embed_model only**. The ID is a correlation token, not an embedding input. Otherwise
vector reuse across moves is invalid and verify will detect the difference.

For vectors computed elsewhere:

```sh
ragwrench sync apply bootstrap.json --qdrant-path .qdrant-vectors \
  --embedder precomputed --dimension 2 --vectors vectors.jsonl
```

`vectors.jsonl` uses the same `id`/`vector` response format, with one row per
required upsert ID. Missing or duplicate IDs and dimension mismatches fail.
To verify, include all new snapshot IDs, even IDs whose vectors were reused.
A collection's embedder mode, dimension, and command argv are fixed by its first
apply; switching mode or dimension requires a new collection. Precomputed file
contents are not part of the embedder identity so a new file can serve each plan.
Command argv is fingerprinted, not stored in collection metadata.

## Reproduce the optimization experiment

```sh
ragwrench sync bench --seed 25 --paragraphs 60 -o bench-results.json
```

Three synthetic Markdown-like documents receive five independent edits. Full
reindex and document-hash strategies use CDC chunks; fixed uses its own chunk
count. Counts come from **actual HashEmbedder calls**, excluding initial setup
and independent verification. Both chunk strategies apply to fresh in-memory
Qdrant collections and verify every resulting point against a from-scratch
index. The document-hash baseline also compares all final vectors. No timing or
ML-quality claim is made. The JSON output includes each strategy's own denominator,
verified point count, seed, parameters and corpus fingerprint. Output is fully
deterministic. A saved reference run is in [examples/sync/bench-results.json](../examples/sync/bench-results.json).

For seed 25 / 60 paragraphs, the measured vector counts are:

| Edit | Full reindex | Document hash | Chunk fixed | Chunk CDC |
|---|---:|---:|---:|---:|
| Insert at top | 172 | 56 | 70 | 3 |
| Modify paragraph | 171 | 55 | 34 | 3 |
| Delete document | 116 | 0 | 0 | 0 |
| Reorder paragraphs | 171 | 55 | 15 | 6 |
| Rename document | 171 | 55 | 0 | 0 |

These results describe this synthetic corpus; they are not a universal reduction
ratio. Deleting a whole document requires no new vectors for incremental methods.

## Scope and provenance

This first version supports local, small, single-writer collections only. It
loads snapshots and point sets in memory. Input/output JSON and aggregate source
documents are limited to 64 MiB; snapshots are limited to 100,000 chunks. Plans
include duplicated snapshot/operation data and may reach the JSON limit sooner.
Split larger datasets into independent collections. Float32 is the fingerprint
precision; changes below that precision are not distinguished. New pending
transactions check journaled point/vector fingerprints; legacy pending upgrades
cannot authenticate vector history from before the upgrade. Missing or undecodable
metadata can recover from a valid sidecar; SQLite corruption and power-loss
durability remain outside scope.
Do not edit a collection during a pending transaction. No server administration,
multi-writer coordination, model inference, semantic deduplication, automatic
model migration, document parsing, or rollback is included. Reuse matches existing
IDs and retired source chunks one-to-one; new duplicates beyond those sources
can require identical content to be embedded again.

Original implementation and fixtures are MIT licensed. Qdrant-client is the Apache-2.0
runtime dependency; Qdrant supplies the target data model. LlamaIndex and LangChain
inform the conceptual document-hash baseline; none of their source code was copied or
imported. See [THIRD_PARTY.md](../THIRD_PARTY.md).
