# ragwrench port

Moving a knowledge base between Dify and RAGFlow, or between two instances of one, usually
means re-uploading files and hoping the chunks come out the same. `ragwrench port`
exports chunk text through each platform's public API into a hash-checked bundle, imports
it into the target with resumable progress and verifies the result. Use it for
migrations, for backups of curated chunks, or to copy a knowledge base between environments.

_Formerly the standalone `kb-bundle-porter` tool (`kbport`)._

Move knowledge-base **document/chunk text** between self-hosted Dify and RAGFlow through their public HTTP APIs. Export an inspectable, versioned JSONL bundle, validate its SHA-256 hashes offline, plan an import, resume interrupted work, and verify the target's content and supported fields. The target builds its own indexes; vectors and original binary files are not transferred. Apps/workflows, database migrations and continuous synchronization are outside scope.

Linux (uses `flock` for local writer locks). Original implementation, MIT licensed. No upstream source code is embedded. API compatibility is scoped to the contracts recorded in [THIRD_PARTY.md](../THIRD_PARTY.md), including the pinned RAGFlow **1.0** and **v0.27.3** dataset contracts. Other releases are not assumed compatible.

## Quick start

```bash
pip install ragwrench
ragwrench port --help
ragwrench port validate examples/port/bundle
ragwrench port import-status --help
ragwrench port capabilities ragflow
python examples/port/demo_roundtrip.py
python examples/port/demo_resume.py
python examples/port/demo_profiles.py
python examples/port/demo_capabilities.py
```

No models, paid APIs, production datasets or browser images are needed by the
tests or demos. The demo starts two temporary loopback HTTP servers and stops them on
exit.

## Export, plan, import, verify

Put the service key in an environment variable using your usual secret-management method. Default variables are `DIFY_KB_API_KEY` and `RAGFLOW_API_KEY`; `--key-env MY_VARIABLE` selects another variable. There is no literal-key CLI argument.

```bash
ragwrench port export dify --base-url http://localhost/v1 \
  --dataset-id 00000000-0000-0000-0000-000000000001 -o ./bundle

ragwrench port import ragflow ./bundle --base-url http://localhost:9380 \
  --dataset-name handbook --state .kbport/handbook.json --dry-run

ragwrench port import ragflow ./bundle --base-url http://localhost:9380 \
  --dataset-name handbook --state .kbport/handbook.json

ragwrench port verify ragflow ./bundle --base-url http://localhost:9380 \
  --state .kbport/handbook.json
```

Instance roots and API roots are accepted: Dify appends `/v1`, RAGFlow appends `/api/v1`. Reverse-proxy prefixes must end in the appropriate API suffix. URLs containing userinfo, query strings or fragments and all redirects are rejected. TLS verification stays enabled; environment proxies and `.netrc` are ignored.

For the reverse direction, use `export ragflow` and `import dify`. Same-system transfers use identical commands. Dify imports default to `--indexing high_quality`, which needs a configured embedding provider on the target. `--indexing economy` selects Dify keyword indexing. The CLI does not configure providers or transfer model credentials. Existing service-side ingestion/model usage remains the operator's responsibility.

Use **exactly one** of `--dataset-name NAME` to create a dataset or `--dataset-id ID` to reuse one. An existing dataset name causes an error, including case-insensitive duplicates; it is never silently reused. Reusing a dataset adds new documents and preserves its existing documents/settings. Source settings are descriptive bundle information, not automatically applied to the target. New RAGFlow datasets use the built-in `general` parser (`naive` in v0.27.3) and the target's configured default embedding model. New Dify datasets are private.

Output is one JSON object per successful command, for example:

```json
{"chunks":15,"documents":3,"valid":true}
```

Import includes `dataset_id`, state path, counts, skipped errors and warnings. Counts describe the bundle's planned/imported scope, not just the writes made in the current invocation. Verify returns `{"verified":true,...}` only after comparing target records.

## API profiles and read-only negotiation

`export`, `import` and `verify` accept `--api-profile`. Defaults remain `dify-f8fc371a` and `ragflow-1.0`. RAGFlow also supports `ragflow-0.27.3`, pinned to upstream commit `8090ffd8ba6098813d7f7ec2c4b8774f36c8423a`. The concrete dataset-create difference is:

| Profile | Dataset creation body, in addition to `name` |
| --- | --- |
| `ragflow-1.0` | `"parse_type": 1, "parser_id": "general"` |
| `ragflow-0.27.3` | `"chunk_method": "naive"` |

Both RAGFlow profiles use empty virtual documents, chunk `questions`/keywords and PATCH availability. This does not add support for older PUT-only APIs.

Use `--api-profile auto` for **RAGFlow** read-only discovery. It paginates dataset GETs and requires every listed dataset to expose one consistent signature: `chunk_method` selects the v0.27.3 contract, while `parser_id` selects 1.0. It identifies a compatible contract family, **not the exact installed release**. Empty listings, mixed signatures, records with both/neither field, and pipeline-only records without these fields fail with exit 1 before migration state or remote writes. Select an explicit profile from the operator-known server version in those cases. Explicit profiles also reject clear contradictory listing signatures before remote writes. No write is used as a probe or retried with alternate fields.

For example, with an existing v0.27.3 fixture dataset available for discovery:

```bash
ragwrench port import ragflow examples/port/bundle --base-url http://localhost:9380 \
  --api-profile auto --dataset-name handbook --state .kbport/handbook.json
```

Expected output includes `"api_profile":"ragflow-0.27.3"`, `"documents":3`, `"chunks":15`, `"skipped":0`. For a new empty instance, use `--api-profile ragflow-0.27.3`. Offline `--dry-run` also requires an explicit profile because it performs no GET discovery. Successful import/verify/export output reports the chosen profile. New migration state and export checkpoints bind that resolved profile; changing it on resume or verification fails. Older state/checkpoints without a profile retain their original default contract.

Field capabilities remain explicit: neither RAGFlow profile restores separate QA `answer` or chunk metadata; Dify does not restore embedding `questions`. Such inputs fail local preflight with `Unsupported fields` before discovery or target writes. `--allow-lossy` reports the acknowledged projections in `warnings`; it does not expand an API's capabilities. The synthetic native dataset fixtures and unsupported-field fixtures under `tests/port/fixtures` cover both profiles. `python examples/port/demo_profiles.py` exercises the installed console script against loopback servers: empty-discovery failure, both profile payloads, 15 verified chunks per import, zero repeat writes, and unsupported-field errors plus lossy warnings without HTTP calls during dry runs.

## Inspect capabilities before choosing a bundle

```bash
# Offline: no URL, bundle, state, credentials or HTTP client needed.
ragwrench port capabilities dify
ragwrench port capabilities ragflow --api-profile ragflow-0.27.3

# Read-only discovery: requires the RAGFlow key in the usual environment variable.
ragwrench port capabilities ragflow --api-profile auto --base-url http://localhost:9380
```

A successful command exits **0** and prints one JSON object with `system`, resolved `api_profile`, `bundle_format: "kbbundle/1"`, `scope: "import"`, `profile_resolution`, `fields`, `dataset_creation`, `limits` and `loss_boundaries`. `fields.documents` and `fields.chunks` cover every portable record field; each entry has an `import` status and a `detail` explanation. Statuses distinguish `preserved`, `conditional`, `requires_allow_lossy`, `remapped`, `renamed`, `text_model`, `not_restored` and `derived`. For example, the RAGFlow answer entry is:

```json
{"import":"requires_allow_lossy","detail":"Nonempty separate answers rejected by default; --allow-lossy replaces content with Question: <content>\\nAnswer: <answer> and removes the separate answer."}
```

Default/explicit selection is offline: `profile_resolution.method` is `default`/`explicit` and `server_checked` is false. Defaults are `dify-f8fc371a` and `ragflow-1.0`. Auto uses the existing paginated dataset-signature negotiation and reports `method: "dataset_signatures"`, `server_checked: true`. `exact_server_version_verified` is always false. This is a pinned **import contract** report; it does not validate a bundle, verify remote write permissions or model/provider readiness, or guarantee compatibility with a live release. Both RAGFlow profiles have the same field support; `dataset_creation` distinguishes their parser defaults.

Nonempty metadata, Dify embedding questions and RAGFlow separate QA answers require the listed `--allow-lossy` projections on import. Dify QA answers are conditional on `qa_model`; text documents with answers still fail even with `--allow-lossy`. IDs are remapped, names gain a recovery marker, positions are not restored as target ordinals, and source settings remain descriptive. The report also discloses binary/vector and document-state exclusions, the 16000-byte default chunk guard and duplicate-content restrictions. Run `import --dry-run` with a selected bundle for actual field/size preflight.

The command never loads or writes a bundle, migration state, checkpoint or lock. Auto makes only dataset GETs and never prints dataset records, URLs or keys. It supports `--key-env`, `--page-size`, `--retries`, `--backoff` and `--timeout` with the same bounds/defaults as migration GETs. These options and `--base-url` are rejected for offline profiles to avoid silently ignoring a requested server check. Auto without a URL/key, cross-system profiles, Dify auto, empty/ambiguous discovery, invalid connection limits and API failures exit **1**, with safe stderr and no JSON output or writes. Unknown profiles/systems/options and bundle/state arguments are syntax errors (exit **2**).

`python examples/port/demo_capabilities.py` exercises the installed CLI without a bundle: credential-free reports, both RAGFlow auto profiles over multiple loopback GET pages, unchanged server state/no local files, and negative selection/credential/discovery cases. No live service compatibility is claimed.

## Bundle format: `kbbundle/1`

The portable bundle consists of three UTF-8 files:

| File | Contents |
| --- | --- |
| `manifest.json` | Format, source system/dataset ID, selected dataset settings, document/chunk counts, SHA-256 of each JSONL file's exact bytes |
| `documents.jsonl` | One object per document: `id`, `name`, `doc_form`, `metadata` |
| `chunks.jsonl` | One object per chunk: `id`, `doc_id`, `position`, `content`, `keywords`, `questions`, `answer`, `enabled`, `metadata`, `content_sha256` |

[The shipped JSON Schema](../src/ragwrench/port/bundle.schema.json) defines a combined validation view `{manifest, documents, chunks}` and reusable `$defs` for each record. Validation additionally checks hashes, counts, referenced documents, unique IDs and positions, nonblank content and canonical ordering. Documents sort by ID; chunks by document ID, position, then chunk ID. Export writes canonical compact JSON with a final newline, without a timestamp, so identical snapshots have identical portable bytes. The manifest is written last as the completed-bundle marker. SHA-256 detects corruption, **not authenticity**; acquire bundles from trusted sources.

Content hashes cover exact UTF-8 content, including whitespace. Dify positions are retained. RAGFlow has no portable ordinal, so its API page order becomes the bundle position. Dify `doc_metadata` lists are wrapped under `metadata.dify_fields`; RAGFlow `meta_fields` objects are retained. Chunk layout/image/tag/attachment references exposed by the supported responses remain descriptive metadata. Source URLs, authorization headers, vectors and credentials are never intentionally serialized. Treat KB text and metadata themselves as private data.

For example data, see [examples/port/bundle](../examples/port/bundle/manifest.json): three synthetic bilingual text documents, five chunks each, including disabled chunks. All example/test content is original and reusable under this repository's MIT license.

## Field mapping and explicit loss boundaries

| Bundle field | Dify target | RAGFlow target (both profiles) |
| --- | --- | --- |
| `content` | Segment content | Chunk content |
| `keywords` | `keywords` | `important_keywords` |
| `enabled` | Segment update `enabled` | Chunk PATCH `available` |
| `answer` | Preserved for `qa_model` documents; each QA chunk needs an answer | Rejected by default; `--allow-lossy` flattens into `Question: …\nAnswer: …` content |
| `questions` | Rejected if nonempty; `--allow-lossy` drops embedding questions | Preserved as `questions` |
| Document/chunk metadata | Retained in bundle; restoring is rejected unless `--allow-lossy` | Same; layout, image and attachment data are not reconstructed |
| Names and IDs | Fresh IDs, source→target mapping in state | Same |

`--allow-lossy` emits explicit warnings and applies only the listed projections. Verification compares against that same projected representation; it does not claim lost fields were preserved. Text documents with answers or QA documents without answers fail Dify preflight. Hierarchical Dify documents are unsupported and fail export. Dify source documents must be enabled, non-archived and fully indexed; disabled RAGFlow documents are also rejected. Document-level disabled/archive state is not recreated. Binary attachments are not fetched.

Target document names keep up to 80 source-name characters plus a deterministic `[kbport-…]` marker. This lets recovery identify a creation whose response was lost. The marker is visible in the target UI. Remote document names are always data, never local filesystem paths.

`--max-chunk-bytes` defaults to **16000 UTF-8 bytes** (configurable up to 1 MiB). This is a conservative local guard, not a guarantee about a target embedding model's token/context limit. Chunks are never silently split. Identical chunk content in a single RAGFlow document is rejected because content-derived IDs can collapse duplicates. RAGFlow parser-specific behavior and service-side content transformations are outside the portable contract; verify after import.

## Recovery and operational safety

**Export:** `--page-size 1..100` (default 100); every successfully checked page is atomically checkpointed in the output directory's `.export-state.json`. Resume with the same source, output and page size:

```bash
ragwrench port export dify --base-url http://localhost/v1 \
  --dataset-id DATASET_ID -o ./bundle --resume
```

GET requests retry 429/5xx and transport failures (`--retries 3`, `--backoff 0.5`, `--timeout 30`). Exponential delays and numeric `Retry-After` are bounded to 30 seconds. Authentication errors and RAGFlow nonzero business codes fail immediately. Totals must remain consistent, with no duplicate IDs, short incomplete pages or contradictory `has_more`. RAGFlow dataset listings have no total, so only document/chunk totals can be checked. Freeze the source during export **and resume**; these APIs provide no transactional snapshot. Restart into a fresh directory after source changes. Completed checkpoint pages are not refetched.

**Import:** keep the state file and use the same target/projection options to resume. State is bound to the portable manifest fingerprint, target URL/system and import options; a new state file is a new migration, not an idempotent re-import. Retry budgets/delays can change between invocations without invalidating an existing checkpoint. Keep one writer per source/target migration, using the same state path. Local locks prevent concurrent access to the same state/output, not distributed writers on different machines. Atomic state writes use owner-only permissions and fsync.

Before each create, the importer saves an intent. On restart, it reconciles pending dataset/document names and pending chunk fields, excluding already mapped chunk IDs. The enabled flag is then applied idempotently. Create requests are **never automatically replayed**, even on 5xx. If no matching pending creation is visible, the command stops: wait for target visibility and retry normally, or inspect the target and explicitly use `--retry-pending` only once you have established the write did not commit. That option authorizes at most one new create attempt per pending chunk per invocation and can create duplicates if your conclusion is wrong. Ambiguous multiple matches always stop. These APIs have no general idempotency key, so unconditional exactly-once behavior cannot be promised.

`import --chunk-retries N` opts into **0..10 extra recovery attempts per unfinished chunk per invocation** (default **0**, preserving immediate-stop behavior). After a transport, HTTP 429 or 5xx failure, recovery first reconciles an uncertain chunk creation through paginated reads, then retries only the enabled flag for its known target ID. A pending chunk absent from listings is polled within the same budget, never automatically recreated. Completed chunk mappings are skipped. `--backoff` exponential delays and numeric `Retry-After` are capped at 30 seconds; listing GETs retain their separate `--retries` budget per page. Authentication, business-envelope, malformed-response and ambiguous-match errors stop immediately. Dataset/document creation and Dify bootstrap cleanup retain manual resume behavior. Exhaustion exits 1 with the durable intent intact; Ctrl-C exits 130 and also retains it.

For example, run this command again after Ctrl-C, keeping the state file:

```bash
ragwrench port import ragflow examples/port/bundle --base-url http://localhost:9380 \
  --dataset-name handbook --state .kbport/handbook.json --chunk-retries 2
```

The shipped bundle's expected completed scope is **3 documents / 15 chunks**, including its original disabled flags. `python examples/port/demo_resume.py` demonstrates this without a service: it sends SIGINT to the installed CLI after the third local HTTP flag write (2 completed mappings plus 1 pending chunk), resumes through a committed chunk whose response is replaced by HTTP 503, verifies all 15 chunks, and asserts exactly 15 chunk-create POSTs and zero writes on a subsequent import. The temporary loopback service and migration state are disposable; no production credentials are used.

### Inspect an interrupted import offline

Before resuming, inspect the saved checkpoint without a bundle, service access or API keys:

```bash
ragwrench port import-status --state .kbport/handbook.json
```

For a checkpoint interrupted while applying the third chunk's enabled flag, the command returns exit **0** and one JSON object of this shape (IDs vary):

```json
{
  "state": ".kbport/handbook.json",
  "target_system": "ragflow",
  "api_profile": "ragflow-1.0",
  "dataset_id": "target-dataset",
  "dataset_pending": false,
  "mapped_documents": 1,
  "completed_chunks": 2,
  "documents": [{
    "source_id": "source-doc", "target_id": "target-doc", "phase": "chunks",
    "completed_chunks": 2, "pending_stage": "enabled_flag",
    "pending_chunk": {"source_id": "three", "target_id": "t-three"}
  }]
}
```

`mapped_documents` counts saved target document IDs, including documents still being prepared. `completed_chunks` counts saved chunk mappings **after** a successful enabled-flag response. The pending chunk is excluded. Documents sort by source ID; each entry reports its saved phase (`new`, `creating`, `clearing`, `chunks` or `complete`). Pending work is described as follows:

| Saved intent | Status output |
| --- | --- |
| Dataset creation not yet mapped | `dataset_pending: true`, `dataset_id: null` |
| Document creation not yet mapped | `pending_stage: "document_creation"`, `target_id: null` |
| Document bootstrap indexing/cleanup | `pending_stage: "bootstrap_cleanup"` |
| Chunk creation not yet mapped | `pending_stage: "chunk_creation"`, `pending_chunk.target_id: null` |
| Chunk ID known, flag response not checkpointed | `pending_stage: "enabled_flag"`, chunk target ID present |
| No saved pending intent for a document | `pending_stage: null`, `pending_chunk: null` |

This is a read-only snapshot of a version 1 import state file. Older states without `api_profile` report their original system default. The command makes no HTTP calls and creates no state, directories or lock files; without `--bundle` it loads no bundle. It can read the atomically replaced checkpoint while an importer holds its writer lock. The snapshot can become stale immediately. A pending write may already have committed remotely; status does not authorize `--retry-pending` or check target visibility.

State does not store bundle totals, unvisited documents/chunks or skipped preflight records. The state-only output therefore has no remaining/skipped counts, overall completion/verification claim or percentage, even when all recorded documents have phase `complete`. Resume with the original bundle/options to finish or rediscover skips, then run `verify` to check the target. A valid partial checkpoint returns exit **0**. Missing/unreadable files, malformed or inconsistent state, unsupported versions/profiles, symlinks, and state larger than 256 MiB return exit **1**, with a safe stderr error and no JSON output or writes. Missing `--state` is an argument error (exit **2**).

To compute local remaining scope, supply the original bundle:

```bash
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle
```

This preserves every snapshot field above and adds `bundle_progress`. For the shipped **3-document / 15-chunk** bundle at that **1 mapped document / 2 completed chunks / 1 pending flag** checkpoint, its aggregate fields are:

```json
{
  "bundle": "examples/port/bundle",
  "manifest_fingerprint": "7e1b8aff4100e1c38d4ee96c7d37be81afed3f6b38a73f8242d37c7a71c7c59e",
  "planned_documents": 3,
  "planned_chunks": 15,
  "skipped_chunks": 0,
  "completed_documents": 0,
  "remaining_documents": 3,
  "remaining_chunks": 13,
  "pending_chunks": 1,
  "unvisited_documents": 2,
  "unvisited_chunks": 12,
  "warnings": []
}
```

Bundle-aware status fully validates the bundle schema, relationships and exact-byte hashes, then matches the saved manifest fingerprint before combining counts. It reuses the state's target system and `allow_lossy`, `max_chunk_bytes`, `on_error` options through the same preflight as import. There are no projection overrides or credentials to supply. Older version 1 states without `api_profile` remain supported with their historical default.

`planned_documents` includes all bundle documents, including empty documents or those with all chunks skipped. `planned_chunks` counts only chunks accepted by the saved preflight; `skipped_chunks` counts exclusions across the whole bundle, including documents not yet visited. `warnings` repeats the preflight's loss warnings. Skipped chunks never count as completed or remaining importable chunks.

`completed_documents` counts saved phase `complete` with all planned chunks mapped. `remaining_documents` is planned minus completed documents, so it includes mapped documents still being prepared or imported. `remaining_chunks` is planned minus completed chunk mappings, **including pending chunks**. `unvisited_documents` counts documents with no state entry. `unvisited_chunks` counts planned chunks with neither a completed mapping nor a pending intent, including untouched chunks in visited documents. Thus remaining chunks equal pending plus unvisited chunks. These are local checkpoint counts, not evidence of remote existence, correctness or full bundle verification. Zero remaining can coexist with skips; run `verify` to check the target.

A valid bundle-aware snapshot exits **0**, even if its nested `skipped_chunks` is nonzero (import's exit **4** policy still applies to import). For example, if preflight skips one chunk and a checkpoint has two completed chunks plus one pending flag, status reports **14 planned / 1 skipped / 12 remaining / 11 unvisited chunks**, with only **2 completed**. A fully processed skip import can report **14 completed / 1 skipped / 0 remaining**; it has not imported all 15 bundle chunks.

A mismatched/corrupt bundle, failed saved preflight, unknown source document/chunk/pending mapping, a mapping to a skipped or wrong-document chunk, or a phase `complete` document missing planned mappings exits **1**, with safe stderr and **no JSON, HTTP calls or writes**. No writer lock is acquired; the state and bundle must remain stable while inspected. State-only mode continues to work without the original bundle and keeps its existing output.

`bundle_progress.documents` adds one row for **every bundle document**, sorted by `source_id`, including unvisited, empty and entirely skipped documents. For the interrupted example, its first row is:

```json
{
  "source_id": "00000000-0000-0000-0000-000000000064",
  "visited": true, "complete": false,
  "planned_chunks": 5, "completed_chunks": 2, "remaining_chunks": 3,
  "pending_chunks": 1, "unvisited_chunks": 2, "skipped_chunks": []
}
```

The other two rows each have `visited: false`, `complete: false`, 5 planned/remaining/unvisited chunks, zero completed/pending chunks and no skips. `visited` means a saved document entry exists (even if creation has not started); `complete` means its saved phase is `complete` and all importable chunks are mapped. Zero importable chunks alone does not make a document complete. Neither field checks the remote target.

Each row's `skipped_chunks` is a list sorted by source chunk ID, with `{ "source_id": "...", "reason": "..." }` entries from the saved preflight. For example, the interrupted skip checkpoint above has **4 planned / 2 completed / 1 pending / 2 remaining / 1 unvisited** chunks in its first document, plus one skipped entry with reason `Unsupported fields: chunk metadata (including image/layout/tags); use --allow-lossy to acknowledge`. Reasons describe the unsupported field or limit without echoing chunk content or metadata values. Chunk IDs are scoped to their document. Row counts sum to the aggregate counts; skipped-list lengths sum to aggregate `skipped_chunks`. A complete document with skips still has exclusions, not completed imports for those skipped chunks. The top-level state `documents` and all existing aggregate fields retain their meanings; state-only JSON is unchanged.

For aggregate-only output on large checkpoints, opt in to `--summary-only`:

```bash
ragwrench port import-status --state .kbport/handbook.json --summary-only
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle --summary-only
```

The state-only response for the interrupted example is exactly:

```json
{
  "state": ".kbport/handbook.json",
  "target_system": "ragflow",
  "api_profile": "ragflow-1.0",
  "dataset_id": "target-dataset",
  "dataset_pending": false,
  "mapped_documents": 1,
  "completed_chunks": 2
}
```

With `--bundle`, the response also retains `bundle_progress` with exactly the aggregate fields shown above (bundle path, manifest fingerprint, counts and warnings). Summary omits top-level `documents` and `bundle_progress.documents` entirely, including per-document skip reasons; it adds no `chunk`, `chunks`, `cursor` or `next_cursor` fields. All retained values equal full status. For the skipped interrupted example, that still means **14 planned / 1 skipped / 2 completed / 12 remaining / 1 pending / 11 unvisited chunks**. Pending creation and pending flag intents each count as one pending chunk; neither counts as completed. An all-skipped checkpoint can have zero remaining and zero completed chunks with nonzero skips. State-only summary cannot infer remaining/skipped counts. Omitting the flag preserves the full JSON exactly.

`--summary-only` cannot be combined with `--document`, `--chunk`, `--list-chunks`, `--chunk-status`, `--page-size` or `--cursor`; these combinations exit **2** before state reads. The direct `import_status(..., summary_only=True)` API rejects the same combinations with `PortError` before reads. All state and bundle schema/relationship/hash/fingerprint checks and saved preflight options still apply, even to omitted details; failures exit **1** without JSON. No credentials, clients, HTTP, locks or writes are used.

This flag reduces returned detail only. It still buffers and validates the entire checkpoint and optional bundle and builds their detail records before omitting the arrays. It does not bound input reads, memory, validation work or the retained warnings array; no streaming or performance guarantee is implied. The counts remain lock-free saved local progress that may immediately become stale, do not verify remote existence, and do not authorize replay of pending writes.

To triage interrupted stages without document arrays, add `--pending-counts` to the summary:

```bash
ragwrench port import-status --state .kbport/handbook.json --summary-only --pending-counts
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle --summary-only --pending-counts
```

This returns the same summary plus exactly one top-level object. For the interrupted flag example:

```json
"pending_counts": {
  "dataset_creation": 0,
  "document_creation": 0,
  "bootstrap_cleanup": 0,
  "chunk_creation": 0,
  "enabled_flag": 1
}
```

These five fields are always integers. `dataset_creation` is 0 or 1 from saved `dataset_pending`; each other field counts **documents** whose full snapshot `pending_stage` equals that field, across the entire checkpoint. A saved chunk creation without a target ID and a saved enabled-flag intent with a known target ID stay distinct. Completed chunks and preflight skips never add pending counts. Initial, `new`, complete and all-skipped documents add no pending intent; only an explicit dataset intent sets `dataset_creation` to 1. These are saved local intents, not the amount of unfinished remote work, verified remote existence or permission to retry.

`--pending-counts` requires `--summary-only`; omission of summary exits **2** before reads. The existing summary selector/filter/pagination conflicts still apply. Direct `import_status(..., summary_only=True, pending_counts=True)` provides the same result and rejects invalid combinations with `PortError` before reads. Omitting `--pending-counts` (or passing `pending_counts=False`) preserves all existing full, summary, selected, filtered and paged JSON exactly. No IDs or detail arrays are added; the existing summary fields, including dataset ID, paths, fingerprint and warnings, remain unchanged. Full state/bundle/hash/fingerprint and saved-preflight validation still run, including validation of omitted details and legacy profile defaults. The output-only limits above also apply: full inputs and detail records remain buffered, warnings are unbounded, reads are lock-free, and no credentials, client, HTTP, locks or writes are used.

To drill into one of those document counts, filter the full snapshot by saved pending stage:

```sh
ragwrench port import-status --state .kbport/handbook.json --pending-stage enabled_flag
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle --pending-stage chunk_creation
```

`--pending-stage` accepts exactly `document_creation`, `bootstrap_cleanup`, `chunk_creation` or `enabled_flag`. It narrows top-level `documents` and, with a bundle, `bundle_progress.documents` to the same matching source IDs in their original sorted order, each once. Rows remain identical to the full snapshot. Every other field and global/bundle aggregate stays unchanged, so totals may exceed the filtered rows. No match returns `documents: []` in both locations. The interrupted flag example above returns its one visited document for `enabled_flag` and empty arrays for the other three stages; its mapped/completed/remaining/unvisited counts remain global.

Matching uses exactly the full snapshot's `pending_stage`, the same saved local intent counted by `--pending-counts`. Chunk creation with no target ID remains distinct from an enabled-flag intent with a known target ID. Documents with no saved intent, including new, complete, skipped-only and unvisited documents, are excluded; remaining work does not imply pending intent. Dataset creation is a scalar intent, so `dataset_creation`, invalid spellings and empty stages are rejected. These snapshots do not verify remote existence or authorize retrying writes.

The filter cannot combine with `--summary-only`, `--pending-counts`, `--document`, `--chunk`, `--list-chunks`, `--chunk-status`, `--page-size` or `--cursor`. Invalid stages or combinations exit **2** before any state/bundle read. Direct `import_status(..., pending_stage="enabled_flag")` follows the same contract and raises `PortError` for argument errors before reads. Omitting the flag (or passing `pending_stage=None`) preserves every previous JSON mode. Filtering is output-only, after full state/bundle/schema/hash/fingerprint/relationship and saved-preflight validation, including nonmatching details and legacy defaults. It still buffers full inputs and constructs all detail rows; warnings and memory are not bounded by the filter. Reads are lock-free and may immediately become stale; no credentials, client, HTTP, lock or write access is used.

To page that pending-stage drill-down in either state-only or bundle mode:

```sh
ragwrench port import-status --state .kbport/handbook.json --pending-stage enabled_flag --document-page-size 20
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --pending-stage enabled_flag --document-page-size 20 --document-cursor "$DOCUMENT_CURSOR"
```

Use the same optional bundle scope on every request. Paging filters by stage **before** slicing the original sorted source-ID order. Both `documents` arrays contain the identical page IDs, with rows unchanged from full status. Every other field and global aggregate remains unchanged. Only paging adds top-level `next_document_cursor`: an opaque string when more matching documents remain, otherwise `null`. Stop on null. No-match, empty, exact-size and oversized final pages return null. For matching IDs `[d02, d05, d09]` and size 2, the pages are `[d02, d05]` with a token, then `[d09]` with null. Omitting both document paging flags preserves the existing filtered JSON exactly, without this extra field.

`--document-page-size` must be a positive integer and requires `--pending-stage`. `--document-cursor` additionally requires `--document-page-size`; it is separate from chunk `--cursor`/`--page-size`. All pending-stage conflicts above remain errors. Argument errors exit **2** before input reads; direct `import_status(..., pending_stage="enabled_flag", document_page_size=20, document_cursor=token)` raises `PortError` for the same errors. Invalid/malformed/unsupported/noncanonical tokens, wrong snapshot/stage/bundle/scope or chunk-kind tokens, and absent/nonmatching/end boundaries exit **1** without JSON (API `PortError`). Token contents are never echoed in these errors.

Document continuations are deterministic, versioned, bounded to 2048 characters and bind the SHA-256 of the **exact state bytes**, pending stage, state-only versus bundle scope, optional manifest fingerprint, and last matching source ID. Any checkpoint byte change, including whitespace, key order or unknown fields, requires restarting without the cursor. Adding/removing/changing the bundle or changing stage also requires a restart. Identical copies at other paths are accepted, and page size may change between requests. Chunk v1/v2 cursors retain their existing formats and cannot be exchanged with document cursors. Tokens contain IDs/hashes and scope, never document text, credentials or target URLs; they are unsigned continuation data, not authorization or retry permission.

This is an **output-only limit**: every request still buffers and validates the entire state/bundle, checks all hashes/fingerprint/relationships and reruns saved preflight before slicing or checking the cursor, including invalid nonmatching and off-page details. Detail construction, warnings, memory and reads remain unbounded by page size. Lock-free snapshots can immediately become stale; paging does not check remote state or permit write replay. No credentials, clients, HTTP, locks or writes are added.

To inspect just one document's detail row, add its **source** ID:

```bash
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --document 00000000-0000-0000-0000-000000000064
```

Only `bundle_progress.documents` is narrowed to that one row; the state snapshot (including top-level `documents`) and every aggregate remain scoped to the **whole bundle**. The selected interrupted row is the 5 planned / 2 completed / 1 pending / 3 remaining / 2 unvisited example above, while global counts remain 15 planned / 2 completed / 13 remaining. Selecting source ID `00000000-0000-0000-0000-000000000065` instead gives one unvisited row with 5 planned/remaining/unvisited chunks and zero completed/pending, with those same global counts. Selected row counts therefore describe only that document and need not sum to global totals. Omit `--document` to retain all rows and the existing JSON exactly.

Unvisited, empty and all-skipped documents are selectable. Skip reasons and saved projection options retain their meanings: selecting the first document of the skip example gives 4 planned / 2 completed / 1 pending / 2 remaining / 1 unvisited plus its one excluded chunk, while global scope remains 14 planned / 1 skipped / 12 remaining. Selection still validates **every** bundle file and all saved mappings/pending intents, including documents outside the selection. An unknown source ID exits **1** with safe stderr and no JSON; `--document` without `--bundle` is an argument error (exit **2**). Selection performs no credentials/client/network/lock/write access and remains local progress only.

To list saved-preflight skips across **all** bundle documents:

```sh
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle --list-skipped-chunks
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --list-skipped-chunks --skip-page-size 20
# Continue with bundle_progress.next_skip_cursor, stopping on null:
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --list-skipped-chunks --skip-page-size 20 --skip-cursor "$SKIP_CURSOR"
```

This adds only `bundle_progress.skipped_chunk_details`, sorted by `(document_source_id, source_id)`. Every row has exactly those two IDs and `reason`, identical to the existing document exclusion reason. For example:

```json
[
  {"document_source_id":"a","source_id":"c2","reason":"Unsupported fields: chunk metadata (including image/layout/tags); use --allow-lossy to acknowledge"},
  {"document_source_id":"b","source_id":"c2","reason":"Chunk exceeds configured target UTF-8 byte limit"}
]
```

Reused chunk IDs stay document-scoped. Visited, unvisited and all-skipped documents contribute every exclusion once; empty and no-skip scopes yield `[]`. Skips remain exclusions rather than completed imports. Both document arrays, all global counts, warnings and existing skip details stay unchanged. Without `--skip-page-size`, the list is complete and no continuation field is added. Omitting all three new flags preserves all previous JSON.

Paging enumerates skips **before** slicing and adds only `bundle_progress.next_skip_cursor`: an opaque string when more skips remain, otherwise `null`, including no matches, exact-size and oversized final pages. A positive integer size requires `--list-skipped-chunks` and `--bundle`; `--skip-cursor` also requires the size. This mode cannot combine with summary/count/pending-stage/document/chunk selectors, chunk listing/filter/pagination, or document pagination. Argument errors exit **2** before input reads. Direct `import_status(state_path, bundle, list_skipped_chunks=True, skip_page_size=20, skip_cursor=token)` uses the same rules with `PortError` (booleans are not integer sizes).

Separate versioned skip tokens bind the SHA-256 of the **exact state bytes**, bundle manifest fingerprint and last compound `[document_source_id, source_id]` boundary. Tokens are canonical base64url JSON, bounded to 2048 characters, and cannot be exchanged with document or chunk cursors. Malformed/noncanonical/stale/cross-kind tokens and absent, non-skipped or final boundaries exit **1** without JSON (API `PortError`); errors never echo tokens. Any checkpoint byte change, even whitespace, key order or unknown fields, requires restarting without the cursor. Identical copies at other paths and page size changes are accepted. Tokens contain IDs/hashes, never chunk text, credentials or target URLs; they are unsigned continuation data, not remote verification or retry authorization.

This is an **output-only limit**. Every call fully buffers and validates state and bundle, checks schema/hash/content-hash/counts/fingerprint/relationships and recomputes saved preflight before projection or cursor checks, including invalid off-page and non-skipped details. Saved lossy/byte/skip options and legacy profile defaults apply. The document arrays, exclusion details, warnings, reads and memory remain unbounded by page size. Lock-free snapshots may immediately become stale. No credentials, clients, HTTP, locks or writes are used.

To list local chunk statuses across **every** bundle document:

```sh
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle --list-bundle-chunks
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --list-bundle-chunks --bundle-chunk-status pending --bundle-chunk-page-size 20
# Continue with bundle_progress.next_bundle_chunk_cursor, stopping on null:
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --list-bundle-chunks --bundle-chunk-status pending --bundle-chunk-page-size 20 \
  --bundle-chunk-cursor "$BUNDLE_CHUNK_CURSOR"
```

This adds only `bundle_progress.chunk_details`: the same six-field rows as document-scoped chunk details, ordered by `(document_source_id, source_id)`. Each original chunk appears once within its document, including repeated IDs across documents, exclusions, visited/unvisited and complete documents. Empty documents contribute no rows; all-skipped documents contribute skipped rows. For example:

```json
[
  {"document_source_id":"a","source_id":"c0","status":"completed","target_id":"t0","pending_stage":null,"reason":null},
  {"document_source_id":"a","source_id":"c1","status":"pending","target_id":null,"pending_stage":"chunk_creation","reason":null},
  {"document_source_id":"b","source_id":"c0","status":"unvisited","target_id":null,"pending_stage":null,"reason":null}
]
```

A pending target ID instead means `pending_stage: "enabled_flag"`. Skipped rows retain their saved-preflight reason with null target and stage, even in complete documents; they are exclusions, not completed imports. Optional `--bundle-chunk-status` accepts exactly one lowercase `completed`, `pending`, `unvisited` or `skipped` and filters **before** slicing. Both document arrays and every global field remain unchanged, including counts, warnings and document skip details. Without page size, all matching rows are returned and no continuation field is added. Omission of these four flags preserves every previous mode and JSON shape; the three-field skip-only mode remains available separately.

Only paging adds `bundle_progress.next_bundle_chunk_cursor`: a string when more matching rows remain, otherwise `null`, including no matches, empty, exact-size and oversized last pages. Size must be a positive integer. Filter/size/cursor require `--bundle` and `--list-bundle-chunks`; the cursor also requires the size. This mode conflicts with all summary/count/pending-stage/document/chunk selectors, document-scoped listing/filter/pagination, document pagination, and skip-only listing/pagination. Argument errors exit **2** before input reads. API `import_status(state_path, bundle, list_bundle_chunks=True, bundle_chunk_status="pending", bundle_chunk_page_size=20, bundle_chunk_cursor=token)` applies the same rules with `PortError`, rejecting boolean sizes.

Separate canonical base64url JSON tokens (version 1, kind `bundle-chunks`, at most 2048 characters) bind the exact state-byte SHA-256, manifest fingerprint, normalized status string or null, and compound `[document_source_id, source_id]` boundary. Page size may change and identical copies at other paths can continue. Any state-byte change (including whitespace/key order/unknown fields), changed filter or bundle requires restarting without the cursor. Malformed, noncanonical, stale, cross-kind, absent/nonmatching or final-boundary cursors exit **1** without JSON (API `PortError`); errors never echo tokens. Tokens contain IDs/hashes and filter, no text, credentials or target URLs. They are unsigned continuation data, not remote verification or retry permission.

These are **output-only limits**: the entire state/bundle and saved preflight are buffered and validated before projection or cursor checking, including every excluded/off-page record, saved lossy/byte/skip option and legacy profile default. Detail construction, both document arrays, skip details, warnings, reads and memory remain unbounded by page size. Snapshots are lock-free and may immediately become stale. No credentials, clients, HTTP, locks or writes are used.

To inspect one chunk's local checkpoint status, supply both source IDs:

```sh
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --document 00000000-0000-0000-0000-000000000064 \
  --chunk 00000000-0000-0000-0000-0000000003ea
```

This adds `bundle_progress.chunk` while preserving every field of the document-selected output, including global counts and the selected document row. At the interrupted example's pending flag, the added field is:

```json
{
  "document_source_id": "00000000-0000-0000-0000-000000000064",
  "source_id": "00000000-0000-0000-0000-0000000003ea",
  "status": "pending",
  "target_id": "t-three",
  "pending_stage": "enabled_flag",
  "reason": null
}
```

The six fields are always present. `status` has these local meanings:

| Status | `target_id` | `pending_stage` | `reason` |
| --- | --- | --- | --- |
| `completed`: durable mapping after the flag response | Saved target chunk ID | `null` | `null` |
| `pending`: create intent without a saved target ID | `null` | `chunk_creation` | `null` |
| `pending`: target ID saved, flag response not checkpointed | Saved target chunk ID | `enabled_flag` | `null` |
| `unvisited`: importable chunk without a mapping or pending intent | `null` | `null` | `null` |
| `skipped`: excluded by saved preflight | `null` | `null` | Safe preflight reason |

For example, chunk `…3e8` is `completed` with target `t-one`; `…3ec` is `unvisited` in that same visited document. If the skip example excludes `…3ec`, it becomes `skipped` with the metadata reason above and never counts as imported. An all-skipped document's chunks are selectable exclusions; an empty document has no selectable chunk. `unvisited` describes the chunk even when its document has already been visited. Reused chunk IDs resolve within the selected document only; an ID found only in another document is unknown.

`--chunk` requires **both** `--bundle` and `--document` (argument error, exit **2**, before reading state). An unknown source document or chunk exits **1**, with safe stderr and no JSON or selector-value echo. Chunk selection retains full-bundle hash/fingerprint/preflight and full-state validation, including invalid mappings outside the selection, saved options and legacy defaults. It adds no credentials, HTTP, locks or writes. A pending write may already exist remotely; these statuses never verify remote data or authorize a retry. Omit `--chunk` to keep the existing document-selected/default/state-only output exactly. Selection still loads and validates the complete state and bundle; it does not reduce validation cost.

To list every source chunk in one document, use `--list-chunks`:

```sh
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --document 00000000-0000-0000-0000-000000000064 --list-chunks
```

This adds `bundle_progress.chunks`, sorted by `source_id`. Each row has exactly the six fields and local meanings of an independent `--chunk` result above. For the interrupted example, the added list is:

```json
[
  {"document_source_id":"00000000-0000-0000-0000-000000000064","source_id":"00000000-0000-0000-0000-0000000003e8","status":"completed","target_id":"t-one","pending_stage":null,"reason":null},
  {"document_source_id":"00000000-0000-0000-0000-000000000064","source_id":"00000000-0000-0000-0000-0000000003e9","status":"completed","target_id":"t-two","pending_stage":null,"reason":null},
  {"document_source_id":"00000000-0000-0000-0000-000000000064","source_id":"00000000-0000-0000-0000-0000000003ea","status":"pending","target_id":"t-three","pending_stage":"enabled_flag","reason":null},
  {"document_source_id":"00000000-0000-0000-0000-000000000064","source_id":"00000000-0000-0000-0000-0000000003eb","status":"unvisited","target_id":null,"pending_stage":null,"reason":null},
  {"document_source_id":"00000000-0000-0000-0000-000000000064","source_id":"00000000-0000-0000-0000-0000000003ec","status":"unvisited","target_id":null,"pending_stage":null,"reason":null}
]
```

The selected document row, state snapshot and global totals stay unchanged: **15 planned / 2 completed / 13 remaining**. If the saved preflight excludes `…3ec` for metadata, that last row instead has `status: "skipped"`, the safe metadata reason from the table above, and null target/stage; global totals become **14 planned / 1 skipped / 2 completed / 12 remaining**. Skips remain exclusions. An unvisited document lists unvisited importable chunks and any skips; a complete document lists completed mappings and any skips. An empty document returns `chunks: []`. Reused chunk IDs stay scoped to the selected document.

`--list-chunks` requires **both** `--bundle` and `--document`, and cannot be combined with `--chunk`. Violations are argument errors (exit **2**) before state reads. Unknown documents exit **1** with safe stderr, no JSON and no selector-value echo. Listing still validates the full bundle hashes/fingerprint, saved preflight and all state, including data outside the selection; saved projection/byte-limit/on-error options and legacy defaults apply. It adds no credentials/client/HTTP/locks/writes and makes no remote-verification or retry-authorization claim. Omit the flag to preserve all previous outputs exactly. Without `--page-size`, listing remains unpaginated and its JSON is unchanged: output grows with every original chunk in the selected document, including skips.

To page the same list, add a positive integer `--page-size`:

```sh
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --document 00000000-0000-0000-0000-000000000064 --list-chunks --page-size 2
# Pass the returned bundle_progress.next_cursor verbatim for the next page:
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --document 00000000-0000-0000-0000-000000000064 --list-chunks --page-size 2 --cursor "$CURSOR"
```

`bundle_progress.chunks` is now one slice of the full `source_id`-ordered list. The only additional field is `bundle_progress.next_cursor`: a string if more rows remain, otherwise `null`. In the interrupted five-chunk example above, size 2 yields `[completed, completed]`, then `[pending, unvisited]`, then `[unvisited]`. Each row equals its independent `--chunk` result; every page retains the selected document detail and global **15 planned / 2 completed / 13 remaining** totals. Skipped chunks participate in page ordering as exclusions. Empty documents return `chunks: []` and `next_cursor: null`; exact-size and oversized final pages also return null. Stop on null. Omit both paging options to preserve the full-list schema (no `next_cursor` field).

A continuation starts strictly after the previous page's last source ID. Its deterministic, versioned opaque token binds the selected document, manifest fingerprint, and SHA-256 of the **exact saved-state bytes** read for that response. Any checkpoint change, including whitespace, key order or unknown fields, invalidates the cursor. Restart without `--cursor` after a change; keep the original matching bundle and saved state to traverse one snapshot. Identical copies at other paths are accepted, and page size may change between requests. Tokens contain IDs and hashes, never chunk text, credentials or target URLs. They are not signed authentication tokens, remote verification or permission to replay writes. Treat them as opaque, private checkpoint data.

`--page-size` and `--cursor` require `--list-chunks` and its bundle/document scope; `--cursor` also requires `--page-size`. Non-positive/non-integer sizes and conflicts with `--chunk` are argument errors (exit **2**) before state reads. Invalid, unsupported, oversized or noncanonical cursors, mismatched document/bundle/state bindings, and unknown/end-of-list boundaries exit **1** with safe stderr and no JSON. Cursors are bounded to 2048 characters. Each request still performs the established full state, bundle hash/fingerprint and saved-preflight validation, including data outside the page. Paging limits only the returned chunk rows; the global snapshot and document details (including all skip reasons) can still grow, and the entire state/bundle remain buffered. Reads are lock-free and local progress may become stale immediately after a response; no credentials, client construction, HTTP, locks or writes are added.

To show one local status, add `--chunk-status completed|pending|unvisited|skipped`:

```sh
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --document 00000000-0000-0000-0000-000000000064 --list-chunks \
  --chunk-status completed --page-size 1
# Continue with the same filter and returned token:
ragwrench port import-status --state .kbport/handbook.json --bundle examples/port/bundle \
  --document 00000000-0000-0000-0000-000000000064 --list-chunks \
  --chunk-status completed --page-size 1 --cursor "$CURSOR"
```

Filtering runs **before** page slicing. `chunks` is the matching subset of the full source-ID-ordered list, with identical six-field rows; without `--page-size` it contains every match and has no `next_cursor`. The interrupted example's completed filter returns `…3e8` then `…3e9` with size 1; pending returns only `…3ea`, for either pending stage. Unvisited returns `…3eb` and `…3ec`. If saved preflight skips `…3ec`, only the skipped filter includes that exclusion and unvisited returns `…3eb`. No matches (including an empty document) give `chunks: []`, plus `next_cursor: null` when paged. Complete/all-skipped/unvisited documents use the same local statuses. All snapshot fields, global counts and selected document detail, including its complete skip list, stay unchanged.

The filter is one exact lowercase choice; invalid choices, a missing `--list-chunks`, missing bundle/document scope or conflicts with `--chunk` exit **2** before reading state. Filtered continuations use cursor **v2**, adding `chunk_status` to the v1 document/bundle/exact-state-byte/after-ID binding. The enum spelling is the normalized filter; there are no aliases or combined filters. Unfiltered requests retain the original **v1 schema and tokens**. Adding, removing or switching filters with a cursor exits **1** with no JSON; restart without `--cursor` to change the filter. Other snapshot/cursor checks and limits above apply to both versions. Tokens remain opaque and unsigned.

Filtering and pagination limit only chunk rows in the response: they still read, buffer and validate the full state/bundle and run saved preflight, including excluded rows and other documents. They add no credentials, clients, HTTP, locks or writes, and describe saved local progress rather than remote verification or permission to retry a pending write.

The installed-CLI `examples/port/demo_resume.py` checks bundle-wide chunk details and every status filter against the combined document-scoped rows, pages through null with changed sizes, and rejects exact-byte stale cursors at interrupted, legacy-skip, multi-document-skip, pending-create and completed checkpoints without HTTP or file changes. It checks bundle-wide skip listing and pages against full exclusion slices at interrupted/no-skip, legacy-skip, multi-document-skip and completed checkpoints, preserving globals and reasons, accepting size changes, ending on null and rejecting exact-byte stale cursors without HTTP or file changes. It checks document paging against matching slices through null in state-only/bundle modes, including a mixed-intent checkpoint derived from a real SIGINT interruption, and rejects changed stage, scope and exact state bytes. It also checks all four pending-stage filters against full snapshot document subsets in state-only and bundle modes, including no matches, interrupted flags, legacy skips, pending creates and completion; it rejects invalid stages, conflicting options and invalid nonmatching details. It compares opt-in pending counts with all saved document stages at interrupted flag, legacy skip, pending-create and complete checkpoints, and rejects pending counts without summary. It compares summary-only state/bundle output with full aggregates, omitting both document arrays at interrupted, legacy skip, pending-create and complete checkpoints; it rejects conflicting summary flags and invalid omitted details without status HTTP calls or file changes. It also checks status at the interrupted **1 mapped document / 2 completed chunks / pending flag** checkpoint and after resuming to **3 / 15**, with no keys or status HTTP calls. It checks the **3 / 13 remaining, 2 / 12 unvisited** scope and each document's counts, illustrates skip reasons with a separate derived legacy checkpoint, checks selected document/chunk details and chunk lists with unchanged global totals (including a derived pending-create intent), and compares every listed row with an independent single-chunk result. All four filters match full-list subsets before paging, including no matches, skips and both pending stages. Paged requests match the selected slices through a null continuation, reject filter/document/state-byte changes and stale v1/v2 cursors after resume, and preserve the same offline guarantees. It rejects unknown/wrong-document/missing-scope/combined selectors and invalid state outside the selection, rejects fingerprint/hash mismatches without changing files, and checks that a missing state file fails without creating its directory.

Dify strategy: create one bootstrap text document, poll its detail endpoint until indexing completes (`--poll-timeout 120`), delete its generated segments, then add each original segment and restore its enabled flag. The `clearing` phase is durable, so an interruption continues clearing before any imported chunk is written. Only documents created/mapped by this migration are cleared; arbitrary existing documents are not overwritten. Dify QA bootstrap can use the target's configured QA processing provider. RAGFlow strategy: create an empty virtual document (`type=empty`), never trigger parsing, add chunks, then set availability with PATCH.

`--on-error abort` rejects all invalid chunks before mutation. `--on-error skip` skips per-chunk **preflight** errors, reports their IDs/reasons and returns exit **4**, even for dry runs. Document-level metadata errors still abort. Network/write ambiguity, authentication failures and remote rejections always stop with exit **1** so recovery state is retained; this version does not silently skip uncertain writes. Do not treat a partial import as verified. Repair the bundle and start a reviewed migration with a new state, or adjust your acknowledged projection before starting.

`--dry-run` performs complete local bundle validation and field/size preflight, emits the counts/errors/warnings and target selection, and makes **zero HTTP calls and zero filesystem writes**. Credentials are not required. It cannot discover remote name collisions or provider/token constraints.

## Verification and resource limits

`verify --state FILE` reads target documents/chunks through the same paginated APIs and compares SHA-256 of content **and** normalized answer, sorted keyword/question lists and enabled status, with multiplicity per mapped document. Missing documents, changed chunks and extra chunks in those documents fail. Unrelated documents in a reused dataset are ignored. Without state, `verify --dataset-id ID` checks total document count and a dataset-wide chunk multiset; use state to detect misplaced chunks between documents. Ordering, IDs, marker names and descriptive metadata are not verification criteria. Skipped chunks prevent a full match.

File validation accepts only the fixed filenames; manifest keys cannot select paths. Symlink bundle roots/files and state/lock files are refused. JSON/JSONL inputs are data; no scripts, templates, SQL, archive extraction or subprocess workflows are executed by the CLI. JSON/JSONL files are bounded to 256 MiB each, JSONL lines to 2 MiB, and content to 1 MiB characters plus the import byte guard. API responses are capped at 32 MiB decompressed (reduce `--page-size` for large chunks), and each paginated resource at 100000 records. Checkpoint/state JSON is also capped at 256 MiB. Data is buffered in memory; this is a small/medium KB tool, not a streaming multi-gigabyte migrator. No export/import rollback or source deletion is implemented. Keep a target backup and run verification.

Exit codes: **0** success, **1** validation/API/I/O failure, **2** argument syntax error, **4** partial preflight skip, **130** interruption.

## Offline fixture servers and tests

```bash
# Optional manual fixture server; terminate with Ctrl-C. It binds loopback only.
python -m ragwrench.port.fakes dify --seed examples/port/bundle --port 9381
# In another terminal, use DIFY_KB_API_KEY=fake-key and base URL http://127.0.0.1:9381.
```

The fixture server is a development aid with in-memory state, not a production service. It implements document/chunk pagination, authenticates requests, creates Dify bootstrap segments, checks mutation payload fields, persists writes in memory and supports injected failures in tests. The offline test suite runs against this fixture server for both systems; it makes no live-instance compatibility or performance claim.

This independent tool is not endorsed by either upstream project.
