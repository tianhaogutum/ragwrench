# ragwrench dify-bridge

Dify can query an external knowledge base over HTTP, but your chunks may already live in
Qdrant or a JSONL file that you do not want to re-ingest. `ragwrench dify-bridge` serves
them through Dify's External Knowledge API, and its `check` command probes any such
endpoint for contract compliance. Use it to connect an existing index to Dify, or to test
an External Knowledge API implementation before wiring it into Dify.

_Formerly the standalone `dify-external-kb-bridge` tool (`difykb-bridge`)._

Expose existing Qdrant chunks to Dify without re-ingestion, or try retrieval with a local JSONL/BM25 corpus. Includes a reusable HTTP contract checker. MIT. No models, API accounts or external services are needed for the demo/tests.

## Quick start

The example corpus is a JSONL file, so the base install is enough:

```bash
pip install ragwrench
export BRIDGE_API_KEY=local-demo
ragwrench dify-bridge serve --config examples/dify-bridge/bridge.toml --port 8765
```

In another terminal:

```bash
curl -sS http://127.0.0.1:8765/retrieval \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-demo' \
  --data-binary @examples/dify-bridge/request.json

BRIDGE_API_KEY=local-demo ragwrench dify-bridge check \
  --endpoint http://127.0.0.1:8765 --knowledge-id handbook --query 'vacation policy'
```

Configure Dify with the base endpoint `http://127.0.0.1:8765`, the same API key, and External Knowledge ID `handbook`. Dify must be able to reach that address: a Dify container's loopback is its own container. Choose an appropriate trusted network address and explicit `--host` for that deployment.

The example JSON request asks for up to three English chunks above score 0.2. The JSONL backend returns two vacation chunks, ordered by score, e.g.:

```json
{"records":[{"content":"Employees accrue 25 days of paid vacation per year. Request vacation through your manager.","title":"handbook.md","score":0.4673488846,"metadata":{"lang":"en","page":4,"team":"people","updated":"2026-10-01","tags":["policy","leave"]}}]}
```

The excerpt shows the first record; scores depend on corpus/tokenization and are not probabilities. No matches returns `{"records":[]}`. Missing metadata becomes `{}`; explicit null/non-object metadata is an error. Only allowlisted metadata leaves the bridge or participates in filtering. `internal_secret` in the fixture demonstrates exclusion.

JSONL serving, the HTTP server, embedding HTTP transport and the checker use only the
Python standard library; Qdrant-backed knowledge needs the `qdrant` extra (`pip install
'ragwrench[qdrant]'`).

## Reload a local corpus

After publishing a replacement JSONL file at the configured path, reload it without restarting the service:

```bash
curl -sS http://127.0.0.1:8765/reload \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-demo' \
  --data '{"knowledge_id":"handbook"}'
# With the unchanged four-record example: {"knowledge_id":"handbook","version":2,"record_count":4}
```

Write a complete candidate beside the corpus and publish it with an atomic rename on the same filesystem (for example, `mv chunks.next chunks.jsonl`). The loader opens the file once, validates every record and builds all BM25 statistics before swapping snapshots. Queries capture one snapshot at retrieval start: an in-flight query finishes with its original records and scores; later queries use the replacement. Queries can continue during the build, and concurrent reloads of the same knowledge ID are serialized. Editing the file alone does not trigger a reload.

Each JSONL knowledge ID starts at version 1. Every successful reload, including unchanged or empty files, increments that ID's version; versions reset on service restart. A missing, malformed, oversized or detectably changing file returns HTTP 503 / `3001` with `"JSONL reload failed; previous corpus retained"`. The previous version stays active and failures do not consume a version. Empty files are valid and clear that corpus. Startup still fails if the initial corpus is invalid.

`/reload` accepts exactly `knowledge_id`, uses the same Bearer key, HTTP framing and request limits as `/retrieval`, and never accepts a path or changes configuration. Unknown IDs return 404 / `2001`; Qdrant IDs and invalid bodies return 400 / `2002`. This is a bridge extension, separate from Dify's retrieval contract; `/retrieval` responses keep their existing shape. Library users can call `JsonlBackend.reload()` for `{version, record_count}` (failure raises `ConfigError`).

Reload is per knowledge ID, even if IDs share a file; there is no transaction across IDs, file watcher, persistent version history or rollback endpoint. Allow memory for the candidate plus the active snapshot and any old snapshots still used by queries. Atomic rename is required for reliable publication: the loader detects size/mtime changes from in-place writes, but cannot guarantee consistency against arbitrary concurrent writers editing the same inode.

## Inspect the active JSONL snapshot

Use authenticated `POST /snapshot/status` to inspect one configured JSONL knowledge ID without reading the source file, rebuilding BM25 statistics or running retrieval:

```bash
curl -sS http://127.0.0.1:8765/snapshot/status \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-demo' \
  --data '{"knowledge_id":"handbook"}'
# Before any reload of the example: {"knowledge_id":"handbook","version":1,"record_count":4}
```

The installed CLI provides the same read-only status without curl:

```bash
BRIDGE_API_KEY=local-demo ragwrench dify-bridge snapshot-status \
  --endpoint http://127.0.0.1:8765 --knowledge-id handbook
# stdout before reload (exit 0):
# {"knowledge_id": "handbook", "version": 1, "record_count": 4}
# After one successful reload of the unchanged example (exit 0):
# {"knowledge_id": "handbook", "version": 2, "record_count": 4}
```

`snapshot-status` makes exactly one `POST /snapshot/status` with `{"knowledge_id":"handbook"}`; it never calls reload or retrieval. It reads the key only from `--key-env` (a variable name matching `[A-Za-z_][A-Za-z0-9_]*`, default `BRIDGE_API_KEY`), with no token argument or fallback to another variable. The key must be a non-empty visible ASCII token without whitespace, at most 4096 characters. `--knowledge-id` must contain 1–256 characters. `--endpoint` is an ASCII HTTP(S) base URL (an optional path prefix is preserved), with a hostname, valid port, and no credentials, whitespace, query or fragment. `--timeout` defaults to 10 seconds and must be finite in (0,120]. Redirects are rejected without following them.

Exit 0 prints only a JSON object with exactly the requested string `knowledge_id`, integer `version` ≥ 1 and integer `record_count` ≥ 0; booleans and floats are rejected. Exit 1 covers HTTP/auth/unknown-ID/non-JSONL failures, transport failures, redirects and invalid responses (including duplicate keys, extra fields, wrong ID/types, non-JSON content, truncation or bodies over 64 KiB). Exit 2 covers local argument/URL/key/timeout/ID errors before HTTP. Failures leave stdout empty and print a generic stderr message without tokens, endpoint/ID values or upstream response content. Status continues to report the active snapshot after the source file is deleted; repeated calls never advance its version.

The body must contain exactly `knowledge_id`, a string of 1–256 characters. This read-only POST uses the existing JSON-body framing, limits and shared Bearer key, and returns HTTP 200 with exactly `knowledge_id`, integer `version` and integer `record_count`. The response has `Cache-Control: no-store`. Invalid bodies (including extra fields or duplicate JSON keys) and Qdrant IDs return 400/`2002`; unknown IDs return 404/`2001`. Missing, duplicate or malformed auth returns 401/`1001`; a wrong key returns 403/`1002`. Error bodies contain only `error_code` and `error_msg`, with no credentials, file paths or record content.

Version and count come from one captured active snapshot. During a reload build, status remains available with the previous snapshot; a request racing publication may report the old or new pair. Repeated status calls never increment the version. File edits, a malformed candidate, a failed reload or a deleted source file leave status unchanged until a successful reload publishes a replacement. `record_count` counts all loaded records, including those with no lexical matches. IDs sharing a file still have independent status. Versions start at 1 and reset on service restart; status reports neither source-file readiness nor persistent history. This is a bridge extension, separate from Dify's retrieval contract and unauthenticated `/healthz`. Library users can call `JsonlBackend.snapshot_status()` for `{version, record_count}`.

Run `python examples/dify-bridge/reload_demo.py` for an installed-console-CLI example that prints status before and after publication, after deleting the source file, and for an empty corpus. It also checks wrong-key/unknown-ID exit 1 and empty-key exit 2.

## Use an existing Qdrant collection

Copy [examples/dify-bridge/qdrant.toml](../examples/dify-bridge/qdrant.toml) and set its collection, vector name, payload fields, metadata allowlist, and embedder. Select either `url` or an **existing** local `path`. Relative paths resolve from the TOML file. The bridge never creates collections or ingests documents. Local Qdrant requires exclusive access to its directory; close other clients first. Knowledge IDs sharing a local path share one client and lock.

The expected payload is:

```json
{"content":"A source chunk","title":"source.md","metadata":{"lang":"en","page":4}}
```

Field names can be mapped with `content_field`, `title_field`, `metadata_field`; all default to the keys shown. Missing title defaults to `""`; missing content, non-string title/content and invalid metadata fail explicitly. Fields are literal simple names, not dotted paths. Arbitrary knowledge IDs in requests cannot select arbitrary collections or filesystem paths.

Choose the same embedding algorithm/model and dimensions used at indexing time. Collection dimensions, distance and named/unnamed vector selection are verified on retrieval. Supported dense distances: cosine, dot, euclid. Sparse vectors, multivectors, Manhattan distance and automatic embedding inference are unsupported.

* `kind = "hash"`: deterministic SHA-256 signed token hashing, L2 normalized, default 64 dimensions. A reproducible offline demonstration, **not a semantic language model**. Empty-token queries embed to a zero vector.
* `kind = "http"`: posts `{"model":"...","input":"..."}` to the exact configured `url`, accepts one `data[0].embedding` vector of the configured dimension. Optional `api_key_env` defaults to `EMBEDDING_API_KEY`; no key is sent if unset. Supports OpenAI-compatible services without an OpenAI SDK. Redirects are rejected, response size is capped at 4 MiB, and a timeout is enforced. Use HTTPS for services outside loopback. Endpoint/model/query text and provider errors are never included in client-facing errors.

`QDRANT_API_KEY` is the default server key environment variable (`api_key_env` overrides it). Environment variables supply secrets; config does not expand variables or run code. Qdrant server mode uses qdrant-client's HTTP API. No live remote Qdrant or paid embedding service is required by tests.

## Filtering and scores

[docs/dify-bridge/filtering.md](dify-bridge/filtering.md) defines all 18 operators, type and missing-value behavior, date interpretation, native Qdrant translations, and the bounded post-filter algorithm. Logical operators are `and` (default) and `or`; an empty conditions list means no filtering. Unknown operators, fields outside the allowlist, malformed values and nested condition groups are rejected with HTTP 400.

| Backend/distance | Normalized score |
| --- | --- |
| Cosine | `clamp((raw + 1) / 2, 0, 1)` |
| Dot | `sigmoid(raw / score_scale)` |
| Euclidean | `1 / (1 + raw / score_scale)` |
| BM25 | `raw / (raw + score_scale)` |

`score_scale` is positive, defaults to 1, and affects dot/Euclidean/BM25 only. These mappings preserve retrieval order; calibration across different corpora/backends is outside scope. Cosine zero similarity maps to 0.5, so a threshold above 0.5 is often useful. Qdrant ties may vary between queries. JSONL uses Unicode word tokens, case folding, BM25 with k1=1.5/b=0.75, unique query terms, and stable file-order ties. It omits documents with zero lexical matches even at threshold zero. No stemming, language-specific segmentation or query expansion.

Filtering and thresholding precede final `top_k`. Qdrant requests pages of at least 16 or `top_k * overfetch` candidates (default overfetch=4), stopping at enough eligible results, an exhausted source, the threshold, or `max_candidates` (default 10,000). Exceeding the budget returns **422/2003**, with no silently incomplete result set. Native Qdrant search may itself be approximate; the bridge cannot promise exhaustive nearest-neighbor recall. Collection mutations during paginated retrieval can change results; use a stable index for repeatable runs.

## Diagnose a query and filter

Send the same Dify request to authenticated `POST /retrieval/diagnostics` to get `records` plus a request-local `diagnostics` object. This is a bridge extension for troubleshooting; configure Dify with the regular base endpoint as before. Both routes share validation, filtering, scoring and bounded pagination. Diagnostics do not issue additional backend requests.

With the example JSONL server running:

```bash
curl -sS http://127.0.0.1:8765/retrieval/diagnostics \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-demo' \
  --data-binary @examples/dify-bridge/request.json
```

The input is the existing `vacation policy` query, `top_k=3`, threshold `0.2`, and `lang is en`. Expect the same `handbook.md` and `leave.md` records as `/retrieval`, with this additional object:

```json
{"backend":"jsonl","query_characters":15,"top_k":3,"score_threshold":0.2,
 "filter":{"logical_operator":"and","condition_count":1,"native_condition_count":0},
 "candidate_budget":null,"page_size":null,"pages_fetched":0,
 "candidates_fetched":4,"candidates_evaluated":4,"filter_rejected":1,
 "threshold_rejected":0,"lexical_misses":1,"eligible_count":2,"returned_count":2,
 "budget_probe_points":0,"stop_reason":"corpus_scanned","snapshot_version":1}
```

Counts describe work actually performed, with these limits:

| Field | Meaning |
| --- | --- |
| `filter.native_condition_count` | Clauses pushed to Qdrant; AND may push a subset, OR needs all clauses translatable. Every candidate still receives the full Python predicate. JSONL reports zero. |
| `pages_fetched`, `page_size` | Qdrant data-page calls (including empty pages) and maximum page size. Excludes schema checks and the final budget probe. JSONL uses zero/null. |
| `candidates_fetched` | Points delivered by Qdrant, including the budget probe; JSONL reports the captured corpus size. Excludes points eliminated by native filtering. |
| `candidates_evaluated` | Candidates visited by the bridge loop. May be less than fetched when top_k/threshold stops within a page; excludes the budget probe. |
| `filter_rejected`, `threshold_rejected`, `lexical_misses` | Rejections among evaluated candidates. Qdrant checks the threshold first and stops at its first failure; JSONL checks filters first, then lexical matches and threshold. These are not totals for unseen candidates. |
| `eligible_count`, `returned_count` | Candidates accepted during the scan and records actually returned. JSONL can find more than top_k. A budget error returns zero records even if some candidates passed. |
| `budget_probe_points` | Zero or one point from the existing final Qdrant exhaustion probe. At most max_candidates + 1 points are fetched. |
| `stop_reason` | `top_k`, `source_exhausted`, `score_threshold`, `candidate_budget`, or JSONL's `corpus_scanned`. Top_k does not imply an exhaustive scan. |
| `snapshot_version` | JSONL only: the exact version captured by this retrieval, including during concurrent reload. |

When the candidate budget runs out, this route returns HTTP **422/2003**, the usual generic error and `diagnostics.stop_reason="candidate_budget"`, with no `records` or partial results. Other failures return the usual structured error without diagnostics. Missing/duplicate/malformed auth returns 401; a wrong key returns 403 with `{"error_code":1002,"error_msg":"Authorization failed"}`. Neither auth failures nor access logs echo credentials, query text or filter values. The diagnostic object includes query length and filter counts, but no raw query, condition names/values, source paths, URLs, embeddings or provider errors. Successful `records` still contain the requested source content and allowlisted metadata. Counters are isolated per request; there is no history, timing profiler or estimate of approximate-search recall.

For a reproducible Qdrant pagination case, run `python examples/dify-bridge/diagnostics_demo.py`. It starts the installed CLI with disposable synthetic named vectors and sends `vacation policy`, `top_k=1`, threshold zero, `team start with keep`. Only candidate 31 matches: expect two pages, 32 fetched, 31 evaluated and 30 filter rejections. Its second configured knowledge ID has a 16-candidate budget: expect 422 with 17 fetched (including the one-point probe), 16 evaluated and no partial records. It also checks redacted wrong-key failure and the ordinary `/retrieval` response.

## Contract checker and errors

`check` sends read-only retrieval probes plus intentionally invalid auth/requests. It checks response shape, non-null metadata, descending finite 0–1 scores, top_k, thresholds and structured errors. It exits 0 for passing checks, 1 for conformance failures, 2 for configuration failures, and writes a JSON report. Give it a known-hit query: empty baseline results fail by default to avoid vacuous success. `--allow-empty` explicitly weakens that check and records the limitation. `--key-env` chooses the API-key variable; no key CLI argument is provided. Checker redirects are rejected. The checker tests a strict full-record profile; Dify itself tolerates some omitted fields. Black-box probes do not prove relevance or every implementation's filter semantics.

Every checker response, including HTTP auth/error responses, must have JSON content type and a body of at most 8 MiB. Like `snapshot-status`, the checker allows an absent `Content-Length`; if present, exactly one header value must contain only ASCII decimal digits and equal the number of body bytes read. Duplicate values (even equal ones), comma lists, signs, non-ASCII/non-decimal values and length mismatches fail conformance. Early EOF fails even when the received prefix is complete valid JSON. These remote failures exit 1 with the existing redacted 13-probe report, empty stderr, and a generic failed-request message for each affected probe. Valid framing preserves exit 0. The bound is enforced with one read of at most 8 MiB plus one overflow byte. Checks use the body delivered by Python's HTTP client; they do not read beyond its message boundary. Both clients accept `Transfer-Encoding` only as one case-insensitive bare `chunked` value, with no `Content-Length` header at all (even empty). Duplicate transfer declarations, comma lists, parameters, trailing whitespace and other codings (including `identity`) are rejected before reading the body. This strict policy uses header values delivered by Python's HTTP parser, which removes leading header whitespace. Unambiguous chunked bodies and bodies with neither framing header remain supported; byte limits apply to the decoded body. The checker applies this policy to success and HTTP error responses; `snapshot-status` rejects every HTTP error and returns only generic failure stderr with empty stdout. The timeout is per socket operation, not a total wall-clock deadline.

Both clients require exactly one `Content-Type` header declaring `application/json` (case-insensitive). Equal or conflicting duplicates, missing/empty types, other media types and comma-separated lists are rejected before body reads. Optional parameters must use `; name=value` with ASCII token names and token or double-quoted values; spaces/tabs around the type and parameter delimiters are allowed. For example, `Application/JSON; Charset="UTF-8"` and `application/json; profile="a,b"` are accepted. Commas/semicolons and backslash-escaped characters inside quoted parameter values are supported; commas outside those quotes, bare/empty parameters and unterminated quotes fail. Parameters do not select a body decoder. The checker applies the same policy to HTTP success and error probes, so an ambiguous auth/error response cannot count as conformance. Failures retain the exit-1/redaction behavior above; local preflight remains exit 2 before HTTP. This client policy does not change the bridge server or embedding transport.

Both clients accept `Content-Encoding` only when absent or when exactly one header contains the bare case-insensitive value `identity`. They never decompress response bodies. Empty declarations, unsupported codings (including `gzip`, `br` and `deflate`), equal/conflicting duplicates, comma lists, parameters and trailing whitespace are rejected before body reads. As with transfer framing, this policy uses header values delivered by Python's HTTP parser, which removes leading header whitespace. Even plain valid JSON falsely declaring a compressed coding fails. The policy covers checker success and HTTP-error probes; failures retain the generic exit-1 reports/errors and redaction above. `snapshot-status` continues to reject every HTTP error without reading its body. Local preflight remains exit 2 before HTTP; server and embedding transports are unchanged.

Run `python examples/dify-bridge/content_encoding_demo.py` for absent/identity success and unsupported/ambiguous coding failures in both checker modes and `snapshot-status`, including HTTP-error-only declarations. It also verifies request bytes, filters/assertions, named-key isolation, redaction, preflight with zero requests and unchanged files/snapshot.

Both clients require strict UTF-8 JSON response bytes without a byte-order mark (BOM), regardless of media-type parameters. UTF-16/UTF-32 (either byte order, with or without BOM), a UTF-8 BOM, malformed UTF-8 and UTF-8 encodings of surrogate code points are rejected. No replacement decoding is performed; a correctly encoded literal U+FFFD remains valid text. Duplicate JSON keys and non-finite constants remain invalid. This also applies to checker HTTP-error probes: valid error fields in a non-UTF-8 body cannot count as conformance. Checker failures exit 1 with the same redacted 13-probe report and empty stderr; snapshot-status failures exit 1 with empty stdout and generic stderr. Local preflight remains exit 2 before HTTP. This response policy does not change server, embedder, corpus or saved-request decoding.

The checker also validates every decoded JSON string and object key, including nested records/metadata, error fields and otherwise ignored fields. Unpaired high/low surrogate escapes (for example, `\ud800` or `\udfff`) fail with the same generic redacted exit-1 report, including on HTTP errors. Valid supplementary-character escape pairs, literal non-ASCII/U+FFFD and text containing a literal backslash followed by `u` remain accepted. This response validation is checker-specific; shared JSON, server, embedder and corpus semantics are unchanged. Request preflight is described below.

Run `python examples/dify-bridge/response_unicode_demo.py` for lone-surrogate failures in both checker modes, including error-only responses, nested values and keys, plus paired escapes/literal Unicode successes. It also checks exact saved-request bytes, filters/assertions, named keys, redaction, preflight with zero requests and unchanged files/snapshot.

Every decoded checker response number must also be finite, including numbers in nested metadata/lists and otherwise ignored success/error fields. JSON exponents such as `1e999` and `-1e999` that overflow to infinity fail with the same generic redacted exit-1 report and empty stderr. Finite floats (including subnormals and underflow to zero), integers, booleans and null remain accepted in those fields; existing record schema/score checks still apply. This is a checker response policy only; shared JSON, saved-request, server, embedder, corpus and snapshot-status semantics are unchanged.

Run `python examples/dify-bridge/response_numbers_demo.py` for overflow failures in both checker modes, including HTTP-error-only responses, nested lists and record metadata, plus finite-number/integer/boolean/null successes. It verifies exact saved-request bytes, filters/assertions, named keys, redaction, local preflight with zero requests and unchanged files/snapshot.

Run `python examples/dify-bridge/response_encoding_demo.py` for valid non-ASCII UTF-8, BOM/UTF-16/UTF-32, malformed-byte and strict-JSON loopback cases in both checker modes and snapshot-status, including error-only bad encodings. It also checks byte-exact saved requests, filters/assertions, named keys, redaction, local preflight and unchanged files/snapshot.

Run `python examples/dify-bridge/content_type_demo.py` to exercise that media-type contract in both checker modes and `snapshot-status` on synthetic loopback, including HTTP-error-only ambiguity, baseline assertions, preflight with zero requests and unchanged files/snapshot state.

Run `python examples/dify-bridge/checker_framing_demo.py` to exercise valid, duplicate, conflicting, invalid and truncated responses on a synthetic loopback endpoint with a `/bridge/` prefix. Both inline and saved-request modes preserve baseline bytes, filters/assertions, named-key isolation, redaction, local exit-2 preflight and unchanged files/snapshot state.

Run `python examples/dify-bridge/transfer_framing_demo.py` for the transfer policy in both checker modes and `snapshot-status`: valid length/absent/chunked framing, conflicting length, duplicate/unsupported transfer declarations, error-only failures and local preflight. It records each CLI command, exit code and output, and verifies baseline bytes, filters/assertions, named-key isolation and unchanged source/snapshot state.

In both modes, `--endpoint` follows the same preflight as `snapshot-status`: a visible ASCII HTTP(S) base URL with a hostname (letters/digits/dots/hyphens or bracketed IPv6) and an optional port in 1–65535. Optional path prefixes and percent-encoded paths are preserved; trailing slashes are removed before appending `/retrieval`. Raw whitespace, controls, DEL, non-ASCII characters, backslashes, credentials (including empty `@`), and query/fragment delimiters (including empty `?` or `#`) are rejected. These local URL failures exit 2 **before any probes**, with empty stdout and a redacted configuration error on stderr. Library checker entrypoints raise `ConfigError` for the same URLs. A valid URL that is unreachable or fails HTTP checks retains the 13-probe report and exit 1. `--timeout` defaults to 10 seconds and must be finite in (0,120].

The public library entrypoints `ragwrench.difybridge.check.check_endpoint` and `check_request_file` enforce the same timeout range before any probes, including auth probes. Their `timeout` must be an exact built-in `int` or `float`, finite and greater than zero through 120 seconds inclusive (default 10). Fractional values such as `0.25` are forwarded unchanged. Booleans, numeric strings, bytes, containers, `None`, numeric subclasses and other numeric/coercible objects (including `Decimal` and `Fraction`) raise `ConfigError`; values are neither coerced nor included in errors. CLI `--timeout` continues to parse numeric text and uses the existing exits 0/1/2. This is a per-socket-operation timeout, not a total deadline for the 13 probes or local file reads.

Run `python examples/dify-bridge/checker_timeout_demo.py` for library timeout rejection with zero requests, valid fractional/upper-bound 13-probe reports, and CLI exit 0/1/2 cases in both modes. It checks exact saved bytes, filters/assertions, named-key isolation and unchanged files/snapshot.

In both inline and saved-request modes, `--key-env` must match `[A-Za-z_][A-Za-z0-9_]*` (default `BRIDGE_API_KEY`). Only that variable is read; there is no fallback. Its key must contain 1–4096 visible ASCII characters (33–126), with no whitespace or control characters, matching `snapshot-status`. Invalid names, unset/empty keys and invalid key characters/lengths exit 2 before **any** probes, including auth probes; stdout stays empty and stderr contains only a redacted configuration error. A syntactically valid but wrong key still makes the 13-probe JSON report fail with exit 1. Library checker entrypoints also reject invalid explicit keys with `ConfigError` before probes. No key values or environment names are included in errors.

```bash
BRIDGE_API_KEY=local-demo ragwrench dify-bridge check \
  --endpoint http://127.0.0.1:8765 --knowledge-id handbook \
  --query 'vacation policy' --top-k 3 --score-threshold 0.2 \
  --metadata-condition '{"logical_operator":"and","conditions":[{"name":"lang","comparison_operator":"is","value":"en"}]}' \
  --expect-count 2 --expect-title handbook.md --expect-title leave.md
```

Use a complete saved request to avoid shell argument limits and copying out settings or filters:

```bash
BRIDGE_API_KEY=local-demo ragwrench dify-bridge check \
  --endpoint http://127.0.0.1:8765 --request-file examples/dify-bridge/request.json \
  --expect-count 2 --expect-title handbook.md --expect-title leave.md
```

`--request-file` accepts a UTF-8 JSON file of at most 1 MiB (including whitespace). It requires exactly `knowledge_id` (string, 1–256 characters), `query` (nonblank string, at most 16384 characters), and `retrieval_setting` (object with exactly integer `top_k` in 1–100 and finite numeric `score_threshold` in 0–1; booleans are not numbers). The only optional field is a non-null `metadata_condition` object using the same flat filter schema and limits described below. No defaults or extra fields are accepted in a saved request. Duplicate keys at any depth, non-finite values (including overflowing exponents), invalid UTF-8 and malformed JSON are rejected.

File mode is mutually exclusive with every inline request flag: `--knowledge-id`, `--query`, `--metadata-condition`, `--top-k`, and `--score-threshold`, even if an explicit value equals the default. Inline mode still requires both knowledge ID and query. Authentication, endpoint, timeout and baseline assertion flags work in both modes. The file is opened once and read with a bound of 1 MiB plus one overflow byte; the baseline POST sends the original bytes unchanged. Applicable later probes retain its filter while deliberately changing settings/auth/errors; generated JSON uses UTF-8 rather than expanding non-ASCII characters into ASCII escapes. Missing/unreadable/oversized files, JSON/schema errors, conflicts and invalid assertions exit 2 before any probes, with empty stdout and a generic stderr configuration message that never includes file paths or contents. Argument parsing errors for `check` are also redacted. HTTP/response/assertion failures retain exit 1 and the same 13-probe redacted JSON report; success exits 0. No source file is changed.

`--metadata-condition` takes one JSON object in Dify's `metadata_condition` format, not a full request or a filename. Duplicate keys, null, non-finite values, malformed groups/operators/typed values and more than 64 clauses are local configuration errors. The checker validates the filter structure using the bridge's documented operators; the endpoint decides which fields are allowlisted. Omit the option to omit the field entirely. Query settings default to `--top-k 3 --score-threshold 0`; their bounds are 1–100 and finite 0–1. The complete baseline request is limited to 1 MiB.

Checker request strings and object keys must contain only Unicode scalar characters, including knowledge IDs, queries, nested filter values (even values ignored by an operator) and expected titles. Unpaired high/low surrogate escapes are rejected locally before **any** HTTP probes: CLI exit 2, empty stdout and a generic redacted configuration error; library entrypoints raise `ConfigError`. Valid supplementary escape pairs, literal Unicode including U+FFFD, and literal backslash-u text remain accepted without normalization. Saved baseline bytes, including escape spelling, whitespace and number spelling, are sent unchanged. This policy is checker-specific; the shared decoder, server, embedding and corpus readers are unchanged.

Run `python examples/dify-bridge/request_unicode_demo.py` for inline/file surrogate preflight, valid Unicode and byte-exact saved requests, assertion failures, named-key isolation, redaction and unchanged files/snapshot.

The first probe (`retrieval shape and ordering`) sends exactly these query settings and the supplied filter. `--expect-count N` requires exactly N baseline records (0–top_k); repeatable `--expect-title TITLE` requires each distinct title to appear, case-sensitively, in any order. Presence does not assert title uniqueness, multiplicity or rank. Expectations apply only to this baseline: the top_k=1 and threshold=0.9/1 probes deliberately change settings, retaining the filter. The remaining auth/error probes also retain the filter where applicable; the invalid-operator probe replaces it intentionally. The checker only calls `/retrieval` and does not reload or mutate data.

With the committed corpus the command exits 0 with `{"passed":true,"checks":[...],"limitations":[...]}` (13 passing checks). A wrong count or missing title makes the baseline check fail, prints the same report shape with `passed:false`, and exits 1. HTTP failures (including an endpoint rejecting a filter field), malformed responses, and scan-budget errors also fail conformance. `--expect-count 0` explicitly permits an empty baseline; `--allow-empty` never disables explicit count/title assertions. Contradictory expectations (count above top_k or fewer records than distinct required titles) exit 2. Other invalid local inputs exit 2 before any probes, with a generic stderr message and no JSON report. Reports and configuration errors do not echo query/filter values, expected or returned titles, credentials or upstream body text. Expectations test only the observed records of these finite requests; they do not prove every filter's semantics, relevance, exhaustive recall or stability during source updates.

Run `python examples/dify-bridge/checker_demo.py` for the inline checker example. Run `python examples/dify-bridge/request_file_demo.py` for saved-request success, exact baseline bytes, preserved filters, count/title failures, explicit zero hits, and missing/malformed/oversized/duplicate-key/conflicting inputs that send no probes. It also checks credential preflight in both request modes: invalid names and missing/empty/unsafe/oversized keys send no probes, while valid wrong keys exit 1. Invalid URLs (including leading whitespace, invalid ports, controls and non-ASCII) also exit 2 without probes in both modes. The demo logs commands, exits, observed request counts and redacted CLI output, and checks that corpus/config/request files and the active snapshot remain unchanged.

Library users can pass `metadata_condition` as a dict, `top_k`, `score_threshold`, `expect_count` and `expect_titles` (list/tuple) to `check_endpoint()`; omitted/None metadata conditions are not sent. `check_request_file(endpoint, key, path, ...)` accepts the same assertion/timeout/empty options and raises `ConfigError` for local file/schema failures. The report format is unchanged, with an additional limitation when an empty baseline is expected.

All application errors contain integer `error_code` and string `error_msg`. Backend failures do not expose upstream errors or keys.

| HTTP / application code | Meaning |
| --- | --- |
| 401 / 1001 | Missing/malformed Bearer header (duplicates rejected) |
| 403 / 1002 | Wrong key; constant-time token comparison |
| 404 / 2001 | Unknown knowledge ID |
| 400, 413, 415, 408 / 2002 | Invalid request, oversized body, content type or read timeout |
| 422 / 2003 | Candidate budget exhausted |
| 404, 405, 501 / 2004 | Unknown route or unsupported method |
| 502, 503 / 3001 | Invalid backend payload/vector/config mapping or unavailable backend |
| 500 / 3002 | Unexpected internal error |
| 503 / 3003 | All request slots busy |

`GET /healthz` is unauthenticated liveness only; it does not probe source readiness. Other requests require a visible ASCII Bearer key from `BRIDGE_API_KEY`. The service defaults to loopback, with a 1 MiB body limit, a 10 second socket read timeout, at most 16 concurrent handlers, at most 64 filter clauses, query length 16,384, knowledge ID length 256, and top_k 1–100. JSON duplicate keys, non-finite scores, unknown request/config fields and chunked request bodies are rejected. JSONL startup/reload limits are 64 MiB, 100,000 records and 1 MiB per line (bytes, including the newline); malformed lines are never skipped. Files are loaded at startup and on explicit `/reload` requests. Qdrant timeout is per SDK request; pagination can involve multiple requests.

This is a small trusted-network adapter, with no TLS termination, rate limiting per tenant, tenant isolation beyond configured knowledge IDs, ingestion, plugin packaging, RAGFlow passthrough or workflow execution. All knowledge IDs share the configured bridge key. Put an authenticated TLS proxy in front of it when appropriate; production deployment is outside this repository's demo.

## Offline validation

```bash
pip install -e '.[qdrant,test]'   # from a clone of the repository
python -m pytest tests/difybridge -q
python examples/dify-bridge/demo.py
python examples/dify-bridge/reload_demo.py
python examples/dify-bridge/diagnostics_demo.py
python examples/dify-bridge/checker_demo.py
python examples/dify-bridge/request_file_demo.py
```

The Qdrant demo creates a disposable directory from the committed synthetic JSONL, indexes named vectors, starts the bridge on an OS-assigned loopback port, runs HTTP retrieval and conformance probes, then shuts down and removes its temporary data. The reload demo launches the installed `ragwrench dify-bridge` CLI with a temporary JSONL corpus and checks publication, malformed-update retention, version progression, empty-corpus clearing and read-only snapshot status even with a missing source file. The diagnostics demo exercises paginated filtering and explicit budget failure through the installed CLI. All HTTP in tests and demos stays on loopback; no model, download or container is required.

Independent implementation of the [documented Dify
contract](https://docs.dify.ai/en/cloud/use-dify/knowledge/external-knowledge-api); no
Dify or RAGFlow source is copied, and this project is not affiliated with them.
