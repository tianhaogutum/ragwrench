# ragwrench migrate

Moving a collection between Chroma and Qdrant usually means re-embedding everything or
writing a one-off script, with no evidence that IDs, metadata and search results
survived. `ragwrench migrate` copies Chroma and Qdrant local collections in either
direction, keeping the stored vectors, and then verifies record and top-k retrieval parity.
Use it when switching vector stores, or to check that two stores hold the same data.

_Formerly the standalone `vector-migration-parity` tool (`vecparity`)._

Copy **Chroma PersistentClient ↔ Qdrant local-mode** collections, then prove record and
retrieval parity without a server, API key, model download, or re-embedding. Original
implementation, MIT licensed.

## Quick start

Needs the `chroma` extra (chromadb and qdrant-client) on Linux. chromadb does not support
Python 3.14 yet, so `ragwrench migrate` needs Python 3.11–3.13.

```bash
pip install 'ragwrench[chroma]'
ragwrench migrate --help
```

From a clone of the repository, create a new synthetic, redistributable fixture directory
(the example refuses to overwrite an existing directory):

```bash
python examples/migrate/make_fixture.py /tmp/vp-demo
ragwrench migrate inspect 'chroma:/tmp/vp-demo/chroma#docs'
ragwrench migrate copy 'chroma:/tmp/vp-demo/chroma#docs' 'qdrant:/tmp/vp-demo/qdrant#docs' --dry-run
ragwrench migrate copy 'chroma:/tmp/vp-demo/chroma#docs' 'qdrant:/tmp/vp-demo/qdrant#docs' --batch-size 2
ragwrench migrate verify 'chroma:/tmp/vp-demo/chroma#docs' 'qdrant:/tmp/vp-demo/qdrant#docs' --sample-queries 20 --top-k 2
ragwrench migrate copy 'qdrant:/tmp/vp-demo/qdrant#docs' 'chroma:/tmp/vp-demo/restored#docs'
ragwrench migrate verify 'chroma:/tmp/vp-demo/chroma#docs' 'chroma:/tmp/vp-demo/restored#docs' --top-k 2
```

Expected forward copy: `copied 5 points; dimension=4 distance=cosine`, with three
UUIDv5 mappings, one null metadata record, and one null document. Verification
reports `count OK`, `ids OK`, zero field mismatches, top-2 mean overlap `1.0`,
and `PARITY OK`. These are real persisted collections, not mock adapters.

The module entry point `python -m ragwrench.migrate` is equivalent.

## Inputs, outputs, and exit codes

URIs have the form `chroma:PATH#collection` or `qdrant:PATH#collection`.
Paths are local, absolute or relative to the current directory; quote the URI.
There is no URL decoding. Collection names must be 3–128 ASCII letters, digits,
`.`, `_`, or `-`, with alphanumeric ends and no `..`. This deliberately excludes
path traversal and some names accepted by upstream clients. Stores must be in
separate, non-nested directories. Chroma uses its default tenant/database.
Symlinks are rejected in the store path, every existing ancestor, and inside
the store, including dangling links and links in paths containing `..`. This
applies to both copy directions, dry-runs, and read commands. Relative paths
and `..` are canonicalized only after checking for links; alternate spellings
of the same or nested source/target directory are refused before opening stores.
Python callers using `engine.copy`, `engine.inspect`, or `engine.verify` receive
the same path checks, including for directly constructed `StoreURI` objects.

Backend kind and collection name are validated when a `StoreURI` is constructed,
including through `dataclasses.replace`. Unsupported kinds and invalid/non-string
names raise `ParityError` before any snapshot or backend access. Path validation
still runs when parsing a URI or accessing a store. For example:

```python
from pathlib import Path
from ragwrench.migrate.model import ParityError, StoreURI

try:
    StoreURI("qdrant", Path("./new-store"), "../outside")
except ParityError as exc:
    print(exc)  # Collection name must be 3–128 ASCII letters/digits/._-, with alphanumeric ends
```

This creates no directory and opens no backend. The same refusal applies to
Chroma and to source and target names. CLI input such as
`ragwrench migrate inspect 'qdrant:./new-store#../outside' --json` exits **2**, with empty
stdout and a JSON `error` on stderr; it also creates no directory.

Direct `StoreURI` paths must be `pathlib.Path` objects (including subclasses).
Strings, bytes, `None`, `PurePath` and other path-like objects raise
`ParityError("Store path must be a local pathlib.Path object")` at construction
or `dataclasses.replace`, before coercion, filesystem probes, snapshots or backend
access. For example, use `StoreURI("qdrant", Path("./store"), "docs")` rather than
passing `"./store"` directly. The shared `checked_store_path` checker enforces
the same contract. CLI URI strings continue to be parsed into `Path` objects.

Paths must not contain embedded NUL bytes.
For example, `StoreURI("qdrant", Path("invalid\x00path"), "docs")` raises
`ParityError` containing `local path without NUL bytes` at construction, also
through `dataclasses.replace`, before filesystem, snapshot, or backend access.
The shared path checker rejects NUL before filesystem access as well.
Filesystem-dependent checks still run at access, so missing paths can be used
for new stores. The CLI parser already refuses NUL with exit 2, empty stdout,
and a JSON `error` on stderr when called with `--json` through Python; an OS
process argument cannot contain NUL.

Paths beginning with `~` or `~username` must have a resolvable home directory.
For example, accessing `StoreURI("qdrant", Path("~no_such_user/store"), "docs")`
when that user does not exist raises `ParityError` containing
`local path with a resolvable home directory` before snapshot or backend access.
This filesystem-dependent check runs at access, including direct `Store` entry.
The equivalent CLI URI exits **2**, with empty stdout and the specific local-path
`error` on JSON stderr. Valid home paths, relative paths and missing target
directories remain supported.

Paths must also be encodable by Python's filesystem codec and error handler.
For example, `StoreURI("qdrant", Path("invalid\ud800path"), "docs")` raises
`ParityError` containing `local path encodable by the filesystem` at construction
or replacement, before filesystem, snapshot or backend access. Expanded home
paths are checked at access too. The equivalent Python CLI `main([...])` call
exits **2**, with empty stdout and that specific JSON `error` on stderr; OS argv
cannot carry this high surrogate. Ordinary Unicode paths and Linux
surrogateescape filename bytes remain valid path inputs, including snapshot
reads from stores in such directories.

| Command | Behavior |
| --- | --- |
| `copy SOURCE TARGET` | Scan and validate the full source and existing target; then copy in paginated batches (default 256, maximum 4096) |
| `copy SOURCE TARGET --checkpoint FILE [--resume]` | Save acknowledged IDs and stable mappings; resume remaining writes and automatically verify parity |
| `copy SOURCE TARGET --dry-run` | Perform the same preflight on temporary snapshots; create no target and change no original store files |
| `verify SOURCE TARGET` | Check collection schema, counts, logical ID sets, physical ID mapping, vectors, metadata, documents, and actual backend top-k queries |
| `inspect SOURCE` | Report count, dimension, distance, logical metadata/payload key types, document count, and null metadata record count |

All commands support `--json` for a JSON report on stdout. Operational errors go
to stderr; with `--json` they are JSON objects containing `error`. Argument parsing
errors use argparse usage/error text on stderr, including with `--json`.
The default verification
output is readable text. For example:

```bash
ragwrench migrate verify 'chroma:./chroma_db#docs' 'qdrant:./qdrant_db#docs' --json > parity.json
```

Exit **0** means successful inspect/copy, valid dry-run, or parity passed.
Exit **1** means verification (including automatic checkpoint parity) found a mismatch. Exit **2** means invalid input,
unsupported format, unsafe mapping, or backend failure. `copy` success means
records were written; run `verify` to establish parity, or use `--checkpoint`
for automatic verification. Reports cap ID examples
at 20 per mismatch category while retaining complete mismatch counts. Record
contents are not printed on backend exceptions.

Python `engine.copy(..., top_k=...)` and `engine.verify(..., top_k=...)` require
a positive `int`, including integer subclasses, and reject booleans. Non-integers
(even `1.0`), zero and negative values raise
`ParityError("top_k must be a positive integer")` before checkpoint path probes,
loading/saving, store path checks, snapshots or backend access. For example,
`engine.copy(source, target, checkpoint=new_path, top_k=1.5)` leaves a fresh
target and checkpoint absent; existing trees remain unchanged. This validation
also applies to copies without checkpoints, dry-runs and verification with
`sample_queries=0`. The default remains 10. Python has no upper cutoff bound;
retrieval uses the smaller of `top_k` and the available record count.

CLI `copy` and `verify` already parse `--top-k` as an integer in **1–4096**.
For example, `--top-k 2` is valid and `--top-k 1.5 --json` exits **2** with empty
stdout and argparse usage/error text on stderr (`expected an integer`), before
URI parsing or store/checkpoint access. Fractional argv is not a JSON API error.

For example, after creating the demo above, this target is refused:

```bash
mkdir /tmp/vp-demo/protected-target
ln -s /tmp/vp-demo/protected-target /tmp/vp-demo/target-link
ragwrench migrate copy 'chroma:/tmp/vp-demo/chroma#docs' 'qdrant:/tmp/vp-demo/target-link#docs' --json
```

Expected: exit **2**, no stdout, and a JSON error on stderr containing `symlink`.
`protected-target` stays empty, the link stays in place, and the source stays
unchanged. A missing target below a symlinked parent is also refused, including
with `--dry-run`. Use an ordinary, separate directory as the target instead.

## Checkpointed migration and recovery

Use a new checkpoint file outside both store directories; its parent must already
exist. Starting with an existing file is refused unless `--resume` is present.
Python checkpoint inputs must be `str` or concrete `pathlib.Path` objects,
including subclasses; `checkpoint=None` disables checkpointing (`resume=True`
still requires a checkpoint). Bytes, numbers, booleans, containers, standalone
`PurePath` and custom `os.PathLike` objects raise
`ParityError("Checkpoint path must be a local str or pathlib.Path object")`
before conversion hooks, checkpoint probes, loading/saving, snapshots or backend
access, with unchanged trees. This deliberately narrows incidental `Path(value)`
coercion to the annotated API contract. Convert other path representations to
`str` or `Path` explicitly. CLI argv supplies text, so unsupported Python types
have no typed CLI equivalent; a string such as `"42"` is a filename.
Unrelated downstream `TypeError` exceptions retain their original identity.
The entire checkpoint path must be free of embedded NUL, before and after home
expansion. For example, `checkpoint=Path("/existing-parent/invalid\x00progress.json")`
raises `ParityError("Checkpoint path must be a local path without NUL bytes")`
before checkpoint filesystem probes, reads/writes, snapshots or backend access,
leaving the trees unchanged. This applies to both directions, creation and resume,
and both `str` and `Path` inputs. The equivalent Python CLI `main([...])` call with
`--json` returns **2**, with empty stdout and
`{"error": "Checkpoint path must be a local path without NUL bytes"}` on stderr.
OS argv cannot carry NUL, so this CLI input is tested through Python `main`.
The entire expanded checkpoint path, including its final filename, must be
encodable by Python's filesystem codec and error handler. For example,
`checkpoint=Path("/existing-parent/invalid\ud800progress.json")` raises
`ParityError("Checkpoint path must be a local path encodable by the filesystem")`
before checkpoint filesystem probes, reads/writes, snapshots or backend access,
leaving the trees unchanged. This applies to creation and resume, with either
`str` or `Path` inputs. The equivalent Python CLI `main([...])` call returns **2**,
with empty stdout and that specific JSON `error` on stderr when using `--json`.
OS argv cannot carry a high surrogate; ordinary Unicode and Linux surrogateescape
checkpoint filenames remain supported, including creation and resume.
Checkpoint home expansion must also resolve. For example,
`checkpoint=Path("~nonexistent_username/progress.json")` (where that user does
not exist) raises
`ParityError("Checkpoint path must be a local path with a resolvable home directory")`.
This applies to `str` and `Path`, creation and resume, in either direction,
before checkpoint probes, loading/saving, snapshots or backend access. Rejection
leaves the trees unchanged. The equivalent CLI `--checkpoint` argument returns
**2**, with empty stdout and that specific JSON `error` on stderr with `--json`.
Valid `~/` and `~username/` expansion and relative/absolute paths remain supported;
NUL and filesystem-encoding errors retain their specific errors. Only a
`RuntimeError` from home expansion is translated; later runtime errors propagate
through the Python API.
The same interface works in either migration direction:

```bash
ragwrench migrate copy 'chroma:/tmp/vp-demo/chroma#docs' 'qdrant:/tmp/vp-demo/checkpoint-target#docs' --batch-size 2 --checkpoint /tmp/vp-demo/copy.json --top-k 2 --json
# After an interrupted copy, fix the backend/disk problem and resume:
ragwrench migrate copy 'chroma:/tmp/vp-demo/chroma#docs' 'qdrant:/tmp/vp-demo/checkpoint-target#docs' --batch-size 1 --checkpoint /tmp/vp-demo/copy.json --resume --top-k 2 --json
```

Version-1 checkpoint JSON stores canonical source/target identities, schema,
vector selection, a SHA-256 hash per source record, each logical-to-target ID
mapping, acknowledged `completed_ids`, and the latest `parity` report. It stores
IDs and hashes rather than document/metadata/vector contents. Treat it as private
migration state and do not edit it. It is replaced atomically and fsynced before
opening the target and after each acknowledged batch. Only IDs saved after a
successful backend write are skipped. An uncertain or partially written batch is
replayed with the same IDs; batch size and pagination order may change on resume.
The full source is still scanned to validate the manifest.

The nested `manifest.source`, `manifest.target`, `manifest.schema` and
`manifest.records` must each be JSON objects.
Missing, null, array, string, number and boolean values for any of these fields
raise `ParityError("Invalid or unsupported checkpoint")` during loading, before any
snapshot or backend access. With `--resume --json`, the CLI exits **2**, with
empty stdout and exactly `{"error": "Invalid or unsupported checkpoint"}` on
stderr; checkpoint and store trees stay unchanged. This is an object-shape check:
loading does not validate identity fields, schema fields, individual record
entries, hashes or target IDs. Empty and populated objects still face the existing
full manifest comparison during resume.

Checkpoint input must contain finite JSON numbers throughout, including nested
manifest and parity objects/arrays. `NaN`, `Infinity`, `-Infinity` and numeric
overflow such as `1e999` raise `ParityError("Invalid or unsupported checkpoint")`
during loading, before snapshots or backend access. With `--resume --json`, the
CLI exits **2**, with empty stdout and exactly
`{"error": "Invalid or unsupported checkpoint"}` on stderr. Refusal leaves
checkpoint and store trees unchanged. Finite numbers and quoted strings such as
`"NaN"` or `"1e999"` remain valid JSON data, subject to the existing checkpoint
shape, version, duplicate-key and resume identity checks.

Resume refuses changed source contents, schema, mapping, vector selection, or
source/target identity before writes. It also checks acknowledged target records
for missing IDs, mapping changes, documents, metadata, and vectors (absolute
component tolerance `1e-5`, normalized for cosine). If these records changed,
repair the target or choose a **new** checkpoint to recopy. Symlink/special-file
checkpoints and checkpoints inside a store are rejected. `--dry-run` cannot be
combined with checkpointing. Concurrent use of the stores or checkpoint is
unsupported; checkpoints do not provide rollback or a distributed transaction.

After writing, the writer closes and read-only snapshots produce a full parity
report, both returned as `parity` and saved in the checkpoint. Defaults are the
same as `verify`: tolerance `1e-5`, 10 sampled queries, seed 0, and required overlap
1.0; `copy --top-k` changes the automatic retrieval cutoff. `copied` counts writes
in this invocation; `skipped` counts previously acknowledged records. Exit **1**
means parity failed (e.g. retained extra target IDs), while exit **2** means an
input/backend error. A parity error retains progress: `--resume` reruns parity,
even if all records were already copied. Use standalone `verify` for other
sampling/tolerance settings. A checkpoint mismatch refusal leaves the checkpoint
and target unchanged. Backend or checkpoint I/O failures can leave a partial
target with only the last saved ID set acknowledged.

For a reproducible failure example on **new synthetic data**:

```bash
python examples/migrate/resume_failure.py /tmp/vp-resume-demo
```

Expected in **each direction**: a single injected error after the second backend
write returns exit 2; only the first two IDs are acknowledged. A fresh CLI process
resumes with batch size 1 and reports `copied=3`, `skipped=2`, zero field/mapping
mismatches, top-2 overlap `1.0`, and `PARITY OK`. A second resume writes zero
records and reruns parity. The example asserts that all source file hashes remain
unchanged. Failure injection lives only in the example/test harness.

## Reversible mappings

**IDs.** Canonical decimal unsigned 64-bit strings become integer Qdrant IDs.
Valid UUIDs are kept (canonicalized). All other strings become
`uuid5(9b33df64-cd30-52b9-9328-3d8ab7287f76, original_id)`; for example `doc`
maps to `a4345041-1b57-581a-875e-2b79d5cac285`. Leading-zero decimal strings,
negative integers, and integers beyond uint64 use UUIDv5. Every generated Qdrant
record carries the exact original string in `__id__`, including UUID spelling.

Native Qdrant IDs are restored as decimal/UUID Chroma strings; a native payload
`__id__` string takes precedence. The physical Qdrant ID is recorded in the
Chroma mapping envelope, so switching back also preserves native ID types and
IDs chosen by other migration tools. Aliases that collide (e.g. upper/lowercase
UUIDs, or a literal UUID equal to another ID's generated UUIDv5) abort the full
preflight before any target writes.

**Documents.** Chroma documents map to payload `document`. Null/absent documents
remain null, distinct from the empty string. On native Qdrant input, `__id__`
and `document` are transport fields; other payload fields are metadata. A
non-string, non-null `document` or a non-string `__id__` is rejected.

**Metadata.** Scalars retain their types. Nested lists/objects, nulls, and
integers outside signed int64 are JSON-encoded for Chroma. A version-1
`__vecparity__` envelope records each key rename, JSON encoding, null-vs-empty
metadata state, and original physical Qdrant ID. Keys in Chroma are renamed to
`vp_m_<UTF-8 hex>`; Qdrant keeps ordinary payload key names. User metadata keys
`document` and `__id__` are escaped and restored, independently of transport
fields. JSON strings are decoded only when the envelope explicitly says so.
Booleans remain distinct from integers. Null records and top-level null values
are counted in copy reports. Nesting is limited to 32 levels; non-finite numbers,
custom Python objects, tuples, sets, and non-string object keys are rejected.
An existing native `__vecparity__` key is reserved: malformed or unfamiliar
versions fail rather than guessing. Do not remove or manually edit envelopes.

**Distance.** `l2 ↔ Euclid`, `cosine ↔ Cosine`, `ip ↔ Dot`. Chroma's squared L2
and Qdrant's Euclidean distances rank the same way; raw score values are not
compared. Cosine parity compares normalized vectors because stores may normalize
on insertion, so original vector magnitudes are not guaranteed for cosine.
Zero cosine vectors, NaN/Infinity, empty vectors, and dimension mismatches fail.
Other distances and index configuration migration are outside v1.

## Verification details

Every logical record is compared, not just the retrieval sample. Vector parity
uses maximum absolute component difference (`--tolerance 1e-5`), after
normalization for cosine. Metadata equality is type-sensitive canonical JSON;
documents are compared exactly. Schema, missing/extra IDs, physical ID mapping,
and field mismatches are separate report fields.

Queries use source embeddings against **both real backends**, with no embedding
function. Defaults are 10 sampled queries, top-10, seed 0, and required mean
overlap 1.0. The denominator is `min(top_k, source_count)`. The sample is drawn
from sorted IDs with a fixed seed; JSON includes individual overlaps and mean.
Use `--sample-queries 20 --top-k 5 --seed 42` to change the sample. Small fixtures
should use a top-k below their count for a meaningful ranking check.

ANN search and ties near the cutoff can yield different valid neighbors. A
chosen `--min-overlap 0.9` relaxes only retrieval overlap, never record checks.
`--sample-queries 0` explicitly skips retrieval and reports it as unchecked.
Empty source collections cannot establish retrieval parity and fail copy/verify;
`inspect` can still describe them (an empty Chroma dimension may be unknown).

A sole named Qdrant dense vector is supported with `--vector-name dense` on
copy/verify/inspect. The flag also selects the name when creating a Qdrant target.
Multiple named fields, multi-vectors, sparse vectors, sparse-only collections,
Chroma URI/image records, and non-HNSW Chroma indexes fail explicitly.

## Operational boundaries

- **Close other processes using both stores before running.** Snapshots are
  filesystem copies, not transactional online backups. Concurrent writes,
  active SQLite journals, concurrent schema changes, and live synchronization
  are unsupported. Make your own backup before migration.
  Path checks also assume a stable filesystem; replacing paths during a command
  is unsupported.
- All reads, inspection, verification, and dry-runs open disposable snapshots.
  They leave original files unchanged, including backend lock/migration writes.
  Temporary disk must fit both store snapshots. Qdrant loads collections into
  memory; copy adds O(N) ID state, and verify holds O(N × dimension) records
  plus backend memory. Checkpoint resumes also hold acknowledged source records
  in memory and rewrite O(N) manifest/progress JSON after each batch; automatic
  parity needs additional snapshots. This release targets embedded development collections,
  not million-vector production migrations.
- Copy replaces source IDs and retains unrelated target records. Re-running the
  same source is idempotent. Extra target records are reported by verify, never
  silently pruned. Existing incompatible dimension/distance or conflicting
  physical/logical IDs abort preflight. Chroma replaces each copied record so
  null documents do not leave stale text behind.
- Copy is **resumable, not atomic** across databases or batches. Disk/backend
  errors can leave a partial target; Chroma's delete-and-upsert replacement can
  also interrupt within a batch. Rerun the same command after fixing the error,
  then verify, or use `--checkpoint FILE --resume` with the original checkpoint.
  There is no rollback or automatic source deletion.
- Store symlinks/special files are rejected. Qdrant metadata collection names
  are checked before backend loading. Its SQLite point blobs are audited with
  an inert allowlisted unpickler: only modern dense PointStruct plus primitive
  data is accepted in every loaded collection; callable reductions, foreign
  classes, legacy dbm storage,
  and SQLite WAL/journal sidecars are unsupported. The per-point serialized
  Qdrant limit is 64 MiB. This is not a general-purpose untrusted database
  sandbox or a resource quota manager.
  Low-level `Store` contexts run the same root/ancestor and descendant
  symlink/special-file preflight before metadata audit or backend construction.
  Missing paths (including missing parents) and empty directories may initialize
  a new store; nonempty directories must contain the selected backend's marker
  (`chroma.sqlite3` or `meta.json`). Rejected entry leaves the directory and any
  outside link target unchanged. Direct contexts are writable; use engine read
  APIs for disposable snapshots.
- No remote URLs/authentication, models, cloud operations, collection aliases,
  payload indexes, sparse/multi-vector migration, filters, or re-embedding.
  Qdrant aliases are not migrated. Compatible local data formats from the
  pinned clients are tested; arbitrary historical database versions are not.

## Tests, provenance, and upstream relationship

`python -m pytest tests/migrate -q` runs in one process, blocks network socket connections and
uses small real Chroma/Qdrant on-disk stores. It checks bidirectional round trips for all
three distances, codec mappings, corruption detection, checkpoint recovery and path-safety
boundaries; rejected copies must leave the file trees unchanged.
The synthetic fixture at `tests/migrate/fixtures/records.json` is dedicated to the public
domain (CC0-1.0).
The example script disables embedding functions and closes its persistent client.
Stored Chroma embedding-function configurations are read as data and are never
reconstructed, so even collections originally using model/API embeddings can be
read without those credentials or model dependencies.

This tool uses [Chroma](https://github.com/chroma-core/chroma) and
[qdrant-client](https://github.com/qdrant/qdrant-client) as runtime dependencies.
[qdrant/migration](https://github.com/qdrant/migration) was a field-name and workflow
reference; no upstream code was copied. ragwrench migrate adds bidirectional local-store
copying and explicit parity checks; its envelope is its own format.
