# Third-party sources and attribution

ragwrench is licensed under MIT (see [LICENSE](LICENSE)). The commands below were
previously separate MIT-licensed projects by the same authors; their notices are
preserved below, one section per command, with paths updated to this repository's
layout. Dependencies are installed by your package manager under their own licenses
and are never vendored; the ragwrench wheel contains only the `ragwrench` package and
these project notices. License texts that must accompany redistribution are in
[licenses/](licenses/) (currently pypdf's BSD-3-Clause license, used by `ragwrench highlight`).

Dependency versions quoted below are the ones each tool was originally locked and
validated against; ragwrench itself declares version ranges in `pyproject.toml`.
Product names identify interoperable formats and APIs only; ragwrench is not
affiliated with or endorsed by Dify, RAGFlow, LangChain, LlamaIndex, Docling,
Unstructured, Qdrant, Chroma, or any other project named here.

## At a glance

### Runtime dependencies (installed separately, never vendored)

| Package | License | Installed by | Used by |
| --- | --- | --- | --- |
| [jsonschema](https://github.com/python-jsonschema/jsonschema) (with attrs, referencing, rpds-py, jsonschema-specifications) | MIT | core | `lint` fix-plan schema, `port` bundle schema |
| [pypdf](https://github.com/py-pdf/pypdf) | BSD-3-Clause ([text](licenses/pypdf-BSD-3-Clause.txt)) | core | `highlight` |
| [httpx](https://github.com/encode/httpx) (with httpcore, h11, anyio, idna, certifi) | BSD-3-Clause (httpcore, idna: BSD-3-Clause; h11, anyio: MIT; certifi: MPL-2.0) | core | `port` |
| [qdrant-client](https://github.com/qdrant/qdrant-client) (with numpy, grpcio, protobuf, pydantic, portalocker, urllib3) | Apache-2.0 (grpcio: Apache-2.0; numpy, protobuf, portalocker: BSD-3-Clause, numpy wheels add bundled notices; pydantic, urllib3: MIT) | `qdrant`, `chroma` extras | `sync`, `bm25`, `migrate`, `dify-bridge` Qdrant backend |
| [chromadb](https://github.com/chroma-core/chroma) and its dependency tree | Apache-2.0 (dependencies under their own licenses; see the `migrate` section) | `chroma` extra (Python < 3.14) | `migrate` |
| [jieba](https://github.com/fxsjy/jieba) | MIT | `jieba` extra | `bm25 --tokenizer jieba` (uses jieba's own bundled dictionary; none is copied here) |
| [tiktoken](https://github.com/openai/tiktoken) | MIT | `tiktoken` extra | optional `table-chunk` token counter |

Test-only dependencies (`test` extra) are [pytest](https://github.com/pytest-dev/pytest) (MIT) and its
dependencies (iniconfig, pluggy: MIT; packaging: Apache-2.0 OR BSD-2-Clause; Pygments: BSD-2-Clause).
Build-only tools are [hatchling](https://github.com/pypa/hatch) (MIT) and
[hatch-fancy-pypi-readme](https://github.com/hynek/hatch-fancy-pypi-readme) (MIT).

### Bundled code and data

- **No third-party source code is vendored.** Every module under `src/ragwrench/` is original MIT work.
- **No third-party dictionaries, models or datasets are bundled.** The CJK tokenizers in
  `ragwrench bm25` are rule-based (character n-grams); dictionary segmentation is delegated to
  the optional, separately installed jieba package.
- Packaged data files are original synthetic fixtures under the root MIT license:
  `src/ragwrench/lint/data/` (demo chunks and Markdown source), `src/ragwrench/lint/schemas/`
  (fix-plan JSON Schema), `src/ragwrench/bm25/data/` (synthetic zh/ja/mixed corpus and queries),
  `src/ragwrench/citecheck/data/benchmark.jsonl` (labelled synthetic citation cases) and
  `src/ragwrench/port/bundle.schema.json` (bundle JSON Schema).
- `tests/migrate/fixtures/records.json` is dedicated to the public domain (CC0-1.0); all other
  test, example and benchmark fixtures are original synthetic data under MIT.

## Contents

- [`ragwrench lint` (formerly chunk-lint)](#ragwrench-lint-formerly-chunk-lint)
- [`ragwrench table-chunk` (formerly table-chunker)](#ragwrench-table-chunk-formerly-table-chunker)
- [`ragwrench sync` (formerly chunk-sync-planner)](#ragwrench-sync-formerly-chunk-sync-planner)
- [`ragwrench bm25` (formerly qdrant-bm25-cjk)](#ragwrench-bm25-formerly-qdrant-bm25-cjk)
- [`ragwrench migrate` (formerly vector-migration-parity)](#ragwrench-migrate-formerly-vector-migration-parity)
- [`ragwrench cite-check` (formerly rag-cite-check)](#ragwrench-cite-check-formerly-rag-cite-check)
- [`ragwrench eval` (formerly retrieval-regress-gate)](#ragwrench-eval-formerly-retrieval-regress-gate)
- [`ragwrench highlight` (formerly cite-highlight-pdf)](#ragwrench-highlight-formerly-cite-highlight-pdf)
- [`ragwrench port` (formerly kb-bundle-porter)](#ragwrench-port-formerly-kb-bundle-porter)
- [`ragwrench dify-bridge` (formerly dify-external-kb-bridge)](#ragwrench-dify-bridge-formerly-dify-external-kb-bridge)

## `ragwrench lint` (formerly chunk-lint)

chunk-lint's implementation, tests, and synthetic fixtures are original work
under the root MIT LICENSE. No LangChain, LlamaIndex, Dify or RAGFlow source is
copied or vendored, and the installed tool does not load or execute those projects.
The quoted-fence development fixture was generated in an isolated environment
using langchain-text-splitters 0.3.11 / langchain-core 0.3.79. The fixture's source,
generator, settings and output are committed under `tests/lint/fixtures/quoted_fence`.
The original quoted-table fixture uses those same pinned LangChain packages;
its source, generator, settings and output are in `tests/lint/fixtures/quoted_table`.
An isolated markdown-it-py 4.0.0 (MIT) CommonMark parser also verified its nesting
and container boundaries; it is not a runtime/test dependency or bundled code.
These projects do not endorse this tool.
Dependencies retain their own licenses and notices in their distributions.

### Format and specification references

| Reference | License | Relationship |
| --- | --- | --- |
| [LangChain](https://github.com/langchain-ai/langchain), including langchain-core Document and libs/text-splitters | MIT | `page_content`, `id`, `metadata`, and data-only constructor-envelope compatibility; no runtime dependency |
| [LlamaIndex](https://github.com/run-llama/llama_index) | MIT | TextNode `text`, `id_`, `metadata`, `start_char_idx`, `end_char_idx` field compatibility; no runtime dependency |
| [CommonMark specification 0.31.2](https://spec.commonmark.org/0.31.2/#fenced-code-blocks), John MacFarlane | CC-BY-SA-4.0 | Fence semantics reference only; no specification prose/examples copied; tests are newly authored |
| [SARIF 2.1.0, OASIS](https://docs.oasis-open.org/sarif/sarif/v2.1.0/os/sarif-v2.1.0-os.html) | OASIS specification terms | Output vocabulary reference only; no schema or specification text bundled |

### Locked runtime and test dependency graph

Versions follow the original lockfile. Conditional dependencies are included even when they
are not installed on every platform.

| Package | Version | License | Source / role |
| --- | --- | --- | --- |
| jsonschema | 4.26.0 | MIT | https://github.com/python-jsonschema/jsonschema — local saved-plan schema validation |
| attrs | 26.1.0 | MIT | https://github.com/python-attrs/attrs — schema validation dependency |
| jsonschema-specifications | 2025.9.1 | MIT | https://github.com/python-jsonschema/jsonschema-specifications — schema validation dependency |
| referencing | 0.37.0 | MIT | https://github.com/python-jsonschema/referencing — local schema references |
| rpds-py | 2026.9.1 | MIT | https://github.com/crate-py/rpds — schema validation dependency |
| tomli | 2.3.0 | MIT | https://github.com/hukkin/tomli — TOML on Python 3.10 only |
| pytest | 8.4.2 | MIT | https://github.com/pytest-dev/pytest — tests |
| iniconfig | 2.3.1 | MIT | https://github.com/pytest-dev/iniconfig — tests |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging — tests/build |
| pluggy | 1.7.0 | MIT | https://github.com/pytest-dev/pluggy — tests/build |
| Pygments | 2.21.0 | BSD-2-Clause | https://github.com/pygments/pygments — tests |
| exceptiongroup | 1.3.1 | MIT | https://github.com/agronholm/exceptiongroup — tests, Python <3.11 |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions — conditional runtime/test dependency |
| colorama | 0.4.6 | BSD-3-Clause | https://github.com/tartley/colorama — tests on Windows |

### Build tools

Hatchling is pinned in `pyproject.toml`; its dependencies are constrained there
for uv builds. These tools are not part of the installed runtime.

| Package | Version | License | Source |
| --- | --- | --- | --- |
| hatchling | 1.27.0 | MIT | https://github.com/pypa/hatch |
| pathspec | 1.1.1 | MPL-2.0 | https://github.com/cpburnz/python-pathspec |
| trove-classifiers | 2026.9.21.13 | Apache-2.0 | https://github.com/pypa/trove-classifiers |

packaging and pluggy are listed above. Python's standard library is provided by Python
under its own PSF license. No model weights, remote APIs, licensed datasets, or
third-party Markdown examples are required.

## `ragwrench table-chunk` (formerly table-chunker)

Original implementation, MIT. No upstream parser, chunker, benchmark code or
issue text was copied or vendored. Runtime uses Python's standard library only.
All document and benchmark fixtures are synthetic, authored for this tool, and
covered by its MIT license.

These projects define the input/output formats `table-chunk` interoperates with;
they are not runtime dependencies or endorsements:

| Project | License | Relationship |
| --- | --- | --- |
| [Docling](https://github.com/docling-project/docling) | MIT | Consumes Markdown/HTML exports |
| [Unstructured](https://github.com/Unstructured-IO/unstructured) | Apache-2.0 | Consumes extracted `metadata.text_as_html` strings |
| [MarkItDown](https://github.com/microsoft/markitdown) | MIT | Consumes Markdown exports |
| [LangChain](https://github.com/langchain-ai/langchain) | MIT | Emits Document constructor-shaped dictionaries |
| [LlamaIndex](https://github.com/run-llama/llama_index) | MIT | Emits TextNode constructor-shaped dictionaries |

Optional dependency: [tiktoken](https://github.com/openai/tiktoken), MIT, pinned
to 0.12.0. Installed only with the `tiktoken` extra; its dependencies and versions
are recorded in the original lockfile, not vendored. First use can download a vocabulary.
Defaults and tests do not use it.

Build/development dependencies: hatchling (MIT, pinned in `pyproject.toml`), pytest
(MIT) and its transitive dependencies pinned in the original lockfile. Python is supplied by your
environment under the Python Software Foundation license.

Upstream license labels above are taken from each project's repository; this tool does not
redistribute those projects' code.

## `ragwrench sync` (formerly chunk-sync-planner)

Original implementation, examples and synthetic fixtures: MIT, copyright 2026
tianhaogutum. No upstream implementation source was copied or vendored.
Dependencies are installed separately and retain their own licenses and notices.
This notice does not relicense those packages. Consult the installed distribution's
license files for full terms, including notices for code bundled inside wheels.
The original lockfile records exact artifacts and hashes.

### Direct dependency and conceptual sources

- **qdrant-client 1.16.2** — [source](https://github.com/qdrant/qdrant-client),
  Apache-2.0. Used as the real embedded local database, including collection
  metadata, point upserts/deletes, payload overwrites and vector retrieval.
- **Qdrant** — [source](https://github.com/qdrant/qdrant), Apache-2.0. Target
  collection/point model and UUID restrictions; server is not bundled or used.
- **LlamaIndex** — [source](https://github.com/run-llama/llama_index), MIT.
  Conceptual document-level ingestion hash baseline only; not a dependency.
- **LangChain** — [source](https://github.com/langchain-ai/langchain), MIT.
  Conceptual indexing/record-manager baseline only; not a dependency.
- **Python** — [source](https://github.com/python/cpython), PSF-2.0 and included
  notices. Standard-library hashing, JSON, subprocess, UUID and CLI facilities.
- **uv** — [source](https://github.com/astral-sh/uv), Apache-2.0 OR MIT.
  Development/installation tool, not bundled or a runtime dependency.

Issue states and dates referenced while designing the tool are historical context, not
current assertions about those projects.

### Locked runtime and development dependencies

License identifiers below come from the installed Linux package metadata/license
files. Windows-only entries are included for completeness of the universal lock.
Packages' own license files take precedence; binary wheels can carry additional
third-party notices (in particular NumPy and gRPC).

| Package | Linux version / locked alternative | License | Source |
|---|---|---|---|
| annotated-types | 0.8.0 | MIT | https://github.com/annotated-types/annotated-types |
| anyio | 4.16.0 | MIT | https://github.com/agronholm/anyio |
| certifi | 2026.10.23 | MPL-2.0 | https://github.com/certifi/python-certifi |
| grpcio | 1.85.1 | Apache-2.0 | https://github.com/grpc/grpc |
| h11 | 0.16.0 | MIT | https://github.com/python-hyper/h11 |
| h2 | 4.4.1 | MIT | https://github.com/python-hyper/h2 |
| hpack | 4.2.0 | MIT | https://github.com/python-hyper/hpack |
| httpcore | 1.0.9 | BSD-3-Clause | https://github.com/encode/httpcore |
| httpx | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| hyperframe | 6.1.0 | MIT | https://github.com/python-hyper/hyperframe |
| idna | 3.20 | BSD-3-Clause | https://github.com/kjd/idna |
| numpy | 2.4.6 / 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; wheel notices apply | https://github.com/numpy/numpy |
| portalocker | 3.2.0 | BSD-3-Clause | https://github.com/wolph/portalocker |
| protobuf | 7.36.2 | BSD-3-Clause | https://github.com/protocolbuffers/protobuf |
| pydantic | 2.14.3 | MIT | https://github.com/pydantic/pydantic |
| pydantic-core | 2.50.4 | MIT | https://github.com/pydantic/pydantic-core |
| typing-extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| typing-inspection | 0.4.5 | MIT | https://github.com/pydantic/typing-inspection |
| urllib3 | 2.8.0 | MIT | https://github.com/urllib3/urllib3 |
| pytest (dev) | 8.4.2 | MIT | https://github.com/pytest-dev/pytest |
| iniconfig (dev) | 2.3.1 | MIT | https://github.com/pytest-dev/iniconfig |
| packaging (dev/build) | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| pluggy (dev) | 1.7.0 | MIT | https://github.com/pytest-dev/pluggy |
| Pygments (dev) | 2.21.0 | BSD-2-Clause | https://github.com/pygments/pygments |
| colorama (Windows dev) | 0.4.6 | BSD-3-Clause | https://github.com/tartley/colorama |
| pywin32 (Windows) | 312 | PSF-style license; see distribution license | https://github.com/mhammond/pywin32 |

Build backend: **hatchling 1.27.0** (MIT), https://github.com/pypa/hatch. Its build
requirements include packaging (Apache-2.0 OR BSD-2-Clause), pathspec (MPL-2.0),
pluggy (MIT), and trove-classifiers (Apache-2.0), with their respective notices.
The source release does not bundle dependency source or binaries.

## `ragwrench bm25` (formerly qdrant-bm25-cjk)

qdrant-bm25-cjk's original code and synthetic text fixtures are MIT licensed.
No FastEmbed, Qdrant or jieba implementation source is copied into this repository.
Dependencies are installed separately; their bundled licenses and notices remain
applicable. The original lockfile records versions, origins and artifact hashes.

### Upstream relationships

- **qdrant-client 1.20.0** — Apache-2.0; Qdrant authors.
  <https://github.com/qdrant/qdrant-client>. Runtime local database, sparse vectors,
  `Modifier.IDF`, collection metadata and `query_points`.
- **FastEmbed / Qdrant BM25** — Apache-2.0; Qdrant authors.
  <https://github.com/qdrant/fastembed>. Architectural reference for splitting
  TF-side encoding from collection-side IDF. Not a dependency. Our normalization,
  SHA-256 dimension hash and tokenizers are independently implemented and **not
  vector-compatible** with FastEmbed.
- **jieba 0.42.1** — MIT; Sun Junyi and contributors.
  <https://github.com/fxsjy/jieba>. Optional dictionary-based tokenizer using its
  bundled dictionary, HMM disabled. No dictionary is copied into this repository.

### Locked dependency inventory

This covers runtime, optional, development and build dependencies. SPDX names
below are from installed package metadata or the upstream license. Alternative
versions selected for Python 3.10 have the same licensing unless noted.
Links point to the corresponding package's source/distribution metadata.

| Package | Locked version(s) | License |
| --- | --- | --- |
| [qdrant-client](https://pypi.org/project/qdrant-client/) | 1.20.0 | Apache-2.0 |
| [annotated-types](https://pypi.org/project/annotated-types/) | 0.8.0 | MIT |
| [anyio](https://pypi.org/project/anyio/) | 4.16.0 | MIT |
| [certifi](https://pypi.org/project/certifi/) | 2026.10.23 | MPL-2.0 |
| [colorama](https://pypi.org/project/colorama/) | 0.4.6 (Windows) | BSD-3-Clause |
| [exceptiongroup](https://pypi.org/project/exceptiongroup/) | 1.3.1 (Python 3.10) | MIT |
| [grpcio](https://pypi.org/project/grpcio/) | 1.85.1 | Apache-2.0 |
| [h11](https://pypi.org/project/h11/) | 0.16.0 | MIT |
| [h2](https://pypi.org/project/h2/) | 4.4.1 | MIT |
| [hpack](https://pypi.org/project/hpack/) | 4.2.0 | MIT |
| [httpcore](https://pypi.org/project/httpcore/) | 1.0.9 | BSD-3-Clause |
| [httpx](https://pypi.org/project/httpx/) | 0.28.1 | BSD-3-Clause |
| [hyperframe](https://pypi.org/project/hyperframe/) | 6.1.0 | MIT |
| [idna](https://pypi.org/project/idna/) | 3.20 | BSD-3-Clause |
| [numpy](https://pypi.org/project/numpy/) | 2.2.6 / 2.4.6 / 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (2.4.6) |
| [portalocker](https://pypi.org/project/portalocker/) | 4.4.2 | BSD-3-Clause |
| [protobuf](https://pypi.org/project/protobuf/) | 7.36.2 | BSD-3-Clause |
| [pydantic](https://pypi.org/project/pydantic/) | 2.14.3 | MIT |
| [pydantic-core](https://pypi.org/project/pydantic-core/) | 2.50.4 | MIT |
| [tomli](https://pypi.org/project/tomli/) | 2.5.0 (Python 3.10) | MIT |
| [typing-extensions](https://pypi.org/project/typing-extensions/) | 4.16.0 | PSF-2.0 |
| [typing-inspection](https://pypi.org/project/typing-inspection/) | 0.4.5 | MIT |
| [urllib3](https://pypi.org/project/urllib3/) | 2.8.0 | MIT |
| [jieba](https://pypi.org/project/jieba/) | 0.42.1 | MIT |
| [pytest](https://pypi.org/project/pytest/) | 9.0.2 | MIT |
| [iniconfig](https://pypi.org/project/iniconfig/) | 2.3.1 | MIT |
| [packaging](https://pypi.org/project/packaging/) | 26.3 | Apache-2.0 OR BSD-2-Clause |
| [pluggy](https://pypi.org/project/pluggy/) | 1.7.0 | MIT |
| [pygments](https://pypi.org/project/pygments/) | 2.21.0 | BSD-2-Clause |
| [ruff](https://pypi.org/project/ruff/) | 0.14.10 | MIT |
| [hatchling](https://pypi.org/project/hatchling/) | 1.30.0 (build backend) | MIT |

NumPy binary wheels may additionally bundle OpenBLAS/LAPACK (BSD-3-Clause),
GCC runtime libraries (GPL-3.0 with GCC Runtime Library Exception), libquadmath
(LGPL-2.1-or-later), and other components depending on platform. The authoritative
per-wheel license inventory is distributed with NumPy; preserve it when
redistributing environments. No binary dependency is vendored here.

Build-backend transitive dependencies (resolved by the isolated builder rather
than runtime the original lockfile) include hatchling's packaging (Apache-2.0 OR BSD-2-Clause),
pathspec (MPL-2.0), pluggy (MIT), and trove-classifiers (Apache-2.0). Runtime and
tests do not invoke the builder or make network requests.

### Design provenance

Upstream issue descriptions that motivated this command are historical context, not
assertions about their current open/closed state.

## `ragwrench migrate` (formerly vector-migration-parity)

Original vecparity code is MIT licensed (see LICENSE). No upstream implementation source
was copied. The committed synthetic fixture is dedicated to CC0-1.0.

### Direct upstreams and compatibility references

| Upstream | License | Use |
| --- | --- | --- |
| [Chroma](https://github.com/chroma-core/chroma) 1.5.9 | Apache-2.0 | Runtime PersistentClient, HNSW collections, no embedding downloads |
| [qdrant-client](https://github.com/qdrant/qdrant-client) 1.20.0 | Apache-2.0 | Runtime embedded/local mode |
| [qdrant/migration](https://github.com/qdrant/migration) | Apache-2.0 | Field-name compatibility reference for `__id__` and `document`; no source copied |
| [pytest](https://github.com/pytest-dev/pytest) 9.2.0 | MIT | Development tests |
| [Hatchling](https://github.com/pypa/hatch) 1.27.0 | MIT | Isolated PEP 517 build backend |

vecparity is an independent tool, not endorsed by these projects. It preserves
logical records, not upstream index files or arbitrary collection configuration.
Its mapping envelope is vecparity-specific, not qdrant/migration's wire format.

### Locked dependency inventory

The following inventory covers every third-party package in the original lockfile, including
conditional dependencies not installed on Linux/Python 3.11. License expressions
come from installed wheel metadata or the exact version's PyPI metadata (2026-11-11).
This is an attribution inventory, not a replacement for each distribution's own
copyright, LICENSE, NOTICE, or bundled-component license files. Those remain in
the upstream wheels installed by uv; dependencies are not vendored in this repo.
Build-backend dependencies are installed separately by PEP 517.

| Package | Version | Declared license |
| --- | --- | --- |
| [aiohappyeyeballs](https://pypi.org/project/aiohappyeyeballs/2.7.1/) | 2.7.1 | PSF-2.0 |
| [aiohttp](https://pypi.org/project/aiohttp/3.14.5/) | 3.14.5 | Apache-2.0 AND MIT |
| [aiohttp-retry](https://pypi.org/project/aiohttp-retry/2.9.1/) | 2.9.1 | MIT |
| [aiosignal](https://pypi.org/project/aiosignal/1.4.0/) | 1.4.0 | Apache-2.0 |
| [annotated-doc](https://pypi.org/project/annotated-doc/0.0.6/) | 0.0.6 | MIT |
| [annotated-types](https://pypi.org/project/annotated-types/0.8.0/) | 0.8.0 | MIT |
| [anyio](https://pypi.org/project/anyio/4.16.0/) | 4.16.0 | MIT |
| [async-timeout](https://pypi.org/project/async-timeout/5.0.1/) | 5.0.1 | Apache 2 |
| [attrs](https://pypi.org/project/attrs/26.1.0/) | 26.1.0 | MIT |
| [bcrypt](https://pypi.org/project/bcrypt/5.0.0/) | 5.0.0 | Apache-2.0 |
| [build](https://pypi.org/project/build/1.6.1/) | 1.6.1 | MIT |
| [certifi](https://pypi.org/project/certifi/2026.10.23/) | 2026.10.23 | MPL-2.0 |
| [charset-normalizer](https://pypi.org/project/charset-normalizer/3.5.2/) | 3.5.2 | MIT |
| [chromadb](https://pypi.org/project/chromadb/1.5.9/) | 1.5.9 | Apache-2.0 |
| [click](https://pypi.org/project/click/8.5.1/) | 8.5.1 | BSD-3-Clause |
| [colorama](https://pypi.org/project/colorama/0.4.6/) | 0.4.6 | OSI Approved :: BSD License |
| [durationpy](https://pypi.org/project/durationpy/0.11/) | 0.11 | MIT |
| [exceptiongroup](https://pypi.org/project/exceptiongroup/1.3.1/) | 1.3.1 | MIT |
| [filelock](https://pypi.org/project/filelock/4.1.0/) | 4.1.0 | MIT |
| [filelock](https://pypi.org/project/filelock/4.2.10/) | 4.2.10 | MIT |
| [flatbuffers](https://pypi.org/project/flatbuffers/25.12.19/) | 25.12.19 | Apache-2.0 |
| [frozenlist](https://pypi.org/project/frozenlist/1.8.0/) | 1.8.0 | Apache-2.0 |
| [fsspec](https://pypi.org/project/fsspec/2026.9.0/) | 2026.9.0 | BSD-3-Clause |
| [googleapis-common-protos](https://pypi.org/project/googleapis-common-protos/1.76.0/) | 1.76.0 | Apache-2.0 |
| [grpcio](https://pypi.org/project/grpcio/1.85.1/) | 1.85.1 | Apache-2.0 |
| [h11](https://pypi.org/project/h11/0.16.0/) | 0.16.0 | MIT |
| [h2](https://pypi.org/project/h2/4.4.1/) | 4.4.1 | MIT |
| [hf-xet](https://pypi.org/project/hf-xet/1.8.0/) | 1.8.0 | Apache-2.0 |
| [hpack](https://pypi.org/project/hpack/4.2.0/) | 4.2.0 | MIT |
| [httpcore](https://pypi.org/project/httpcore/1.0.9/) | 1.0.9 | BSD-3-Clause |
| [httpcore2](https://pypi.org/project/httpcore2/2.14.0/) | 2.14.0 | BSD-3-Clause |
| [httpcore2](https://pypi.org/project/httpcore2/2.16.0/) | 2.16.0 | BSD-3-Clause |
| [httptools](https://pypi.org/project/httptools/0.9.0/) | 0.9.0 | MIT |
| [httpx](https://pypi.org/project/httpx/0.28.1/) | 0.28.1 | BSD-3-Clause |
| [httpx2](https://pypi.org/project/httpx2/2.14.0/) | 2.14.0 | BSD-3-Clause |
| [httpx2](https://pypi.org/project/httpx2/2.16.0/) | 2.16.0 | BSD-3-Clause |
| [httpx2-jsfetch](https://pypi.org/project/httpx2-jsfetch/1.0/) | 1.0 | BSD-3-Clause |
| [huggingface-hub](https://pypi.org/project/huggingface-hub/2.4.0/) | 2.4.0 | Apache-2.0 |
| [hyperframe](https://pypi.org/project/hyperframe/6.1.0/) | 6.1.0 | MIT |
| [idna](https://pypi.org/project/idna/3.20/) | 3.20 | BSD-3-Clause |
| [importlib-metadata](https://pypi.org/project/importlib-metadata/9.0.1/) | 9.0.1 | Apache-2.0 |
| [importlib-resources](https://pypi.org/project/importlib-resources/7.1.0/) | 7.1.0 | Apache-2.0 |
| [iniconfig](https://pypi.org/project/iniconfig/2.3.1/) | 2.3.1 | MIT |
| [jsonschema](https://pypi.org/project/jsonschema/4.26.0/) | 4.26.0 | MIT |
| [jsonschema-specifications](https://pypi.org/project/jsonschema-specifications/2025.9.1/) | 2025.9.1 | MIT |
| [kubernetes](https://pypi.org/project/kubernetes/37.0.2/) | 37.0.2 | Apache-2.0 |
| [markdown-it-py](https://pypi.org/project/markdown-it-py/4.3.0/) | 4.3.0 | MIT |
| [mdurl](https://pypi.org/project/mdurl/0.1.2/) | 0.1.2 | MIT |
| [mmh3](https://pypi.org/project/mmh3/5.3.1/) | 5.3.1 | MIT |
| [mpmath](https://pypi.org/project/mpmath/1.3.0/) | 1.3.0 | BSD |
| [multidict](https://pypi.org/project/multidict/7.1.0/) | 7.1.0 | Apache-2.0 |
| [multidict](https://pypi.org/project/multidict/7.2.1/) | 7.2.1 | Apache-2.0 |
| [numpy](https://pypi.org/project/numpy/2.2.6/) | 2.2.6 | BSD-3-Clause; binary wheels also bundle OpenBLAS, LAPACK and GCC runtime libraries whose notices (including GPL-3.0-or-later WITH GCC-exception-3.1) ship in the wheel's `LICENSE.txt` |
| [numpy](https://pypi.org/project/numpy/2.4.6/) | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| [numpy](https://pypi.org/project/numpy/2.5.4/) | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| [oauthlib](https://pypi.org/project/oauthlib/4.0.1/) | 4.0.1 | BSD-3-Clause |
| [onnxruntime](https://pypi.org/project/onnxruntime/1.24.3/) | 1.24.3 | MIT License |
| [onnxruntime](https://pypi.org/project/onnxruntime/1.31.1/) | 1.31.1 | MIT |
| [opentelemetry-api](https://pypi.org/project/opentelemetry-api/1.45.2/) | 1.45.2 | Apache-2.0 |
| [opentelemetry-exporter-otlp-common](https://pypi.org/project/opentelemetry-exporter-otlp-common/0.66b2/) | 0.66b2 | Apache-2.0 |
| [opentelemetry-exporter-otlp-proto-common](https://pypi.org/project/opentelemetry-exporter-otlp-proto-common/1.45.2/) | 1.45.2 | Apache-2.0 |
| [opentelemetry-exporter-otlp-proto-grpc](https://pypi.org/project/opentelemetry-exporter-otlp-proto-grpc/1.45.2/) | 1.45.2 | Apache-2.0 |
| [opentelemetry-proto](https://pypi.org/project/opentelemetry-proto/1.45.2/) | 1.45.2 | Apache-2.0 |
| [opentelemetry-sdk](https://pypi.org/project/opentelemetry-sdk/1.45.2/) | 1.45.2 | Apache-2.0 |
| [opentelemetry-semantic-conventions](https://pypi.org/project/opentelemetry-semantic-conventions/0.66b2/) | 0.66b2 | Apache-2.0 |
| [orjson](https://pypi.org/project/orjson/3.13.0/) | 3.13.0 | MPL-2.0 AND (Apache-2.0 OR MIT) |
| [overrides](https://pypi.org/project/overrides/7.7.0/) | 7.7.0 | Apache-2.0 |
| [packaging](https://pypi.org/project/packaging/26.3/) | 26.3 | Apache-2.0 OR BSD-2-Clause |
| [pluggy](https://pypi.org/project/pluggy/1.7.0/) | 1.7.0 | MIT |
| [portalocker](https://pypi.org/project/portalocker/4.4.2/) | 4.4.2 | BSD-3-Clause |
| [propcache](https://pypi.org/project/propcache/0.5.4/) | 0.5.4 | Apache-2.0 |
| [protobuf](https://pypi.org/project/protobuf/7.36.2/) | 7.36.2 | BSD-3-Clause |
| [pybase64](https://pypi.org/project/pybase64/1.5.2/) | 1.5.2 | BSD-2-Clause |
| [pydantic](https://pypi.org/project/pydantic/2.14.3/) | 2.14.3 | MIT |
| [pydantic-core](https://pypi.org/project/pydantic-core/2.50.4/) | 2.50.4 | MIT |
| [pydantic-settings](https://pypi.org/project/pydantic-settings/2.15.0/) | 2.15.0 | MIT |
| [pydantic-settings](https://pypi.org/project/pydantic-settings/2.16.2/) | 2.16.2 | MIT |
| [pygments](https://pypi.org/project/pygments/2.21.0/) | 2.21.0 | BSD-2-Clause |
| [pypika](https://pypi.org/project/pypika/0.51.1/) | 0.51.1 | Apache-2.0 |
| [pyproject-hooks](https://pypi.org/project/pyproject-hooks/1.3.3/) | 1.3.3 | MIT |
| [pytest](https://pypi.org/project/pytest/9.2.0/) | 9.2.0 | MIT |
| [python-dateutil](https://pypi.org/project/python-dateutil/3.0.0/) | 3.0.0 | Dual License |
| [python-dateutil](https://pypi.org/project/python-dateutil/4.0.0/) | 4.0.0 | BSD-3-Clause |
| [python-dotenv](https://pypi.org/project/python-dotenv/1.2.4/) | 1.2.4 | BSD-3-Clause |
| [pyyaml](https://pypi.org/project/pyyaml/6.0.3/) | 6.0.3 | MIT |
| [qdrant-client](https://pypi.org/project/qdrant-client/1.20.0/) | 1.20.0 | Apache-2.0 |
| [referencing](https://pypi.org/project/referencing/0.37.0/) | 0.37.0 | MIT |
| [requests](https://pypi.org/project/requests/2.34.2/) | 2.34.2 | Apache-2.0 |
| [requests-oauthlib](https://pypi.org/project/requests-oauthlib/2.0.0/) | 2.0.0 | ISC |
| [rich](https://pypi.org/project/rich/15.0.0/) | 15.0.0 | MIT |
| [rpds-py](https://pypi.org/project/rpds-py/0.30.0/) | 0.30.0 | MIT |
| [rpds-py](https://pypi.org/project/rpds-py/2026.9.1/) | 2026.9.1 | MIT |
| [shellingham](https://pypi.org/project/shellingham/1.5.4/) | 1.5.4 | ISC |
| [six](https://pypi.org/project/six/1.17.0/) | 1.17.0 | MIT |
| [sympy](https://pypi.org/project/sympy/1.14.0/) | 1.14.0 | BSD |
| [tenacity](https://pypi.org/project/tenacity/9.2.1/) | 9.2.1 | Apache-2.0 |
| [tokenizers](https://pypi.org/project/tokenizers/0.23.3/) | 0.23.3 | Apache-2.0 |
| [tomli](https://pypi.org/project/tomli/2.5.0/) | 2.5.0 | MIT |
| [tqdm](https://pypi.org/project/tqdm/4.70.1/) | 4.70.1 | MPL-2.0 AND MIT |
| [truststore](https://pypi.org/project/truststore/0.10.5/) | 0.10.5 | MIT |
| [typer](https://pypi.org/project/typer/0.29.1/) | 0.29.1 | MIT |
| [typing-extensions](https://pypi.org/project/typing-extensions/4.16.0/) | 4.16.0 | PSF-2.0 |
| [typing-inspection](https://pypi.org/project/typing-inspection/0.4.5/) | 0.4.5 | MIT |
| [tzdata](https://pypi.org/project/tzdata/2026.6/) | 2026.6 | Apache-2.0 |
| [urllib3](https://pypi.org/project/urllib3/2.8.0/) | 2.8.0 | MIT |
| [uvicorn](https://pypi.org/project/uvicorn/0.54.0/) | 0.54.0 | BSD-3-Clause |
| [uvloop](https://pypi.org/project/uvloop/0.23.0/) | 0.23.0 | MIT |
| [watchfiles](https://pypi.org/project/watchfiles/1.3.0/) | 1.3.0 | MIT |
| [websocket-client](https://pypi.org/project/websocket-client/1.9.2/) | 1.9.2 | Apache-2.0 |
| [websockets](https://pypi.org/project/websockets/16.1.1/) | 16.1.1 | BSD-3-Clause |
| [websockets](https://pypi.org/project/websockets/17.4/) | 17.4 | BSD-3-Clause |
| [yarl](https://pypi.org/project/yarl/1.25.1/) | 1.25.1 | Apache-2.0 |
| [zipp](https://pypi.org/project/zipp/4.1.2/) | 4.1.2 | MIT |

## `ragwrench cite-check` (formerly rag-cite-check)

All citecheck implementation, tests, fixtures and the labelled synthetic corpus
are original code/data released under the repository's MIT LICENSE. No upstream
implementation, datasets, model weights, or documentation prose are vendored.
The Python standard library is the only runtime dependency (Python: PSF License).
This is an independent project, not an official component of the products below.

### Interoperability references (no runtime installation or source copying)

| Project | Upstream license | Use |
| --- | --- | --- |
| [LlamaIndex](https://github.com/run-llama/llama_index) | [MIT](https://github.com/run-llama/llama_index/blob/main/LICENSE) | JSON response/source_nodes shape and ordered 1-based CitationQueryEngine numbering |
| [Dify](https://github.com/langgenius/dify) | [Dify Open Source License](https://github.com/langgenius/dify/blob/main/LICENSE), modified Apache-2.0 with additional conditions | Chat export answer/metadata.retriever_resources fields; no Dify code incorporated or relicensed |
| [RAGFlow](https://github.com/infiniflow/ragflow) | [Apache-2.0](https://github.com/infiniflow/ragflow/blob/main/LICENSE) | Completion answer/reference.chunks list shape and zero-based marker convention |

Upstream licenses govern their respective projects; the Dify license is not represented
as unmodified Apache-2.0 or as an OSI-approved license here.

### Development and build dependencies

Versions of test dependencies, including conditional dependencies for other
platforms/Python versions, are recorded with distribution hashes in the original lockfile.
No dependency source is copied into this repository. Distributed dependency
packages retain their own bundled license notices.

| Distribution | Version | License | Role |
| --- | --- | --- | --- |
| [pytest](https://github.com/pytest-dev/pytest) | 8.3.5 | MIT | Tests |
| [iniconfig](https://github.com/pytest-dev/iniconfig) | 2.3.1 | MIT | pytest dependency |
| [packaging](https://github.com/pypa/packaging) | 26.3 | Apache-2.0 OR BSD-2-Clause | pytest/build dependency |
| [pluggy](https://github.com/pytest-dev/pluggy) | 1.7.0 | MIT | pytest dependency |
| [colorama](https://github.com/tartley/colorama) | 0.4.6 | BSD-3-Clause | pytest on Windows (not installed on this Linux run) |
| [exceptiongroup](https://github.com/agronholm/exceptiongroup) | 1.3.1 | MIT | pytest on Python <3.11 |
| [tomli](https://github.com/hukkin/tomli) | 2.5.0 | MIT | pytest on Python <3.11 |
| [typing-extensions](https://github.com/python/typing_extensions) | 4.16.0 | PSF-2.0 | exceptiongroup on Python <3.11 |
| [hatchling](https://github.com/pypa/hatch) | 1.27.0 (pinned in pyproject.toml) | MIT | Isolated build backend |
| [pathspec](https://github.com/cpburnz/python-pathspec) | Backend-resolved | MPL-2.0 | hatchling build dependency |
| [trove-classifiers](https://github.com/pypa/trove-classifiers) | Backend-resolved | Apache-2.0 | hatchling build dependency |

Hatchling also conditionally uses tomli on Python <3.11. Build backend
transitive versions are resolved by the build frontend; runtime is dependency
free. The original lockfile fixes the test environment. The `uv` tool itself is
MIT OR Apache-2.0 and is not bundled. Build outputs and virtual environments are
excluded from git.

## `ragwrench eval` (formerly retrieval-regress-gate)

All retgate implementation code and synthetic fixtures are original MIT-licensed
work. No third-party implementation or dataset is vendored. Python's standard
library (PSF license) is the only runtime requirement. This file records references
and development/build tooling, not a claim that upstream licenses cover retgate.

### Behavioral and format references

| Project | License / notice | Relationship |
|---|---|---|
| [ranx](https://github.com/AmenRa/ranx) | MIT | Comparison baseline; optional, tie-free cross-check if installed. No runtime dependency or copied code. |
| [pytrec_eval](https://github.com/cvangysel/pytrec_eval) | MIT (Christophe Van Gysel) | Optional cross-check of metrics and tied-score ordering. No copied code. |
| [trec_eval](https://github.com/usnistgov/trec_eval) | File-specific historical notices; inspected metric/comparator files state research, non-commercial permission | Semantic reference only. No C code, source files, binaries or fixtures are redistributed. |
| [BEIR](https://github.com/beir-cellar/beir) | Apache-2.0 | Qrels TSV schema; no dataset or loader code copied. |
| [LlamaIndex](https://github.com/run-llama/llama_index) | MIT | EmbeddingQAFinetuneDataset exchange schema; no framework code copied. |

Upstream issue status and maintenance notes reflect the time this command was designed;
they are not maintained assertions about those projects.

### Development / packaging dependencies

the original lockfile records exact versions, hashes and Python/platform markers for the dev
group, including packages not used on every platform.
They are installed by the package manager under their respective licenses and are
not copied into the source tree or runtime wheel.

| Package | License | Use |
|---|---|---|
| pytest 8.4.2 | MIT | Tests |
| iniconfig 2.3.1 | MIT | pytest dependency |
| packaging 26.3 | Apache-2.0 OR BSD-2-Clause | pytest/build dependency |
| pluggy 1.7.0 | MIT | pytest/build dependency |
| Pygments 2.21.0 | BSD-2-Clause | pytest dependency |
| colorama 0.4.6 | BSD-3-Clause | pytest dependency, Windows only |
| exceptiongroup 1.3.1 | MIT | pytest dependency, Python <3.11 |
| tomli 2.5.0 | MIT | pytest dependency, Python <3.11 |
| typing_extensions 4.16.0 | PSF-2.0 | exceptiongroup dependency |
| hatchling 1.27.0 | MIT | Pinned isolated build backend |
| pathspec | MPL-2.0 | Isolated hatchling build dependency |
| trove-classifiers | Apache-2.0 | Isolated hatchling build dependency |

Optional pytrec_eval and ranx are not part of the locked/default environment.
uv is an external installer, not a bundled dependency (MIT OR Apache-2.0).

## `ragwrench highlight` (formerly cite-highlight-pdf)

Original cite-highlight-pdf source and synthetic fixtures are MIT licensed. No Docling
or Unstructured source code, PDFs, OCR models, or datasets are copied or bundled. This
is an independent format adapter, not an official upstream tool.

### Runtime and format integrations

| Project | Version / use | License | Source |
| --- | --- | --- | --- |
| pypdf | 7.0.1; PDF reading, cloning, writing, text visitors | BSD-3-Clause | https://github.com/py-pdf/pypdf |
| typing_extensions | 4.16.0; conditional pypdf dependency on Python 3.10 | PSF-2.0 | https://github.com/python/typing_extensions |
| docling-core | Format reference only; not installed or bundled | MIT | https://github.com/docling-project/docling-core |
| Unstructured | Format reference only; not installed or bundled | Apache-2.0 | https://github.com/Unstructured-IO/unstructured |

The complete pypdf copyright, BSD conditions, and disclaimer are retained in
[`licenses/pypdf-BSD-3-Clause.txt`](licenses/pypdf-BSD-3-Clause.txt). Dependencies
are installed separately from PyPI with their own license notices intact.

### Test and build dependencies

Versions of test dependencies, including platform/Python-conditional ones, and
artifact hashes are recorded in the original lockfile. Build backend is pinned in
`pyproject.toml`; its transitive requirements are resolved in build isolation.
These tools are not included in the distributed runtime wheel.

| Project | Version / role | License | Source |
| --- | --- | --- | --- |
| pytest | 9.0.2; tests | MIT | https://github.com/pytest-dev/pytest |
| iniconfig | 2.3.1; pytest | MIT | https://github.com/pytest-dev/iniconfig |
| packaging | 26.3; pytest / build | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| pluggy | 1.7.0; pytest / build | MIT | https://github.com/pytest-dev/pluggy |
| Pygments | 2.21.0; pytest | BSD-2-Clause | https://github.com/pygments/pygments |
| colorama | 0.4.6; pytest on Windows only | BSD-3-Clause | https://github.com/tartley/colorama |
| exceptiongroup | 1.3.1; pytest on Python 3.10 | MIT | https://github.com/agronholm/exceptiongroup |
| tomli | 2.5.0; pytest on Python 3.10 / build | MIT | https://github.com/hukkin/tomli |
| hatchling | 1.27.0; isolated build backend | MIT | https://github.com/pypa/hatch |
| pathspec | hatchling build dependency | MPL-2.0 | https://github.com/cpburnz/python-pathspec |
| trove-classifiers | hatchling build dependency | Apache-2.0 | https://github.com/pypa/trove-classifiers |

### Public schema references checked during implementation

Checked 2026-11-11, used to implement independent field mappings (not source copying):

- [Docling BoundingBox / CoordOrigin](https://github.com/docling-project/docling-core/blob/main/docling_core/types/doc/base.py)
- [Docling chunk export](https://github.com/docling-project/docling-core/blob/main/docling_core/transforms/chunker/base.py)
- [Unstructured PixelSpace and PointSpace](https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/documents/coordinates.py)
- [Unstructured orig_elements serialization](https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/staging/base.py)
- [pypdf annotation API](https://pypdf.readthedocs.io/en/stable/user/adding-pdf-annotations.html)

Unstructured's compressed orig_elements is historically called "gzipped" JSON;
its current staging serializer uses zlib. Both zlib and gzip headers are accepted,
with explicit decompression limits and integrity checks.

The example PDF uses only synthetic text and a reference to PDF's built-in Helvetica
font; no font file is bundled.

## `ragwrench port` (formerly kb-bundle-porter)

kb-bundle-porter's original code and synthetic fixtures are MIT licensed (see LICENSE). It calls public HTTP APIs. No Dify or RAGFlow source code, assets, embeddings or datasets are copied, linked or bundled. Product names identify compatibility and remain their owners' names/trademarks. This project is independent of both upstreams.

### API contract sources

- **Dify**, LangGenius: https://github.com/langgenius/dify
  - License: **Dify Open Source License**, a modified Apache-2.0 license with additional conditions; it must not be represented as plain Apache-2.0. Upstream license: https://github.com/langgenius/dify/blob/f8fc371a/LICENSE
  - Contract revision: `f8fc371a`.
  - Dataset endpoints: https://github.com/langgenius/dify/blob/f8fc371a/api/controllers/service_api/dataset/dataset.py
  - Document create/list/detail endpoints: https://github.com/langgenius/dify/blob/f8fc371a/api/controllers/service_api/dataset/document.py
  - Segment list/create/update/delete endpoints: https://github.com/langgenius/dify/blob/f8fc371a/api/controllers/service_api/dataset/segment.py
  - Response/payload fields: https://github.com/langgenius/dify/blob/f8fc371a/api/fields/document_fields.py and https://github.com/langgenius/dify/blob/f8fc371a/api/services/knowledge/entities/segments.py
- **RAGFlow**, InfiniFlow: https://github.com/infiniflow/ragflow
  - License: **Apache-2.0**. Upstream license: https://github.com/infiniflow/ragflow/blob/d7d4dd0b/LICENSE
  - Contract revision: `d7d4dd0b`, including the RAGFlow 1.0 migration notes.
  - API reference: https://github.com/infiniflow/ragflow/blob/d7d4dd0b/docs/references/http_api_reference.md
  - Used contracts: dataset listing/creation, paginated documents, empty virtual document creation, chunk listing/creation, PATCH availability, and nonzero error envelopes.
  - Additional profile: **v0.27.3**, immutable revision `8090ffd8ba6098813d7f7ec2c4b8774f36c8423a`.
  - v0.27.3 API reference: https://github.com/infiniflow/ragflow/blob/8090ffd8ba6098813d7f7ec2c4b8774f36c8423a/docs/references/http_api_reference.md
  - Dataset selection uses `chunk_method: "naive"` instead of 1.0's `parse_type: 1, parser_id: "general"`. Both profiles document empty virtual documents and PATCH chunk updates; no PUT fallback is implemented.
  - Negotiation reads dataset field signatures from these pinned list responses. This identifies the selected contract family, not a verified server release; empty/ambiguous listings require an explicit profile.

The synthetic native JSON fixtures were written independently to match these documented field shapes. They contain no production data or copied upstream example text. API surface inspection does not establish compatibility with every released version or configured provider; see README for the precise scope and limits.

### Installed Python dependencies

Dependencies are fetched separately by uv/pip, not vendored source. Exact versions, URLs and artifact hashes are in the original lockfile; installed distributions retain their own license notices. Direct runtime dependencies:

| Distribution | Version | License | Source |
| --- | --- | --- | --- |
| httpx | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| jsonschema | 4.26.0 | MIT | https://github.com/python-jsonschema/jsonschema |

Runtime transitives: anyio (MIT), attrs (MIT), certifi (MPL-2.0), h11 (MIT), httpcore (BSD-3-Clause), idna (BSD-3-Clause), jsonschema-specifications (MIT), referencing (MIT), rpds-py (MIT), typing-extensions (PSF-2.0). Build/test tools include hatchling (MIT), pytest (MIT), iniconfig (MIT), packaging (Apache-2.0 OR BSD-2-Clause), pluggy (MIT), Pygments (BSD-2-Clause). Consult each installed distribution's metadata/license files for full notices. No dependency's license is replaced by this project's MIT license.

## `ragwrench dify-bridge` (formerly dify-external-kb-bridge)

Original bridge code, tests, synthetic fixtures and documentation are MIT licensed (LICENSE). No Dify, Qdrant server, RAGFlow, model weights or other upstream source code is vendored. Installed dependency wheels retain their own license files; this project's MIT license does not relicense them.

### Contract and references

| Project | Source | License / relationship |
| --- | --- | --- |
| Dify | https://github.com/langgenius/dify | Dify Open Source License (modified Apache-2.0); documented HTTP contract implemented independently, no code copied |
| qdrant-client | https://github.com/qdrant/qdrant-client | Apache-2.0; public runtime client library, pinned 1.15.1 |
| RAGFlow issue 3377 | https://github.com/infiniflow/ragflow/issues/3377 | Historical interoperability evidence only; no code or service integration used |

Dify and Qdrant names identify compatible upstream systems and do not imply endorsement.

### Locked dependencies

Runtime/test versions below are from the original lockfile. License names were checked against installed distribution metadata on Linux; platform-only dependencies are separately marked. License texts and upstream notices ship inside the respective dependency distributions (typically `*.dist-info/licenses`). Source and release links are available at `https://pypi.org/project/<package>/<version>/` and in the original lockfile.

| Package | Version | License |
| --- | --- | --- |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.16.0 | MIT |
| certifi | 2026.10.23 | MPL-2.0 |
| grpcio | 1.85.1 | Apache-2.0 |
| h11 | 0.16.0 | MIT |
| h2 | 4.4.1 | MIT |
| hpack | 4.2.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| hyperframe | 6.1.0 | MIT |
| idna | 3.20 | BSD-3-Clause |
| iniconfig | 2.3.1 | MIT |
| numpy | 2.4.6 (Python 3.11), 2.5.4 (newer Python lock branch) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; binary wheels can include additional bundled-library notices |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pluggy | 1.7.0 | MIT |
| portalocker | 3.2.0 | BSD-3-Clause |
| protobuf | 7.36.2 | BSD-3-Clause |
| pydantic | 2.14.3 | MIT |
| pydantic-core | 2.50.4 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pytest | 8.4.2 | MIT |
| qdrant-client | 1.15.1 | Apache-2.0 |
| typing-extensions | 4.16.0 | PSF-2.0 |
| typing-inspection | 0.4.5 | MIT |
| urllib3 | 2.8.0 | MIT |
| colorama (Windows only) | 0.4.6 | BSD-3-Clause |
| pywin32 (Windows only) | 312 | PSF-based license; see its distribution's License.txt |

Build tooling: hatchling 1.27.0 (MIT), with packaging (Apache-2.0 OR BSD-2-Clause), pathspec (MPL-2.0), pluggy (MIT) and trove-classifiers (Apache-2.0). These are isolated build tools rather than vendored application code. Python's standard library is under the PSF license. uv is an external installation/development tool (MIT OR Apache-2.0), not a package dependency.

When redistributing a bundled environment, preserve all dependency license and notice files, including NumPy's platform-specific binary-library notices and certifi's MPL notice. This repository ships no bundled environment.
