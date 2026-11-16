# ragwrench documentation

One page per command. Each page starts with the problem the command solves and a quick
start, followed by the full reference. Run `ragwrench <command> --help` for the flags.
Quick starts use the sample inputs under [`examples/`](../examples/), so run them from a
clone of the repository (`git clone https://github.com/tianhaogutum/ragwrench`).

## Chunking

| Command | Use it to |
| --- | --- |
| [`ragwrench audit`](audit.md) | Get one read-only health report for a chunk export, with the command that fixes each problem |
| [`ragwrench lint`](lint.md) | Find and conservatively repair chunks that split code fences, tables, headings or lose text |
| [`ragwrench table-chunk`](table-chunk.md) | Split Markdown/HTML so every table chunk keeps its headers, caption and heading path |

## Indexing and vector stores

| Command | Use it to |
| --- | --- |
| [`ragwrench sync`](sync.md) | Plan, review and apply incremental chunk updates to a local Qdrant collection |
| [`ragwrench bm25`](bm25.md) | Build CJK-aware BM25 sparse vectors and search them in Qdrant local mode |
| [`ragwrench migrate`](migrate.md) | Copy collections between Chroma and Qdrant and verify record and retrieval parity |

## Answers and evaluation

| Command | Use it to |
| --- | --- |
| [`ragwrench cite-check`](cite-check.md) | Check citations in exported RAG answers against their retrieved sources |
| [`ragwrench eval`](eval.md) | Score retrieval runs and gate regressions in CI with paired statistics |

## Citations and PDFs

| Command | Use it to |
| --- | --- |
| [`ragwrench highlight`](highlight.md) | Turn Docling/Unstructured provenance into PDF highlights and `#page=` citation links |

## Platforms

| Command | Use it to |
| --- | --- |
| [`ragwrench port`](port.md) | Export, import and verify hash-checked knowledge bases between Dify and RAGFlow |
| [`ragwrench dify-bridge`](dify-bridge.md) | Serve Qdrant or JSONL chunks as a Dify External Knowledge API and check such endpoints |

Additional reference: [Dify bridge filter semantics](dify-bridge/filtering.md).
Third-party attribution is in [THIRD_PARTY.md](../THIRD_PARTY.md).
