"""Reproducible count experiment, not a timing/ML quality benchmark."""
import random

from .core import SyncError, build_snapshot, chunk_text, digest, make_plan
from .embedders import HashEmbedder
from .store import apply_plan, local_client, verify_snapshot


def run_bench(seed=25, paragraphs=60):
    if not 10 <= paragraphs <= 1000:
        raise SyncError("bench paragraphs must be between 10 and 1000")
    rng = random.Random(seed)
    words = "index vector boundary stable payload document retrieval orchard river mountain autumn compass notebook library lantern meadow forest planet window".split()
    docs = {}
    for doc in range(3):
        docs[f"guide-{doc}.md"] = [f"Section {doc}.{i}. " + " ".join(rng.choices(words, k=36)) + "." for i in range(paragraphs)]
    scenarios = {}
    for name in ("insert_at_top", "modify_paragraph", "delete", "reorder", "rename"):
        changed = {k: list(v) for k, v in docs.items()}
        if name == "insert_at_top":
            changed["guide-0.md"].insert(0, "New introduction. " + " ".join(rng.choices(words, k=29)) + ".")
        elif name == "modify_paragraph":
            changed["guide-0.md"][paragraphs // 2] += " A newly discovered boundary needs special care."
        elif name == "delete":
            del changed["guide-0.md"]
        elif name == "reorder":
            items = changed["guide-0.md"]
            items[4:8], items[12:16] = items[12:16], items[4:8]
        elif name == "rename":
            changed["renamed.md"] = changed.pop("guide-0.md")
        scenarios[name] = changed

    def snap(corpus, method):
        rows = []
        for name, paras in sorted(corpus.items()):
            rows.extend({"doc_id": name, "text": text} for text in chunk_text("\n\n".join(paras), method))
        return build_snapshot(rows)

    old = {method: snap(docs, method) for method in ("fixed", "cdc")}
    results = []
    for scenario, corpus in scenarios.items():
        new = {method: snap(corpus, method) for method in ("fixed", "cdc")}
        changed_docs = {name for name, paras in corpus.items() if name not in docs or paras != docs[name]}
        cdc_total = len(new["cdc"]["chunks"])
        # Run the baselines, rather than treating planned counts as measured
        # embedding calls. Old embeddings are setup work, excluded for all.
        previous_vectors = {c["id"]: HashEmbedder(8).embed(c, "hash-v1") for c in old["cdc"]["chunks"]}
        full, document = HashEmbedder(8), HashEmbedder(8)
        expected = {c["id"]: full.embed(c, "hash-v1") for c in new["cdc"]["chunks"]}
        document_vectors = {}
        for c in new["cdc"]["chunks"]:
            document_vectors[c["id"]] = (document.embed(c, "hash-v1") if c["doc_id"] in changed_docs
                                          else previous_vectors[c["id"]])
        if document_vectors != expected:
            raise SyncError("document-level baseline differs from from-scratch vectors")
        counts = {
            "full_reindex": {"embedded": full.calls, "total": cdc_total},
            "document_hash": {"embedded": document.calls, "total": cdc_total},
        }
        for method in ("fixed", "cdc"):
            plan = make_plan(old[method], new[method])
            with local_client(":memory:") as client:
                embedder = HashEmbedder(8)
                apply_plan(client, "bench", make_plan(build_snapshot([]), old[method]), embedder)
                setup_calls = embedder.calls
                applied = apply_plan(client, "bench", plan, embedder)
                measured = embedder.calls - setup_calls
                if measured != plan["summary"]["upsert"] or measured != applied["embedded"]:
                    raise SyncError("measured embedding calls differ from plan")
                verify_snapshot(client, "bench", new[method], HashEmbedder(8))
            counts[f"chunk_{method}"] = {"embedded": measured, "total": len(new[method]["chunks"]),
                                        "moved": plan["summary"]["moved"], "deleted": plan["summary"]["delete"]}
        for values in counts.values():
            values["avoided"] = values["total"] - values["embedded"]
            values["verified_points"] = values["total"]
        results.append({"scenario": scenario, "strategies": counts})
    return {"kind": "chunksync.bench", "version": 1, "seed": seed, "documents": 3, "paragraphs_per_document": paragraphs,
            "chunk_sizes": {"min": 128, "target": 256, "max": 512}, "corpus_fingerprint": digest(docs),
            "metric": "actual HashEmbedder(8) calls during update; setup and independent verification excluded",
            "baseline": "full and document-hash use the same CDC chunks; fixed has its own total; moves reuse vectors",
            "results": results}
