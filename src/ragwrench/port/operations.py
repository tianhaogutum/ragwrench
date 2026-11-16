from collections import Counter, defaultdict
from copy import deepcopy

from .bundle import fingerprint, load_bundle, write_bundle
from .common import PortError, TransientAPIError, atomic_json, content_hash, digest, dumps, identifier, locked, read_json, safe_file


def export_bundle(api, dataset_id, output, *, resume=False):
    dataset_id = identifier(dataset_id)
    api.negotiate()
    output = safe_file(output)
    output.mkdir(parents=True, exist_ok=True)
    with locked(output / ".export.lock"):
        checkpoint = output / ".export-state.json"
        identity = {"version": 1, "system": api.system, "base_url": api.base_url,
                    "dataset_id": dataset_id, "page_size": api.page_size, "api_profile": api.api_profile}
        if checkpoint.exists():
            if not resume:
                raise PortError("Export checkpoint exists; use --resume or a new output directory")
            state = read_json(checkpoint)
            if api.state_identity(state.get("identity", {})) != identity:
                raise PortError("Export checkpoint belongs to different source/options")
        else:
            if (output / "manifest.json").exists():
                raise PortError("Output already contains a bundle; choose another directory")
            state = {"identity": identity, "pages": {}}
        def save():
            atomic_json(checkpoint, state)
        save()
        datasets = api.pages("datasets", cache=state["pages"], save=save)
        api.negotiate(datasets)
        found = [d for d in datasets if d["id"] == dataset_id]
        if len(found) != 1:
            raise PortError("Source dataset not found or ambiguous")
        documents = api.pages("documents", dataset_id, cache=state["pages"], save=save)
        chunks = []
        for document in documents:
            chunks.extend(api.pages("chunks", dataset_id, document["id"], cache=state["pages"], save=save))
        manifest = write_bundle(output, {"system": api.system, "dataset_id": dataset_id},
                                found[0]["settings"], documents, chunks)
        state["complete"] = True
        save()
        return {"bundle": str(output), "api_profile": api.api_profile, **manifest["counts"]}


def project_chunk(chunk, system, allow_lossy=False):
    chunk = deepcopy(chunk)
    losses = []
    if chunk["metadata"]:
        losses.append("chunk metadata (including image/layout/tags)")
        chunk["metadata"] = {}
    if system == "ragflow" and chunk["answer"]:
        losses.append("QA structure flattened to Question/Answer text")
        chunk["content"] = f"Question: {chunk['content']}\nAnswer: {chunk['answer']}"
        chunk["answer"] = None
    if system == "dify" and chunk["questions"]:
        losses.append("embedding questions unsupported by Dify")
        chunk["questions"] = []
    if losses and not allow_lossy:
        raise PortError("Unsupported fields: " + "; ".join(losses) + "; use --allow-lossy to acknowledge")
    chunk["content_sha256"] = content_hash(chunk["content"])
    return chunk, losses


def make_plan(manifest, documents, chunks, system, *, allow_lossy=False, max_chunk_bytes=16000, on_error="abort"):
    if not 1 <= max_chunk_bytes <= 1024 * 1024:
        raise PortError("max-chunk-bytes must be 1..1048576")
    projected, errors, warnings = {}, {}, set()
    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk["doc_id"]].append(chunk)
    doc_forms = {}
    for document in documents:
        if document["metadata"]:
            if not allow_lossy:
                raise PortError("Document metadata cannot be restored; use --allow-lossy to acknowledge")
            warnings.add("Document metadata preserved in bundle only")
        doc_forms[document["id"]] = "qa_model" if system == "dify" and document["doc_form"] == "qa_model" else "text_model"
        seen_content = set()
        for chunk in groups[document["id"]]:
            key = chunk_key(chunk)
            try:
                mapped, losses = project_chunk(chunk, system, allow_lossy)
                warnings.update(losses)
                if len(mapped["content"].encode("utf-8")) > max_chunk_bytes:
                    raise PortError("Chunk exceeds configured target UTF-8 byte limit")
                if system == "dify" and (doc_forms[document["id"]] == "qa_model") != bool(mapped["answer"]):
                    raise PortError("Dify QA documents require an answer for every chunk; text documents require no answer")
                if system == "ragflow" and mapped["content"] in seen_content:
                    raise PortError("Repeated content in one document can collide in RAGFlow chunk IDs")
                seen_content.add(mapped["content"])
                projected[key] = mapped
            except PortError as exc:
                if on_error != "skip":
                    raise
                errors[key] = str(exc)
    return {"documents": len(documents), "chunks": len(projected), "skipped": len(errors),
            "warnings": sorted(warnings), "errors": errors}, projected, doc_forms


def chunk_key(chunk):
    return chunk["doc_id"] + "/" + chunk["id"]


def semantic(chunk, *, enabled=True):
    value = {k: chunk[k] for k in ("content", "answer", "keywords", "questions")}
    value["keywords"] = sorted(value["keywords"])
    value["questions"] = sorted(value["questions"])
    if enabled:
        value["enabled"] = chunk["enabled"]
    return digest(dumps(value).encode())


def document_name(document, bundle_id):
    marker = digest((bundle_id + "/" + document["id"]).encode())[:16]
    return document["name"][:80] + f" [kbport-{marker}]"


class _PendingChunkNotVisible(PortError):
    pass


def _import_chunk(api, target, doc, chunk, save, *, retry_pending, chunk_retries):
    """Bound recovery of one durable intent, never automatically replay a create."""
    for attempt in range(chunk_retries + 1):
        try:
            pending = doc["pending"]
            if pending and pending["source"] != chunk["id"]:
                raise PortError("State has an unexpected pending chunk")
            if pending and not pending.get("target"):
                existing = api.pages("chunks", target, doc["id"])
                used = set(doc["chunks"].values())
                matches = [c for c in existing if c["id"] not in used and semantic(c, enabled=False) == semantic(chunk, enabled=False)]
                if len(matches) > 1:
                    raise PortError("Ambiguous pending chunk; inspect target before resuming")
                if matches:
                    pending["target"] = matches[0]["id"]
                    save()
                elif not (retry_pending and attempt == 0):
                    # Even --retry-pending authorizes only one create attempt per invocation.
                    raise _PendingChunkNotVisible("Pending chunk has no visible match; wait or inspect target before --retry-pending")
            if not pending:
                pending = doc["pending"] = {"source": chunk["id"], "target": None}
                save()
            if not pending["target"]:
                pending["target"] = api.add_chunk(target, doc["id"], chunk)
                if pending["target"] in doc["chunks"].values():
                    raise PortError("Target collapsed chunk IDs; cannot preserve multiplicity")
                save()
            # Repeating a known chunk's enabled flag is idempotent.
            api.set_enabled(target, doc["id"], pending["target"], chunk["enabled"])
            doc["chunks"][chunk["id"]] = pending["target"]
            doc["pending"] = None
            save()
            return
        except (TransientAPIError, _PendingChunkNotVisible) as exc:
            if attempt == chunk_retries:
                raise
            api.wait_before_retry(attempt, getattr(exc, "retry_after", 0))


def import_bundle(api, bundle, state_path, *, dataset_name=None, dataset_id=None, dry_run=False,
                  allow_lossy=False, max_chunk_bytes=16000, on_error="abort", indexing="high_quality",
                  retry_pending=False, chunk_retries=0):
    if type(chunk_retries) is not int or not 0 <= chunk_retries <= 10:
        raise PortError("chunk-retries must be 0..10")
    manifest, documents, chunks = load_bundle(bundle)
    plan, projected, forms = make_plan(manifest, documents, chunks, api.system, allow_lossy=allow_lossy,
                                       max_chunk_bytes=max_chunk_bytes, on_error=on_error)
    if bool(dataset_id) == bool(dataset_name):
        raise PortError("Specify exactly one of --dataset-id (reuse) or --dataset-name (create)")
    if dataset_id:
        identifier(dataset_id)
    if dataset_name and (not dataset_name.strip() or len(dataset_name) > (40 if api.system == "dify" else 128)
                         or any(ord(c) > 0xffff for c in dataset_name)):
        raise PortError("Invalid target dataset name length/characters")
    if dry_run:
        api.negotiate(offline=True)
        # No network, state writes, or locks, even if credentials are absent.
        return {"dry_run": True, "target_system": api.system, "dataset_id": dataset_id,
                "dataset_name": dataset_name, "api_profile": api.api_profile, **plan}
    # Auto discovery must finish before creating state or mutating a target.
    api.negotiate()
    state_path = safe_file(state_path)
    identity = {"version": 1, "bundle": fingerprint(manifest), "system": api.system,
                "base_url": api.base_url, "dataset_name": dataset_name, "reuse_dataset": dataset_id,
                "allow_lossy": allow_lossy, "max_chunk_bytes": max_chunk_bytes, "on_error": on_error,
                "indexing": indexing, "api_profile": api.api_profile}
    with locked(str(state_path) + ".lock"):
        if state_path.exists():
            state = read_json(state_path)
            if api.state_identity(state.get("identity", {})) != identity:
                raise PortError("Import state belongs to a different bundle/target/options (including API profile)")
        else:
            state = {"identity": identity, "dataset": dataset_id, "dataset_pending": False, "documents": {}}
        datasets = api.datasets()
        api.negotiate(datasets)
        def save():
            atomic_json(state_path, state)
        save()
        if state["dataset"]:
            if not any(d["id"] == state["dataset"] for d in datasets):
                raise PortError("Mapped target dataset no longer exists")
        else:
            matches = [d for d in datasets if d["name"].casefold() == dataset_name.casefold()]
            if matches:
                if not state["dataset_pending"] or len(matches) != 1 or matches[0]["name"] != dataset_name:
                    raise PortError("Duplicate target dataset name; use --dataset-id to explicitly reuse")
                state["dataset"] = matches[0]["id"]
            else:
                if state["dataset_pending"] and not retry_pending:
                    raise PortError("Pending dataset creation has no visible match; wait or use --retry-pending after inspecting target")
                state["dataset_pending"] = True
                save()
                state["dataset"] = api.create_dataset(dataset_name, indexing)
            state["dataset_pending"] = False
            save()
        target = identifier(state["dataset"])
        for document in documents:
            doc = state["documents"].setdefault(document["id"], {"id": None, "phase": "new", "chunks": {}, "pending": None})
            name = document_name(document, identity["bundle"])
            if not doc["id"]:
                remote_docs = api.pages("documents", target, allow_pending=True)
                matches = [d for d in remote_docs if d["name"] == name]
                if matches:
                    if doc["phase"] != "creating" or len(matches) != 1:
                        raise PortError("Target document marker collision; refusing to overwrite")
                    doc["id"] = matches[0]["id"]
                else:
                    if doc["phase"] == "creating" and not retry_pending:
                        raise PortError("Pending document has no visible match; wait or inspect target before --retry-pending")
                    doc["phase"] = "creating"
                    save()
                    doc["id"] = api.create_document(target, name, forms[document["id"]], indexing)
                doc["phase"] = "clearing"
                save()
            remote_doc = identifier(doc["id"])
            if doc["phase"] == "clearing":
                api.ready_and_clear(target, remote_doc)
                doc["phase"] = "chunks"
                save()
            for original in chunks:
                if original["doc_id"] != document["id"]:
                    continue
                key = chunk_key(original)
                if key not in projected or original["id"] in doc["chunks"]:
                    continue
                _import_chunk(api, target, doc, projected[key], save,
                              retry_pending=retry_pending, chunk_retries=chunk_retries)
            doc["phase"] = "complete"
            save()
        return {"dataset_id": target, "state": str(state_path), "api_profile": api.api_profile, **plan}


def verify_bundle(api, bundle, *, dataset_id=None, state_path=None, allow_lossy=False, max_chunk_bytes=16000):
    manifest, documents, chunks = load_bundle(bundle)
    plan, projected, _ = make_plan(manifest, documents, chunks, api.system,
                                   allow_lossy=allow_lossy, max_chunk_bytes=max_chunk_bytes)
    api.negotiate()
    mapping = None
    if state_path:
        state = read_json(state_path)
        identity = state.get("identity", {})
        if identity.get("bundle") != fingerprint(manifest) or identity.get("system") != api.system or identity.get("base_url") != api.base_url:
            raise PortError("Verification state belongs to another bundle/target")
        if api.state_identity(identity)["api_profile"] != api.api_profile:
            raise PortError("Verification API profile differs from state")
        if dataset_id and dataset_id != state["dataset"]:
            raise PortError("Verification dataset differs from state")
        dataset_id = state["dataset"]
        mapping = {source: doc["id"] for source, doc in state["documents"].items()}
        if set(mapping) != {d["id"] for d in documents} or any(d["pending"] or d["phase"] != "complete" for d in state["documents"].values()):
            raise PortError("Import is incomplete; cannot verify")
    if not dataset_id:
        raise PortError("Verification needs --dataset-id or --state")
    dataset_id = identifier(dataset_id)
    remote_docs = api.pages("documents", dataset_id, allow_pending=True)
    selected = set(mapping.values()) if mapping else {d["id"] for d in remote_docs}
    actual_by_doc = {d["id"]: api.pages("chunks", dataset_id, d["id"]) for d in remote_docs if d["id"] in selected}
    expected = Counter()
    actual = Counter()
    if mapping:
        for doc in documents:
            remote_id = mapping[doc["id"]]
            if remote_id not in actual_by_doc:
                raise PortError("Mapped target document is missing")
            expected.update((remote_id, semantic(c)) for c in projected.values() if c["doc_id"] == doc["id"])
            actual.update((remote_id, semantic(c)) for c in actual_by_doc[remote_id])
    else:
        if len(remote_docs) != len(documents):
            raise PortError("Verification document count mismatch")
        expected.update(semantic(c) for c in projected.values())
        actual.update(semantic(c) for rows in actual_by_doc.values() for c in rows)
    missing, extra = sum((expected - actual).values()), sum((actual - expected).values())
    if missing or extra:
        raise PortError(f"Verification failed: {missing} missing/changed chunks, {extra} extra/changed chunks")
    return {"verified": True, "api_profile": api.api_profile, "documents": len(documents),
            "chunks": len(projected), "warnings": plan["warnings"]}
