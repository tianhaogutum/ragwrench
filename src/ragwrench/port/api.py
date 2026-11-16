"""Independent public-HTTP API adapters, with conservative mutation semantics."""
import json
import math
import os
import time
from urllib.parse import urlsplit

import httpx

from .common import PortError, TransientAPIError, content_hash, identifier


DEFAULT_PROFILES = {"dify": "dify-f8fc371a", "ragflow": "ragflow-1.0"}
API_PROFILES = ("auto", "dify-f8fc371a", "ragflow-1.0", "ragflow-0.27.3")


def select_profile(system, api_profile=None):
    """Validate an offline contract selection without credentials or an HTTP client."""
    if system not in DEFAULT_PROFILES:
        raise PortError("Unknown system")
    profile = api_profile or DEFAULT_PROFILES[system]
    if (profile not in API_PROFILES or
            (profile == "auto" and system != "ragflow") or
            (profile != "auto" and not profile.startswith(system + "-"))):
        raise PortError("Unsupported API profile for this system; auto is supported only for RAGFlow")
    return profile


class API:
    def __init__(self, system, base_url, *, key=None, key_env=None, page_size=100,
                 retries=3, backoff=0.5, timeout=30, poll_timeout=120, api_profile=None):
        api_profile = select_profile(system, api_profile)
        self.api_profile = None if api_profile == "auto" else api_profile
        self.auto_profile = api_profile == "auto"
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise PortError("Base URL must be HTTP(S), without credentials, query or fragment")
        suffix = "/v1" if system == "dify" else "/api/v1"
        path = parts.path.rstrip("/")
        if path and not path.endswith(suffix):
            raise PortError(f"Base URL path must be empty or end in {suffix}")
        self.base_url = base_url.rstrip("/") + ("" if path else suffix)
        key = key or os.environ.get(key_env or ("DIFY_KB_API_KEY" if system == "dify" else "RAGFLOW_API_KEY"))
        if not key or any(c in key for c in "\r\n"):
            raise PortError("Missing or invalid API key in environment")
        if (not 1 <= page_size <= 100 or not 0 <= retries <= 10 or backoff < 0 or timeout <= 0 or poll_timeout < 0
                or not all(math.isfinite(n) for n in (backoff, timeout, poll_timeout))):
            raise PortError("Invalid page size, retry, backoff or timeout option")
        self.system, self.page_size = system, page_size
        self.retries, self.backoff, self.poll_timeout = retries, backoff, poll_timeout
        self.http = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=timeout,
                                 follow_redirects=False, trust_env=False)

    def close(self):
        self.http.close()

    def negotiate(self, datasets=None, *, offline=False):
        """Select a pinned dataset contract from read-only evidence, never trial writes.

        Field signatures identify a compatible contract family, not an exact server
        release. Explicit profiles remain necessary for empty or ambiguous servers.
        """
        if self.system != "ragflow":
            return self.api_profile
        if datasets is None:
            if self.api_profile is not None:
                return self.api_profile
            if offline:
                raise PortError("Offline --dry-run needs an explicit --api-profile; auto requires GET discovery")
            datasets = self.datasets()
        observed, unknown = set(), not datasets
        for dataset in datasets:
            settings = dataset["settings"]
            legacy = isinstance(settings.get("chunk_method"), str) and bool(settings["chunk_method"])
            modern = isinstance(settings.get("parser_id"), str) and bool(settings["parser_id"])
            if legacy == modern:
                unknown = True
            else:
                observed.add("ragflow-0.27.3" if legacy else "ragflow-1.0")
        if self.auto_profile and (unknown or len(observed) != 1):
            raise PortError("Ambiguous RAGFlow dataset contract; select an explicit --api-profile")
        if self.api_profile is None:
            self.api_profile = observed.pop()
        elif observed - {self.api_profile}:
            raise PortError("Selected API profile conflicts with the RAGFlow dataset contract; check --api-profile")
        return self.api_profile

    def state_identity(self, identity):
        # State/checkpoints from before profiles used the original default contract.
        return {"api_profile": DEFAULT_PROFILES[self.system], **identity}

    @staticmethod
    def retry_after(response):
        try:
            return min(30, max(0, float(response.headers.get("Retry-After", "0")))) if response is not None else 0
        except ValueError:
            return 0

    def wait_before_retry(self, attempt, retry_after=0):
        time.sleep(min(30, max(self.backoff * 2 ** attempt, retry_after)))

    def request(self, method, path, *, params=None, body=None):
        # Never replay writes here. Import recovery must reconcile creates first.
        attempts = self.retries + 1 if method == "GET" else 1
        for attempt in range(attempts):
            response = None
            try:
                with self.http.stream(method, self.base_url + path, params=params, json=body) as response:
                    payload = bytearray()
                    if response.is_success and response.status_code != 204:
                        for part in response.iter_bytes():
                            payload.extend(part)
                            if len(payload) > 32 * 1024 * 1024:
                                raise PortError("API response exceeds 32 MiB safety limit; reduce --page-size")
            except httpx.HTTPError as exc:
                if attempt + 1 == attempts:
                    raise TransientAPIError(f"{method} transport failure; resume using the same state") from exc
            else:
                if response.status_code < 400 and response.status_code >= 300:
                    raise PortError("Redirect refused; use the final API base URL")
                retryable = response.status_code == 429 or response.status_code >= 500
                if response.is_success:
                    if response.status_code == 204:
                        return None
                    try:
                        data = json.loads(payload)
                    except ValueError as exc:
                        raise PortError("API returned invalid JSON") from exc
                    if not isinstance(data, dict):
                        raise PortError("API returned a non-object JSON envelope")
                    if self.system == "ragflow":
                        code = data.get("code")
                        if type(code) is not int or code != 0:
                            raise PortError(f"RAGFlow error envelope code {code if isinstance(code, int) else 'invalid'}")
                        return data.get("data")
                    return data
                if not retryable or attempt + 1 == attempts:
                    if retryable:
                        raise TransientAPIError(f"{method} API HTTP {response.status_code}; resume using the same state",
                                                retry_after=self.retry_after(response))
                    raise PortError(f"{method} API HTTP {response.status_code}; check access/version, then resume")
            self.wait_before_retry(attempt, self.retry_after(response))
        raise PortError("API retry budget exhausted")

    def pages(self, kind, dataset_id=None, doc_id=None, *, cache=None, save=lambda: None, allow_pending=False):
        """Checkpoint every validated page, never trust a short page against a total."""
        dataset_id = identifier(dataset_id) if dataset_id else None
        doc_id = identifier(doc_id) if doc_id else None
        path = "/datasets"
        if kind == "documents":
            path += f"/{dataset_id}/documents"
        elif kind == "chunks":
            path += f"/{dataset_id}/documents/{doc_id}/" + ("segments" if self.system == "dify" else "chunks")
        elif kind != "datasets":
            raise PortError("Unknown resource kind")
        cache = cache if cache is not None else {}
        state = cache.setdefault(path, {"next": 1, "items": [], "total": None, "done": False})
        while not state["done"]:
            page = state["next"]
            params = {"page": page, "limit" if self.system == "dify" else "page_size": self.page_size}
            if self.system == "ragflow" and kind != "chunks":
                params.update(orderby="create_time", desc="false")
            result = self.request("GET", path, params=params)
            if not isinstance(result, dict if self.system == "dify" or kind != "datasets" else list):
                raise PortError("Malformed API pagination envelope")
            if self.system == "dify":
                items, total = result.get("data"), result.get("total")
                more = result.get("has_more")
            elif kind == "datasets":
                items, total, more = result, None, None
            else:
                items = result.get("docs" if kind == "documents" else "chunks")
                total = result.get("total", result.get("total_datasets"))
                more = None
            if not isinstance(items, list) or len(items) > self.page_size:
                raise PortError("Malformed or overfull API page")
            if total is not None and (type(total) is not int or total < 0):
                raise PortError("Invalid API total")
            if kind != "datasets" and total is None:
                raise PortError("API page omitted total; completeness cannot be established")
            if state["total"] is not None and total != state["total"]:
                raise PortError("API total changed during pagination; restart export from a frozen source")
            mapped = [self.normalize(kind, item, doc_id, len(state["items"]) + i, allow_pending) for i, item in enumerate(items)]
            ids = {item["id"] for item in state["items"]}
            if any(item["id"] in ids for item in mapped) or len({item["id"] for item in mapped}) != len(mapped):
                raise PortError("Duplicate IDs across API pages; source may be changing")
            count = len(state["items"]) + len(mapped)
            if count > 100000 or (total is not None and total > 100000):
                raise PortError("API resource exceeds 100000 record safety limit")
            if total is not None:
                if count > total or (count < total and len(items) < self.page_size):
                    raise PortError("Truncated API page / total mismatch")
                done = count == total
                if more is not None and bool(more) == done:
                    raise PortError("API has_more disagrees with total")
            else:
                done = len(items) < self.page_size
            state["items"].extend(mapped)
            state.update(next=page + 1, total=total, done=done)
            save()
        return state["items"]

    def normalize(self, kind, item, doc_id=None, ordinal=0, allow_pending=False):
        if not isinstance(item, dict):
            raise PortError("API record is not an object")
        ident = identifier(item.get("id"))
        if kind == "datasets":
            return {"id": ident, "name": item.get("name"), "settings": {
                k: item[k] for k in ("language", "indexing_technique", "parser_id", "chunk_method") if k in item}}
        if kind == "documents":
            form = item.get("doc_form") or "text_model"
            if form not in ("text_model", "qa_model"):
                raise PortError("Hierarchical or unknown Dify doc_form is unsupported")
            if self.system == "dify" and not allow_pending and (item.get("enabled") is False or item.get("archived") is True or item.get("indexing_status") not in (None, "completed")):
                raise PortError("Export requires enabled, non-archived, fully indexed Dify documents")
            if self.system == "ragflow" and not allow_pending and item.get("status") in (0, "0"):
                raise PortError("Export of disabled RAGFlow documents is unsupported")
            metadata = item.get("doc_metadata") or item.get("meta_fields") or {}
            if isinstance(metadata, list):
                metadata = {"dify_fields": metadata}
            if not isinstance(metadata, dict):
                raise PortError("Unsupported document metadata representation")
            return {"id": ident, "name": item.get("name"), "doc_form": form, "metadata": metadata}
        content = item.get("content")
        if not isinstance(content, str):
            raise PortError("Chunk content missing or non-text")
        enabled = item.get("enabled", True) if self.system == "dify" else item.get("available", bool(item.get("available_int", 1)))
        return {"id": ident, "doc_id": doc_id, "position": item.get("position", ordinal) if self.system == "dify" else ordinal,
                "content": content, "keywords": item.get("keywords" if self.system == "dify" else "important_keywords", []) or [],
                "questions": item.get("questions", item.get("question_kwd", [])) or [],
                "answer": item.get("answer") or None, "enabled": enabled,
                "metadata": {k: item[k] for k in ("positions", "tag_kwd", "image_id", "attachment_ids") if item.get(k)},
                "content_sha256": content_hash(content)}

    def datasets(self):
        return self.pages("datasets")

    def create_dataset(self, name, indexing="high_quality"):
        body = {"name": name}
        if self.system == "dify":
            body.update(indexing_technique=indexing, permission="only_me")
        elif self.negotiate() == "ragflow-0.27.3":
            body.update(chunk_method="naive")
        else:
            body.update(parse_type=1, parser_id="general")
        result = self.request("POST", "/datasets", body=body)
        if result.get("name") != name:
            raise PortError("Target renamed dataset (name collision); inspect target before resuming")
        return identifier(result["id"])

    def create_document(self, dataset, name, form="text_model", indexing="high_quality"):
        if self.system == "ragflow":
            result = self.request("POST", f"/datasets/{identifier(dataset)}/documents", params={"type": "empty"}, body={"name": name})
            if not isinstance(result, list) or len(result) != 1:
                raise PortError("Unexpected RAGFlow document creation response")
            return identifier(result[0]["id"])
        result = self.request("POST", f"/datasets/{identifier(dataset)}/document/create-by-text", body={
            "name": name, "text": ("Q1: What is kbport?\nA1: A temporary bootstrap." if form == "qa_model" else "kbport temporary bootstrap text."), "doc_form": form,
            "indexing_technique": indexing,
            "process_rule": {"mode": "custom", "rules": {"pre_processing_rules": [],
                              "segmentation": {"separator": "\n\n", "max_tokens": 500}}}})
        return identifier(result["document"]["id"])

    def ready_and_clear(self, dataset, document):
        if self.system == "dify":
            deadline = time.monotonic() + self.poll_timeout
            while True:
                detail = self.request("GET", f"/datasets/{identifier(dataset)}/documents/{identifier(document)}")
                status = detail.get("indexing_status")
                if status == "completed":
                    break
                if status in ("error", "paused", "stopped"):
                    raise PortError("Dify bootstrap indexing failed or paused; inspect target, then resume")
                if time.monotonic() >= deadline:
                    raise PortError("Dify document indexing timed out; resume later")
                time.sleep(0.25)
        for chunk in self.pages("chunks", dataset, document):
            if self.system != "dify":
                raise PortError("New RAGFlow document unexpectedly contains chunks; refusing to overwrite")
            self.request("DELETE", f"/datasets/{dataset}/documents/{document}/segments/{chunk['id']}")

    def add_chunk(self, dataset, document, chunk):
        path = f"/datasets/{identifier(dataset)}/documents/{identifier(document)}/"
        if self.system == "dify":
            data = {"content": chunk["content"], "keywords": chunk["keywords"]}
            if chunk["answer"]:
                data["answer"] = chunk["answer"]
            result = self.request("POST", path + "segments", body={"segments": [data]})
            return identifier(result["data"][0]["id"])
        result = self.request("POST", path + "chunks", body={"content": chunk["content"],
                              "important_keywords": chunk["keywords"], "questions": chunk["questions"]})
        return identifier(result["chunk"]["id"])

    def set_enabled(self, dataset, document, chunk_id, enabled):
        path = f"/datasets/{identifier(dataset)}/documents/{identifier(document)}/"
        if self.system == "dify":
            return self.request("POST", path + f"segments/{identifier(chunk_id)}", body={"segment": {"enabled": enabled}})
        return self.request("PATCH", path + f"chunks/{identifier(chunk_id)}", body={"available": enabled})
