"""Offline, stateful HTTP contract fixtures. No external services or model calls."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from .bundle import load_bundle
from .common import content_hash


class FakeState:
    def __init__(self, system, *, key="fake-key", api_profile=None):
        self.system, self.key = system, key
        self.api_profile = api_profile or ("ragflow-1.0" if system == "ragflow" else "dify-f8fc371a")
        self.datasets = {}
        self.requests = []
        self.faults = []
        self.truncate = False
        self.indexing_polls = 0

    def seed(self, bundle, dataset_id=None):
        manifest, docs, chunks = load_bundle(bundle)
        dataset_id = dataset_id or manifest["source"]["dataset_id"]
        self.datasets[dataset_id] = {"id": dataset_id, "name": "Synthetic handbook", "documents": {}}
        for doc in docs:
            self.datasets[dataset_id]["documents"][doc["id"]] = {**deepcopy(doc), "chunks": {}, "indexing_status": "completed", "enabled": True}
        for chunk in chunks:
            self.datasets[dataset_id]["documents"][chunk["doc_id"]]["chunks"][chunk["id"]] = deepcopy(chunk)
        return dataset_id

    def new_id(self):
        return str(uuid4()) if self.system == "dify" else uuid4().hex

    def envelope(self, value):
        return {"code": 0, "data": value} if self.system == "ragflow" else value

    def error(self, status, message="synthetic API error"):
        return status, {"code": status, "message": message}

    def fault(self, method, path, query, phase):
        for rule in list(self.faults):
            if (rule.get("method", "GET") == method and rule.get("path", path) == path
                    and path.endswith(rule.get("path_suffix", ""))
                    and rule.get("page", query.get("page", ["1"])[0]) == query.get("page", ["1"])[0]
                    and rule.get("phase", "before") == phase):
                self.faults.remove(rule)
                return rule.get("status", 503), rule.get("body", {"message": "injected failure"})
        return None

    def doc_view(self, doc):
        return {"id": doc["id"], "name": doc["name"], "doc_form": doc["doc_form"],
                "indexing_status": doc["indexing_status"], "enabled": doc["enabled"],
                "doc_metadata" if self.system == "dify" else "meta_fields": doc["metadata"]}

    def chunk_view(self, chunk):
        if self.system == "dify":
            return {k: deepcopy(chunk[k]) for k in ("id", "content", "position", "keywords", "answer", "enabled")}
        return {"id": chunk["id"], "content": chunk["content"], "available": chunk["enabled"],
                "important_keywords": deepcopy(chunk["keywords"]), "questions": deepcopy(chunk["questions"])}

    def dispatch(self, method, path, query, body):
        parts = path.strip("/").split("/")
        prefix = ["v1"] if self.system == "dify" else ["api", "v1"]
        if parts[:len(prefix)] != prefix:
            return self.error(404)
        parts = parts[len(prefix):]
        if not parts or parts[0] != "datasets":
            return self.error(404)
        if len(parts) == 1:
            if method == "GET":
                rows = [{"id": d["id"], "name": d["name"], "language": "English"} for d in self.datasets.values()]
                if self.system == "ragflow":
                    for row in rows:
                        row.update({"chunk_method": "naive"} if self.api_profile == "ragflow-0.27.3" else {"parser_id": "general"})
                return self.paginate(rows, "datasets", query)
            if method == "POST":
                expected = ({"name", "indexing_technique", "permission"} if self.system == "dify" else
                            {"name", "chunk_method"} if self.api_profile == "ragflow-0.27.3" else {"name", "parse_type", "parser_id"})
                if set(body) != expected:
                    return self.error(400, "bad dataset fields")
                if self.system == "ragflow" and (body.get("chunk_method") != "naive" if self.api_profile == "ragflow-0.27.3" else
                                                  body.get("parse_type") != 1 or body.get("parser_id") != "general"):
                    return self.error(400, "bad parser selection")
                if any(d["name"].casefold() == body["name"].casefold() for d in self.datasets.values()):
                    return self.error(409)
                ident = self.new_id()
                self.datasets[ident] = {"id": ident, "name": body["name"], "documents": {}}
                return 200, self.envelope({"id": ident, "name": body["name"]})
        if len(parts) < 2 or parts[1] not in self.datasets:
            return self.error(404)
        dataset = self.datasets[parts[1]]
        documents = dataset["documents"]
        if len(parts) == 3 and parts[2] == "documents" and method == "GET":
            return self.paginate([self.doc_view(d) for d in documents.values()], "documents", query)
        create_dify = parts[2:] == ["document", "create-by-text"] and self.system == "dify"
        create_rag = parts[2:] == ["documents"] and self.system == "ragflow"
        if method == "POST" and (create_dify or create_rag):
            if create_rag and (query.get("type") != ["empty"] or set(body) != {"name"}):
                return self.error(400, "empty document contract")
            if create_dify and ("text" not in body or "process_rule" not in body):
                return self.error(400, "text document contract")
            ident = self.new_id()
            doc = {"id": ident, "name": body["name"], "doc_form": body.get("doc_form", "text_model"),
                   "metadata": {}, "chunks": {}, "enabled": True,
                   "indexing_status": "indexing" if create_dify else "completed", "polls": 0}
            documents[ident] = doc
            if create_dify:
                chunk = self.new_chunk(body["text"], ident, 0)
                doc["chunks"][chunk["id"]] = chunk
                return 200, {"document": self.doc_view(doc), "batch": self.new_id()}
            return 200, self.envelope([self.doc_view(doc)])
        if len(parts) < 4 or parts[2] != "documents" or parts[3] not in documents:
            return self.error(404)
        doc = documents[parts[3]]
        if len(parts) == 4 and method == "GET":
            doc["polls"] = doc.get("polls", 0) + 1
            if doc["polls"] > self.indexing_polls:
                doc["indexing_status"] = "completed"
            return 200, self.doc_view(doc)
        resource = "segments" if self.system == "dify" else "chunks"
        if len(parts) < 5 or parts[4] != resource:
            return self.error(404)
        chunks = doc["chunks"]
        if self.system == "dify" and doc["indexing_status"] != "completed":
            return self.error(404)
        if len(parts) == 5:
            if method == "GET":
                return self.paginate([self.chunk_view(c) for c in chunks.values()], "chunks", query)
            if method == "POST":
                data = body["segments"][0] if self.system == "dify" else body
                expected = {"content", "keywords", "answer"} if self.system == "dify" else {"content", "important_keywords", "questions"}
                if set(data) - expected:
                    return self.error(400, "bad chunk fields")
                if doc["doc_form"] == "qa_model" and self.system == "dify" and not data.get("answer"):
                    return self.error(400, "QA answer required")
                chunk = self.new_chunk(data["content"], doc["id"], max([c["position"] for c in chunks.values()], default=0) + 1)
                chunk.update(keywords=data.get("keywords", data.get("important_keywords", [])),
                             questions=data.get("questions", []), answer=data.get("answer"))
                if self.system == "ragflow":
                    # Real RAGFlow chunk IDs can be content-derived: exercise collision boundary.
                    chunk["id"] = content_hash(doc["id"] + data["content"])[:32]
                chunks[chunk["id"]] = chunk
                view = self.chunk_view(chunk)
                return 200, {"data": [view], "doc_form": doc["doc_form"]} if self.system == "dify" else self.envelope({"chunk": view})
        if len(parts) == 6 and parts[5] in chunks:
            if method == "DELETE" and self.system == "dify":
                del chunks[parts[5]]
                return 204, None
            if method == ("POST" if self.system == "dify" else "PATCH"):
                enabled = body["segment"]["enabled"] if self.system == "dify" else body["available"]
                chunks[parts[5]]["enabled"] = enabled
                return 200, {"data": self.chunk_view(chunks[parts[5]])} if self.system == "dify" else self.envelope(None)
        return self.error(404)

    def new_chunk(self, content, doc_id, position):
        return {"id": self.new_id(), "doc_id": doc_id, "position": position, "content": content,
                "keywords": [], "questions": [], "answer": None, "enabled": True, "metadata": {},
                "content_sha256": content_hash(content)}

    def paginate(self, rows, kind, query):
        page = int(query.get("page", ["1"])[0])
        limit = int(query.get("limit" if self.system == "dify" else "page_size", ["30"])[0])
        if page < 1 or not 1 <= limit <= 100:
            return self.error(400)
        start = (page - 1) * limit
        items = rows[start:start + limit]
        if self.truncate and items:
            items = items[:-1]
        if self.system == "dify":
            return 200, {"data": items, "total": len(rows), "page": page, "limit": limit, "has_more": start + limit < len(rows)}
        if kind == "datasets":
            return 200, self.envelope(items)
        return 200, self.envelope({"docs" if kind == "documents" else "chunks": items,
                                   "total" if kind == "chunks" else "total_datasets": len(rows)})


def make_server(state, host="127.0.0.1", port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle_request(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            if self.headers.get("Authorization") != "Bearer " + state.key:
                status, result = state.error(401)
            else:
                size = int(self.headers.get("Content-Length", 0))
                if size > 4 * 1024 * 1024:
                    self.send_error(413)
                    return
                try:
                    body = json.loads(self.rfile.read(size)) if size else None
                    state.requests.append({"method": self.command, "path": parsed.path, "query": query, "body": deepcopy(body)})
                    fault = state.fault(self.command, parsed.path, query, "before")
                    status, result = fault or state.dispatch(self.command, parsed.path, query, body)
                    if not fault:
                        status, result = state.fault(self.command, parsed.path, query, "after") or (status, result)
                except (KeyError, TypeError, ValueError):
                    status, result = state.error(400)
            blob = json.dumps(result).encode() if result is not None else b""
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when an interrupted client closes before reading its response.

        do_GET = do_POST = do_DELETE = do_PATCH = handle_request

    return HTTPServer((host, port), Handler)


@contextmanager
def running_fake(state):
    server = make_server(state)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("system", choices=("dify", "ragflow"))
    p.add_argument("--seed", type=Path)
    p.add_argument("--port", type=int, default=9381)
    args = p.parse_args()
    state = FakeState(args.system)
    if args.seed:
        state.seed(args.seed)
    server = make_server(state, port=args.port)
    print(f"Offline {args.system} fixture on 127.0.0.1:{server.server_port}; key=fake-key", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
