import hmac
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import Backends, JsonlBackend
from .diagnostics import Scan
from .errors import BridgeError, ConfigError, invalid
from .filters import Predicate, number
from .jsonutil import loads

MAX_BODY = 1024 * 1024


def parse_json(raw):
    try:
        return loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise invalid("Malformed JSON") from None


class Application:
    def __init__(self, config, api_key):
        if not isinstance(api_key, str) or not api_key or len(api_key) > 4096 or not api_key.isascii() or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in api_key):
            raise ValueError("BRIDGE_API_KEY must be a non-empty visible ASCII token without whitespace (max 4096 characters)")
        self.config, self.api_key = config, api_key.encode("utf-8")
        self.backends = Backends(config)

    def _authorize(self, authorization):
        if not isinstance(authorization, str):
            raise BridgeError(401, 1001, "Expected Authorization: Bearer <token>")
        parts = authorization.split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1] or any(c.isspace() for c in parts[1]):
            raise BridgeError(401, 1001, "Expected Authorization: Bearer <token>")
        if not hmac.compare_digest(parts[1].encode("utf-8"), self.api_key):
            raise BridgeError(403, 1002, "Authorization failed")

    def _jsonl_request(self, authorization, raw, operation):
        self._authorize(authorization)
        body = parse_json(raw)
        if not isinstance(body, dict) or set(body) != {"knowledge_id"}:
            raise invalid(f"{operation} requires only knowledge_id")
        knowledge = body["knowledge_id"]
        if not isinstance(knowledge, str) or not 1 <= len(knowledge) <= 256:
            raise invalid("knowledge_id must be a string of 1..256 characters")
        if knowledge not in self.backends.items:
            raise BridgeError(404, 2001, "Knowledge base not found")
        backend = self.backends.items[knowledge]
        if not isinstance(backend, JsonlBackend):
            raise invalid(f"{operation} supports only JSONL knowledge bases")
        return knowledge, backend

    def snapshot_status(self, authorization, raw):
        knowledge, backend = self._jsonl_request(authorization, raw, "Snapshot status")
        return {"knowledge_id": knowledge, **backend.snapshot_status()}

    def reload(self, authorization, raw):
        knowledge, backend = self._jsonl_request(authorization, raw, "Reload")
        try:
            info = backend.reload()
        except ConfigError:
            raise BridgeError(503, 3001, "JSONL reload failed; previous corpus retained") from None
        return {"knowledge_id": knowledge, **info}

    def _retrieval_request(self, authorization, raw):
        self._authorize(authorization)
        body = parse_json(raw)
        if not isinstance(body, dict) or set(body) - {"knowledge_id", "query", "retrieval_setting", "metadata_condition"}:
            raise invalid("Expected knowledge_id, query, retrieval_setting and optional metadata_condition")
        knowledge, query, settings = body.get("knowledge_id"), body.get("query"), body.get("retrieval_setting")
        if not isinstance(knowledge, str) or not 1 <= len(knowledge) <= 256 or not isinstance(query, str) or not query.strip() or len(query) > 16384:
            raise invalid("knowledge_id and non-empty query are required strings (max 256/16384 characters)")
        if not isinstance(settings, dict) or set(settings) != {"top_k", "score_threshold"}:
            raise invalid("retrieval_setting requires top_k and score_threshold")
        top_k, threshold = settings["top_k"], settings["score_threshold"]
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise invalid("top_k must be an integer in 1..100")
        if not number(threshold) or not 0 <= threshold <= 1:
            raise invalid("score_threshold must be a finite number in 0..1")
        if knowledge not in self.config.knowledge:
            raise BridgeError(404, 2001, "Knowledge base not found")
        if "metadata_condition" in body and body["metadata_condition"] is None:
            raise invalid("metadata_condition must be an object when supplied")
        predicate = Predicate(body.get("metadata_condition"), self.config.knowledge[knowledge].metadata_allowlist)
        return knowledge, query, predicate, top_k, threshold

    def retrieve(self, authorization, raw):
        knowledge, query, predicate, top_k, threshold = self._retrieval_request(authorization, raw)
        return {"records": self.backends.items[knowledge].retrieve(query, predicate, top_k, threshold)}

    def diagnose(self, authorization, raw):
        knowledge, query, predicate, top_k, threshold = self._retrieval_request(authorization, raw)
        backend = self.backends.items[knowledge]
        scan = Scan()
        try:
            records = backend.retrieve(query, predicate, top_k, threshold, scan=scan)
        except BridgeError as exc:
            # Only the explicit budget failure has a complete, safe scan report.
            # Auth/validation/provider/internal failures never attach diagnostics.
            if (exc.status, exc.code) != (422, 2003):
                raise
            return exc.status, {**exc.body(), "diagnostics": scan.report(
                backend.config.backend, query, predicate, top_k, threshold)}
        return 200, {"records": records, "diagnostics": scan.report(
            backend.config.backend, query, predicate, top_k, threshold)}

    def close(self):
        self.backends.close()


class Server(ThreadingHTTPServer):
    daemon_threads = False
    request_queue_size = 16

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            body = b'{"error_code":3003,"error_msg":"Bridge busy"}'
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def make_server(app, host="127.0.0.1", port=8765):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ragwrench-dify-bridge/0.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            # Neither queries nor credentials go into access logs.
            pass

        def send_json(self, status, body):
            raw = json.dumps(body, ensure_ascii=True, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(raw)

        def send_error(self, code, message=None, explain=None):
            self.send_json(code, {"error_code": 2004, "error_msg": "Invalid HTTP request or unsupported method"})

        def do_GET(self):
            if self.path == "/healthz":
                self.send_json(200, {"status": "ok"})
            else:
                self.send_json(404, {"error_code": 2004, "error_msg": "Route not found"})

        def do_POST(self):
            try:
                if self.path not in {"/retrieval", "/retrieval/diagnostics", "/reload", "/snapshot/status"}:
                    raise BridgeError(404, 2004, "Route not found")
                if self.headers.get_all("Transfer-Encoding"):
                    raise invalid("Transfer-Encoding is unsupported; send Content-Length")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit() or len(lengths[0]) > 10:
                    raise invalid("One non-negative Content-Length is required")
                size = int(lengths[0])
                if size > MAX_BODY:
                    raise BridgeError(413, 2002, "Request exceeds 1 MiB")
                if self.headers.get_content_type() != "application/json":
                    raise BridgeError(415, 2002, "Content-Type must be application/json")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise invalid("Incomplete request body")
                auth = self.headers.get_all("Authorization", [])
                authorization = auth[0] if len(auth) == 1 else None
                if self.path == "/retrieval/diagnostics":
                    status, result = app.diagnose(authorization, raw)
                    self.send_json(status, result)
                    return
                action = {"/reload": app.reload, "/snapshot/status": app.snapshot_status,
                          "/retrieval": app.retrieve}[self.path]
                result = action(authorization, raw)
                self.send_json(200, result)
            except (BrokenPipeError, ConnectionResetError):
                return
            except BridgeError as exc:
                self.send_json(exc.status, exc.body())
            except (socket.timeout, TimeoutError):
                self.send_json(408, {"error_code": 2002, "error_msg": "Request read timeout"})
            except Exception:
                self.send_json(500, {"error_code": 3002, "error_msg": "Internal bridge error"})

        def do_PUT(self):
            self.send_json(405, {"error_code": 2004, "error_msg": "Method not allowed"})

        do_DELETE = do_PATCH = do_OPTIONS = do_PUT

    return Server((host, port), Handler)
