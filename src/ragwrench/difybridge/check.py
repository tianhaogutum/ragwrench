"""Black-box, read-only contract probes. No dependencies beyond Python stdlib."""
import json
import math
import uuid
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from .client_url import client_base_url
from .client_response import validate_content_encoding, validate_json_content_type, validate_transfer_encoding
from .config import http_url
from .credentials import validate_key
from .embedding import NoRedirect
from .errors import BridgeError, ConfigError
from .filters import Predicate, number
from .jsonutil import loads

MAX_REQUEST = 1024 * 1024
MAX_RESPONSE = 8 * 1024 * 1024


def _validate_timeout(timeout):
    """Accept only built-in numeric seconds; do not coerce caller objects."""
    # Check the range before isfinite to avoid converting huge integers.
    if type(timeout) not in (int, float) or not 0 < timeout <= 120 or not math.isfinite(timeout):
        raise ConfigError("Checker timeout must be a finite int or float in (0,120]")


def _validate_request_strings(value):
    """Check scalar strings/keys without normalizing or reserializing input."""
    if isinstance(value, str):
        value.encode("utf-8")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_request_strings(key)
            _validate_request_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_request_strings(item)


def _validate_response_values(value):
    """Validate decoded values/keys, including otherwise ignored fields."""
    if isinstance(value, str):
        # JSON combines valid surrogate escape pairs into scalar characters.
        # Strict encoding rejects any unpaired surrogate left after parsing.
        value.encode("utf-8")
    elif isinstance(value, float):
        # Valid JSON exponents can overflow the decoder's float to infinity.
        # Integers need no float conversion: arbitrarily large ones are finite.
        if not math.isfinite(value):
            raise ValueError("Non-finite response number")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_response_values(key)
            _validate_response_values(item)
    elif isinstance(value, list):
        for item in value:
            _validate_response_values(item)


def request(endpoint, payload, key=None, timeout=10, *, authorization=None):
    url = http_url(endpoint) + "/retrieval"
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if authorization is not None:
        headers["Authorization"] = authorization
    elif key is not None:
        headers["Authorization"] = "Bearer " + key
    req = Request(url, data=raw, headers=headers)
    try:
        response = build_opener(NoRedirect()).open(req, timeout=timeout)
    except HTTPError as exc:
        response = exc
    with response:
        validate_transfer_encoding(response.headers)
        validate_json_content_type(response.headers)
        validate_content_encoding(response.headers)
        raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise ValueError("Response exceeds 8 MiB")
        # Bounded HTTPResponse.read(n) may return a valid JSON prefix at EOF
        # without raising IncompleteRead. Check framing even on HTTP errors.
        lengths = response.headers.get_all("Content-Length", [])
        if (len(lengths) > 1 or
                (lengths and (not lengths[0].isascii() or not lengths[0].isdecimal() or
                              int(lengths[0]) != len(raw)))):
            raise ValueError("Invalid response Content-Length")
        # Decode strictly before JSON parsing: bytes input auto-detects UTF-16/32
        # and accepts encoded surrogates. Like snapshot-status, reject a BOM.
        body = loads(raw.decode("utf-8"))
        _validate_response_values(body)
        return response.status, body


def record_errors(body, top_k, threshold):
    if not isinstance(body, dict) or not isinstance(body.get("records"), list):
        return ["response must contain a records array"]
    records = body["records"]
    errors = []
    if len(records) > top_k:
        errors.append("records exceeds top_k")
    previous = float("inf")
    for rec in records:
        if not isinstance(rec, dict):
            errors.append("record must be an object")
            continue
        if not isinstance(rec.get("content"), str) or not isinstance(rec.get("title"), str):
            errors.append("content/title must be strings")
        if not isinstance(rec.get("metadata"), dict):
            errors.append("metadata must be a non-null object")
        score = rec.get("score")
        if not number(score) or not 0 <= score <= 1:
            errors.append("score must be finite and in 0..1")
        else:
            if score < threshold:
                errors.append("score below score_threshold")
            if score > previous:
                errors.append("scores must be descending")
            previous = score
    return sorted(set(errors))


def baseline_request(knowledge_id, query, metadata_condition, top_k, threshold, expect_count, expect_titles,
                     *, request_bytes=None):
    """Validate local inputs before any probes; never include their values in errors."""
    if (not isinstance(knowledge_id, str) or not 1 <= len(knowledge_id) <= 256 or
            not isinstance(query, str) or not query.strip() or len(query) > 16384):
        raise ConfigError("Checker requires knowledge_id of 1..256 and non-empty query of at most 16384 characters")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ConfigError("Checker top_k must be an integer in 1..100")
    if not number(threshold) or not 0 <= threshold <= 1:
        raise ConfigError("Checker score_threshold must be finite and in 0..1")
    if expect_count is not None and (type(expect_count) is not int or not 0 <= expect_count <= top_k):
        raise ConfigError("Checker expect_count must be an integer in 0..top_k")
    if not isinstance(expect_titles, (list, tuple)) or not all(isinstance(t, str) for t in expect_titles):
        raise ConfigError("Checker expect_titles must be a list or tuple of strings")
    if len(set(expect_titles)) > (top_k if expect_count is None else expect_count):
        raise ConfigError("Checker distinct expected titles exceed the expected or maximum record count")
    base = {"knowledge_id": knowledge_id, "query": query,
            "retrieval_setting": {"top_k": top_k, "score_threshold": threshold}}
    # Visit even filter values ignored by operators such as empty/not empty.
    # JSON's ASCII escaping would otherwise hide invalid surrogate code points
    # until UTF-8 probe construction, after some HTTP requests have been sent.
    try:
        _validate_request_strings([base, metadata_condition, expect_titles])
    except (UnicodeError, RecursionError):
        raise ConfigError("Checker request and expected titles must contain Unicode scalar strings and keys") from None
    if metadata_condition is not None:
        try:
            if not isinstance(metadata_condition, dict) or not isinstance(metadata_condition.get("conditions"), list):
                raise ValueError
            # The remote allowlist is unknown: validate structure/types locally,
            # leaving field authorization to the endpoint's baseline response.
            names = {c["name"] for c in metadata_condition["conditions"]
                     if isinstance(c, dict) and isinstance(c.get("name"), str)}
            Predicate(metadata_condition, names)
        except (BridgeError, ValueError, TypeError):
            raise ConfigError("Checker metadata_condition must be a valid flat Dify filter object") from None
        base["metadata_condition"] = metadata_condition
    try:
        raw = json.dumps(base, allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError):
        raise ConfigError("Checker request must contain finite, serializable JSON") from None
    if len(raw if request_bytes is None else request_bytes) > MAX_REQUEST:
        raise ConfigError("Checker baseline request exceeds 1 MiB")
    return base


def check_endpoint(endpoint, key, knowledge_id, query, *, allow_empty=False, timeout=10,
                   metadata_condition=None, top_k=3, score_threshold=0.0,
                   expect_count=None, expect_titles=()):
    """Return a JSON report, never echoing response text, query or credentials."""
    _validate_timeout(timeout)
    validate_key(key, context="Checker")
    endpoint = client_base_url(endpoint, context="Checker")
    base = baseline_request(knowledge_id, query, metadata_condition, top_k, score_threshold,
                            expect_count, expect_titles)
    return _check_endpoint(endpoint, key, base, base, allow_empty=allow_empty, timeout=timeout,
                           expect_count=expect_count, expect_titles=expect_titles)


def check_request_file(endpoint, key, path, *, allow_empty=False, timeout=10,
                       expect_count=None, expect_titles=()):
    """Read one bounded UTF-8 complete request and preserve its baseline bytes."""
    _validate_timeout(timeout)
    validate_key(key, context="Checker")
    endpoint = client_base_url(endpoint, context="Checker")
    try:
        with open(path, "rb") as source:
            raw = source.read(MAX_REQUEST + 1)
        if len(raw) > MAX_REQUEST:
            raise ValueError
        body = loads(raw.decode("utf-8"))
        required = {"knowledge_id", "query", "retrieval_setting"}
        if (not isinstance(body, dict) or not required <= set(body) or
                set(body) - required - {"metadata_condition"}):
            raise ValueError
        settings = body["retrieval_setting"]
        if not isinstance(settings, dict) or set(settings) != {"top_k", "score_threshold"}:
            raise ValueError
        if "metadata_condition" in body and not isinstance(body["metadata_condition"], dict):
            raise ValueError
    except (OSError, ValueError, RecursionError):
        raise ConfigError("Checker request file must be readable UTF-8 JSON of at most 1 MiB with a complete request schema") from None
    # Reuse all inline value/filter/assertion checks. Count the bytes actually
    # sent, rather than an ASCII-escaped serialization of a valid UTF-8 file.
    base = baseline_request(body["knowledge_id"], body["query"], body.get("metadata_condition"),
                            settings["top_k"], settings["score_threshold"], expect_count,
                            expect_titles, request_bytes=raw)
    return _check_endpoint(endpoint, key, base, raw, allow_empty=allow_empty, timeout=timeout,
                           expect_count=expect_count, expect_titles=expect_titles)


def _check_endpoint(endpoint, key, base, baseline_payload, *, allow_empty, timeout, expect_count, expect_titles):
    checks = []
    top_k = base["retrieval_setting"]["top_k"]
    score_threshold = base["retrieval_setting"]["score_threshold"]

    def probe(name, payload, auth=key, *, error=False, top_k=3, threshold=0.0, hits=False, authorization=None):
        try:
            status, body = request(endpoint, payload, auth, timeout, authorization=authorization)
            if error:
                failures = []
                if not 400 <= status <= 499:
                    failures.append("invalid request/auth must return a 4xx status")
                if not isinstance(body, dict) or type(body.get("error_code")) is not int or not isinstance(body.get("error_msg"), str) or not body.get("error_msg"):
                    failures.append("error must contain integer error_code and non-empty error_msg")
            else:
                failures = ([] if status == 200 else ["valid retrieval must return HTTP 200"])
                failures += record_errors(body, top_k, threshold)
                if hits and isinstance(body, dict) and isinstance(body.get("records"), list):
                    records = body["records"]
                    if not records and not allow_empty and expect_count != 0:
                        failures.append("no hits: provide a known matching query or use --allow-empty (weaker check)")
                    if expect_count is not None and len(records) != expect_count:
                        failures.append("baseline record count does not match --expect-count")
                    titles = {r["title"] for r in records if isinstance(r, dict) and isinstance(r.get("title"), str)}
                    if not set(expect_titles) <= titles:
                        failures.append("baseline is missing an --expect-title")
            checks.append({"name": name, "passed": not failures, "errors": failures})
        except Exception:
            checks.append({"name": name, "passed": False, "errors": ["request failed or response is not bounded valid JSON"]})

    probe("retrieval shape and ordering", baseline_payload, hits=True, top_k=top_k, threshold=score_threshold)
    probe("top_k=1", dict(base, retrieval_setting={"top_k": 1, "score_threshold": 0.0}), top_k=1)
    probe("threshold=0.9", dict(base, retrieval_setting={"top_k": 3, "score_threshold": 0.9}), threshold=0.9)
    probe("threshold=1", dict(base, retrieval_setting={"top_k": 3, "score_threshold": 1.0}), threshold=1.0)
    probe("missing authorization", base, None, error=True)
    probe("wrong authorization", base, "invalid-" + uuid.uuid4().hex, error=True)
    probe("malformed authorization", base, error=True, authorization="Basic invalid")
    probe("unknown knowledge", dict(base, knowledge_id="__conformance_missing_" + uuid.uuid4().hex), error=True)
    probe("malformed JSON", b"{", error=True)
    probe("missing required fields", {}, error=True)
    probe("top_k=0", dict(base, retrieval_setting={"top_k": 0, "score_threshold": 0.0}), error=True)
    probe("invalid threshold", dict(base, retrieval_setting={"top_k": 1, "score_threshold": 1.1}), error=True)
    probe("unknown comparison operator", dict(base, metadata_condition={"conditions": [
        {"name": "__probe", "comparison_operator": "exec", "value": "x"}]}), error=True)
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "limitations": ["Finite black-box probes cannot prove retrieval relevance or complete filter semantics."] +
            (["Empty results allowed: record checks may be vacuous."] if allow_empty else []) +
            (["Empty baseline expected: baseline record checks may be vacuous."] if expect_count == 0 else [])}
