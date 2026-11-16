import hashlib
import json
import math
import os
import re
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .errors import BridgeError
from .filters import number
from .jsonutil import loads


def tokens(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Credentials must never follow redirects to another host.
        return None


def hash_embed(text, dimensions=64):
    vector = [0.0] * dimensions
    for word in tokens(text):
        digest = hashlib.sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:8], "big") % dimensions] += 1.0 if digest[8] & 1 else -1.0
    norm = math.sqrt(sum(x*x for x in vector))
    return [x / norm for x in vector] if norm else vector


def embed(text, config):
    if config["kind"] == "hash":
        return hash_embed(text, config["dimensions"])
    try:
        key = os.environ.get(config.get("api_key_env", "EMBEDDING_API_KEY"), "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        req = Request(config["url"], data=json.dumps({"model": config["model"], "input": text}).encode(), headers=headers)
        with build_opener(NoRedirect()).open(req, timeout=config.get("timeout", 10)) as response:
            raw = response.read(4*1024*1024 + 1)
            if len(raw) > 4*1024*1024:
                raise ValueError("Oversized embedding response")
            data = loads(raw)
        rows = data["data"]
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("index", 0) != 0:
            raise ValueError("Expected one embedding")
        vector = rows[0]["embedding"]
        if not isinstance(vector, list) or len(vector) != config["dimensions"] or not all(number(x) for x in vector):
            raise ValueError("Invalid embedding dimensions/values")
        return vector
    except Exception:
        # Do not echo endpoint URLs, payloads, tokens or provider errors.
        raise BridgeError(502, 3001, "Embedding backend unavailable or returned an invalid vector") from None
