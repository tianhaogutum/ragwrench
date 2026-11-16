"""TOML configuration; no dynamic imports, scripts, templating or interpolation."""
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ConfigError
from .filters import number

FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def http_url(value):
    if not isinstance(value, str):
        raise ConfigError("URL must be a string")
    u = urlsplit(value)
    if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ConfigError("URL requires http(s), hostname, and no credentials/query/fragment")
    return value.rstrip("/")


@dataclass(frozen=True)
class Knowledge:
    backend: str
    content_field: str = "content"
    title_field: str = "title"
    metadata_field: str = "metadata"
    metadata_allowlist: tuple = ()
    path: Path | None = None
    url: str | None = None
    api_key_env: str = "QDRANT_API_KEY"
    collection: str = ""
    vector_name: str | None = None
    distance: str = "cosine"
    score_scale: float = 1.0
    embedder: dict | None = None
    max_candidates: int = 10000
    overfetch: int = 4
    timeout: float = 10.0


@dataclass(frozen=True)
class Config:
    knowledge: dict[str, Knowledge]


def load_config(filename):
    filename = Path(filename).resolve()
    try:
        with filename.open("rb") as f:
            raw = tomllib.load(f)
        if set(raw) != {"knowledge"} or not isinstance(raw["knowledge"], dict) or not raw["knowledge"]:
            raise ConfigError("Config requires a non-empty [knowledge.<id>] mapping")
        result = {}
        for name, item in raw["knowledge"].items():
            if not isinstance(item, dict) or not name or len(name) > 256:
                raise ConfigError("Invalid knowledge mapping")
            if set(item) - set(Knowledge.__dataclass_fields__):
                raise ConfigError("Unknown knowledge config option")
            opts = dict(item)
            backend = opts.get("backend")
            if backend not in {"jsonl", "qdrant"}:
                raise ConfigError("backend must be jsonl or qdrant")
            allowed = opts.get("metadata_allowlist", [])
            if not isinstance(allowed, list) or any(not isinstance(v, str) or not FIELD.fullmatch(v) for v in allowed) or len(set(allowed)) != len(allowed):
                raise ConfigError("metadata_allowlist must contain unique simple field names")
            opts["metadata_allowlist"] = tuple(allowed)
            for field in ("content_field", "title_field", "metadata_field"):
                if field in opts and (not isinstance(opts[field], str) or not FIELD.fullmatch(opts[field])):
                    raise ConfigError("Payload fields must be simple names (nested paths unsupported)")
            for field, low, high in (("max_candidates", 1, 100000), ("overfetch", 1, 100)):
                if field in opts and (type(opts[field]) is not int or not low <= opts[field] <= high):
                    raise ConfigError(f"{field} outside supported range {low}..{high}")
            for field in ("score_scale", "timeout"):
                if field in opts and (not number(opts[field]) or opts[field] <= 0 or opts[field] > 3600):
                    raise ConfigError(f"{field} must be finite and in (0,3600]")
            if "path" in opts:
                if not isinstance(opts["path"], str):
                    raise ConfigError("path must be a string")
                opts["path"] = (filename.parent / opts["path"]).resolve()
                if not opts["path"].exists():
                    raise ConfigError("Configured path does not exist")
            if "url" in opts:
                opts["url"] = http_url(opts["url"])
            if backend == "jsonl":
                forbidden = {"url", "collection", "embedder", "vector_name", "distance", "api_key_env"} & set(item)
                if "path" not in opts or not opts["path"].is_file() or forbidden:
                    raise ConfigError("jsonl requires a file path and no Qdrant/embedder options")
            else:
                if ("path" in opts) == ("url" in opts):
                    raise ConfigError("qdrant requires exactly one of path or url")
                if "path" in opts and not opts["path"].is_dir():
                    raise ConfigError("Qdrant local path must be an existing directory")
                if not isinstance(opts.get("collection"), str) or not opts.get("collection"):
                    raise ConfigError("qdrant requires a collection")
                if "vector_name" in opts and (not isinstance(opts["vector_name"], str) or not opts["vector_name"]):
                    raise ConfigError("vector_name must be non-empty")
                if opts.get("distance", "cosine") not in {"cosine", "dot", "euclid"}:
                    raise ConfigError("Supported distances: cosine, dot, euclid")
                if not isinstance(opts.get("api_key_env", "QDRANT_API_KEY"), str):
                    raise ConfigError("api_key_env must name an environment variable")
                embed = opts.get("embedder", {"kind": "hash", "dimensions": 64})
                if not isinstance(embed, dict) or set(embed) - {"kind", "dimensions", "url", "model", "api_key_env", "timeout"}:
                    raise ConfigError("Invalid embedder options")
                kind = embed.get("kind")
                if kind not in {"hash", "http"}:
                    raise ConfigError("embedder kind must be hash or http")
                dim = embed.get("dimensions", 64)
                if type(dim) is not int or not 1 <= dim <= 65536:
                    raise ConfigError("embedder dimensions must be 1..65536")
                if kind == "hash" and set(embed) - {"kind", "dimensions"}:
                    raise ConfigError("hash embedder accepts only kind and dimensions")
                if kind == "http":
                    http_url(embed.get("url"))
                    if not isinstance(embed.get("model"), str) or not embed["model"]:
                        raise ConfigError("http embedder requires model")
                    if "api_key_env" in embed and not isinstance(embed["api_key_env"], str):
                        raise ConfigError("embedder api_key_env must be a string")
                    if not number(embed.get("timeout", 10)) or not 0 < embed.get("timeout", 10) <= 120:
                        raise ConfigError("embedder timeout must be in (0,120]")
                opts["embedder"] = dict(embed, dimensions=dim)
            result[name] = Knowledge(**opts)
        return Config(result)
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError("Cannot parse configuration; check TOML types and file access") from None
