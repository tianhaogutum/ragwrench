import json
import math
import os
import threading
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

from .diagnostics import Scan
from .embedding import embed, tokens
from .errors import BridgeError, ConfigError
from .filters import number
from .jsonutil import loads


def normalize(score, distance, scale=1.0):
    if not number(score):
        raise BridgeError(502, 3001, "Backend returned a non-finite score")
    if distance == "cosine":
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    if distance == "dot":
        x = score / scale
        return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    if distance in {"euclid", "bm25"}:
        if score < 0:
            raise BridgeError(502, 3001, "Backend returned a negative distance/BM25 score")
        return 1.0 / (1.0 + score / scale) if distance == "euclid" else score / (score + scale)
    raise ValueError("Unknown distance")


def record(payload, config, score):
    if not isinstance(payload, dict):
        raise BridgeError(502, 3001, "Chunk payload must be an object")
    content, title = payload.get(config.content_field), payload.get(config.title_field, "")
    metadata = payload.get(config.metadata_field, {})
    if not isinstance(content, str) or not isinstance(title, str) or not isinstance(metadata, dict):
        raise BridgeError(502, 3001, "Chunk content/title must be strings and metadata must be an object")
    output = {"content": content, "title": title, "score": score,
              "metadata": {k: metadata[k] for k in config.metadata_allowlist if k in metadata}}
    try:
        json.dumps(output, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise BridgeError(502, 3001, "Allowlisted metadata must contain finite JSON values") from None
    return output


@dataclass(frozen=True)
class _CorpusSnapshot:
    # Privately owned data: never mutate after publication or return it to callers.
    version: int
    records: tuple
    tf: tuple
    df: Counter
    lengths: tuple
    avgdl: float


class JsonlBackend:
    def __init__(self, config):
        self.config = config
        self._snapshot_lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self._snapshot = self._load_snapshot(1)

    def _load_snapshot(self, version):
        records, term_frequencies, df = [], [], Counter()
        try:
            # Open once so an atomic path replacement cannot mix file versions.
            # Bound reads as well as the initial size (a file may grow mid-load).
            with self.config.path.open("rb") as f:
                before = os.fstat(f.fileno())
                if before.st_size > 64 * 1024 * 1024:
                    raise ValueError("JSONL exceeds 64 MiB")
                total = 0
                while line := f.readline(1024 * 1024 + 1):
                    total += len(line)
                    if total > 64 * 1024 * 1024 or len(records) >= 100000 or len(line) > 1024 * 1024:
                        raise ValueError("JSONL exceeds record/line limit")
                    payload = loads(line.decode("utf-8"))
                    rec = record(payload, self.config, 0.0)
                    tf = Counter(tokens(rec["content"]))
                    records.append(rec)
                    term_frequencies.append(tf)
                    df.update(tf.keys())
                after = os.fstat(f.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("JSONL changed during loading; publish with atomic rename")
            lengths = tuple(sum(t.values()) for t in term_frequencies)
            avgdl = sum(lengths) / len(lengths) if lengths else 1.0
            return _CorpusSnapshot(version, tuple(records), tuple(term_frequencies), df, lengths, avgdl)
        except (OSError, ValueError, BridgeError, UnicodeError, RecursionError):
            raise ConfigError("Invalid JSONL corpus (JSON/object/content/title/metadata or size limit)") from None

    def _get_snapshot(self):
        with self._snapshot_lock:
            return self._snapshot

    def snapshot_status(self):
        # Capture once so the version and count describe the same publication.
        # No source I/O, indexing, retrieval or waiting for a reload build.
        snapshot = self._get_snapshot()
        return {"version": snapshot.version, "record_count": len(snapshot.records)}

    def reload(self):
        # Serialize writers, but keep the expensive parse/index build off the
        # reader lock. A failed build never changes the active pointer/version.
        with self._reload_lock:
            candidate = self._load_snapshot(self._get_snapshot().version + 1)
            with self._snapshot_lock:
                self._snapshot = candidate
            return {"version": candidate.version, "record_count": len(candidate.records)}

    def retrieve(self, query, predicate, top_k, threshold, *, scan=None):
        scan = scan if scan is not None else Scan()
        snapshot = self._get_snapshot()
        scan.snapshot_version = snapshot.version
        scan.candidates_fetched = len(snapshot.records)
        result = []
        n = len(snapshot.records)
        query_terms = sorted(set(tokens(query)))
        for rec, tf, length in zip(snapshot.records, snapshot.tf, snapshot.lengths):
            scan.candidates_evaluated += 1
            if not predicate.matches(rec["metadata"]):
                scan.filter_rejected += 1
                continue
            score = 0.0
            for term in query_terms:
                count = tf[term]
                if count:
                    idf = math.log(1.0 + (n - snapshot.df[term] + 0.5) / (snapshot.df[term] + 0.5))
                    score += idf * count * 2.5 / (count + 1.5 * (0.25 + 0.75 * length / snapshot.avgdl))
            # No lexical match means no result, including when threshold is zero.
            if score > 0:
                score = normalize(score, "bm25", self.config.score_scale)
                if score >= threshold:
                    result.append(dict(rec, score=score))
                    scan.eligible_count += 1
                else:
                    scan.threshold_rejected += 1
            else:
                scan.lexical_misses += 1
        result.sort(key=lambda r: r["score"], reverse=True)
        # Nested allowlisted metadata must not give library callers a mutable
        # reference into a snapshot used by other/in-flight queries.
        return scan.finish(deepcopy(result[:top_k]), "corpus_scanned")

class QdrantBackend:
    def __init__(self, config, client):
        self.config, self.client = config, client
        self.lock = threading.Lock()

    def _check_collection(self):
        info = self.client.get_collection(self.config.collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            params = vectors.get(self.config.vector_name)
        else:
            params = vectors if self.config.vector_name is None else None
        if params is None or params.size != self.config.embedder["dimensions"] or params.distance.value.lower() != self.config.distance:
            raise BridgeError(502, 3001, "Collection vector name/dimensions/distance does not match configuration")

    def retrieve(self, query, predicate, top_k, threshold, *, scan=None):
        scan = scan if scan is not None else Scan()
        try:
            vector = embed(query, self.config.embedder)
            prefilter = predicate.native(self.config.metadata_field)
            if prefilter is not None:
                scan.native_condition_count = len(prefilter.must or prefilter.should or [])
            result = []
            offset = 0
            batch = min(self.config.max_candidates, max(top_k * self.config.overfetch, 16))
            scan.candidate_budget = self.config.max_candidates
            scan.page_size = batch
            # Qdrant local uses SQLite; serialize access to shared local clients.
            with self.lock:
                self._check_collection()
                while offset < self.config.max_candidates:
                    limit = min(batch, self.config.max_candidates - offset)
                    points = self.client.query_points(
                        collection_name=self.config.collection, query=vector,
                        using=self.config.vector_name, query_filter=prefilter,
                        limit=limit, offset=offset, with_payload=True, with_vectors=False,
                    ).points
                    scan.pages_fetched += 1
                    scan.candidates_fetched += len(points)
                    for point in points:
                        scan.candidates_evaluated += 1
                        score = normalize(point.score, self.config.distance, self.config.score_scale)
                        # Distances are monotonic after normalization: all later points fail too.
                        if score < threshold:
                            scan.threshold_rejected += 1
                            return scan.finish(result, "score_threshold")
                        rec = record(point.payload, self.config, score)
                        if predicate.matches(rec["metadata"]):
                            scan.eligible_count += 1
                            result.append(rec)
                            if len(result) == top_k:
                                return scan.finish(result, "top_k")
                        else:
                            scan.filter_rejected += 1
                    offset += len(points)
                    if len(points) < limit:
                        return scan.finish(result, "source_exhausted")
                # Probe one point to distinguish actual exhaustion from the explicit budget.
                more = self.client.query_points(
                    collection_name=self.config.collection, query=vector, using=self.config.vector_name,
                    query_filter=prefilter, limit=1, offset=offset, with_payload=False, with_vectors=False,
                ).points
                scan.budget_probe_points = len(more)
                scan.candidates_fetched += len(more)
                if more and normalize(more[0].score, self.config.distance, self.config.score_scale) >= threshold:
                    scan.stop_reason = "candidate_budget"
                    raise BridgeError(422, 2003, "Candidate scan budget exhausted; narrow filters or raise max_candidates")
                return scan.finish(result, "score_threshold" if more else "source_exhausted")
        except BridgeError:
            raise
        except Exception:
            raise BridgeError(503, 3001, "Qdrant backend unavailable or incompatible") from None


class Backends:
    def __init__(self, config):
        self.items, self.clients = {}, {}
        try:
            for name, k in config.knowledge.items():
                if k.backend == "jsonl":
                    self.items[name] = JsonlBackend(k)
                else:
                    from qdrant_client import QdrantClient
                    identity = ("local", str(k.path)) if k.path else ("url", k.url, k.api_key_env, k.timeout)
                    if identity not in self.clients:
                        if k.path:
                            client = QdrantClient(path=str(k.path), force_disable_check_same_thread=True)
                        else:
                            client = QdrantClient(url=k.url, api_key=os.environ.get(k.api_key_env), timeout=k.timeout)
                        self.clients[identity] = client
                    backend = QdrantBackend(k, self.clients[identity])
                    # Shared lock for multiple knowledge IDs referencing the same local DB.
                    backend.lock = next((b.lock for b in self.items.values() if isinstance(b, QdrantBackend) and b.client is backend.client), backend.lock)
                    self.items[name] = backend
        except ModuleNotFoundError as exc:
            self.close()
            if exc.name == "qdrant_client":
                raise ConfigError("Qdrant backends need qdrant-client; install it with: "
                                  "pip install 'ragwrench[qdrant]'") from None
            raise ConfigError("Cannot initialize backend; check corpus, local lock and configuration") from None
        except Exception:
            self.close()
            raise ConfigError("Cannot initialize backend; check corpus, local lock and configuration") from None

    def close(self):
        for client in self.clients.values():
            client.close()
