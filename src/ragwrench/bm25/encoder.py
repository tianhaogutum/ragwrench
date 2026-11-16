"""BM25 TF-side weights and stable sparse dimensions, independent of Qdrant."""

from __future__ import annotations

import hashlib
import math
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Iterable

from .tokenizers import TOKENIZER_VERSION, Tokenizer

HASH_VERSION = "sha256-first32-big-endian-v1"


def token_hash(token: str) -> int:
    """Unsigned uint32 of the first four SHA-256 bytes of UTF-8, big endian."""
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]

    def as_dict(self) -> dict:
        return asdict(self)


class Encoder:
    def __init__(
        self,
        tokenizer: str = "cjk-bigram",
        *,
        k1: float = 1.2,
        b: float = 0.75,
        avgdl: float = 100.0,
        stopwords: Iterable[str] = (),
    ):
        for name, value in (("k1", k1), ("b", b), ("avgdl", avgdl)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if k1 <= 0 or avgdl <= 0 or not 0 <= b <= 1:
            raise ValueError("require k1 > 0, avgdl > 0 and 0 <= b <= 1")
        self.k1, self.b, self.avgdl = float(k1), float(b), float(avgdl)
        self.tokenizer = Tokenizer(tokenizer, tuple(stopwords))
        self._vocabulary: dict[int, set[str]] = defaultdict(set)

    @property
    def config(self) -> dict:
        return {
            "schema": 1,
            "tokenizer": self.tokenizer.name,
            "tokenizer_version": TOKENIZER_VERSION,
            "unicode_version": unicodedata.unidata_version,
            "jieba_version": version("jieba") if self.tokenizer.name == "jieba" else None,
            "hash": HASH_VERSION,
            "stopwords": sorted(self.tokenizer.stopwords),
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
        }

    @classmethod
    def from_config(cls, config: dict) -> Encoder:
        if not isinstance(config, dict):
            raise ValueError("invalid collection encoder config")
        # Dict equality treats True and 1.0 as version 1; require JSON integers
        # before the full saved-config/runtime compatibility comparison below.
        for field in ("schema", "tokenizer_version"):
            if type(config.get(field)) is not int:
                raise ValueError(f"invalid collection encoder config: {field} must be an integer")
        try:
            encoder = cls(
                config["tokenizer"],
                k1=config["k1"],
                b=config["b"],
                avgdl=config["avgdl"],
                stopwords=config["stopwords"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid collection encoder config") from exc
        if encoder.config != config:
            raise ValueError("collection config/version mismatch; rebuild with this runtime")
        return encoder

    def _encode(self, text: str, query: bool) -> SparseVector:
        tokens = self.tokenizer(text)
        if not tokens:
            warnings.warn("empty or stopword-only text skipped", UserWarning, stacklevel=3)
            return SparseVector([], [])
        counts = Counter(tokens)
        weights: dict[int, float] = defaultdict(float)
        for token, count in counts.items():
            index = token_hash(token)
            bucket = self._vocabulary[index]
            if bucket and token not in bucket:
                warnings.warn(
                    f"32-bit hash collision at {index}: {sorted(bucket)!r}, {token!r}",
                    UserWarning,
                    stacklevel=3,
                )
            bucket.add(token)
            if query:
                weights[index] = 1.0
            else:
                weight = (
                    count
                    * (self.k1 + 1)
                    / (count + self.k1 * (1 - self.b + self.b * len(tokens) / self.avgdl))
                )
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("BM25 parameters produce non-finite or underflowed TF weights")
                weights[index] += weight
        indices = sorted(weights)
        return SparseVector(indices, [weights[index] for index in indices])

    def encode_documents(self, documents: Iterable[str]) -> list[SparseVector]:
        if isinstance(documents, str):
            raise ValueError("encode_documents expects an iterable of strings, not one string")
        return [self._encode(text, query=False) for text in documents]

    def encode_query(self, query: str) -> SparseVector:
        return self._encode(query, query=True)

    @property
    def collision_stats(self) -> dict:
        collisions = {
            str(index): sorted(tokens)
            for index, tokens in self._vocabulary.items()
            if len(tokens) > 1
        }
        return {
            "unique_tokens": sum(map(len, self._vocabulary.values())),
            "unique_indices": len(self._vocabulary),
            "collision_count": sum(len(tokens) - 1 for tokens in collisions.values()),
            "collisions": collisions,
        }
