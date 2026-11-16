"""Deterministic, model-free tokenizers. No stemming or language detection."""

from __future__ import annotations

import importlib
import unicodedata
from pathlib import Path

TOKENIZERS = ("whitespace", "unicode", "cjk-bigram", "jieba")
TOKENIZER_VERSION = 1
MAX_TEXT_CHARS = 1_000_000


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def is_cjk(char: str) -> bool:
    n = ord(char)
    return any(
        lo <= n <= hi
        for lo, hi in (
            (0x3040, 0x30FF),
            (0x31F0, 0x31FF),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x323AF),
            (0x1100, 0x11FF),
            (0x3130, 0x318F),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7FF),
            (0x1B000, 0x1B16F),
            (0x3007, 0x3007),
        )
    )


def word_runs(text: str, separate_cjk: bool = False) -> list[tuple[str, bool]]:
    result: list[tuple[str, bool]] = []
    run = ""
    cjk = False
    for char in text:
        category = unicodedata.category(char)[0]
        char_cjk = is_cjk(char)
        if category not in "LN" and not (category == "M" and run):
            if run:
                result.append((run, cjk))
            run = ""
            continue
        if run and separate_cjk and char_cjk != cjk and category != "M":
            result.append((run, cjk))
            run = ""
        if not run:
            cjk = char_cjk
        run += char
    if run:
        result.append((run, cjk))
    return result


def read_stopwords(path: str | Path) -> tuple[str, ...]:
    with Path(path).open("rb") as stream:
        data = stream.read(1_048_577)
    if len(data) > 1_048_576:
        raise ValueError("stopword file exceeds 1 MiB")
    # Literal emitted tokens, one per line; no executable config or regex.
    words = {normalize(line.strip()) for line in data.decode("utf-8").splitlines()}
    words.discard("")
    if any(any(c.isspace() for c in word) for word in words):
        raise ValueError("each stopword must be a single token without whitespace")
    return tuple(sorted(words))


class Tokenizer:
    def __init__(self, name: str = "cjk-bigram", stopwords: tuple[str, ...] = ()):
        if name not in TOKENIZERS:
            raise ValueError(f"unsupported tokenizer: {name}")
        self.name = name
        self.stopwords = frozenset(normalize(word) for word in stopwords)
        self._jieba = None
        if name == "jieba":
            try:
                module = importlib.import_module("jieba")
            except ImportError as exc:
                raise ValueError("jieba is not installed; install it with: pip install 'ragwrench[jieba]'") from exc
            self._jieba = module.Tokenizer()

    def __call__(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters")
        text = normalize(text)
        if self.name == "whitespace":
            tokens = text.split()
        elif self.name == "unicode":
            tokens = [run for run, _ in word_runs(text)]
        elif self.name == "cjk-bigram":
            tokens = []
            for run, cjk in word_runs(text, separate_cjk=True):
                if cjk and len(run) > 1:
                    tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
                else:
                    tokens.append(run)
        else:
            # HMM disabled: fixed bundled dictionary, no learned assets/network.
            tokens = [
                token
                for token in self._jieba.cut(text, HMM=False)
                if any(char.isalnum() for char in token)
            ]
        return [token for token in tokens if token not in self.stopwords]
