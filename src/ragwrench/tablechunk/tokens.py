"""Explicit, deterministic approximate tokens; no downloaded vocabulary by default."""
import re
from typing import Callable

TokenCounter = Callable[[str], int]
# Individual CJK characters, runs of other letters/digits, and punctuation.
TOKEN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[^\W_\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def tiktoken_counter(encoding: str = "cl100k_base") -> TokenCounter:
    """Optional; tiktoken may fetch its vocabulary on first use (not used by tests)."""
    try:
        import tiktoken
    except ImportError as exc:
        raise ValueError("install ragwrench[tiktoken] to use this counter") from exc
    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))
