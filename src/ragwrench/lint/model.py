from dataclasses import asdict, dataclass, field
from typing import Any


class InputError(ValueError):
    """Invalid input or unsupported configuration."""


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_hint: int | None = None
    end_hint: int | None = None

    def record(self):
        value = {"id": self.id, "text": self.text, "metadata": self.metadata}
        if self.start_hint is not None and self.end_hint is not None:
            value.update(start_char_idx=self.start_hint, end_char_idx=self.end_hint)
        return value


@dataclass
class Span:
    start_char: int
    end_char: int
    start_line: int
    end_line: int


@dataclass
class Alignment:
    chunk_id: str
    status: str
    spans: list[Span] = field(default_factory=list)
    matched_chars: int = 0
    total_chars: int = 0
    candidates: int = 0

    def record(self):
        return asdict(self)


@dataclass
class Issue:
    rule: str
    severity: str
    chunk_id: str | None
    message: str
    line: int = 1
    related: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def record(self):
        return asdict(self)


RULES = {
    "CL001": ("error", "Unbalanced code fence"),
    "CL002": ("warning", "Table fragment without header"),
    "CL003": ("warning", "Orphan or trailing heading"),
    "CL004": ("warning", "Mid-sentence boundary"),
    "CL005": ("warning", "Size outlier"),
    "CL006": ("warning", "Duplicate or excessive overlap"),
    "CL007": ("error", "Source coverage or alignment failure"),
    "CL008": ("warning", "Split list"),
}


@dataclass
class Config:
    min_chars: int = 40
    max_chars: int = 4000
    near_duplicate: float = 0.90
    max_overlap: float = 0.50
    disable: list[str] = field(default_factory=list)
    fail_on: str = "error"
    max_pair_checks: int = 100000

    def validate(self):
        if (isinstance(self.min_chars, bool) or not isinstance(self.min_chars, int)
                or self.min_chars < 0):
            raise InputError("min_chars must be a non-negative integer")
        if (isinstance(self.max_chars, bool) or not isinstance(self.max_chars, int)
                or self.max_chars < max(1, self.min_chars)):
            raise InputError("max_chars must be >= min_chars and positive")
        for name in ("near_duplicate", "max_overlap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise InputError(f"{name} must be in (0, 1]")
        if not isinstance(self.disable, list) or any(not isinstance(x, str) or x not in RULES for x in self.disable):
            raise InputError("disable must be an array of rule IDs CL001..CL008")
        if self.fail_on not in ("error", "warning", "note", "none"):
            raise InputError("fail_on must be error, warning, note, or none")
        if (isinstance(self.max_pair_checks, bool) or not isinstance(self.max_pair_checks, int)
                or self.max_pair_checks < 1):
            raise InputError("max_pair_checks must be a positive integer")
        return self
