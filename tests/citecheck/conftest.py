import json
from pathlib import Path

import pytest

from ragwrench.citecheck.adapters import adapt

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def example():
    def load(format):
        return adapt(json.loads((ROOT / "examples" / "cite-check" / f"{format}_response.json").read_text()), format)
    return load
