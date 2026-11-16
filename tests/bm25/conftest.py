import tempfile
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / ".artifacts" / "tmp"


@pytest.fixture(autouse=True)
def bm25_tempdir(monkeypatch):
    """Keep this suite's temporary files (including subprocesses') under .artifacts/tmp."""
    TEMP.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TMPDIR", str(TEMP))
    monkeypatch.setattr(tempfile, "tempdir", str(TEMP))


@pytest.fixture
def client():
    instance = QdrantClient(":memory:")
    yield instance
    instance.close()


@pytest.fixture
def corpus():
    return [
        {
            "id": "d1",
            "text": "Qdrant 向量数据库支持快照备份与恢复。",
            "metadata": {"tag": "backup"},
        },
        {"id": "d2", "text": "数据库检索使用稀疏向量，向量检索很快。"},
        {"id": 7, "text": "面包酵母烘焙。"},
        {"id": "7", "text": "数据库数据库管理。"},
    ]
