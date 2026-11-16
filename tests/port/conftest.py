from pathlib import Path
import pytest

from ragwrench.port.api import API
from ragwrench.port.bundle import load_bundle, write_bundle

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/port/bundle"


@pytest.fixture
def example():
    return EXAMPLE


@pytest.fixture
def bundle(tmp_path, example):
    m, docs, chunks = load_bundle(example)
    path = tmp_path / "input"
    write_bundle(path, m["source"], m["settings"], docs, chunks)
    return path


@pytest.fixture
def api_factory():
    clients = []
    def factory(system, url, **kwargs):
        client = API(system, url, key=kwargs.pop("key", "fake-key"), page_size=kwargs.pop("page_size", 2),
                     backoff=0, retries=kwargs.pop("retries", 0), **kwargs)
        clients.append(client)
        return client
    yield factory
    for client in clients:
        client.close()
