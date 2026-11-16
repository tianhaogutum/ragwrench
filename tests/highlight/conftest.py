import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("tests must not access the network")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def examples():
    return Path(__file__).resolve().parents[2] / "examples" / "highlight"
