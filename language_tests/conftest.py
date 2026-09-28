"""Library validation is offline; live providers require a separate explicit run."""
import os
import socket
import pytest

os.environ['HAYSTACK_TELEMETRY_ENABLED'] = 'false'
os.environ['ECHORA_RUN_FISH_LIVE'] = '0'

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError('Network access is disabled in the language test suite')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)
