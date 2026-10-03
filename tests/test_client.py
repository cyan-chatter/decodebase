from __future__ import annotations

import pytest

import httpx

from cfl.config import Settings
from cfl.core.client import OllamaClient
from tests.fakes import FakeOllama


def test_version(client, fake_ollama):
    v = client.version()
    assert isinstance(v, str)
    assert len(fake_ollama.call_log) >= 1


def test_tags(client):
    tags = client.tags()
    assert isinstance(tags, list)


def test_ps(client):
    models = client.ps()
    assert isinstance(models, list)


def test_fail_next(fake_ollama):
    fake_ollama.fail_next(2, status=503)
    settings = Settings()
    c = OllamaClient(settings, transport=fake_ollama.transport)
    with pytest.raises(httpx.HTTPStatusError):
        c.version()
    with pytest.raises(httpx.HTTPStatusError):
        c.version()
    v = c.version()
    assert isinstance(v, str)


def test_deterministic_embeddings(fake_ollama):
    e1 = fake_ollama._make_embedding("hello world")
    e2 = fake_ollama._make_embedding("hello world")
    e3 = fake_ollama._make_embedding("different text")
    assert e1 == e2
    assert e1 != e3
    assert len(e1) == fake_ollama.embed_dim
