from __future__ import annotations

from typing import Any

import httpx

from cfl.config import Settings


class OllamaClient:
    """Thin synchronous wrapper around the Ollama HTTP API."""

    def __init__(self, settings: Settings, *, transport: Any = None) -> None:
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(600, connect=10),
            "base_url": settings.ollama_url,
        }
        if transport is not None:
            kwargs["transport"] = transport
        self._client = httpx.Client(**kwargs)

    # ------------------------------------------------------------------
    # Health / info
    # ------------------------------------------------------------------

    def version(self) -> str:
        """Return the Ollama server version string."""
        resp = self._client.get("/api/version")
        resp.raise_for_status()
        return resp.json()["version"]

    def tags(self) -> list[dict]:
        """Return all locally available model descriptors."""
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json()["models"]

    def ps(self) -> list[dict]:
        """Return currently loaded (resident) models."""
        resp = self._client.get("/api/ps")
        resp.raise_for_status()
        return resp.json().get("models", [])

    def show(self, model: str) -> dict:
        """Return model details from Ollama."""
        resp = self._client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
