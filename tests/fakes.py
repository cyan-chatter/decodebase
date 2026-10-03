from __future__ import annotations

import hashlib
import json
import random
import threading

import httpx


class FakeOllama:
    def __init__(self, embed_dim: int = 768) -> None:
        self.embed_dim = embed_dim
        self.call_log: list[dict] = []
        self.max_in_flight: int = 0
        self._lock = threading.Lock()
        self._in_flight: int = 0
        self._fail_queue: list[int] = []

    def fail_next(self, n: int, status: int = 503) -> None:
        """Cause next n calls to return HTTP status."""
        self._fail_queue.extend([status] * n)

    def _make_embedding(self, text: str) -> list[float]:
        """Deterministic embedding: seed RNG with hash of text."""
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.gauss(0, 1) for _ in range(self.embed_dim)]

    def _responder(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            if self._fail_queue:
                status = self._fail_queue.pop(0)
                self._in_flight -= 1
                return httpx.Response(status, json={"error": "simulated failure"})

        try:
            body = json.loads(request.content) if request.content else {}
            path = request.url.path
            entry = {"path": path, "body": body}

            if path == "/api/version":
                resp_json = {"version": "0.35.0"}
            elif path == "/api/tags":
                resp_json = {"models": [{"name": "qwen2.5-coder:7b", "model": "qwen2.5-coder:7b"}]}
            elif path == "/api/ps":
                resp_json = {"models": [{"name": "qwen2.5-coder:7b", "size": 5000000000, "size_vram": 5000000000}]}
            elif path == "/api/generate":
                prompt = body.get("prompt", "")
                code = prompt
                summary = self._make_summary(code, body.get("stream", False), body.get("options", {}))
                resp_json = {"response": summary, "done": True, "prompt_eval_count": 100, "eval_count": 50}
            elif path == "/api/chat":
                messages = body.get("messages", [])
                code = messages[-1]["content"] if messages else ""
                summary = self._make_summary(code, body.get("stream", False), body.get("options", {}))
                resp_json = {"message": {"role": "assistant", "content": summary}, "done": True, "prompt_eval_count": 100, "eval_count": 50}
            elif path == "/api/embed":
                input_data = body.get("input", body.get("prompt", ""))
                if isinstance(input_data, list):
                    embeddings = [self._make_embedding(t) for t in input_data]
                else:
                    embeddings = [self._make_embedding(input_data)]
                resp_json = {"embeddings": embeddings, "model": body.get("model", "nomic-embed-text")}
            else:
                resp_json = {"error": "unknown endpoint"}
                with self._lock:
                    self._in_flight -= 1
                return httpx.Response(404, json=resp_json)

            self.call_log.append(entry)
            return httpx.Response(200, json=resp_json)
        finally:
            with self._lock:
                self._in_flight -= 1

    def _make_summary(self, code: str, stable: bool = False, options: dict | None = None) -> str:
        """Return a schema-valid symbol summary JSON string."""
        if options is None:
            options = {}
        if options.get("stable_summary") or stable:
            one_liner = "Mock summary (stable)"
        else:
            one_liner = f"Mock summary of {hash(code)}"
        summary = {
            "one_liner": one_liner[:25 * 5],
            "purpose": "Mock purpose for testing.",
            "inputs": "Mock inputs.",
            "returns": "Mock returns.",
            "side_effects": [],
            "raises": [],
            "notable_logic": "Mock logic.",
        }
        return json.dumps(summary)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._responder)
