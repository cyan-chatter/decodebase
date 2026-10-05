from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time

import httpx


class _TrackedStream(httpx.SyncByteStream):
    def __init__(self, owner, data: bytes) -> None:
        self.owner = owner
        self.data = data
        self.closed = False

    def __iter__(self):
        yield from self.data.splitlines(keepends=True)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with self.owner._lock:
                self.owner._in_flight -= 1


class FakeOllama:
    def __init__(self, embed_dim: int = 768) -> None:
        self.source_answers = False
        self.review_override = None
        self.loaded_generator = "qwen3.5:9b"
        self.answer_text = None
        self.stable_summary = False
        self.invalid_outputs = 0
        self.kill_after = None
        self.generated = 0
        self.context_length = 8192
        self.delay = 0.0
        self.prompt_tokens = 100
        self.cached_prompt_tokens: int | None = 0
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
        body = json.loads(request.content) if request.content else {}
        self.call_log.append({"path": request.url.path, "body": body})
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            if self._fail_queue:
                status = self._fail_queue.pop(0)
                self._in_flight -= 1
                return httpx.Response(status, json={"error": "simulated failure"})

        transferred = False
        try:
            body = json.loads(request.content) if request.content else {}
            path = request.url.path
            time.sleep(self.delay)

            if path == "/api/version":
                resp_json = {"version": "0.35.0"}
            elif path == "/api/tags":
                resp_json = {
                    "models": [
                        {
                            "name": "qwen3.5:9b",
                            "model": "qwen3.5:9b",
                            "digest": "generator-digest",
                        },
                        {"name": "nomic-embed-text:latest", "digest": "embedder-digest"},
                        {"name": "qwen2.5-coder:7b", "digest": "verifier-digest"},
                    ]
                }
            elif path == "/api/ps":
                resp_json = {
                    "models": [
                        {
                            "name": self.loaded_generator,
                            "size": 5000000000,
                            "size_vram": 5000000000,
                            "context_length": self.context_length,
                        },
                        {
                            "name": "nomic-embed-text:latest",
                            "size": 300000000,
                            "size_vram": 300000000,
                        },
                    ]
                }
            elif path == "/api/generate":
                if body.get("keep_alive") == 0:
                    if self.loaded_generator == body.get("model"):
                        self.loaded_generator = None
                else:
                    self.loaded_generator = body.get("model", self.loaded_generator)
                if self.kill_after is not None and self.generated >= self.kill_after:
                    raise KeyboardInterrupt
                self.generated += 1
                self.context_length = body.get("options", {}).get("num_ctx", 8192)
                prompt = body.get("prompt", "")
                code = prompt
                summary = self._make_summary(code, self.stable_summary, body.get("options", {}))
                fmt = body.get("format", {})
                if isinstance(fmt, dict) and "claims" in fmt.get("properties", {}):
                    data = json.loads(prompt[prompt.index('{"question"') :])
                    source = data["sources"][0]
                    evidence = [
                        {
                            "id": source["id"],
                            "quote": next(
                                line for line in source["code"].splitlines() if line.strip()
                            ),
                        }
                    ]
                    value = {
                        "complete": True,
                        "missing": [],
                        "claims": [
                            {
                                "index": i,
                                "verdict": "supported",
                                "reason": "Test review fixture",
                                "evidence": evidence,
                            }
                            for i, _ in data["claims"]
                        ],
                    }
                    if self.review_override is not None:
                        value = (
                            self.review_override(data)
                            if callable(self.review_override)
                            else self.review_override
                        )
                    summary = json.dumps(value)
                elif isinstance(fmt, dict) and "explanation" in fmt.get("properties", {}):
                    data = json.loads(prompt[prompt.index('{"question"') :])
                    summary = json.dumps(
                        {
                            "explanation": "Source-grounded test explanation "
                            + data["sources"][0]["citation"],
                            "limitations": [],
                        }
                    )
                elif isinstance(fmt, dict) and "features" in fmt.get("properties", {}):
                    summary = json.dumps({"features": []})
                elif (
                    isinstance(fmt, dict)
                    and fmt.get("properties")
                    and "one_liner" not in fmt["properties"]
                ):
                    summary = json.dumps(
                        {
                            id: json.loads(self._make_summary(code + id, self.stable_summary))
                            for id in fmt["properties"]
                        }
                    )
                if self.invalid_outputs:
                    self.invalid_outputs -= 1
                    summary = "{invalid"
                resp_json = {
                    "response": summary,
                    "done": True,
                    "prompt_eval_count": self.prompt_tokens,
                    "prompt_eval_cached_count": self.cached_prompt_tokens,
                    "eval_count": 50,
                    "prompt_eval_duration": 1000000000,
                    "eval_duration": 2000000000,
                    "load_duration": 0,
                    "done_reason": "stop",
                }
            elif path == "/api/chat":
                messages = body.get("messages", [])
                code = messages[-1]["content"] if messages else ""
                summary = self._make_summary(code, False, body.get("options", {}))
                if self.source_answers:
                    cites = list(dict.fromkeys(re.findall(r"\[[^\]\n]+:\d+-\d+\]", code)))
                    summary = (
                        self.answer_text
                        if self.answer_text is not None
                        else "Source-grounded test answer: " + " ".join(cites[:5])
                    )
                resp_json = {
                    "message": {"role": "assistant", "content": summary},
                    "done": True,
                    "prompt_eval_count": self.prompt_tokens,
                    "prompt_eval_cached_count": self.cached_prompt_tokens,
                    "eval_count": 50,
                    "prompt_eval_duration": 1000000000,
                    "eval_duration": 2000000000,
                    "load_duration": 0,
                    "done_reason": "stop",
                }
            elif path == "/api/embed":
                input_data = body.get("input", body.get("prompt", ""))
                if isinstance(input_data, list):
                    embeddings = [self._make_embedding(t) for t in input_data]
                else:
                    embeddings = [self._make_embedding(input_data)]
                resp_json = {
                    "embeddings": embeddings,
                    "model": body.get("model", "nomic-embed-text"),
                }
            else:
                resp_json = {"error": "unknown endpoint"}
                return httpx.Response(404, json=resp_json)

            if body.get("stream") and path in ("/api/generate", "/api/chat"):
                key = "response" if path == "/api/generate" else "message"
                piece = {
                    key: "hello"
                    if key == "response"
                    else {"content": summary if self.source_answers else "hello"},
                    "done": False,
                }
                resp_json[key] = "" if key == "response" else {"content": ""}
                data = (json.dumps(piece) + "\n" + json.dumps(resp_json) + "\n").encode()
                transferred = True
                return httpx.Response(200, stream=_TrackedStream(self, data))
            return httpx.Response(200, json=resp_json)
        finally:
            if not transferred:
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
            "one_liner": one_liner[: 25 * 5],
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
