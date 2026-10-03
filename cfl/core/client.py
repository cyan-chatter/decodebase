from __future__ import annotations

import json
import math
import random
import threading
import time
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self

import httpx

from cfl.config import Settings
from cfl.core.budget import TokenCounter, assert_fits
from cfl.core.errors import LLMTransportError, LLMValidationError
from cfl.core.trace_log import trace_log

GENERATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class GenResult:
    """Native Ollama metrics; duration fields are nanoseconds, latency is seconds."""

    text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    prompt_eval_duration: int
    eval_duration: int
    load_duration: int
    done_reason: str | None
    cached_prompt_tokens: int | None = None

    @property
    def uncached_prompt_tokens(self) -> int | None:
        return (
            None
            if self.cached_prompt_tokens is None
            else self.prompt_tokens - self.cached_prompt_tokens
        )


class _Retryable(LLMTransportError):
    pass


class OllamaClient:
    """Budgeted native Ollama requests. Consume or close streaming iterators."""

    def __init__(self, settings: Settings, *, transport: Any = None) -> None:
        assert_fits(0, 0, settings.num_ctx)
        if settings.template_overhead < 0:
            raise ValueError("Template overhead cannot be negative")
        self.settings = settings.model_copy(deep=True)
        self.counter = TokenCounter(self.settings)
        self._client = httpx.Client(
            timeout=httpx.Timeout(600, connect=10),
            base_url=settings.ollama_url,
            transport=transport,
        )

    def version(self) -> str:
        resp = self._client.get("/api/version")
        resp.raise_for_status()
        return resp.json()["version"]

    def tags(self) -> list[dict]:
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json()["models"]

    def ps(self) -> list[dict]:
        resp = self._client.get("/api/ps")
        resp.raise_for_status()
        return resp.json().get("models", [])

    def show(self, model: str) -> dict:
        resp = self._client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _status(response: httpx.Response) -> None:
        if response.status_code >= 500:
            raise _Retryable(f"Ollama HTTP {response.status_code}")
        if response.is_error:
            raise LLMTransportError(f"Ollama HTTP {response.status_code}: {response.text[:500]}")

    @staticmethod
    def _data(response: httpx.Response) -> dict:
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMValidationError("Ollama returned invalid JSON") from exc
        return OllamaClient._validate_data(data)

    @staticmethod
    def _validate_data(data: Any) -> dict:
        if not isinstance(data, dict):
            raise LLMValidationError("Ollama response must be an object")
        if error := data.get("error"):
            if "out of memory" in str(error).lower() or "oom" in str(error).lower():
                raise _Retryable(str(error))
            raise LLMTransportError(str(error))
        return data

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(2 * 2**attempt + random.uniform(0, 1))

    def _post(self, endpoint: str, body: dict) -> dict:
        for attempt in range(3):
            try:
                response = self._client.post(endpoint, json=body)
                self._status(response)
                return self._data(response)
            except (httpx.TransportError, _Retryable) as exc:
                if attempt == 2:
                    raise LLMTransportError(f"Ollama failed after 3 attempts: {exc}") from exc
                self._backoff(attempt)
        raise AssertionError("Unreachable")

    def _estimate(self, text: str, num_predict: int) -> int:
        estimate = self.counter.count(text) + self.settings.template_overhead
        assert_fits(estimate, num_predict, self.settings.num_ctx)
        return estimate

    def generate(
        self,
        prompt: str,
        system: str,
        *,
        fmt: dict | str | None = None,
        num_predict: int,
        task: str,
        symbol_id: str | None = None,
        stream: bool = False,
    ) -> GenResult | Iterator[str]:
        text = "\n\n".join(part for part in (system, prompt) if part)
        body = {"prompt": prompt, "system": system}
        return self._generation(
            "/api/generate", body, text, fmt, num_predict, task, symbol_id, stream
        )

    def chat(
        self,
        messages: list[dict],
        *,
        fmt: dict | str | None = None,
        num_predict: int,
        task: str,
        symbol_id: str | None = None,
        stream: bool = False,
    ) -> GenResult | Iterator[str]:
        # Include roles and tool metadata rather than counting only message content.
        text = json.dumps(messages, ensure_ascii=False)
        return self._generation(
            "/api/chat", {"messages": messages}, text, fmt, num_predict, task, symbol_id, stream
        )

    def _generation(
        self,
        endpoint: str,
        body: dict,
        text: str,
        fmt: dict | str | None,
        num_predict: int,
        task: str,
        symbol_id: str | None,
        stream: bool,
    ) -> GenResult | Iterator[str]:
        if num_predict <= 0:
            raise ValueError("num_predict must be positive")
        self._estimate(text, num_predict)  # Streaming callers fail before receiving an iterator.
        body = deepcopy(body)
        body.update(
            model=self.settings.gen_model,
            stream=stream,
            keep_alive=-1,
            options={
                "num_ctx": self.settings.num_ctx,
                "num_predict": num_predict,
                "temperature": self.settings.temperature,
                "seed": self.settings.seed,
            },
        )
        if fmt is not None:
            body["format"] = deepcopy(fmt)
        if self.settings.gen_disable_thinking:
            body["think"] = False
        iterator = self._run(endpoint, body, text, num_predict, task, symbol_id, stream)
        return iterator if stream else next(iterator)

    @staticmethod
    def _content(data: dict, endpoint: str) -> str:
        if endpoint == "/api/chat":
            message = data.get("message", {})
            if not isinstance(message, dict):
                raise LLMValidationError("Ollama message must be an object")
            value = message.get("content", "")
        else:
            value = data.get("response", "")
        if not isinstance(value, str):
            raise LLMValidationError("Ollama content must be text")
        return value

    def _finish(
        self,
        data: dict,
        content: str,
        text: str,
        estimate: int,
        num_predict: int,
        started: float,
        task: str,
        symbol_id: str | None,
        attempt: int,
    ) -> GenResult:
        metrics = {}
        for key in (
            "prompt_eval_count",
            "eval_count",
            "prompt_eval_duration",
            "eval_duration",
            "load_duration",
        ):
            value = data.get(key, 0 if key.endswith("duration") else None)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LLMValidationError(f"Invalid Ollama metric: {key}")
            metrics[key] = value
        cached = data.get("prompt_eval_cached_count")
        if cached is not None and (
            not isinstance(cached, int)
            or isinstance(cached, bool)
            or not 0 <= cached <= metrics["prompt_eval_count"]
        ):
            raise LLMValidationError("Invalid Ollama metric: prompt_eval_cached_count")
        latency = time.perf_counter() - started
        exceeded = metrics["prompt_eval_count"] + num_predict > self.settings.num_ctx
        trace_log(
            self.settings.state_dir,
            task=task,
            symbol_id=symbol_id,
            est_prompt_tokens=estimate,
            actual_prompt_tokens=metrics["prompt_eval_count"],
            output_tokens=metrics["eval_count"],
            latency_s=latency,
            validation="budget_exceeded" if exceeded else "ok",
            attempt=attempt,
            cached_prompt_tokens=cached,
            prompt_eval_duration=metrics["prompt_eval_duration"],
            eval_duration=metrics["eval_duration"],
            load_duration=metrics["load_duration"],
        )
        if self.counter.tokenizer is None:
            self.counter.calibrate(len(text), metrics["prompt_eval_count"])
        assert_fits(metrics["prompt_eval_count"], num_predict, self.settings.num_ctx)
        return GenResult(
            content,
            metrics["prompt_eval_count"],
            metrics["eval_count"],
            latency,
            metrics["prompt_eval_duration"],
            metrics["eval_duration"],
            metrics["load_duration"],
            data.get("done_reason"),
            cached,
        )

    def _run(
        self,
        endpoint: str,
        body: dict,
        text: str,
        num_predict: int,
        task: str,
        symbol_id: str | None,
        stream: bool,
    ) -> Iterator[Any]:
        with GENERATION_LOCK:
            started = time.perf_counter()
            for attempt in range(3):
                estimate = self._estimate(text, num_predict)
                emitted = False
                finished = False
                try:
                    if not stream:
                        response = self._client.post(endpoint, json=body)
                        self._status(response)
                        data = self._data(response)
                        if data.get("done") is not True:
                            raise LLMValidationError("Ollama generation did not finish")
                        result = self._finish(
                            data,
                            self._content(data, endpoint),
                            text,
                            estimate,
                            num_predict,
                            started,
                            task,
                            symbol_id,
                            attempt + 1,
                        )
                    else:
                        with self._client.stream("POST", endpoint, json=body) as response:
                            if response.is_error:
                                response.read()
                            self._status(response)
                            content = ""
                            for line in response.iter_lines():
                                if not line:
                                    continue
                                try:
                                    data = self._validate_data(json.loads(line))
                                except ValueError as exc:
                                    raise LLMValidationError("Invalid streaming JSON") from exc
                                piece = self._content(data, endpoint)
                                content += piece
                                if data.get("done") is True:
                                    self._finish(
                                        data,
                                        content,
                                        text,
                                        estimate,
                                        num_predict,
                                        started,
                                        task,
                                        symbol_id,
                                        attempt + 1,
                                    )
                                    finished = True
                                    if piece:
                                        yield piece
                                    return
                                if piece:
                                    emitted = True
                                    yield piece
                            raise LLMValidationError("Ollama stream ended without final metrics")
                except GeneratorExit:
                    if not finished:
                        trace_log(
                            self.settings.state_dir,
                            task=task,
                            symbol_id=symbol_id,
                            est_prompt_tokens=estimate,
                            actual_prompt_tokens=None,
                            output_tokens=None,
                            latency_s=time.perf_counter() - started,
                            validation="cancelled",
                            attempt=attempt + 1,
                        )
                    raise
                except (httpx.TransportError, LLMTransportError, LLMValidationError) as exc:
                    retry = (
                        isinstance(exc, (httpx.TransportError, _Retryable))
                        and not emitted
                        and attempt < 2
                    )
                    trace_log(
                        self.settings.state_dir,
                        task=task,
                        symbol_id=symbol_id,
                        est_prompt_tokens=estimate,
                        actual_prompt_tokens=None,
                        output_tokens=None,
                        latency_s=time.perf_counter() - started,
                        validation="retry" if retry else "error",
                        attempt=attempt + 1,
                    )
                    if retry:
                        self._backoff(attempt)
                        continue
                    if isinstance(exc, httpx.TransportError | _Retryable):
                        raise LLMTransportError(f"Ollama generation failed: {exc}") from exc
                    raise
                if not stream:
                    break
        # Release the lock before exposing the completed non-streaming result.
        yield result

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            assert_fits(self.counter.count(text), 0, self.settings.embed_max_tokens)
        for offset in range(0, len(texts), 32):
            batch = texts[offset : offset + 32]
            data = self._post(
                "/api/embed",
                {
                    "model": self.settings.embed_model,
                    "input": batch,
                    "truncate": False,
                    "keep_alive": -1,
                },
            )
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise LLMValidationError("Embedding count differs from input count")
            for vector in embeddings:
                if not isinstance(vector, list) or len(vector) != self.settings.embed_dim:
                    raise LLMValidationError("Embedding dimension differs from settings.embed_dim")
                if any(
                    not isinstance(n, int | float) or isinstance(n, bool) or not math.isfinite(n)
                    for n in vector
                ):
                    raise LLMValidationError("Embedding contains non-finite or nonnumeric values")
                vectors.append(vector)
        return vectors

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
