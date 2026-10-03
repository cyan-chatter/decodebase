from __future__ import annotations

import httpx
import pytest

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


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch, request):
    if request.node.name == "test_backoff_is_exponential_with_jitter":
        return
    monkeypatch.setattr(OllamaClient, "_backoff", lambda *args: None)


def logs(client):
    import json
    from pathlib import Path

    return [
        json.loads(line)
        for file in (Path(client.settings.state_dir) / "logs").glob("*.jsonl")
        for line in file.read_text().splitlines()
    ]


@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("stream", [False, True])
def test_oversized_never_sends(client, fake_ollama, method, stream):
    from cfl.core.errors import BudgetExceeded

    text = "x" * 100000
    with pytest.raises(BudgetExceeded):
        if method == "generate":
            client.generate("small", text, num_predict=350, task="test", stream=stream)
        else:
            client.chat(
                [{"role": "user", "content": "small", "tool_calls": text}],
                num_predict=350,
                task="test",
                stream=stream,
            )
    assert fake_ollama.call_log == []


@pytest.mark.parametrize("method", ["generate", "chat"])
def test_native_options_metrics_and_trace(client, fake_ollama, method):
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    if method == "generate":
        result = client.generate(
            "source",
            "system",
            fmt=schema,
            num_predict=350,
            task="symbol",
            symbol_id="src::function",
        )
    else:
        result = client.chat(
            [{"role": "user", "content": "source"}],
            fmt=schema,
            num_predict=350,
            task="symbol",
            symbol_id="src::function",
        )
    body = fake_ollama.call_log[-1]["body"]
    assert body["options"] == {
        "num_ctx": client.settings.num_ctx,
        "num_predict": 350,
        "temperature": client.settings.temperature,
        "seed": client.settings.seed,
    }
    assert body["keep_alive"] == -1 and body["think"] is False
    assert body["format"] == schema and body["stream"] is False
    assert result.prompt_tokens == 100 and result.output_tokens == 50
    assert result.prompt_eval_duration == 1000000000
    assert result.eval_duration == 2000000000 and result.done_reason == "stop"
    record = logs(client)[-1]
    assert record["actual_prompt_tokens"] == 100
    assert record["output_tokens"] == 50 and record["validation"] == "ok"
    assert record["task"] == "symbol" and record["symbol_id"] == "src::function"
    assert record["est_prompt_tokens"] > 0 and record["latency_s"] >= 0
    assert "source" not in str(record)


@pytest.mark.parametrize("stream", [False, True])
def test_cached_tokens_are_preserved_in_metrics_and_trace(client, fake_ollama, stream):
    fake_ollama.cached_prompt_tokens = 75
    result = client.generate("source", "system", num_predict=64, task="cache", stream=stream)
    if stream:
        list(result)
    else:
        assert result.prompt_tokens == 100 and result.cached_prompt_tokens == 75
        assert result.uncached_prompt_tokens == 25
    record = logs(client)[-1]
    assert record["actual_prompt_tokens"] == 100 and record["cached_prompt_tokens"] == 75
    assert record["prompt_eval_duration"] == 1000000000


@pytest.mark.parametrize("cached", [-1, 101, True, "75"])
def test_invalid_cached_token_metric_rejected(client, fake_ollama, cached):
    from cfl.core.errors import LLMValidationError

    fake_ollama.cached_prompt_tokens = cached
    with pytest.raises(LLMValidationError, match="prompt_eval_cached_count"):
        client.generate("source", "", num_predict=64, task="cache")


def test_unknown_and_nearly_complete_cache_do_not_inflate_benchmark(client, fake_ollama):
    from cfl.core.preflight import warm_and_benchmark

    for cached in [None, 99]:
        fake_ollama.cached_prompt_tokens = cached
        assert warm_and_benchmark(client, client.settings).prefill_tps == 0
    fake_ollama.cached_prompt_tokens = 60
    bench = warm_and_benchmark(client, client.settings)
    assert bench.prefill_tps == 40 and bench.extra["cached_prompt_tokens"] == 60


def test_cached_tokens_still_count_toward_context_budget(client, fake_ollama):
    from cfl.core.errors import BudgetExceeded

    fake_ollama.prompt_tokens = client.settings.num_ctx
    fake_ollama.cached_prompt_tokens = client.settings.num_ctx - 1
    with pytest.raises(BudgetExceeded):
        client.generate("small", "", num_predict=64, task="cache")


@pytest.mark.parametrize("failures", [2, 3])
@pytest.mark.parametrize("stream", [False, True])
def test_retries(client, fake_ollama, failures, stream):
    from cfl.core.errors import LLMTransportError

    fake_ollama.fail_next(failures, 503)

    def invoke():
        result = client.generate("code", "", num_predict=100, task="retry", stream=stream)
        return list(result) if stream else result

    if failures == 3:
        with pytest.raises(LLMTransportError):
            invoke()
    else:
        assert invoke()
    assert len(fake_ollama.call_log) == 3
    assert len(logs(client)) == 3
    assert all(
        entry["body"]["options"]["num_ctx"] == client.settings.num_ctx
        for entry in fake_ollama.call_log
    )


def test_4xx_not_retried(client, fake_ollama):
    from cfl.core.errors import LLMTransportError

    fake_ollama.fail_next(3, 400)
    with pytest.raises(LLMTransportError):
        client.generate("code", "", num_predict=100, task="test")
    assert len(fake_ollama.call_log) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_reported_budget_overflow(client, fake_ollama, stream):
    from cfl.core.errors import BudgetExceeded

    fake_ollama.prompt_tokens = client.settings.num_ctx - 50
    with pytest.raises(BudgetExceeded):
        result = client.generate("code", "", num_predict=100, task="test", stream=stream)
        if stream:
            list(result)
    assert len(fake_ollama.call_log) == 1
    assert logs(client)[-1]["validation"] == "budget_exceeded"


@pytest.mark.parametrize("method", ["generate", "chat"])
def test_streaming_final_metrics(client, fake_ollama, method):
    if method == "generate":
        stream = client.generate("code", "", num_predict=100, task="test", stream=True)
    else:
        stream = client.chat(
            [{"role": "user", "content": "code"}], num_predict=100, task="test", stream=True
        )
    assert "".join(stream) == "hello"
    assert logs(client)[-1]["actual_prompt_tokens"] == 100


def test_global_generation_lock_across_clients(tmp_path, fake_ollama):
    from concurrent.futures import ThreadPoolExecutor

    fake_ollama.delay = 0.02
    settings = Settings(state_dir=str(tmp_path))
    with (
        OllamaClient(settings, transport=fake_ollama.transport) as first,
        OllamaClient(settings, transport=fake_ollama.transport) as second,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [
            pool.submit(first.generate, "code", "", num_predict=100, task="test"),
            pool.submit(
                second.chat, [{"role": "user", "content": "code"}], num_predict=100, task="test"
            ),
        ]
        assert all(future.result(timeout=2).text for future in futures)
    assert fake_ollama.max_in_flight == 1


def test_stream_holds_lock_until_closed(client, fake_ollama):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from cfl.core.client import GENERATION_LOCK

    stream = client.generate("code", "", num_predict=100, task="test", stream=True)
    assert next(stream) == "hello"
    assert not GENERATION_LOCK.acquire(blocking=False)
    started = threading.Event()

    def invoke():
        started.set()
        return client.generate("second", "", num_predict=100, task="test")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(invoke)
        assert started.wait(timeout=2)
        assert len(fake_ollama.call_log) == 1
        stream.close()
        assert future.result(timeout=2).text
    assert len(fake_ollama.call_log) == 2


def test_embeddings_batch_and_dimension_guard(client, fake_ollama):
    from cfl.core.errors import LLMValidationError

    texts = [f"text {i}" for i in range(65)]
    vectors = client.embed(texts)
    assert len(vectors) == 65 and vectors[0] == fake_ollama._make_embedding(texts[0])
    assert [len(entry["body"]["input"]) for entry in fake_ollama.call_log] == [32, 32, 1]
    assert all(entry["body"]["truncate"] is False for entry in fake_ollama.call_log)
    assert client.embed([]) == []
    fake_ollama.embed_dim = 3
    with pytest.raises(LLMValidationError, match="dimension"):
        client.embed(["test"])


def test_benchmark_uses_guarded_client(client, fake_ollama):
    from cfl.core.preflight import warm_and_benchmark

    bench = warm_and_benchmark(client, client.settings)
    assert bench.prefill_tps == 100 and bench.gen_tps == 25
    assert bench.embed_dim == 768
    assert [entry["path"] for entry in fake_ollama.call_log] == ["/api/generate", "/api/embed"]
    assert fake_ollama.call_log[0]["body"]["options"]["seed"] == client.settings.seed
    assert logs(client)[-1]["task"] == "benchmark"


@pytest.mark.parametrize("failure", ["connect", "oom"])
def test_connect_and_oom_retry(tmp_path, failure):

    fake = FakeOllama()
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) < 3:
            if failure == "connect":
                raise httpx.ConnectError("unreachable", request=request)
            return httpx.Response(200, json={"error": "out of memory"})
        return fake._responder(request)

    with OllamaClient(
        Settings(state_dir=str(tmp_path)), transport=httpx.MockTransport(handler)
    ) as client:
        assert client.generate("code", "", num_predict=100, task="test").text
    assert len(attempts) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"done": True, "response": 42},
        {"done": True, "prompt_eval_count": -1, "eval_count": 2},
    ],
)
def test_invalid_responses(tmp_path, payload):
    from cfl.core.errors import LLMValidationError

    with (
        OllamaClient(
            Settings(state_dir=str(tmp_path)),
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        ) as client,
        pytest.raises(LLMValidationError),
    ):
        client.generate("code", "", num_predict=100, task="test")


@pytest.mark.parametrize("error", ['{"error":"out of memory"}', "{broken", ""])
def test_stream_failure_after_output_never_replays(tmp_path, error):
    from cfl.core.errors import LLMTransportError, LLMValidationError

    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(200, content='{"response":"partial","done":false}\n' + error + "\n")

    with OllamaClient(
        Settings(state_dir=str(tmp_path)), transport=httpx.MockTransport(handler)
    ) as client:
        stream = client.generate("code", "", num_predict=100, task="test", stream=True)
        assert next(stream) == "partial"
        with pytest.raises((LLMTransportError, LLMValidationError)):
            list(stream)
    assert len(attempts) == 1


def test_cancelled_stream_is_traced(client):
    stream = client.generate("code", "", num_predict=100, task="test", stream=True)
    next(stream)
    stream.close()
    assert logs(client)[-1]["validation"] == "cancelled"


def test_backoff_is_exponential_with_jitter(monkeypatch):
    import cfl.core.client as module

    sleeps = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module.random, "uniform", lambda lower, upper: 0.25)
    OllamaClient._backoff(0)
    OllamaClient._backoff(1)
    assert sleeps == [2.25, 4.25]


def test_stream_snapshots_messages_before_budget_check(client, fake_ollama):
    messages = [{"role": "user", "content": "code"}]
    stream = client.chat(messages, num_predict=100, task="test", stream=True)
    messages[0]["content"] = "x" * 100000
    assert "".join(stream) == "hello"
    assert fake_ollama.call_log[-1]["body"]["messages"][0]["content"] == "code"


@pytest.mark.parametrize("stream", [False, True])
def test_chat_bad_message_wrapped(tmp_path, stream):
    from cfl.core.errors import LLMValidationError

    payload = '{"message":null,"done":true,"prompt_eval_count":10,"eval_count":1}\n'
    with (
        OllamaClient(
            Settings(state_dir=str(tmp_path)),
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload)),
        ) as instance,
        pytest.raises(LLMValidationError, match="message"),
    ):
        result = instance.chat([], num_predict=100, task="test", stream=stream)
        if stream:
            list(result)


@pytest.mark.parametrize("vectors", [[], [[0.0] * 767], [[float("nan")] * 768], [["bad"] * 768]])
def test_invalid_embeddings(tmp_path, vectors):
    import json

    from cfl.core.errors import LLMValidationError

    with (
        OllamaClient(
            Settings(state_dir=str(tmp_path)),
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=json.dumps({"embeddings": vectors}))
            ),
        ) as instance,
        pytest.raises(LLMValidationError),
    ):
        instance.embed(["test"])


def test_negative_overhead_cannot_bypass_budget():
    with pytest.raises(ValueError, match="overhead"):
        OllamaClient(Settings(template_overhead=-1))
