import pytest

from cfl.config import Settings
from cfl.core import db
from cfl.core.client import OllamaClient
from cfl.pipeline.scan import run_stage1, run_stage2
from cfl.pipeline.symbol_pass import is_trivial, run_symbol_pass
from tests.fakes import FakeOllama


def scan(conn, repo, settings):
    run_stage1(conn, settings, str(repo))
    run_stage2(conn, settings, str(repo))


def calls(fake):
    return [c for c in fake.call_log if c["path"] == "/api/generate"]


@pytest.mark.db
def test_bottom_up_resume_edit_and_prompt_invalidation(pg_conn, tmp_path, monkeypatch):
    from cfl.prompts import prompts

    (tmp_path / "a.py").write_text(
        "def leaf(x):\n    y = x + 1\n    return y\n\ndef caller(x):\n    y = leaf(x)\n    return y * 2\n"
    )
    settings = Settings(state_dir=str(tmp_path / ".cfl"))
    fake = FakeOllama()
    fake.stable_summary = True
    with OllamaClient(settings, transport=fake.transport) as client:
        scan(pg_conn, tmp_path, settings)
        result = run_symbol_pass(pg_conn, client, settings)
        assert result.saved == 2 and result.llm_calls == 2
        assert "def leaf(" in calls(fake)[0]["body"]["prompt"]
        assert "def caller(" in calls(fake)[1]["body"]["prompt"]
        assert run_symbol_pass(pg_conn, client, settings).cached == 2
        assert len(calls(fake)) == 2
        (tmp_path / "a.py").write_text((tmp_path / "a.py").read_text().replace("x + 1", "x + 3"))
        scan(pg_conn, tmp_path, settings)
        assert run_symbol_pass(pg_conn, client, settings).llm_calls == 1
        assert len(calls(fake)) == 3
        monkeypatch.setattr(prompts, "PROMPT_VERSION", "new-version")
        assert run_symbol_pass(pg_conn, client, settings).llm_calls == 2


@pytest.mark.db
def test_interrupt_resumes_without_repeating_completed_requests(pg_conn, tmp_path):
    (tmp_path / "a.py").write_text(
        "\n".join(f"def f{i}(x):\n    y=x+{i}\n    return y\n" for i in range(4))
    )
    settings = Settings(state_dir=str(tmp_path / ".cfl"))
    fake = FakeOllama()
    fake.kill_after = 2
    with OllamaClient(settings, transport=fake.transport) as client:
        scan(pg_conn, tmp_path, settings)
        result = run_symbol_pass(pg_conn, client, settings)
        assert result.interrupted and result.saved == 2
        fake.kill_after = None
        result = run_symbol_pass(pg_conn, client, settings)
        assert result.saved == 2 and result.cached == 2
        assert fake.generated == 4
        assert db.status_counts(pg_conn) == {"done": 4}


@pytest.mark.db
def test_mutual_recursion_combined_once_and_cached(pg_conn, tmp_path):
    (tmp_path / "a.py").write_text(
        "def even(n):\n    return True if n == 0 else odd(n-1)\n\ndef odd(n):\n    return False if n == 0 else even(n-1)\n"
    )
    settings = Settings(state_dir=str(tmp_path / ".cfl"))
    fake = FakeOllama()
    with OllamaClient(settings, transport=fake.transport) as client:
        scan(pg_conn, tmp_path, settings)
        result = run_symbol_pass(pg_conn, client, settings)
        assert result.saved == 2 and result.llm_calls == 1
        assert len(calls(fake)) == 1
        assert run_symbol_pass(pg_conn, client, settings).cached == 2


@pytest.mark.db
def test_quarantine_after_one_repair_then_continue(pg_conn, tmp_path):
    (tmp_path / "a.py").write_text(
        "def a(x):\n    y=x+1\n    return y\n\ndef b(x):\n    y=x+2\n    return y\n"
    )
    settings = Settings(state_dir=str(tmp_path / ".cfl"))
    fake = FakeOllama()
    fake.invalid_outputs = 2
    with OllamaClient(settings, transport=fake.transport) as client:
        scan(pg_conn, tmp_path, settings)
        result = run_symbol_pass(pg_conn, client, settings)
        assert result.failed == 1 and result.saved == 1 and result.llm_calls == 3
        assert db.get_symbol(pg_conn, "a.py::a")["attempts"] == 2
        assert run_symbol_pass(pg_conn, client, settings, retry_failed=True).saved == 1


def test_trivial_rules_do_not_skip_real_computation():
    def symbol(code, name="f"):
        return {
            "raw_code": code,
            "kind": "function",
            "name": name,
            "start_line": 1,
            "end_line": len(code.splitlines()),
        }

    assert is_trivial(symbol("def f(self):\n    return self.value"))
    assert is_trivial(symbol("def f(x):\n    return delegate(x)"))
    assert not is_trivial(symbol("def f(x):\n    return delegate(x+1)"))
    assert not is_trivial(symbol("def f(x):\n    return delegate(3)"))
    assert is_trivial(symbol("def f():\n    raise NotImplementedError"))


@pytest.mark.db
def test_changed_summary_ripples_and_trivial_never_generates(pg_conn, tmp_path):
    path = tmp_path / "a.py"
    path.write_text(
        "def leaf(x):\n    y=x+1\n    return y\n\ndef caller(x):\n    y=leaf(x)\n    return y*2\n\ndef forward(x):\n    return caller(x)\n"
    )
    settings = Settings(state_dir=str(tmp_path / ".cfl"))
    fake = FakeOllama()
    with OllamaClient(settings, transport=fake.transport) as client:
        scan(pg_conn, tmp_path, settings)
        first = run_symbol_pass(pg_conn, client, settings)
        assert first.saved == 3 and first.trivial == 1 and first.llm_calls == 2
        path.write_text(path.read_text().replace("x+1", "x+5"))
        scan(pg_conn, tmp_path, settings)
        second = run_symbol_pass(pg_conn, client, settings)
        assert second.llm_calls == 2  # Changed leaf description invalidates its caller.
        assert run_symbol_pass(pg_conn, client, settings).llm_calls == 0


@pytest.mark.db
def test_physical_transport_attempts_are_counted_and_quarantined(pg_conn, tmp_path, monkeypatch):
    import cfl.core.client as client_module

    monkeypatch.setattr(client_module.time, "sleep", lambda _: None)
    (tmp_path / "a.py").write_text("def f(x):\n    y=x+1\n    return y\n")
    settings = Settings(state_dir=str(tmp_path / ".cfl"))
    fake = FakeOllama()
    # Fail only generation, so digest resolution is unaffected.
    import httpx

    def responder(request):
        if request.url.path == "/api/generate" and len(calls(fake)) < 3:
            fake.call_log.append({"path": request.url.path, "body": {}})
            return httpx.Response(503, json={"error": "temporary"})
        return fake._responder(request)

    with OllamaClient(settings, transport=httpx.MockTransport(responder)) as client:
        scan(pg_conn, tmp_path, settings)
        result = run_symbol_pass(pg_conn, client, settings)
        assert result.failed == 1 and result.llm_calls == 3
        assert db.get_symbol(pg_conn, "a.py::f")["attempts"] == 3


@pytest.mark.db
def test_class_runs_after_methods_and_corrupt_cached_summary_regenerates(pg_conn, tmp_path):
    (tmp_path / "a.py").write_text(
        'class Worker:\n    """A computation worker."""\n    def calculate(self, x):\n        y=x+1\n        return y\n'
    )
    settings = Settings(state_dir=str(tmp_path / ".cfl"))
    fake = FakeOllama()
    with OllamaClient(settings, transport=fake.transport) as client:
        scan(pg_conn, tmp_path, settings)
        result = run_symbol_pass(pg_conn, client, settings)
        assert result.llm_calls == 2
        assert "def calculate(" in calls(fake)[0]["body"]["prompt"]
        assert "Worker.calculate(" in calls(fake)[1]["body"]["prompt"]
        row = db.get_symbol(pg_conn, "a.py::Worker.calculate")
        db.save_symbol_summary(
            pg_conn, row["id"], {"invalid": True}, row["summary_short"], None, row["ctx_hash"]
        )
        assert run_symbol_pass(pg_conn, client, settings).llm_calls >= 1


@pytest.mark.db
def test_oversized_symbol_partials_are_combined_without_source_loss(pg_conn, tmp_path):
    (tmp_path / "a.py").write_text(
        "def large(x):\n"
        + "".join(f"    value_{i}=x+{i}\n" for i in range(500))
        + "    return value_499\n"
    )
    settings = Settings(num_ctx=2048, state_dir=str(tmp_path / ".cfl"))
    fake = FakeOllama()
    with OllamaClient(settings, transport=fake.transport) as client:
        scan(pg_conn, tmp_path, settings)
        result = run_symbol_pass(pg_conn, client, settings)
        assert result.failed == 0 and result.llm_calls > 2
        prompts = [c["body"]["prompt"] for c in calls(fake)]
        assert any("value_0=" in p for p in prompts) and any("value_499=" in p for p in prompts)
        assert "Combine all partials" in prompts[-1]
