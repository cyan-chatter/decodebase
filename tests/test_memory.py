from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from psycopg.conninfo import make_conninfo

from cfl.core import db
from cfl.core.errors import LLMValidationError
from cfl.core.memory import KnowledgeMemory, SessionMemory, answer_key, evidence_reference
from cfl.pipeline.scan import run_stage1
from cfl.prompts.prompts import SUMMARY_PREFIX, build_symbol_prompt
from tests.conftest import CFL_DSN


def test_stable_prompt_prefix_and_dependency_order():
    symbol = {"file_path": "a.py", "qualname": "f", "raw_code": "def f(): pass"}
    system, prompt = build_symbol_prompt(symbol, [("b", "B"), ("a", "A")])
    assert prompt.startswith(SUMMARY_PREFIX) and "precise" in system
    assert prompt.index("a: A") < prompt.index("b: B") < prompt.index("[CODE]")
    assert build_symbol_prompt(symbol, [("a", "A"), ("b", "B")]) == (system, prompt)


@pytest.mark.db
def test_summary_cache_and_embedding_cache_skip_generation(indexed_repo, fake_ollama, settings):
    # indexed_repo prohibits constructing Ollama; provide the existing fake explicitly.
    class Client:
        tags = lambda self: [
            {"name": "qwen3.5:9b", "digest": "gen-a"},
            {"name": "nomic-embed-text:latest", "digest": "embed-a"},
        ]

        def generate(self, *args, **kwargs):
            self.calls += 1
            from types import SimpleNamespace

            return SimpleNamespace(text=fake_ollama._make_summary(args[0], True))

        def embed(self, texts):
            self.embedding_calls += 1
            return [fake_ollama._make_embedding(text) for text in texts]

        calls = 0
        embedding_calls = 0

    client = Client()
    memory = KnowledgeMemory(indexed_repo, client, settings)
    id = "auth/tokens.py::issue_token"
    first = memory.summarize_symbol(id, [])
    assert memory.summarize_symbol(id, []) == first and client.calls == 1
    assert db.get_symbol(indexed_repo, id)["summary_long"] is None
    memory.summarize_symbol(id, [("a::save", "Save")])
    memory.summarize_symbol(id, [("b::save", "Save")])
    assert client.calls == 3
    vectors = memory.embed(["hello", "world", "hello"])
    assert vectors[0] == vectors[2] and client.embedding_calls == 1
    # New instances reuse durable data, not just an in-process dictionary.
    restored = KnowledgeMemory(indexed_repo, client, settings)
    assert restored.embed(["hello", "world", "hello"]) == vectors
    assert client.embedding_calls == 1
    restored.summarize_symbol(id, [("b::save", "Save")])
    assert client.calls == 3
    client.tags = lambda: [
        {"name": "qwen3.5:9b", "digest": "gen-b"},
        {"name": "nomic-embed-text:latest", "digest": "embed-b"},
    ]
    changed = KnowledgeMemory(indexed_repo, client, settings)
    changed.summarize_symbol(id, [("b::save", "Save")])
    changed.embed(["hello"])
    assert client.calls == 4 and client.embedding_calls == 2


@pytest.mark.db
def test_verified_answers_require_current_source(indexed_repo):
    conn = indexed_repo
    symbol = db.get_symbol(conn, "trees.py::is_even")
    evidence = [evidence_reference(symbol)]
    version = db.index_version(conn)
    key = answer_key("Explain even", "brief", {"model": "a"})
    assert key != answer_key("Explain even", "detailed", {"model": "a"})
    assert key != answer_key("Explain even", "brief", {"model": "b"})
    assert key != answer_key("Explain even", "brief", {"model": "a"}, {"turn": 1})
    db.put_cached_answer(conn, key, "brief", "Answer", version, evidence=evidence)
    assert db.get_cached_answer(conn, key, version) is None
    db.put_cached_answer(conn, key, "brief", "Answer", version, verified=True, evidence=evidence)
    assert db.get_cached_answer(conn, key, version) == "Answer"
    altered = [{**evidence[0], "start_line": evidence[0]["start_line"] + 1}]
    db.put_cached_answer(conn, key, "brief", "Answer", version, verified=True, evidence=altered)
    assert db.get_cached_answer(conn, key, version) is None
    db.bump_epoch(conn)
    assert db.index_version(conn) != version


@pytest.mark.db
def test_session_resumes_with_bounds_and_revalidates_sources(indexed_repo):
    conn = indexed_repo
    symbol = db.get_symbol(conn, "trees.py::is_even")
    session = SessionMemory("test", constraints=["Use concise explanations"])
    for index in range(7):
        session.add_turn(str(index), "Answer", [symbol])
    session.save(conn)
    restored = SessionMemory.load(conn, "test")
    assert len(restored.turns) == 4 and restored.turns[0]["question"] == "3"
    assert restored.context(conn)["sources"][0]["raw_code"] == symbol["raw_code"]
    db.bump_epoch(conn)
    changed = SessionMemory.load(conn, "test")
    assert changed.turns == [] and changed.constraints == session.constraints
    assert len(changed.direction_seed) == 1
    assert restored.context(conn)["recent_turns"] == []
    changed.direction_seed[0]["code_hash"] = "stale"
    changed.save(conn)
    assert SessionMemory.load(conn, "test").direction_seed == []
    db.set_meta(conn, "repo_root", "/different/repo")
    with pytest.raises(ValueError, match="different repository"):
        SessionMemory.load(conn, "test")


@pytest.mark.db
def test_memory_prompt_prioritizes_current_sources_over_old_answers(indexed_repo, settings):
    session = SessionMemory("bounded", constraints=["Answer with source citations"])
    symbol = db.get_symbol(indexed_repo, "trees.py::is_even")
    session.add_turn("Old question", "x" * 20000, [symbol])
    settings = settings.model_copy(update={"num_ctx": 1024})
    assembled = session.assemble_prompt(indexed_repo, "Explain is_even", settings, num_predict=120)
    assert assembled.prompt_tokens + 120 <= 1024
    assert symbol["raw_code"] in assembled.prompt
    assert "Answer with source citations" in assembled.prompt
    assert "history" in assembled.dropped or "history" in assembled.truncated
    assert len(session.turns[0]["answer"]) == 20000


def test_session_limits_reject_unbounded_or_zero_values():
    for limits in [{"max_turns": 0}, {"max_seed": 0}, {"max_turns": 1000}]:
        with pytest.raises(ValueError, match="bounded"):
            SessionMemory("test", **limits)


@pytest.mark.db
def test_epoch_increments_are_atomic_across_connections(pg_conn):
    dsn = make_conninfo(CFL_DSN, dbname=pg_conn.info.dbname)

    def increment():
        with db.connect(dsn) as conn:
            for _ in range(10):
                db.bump_epoch(conn)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: increment(), range(4)))
    assert db.get_meta(pg_conn, "epoch") == "40"


@pytest.mark.db
def test_scanning_content_edits_invalidates_answers_without_noop_churn(pg_conn, settings, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "a.py"
    path.write_text("def f(flag):\n    if flag:\n        work()\n    finish()\n")
    run_stage1(pg_conn, settings, str(repo))
    first = db.index_version(pg_conn)
    old_hash = db.get_symbol(pg_conn, "a.py::f")["code_hash"]
    run_stage1(pg_conn, settings, str(repo))
    assert db.index_version(pg_conn) == first
    path.write_text("def f(flag):\n    if flag:\n        work()\n        finish()\n")
    run_stage1(pg_conn, settings, str(repo))
    assert db.index_version(pg_conn) != first
    assert db.get_symbol(pg_conn, "a.py::f")["code_hash"] != old_hash


@pytest.mark.db
@pytest.mark.parametrize("failure", ["invalid_json", "source_edit"])
def test_unusable_generated_memory_is_not_saved(
    indexed_repo, settings, fake_ollama, monkeypatch, failure
):
    conn = indexed_repo
    id = "trees.py::is_even"
    original = db.get_symbol
    symbol = original(conn, id)
    client = SimpleNamespace(
        tags=lambda: [
            {"name": "qwen3.5:9b", "digest": "gen"},
            {"name": "nomic-embed-text:latest", "digest": "embed"},
        ],
        generate=lambda *args, **kwargs: SimpleNamespace(
            text="invalid" if failure == "invalid_json" else fake_ollama._make_summary("code", True)
        ),
    )
    memory = KnowledgeMemory(conn, client, settings)
    if failure == "source_edit":
        calls = 0

        def changed(*args):
            nonlocal calls
            calls += 1
            return symbol if calls == 1 else {**symbol, "code_hash": "changed"}

        monkeypatch.setattr(db, "get_symbol", changed)
    with pytest.raises(LLMValidationError):
        memory.summarize_symbol(id, [])
    assert original(conn, id)["summary_json"] is None


def test_summary_json_schema_matches_nonempty_string_contract():
    from cfl.prompts.schemas import SYMBOL_SUMMARY_JSON_SCHEMA

    for name in ("one_liner", "purpose", "inputs", "returns", "notable_logic"):
        assert SYMBOL_SUMMARY_JSON_SCHEMA["properties"][name]["minLength"] == 1
