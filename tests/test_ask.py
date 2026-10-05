from pathlib import Path

import pytest

from cfl.core import db
from cfl.engines.ask import Candidate, answer, classify, render_blocks, retrieve
from cfl.engines.explain import explain
from cfl.pipeline.indexer import build_dense


def generations(fake):
    return sum(c["path"] in {"/api/chat", "/api/generate"} for c in fake.call_log)


@pytest.mark.db
def test_router_structural_has_no_generation_and_brief_is_reviewed(ready):
    conn, client, settings, fake = ready
    for question, route in [
        ("Who calls with_retries?", "structural"),
        ("trace run_pipeline", "flow"),
        ("Explain with_retries", "explain"),
        ("Where are retries handled?", "feature"),
        ("What else should I know?", "retrieval"),
    ]:
        assert classify(question, conn).kind == route
    before = generations(fake)
    result = answer(conn, client, settings, "Who calls with_retries?")
    assert result["sufficient"] and result["citations"]
    assert generations(fake) == before
    assert answer(conn, client, settings, "Who calls with_retries?")["cached"]
    assert explain(conn, client, settings, "with_retries")["sufficient"]
    assert generations(fake) > before
    before = generations(fake)
    narrative = answer(conn, client, settings, "Who calls with_retries?", explain_graph=True)
    assert narrative["sufficient"] and generations(fake) > before
    request = [c for c in fake.call_log if c["path"] == "/api/chat"][-1]["body"]
    assert "Verified graph results" in request["messages"][0]["content"]


@pytest.mark.db
def test_dense_exact_index_and_resume_are_cached(ready):
    conn, client, settings, fake = ready
    first = build_dense(conn, client, settings)
    assert not first["failed"] and first["count"] == 60 + len(db.published_knowledge(conn))
    embeds = sum(c["path"] == "/api/embed" for c in fake.call_log)
    version = db.index_version(conn)
    assert build_dense(conn, client, settings) == first
    assert sum(c["path"] == "/api/embed" for c in fake.call_log) == embeds
    assert db.index_version(conn) == version
    assert all(s["embed_hash"] for s in db.get_all_symbols(conn))
    assert all(
        f["skeleton"] is not None and f["line_count"]
        for f in db.list_files(conn)
        if f["parse_status"] in {"done", "parsed"}
    )
    rows = db.dense_units(conn, client.embed(["search_query: retries"])[0], 5)
    assert len(rows) == 5 and all("distance" in r for r in rows)
    assert retrieve(conn, client, "retries", settings)


@pytest.mark.db
def test_rrf_and_whole_block_budget(ready, monkeypatch):
    import cfl.engines.ask as engine

    conn, client, settings, _ = ready
    a, b, c = db.get_all_symbols(conn)[:3]
    monkeypatch.setattr(engine, "retrieve_lexical", lambda *args: [a, b])
    monkeypatch.setattr(
        db,
        "get_view_status",
        lambda *args: {"dense": {"status": "fresh", "config": {"digest": "digest-nomic"}}},
    )
    # Read the actual fake digest rather than assuming the tag's implementation.
    from cfl.core.memory import KnowledgeMemory

    digest = KnowledgeMemory(conn, client, settings).digests[settings.embed_model]
    monkeypatch.setattr(
        db,
        "get_view_status",
        lambda *args: {"dense": {"status": "fresh", "config": {"digest": digest}}},
    )
    monkeypatch.setattr(db, "dense_units", lambda *args: [b, c])
    candidates = retrieve(conn, client, "generic question", settings)
    assert candidates[0].row["id"] == b["id"]
    assert candidates[0].score == pytest.approx(1 / (settings.rrf_k + 2) + 1 / (settings.rrf_k + 1))
    blocks = render_blocks(candidates, 500, counter=client.counter)
    assert sum(client.counter.count(block["text"]) + 8 for block in blocks) <= 500
    assert (
        render_blocks([Candidate({**a, "raw_code": "x " * 10000}, 1)], 100, counter=client.counter)
        == []
    )


@pytest.mark.db
def test_answer_cache_verification_and_invalidation(ready):
    conn, client, settings, fake = ready
    question = "Where are retries handled?"
    first = answer(conn, client, settings, question)
    assert first["sufficient"] and not first["cached"]
    before = generations(fake)
    second = answer(conn, client, settings, question)
    assert second["cached"] and second["blocks"] and second["text"] == first["text"]
    assert generations(fake) == before
    db.bump_epoch(conn)
    assert not answer(conn, client, settings, question)["cached"]
    fake.answer_text = "Uncited invention from `imaginary` [missing.py:1-99]."
    bad = answer(conn, client, settings, "Where are impossible tasks handled?")
    assert not bad["sufficient"] and "insufficient context" in bad["text"]
    assert any("Unsupported citation" in w for w in bad["warnings"])
    assert any("Unknown identifier: imaginary" == w for w in bad["warnings"])
    assert not answer(conn, client, settings, "Where are impossible tasks handled?")["cached"]


@pytest.mark.db
def test_full_evaluation_reports_answer_metrics(ready):
    from cfl.eval.runner import run_eval

    conn, client, settings, _ = ready
    questions = Path(__file__).resolve().parents[1] / "eval/questions.yaml"
    result = run_eval(
        conn,
        {"questions": str(questions)},
        ("full", "structural"),
        client=client,
        settings=settings,
    )
    assert len(result["rows"]) == 48
    for row in result["rows"]:
        metrics = row["metrics"]
        assert 0 <= metrics["retrieved_recall_at_5"] <= 1
        assert 0 <= metrics["cited_recall_at_5"] <= 1
        assert 0 <= metrics["citation_validity"] <= 1
        assert metrics["tokens_per_answer"] >= 0
    assert any(row["metrics"]["tokens_per_answer"] > 0 for row in result["rows"])


@pytest.mark.db
def test_dense_failure_preserves_lexical_answers(ready, monkeypatch):
    conn, client, settings, _ = ready

    def unavailable(*args):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(client, "embed", unavailable)
    assert build_dense(conn, client, settings)["failed"]
    assert db.get_view_status(conn)["dense"]["status"] == "failed"
    assert db.get_view_status(conn)["lexical"]["status"] == "fresh"
    assert answer(conn, client, settings, "Where are retries handled?")["sufficient"]


@pytest.mark.db
def test_registered_vector_connection_can_reuse_embeddings(ready):
    conn, client, settings, _ = ready
    assert not build_dense(conn, client, settings)["failed"]
    # Fresh connections register pgvector's Vector decoder, unlike a newly migrated DB.
    from tests.conftest import CFL_DSN

    with db.connect(CFL_DSN.rsplit("/", 1)[0] + "/" + conn.info.dbname) as reopened:
        result = build_dense(reopened, client, settings)
        assert not result["failed"] and result["count"] == 60 + len(db.published_knowledge(conn))


@pytest.mark.db
def test_natural_graph_queries_and_trace_entrypoint_are_resolved(ready):
    conn, _, _, _ = ready
    from cfl.engines.graph_queries import resolve_symbol

    assert classify("What calls validate_token?", conn).kind == "structural"
    assert classify("Which functions call issue_token?", conn).kind == "structural"
    route = classify("Trace run_pipeline to Repository.save.", conn)
    assert route.kind == "flow" and route.target == "run_pipeline"
    route = classify("Trace services.tasks.process to task persistence.", conn)
    assert route.target == "services.tasks.process"
    assert resolve_symbol(conn, route.target).id == "services/tasks.py::process"
    assert classify("Trace is_odd to is_even.", conn).target == "is_odd"


def test_embedding_cap_preserves_source_head():
    from cfl.core.budget import TokenCounter
    from cfl.pipeline.indexer import embedding_head

    counter = TokenCounter()
    text = "search_document: path signature summary\nHEAD\n" + "statement = value\n" * 1000 + "TAIL"
    result = embedding_head(text, 1024, counter)
    assert counter.count(result) <= 1024
    assert result.startswith("search_document: path signature summary\nHEAD")
    assert "TAIL" not in result and "omitted from embedding" in result


@pytest.mark.db
def test_module_scope_excludes_same_named_members_from_other_files(ready):
    conn, client, settings, _ = ready
    candidates = retrieve(conn, client, "What does services.tasks provide?", settings)
    assert candidates and all(c.row["file_path"] == "services/tasks.py" for c in candidates)


@pytest.mark.db
def test_uncited_answer_gets_only_one_bounded_repair(ready):
    conn, client, settings, fake = ready
    fake.answer_text = "An answer without a citation."
    before = generations(fake)
    result = answer(conn, client, settings, "Where are uncited operations handled?")
    assert not result["sufficient"]
    assert generations(fake) == before + 2


@pytest.mark.db
def test_unknown_identifier_answer_is_rejected_and_not_cached(ready):
    conn, client, settings, fake = ready
    fake.answer_text = "Uses `ghost` [utils/retry.py:6-23]."
    first = answer(conn, client, settings, "Where are retries handled?")
    assert not first["sufficient"] and first["status"] == "abstained"
    assert "ghost" not in first["text"]
    second = answer(conn, client, settings, "Where are retries handled?")
    assert not second["cached"]
