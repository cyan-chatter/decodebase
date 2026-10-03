from pathlib import Path

import pytest

from cfl.core import db
from cfl.engines import retrieval
from cfl.engines.retrieval import retrieve_evidence
from cfl.eval.runner import load_questions
from cfl.pipeline.indexer import build_lexical, retrieve_lexical

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.db


def test_traversal_questions_preserve_source_and_callsite_evidence(indexed_repo):
    build_lexical(indexed_repo)
    questions = load_questions(ROOT / "eval/questions.yaml")
    for question in questions:
        if question["type"] != "traversal":
            continue
        # Only natural-language text enters retrieval; the oracle stays in assertions.
        result = retrieve_evidence(indexed_repo, question["question"])
        assert result.complete, (question["id"], result.status, result.candidates)
        assert result.path_ids == question["expected"]["ordered_symbols"]
        assert [row["id"] for row in result.blocks[: len(result.path_ids)]] == result.path_ids
        for row in result.blocks[: len(result.path_ids)]:
            source = (ROOT / "eval/sample_repo" / row["file_path"]).read_text().splitlines()
            assert row["raw_code"] == "\n".join(source[row["start_line"] - 1 : row["end_line"]])
            assert row["retrieval_source"] == "graph_path"
        assert result.blocks[0]["incoming_call"] is None
        for caller, callee in zip(result.blocks, result.blocks[1 : len(result.path_ids)]):
            edge = callee["incoming_call"]
            assert edge["caller_id"] == caller["id"] and edge["callee_id"] == callee["id"]
            assert edge["callsite_file"] == caller["file_path"]
            assert caller["start_line"] <= edge["line"] <= caller["end_line"]


def test_long_path_is_complete_without_lexical_hits(indexed_repo):
    result = retrieve_evidence(
        indexed_repo, "Trace cli.main to db.repository.Repository.save.", 2, lexical_rows=[]
    )
    assert result.complete and result.requires_batching
    assert len(result.path_ids) == len(result.blocks) == 5
    assert result.lexical_blocks == []
    assert result.blocks[-1]["id"] == "db/repository.py::Repository.save"


@pytest.mark.parametrize(
    ("query", "status"),
    [
        ("Trace process to Repository.save.", "ambiguous_source"),
        ("Trace run_pipeline to process.", "ambiguous_target"),
        ("Trace absent_symbol to Repository.save.", "unresolved_source"),
        ("Trace run_pipeline to imaginary destination.", "unresolved_target"),
        ("Trace issue_token to Repository.save.", "no_path"),
    ],
)
def test_unresolved_paths_do_not_claim_complete_evidence(indexed_repo, query, status):
    result = retrieve_evidence(indexed_repo, query, lexical_rows=[])
    assert result.status == status
    assert not result.complete and result.path_ids == []
    if status.startswith("ambiguous"):
        assert len(result.candidates) > 1


def test_depth_and_confidence_limits_apply_to_evidence(indexed_repo):
    for options in [{"max_depth": 1}, {"min_conf": 0.99}]:
        result = retrieve_evidence(
            indexed_repo, "Trace run_pipeline to Repository.save.", lexical_rows=[], **options
        )
        assert result.status == "no_path" and not result.complete


def test_recursive_path_and_self_path_terminate(indexed_repo):
    result = retrieve_evidence(indexed_repo, "Trace is_even to is_odd.", lexical_rows=[])
    assert result.path_ids == ["trees.py::is_even", "trees.py::is_odd"]
    result = retrieve_evidence(indexed_repo, "Trace is_even to is_even.", lexical_rows=[])
    assert result.complete and result.path_ids == ["trees.py::is_even"]


def test_missing_source_is_reported(indexed_repo, monkeypatch):
    monkeypatch.setattr(db, "get_symbols", lambda *args: [])
    result = retrieve_evidence(indexed_repo, "Trace is_even to is_odd.", lexical_rows=[])
    assert result.status == "missing_source" and not result.complete


def test_equally_supported_prose_targets_are_reported(indexed_repo, monkeypatch):
    rows = db.get_symbols(
        indexed_repo, ["db/repository.py::Repository.save", "pipeline.py::write_to_db"]
    )
    monkeypatch.setattr(retrieval, "retrieve_lexical", lambda *args: rows)
    monkeypatch.setattr(retrieval.graph_queries, "path", lambda *args: [{"id": args[2]}])
    result = retrieve_evidence(indexed_repo, "Trace main to database persistence.", lexical_rows=[])
    assert result.status == "ambiguous_target" and not result.complete
    assert set(result.candidates) == {row["id"] for row in rows}


def test_ordinary_queries_keep_lexical_results(indexed_repo):
    build_lexical(indexed_repo)
    query = "Where is validate_token defined?"
    lexical = retrieve_lexical(indexed_repo, query)
    result = retrieve_evidence(indexed_repo, query, lexical_rows=lexical)
    assert result.blocks == result.lexical_blocks == lexical
    assert result.status == "lexical"


def test_path_priority_does_not_mutate_lexical_baseline(indexed_repo):
    distractor = db.get_symbol(indexed_repo, "auth/tokens.py::issue_token")
    lexical = [distractor]
    result = retrieve_evidence(
        indexed_repo, "Trace run_pipeline to Repository.save.", lexical_rows=lexical
    )
    assert result.complete and result.lexical_blocks == lexical
    assert result.blocks[0]["id"] == "pipeline.py::run_pipeline"
    assert result.blocks[-1]["retrieval_source"] == "lexical"
    assert "retrieval_source" not in distractor
