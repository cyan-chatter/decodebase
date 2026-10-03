from pathlib import Path
from types import SimpleNamespace

import pytest

from cfl.core.client import GenResult
from scripts.bakeoff import (
    dense_rank,
    embedding_text,
    generation_metrics,
    summary_result,
    symbols_from,
)


def test_fixture_corpus_has_stable_source_identity():
    rows = symbols_from(Path("eval/sample_repo"))
    assert len(rows) == 40
    assert len({r["id"] for r in rows}) == 40
    assert all(len(r["source_sha256"]) == 64 for r in rows)
    assert any(r["id"] == "utils/retry.py::with_retries.decorate.wrapped" for r in rows)


def test_invalid_summary_is_recorded_with_actual_cost():
    result = GenResult("{", 100, 2, 0.5, 100000000, 20000000, 0, "length", 20)
    client = SimpleNamespace(
        settings=SimpleNamespace(num_predict_symbol=350), generate=lambda *a, **kw: result
    )
    row = summary_result(
        client,
        {
            "id": "x.py::f",
            "source_sha256": "a",
            "file_path": "x.py",
            "qualname": "f",
            "raw_code": "def f(): pass",
        },
    )
    assert not row["json_valid"]
    assert row["output_tokens"] == 2
    assert row["done_reason"] == "length"
    assert row["wall_s"] >= 0


def test_projection_includes_invalid_samples_and_uses_native_eval_time():
    metrics = generation_metrics(
        [
            {
                "json_valid": True,
                "schema_valid": True,
                "wall_s": 1,
                "output_tokens": 20,
                "eval_duration": 500000000,
            },
            {
                "json_valid": False,
                "schema_valid": False,
                "wall_s": 3,
                "output_tokens": 20,
                "eval_duration": 500000000,
            },
        ],
        3600,
    )
    assert metrics["json_validity"] == 0.5
    assert metrics["gen_tps"] == 40
    assert metrics["projected_ingest_hours"] == 2


def test_dense_only_ranking_normalizes_vectors_and_is_stable():
    assert dense_rank([[100, 0], [0, 1], [1, 1]], [[0, 2]]) == [[1, 2, 0]]
    assert dense_rank([[1, 0], [1, 0]], [[1, 0]]) == [[0, 1]]


@pytest.mark.parametrize(
    "documents,queries", [([[0, 0]], [[1, 0]]), ([[1, 0]], [[1]]), ([[float("nan"), 1]], [[1, 0]])]
)
def test_dense_rank_rejects_invalid_embeddings(documents, queries):
    with pytest.raises(ValueError):
        dense_rank(documents, queries)


def test_model_specific_retrieval_instructions():
    assert embedding_text("code", "nomic-embed-text", query=False) == "search_document: code"
    assert (
        embedding_text("question", "nomic-embed-text:latest", query=True)
        == "search_query: question"
    )
    assert embedding_text("code", "qwen3-embedding:0.6b", query=False) == "code"
    assert embedding_text("question", "qwen3-embedding:0.6b", query=True).endswith("Query:question")


def test_archived_contract_replays_original_prompt_and_schema():
    import json

    contract = json.loads(Path("docs/validation/milestone-6/prompt-v1.json").read_text())
    calls = []

    def generate(prompt, system, **kwargs):
        calls.append((prompt, system, kwargs))
        return GenResult("{", 100, 2, 0.5, 100000000, 20000000, 0, "length", 20)

    client = SimpleNamespace(settings=SimpleNamespace(num_predict_symbol=350), generate=generate)
    summary_result(
        client,
        {
            "id": "x.py::f",
            "source_sha256": "a",
            "file_path": "x.py",
            "qualname": "f",
            "raw_code": "def f(): pass",
        },
        contract,
    )
    prompt, system, kwargs = calls[0]
    assert prompt.startswith(contract["summary_prefix"] + "\n[FILE] x.py")
    assert system == contract["system"]
    assert "minLength" not in kwargs["fmt"]["properties"]["inputs"]
    assert kwargs["num_predict"] == 350
