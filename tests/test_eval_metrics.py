from __future__ import annotations

import json

import pytest

from cfl.eval.metrics import (
    Span,
    answer_recall_at_5,
    citation_validity,
    structural_exactness,
    tokens_per_answer,
)


def test_recall_uses_first_five_distinct_and_inclusive_overlap():
    wanted = [("a.py", 10, 20), ("b.py", 1, 4)]
    returned = [("a.py", 20, 25)] * 10 + [
        ("c.py", 1, 3),
        ("d.py", 1, 3),
        ("e.py", 1, 3),
        ("f.py", 1, 3),
        ("b.py", 1, 4),
    ]
    assert answer_recall_at_5(wanted, returned) == 0.5
    assert answer_recall_at_5(wanted, [("other.py", 10, 20)]) == 0
    assert answer_recall_at_5(wanted, []) == 0
    assert answer_recall_at_5([], []) == 1


def test_citation_requires_containment_and_valid_syntax():
    citations = ["[a.py:10-12]", "a.py:8-12", "other.py:10-12", "invalid"]
    assert citation_validity(citations, [("a.py", 10, 20)]) == 0.25
    assert citation_validity([], []) == 0
    assert Span.parse({"file_path": "./a.py", "start_line": 10, "end_line": 12}) == Span(
        "a.py", 10, 12
    )
    assert Span.parse("[C:\\src\\a.py:10-12]").path == "C:/src/a.py"
    with pytest.raises(ValueError):
        Span("a.py", 20, 10)


def test_exactness_is_set_equality():
    assert structural_exactness(["a", "b"], ["b", "a", "a"]) == 1
    assert structural_exactness(["a"], ["a", "b"]) == 0
    assert structural_exactness([], []) == 1


def test_token_cost_from_successful_actual_trace_counts(tmp_path):
    records = [
        {
            "validation": "ok",
            "task": "answer",
            "symbol_id": "q1",
            "actual_prompt_tokens": 100,
            "output_tokens": 20,
        },
        {
            "validation": "retry",
            "task": "answer",
            "actual_prompt_tokens": None,
            "output_tokens": None,
        },
        {
            "validation": "ok",
            "task": "answer",
            "symbol_id": "q1",
            "actual_prompt_tokens": 40,
            "output_tokens": 10,
        },
    ]
    assert tokens_per_answer(records) == 85
    assert tokens_per_answer(records, answer_count=1) == 170
    assert tokens_per_answer(records, task="unrelated") is None
    assert tokens_per_answer(records, symbol_id="q2") is None
    file = tmp_path / "calls-20261004.jsonl"
    file.write_text("\n".join(json.dumps(record) for record in records))
    assert tokens_per_answer(tmp_path, task="answer") == 85
    with pytest.raises(ValueError):
        tokens_per_answer(records, answer_count=0)


@pytest.mark.parametrize(
    "citation",
    [
        "[a.py:10-12",
        "a.py:10-12]",
        {"path": 42, "start_line": 10, "end_line": 12},
        {"path": "a.py", "start_line": True, "end_line": 12},
    ],
)
def test_malformed_citations_are_invalid(citation):
    assert citation_validity([citation], [("a.py", 10, 20)]) == 0
