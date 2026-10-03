from __future__ import annotations

import hashlib
import tomllib
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from cfl.core import db
from cfl.engines import graph_queries
from cfl.engines.retrieval import RETRIEVAL_VERSION, retrieve_evidence
from cfl.eval.metrics import Span, answer_recall_at_5, structural_exactness
from cfl.pipeline.indexer import build_lexical, retrieve_lexical

if TYPE_CHECKING:
    from psycopg import Connection

TYPES = {"symbol", "module", "behavioral", "traversal", "structural", "reasoning"}


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as file:
        config = tomllib.load(file).get("eval", {})
    questions = Path(config.get("questions", "../questions.yaml"))
    config["questions"] = str(questions if questions.is_absolute() else path.parent / questions)
    return config


def load_questions(path: str | Path, *, verified_only: bool = False) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or not isinstance(data.get("questions"), list)
    ):
        raise ValueError("Expected a version 1 question suite")
    ids = set()
    questions = []
    for question in data["questions"]:
        if not isinstance(question, dict):
            raise TypeError("Invalid question schema")
        id = question.get("id")
        if not isinstance(id, str) or not id or id in ids:
            raise ValueError(f"Missing or duplicate question ID: {id}")
        ids.add(id)
        expected = question.get("expected")
        if (
            question.get("type") not in TYPES
            or not isinstance(question.get("question"), str)
            or not question["question"].strip()
            or not isinstance(expected, dict)
            or not isinstance(question.get("verified"), bool)
        ):
            raise ValueError(f"Invalid question: {id}")
        for key in ("symbols", "ordered_symbols", "callers", "keywords"):
            if key in expected and (
                not isinstance(expected[key], list)
                or any(not isinstance(item, str) or not item for item in expected[key])
            ):
                raise ValueError(f"Invalid expected {key}: {id}")
        if not expected.get("symbols"):
            raise ValueError(f"Expected evidence symbols required: {id}")
        if question["type"] == "structural" and (
            "callers" not in expected or not question.get("target")
        ):
            raise ValueError(f"Structural target and caller set required: {id}")
        if question["type"] == "traversal" and (
            not expected.get("ordered_symbols")
            or not question.get("source")
            or not question.get("target")
        ):
            raise ValueError(f"Traversal endpoints and ordered symbols required: {id}")
        for key in ("source", "target"):
            if key in question and (not isinstance(question[key], str) or not question[key]):
                raise ValueError(f"Invalid {key}: {id}")
        if question["type"] == "reasoning" and not expected.get("keywords"):
            raise ValueError(f"Reasoning keywords required: {id}")
        if question["type"] == "traversal" and (
            expected["ordered_symbols"][0] != question["source"]
            or expected["ordered_symbols"][-1] != question["target"]
        ):
            raise ValueError(f"Traversal expectation must match endpoints: {id}")
        review = question.get("review")
        if review is not None and (
            not isinstance(review, dict)
            or review.get("kind") not in {"agent", "human"}
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"].strip()
            or not isinstance(review.get("notes"), str)
            or not review["notes"].strip()
        ):
            raise ValueError(f"Invalid review provenance: {id}")
        if not verified_only or question["verified"]:
            questions.append(question)
    if not questions:
        raise ValueError("No questions selected; expectation verification may still be pending")
    return questions


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["metrics"]["type"]].append(row["metrics"])
    result = {}
    for type, metrics in sorted(groups.items()):
        keys = sorted(
            {
                key
                for item in metrics
                for key, value in item.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
        )
        result[type] = {
            "count": len(metrics),
            **{
                key: sum(item[key] for item in metrics if key in item)
                / sum(key in item for item in metrics)
                for key in keys
            },
        }
    return result


def run_eval(conn: Connection, config: dict, modes=None) -> dict:
    modes = set(("retrieval", "structural") if modes is None else modes)
    if not modes or modes - {"retrieval", "structural"}:
        raise ValueError(
            "Only retrieval and structural modes exist at milestone 5; full answers are unavailable"
        )
    limit, depth, confidence = (
        config.get("retrieval_limit", 5),
        config.get("max_depth", 8),
        config.get("min_conf", 0.6),
    )
    if (
        not isinstance(limit, int)
        or not isinstance(depth, int)
        or limit < 5
        or depth < 1
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Require retrieval_limit >= 5, max_depth >= 1 and min_conf in [0,1]")
    questions = load_questions(
        config["questions"], verified_only=config.get("verified_only", False)
    )
    if modes == {"structural"}:
        questions = [
            question for question in questions if question["type"] in {"structural", "traversal"}
        ]
    if not questions:
        raise ValueError("No questions selected for these modes")
    # Resolve all draft expectations before scoring: never treat absent evidence as success.
    symbols = {}
    for question in questions:
        expected = question["expected"]
        references = [
            *expected["symbols"],
            *expected.get("callers", []),
            *expected.get("ordered_symbols", []),
        ]
        references += [question[key] for key in ("source", "target") if key in question]
        for reference in references:
            if reference not in symbols:
                row = db.get_symbol(conn, reference)
                if row is None:
                    raise ValueError(
                        f"Expected symbol missing from index: {reference}; scan the evaluation fixture"
                    )
                symbols[reference] = row
    review_methods = sorted(
        {
            question.get("review", {}).get("kind", "unspecified")
            for question in questions
            if question["verified"]
        }
    )
    run_id = uuid.uuid4().hex
    snapshot = {
        **config,
        "questions": str(Path(config["questions"]).resolve()),
        "retrieval_limit": limit,
        "max_depth": depth,
        "min_conf": confidence,
        "parser_version": db.get_meta(conn, "parser_version"),
        "resolve_fingerprint": db.get_meta(conn, "resolve_fingerprint"),
        "modes": sorted(modes),
        "suite_hash": hashlib.sha256(Path(config["questions"]).read_bytes()).hexdigest(),
        "questions_snapshot": questions,
        "review_methods": review_methods,
        "retrieval_version": RETRIEVAL_VERSION,
    }
    rows = []
    with conn.transaction():
        if "retrieval" in modes:
            build_lexical(conn)
        for question in questions:
            expected = question["expected"]
            metrics = {
                "type": question["type"],
                "verified": question["verified"],
                "review_method": question.get("review", {}).get("kind", "unspecified")
                if question["verified"]
                else None,
            }
            if "retrieval" in modes:
                retrieved = retrieve_lexical(conn, question["question"], limit)
                metrics["answer_recall_at_5"] = answer_recall_at_5(
                    [symbols[id] for id in expected["symbols"]], retrieved
                )
                metrics["expected_spans"] = [
                    vars(Span.parse(symbols[id])) for id in expected["symbols"]
                ]
                metrics["returned_spans"] = [vars(Span.parse(row)) for row in retrieved]
                metrics["returned_symbols"] = [row["id"] for row in retrieved]
                evidence = retrieve_evidence(
                    conn,
                    question["question"],
                    limit,
                    max_depth=depth,
                    min_conf=confidence,
                    lexical_rows=retrieved,
                )
                metrics["evidence_recall_at_5"] = answer_recall_at_5(
                    [symbols[id] for id in expected["symbols"]],
                    evidence.blocks,
                )
                metrics["evidence_symbols"] = [row["id"] for row in evidence.blocks]
                metrics["evidence_spans"] = [vars(Span.parse(row)) for row in evidence.blocks]
                metrics["path_evidence_status"] = evidence.status
                metrics["path_evidence_complete"] = evidence.complete
                metrics["evidence_requires_batching"] = evidence.requires_batching
                metrics["path_evidence_symbols"] = evidence.path_ids

            if "structural" in modes and question["type"] == "structural":
                callers = graph_queries.callers(conn, question["target"], 1, confidence)
                got = sorted({row["id"] for row in callers if row["id"] is not None})
                metrics["structural_exactness"] = structural_exactness(expected["callers"], got)
                metrics["actual_callers"] = got
            if "structural" in modes and question["type"] == "traversal":
                hops = graph_queries.path(
                    conn, question["source"], question["target"], depth, confidence
                )
                got = [question["source"], *(row["id"] for row in hops)] if hops is not None else []
                metrics["traversal_exactness"] = float(got == expected["ordered_symbols"])
                metrics["actual_ordered_symbols"] = got
            db.save_eval_metrics(conn, run_id, question["id"], metrics, snapshot)
            rows.append({"question_id": question["id"], "metrics": metrics})
    return {
        "run_id": run_id,
        "provisional": any(not question["verified"] for question in questions),
        "review_methods": review_methods,
        "summary": summarize(rows),
        "rows": rows,
    }


def compare_runs(conn: Connection, run_a: str, run_b: str) -> dict:
    first, second = db.get_eval_run(conn, run_a), db.get_eval_run(conn, run_b)
    if not first or not second:
        raise ValueError("Both run IDs must exist")
    # Compare common question IDs, recording baseline differences instead of hiding them.
    left, right = (
        {row["question_id"]: row for row in first},
        {row["question_id"]: row for row in second},
    )
    common = sorted(left.keys() & right.keys())
    differences = []
    for id in common:
        a, b = left[id]["metrics"], right[id]["metrics"]
        for metric in (
            "answer_recall_at_5",
            "evidence_recall_at_5",
            "structural_exactness",
            "traversal_exactness",
        ):
            if metric in a and metric in b:
                differences.append(
                    {
                        "question_id": id,
                        "metric": metric,
                        "run_a": a[metric],
                        "run_b": b[metric],
                        "delta": b[metric] - a[metric],
                    }
                )
    return {
        "run_a": run_a,
        "run_b": run_b,
        "summary_a": summarize(first),
        "summary_b": summarize(second),
        "only_a": sorted(left.keys() - right.keys()),
        "only_b": sorted(right.keys() - left.keys()),
        "suite_changed": first[0]["config"].get("suite_hash")
        != second[0]["config"].get("suite_hash"),
        "differences": differences,
    }
