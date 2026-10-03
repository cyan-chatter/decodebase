from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cfl.cli import app
from cfl.core import db
from cfl.eval.runner import compare_runs, load_config, load_questions, run_eval

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "eval/configs/lexical.toml"


def test_agent_reviewed_suite_and_relative_config():
    config = load_config(CONFIG)
    questions = load_questions(config["questions"])
    assert len(questions) == 48
    assert Counter(question["type"] for question in questions) == dict.fromkeys(
        ["symbol", "module", "behavioral", "traversal", "structural", "reasoning"], 8
    )
    assert all(
        question["verified"] and question["review"]["kind"] == "agent" for question in questions
    )
    assert all(
        question["review"]["notes"] and len(question["review"]["fixture_sha256"]) == 64
        for question in questions
    )
    assert len(load_questions(config["questions"], verified_only=True)) == 48


@pytest.mark.parametrize("mutation", ["duplicate", "missing_expected", "bad_type", "bad_verified"])
def test_invalid_suite_rejected(tmp_path, mutation):
    questions = load_questions(ROOT / "eval/questions.yaml")
    if mutation == "duplicate":
        questions.append(questions[0])
    elif mutation == "missing_expected":
        questions[0]["expected"]["symbols"] = []
    elif mutation == "bad_type":
        questions[0]["type"] = "unknown"
    else:
        questions[0]["verified"] = "yes"
    file = tmp_path / "questions.yaml"
    file.write_text(yaml.safe_dump({"version": 1, "questions": questions}))
    with pytest.raises(ValueError):
        load_questions(file)


@pytest.mark.db
def test_runner_metrics_persist_and_compare(indexed_repo):
    conn = indexed_repo
    config = load_config(CONFIG)
    first = run_eval(conn, config, ("retrieval", "structural"))
    assert first["provisional"] is False and len(first["rows"]) == 48
    assert first["review_methods"] == ["agent"]
    assert first["summary"]["structural"]["structural_exactness"] == 1
    assert first["summary"]["traversal"]["traversal_exactness"] == 1
    assert first["summary"]["traversal"]["evidence_recall_at_5"] == 1
    assert first["summary"]["traversal"]["answer_recall_at_5"] < 1
    assert all(
        row["metrics"]["path_evidence_complete"]
        for row in first["rows"]
        if row["metrics"]["type"] == "traversal"
    )
    assert first["summary"]["symbol"]["answer_recall_at_5"] > 0.5
    saved = db.get_eval_run(conn, first["run_id"])
    assert len(saved) == 48 and len(saved[0]["config"]["questions_snapshot"]) == 48
    assert saved[0]["config"]["modes"] == ["retrieval", "structural"]
    assert saved[0]["config"]["review_methods"] == ["agent"]
    second = run_eval(conn, config, ("structural",))
    comparison = compare_runs(conn, first["run_id"], second["run_id"])
    assert len(comparison["only_a"]) == 32 and comparison["only_b"] == []
    assert all(row["delta"] == 0 for row in comparison["differences"])
    with pytest.raises(ValueError, match="exist"):
        compare_runs(conn, "unknown", first["run_id"])


@pytest.mark.db
def test_runner_missing_symbols_and_full_answers_fail(indexed_repo, tmp_path):
    conn = indexed_repo
    config = load_config(CONFIG)
    with pytest.raises(ValueError, match="full answers"):
        run_eval(conn, config, ("full",))
    questions = load_questions(config["questions"])
    questions[0]["expected"]["symbols"] = ["missing.py::missing"]
    file = tmp_path / "missing.yaml"
    file.write_text(yaml.safe_dump({"version": 1, "questions": questions}))
    with pytest.raises(ValueError, match="missing from index"):
        run_eval(conn, {**config, "questions": str(file)})


@pytest.mark.db
def test_eval_cli_config_json_and_comparison(indexed_repo, monkeypatch):
    # Keep the shared fixture connection open across multiple CLI invocations.
    class Connection:
        def __getattr__(self, name):
            return getattr(indexed_repo, name)

        def close(self):
            pass

    monkeypatch.setattr(db, "connect", lambda *args, **kwargs: Connection())
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "--config", str(CONFIG), "--retrieval-only", "--json"])
    assert result.exit_code == 0, result.output
    first = json.loads(result.stdout)
    assert len(first["rows"]) == 48
    result = runner.invoke(app, ["eval", "--structural-only", "--json"])
    assert result.exit_code == 0, result.output
    second = json.loads(result.stdout)
    result = runner.invoke(app, ["eval", "--compare", first["run_id"], second["run_id"], "--json"])
    assert result.exit_code == 0 and json.loads(result.stdout)["differences"]
    result = runner.invoke(app, ["eval", "--verified-only", "--json"])
    assert result.exit_code == 0 and len(json.loads(result.stdout)["rows"]) == 48


@pytest.mark.db
def test_eval_is_atomic_on_persistence_failure(indexed_repo, monkeypatch):
    conn = indexed_repo
    original = db.save_eval_metrics
    calls = []

    def failing(*args):
        calls.append(args[1])
        if len(calls) == 2:
            raise RuntimeError("disk failure")
        original(*args)

    monkeypatch.setattr(db, "save_eval_metrics", failing)
    with pytest.raises(RuntimeError, match="disk failure"):
        run_eval(conn, load_config(CONFIG))
    assert len(calls) == 2 and db.get_eval_run(conn, calls[0]) == []
    assert "lexical" not in db.get_view_status(conn)


@pytest.mark.db
def test_verified_selection_and_changed_suite_comparison(indexed_repo, tmp_path):
    conn = indexed_repo
    questions = load_questions(ROOT / "eval/questions.yaml")
    for question in questions:
        question["verified"] = False
    questions[0]["verified"] = True
    file = tmp_path / "suite.yaml"
    file.write_text(yaml.safe_dump({"version": 1, "questions": questions}))
    config = {"questions": str(file), "verified_only": True}
    first = run_eval(conn, config)
    assert not first["provisional"] and len(first["rows"]) == 1
    assert first["rows"][0]["metrics"]["answer_recall_at_5"] == 1
    questions[0]["question"] = "zzznonexistentword"
    file.write_text(yaml.safe_dump({"version": 1, "questions": questions}))
    second = run_eval(conn, config)
    comparison = compare_runs(conn, first["run_id"], second["run_id"])
    assert comparison["suite_changed"] is True
    assert comparison["differences"][0]["delta"] == -1


@pytest.mark.db
def test_eval_cli_prints_per_type_metrics(indexed_repo, monkeypatch):
    monkeypatch.setattr(db, "connect", lambda *args, **kwargs: indexed_repo)
    result = CliRunner().invoke(app, ["eval", "--retrieval-only"])
    assert result.exit_code == 0, result.output
    assert "Expectation review: agent" in result.output and "AnswerRecall@5" in result.output
    assert "Structural" in result.output and "traversal" in result.output
    assert indexed_repo.closed


def test_unreviewed_suite_still_rejected_by_verified_only(tmp_path):
    questions = load_questions(ROOT / "eval/questions.yaml")
    for question in questions:
        question["verified"] = False
    file = tmp_path / "draft.yaml"
    file.write_text(yaml.safe_dump({"version": 1, "questions": questions}))
    with pytest.raises(ValueError, match="verification"):
        load_questions(file, verified_only=True)
    assert len(load_questions(file)) == 48


def test_invalid_review_provenance_rejected(tmp_path):
    questions = load_questions(ROOT / "eval/questions.yaml")
    questions[0]["review"]["kind"] = "automatic-pass"
    file = tmp_path / "invalid.yaml"
    file.write_text(yaml.safe_dump({"version": 1, "questions": questions}))
    with pytest.raises(ValueError, match="provenance"):
        load_questions(file)
