from itertools import pairwise

import pytest

from cfl.config import Settings
from cfl.core import db
from cfl.core.errors import LLMValidationError
from cfl.engines.ask import answer, block_for
from cfl.engines.guardrails import (
    PARTIAL,
    ClaimReview,
    Review,
    Support,
    guard_answer,
    refresh_review,
    review_claims,
    verifier_session,
)
from cfl.pipeline.knowledge import generate_knowledge, validate_knowledge


def review_output(data, rejected=(), missing=()):
    s = data["sources"][0]
    return {
        "complete": not missing,
        "missing": list(missing),
        "claims": [
            {
                "index": i,
                "verdict": "contradicted" if i in rejected else "supported",
                "reason": "unsupported behavior"
                if i in rejected
                else "implementation supports this",
                "evidence": []
                if i in rejected
                else [{"id": s["id"], "quote": s["code"].splitlines()[0]}],
            }
            for i, _ in data["claims"]
        ],
    }


def test_invented_quote_duplicate_and_missing_claims_are_not_accepted(client, fake_ollama):
    sources = [
        {
            "id": "a.py::f",
            "file_path": "a.py",
            "name": "f",
            "qualname": "f",
            "start_line": 1,
            "end_line": 2,
            "raw_code": "def f(x):\n    return x",
        }
    ]
    for mutation in [
        lambda r: r["claims"][0]["evidence"][0].update(quote="network.send(x)"),
        lambda r: r["claims"].append(r["claims"][0].copy()),
        lambda r: r["claims"].clear(),
    ]:

        def result(data, mutation=mutation):
            r = review_output(data)
            mutation(r)
            return r

        fake_ollama.review_override = result
        reviewed = review_claims(client, Settings(), ["Returns input"], sources, "What does f do?")
        assert not reviewed.complete and reviewed.claims[0].verdict == "unsupported"


def test_return_only_stub_contradiction_survives_overpermissive_verifier(client, fake_ollama):
    sources = [
        {
            "id": "a.py::send_email",
            "file_path": "a.py",
            "name": "send_email",
            "qualname": "send_email",
            "start_line": 1,
            "end_line": 2,
            "raw_code": "def send_email(message):\n    return message",
        }
    ]
    result = review_claims(
        client, Settings(), ["send_email delivers an email"], sources, "What happens?"
    )
    assert result.claims[0].verdict == "contradicted"


@pytest.mark.db
def test_partial_output_filters_bad_paragraph_and_buffers_stream(ready):
    conn, client, settings, fake = ready
    fake.answer_text = "Retries RuntimeError [utils/retry.py:6-23].\n\nAlso deletes the database [utils/retry.py:6-23]."
    fake.review_override = lambda data: review_output(data, rejected=(1,))
    emitted = []
    result = answer(conn, client, settings, "Where are retries handled?", on_token=emitted.append)
    assert result["status"] == "partial" and result["text"].startswith(PARTIAL)
    assert "deletes the database" not in result["text"]
    assert emitted == [result["text"]]
    assert not answer(conn, client, settings, "Where are retries handled?")["cached"]


@pytest.mark.db
def test_mixed_valid_and_unsupported_citations_abstain_instead_of_stripping_only_tag(ready):
    conn, client, settings, fake = ready
    fake.answer_text = "True fact [utils/retry.py:6-23]. Fake fact [missing.py:1-9]."
    result = answer(conn, client, settings, "Where are retries handled?")
    assert result["status"] == "abstained" and not result["sufficient"]
    assert "Fake fact" not in result["text"]


@pytest.mark.db
def test_missing_verifier_abstains_without_releasing_draft(ready):
    conn, client, settings, _ = ready
    settings = settings.model_copy(update={"verifier_model": "missing:model"})
    emitted = []
    result = answer(conn, client, settings, "Where are retries handled?", on_token=emitted.append)
    assert result["status"] == "abstained" and emitted == [result["text"]]
    assert "Source-grounded test answer" not in result["text"]


@pytest.mark.db
def test_knowledge_rejection_removed_from_publication_and_embedding_inputs(ready):
    conn, client, settings, fake = ready
    rows = db.knowledge_drafts(conn)
    assert db.published_knowledge(conn)
    fake.review_override = lambda data: review_output(
        data, rejected=tuple(i for i, _ in data["claims"])
    )
    # Force re-review by changing verifier certificate, while preserving draft/source.
    for row in rows:
        db.save_knowledge_review(
            conn, row["id"], row["draft_hash"], "pending", None, {}, None, None
        )
    validate_knowledge(conn, client, settings)
    assert not db.published_knowledge(conn) and not db.summary_units(conn)
    assert all(r["status"] == "rejected" for r in db.knowledge_drafts(conn))


@pytest.mark.db
def test_raw_summary_is_not_retrieval_evidence(ready):
    conn, _, _, _ = ready
    s = db.get_symbol(conn, "services/notifications.py::send_email")
    s.update(summary_short="Actually sends real emails", summary_json={"purpose": "delivers email"})
    block = block_for(s)[0]
    assert "Actually sends" not in block["text"] and block["summary_json"] is None


@pytest.mark.db
def test_unchanged_two_pass_resume_reuses_drafts_and_review_certificates(ready):
    conn, client, settings, fake = ready
    before = {r["id"]: (r["draft_hash"], r["published"]) for r in db.knowledge_drafts(conn)}
    fake.call_log.clear()
    generated = generate_knowledge(conn, client, settings)
    reviewed = validate_knowledge(conn, client, settings)
    assert generated["drafts"] == len(before) and not generated["generation_failures"]
    assert {
        "feature:api/routes.py::create_handler",
        "feature:api/routes.py::complete_handler",
    } <= before.keys(), [
        (s["id"], s["decorators"]) for s in db.get_all_symbols(conn) if "handler" in s["name"]
    ]
    assert reviewed["cached"] == len(before)
    generations = [
        c
        for c in fake.call_log
        if c["path"] == "/api/chat" or (c["path"] == "/api/generate" and c["body"].get("prompt"))
    ]
    assert not generations, generations
    assert before == {r["id"]: (r["draft_hash"], r["published"]) for r in db.knowledge_drafts(conn)}


@pytest.mark.db
def test_physical_source_change_cannot_reuse_reviewed_prose(ready, tmp_path):
    from pathlib import Path

    conn, client, settings, fake = ready
    symbol = db.get_symbol(conn, "services/notifications.py::send_email")
    path = tmp_path / symbol["file_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    original = Path(db.get_meta(conn, "repo_root")) / symbol["file_path"]
    path.write_text(original.read_text() + "\n# source changed since scan\n")
    with conn.transaction():
        db.set_meta(conn, "repo_root", str(tmp_path))
    fake.call_log.clear()
    checked = guard_answer(
        conn,
        client,
        settings,
        "send_email returns message unchanged [services/notifications.py:9-11].",
        block_for(symbol),
        "Explain send_email",
    )
    assert checked.status == "abstained" and not checked.citations
    assert not any(c["path"] == "/api/generate" for c in fake.call_log)


def test_same_checkpoint_verifier_is_rejected(client):
    with (
        pytest.raises(LLMValidationError),
        verifier_session(client, Settings(verifier_model="qwen3.5:9b")),
    ):
        pass


def test_cached_review_refresh_rejects_docstring_proof_and_empty_guarantees():
    source = {
        "id": "a.py::f",
        "file_path": "a.py",
        "name": "f",
        "qualname": "f",
        "start_line": 1,
        "end_line": 3,
        "raw_code": 'def f(x):\n    """Sends an email."""\n    return x',
    }
    claims = ["f sends an email", "Scope `f`: raises: []", "f returns x"]
    review = Review(
        complete=True,
        missing=[],
        claims=[
            ClaimReview(
                index=i,
                verdict="supported",
                reason="previous review",
                evidence=[Support(id=source["id"], quote=quote)],
            )
            for i, quote in enumerate(['    """Sends an email."""', "def f(x):", "    return x"])
        ],
    )
    refreshed = refresh_review(review, claims, [source])
    assert [c.verdict for c in refreshed.claims] == ["contradicted", "contradicted", "supported"]


@pytest.mark.parametrize(
    "claim",
    [
        "Connection.execute is not thread-safe.",
        "The annotations import evaluates annotations later at runtime.",
        "The import does not affect runtime performance significantly.",
        "This securely stores passwords and prevents timing attacks.",
        "All inputs are immutable strings.",
    ],
)
def test_external_guarantees_cannot_pass_a_permissive_reviewer(client, claim):
    source = {
        "id": "a.py::f",
        "file_path": "a.py",
        "name": "f",
        "qualname": "f",
        "start_line": 1,
        "end_line": 3,
        "raw_code": "from __future__ import annotations\ndef f(x: str):\n    return x",
    }
    reviewed = review_claims(client, Settings(), [claim], [source], "Explain the code")
    assert reviewed.claims[0].verdict == "contradicted"


def test_disclaimers_are_not_answer_claims_and_invented_errors_cannot_pass(client):
    from cfl.engines.guardrails import claims_from_text

    assert claims_from_text(PARTIAL) == []
    source = {
        "id": "a.py::f",
        "file_path": "a.py",
        "name": "f",
        "qualname": "f",
        "start_line": 1,
        "end_line": 2,
        "raw_code": "def f(x):\n    return x",
    }
    for claim in [
        "f raises json.JSONEncodeError",
        "f raises UnicodeDecodeError if the file encoding is not UTF-8",
        "f has no side effects",
        "f notifies subscribers",
        "Task notifications are delivered via email through send_email",
        "Citation: [a.py:1-2]",
        "- Citation: [a.py:1-2]",
        "This is an unconditional call within the function body",
        "This call is guarded by the enclosing boolean expression",
        "repo.get retrieves a task or raises KeyError if missing",
        "No additional evidence is required",
    ]:
        checked = review_claims(client, Settings(), [claim], [source], "Explain f")
        assert checked.claims[0].verdict == "contradicted"


@pytest.mark.db
def test_reviewer_prompt_contains_raw_source_without_draft_summaries(ready):
    conn, client, settings, fake = ready
    s = db.get_symbol(conn, "services/notifications.py::send_email")
    fake.review_override = lambda data: review_output(data)
    checked = guard_answer(
        conn,
        client,
        settings,
        "Returns input unchanged [services/notifications.py:9-11].",
        block_for(s),
        "What does send_email do?",
    )
    assert checked.sufficient
    requests = [
        c["body"]
        for c in fake.call_log
        if c["path"] == "/api/generate"
        and "claims" in c["body"].get("format", {}).get("properties", {})
    ]
    import json

    data = json.loads(requests[-1]["prompt"][requests[-1]["prompt"].index('{"question"') :])
    assert data["sources"][0]["code"] == s["raw_code"]
    assert "summary_json" not in data["sources"][0]


def test_short_circuit_and_copy_errors_survive_permissive_review(client):
    cases = [
        (
            {
                "id": "a.py::validate_token",
                "file_path": "a.py",
                "name": "validate_token",
                "qualname": "validate_token",
                "start_line": 1,
                "end_line": 4,
                "raw_code": 'def validate_token(token, secret):\n    uid, separator, value = token.partition(":")\n    if not separator or not compare(token, issue_token(uid, secret)):\n        raise ValueError()',
            },
            "validate_token invokes issue_token if separator is missing or comparison fails",
        ),
        (
            {
                "id": "a.py::save",
                "file_path": "a.py",
                "name": "save",
                "qualname": "save",
                "start_line": 1,
                "end_line": 3,
                "raw_code": "def save(task):\n    task = dict(task, audited=True)\n    return task",
            },
            "save modifies the input dictionary",
        ),
    ]
    for source, claim in cases:
        reviewed = review_claims(
            client, Settings(), [claim], [source], "Explain the implementation"
        )
        assert reviewed.claims[0].verdict == "contradicted"


def test_unrelated_language_history_is_not_proved_by_import(client):
    source = {
        "id": "a.py::f",
        "file_path": "a.py",
        "name": "f",
        "qualname": "f",
        "start_line": 1,
        "end_line": 3,
        "raw_code": "from __future__ import annotations\ndef f():\n    return 1",
    }
    r = review_claims(
        client,
        Settings(),
        ["PEP 563 enables type hints in Python versions prior to 3.7"],
        [source],
        "Explain file",
    )
    assert r.claims[0].verdict == "contradicted"


def test_large_file_parts_cover_every_line_without_truncation(client):
    from cfl.pipeline.knowledge import file_parts

    raw = "".join(f"def f{i}():\n    return {i}\n" for i in range(300))
    source = {
        "id": "file:a.py",
        "file_path": "a.py",
        "start_line": 1,
        "end_line": 600,
        "raw_code": raw,
    }
    parts = file_parts(source, client, Settings(num_ctx=2048, num_predict_detailed=350))
    assert len(parts) > 1 and "".join(p["raw_code"] for p in parts) == raw
    assert parts[0]["start_line"] == 1 and parts[-1]["end_line"] == 600
    assert all(a["end_line"] + 1 == b["start_line"] for a, b in pairwise(parts))


def test_filter_keeps_scope_when_a_feature_heading_is_removed():
    from cfl.engines.guardrails import claims_from_text

    claims = claims_from_text(
        "2. **`db/repository.py::Repository.get`:**\n\n* **Returns:** Returns a dict or None.\n\n3. **`services/tasks.py::complete_task`**:\n\n* **Returns:** Returns the updated task.",
        subject="services/tasks.py::complete_task",
    )
    assert len(claims) == 2
    assert claims[0].startswith("Scope `db/repository.py::Repository.get`:")
    assert claims[1].startswith("Scope `services/tasks.py::complete_task`:")
