from __future__ import annotations

import pytest

from cfl.core import db
from cfl.pipeline.indexer import (
    build_lexical,
    build_search_text,
    retrieve_lexical,
    split_identifiers,
)


def test_identifier_split_and_fields():
    assert split_identifiers("parse_config HTTPResponse getURL2 parse_config") == [
        "parse",
        "config",
        "http",
        "response",
        "get",
        "url",
        "2",
    ]
    text = build_search_text(
        {
            "name": "validate_token",
            "qualname": "Auth.validate_token",
            "signature": "(accessToken)",
            "decorators": ["with_retries(3)"],
            "docstring": "Verify HMAC.",
            "summary_short": "Reject expired credentials.",
        }
    )
    assert text == text.lower()
    assert all(
        word in text
        for word in ("auth", "validate", "token", "access", "with", "retries", "hmac", "expired")
    )
    assert build_search_text({"name": "bare"}) == "bare\nbare"


@pytest.mark.db
def test_lexical_without_summaries(indexed_repo):
    conn = indexed_repo
    count = build_lexical(conn, batch_size=3)
    assert count == len(db.get_all_symbols(conn)) and count >= 40
    assert db.get_view_status(conn)["lexical"]["status"] == "fresh"
    assert all(row["summary_short"] is None for row in db.get_all_symbols(conn))
    results = retrieve_lexical(conn, "validate_token", 5)
    assert results[0]["id"] == "auth/tokens.py::validate_token"
    assert results[0]["start_line"] > 0
    assert retrieve_lexical(conn, "!!!", 5) == []
    assert retrieve_lexical(conn, "'); drop table symbols; --", 5) is not None
    assert build_lexical(conn, batch_size=4) == count


@pytest.mark.db
def test_rank_merge_deduplicates_and_orders_ties(indexed_repo, monkeypatch):
    conn = indexed_repo
    a, b = "auth/tokens.py::validate_token", "auth/tokens.py::issue_token"
    monkeypatch.setattr(db, "lexical_search", lambda *args: [{"id": a}, {"id": b}])
    monkeypatch.setattr(db, "trigram_search", lambda *args: [{"id": b}, {"id": a}])
    rows = retrieve_lexical(conn, "token", 5)
    assert [row["id"] for row in rows] == sorted([a, b])
    assert rows[0]["rank"] == rows[1]["rank"]


@pytest.mark.db
def test_module_namespace_queries_retrieve_source_files(indexed_repo):
    conn = indexed_repo
    build_lexical(conn)
    for query, path in [
        ("What does api.routes provide?", "api/routes.py"),
        ("What does services.notifications provide?", "services/notifications.py"),
    ]:
        rows = retrieve_lexical(conn, query, 5)
        assert len(rows) >= 2 and all(row["file_path"] == path for row in rows[:2])
    rows = retrieve_lexical(conn, "What is auth responsible for?", 5)
    assert any(row["file_path"] == "auth/tokens.py" for row in rows)
    assert any(row["file_path"] == "auth/passwords.py" for row in rows)
