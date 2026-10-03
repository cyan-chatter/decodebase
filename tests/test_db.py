from __future__ import annotations

import pytest

from cfl.core.db import (
    get_meta,
    set_meta,
    bump_epoch,
    upsert_file,
    list_files,
    delete_missing_files,
    sync_symbols,
    ParsedRow,
    replace_edges,
    fetch_edges,
    status_counts,
)

pytestmark = pytest.mark.db


def test_meta_roundtrip(pg_conn):
    set_meta(pg_conn, "test_key", "hello")
    assert get_meta(pg_conn, "test_key") == "hello"


def test_meta_missing(pg_conn):
    assert get_meta(pg_conn, "nonexistent") is None


def test_bump_epoch(pg_conn):
    bump_epoch(pg_conn)
    v1 = get_meta(pg_conn, "epoch")
    bump_epoch(pg_conn)
    v2 = get_meta(pg_conn, "epoch")
    assert int(v2) == int(v1) + 1


def test_upsert_and_list_files(pg_conn):
    upsert_file(pg_conn, "src/a.py", "abc123", "python", 500, 100)
    files = list_files(pg_conn)
    assert any(f["path"] == "src/a.py" for f in files)


def test_delete_missing_files(pg_conn):
    upsert_file(pg_conn, "src/a.py", "abc", "python", 100, None)
    upsert_file(pg_conn, "src/b.py", "def", "python", 100, None)
    removed = delete_missing_files(pg_conn, {"src/a.py"})
    assert removed == 1
    files = list_files(pg_conn)
    assert all(f["path"] != "src/b.py" for f in files)


def test_delete_missing_empty_set(pg_conn):
    upsert_file(pg_conn, "src/a.py", "abc", "python", 100, None)
    removed = delete_missing_files(pg_conn, set())
    assert removed == 1
    assert list_files(pg_conn) == []


def test_sync_symbols(pg_conn):
    upsert_file(pg_conn, "src/a.py", "abc", "python", 100, None)
    rows = [
        ParsedRow(
            kind="function",
            qualname="foo",
            name="foo",
            parent_qualname=None,
            signature="def foo():",
            decorators=[],
            docstring=None,
            start_line=1,
            end_line=5,
            raw_code="def foo(): pass",
            code_hash="hash1",
            token_est=10,
            call_sites=[],
            extra={},
            is_async=False,
        )
    ]
    result = sync_symbols(pg_conn, "src/a.py", rows)
    assert result.inserted == 1
    assert result.unchanged == 0


def test_sync_symbols_unchanged(pg_conn):
    upsert_file(pg_conn, "src/a.py", "abc", "python", 100, None)
    row = ParsedRow(
        kind="function",
        qualname="foo",
        name="foo",
        parent_qualname=None,
        signature="def foo():",
        decorators=[],
        docstring=None,
        start_line=1,
        end_line=5,
        raw_code="def foo(): pass",
        code_hash="hash1",
        token_est=10,
        call_sites=[],
        extra={},
        is_async=False,
    )
    sync_symbols(pg_conn, "src/a.py", [row])
    result2 = sync_symbols(pg_conn, "src/a.py", [row])
    assert result2.unchanged == 1
    assert result2.inserted == 0


def test_status_counts(pg_conn):
    upsert_file(pg_conn, "src/a.py", "abc", "python", 100, None)
    sync_symbols(
        pg_conn,
        "src/a.py",
        [
            ParsedRow(
                kind="function",
                qualname="bar",
                name="bar",
                parent_qualname=None,
                signature="def bar():",
                decorators=[],
                docstring=None,
                start_line=1,
                end_line=3,
                raw_code="def bar(): pass",
                code_hash="h2",
                token_est=5,
                call_sites=[],
                extra={},
                is_async=False,
            )
        ],
    )
    counts = status_counts(pg_conn)
    assert counts.get("pending", 0) >= 1


def test_replace_and_fetch_edges(pg_conn):
    from cfl.core.db import get_symbol
    upsert_file(pg_conn, "src/a.py", "abc", "python", 100, None)
    # Create both symbols first
    sync_symbols(
        pg_conn,
        "src/a.py",
        [
            ParsedRow(
                kind="function",
                qualname="foo",
                name="foo",
                parent_qualname=None,
                signature="def foo():",
                decorators=[],
                docstring=None,
                start_line=1,
                end_line=3,
                raw_code="def foo(): pass",
                code_hash="h_foo",
                token_est=5,
                call_sites=[],
                extra={},
                is_async=False,
            ),
            ParsedRow(
                kind="function",
                qualname="bar",
                name="bar",
                parent_qualname=None,
                signature="def bar():",
                decorators=[],
                docstring=None,
                start_line=5,
                end_line=7,
                raw_code="def bar(): pass",
                code_hash="h_bar",
                token_est=5,
                call_sites=[],
                extra={},
                is_async=False,
            ),
        ],
    )
    # Verify symbols exist
    foo_sym = get_symbol(pg_conn, "src/a.py::foo")
    bar_sym = get_symbol(pg_conn, "src/a.py::bar")
    assert foo_sym is not None
    assert bar_sym is not None

    edge_rows = [
        {
            "caller_id": "src/a.py::foo",
            "callee_id": "src/a.py::bar",
            "callee_expr": "bar()",
            "line": 2,
            "kind": "call",
            "resolution": "local",
            "source": "ast",
            "confidence": 0.9,
            "control_ctx": None,
        }
    ]
    replace_edges(pg_conn, "ast", edge_rows)
    # Just verify the edge was inserted
    edge_count = pg_conn.execute("SELECT COUNT(*) FROM edges WHERE source = 'ast'").fetchone()[0]
    assert edge_count == 1
