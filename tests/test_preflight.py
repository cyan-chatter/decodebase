from __future__ import annotations

import pytest

from cfl.config import Settings
from cfl.core import db
from cfl.core.preflight import check_db


@pytest.mark.db
def test_doctor_reads_key_value_schema_metadata(pg_conn, monkeypatch):
    monkeypatch.setattr(db, "connect", lambda dsn: pg_conn)
    status = check_db(Settings())
    assert status == "Connected; extensions OK (vector, pg_trgm); schema_version=1"
    assert pg_conn.closed


@pytest.mark.db
def test_diagnostics_accepts_unmigrated_database(pg_conn):
    pg_conn.execute("DROP TABLE meta")
    extensions, version = db.database_diagnostics(pg_conn)
    assert extensions == {"vector", "pg_trgm"}
    assert version is None


def test_doctor_reports_connection_failure(monkeypatch):
    def fail(dsn):
        raise OSError("unreachable")

    monkeypatch.setattr(db, "connect", fail)
    assert check_db(Settings()) == "Cannot connect to DB: unreachable"


def test_preflight_requires_exact_candidate_tag():
    from types import SimpleNamespace

    from cfl.core.errors import PreflightError
    from cfl.core.preflight import check_models_pulled

    client = SimpleNamespace(
        tags=lambda: [{"name": "qwen3:4b"}, {"name": "nomic-embed-text:latest"}]
    )
    with pytest.raises(PreflightError, match="qwen3:8b"):
        check_models_pulled(client, Settings(gen_model="qwen3:8b"))


@pytest.mark.parametrize("failure", ["missing", "spill", "budget", "context"])
def test_preflight_rejects_invalid_gpu_or_context(failure):
    from types import SimpleNamespace

    from cfl.core.errors import PreflightError
    from cfl.core.preflight import check_residency

    rows = [
        {
            "name": "qwen2.5-coder:7b",
            "size": 5_000_000_000,
            "size_vram": 5_000_000_000,
            "context_length": 8192,
        },
        {
            "name": "nomic-embed-text:latest",
            "size": 300_000_000,
            "size_vram": 300_000_000,
            "context_length": 2048,
        },
    ]
    if failure == "missing":
        rows.pop()
    elif failure == "spill":
        rows[1]["size_vram"] = 0
    elif failure == "budget":
        rows[0].update(size=10_000_000_000, size_vram=10_000_000_000)
    else:
        rows[0]["context_length"] = 4096
    with pytest.raises(PreflightError):
        check_residency(SimpleNamespace(ps=lambda: rows), Settings())


def test_preflight_accepts_untagged_latest_alias():
    from types import SimpleNamespace

    from cfl.core.preflight import check_residency

    rows = [
        {"name": "qwen2.5-coder:7b", "size": 5, "size_vram": 5, "context_length": 8192},
        {"name": "nomic-embed-text:latest", "size": 1, "size_vram": 1},
    ]
    assert check_residency(SimpleNamespace(ps=lambda: rows), Settings()) == []
