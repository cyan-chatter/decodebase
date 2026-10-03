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
