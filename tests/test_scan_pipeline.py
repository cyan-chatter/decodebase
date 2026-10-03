from __future__ import annotations

import shutil
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from cfl.cli import app
from cfl.core import db
from cfl.parser.python_adapter import PythonAdapter
from cfl.pipeline import scan


def test_cli_scan_runs_stages_and_closes(monkeypatch):
    conn = Mock()
    monkeypatch.setattr(db, "connect", lambda dsn: conn)
    stages = []
    monkeypatch.setattr(scan, "run_stage1", lambda c, s, repo: stages.append((1, c, repo)))
    monkeypatch.setattr(scan, "run_stage2", lambda c, s, repo: stages.append((2, c, repo)))
    result = CliRunner().invoke(app, ["scan", "some/repo"])
    assert result.exit_code == 0, result.output
    assert stages == [(1, conn, "some/repo"), (2, conn, "some/repo")]
    assert "Scan complete." in result.output
    conn.close.assert_called_once()


def test_cli_scan_closes_on_failure(monkeypatch):
    conn = Mock()
    monkeypatch.setattr(db, "connect", lambda dsn: conn)

    def fail(*args):
        raise RuntimeError("scan failed")

    stage2 = Mock()
    monkeypatch.setattr(scan, "run_stage1", fail)
    monkeypatch.setattr(scan, "run_stage2", stage2)
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code == 1
    assert "Scan complete." not in result.output
    conn.close.assert_called_once()
    stage2.assert_not_called()


@pytest.mark.db
def test_scan_fixture_incremental(pg_conn, settings, fixture_repo_path, tmp_path, mocker):
    repo = tmp_path / "repo"
    shutil.copytree(fixture_repo_path, repo)
    parse = mocker.spy(PythonAdapter, "parse")
    replace = mocker.spy(db, "replace_edges")
    scan.run_stage1(pg_conn, settings, str(repo))
    scan.run_stage2(pg_conn, settings, str(repo))
    assert parse.call_count == len(list(repo.rglob("*.py")))
    assert replace.call_count == 1
    symbols = {s["id"]: s for s in db.get_all_symbols(pg_conn)}
    assert all(s["scc_id"] and s["layer"] is not None for s in symbols.values())
    assert symbols["recursion.py::is_even"]["scc_id"] == symbols["recursion.py::is_odd"]["scc_id"]
    assert symbols["entrypoint.py::entrypoint::<module>"]["entry_kind"] == "main_guard"
    ambiguous = [e for e in db.fetch_edges(pg_conn, 0.0) if e["resolution"] == "ambiguous"]
    assert len(ambiguous) == 4
    assert db.get_view_status(pg_conn)["graph"]["status"] == "fresh"

    parse.reset_mock()
    replace.reset_mock()
    scan.run_stage1(pg_conn, settings, str(repo))
    scan.run_stage2(pg_conn, settings, str(repo))
    parse.assert_not_called()
    replace.assert_not_called()

    # A line-only edit retains code hashes but must update call-site locations.
    path = repo / "recursion.py"
    path.write_text("\n" + path.read_text())
    scan.run_stage1(pg_conn, settings, str(repo))
    scan.run_stage2(pg_conn, settings, str(repo))
    parse.assert_called_once()
    assert parse.call_args.args[1].name == "recursion.py"
    replace.assert_called_once()
    factorial = db.get_symbol(pg_conn, "recursion.py::factorial")
    edge = next(
        e for e in db.fetch_edges(pg_conn, 0.0) if e["caller_id"] == "recursion.py::factorial"
    )
    assert edge["line"] == factorial["call_sites"][0]["line"]


@pytest.mark.db
def test_scan_removes_stale_edges_and_entrypoints(pg_conn, settings, tmp_path):
    path = tmp_path / "a.py"
    path.write_text("def main(): target()\ndef target(): pass\n")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    assert len(db.fetch_edges(pg_conn, 0.0)) == 1
    # Removing the call must clear edges even though the caller still exists.
    path.write_text("def main(): pass\ndef target(): pass\n")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    assert db.fetch_edges(pg_conn, 0.0) == []
    # Moving an otherwise identical file must invalidate the resolution fingerprint.
    fingerprint = db.get_meta(pg_conn, "resolve_fingerprint")
    path.rename(tmp_path / "b.py")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    assert db.get_meta(pg_conn, "resolve_fingerprint") != fingerprint
    assert db.get_symbol(pg_conn, "b.py::main")["is_entrypoint"]
    (tmp_path / "b.py").unlink()
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    assert db.list_files(pg_conn) == []
    assert db.get_all_symbols(pg_conn) == []


@pytest.mark.db
def test_scan_parse_failure_and_encoding(pg_conn, settings, tmp_path, mocker):
    path = tmp_path / "a.py"
    path.write_text("def main(): pass\n")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    path.write_text("def main(:")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    assert db.get_all_symbols(pg_conn) == []
    file = db.list_files(pg_conn)[0]
    assert file["parse_status"] == "parse_failed"
    assert file["parse_error"]
    parse = mocker.spy(PythonAdapter, "parse")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    parse.assert_not_called()
    path.write_bytes(b'# coding: latin-1\ndef cafe():\n return "caf\xe9"\n')
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    assert "café" in db.get_symbol(pg_conn, "a.py::cafe")["raw_code"]
    assert db.list_files(pg_conn)[0]["parse_status"] == "parsed"


@pytest.mark.db
def test_scan_recomputes_exports_and_threshold(pg_conn, settings, tmp_path, mocker):
    path = tmp_path / "a.py"
    path.write_text("__all__ = ['f']\ndef f(): pass\n")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    assert db.get_symbol(pg_conn, "a.py::f")["entry_kind"] == "exported"
    path.write_text("def f(): pass\n")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    symbol = db.get_symbol(pg_conn, "a.py::f")
    assert not symbol["is_entrypoint"]
    assert symbol["entry_kind"] is None
    replace = mocker.spy(db, "replace_edges")
    settings.edge_conf_threshold = 0.3
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    replace.assert_called_once()
    assert db.get_view_status(pg_conn)["graph"]["config"]["threshold"] == 0.3


@pytest.mark.db
def test_cli_scan_fixture_with_postgres(pg_conn, fixture_repo_path, monkeypatch):
    from psycopg.conninfo import make_conninfo

    from tests.conftest import CFL_DSN

    real_connect = db.connect
    opened = []

    def connect_test_database(dsn):
        conn = real_connect(make_conninfo(CFL_DSN, dbname=pg_conn.info.dbname))
        opened.append(conn)
        return conn

    monkeypatch.setattr(db, "connect", connect_test_database)
    result = CliRunner().invoke(app, ["scan", str(fixture_repo_path)])
    assert result.exit_code == 0, result.exception
    assert "Scan complete." in result.output
    assert opened[0].closed
    assert len([e for e in db.fetch_edges(pg_conn, 0.0) if e["resolution"] == "ambiguous"]) == 4


@pytest.mark.db
def test_scan_repeated_calls_on_one_line(pg_conn, settings, tmp_path):
    (tmp_path / "a.py").write_text("def f(): g(); g()\ndef g(): pass\n")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    assert len(db.fetch_edges(pg_conn, 0.0)) == 1


def test_invalid_repository_does_not_register(monkeypatch, settings, tmp_path):
    register = Mock()
    monkeypatch.setattr(scan, "register_files", register)
    with pytest.raises(ValueError, match="Repository directory does not exist"):
        scan.run_stage1(None, settings, str(tmp_path / "missing"))
    register.assert_not_called()


@pytest.mark.db
def test_encoding_fallback_and_failure_counts(pg_conn, settings, tmp_path):
    (tmp_path / "bad.py").write_text("def f(:")
    (tmp_path / "cookie.py").write_bytes(b'# coding: nonexistent\ndef f(): return "ok"\n')
    (tmp_path / "bytes.py").write_bytes(b'def f(): return "caf\xff"\n')
    result = scan.run_stage1(pg_conn, settings, str(tmp_path))
    assert (result.parsed, result.failed) == (2, 1)
    assert "caf�" in db.get_symbol(pg_conn, "bytes.py::f")["raw_code"]
    second = scan.run_stage1(pg_conn, settings, str(tmp_path))
    assert (second.parsed, second.failed) == (0, 0)


@pytest.mark.db
def test_parser_upgrade_invalidates_cached_symbols(pg_conn, settings, tmp_path, mocker):
    (tmp_path / "a.py").write_text("async def f(): pass")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    # Simulate symbols parsed by the previous adapter, with unchanged source hashes.
    db.set_meta(pg_conn, "parser_version", "python_v1")
    pg_conn.execute("UPDATE symbols SET signature = ''")
    parse = mocker.spy(PythonAdapter, "parse")
    result = scan.run_stage1(pg_conn, settings, str(tmp_path))
    assert result.parsed == 1
    parse.assert_called_once()
    assert db.get_symbol(pg_conn, "a.py::f")["signature"] == "async "
    parse.reset_mock()
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    parse.assert_not_called()


@pytest.mark.db
def test_package_exports_are_entrypoints(pg_conn, settings, fixture_repo_path):
    scan.run_stage1(pg_conn, settings, str(fixture_repo_path))
    scan.run_stage2(pg_conn, settings, str(fixture_repo_path))
    assert db.get_symbol(pg_conn, "pkg/mod.py::leaf")["entry_kind"] == "exported"


def test_scan_hidden_from_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "│ scan " not in result.stdout
    assert CliRunner().invoke(app, ["scan", "--help"]).exit_code == 0


@pytest.mark.db
def test_parser_version_invalidation_rebuilds_edges(
    pg_conn, settings, tmp_path, mocker, monkeypatch
):
    (tmp_path / "a.py").write_text("def f(): g()\ndef g(): pass")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    parse, replace = mocker.spy(PythonAdapter, "parse"), mocker.spy(db, "replace_edges")
    monkeypatch.setattr(scan, "PARSER_VERSION", "python_next")
    scan.run_stage1(pg_conn, settings, str(tmp_path))
    scan.run_stage2(pg_conn, settings, str(tmp_path))
    parse.assert_called_once()
    replace.assert_called_once()
