from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from psycopg.conninfo import make_conninfo
from typer.testing import CliRunner

from cfl import cli
from cfl.core import db
from cfl.core.client import OllamaClient
from cfl.core.errors import AmbiguousSymbol, CflError, SymbolNotFound
from cfl.engines import graph_queries as queries
from cfl.pipeline.scan import run_stage1, run_stage2
from tests.conftest import CFL_DSN


@pytest.fixture
def indexed(pg_conn, fixture_repo_path, settings, monkeypatch, fake_ollama):
    def forbidden(*args, **kwargs):
        pytest.fail("Graph queries must not create an Ollama client")

    monkeypatch.setattr(OllamaClient, "__init__", forbidden)
    run_stage1(pg_conn, settings, str(fixture_repo_path))
    run_stage2(pg_conn, settings, str(fixture_repo_path))
    yield pg_conn
    assert fake_ollama.call_log == []


@pytest.fixture
def query_cli(indexed, monkeypatch, settings):
    config = settings.model_copy(update={"dsn": make_conninfo(CFL_DSN, dbname=indexed.info.dbname)})
    monkeypatch.setattr(cli, "get_settings", lambda: config)
    return CliRunner()


@pytest.mark.db
def test_callers_match_expected_fixture(indexed, fixture_repo_path):
    expected = json.loads((fixture_repo_path / "expected_edges.json").read_text())
    for qualname in ["process", "Repository.save", "is_even", "factorial"]:
        symbol = queries.resolve_symbol(indexed, qualname)
        rows = queries.callers(indexed, symbol, min_conf=0.3)
        caller_set = {row["caller_id"] for row in rows}
        expected_set = {e["caller_id"] for e in expected if e["callee_qualname"] == qualname}
        assert caller_set == expected_set
        assert all(row["location"].endswith(f":{row['line']}") for row in rows)
        assert all(row["source"] == "ast" for row in rows)


@pytest.mark.db
def test_confidence_and_external_edges(indexed):
    rows = queries.callers(indexed, "models.py::Repository.save", min_conf=0.6)
    assert {row["caller_id"] for row in rows} == {
        "services.py::create_task",
        "models.py::Manager.commit",
    }
    rows = queries.callees(indexed, "external.py::do_work", min_conf=0.0)
    assert {row["symbol"] for row in rows} == {"os.path.join", "len", "d.get"}
    assert all(row["id"] is None and row["confidence"] == 0 for row in rows)


@pytest.mark.db
def test_resolution_ranking_and_ambiguity(indexed):
    assert (
        queries.resolve_symbol(indexed, "models.py::Repository.save").qualname == "Repository.save"
    )
    assert queries.resolve_symbol(indexed, "Repository.save").id == "models.py::Repository.save"
    assert queries.resolve_symbol(indexed, "aliases.py::run").name == "run"
    with pytest.raises(AmbiguousSymbol) as exc:
        queries.resolve_symbol(indexed, "save")
    assert {s.qualname for s in exc.value.candidates} == {"Base.save", "Repository.save"}
    with pytest.raises(SymbolNotFound):
        queries.resolve_symbol(indexed, "nonexistent_xyz_symbol")
    assert queries.resolve_symbol(indexed, "factrial").name == "factorial"
    assert queries.resolve_symbol(indexed, "save_record").qualname == "save_record"


@pytest.mark.db
def test_where_all_locations_and_literal_lookup(indexed):
    rows = queries.where(indexed, "save")
    assert {"Base.save", "Repository.save", "save_record"} <= {row["qualname"] for row in rows}
    assert all(row["location"].startswith(row["file_path"] + ":") for row in rows)
    assert queries.where(indexed, "%") == []
    assert len(queries.where(indexed, "factorial")) == 1


@pytest.fixture
def small_graph(pg_conn, settings, tmp_path):
    (tmp_path / "graph.py").write_text("\n".join(f"def {name}(): pass" for name in "abcdxz"))
    run_stage1(pg_conn, settings, str(tmp_path))
    pairs = [("a", "b"), ("b", "c"), ("c", "b"), ("a", "d"), ("d", "c"), ("c", "x")]
    edges = [
        {
            "caller_id": f"graph.py::{a}",
            "callee_id": f"graph.py::{b}",
            "callee_expr": b,
            "line": i + 10,
            "kind": "call",
            "resolution": "local",
            "confidence": 0.9,
            "source": "ast",
            "control_ctx": "",
        }
        for i, (a, b) in enumerate(pairs)
    ]
    db.replace_edges(pg_conn, "ast", edges)
    return pg_conn


@pytest.mark.db
def test_shortest_path_backtracks_acyclic_and_cyclic_graph(small_graph):
    hops = queries.path(small_graph, "graph.py::a", "graph.py::x")
    assert [hop["callee_id"] for hop in hops] == ["graph.py::b", "graph.py::c", "graph.py::x"]
    assert [hop["line"] for hop in hops] == [10, 11, 15]
    assert queries.path(small_graph, "graph.py::a", "graph.py::x", max_depth=2) is None
    assert queries.path(small_graph, "graph.py::x", "graph.py::a") is None
    assert queries.path(small_graph, "graph.py::a", "graph.py::a") == []
    cycle = queries.path(small_graph, "graph.py::c", "graph.py::b")
    assert len(cycle) == 1


@pytest.mark.db
def test_impact_terminates_on_cycles_and_counts_unique_symbols(small_graph):
    result = queries.impact(small_graph, "graph.py::x", depth=100)
    assert result["total"] == 4
    assert [(g["depth"], g["count"]) for g in result["groups"]] == [(1, 1), (2, 2), (3, 1)]
    assert {s["id"] for group in result["groups"] for s in group["symbols"]} == {
        f"graph.py::{name}" for name in "abcd"
    }
    assert queries.impact(small_graph, "graph.py::c")["total"] == 3


@pytest.mark.db
def test_hubs_distinct_counts_and_threshold(indexed):
    rows = queries.hubs(indexed, n=3)
    assert len(rows) == 3
    assert [r["pagerank"] for r in rows] == sorted([r["pagerank"] for r in rows], reverse=True)
    all_hubs = {r["id"]: r for r in queries.hubs(indexed, n=100)}
    assert all_hubs["models.py::Repository.save"]["fan_in"] == 2
    all_hubs = {r["id"]: r for r in queries.hubs(indexed, n=100, min_conf=0.3)}
    assert all_hubs["models.py::Repository.save"]["fan_in"] == 3


@pytest.mark.db
def test_dead_exclusions(pg_conn, settings, tmp_path):
    (tmp_path / "a.py").write_text(
        "__all__ = ['public']\ndef public(): pass\ndef main(): pass\n"
        "def unused(): pass\nclass Base:\n def save(self): pass\n"
        "class Child(Base):\n def save(self): pass\n def own(self): pass\n"
        " def __init__(self): pass\n def __repr__(self): pass\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "helper.py").write_text("def test_helper(): pass")
    (tmp_path / "test_misc.py").write_text("def test_misc(): pass")
    (tmp_path / "misc_test.py").write_text("def misc(): pass")
    run_stage1(pg_conn, settings, str(tmp_path))
    run_stage2(pg_conn, settings, str(tmp_path))
    rows = queries.dead(pg_conn)
    names = {row["qualname"] for row in rows}
    assert {"unused", "Base.save", "Child.own"} <= names
    assert (
        not {
            "public",
            "main",
            "Child.save",
            "Child.__init__",
            "Child.__repr__",
            "test_helper",
            "test_misc",
            "misc",
        }
        & names
    )
    assert all(row["label"] == "candidate" for row in rows)
    names = {row["qualname"] for row in queries.dead(pg_conn, include_tests=True)}
    assert {"test_helper", "test_misc", "misc"} <= names


@pytest.mark.db
@pytest.mark.parametrize(
    "args",
    [
        ["callers", "is_even"],
        ["callees", "is_even"],
        ["path", "is_even", "is_odd"],
        ["impact", "is_even"],
        ["hubs", "--top", "3"],
        ["dead"],
        ["where", "save"],
    ],
)
def test_cli_json_all_graph_commands(query_cli, args):
    result = query_cli.invoke(cli.app, [*args, "--json"])
    assert result.exit_code == 0, result.exception
    payload = json.loads(result.stdout)
    assert isinstance(payload, (list, dict))
    assert result.stderr == ""


@pytest.mark.db
def test_cli_table_and_ambiguity(query_cli):
    result = query_cli.invoke(cli.app, ["callers", "is_even"])
    assert result.exit_code == 0
    assert "Confidence" in result.stderr and "Source" in result.stderr
    assert "is_odd" in result.stderr
    result = query_cli.invoke(cli.app, ["callers", "save", "--json"])
    assert result.exit_code == 2
    assert len(json.loads(result.stdout)["candidates"]) == 2
    result = query_cli.invoke(cli.app, ["callers", "save"])
    assert result.exit_code == 2
    assert "models.py::Base.save" in result.stderr
    result = query_cli.invoke(cli.app, ["callees", "save_record", "--min-conf", "0.3"])
    assert result.exit_code == 0
    assert "? ambiguous" in result.stderr
    result = query_cli.invoke(cli.app, ["where", "no_xyz_match", "--json"])
    assert json.loads(result.stdout) == []
    result = query_cli.invoke(cli.app, ["callers", "no_xyz_match", "--json"])
    assert result.exit_code == 1
    assert "No symbol found" in json.loads(result.stdout)["error"]


def test_interactive_ambiguity_selection(monkeypatch):
    symbols = [queries.Symbol(str(i), f"C{i}.save", "save", "f.py", "method", 1, 2) for i in (1, 2)]

    def ambiguous(conn, query):
        raise AmbiguousSymbol(symbols)

    monkeypatch.setattr(queries, "resolve_symbol", ambiguous)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "console", Mock(is_terminal=True))
    prompt = Mock(return_value=2)
    monkeypatch.setattr(cli.typer, "prompt", prompt)
    assert cli._resolve_query_symbol(None, "save", False) == symbols[1]
    prompt.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [
        ["callers", "f", "--depth", "0"],
        ["callees", "f", "--min-conf", "1.5"],
        ["hubs", "--top", "0"],
        ["db", "reset-embeddings", "--dim", "0"],
    ],
)
def test_cli_rejects_invalid_options(args, monkeypatch):
    connect = Mock()
    monkeypatch.setattr(db, "connect", connect)
    assert CliRunner().invoke(cli.app, args).exit_code == 2
    connect.assert_not_called()


def test_cli_closes_connection_on_query_failure(monkeypatch, settings):
    conn = Mock()
    monkeypatch.setattr(db, "connect", lambda *args, **kwargs: conn)

    def fail(*args):
        raise SymbolNotFound("missing")

    monkeypatch.setattr(queries, "resolve_symbol", fail)
    result = CliRunner().invoke(cli.app, ["callers", "missing", "--json"])
    assert result.exit_code == 1
    conn.close.assert_called_once()


@pytest.mark.db
def test_db_migrate_guard_and_reset_preserves_graph(indexed, settings):
    migrations = str(Path(__file__).parents[1] / "migrations")
    db.run_migrations(indexed, 768, migrations)
    assert db.get_meta(indexed, "embed_dim") == "768"
    with pytest.raises(CflError, match="reset-embeddings --dim 4"):
        db.run_migrations(indexed, 4, migrations)
    # Existing rows and caches are cleared; structural index and summaries survive.
    indexed.execute(
        "INSERT INTO embeddings VALUES ('hash', %s)", ("[" + ",".join(["0"] * 768) + "]",)
    )
    indexed.execute("UPDATE symbols SET embed_hash='hash', summary_short='kept'")
    indexed.execute("UPDATE files SET embed_hash='hash'")
    symbols_before = len(db.get_all_symbols(indexed))
    edges_before = db.fetch_edges(indexed, 0.0)
    db.reset_embeddings(indexed, 4)
    assert db.embedding_dimension(indexed) == 4
    assert db.get_meta(indexed, "embed_dim") == "4"
    assert indexed.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert all(
        s["embed_hash"] is None and s["summary_short"] == "kept"
        for s in db.get_all_symbols(indexed)
    )
    assert all(f["embed_hash"] is None for f in db.list_files(indexed))
    assert len(db.get_all_symbols(indexed)) == symbols_before
    assert db.fetch_edges(indexed, 0.0) == edges_before
    assert db.get_view_status(indexed)["dense"]["status"] == "stale"
    db.run_migrations(indexed, 4, migrations)
    with pytest.raises(ValueError, match="positive integer"):
        db.reset_embeddings(indexed, 0)
    assert db.embedding_dimension(indexed) == 4


@pytest.mark.db
def test_database_cli_migrate_and_reset(query_cli, indexed, monkeypatch, settings):
    config = settings.model_copy(
        update={
            "dsn": make_conninfo(CFL_DSN, dbname=indexed.info.dbname),
            "migrations_dir": str(Path(__file__).parents[1] / "migrations"),
        }
    )
    monkeypatch.setattr(cli, "get_settings", lambda: config)
    result = query_cli.invoke(cli.app, ["db", "migrate"])
    assert result.exit_code == 0, result.exception
    assert "migrations complete" in result.stderr
    result = query_cli.invoke(cli.app, ["db", "reset-embeddings", "--dim", "4"])
    assert result.exit_code == 0, result.exception
    assert db.embedding_dimension(indexed) == 4
    result = query_cli.invoke(cli.app, ["db", "migrate"])
    assert result.exit_code == 1
    assert "reset-embeddings" in result.stderr


def test_repository_settings_environment_precedence(tmp_path, monkeypatch):
    from cfl.config import load_settings

    (tmp_path / "cfl.toml").write_text("edge_conf_threshold = 0.2\nembed_dim = 4\n")
    monkeypatch.setenv("CFL_EDGE_CONF_THRESHOLD", "0.8")
    settings = load_settings(tmp_path)
    assert settings.edge_conf_threshold == 0.8
    assert settings.embed_dim == 4


@pytest.mark.db
def test_path_on_wholly_acyclic_graph(pg_conn, settings, tmp_path):
    (tmp_path / "a.py").write_text("def a(): b()\ndef b(): c()\ndef c(): pass")
    run_stage1(pg_conn, settings, str(tmp_path))
    run_stage2(pg_conn, settings, str(tmp_path))
    hops = queries.path(pg_conn, "a.py::a", "a.py::c")
    assert [(hop["caller_id"], hop["callee_id"], hop["line"]) for hop in hops] == [
        ("a.py::a", "a.py::b", 1),
        ("a.py::b", "a.py::c", 2),
    ]
    assert queries.path(pg_conn, "a.py::a", "a.py::c", max_depth=1) is None
