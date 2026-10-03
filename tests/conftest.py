from __future__ import annotations

import os
import pathlib
import uuid

import psycopg
import pytest

from cfl.config import Settings
from cfl.core.client import OllamaClient
from cfl.core.db import connect, run_migrations
from tests.fakes import FakeOllama

CFL_DSN = os.environ.get("CFL_DSN", "postgresql://cfl:cfl@localhost:5432/cfl")


def _admin_dsn() -> str:
    """Derive an admin DSN by replacing the dbname with 'postgres'."""
    base = CFL_DSN.rsplit("/", 1)[0]
    return f"{base}/postgres"


@pytest.fixture
def pg_conn(tmp_path):
    """Create a temporary test database, run migrations, yield connection, drop DB."""
    db_name = f"cfl_test_{uuid.uuid4().hex[:12]}"
    admin_dsn = _admin_dsn()

    # Create the test database using admin connection
    with psycopg.connect(admin_dsn, autocommit=True) as admin_conn:
        admin_conn.execute(f'CREATE DATABASE "{db_name}"')

    # Build DSN for the new test DB
    test_dsn = CFL_DSN.rsplit("/", 1)[0] + f"/{db_name}"

    conn = connect(test_dsn)
    run_migrations(conn, 768, str(pathlib.Path(__file__).parent.parent / "migrations"))

    yield conn

    conn.close()

    # Drop the test database
    with psycopg.connect(admin_dsn, autocommit=True) as admin_conn:
        admin_conn.execute(f'DROP DATABASE "{db_name}" WITH (FORCE)')


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def client(fake_ollama: FakeOllama) -> OllamaClient:
    settings = Settings()
    return OllamaClient(settings, transport=fake_ollama.transport)


@pytest.fixture
def settings() -> Settings:
    return Settings(num_ctx=2048)


@pytest.fixture
def fixture_repo_path() -> pathlib.Path:
    p = pathlib.Path(__file__).parent / "fixtures" / "resolution_repo"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def resolution_data(fixture_repo_path):
    """Parsed fixture records in the same shape as the DB helpers return."""
    from dataclasses import asdict

    from cfl.core.hashing import file_sha256, symbol_id
    from cfl.parser.python_adapter import PythonAdapter

    symbols, files = [], []
    for path in sorted(fixture_repo_path.rglob("*.py")):
        relative_path = path.relative_to(fixture_repo_path).as_posix()
        parsed = PythonAdapter().parse(pathlib.Path(relative_path), path.read_text())
        assert parsed.parse_error is None
        files.append(
            {
                "path": relative_path,
                "sha256": file_sha256(path),
                "imports": [asdict(i) for i in parsed.imports],
                "exports": parsed.exports,
            }
        )
        for symbol in parsed.symbols:
            data = asdict(symbol)
            extra = {
                key: data.pop(key) for key in ("bases", "init_attrs", "param_types", "local_types")
            }
            parent = data.pop("parent_qualname")
            symbols.append(
                {
                    **data,
                    "extra": extra,
                    "file_path": relative_path,
                    "id": symbol_id(relative_path, symbol.qualname, symbol.start_line, False),
                    "parent_id": f"{relative_path}::{parent}" if parent else None,
                }
            )
    return symbols, files
