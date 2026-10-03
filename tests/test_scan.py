from __future__ import annotations

import pytest

from cfl.config import Settings
from cfl.core.db import list_files
from cfl.pipeline.scan import discover_files, register_files


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass\n")
    (tmp_path / "src" / "utils.py").write_text("def helper(): return 1\n")
    (tmp_path / ".env").write_text("SECRET=abc\n")
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 2}\n')
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")
    return tmp_path


def test_discover_files_finds_python(sample_repo):
    settings = Settings()
    files = list(discover_files(sample_repo, settings))
    paths = [f.path for f in files]
    assert "src/main.py" in paths
    assert "src/utils.py" in paths


def test_discover_files_excludes_env(sample_repo):
    settings = Settings()
    files = list(discover_files(sample_repo, settings))
    paths = [f.path for f in files]
    assert ".env" not in paths


def test_discover_files_excludes_lockfile(sample_repo):
    settings = Settings()
    files = list(discover_files(sample_repo, settings))
    paths = [f.path for f in files]
    assert "package-lock.json" not in paths


@pytest.mark.db
def test_register_files(pg_conn, sample_repo):
    settings = Settings()
    files = list(discover_files(sample_repo, settings))
    result = register_files(pg_conn, files, str(sample_repo))
    assert result.changed >= 2
    db_files = list_files(pg_conn)
    db_paths = {f["path"] for f in db_files}
    assert "src/main.py" in db_paths
