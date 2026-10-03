from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pathspec

if TYPE_CHECKING:
    from psycopg import Connection

    from cfl.config import Settings

# Default directories to exclude
DEFAULT_EXCLUDES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".eggs",
    "htmlcov",
    ".ruff_cache",
}

# Lockfile patterns to skip
LOCKFILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
    "poetry.lock",
    "Cargo.lock",
    "composer.lock",
}

# Secret file patterns to skip
SECRET_PATTERNS = (".env", ".pem", ".key", ".p12", ".pfx", ".crt", ".cer", ".der")

LANGUAGE_MAP = {
    ".py": "python",
    ".toml": "config",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
    ".ini": "config",
    ".cfg": "config",
    ".conf": "config",
}


@dataclass
class FileInfo:
    path: str
    sha256: str
    size_bytes: int
    language: str | None


@dataclass
class RegisterResult:
    changed: int = 0
    unchanged: int = 0
    removed: int = 0


def _is_binary(path: pathlib.Path) -> bool:
    """Check if file is binary by looking for NUL bytes in first 8KB."""
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def _is_minified(path: pathlib.Path) -> bool:
    """Heuristic: if avg line length > 1000 chars, likely minified."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            return False
        avg = sum(len(l) for l in lines) / len(lines)
        return avg > 1000
    except OSError:
        return False


def _is_secret(name: str) -> bool:
    """Check if filename matches secret file patterns."""
    lower = name.lower()
    if lower == ".env" or lower.startswith(".env."):
        return True
    for pat in SECRET_PATTERNS:
        if lower.endswith(pat):
            return True
    return False


def _detect_language(path: pathlib.Path) -> str | None:
    """Detect language from file extension."""
    return LANGUAGE_MAP.get(path.suffix.lower())


def _load_gitignore(root: pathlib.Path) -> pathspec.PathSpec | None:
    """Load .gitignore from root if it exists."""
    gitignore = root / ".gitignore"
    if gitignore.exists():
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return None


def discover_files(root: pathlib.Path, settings: Settings) -> Iterator[FileInfo]:
    """Walk root and yield FileInfo for each eligible file."""
    from cfl.core.hashing import file_sha256

    gitignore = _load_gitignore(root)

    def _walk(dirpath: pathlib.Path) -> Iterator[pathlib.Path]:
        try:
            entries = list(os.scandir(dirpath))
        except PermissionError:
            return
        for entry in sorted(entries, key=lambda e: e.name):
            if entry.is_dir(follow_symlinks=False):
                if entry.name in DEFAULT_EXCLUDES:
                    continue
                yield from _walk(pathlib.Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield pathlib.Path(entry.path)

    for abs_path in _walk(root):
        rel = abs_path.relative_to(root)
        rel_posix = rel.as_posix()

        if gitignore and gitignore.match_file(rel_posix):
            continue

        name = abs_path.name

        if name in LOCKFILE_NAMES:
            continue

        if _is_secret(name):
            continue

        try:
            size = abs_path.stat().st_size
        except OSError:
            continue
        if size > settings.max_file_bytes:
            continue

        if _is_binary(abs_path):
            continue

        if _is_minified(abs_path):
            continue

        sha = file_sha256(abs_path)
        lang = _detect_language(abs_path)

        yield FileInfo(
            path=rel_posix,
            sha256=sha,
            size_bytes=size,
            language=lang,
        )


def register_files(
    conn: Connection,
    files: list[FileInfo],
    repo_root: str,
) -> RegisterResult:
    """Upsert discovered files into DB, remove missing ones."""
    from cfl.core.db import delete_missing_files, upsert_file

    result = RegisterResult()
    present_paths: set[str] = set()

    for fi in files:
        present_paths.add(fi.path)
        changed = upsert_file(
            conn,
            path=fi.path,
            sha256=fi.sha256,
            language=fi.language,
            size_bytes=fi.size_bytes,
            token_est=None,
        )
        if changed:
            result.changed += 1
        else:
            result.unchanged += 1

    removed = delete_missing_files(conn, present_paths)
    result.removed = removed
    return result


def run_stage1(conn: Connection, settings: Settings, repo_root: str) -> None:
    """Stage 1: scan files and register in DB."""
    root = pathlib.Path(repo_root)
    files = list(discover_files(root, settings))
    register_files(conn, files, repo_root)
