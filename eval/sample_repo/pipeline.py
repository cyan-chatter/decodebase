from __future__ import annotations

from db.repository import Repository


def transform(repo: Repository, record: dict) -> dict:
    """Normalize a raw task record's identifier and whitespace in the title."""
    normalized = {"id": str(record["id"]), "title": record["title"].strip()}
    return write_to_db(repo, normalized)


def write_to_db(repo: Repository, record: dict) -> dict:
    """Persist a normalized incoming record through the repository."""
    return repo.save(record)


def run_pipeline(repo: Repository, records: list[dict]) -> list[dict]:
    """Normalize and write each incoming task record."""
    return [transform(repo, record) for record in records]
