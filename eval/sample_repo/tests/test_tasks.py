from __future__ import annotations

from db.connection import Connection
from db.repository import Repository
from services.tasks import complete_task, create_task


def test_create_and_complete() -> None:
    repository = Repository(Connection())
    task = create_task(repository, "Write report")
    assert complete_task(repository, task["id"])["done"] is True
