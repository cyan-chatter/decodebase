from __future__ import annotations

from db.connection import Connection


class Repository:
    """Store and retrieve tasks through a connection."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def save(self, task: dict) -> dict:
        """Persist a task; missing tasks or identifiers are invalid."""
        if task is None or not task.get("id"):
            raise ValueError("Task requires an id")
        self.connection.execute(task["id"], task)
        return task

    def get(self, task_id: str) -> dict | None:
        """Find a task by its identifier."""
        return self.connection.execute(task_id)


class AuditedRepository(Repository):
    """Repository whose writes include an audit marker."""

    def save(self, task: dict) -> dict:
        """Mark a task audited before delegating to the base repository."""
        task = dict(task, audited=True)
        return super().save(task)
