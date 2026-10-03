from __future__ import annotations

from db.repository import Repository
from utils.retry import with_retries

from services.notifications import notify


@with_retries(attempts=3)
def create_task(repo: Repository, title: str) -> dict:
    """Create and persist a task, then notify subscribers; retry transient errors."""
    task = repo.save({"id": title.lower().replace(" ", "-"), "title": title})
    notify(task)
    return task


def complete_task(repo: Repository, task_id: str) -> dict:
    """Mark an existing task complete; missing tasks raise KeyError."""
    task = repo.get(task_id)
    if task is None:
        raise KeyError(task_id)
    task["done"] = True
    return repo.save(task)


def process(repo: Repository, titles: list[str]) -> list[dict]:
    """Process titles into stored task records."""
    return [create_task(repo, title) for title in titles]
