from __future__ import annotations

import json


def save(task: dict, path: str) -> None:
    """Export a task as JSON to disk, independently of Repository.save."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(task, file)


def process(tasks: list[dict]) -> str:
    """Serialize tasks for export, independently of services.tasks.process."""
    return json.dumps(tasks, sort_keys=True)
