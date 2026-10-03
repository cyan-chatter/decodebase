from __future__ import annotations

from auth.tokens import validate_token as authenticate
from db.repository import Repository
from services.tasks import complete_task, create_task


def route(path: str):
    """Attach a route path to an HTTP handler."""

    def decorate(handler):
        handler.route_path = path
        return handler

    return decorate


@route("/tasks/create")
def create_handler(token: str, secret: str, repo: Repository, title: str) -> dict:
    """Authenticate the request before creating a task."""
    authenticate(token, secret)
    return create_task(repo, title)


@route("/tasks/complete")
def complete_handler(token: str, secret: str, repo: Repository, task_id: str) -> dict:
    """Authenticate the request before completing a task."""
    authenticate(token, secret)
    return complete_task(repo, task_id)
