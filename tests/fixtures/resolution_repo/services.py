from __future__ import annotations
from models import Repository


def create_task(repo: Repository) -> None:
    repo.save()
    r = Repository(None)
    r.save()
