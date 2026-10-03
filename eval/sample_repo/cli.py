from __future__ import annotations

from db.connection import Connection
from db.repository import Repository
from pipeline import run_pipeline
from utils.config import parse_config


def main(config_text: str, records: list[dict]) -> list[dict]:
    """Validate configuration and import task records through the pipeline."""
    parse_config(config_text)
    connection = Connection()
    repository = Repository(connection)
    return run_pipeline(repository, records)
