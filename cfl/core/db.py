from __future__ import annotations

import logging

import psycopg

logger = logging.getLogger(__name__)


def connect(
    dsn: str,
    *,
    autocommit: bool = True,
    statement_timeout_ms: int | None = None,
) -> psycopg.Connection:
    """Open a psycopg 3 connection and configure it for CFL use.

    - Forces client_encoding=UTF8.
    - Optionally sets statement_timeout.
    - Tries to register the pgvector codec; silently skips if the extension
      is not installed yet.
    """
    conn = psycopg.connect(dsn, options="-c client_encoding=UTF8")
    conn.autocommit = autocommit

    if statement_timeout_ms is not None:
        conn.execute(f"SET statement_timeout = {statement_timeout_ms}")

    # Register pgvector type codec only if the extension exists.
    try:
        row = conn.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1"
        ).fetchone()
        if row:
            from pgvector.psycopg import register_vector  # type: ignore[import]

            register_vector(conn)
    except Exception as exc:  # noqa: BLE001
        logger.debug("pgvector registration skipped: %s", exc)

    return conn
