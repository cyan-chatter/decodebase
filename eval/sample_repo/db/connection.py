from __future__ import annotations


class Connection:
    """An in-memory taskboard connection with explicit close behavior."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.closed = False

    def execute(self, key: str, value: dict | None = None) -> dict | None:
        """Read or write a row; reject operations after close."""
        if self.closed:
            raise RuntimeError("Connection is closed")
        if value is not None:
            self.rows[key] = value
        return self.rows.get(key)

    def close(self) -> None:
        """Close the connection."""
        self.closed = True
