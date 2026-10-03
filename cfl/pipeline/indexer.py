from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cfl.core import db

if TYPE_CHECKING:
    from psycopg import Connection

LEXICAL_VERSION = "2"
_WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+|[^\W\d_]+", re.UNICODE)

_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "does",
        "do",
        "how",
        "why",
        "where",
        "who",
        "which",
        "when",
        "can",
        "could",
        "would",
        "should",
        "to",
        "of",
        "for",
        "from",
        "with",
        "and",
        "or",
        "in",
        "on",
        "by",
        "it",
        "this",
        "that",
        "these",
        "those",
        "as",
    }
)


def split_identifiers(text: str) -> list[str]:
    """Split snake_case, CamelCase, acronym transitions and digits into safe lexemes."""
    return list(dict.fromkeys(word.lower() for word in _WORDS.findall(text)))


def build_search_text(symbol: dict) -> str:
    fields = [
        symbol.get(key) or ""
        for key in ("file_path", "qualname", "name", "signature", "docstring", "summary_short")
    ]
    fields.extend(symbol.get("decorators") or [])
    text = "\n".join(field for field in fields if field)
    return (text + "\n" + " ".join(split_identifiers(text))).lower()


def build_lexical(conn: Connection, *, batch_size: int = 256) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    count = 0
    with conn.transaction():
        for batch in db.lexical_symbol_batches(conn, batch_size):
            db.set_search_text(
                conn, [{"id": row["id"], "search_text": build_search_text(row)} for row in batch]
            )
            count += len(batch)
        db.set_view_status(conn, "lexical", "fresh", {"version": LEXICAL_VERSION})
    return count


def retrieve_lexical(conn: Connection, query: str, limit: int = 5) -> list[dict]:
    """Fuse FTS and qualname/name trigram ranks, with stable ID tie-breaking."""
    if limit < 1:
        raise ValueError("limit must be positive")
    terms = [term for term in split_identifiers(query) if term not in _QUERY_STOP_WORDS]
    if not terms:
        return []
    candidates = max(20, limit * 4)
    ranked = [
        db.lexical_search(conn, terms, candidates),
        db.trigram_search(conn, query, candidates),
    ]
    scores: dict[str, float] = {}
    for result in ranked:
        for rank, row in enumerate(result, 1):
            scores[row["id"]] = scores.get(row["id"], 0) + 1 / rank
    ids = sorted(scores, key=lambda id: (-scores[id], id))[:limit]
    # Hydrate spans from the current index; never use static line numbers in the suite.
    hydrated = {row["id"]: row for row in db.get_symbols(conn, ids)}
    return [{**hydrated[id], "rank": scores[id]} for id in ids if id in hydrated]
