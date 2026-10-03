from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from dataclasses import dataclass

import psycopg

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0


@dataclass
class ParsedRow:
    kind: str
    qualname: str
    name: str
    parent_qualname: str | None
    signature: str | None
    decorators: list
    docstring: str | None
    start_line: int
    end_line: int
    raw_code: str
    code_hash: str
    token_est: int | None
    call_sites: list
    extra: dict
    is_async: bool


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


# -----------------------------------------------------------------------------
# Migrations
# -----------------------------------------------------------------------------


def run_migrations(
    conn: psycopg.Connection,
    embed_dim: int,
    migrations_dir: str,
) -> None:
    """Run database migrations from the migrations directory.

    Reads all .sql files sorted by filename, applies them in order.
    For files containing :EMBED_DIM placeholder, replaces with actual embed_dim.
    """
    migrations_path = pathlib.Path(migrations_dir)
    if not migrations_path.exists():
        raise FileNotFoundError(f"Migrations directory not found: {migrations_dir}")

    # Get all SQL files sorted by name
    sql_files = sorted(migrations_path.glob("*.sql"))

    for sql_file in sql_files:
        content = sql_file.read_text(encoding="utf-8")

        # Replace :EMBED_DIM placeholder for embedding migrations
        if ":EMBED_DIM" in content:
            content = content.replace(":EMBED_DIM", str(embed_dim))

        # Execute the migration
        conn.execute(content)

    # Insert schema_version if not present
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', '1') "
        "ON CONFLICT (key) DO NOTHING"
    )


# -----------------------------------------------------------------------------
# Meta
# -----------------------------------------------------------------------------


def get_meta(conn: psycopg.Connection, key: str) -> str | None:
    """Get a meta value by key."""
    row = conn.execute(
        "SELECT value FROM meta WHERE key = %s",
        (key,),
    ).fetchone()
    return row[0] if row else None


def set_meta(conn: psycopg.Connection, key: str, value: str) -> None:
    """Set a meta value."""
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )


def bump_epoch(conn: psycopg.Connection) -> None:
    """Increment the epoch counter in meta table."""
    # Get current epoch
    current = get_meta(conn, "epoch")
    if current is None:
        new_epoch = 1
    else:
        new_epoch = int(current) + 1
    set_meta(conn, "epoch", str(new_epoch))


def index_version(conn: psycopg.Connection) -> str:
    """Return index version hash from meta, or 'v0' if missing."""
    gen_model = get_meta(conn, "gen_model_tag")
    embed_model = get_meta(conn, "embed_model_tag")
    prompt_version = get_meta(conn, "prompt_version")

    if gen_model is None and embed_model is None and prompt_version is None:
        return "v0"

    # Generate hash from the components
    parts = [gen_model or "", embed_model or "", prompt_version or ""]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------


def upsert_file(
    conn: psycopg.Connection,
    path: str,
    sha256: str,
    language: str | None,
    size_bytes: int,
    token_est: int | None,
) -> bool:
    """Upsert a file record. Returns True if inserted or sha256 changed."""
    # Check existing sha256
    existing = conn.execute(
        "SELECT sha256 FROM files WHERE path = %s",
        (path,),
    ).fetchone()

    if existing is None:
        # New insert
        conn.execute(
            "INSERT INTO files (path, sha256, language, size_bytes, token_est) "
            "VALUES (%s, %s, %s, %s, %s)",
            (path, sha256, language, size_bytes, token_est),
        )
        return True

    if existing[0] != sha256:
        # sha256 changed, update
        conn.execute(
            "UPDATE files SET sha256 = %s, language = %s, size_bytes = %s, token_est = %s, parse_status = 'pending' "
            "WHERE path = %s",
            (sha256, language, size_bytes, token_est, path),
        )
        return True

    # Unchanged
    return False


def delete_missing_files(conn: psycopg.Connection, present_paths: set[str]) -> int:
    """Delete files not in present_paths. Returns rowcount."""
    result = conn.execute(
        "DELETE FROM files WHERE path != ALL(%s)",
        (list(present_paths),),
    )
    return result.rowcount


def mark_parsed(
    conn: psycopg.Connection,
    path: str,
    status: str,
    error: str | None,
    imports: list,
    exports: list,
) -> None:
    """Mark a file as parsed with import/export info."""
    conn.execute(
        "UPDATE files SET parsed_at = now(), parse_status = %s, parse_error = %s, "
        "imports = %s, exports = %s WHERE path = %s",
        (status, error, json.dumps(imports), json.dumps(exports), path),
    )


def files_needing_parse(conn: psycopg.Connection) -> list[dict]:
    """Return files that need parsing."""
    rows = conn.execute(
        "SELECT path, sha256, language, size_bytes FROM files "
        "WHERE parse_status IS NULL OR parse_status = 'pending' ORDER BY path"
    ).fetchall()
    return [
        {"path": r[0], "sha256": r[1], "language": r[2], "size_bytes": r[3]}
        for r in rows
    ]


def list_files(conn: psycopg.Connection) -> list[dict]:
    """List all files."""
    rows = conn.execute("SELECT * FROM files ORDER BY path").fetchall()
    columns = ["path", "sha256", "language", "size_bytes", "token_est", "parsed_at",
               "summary", "summary_ctx_hash", "parse_status", "parse_error",
               "imports", "exports", "embed_hash"]
    return [dict(zip(columns, row)) for row in rows]


# -----------------------------------------------------------------------------
# Symbols
# -----------------------------------------------------------------------------


def sync_symbols(
    conn: psycopg.Connection,
    file_path: str,
    parsed_rows: list[ParsedRow],
) -> SyncResult:
    """Synchronize symbols for a file with parsed data."""
    # Get existing symbols for this file
    existing = conn.execute(
        "SELECT id, code_hash FROM symbols WHERE file_path = %s",
        (file_path,),
    ).fetchall()
    existing_map = {row[0]: row[1] for row in existing}

    # Build new symbol ids and track duplicates for disambiguation
    qualname_counts: dict[str, int] = {}
    for row in parsed_rows:
        qualname_counts[row.qualname] = qualname_counts.get(row.qualname, 0) + 1

    new_ids: dict[int, str] = {}  # index -> symbol_id
    ambiguous: dict[int, bool] = {}
    for i, row in enumerate(parsed_rows):
        is_ambiguous = qualname_counts[row.qualname] > 1
        ambiguous[i] = is_ambiguous
        base_id = f"{file_path}::{row.qualname}"
        if is_ambiguous:
            new_ids[i] = f"{base_id}@{row.start_line}"
        else:
            new_ids[i] = base_id

    result = SyncResult()

    with conn.transaction():
        # Process each parsed row
        for i, row in enumerate(parsed_rows):
            symbol_id = new_ids[i]
            is_ambiguous = ambiguous[i]

            # Determine parent_id
            parent_id = None
            if row.parent_qualname is not None:
                parent_qualname_counts = qualname_counts.get(row.parent_qualname, 0)
                if parent_qualname_counts > 1:
                    # Find parent in parsed_rows to get its line number
                    for j, prow in enumerate(parsed_rows):
                        if prow.qualname == row.parent_qualname:
                            parent_id = f"{file_path}::{row.parent_qualname}@{prow.start_line}"
                            break
                else:
                    parent_id = f"{file_path}::{row.parent_qualname}"

            existing_hash = existing_map.get(symbol_id)

            if existing_hash is None:
                # Insert new symbol
                conn.execute(
                    """INSERT INTO symbols (id, file_path, kind, qualname, name, parent_id,
                       signature, decorators, docstring, start_line, end_line, raw_code,
                       code_hash, token_est, call_sites, extra, is_async, status, attempts)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', 0)""",
                    (symbol_id, file_path, row.kind, row.qualname, row.name, parent_id,
                     row.signature, json.dumps(row.decorators), row.docstring,
                     row.start_line, row.end_line, row.raw_code, row.code_hash,
                     row.token_est, json.dumps(row.call_sites), json.dumps(row.extra),
                     row.is_async),
                )
                result.inserted += 1
            elif existing_hash != row.code_hash:
                # Update existing symbol
                conn.execute(
                    """UPDATE symbols SET kind = %s, qualname = %s, name = %s, parent_id = %s,
                       signature = %s, decorators = %s, docstring = %s, start_line = %s,
                       end_line = %s, raw_code = %s, code_hash = %s, token_est = %s,
                       call_sites = %s, extra = %s, is_async = %s
                       WHERE id = %s""",
                    (row.kind, row.qualname, row.name, parent_id, row.signature,
                     json.dumps(row.decorators), row.docstring, row.start_line,
                     row.end_line, row.raw_code, row.code_hash, row.token_est,
                     json.dumps(row.call_sites), json.dumps(row.extra), row.is_async,
                     symbol_id),
                )
                result.updated += 1
            else:
                # Unchanged, just update basic fields
                conn.execute(
                    "UPDATE symbols SET start_line = %s, end_line = %s, token_est = %s, "
                    "raw_code = %s, call_sites = %s, extra = %s WHERE id = %s",
                    (row.start_line, row.end_line, row.token_est, row.raw_code,
                     json.dumps(row.call_sites), json.dumps(row.extra), symbol_id),
                )
                result.unchanged += 1

        # Delete symbols that are no longer present
        new_symbol_ids = set(new_ids.values())
        to_delete = set(existing_map.keys()) - new_symbol_ids
        if to_delete:
            conn.execute(
                "DELETE FROM symbols WHERE id = ANY(%s)",
                (list(to_delete),),
            )
            result.deleted = len(to_delete)

    return result


# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------


def replace_edges(
    conn: psycopg.Connection,
    source: str,
    rows: list[dict],
) -> None:
    """Replace edges for given source with new rows."""
    # Each invocation supplies the complete edge set for this source.
    conn.execute("DELETE FROM edges WHERE source = %s", (source,))

    # Insert new edges
    for row in rows:
        conn.execute(
            """INSERT INTO edges (caller_id, callee_id, callee_expr, line, kind, resolution, source, confidence, control_ctx)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT DO NOTHING""",
            (row.get("caller_id"), row.get("callee_id"), row.get("callee_expr"),
             row.get("line"), row.get("kind"), row.get("resolution"), source,
             row.get("confidence"), row.get("control_ctx")),
        )


def fetch_edges(conn: psycopg.Connection, min_conf: float) -> list[dict]:
    """Fetch edges above confidence threshold."""
    rows = conn.execute(
        """SELECT caller_id, callee_id, callee_expr, line, kind, resolution, source, confidence, control_ctx
           FROM edges WHERE confidence >= %s""",
        (min_conf,),
    ).fetchall()
    columns = ["caller_id", "callee_id", "callee_expr", "line", "kind", "resolution",
               "source", "confidence", "control_ctx"]
    return [dict(zip(columns, row)) for row in rows]


def set_graph_metrics(conn: psycopg.Connection, rows: list[dict]) -> None:
    """Set pagerank, scc_id, layer for symbols."""
    for row in rows:
        conn.execute(
            "UPDATE symbols SET pagerank = %s, scc_id = %s, layer = %s WHERE id = %s",
            (row.get("pagerank"), row.get("scc_id"), row.get("layer"), row.get("id")),
        )


# -----------------------------------------------------------------------------
# Traversal
# -----------------------------------------------------------------------------


def callers_rows(
    conn: psycopg.Connection,
    id: str,
    depth: int,
    min_conf: float,
) -> list[dict]:
    """Find all callers of a symbol up to given depth."""
    # Recursive CTE to find all callers
    cte_query = """WITH RECURSIVE cte AS (
        SELECT caller_id, callee_id, 1 as d FROM edges
        WHERE callee_id = %s AND confidence >= %s
        UNION ALL
        SELECT e.caller_id, e.callee_id, cte.d + 1 FROM edges e
        JOIN cte ON e.callee_id = cte.caller_id
        WHERE cte.d < %s AND e.confidence >= %s
    )
    SELECT DISTINCT caller_id FROM cte"""

    caller_ids = conn.execute(cte_query, (id, min_conf, depth, min_conf)).fetchall()
    if not caller_ids:
        return []

    ids = [row[0] for row in caller_ids]
    symbol_rows = conn.execute(
        "SELECT * FROM symbols WHERE id = ANY(%s)",
        (ids,),
    ).fetchall()

    columns = ["id", "file_path", "kind", "qualname", "name", "parent_id", "signature",
               "decorators", "docstring", "start_line", "end_line", "raw_code", "code_hash",
               "token_est", "pagerank", "summary_short", "summary_json", "summary_long",
               "ctx_hash", "status", "attempts", "error", "search_text", "search", "embed_hash",
               "call_sites", "extra", "is_async", "is_entrypoint", "entry_kind", "scc_id", "layer"]
    return [dict(zip(columns, row)) for row in symbol_rows]


def callees_rows(
    conn: psycopg.Connection,
    id: str,
    depth: int,
    min_conf: float,
) -> list[dict]:
    """Find all callees of a symbol up to given depth."""
    cte_query = """WITH RECURSIVE cte AS (
        SELECT callee_id, 1 as d FROM edges
        WHERE caller_id = %s AND confidence >= %s AND callee_id IS NOT NULL
        UNION ALL
        SELECT e.callee_id, cte.d + 1 FROM edges e
        JOIN cte ON e.caller_id = cte.callee_id
        WHERE cte.d < %s AND e.confidence >= %s AND e.callee_id IS NOT NULL
    )
    SELECT DISTINCT callee_id FROM cte"""

    callee_ids = conn.execute(cte_query, (id, min_conf, depth, min_conf)).fetchall()
    if not callee_ids:
        return []

    ids = [row[0] for row in callee_ids if row[0] is not None]
    if not ids:
        return []

    symbol_rows = conn.execute(
        "SELECT * FROM symbols WHERE id = ANY(%s)",
        (ids,),
    ).fetchall()

    columns = ["id", "file_path", "kind", "qualname", "name", "parent_id", "signature",
               "decorators", "docstring", "start_line", "end_line", "raw_code", "code_hash",
               "token_est", "pagerank", "summary_short", "summary_json", "summary_long",
               "ctx_hash", "status", "attempts", "error", "search_text", "search", "embed_hash",
               "call_sites", "extra", "is_async", "is_entrypoint", "entry_kind", "scc_id", "layer"]
    return [dict(zip(columns, row)) for row in symbol_rows]


def reach_rows(
    conn: psycopg.Connection,
    seed: str,
    direction: str,
    depth: int,
    min_conf: float,
) -> list[dict]:
    """Find reachable symbols in given direction."""
    if direction == "callers":
        return callers_rows(conn, seed, depth, min_conf)
    elif direction == "callees":
        return callees_rows(conn, seed, depth, min_conf)
    return []


def top_pagerank(conn: psycopg.Connection, n: int) -> list[dict]:
    """Get top N symbols by pagerank."""
    rows = conn.execute(
        """SELECT id, qualname, file_path, pagerank, summary_short FROM symbols
           WHERE pagerank IS NOT NULL ORDER BY pagerank DESC LIMIT %s""",
        (n,),
    ).fetchall()
    return [{"id": r[0], "qualname": r[1], "file_path": r[2],
             "pagerank": r[3], "summary_short": r[4]} for r in rows]


def dead_candidates(
    conn: psycopg.Connection,
    min_conf: float,
    include_tests: bool,
) -> list[dict]:
    """Find symbols that are never called (dead code candidates)."""
    query = """SELECT s.id, s.qualname, s.file_path, s.kind FROM symbols s
               WHERE s.is_entrypoint = FALSE AND NOT EXISTS (
                   SELECT 1 FROM edges e WHERE e.callee_id = s.id AND e.confidence >= %s
               )"""
    if not include_tests:
        query += """ AND s.file_path NOT LIKE 'tests/%' AND s.file_path NOT LIKE '%/test_%'
                     AND s.file_path NOT LIKE '%_test.py'"""

    rows = conn.execute(query, (min_conf,)).fetchall()
    return [{"id": r[0], "qualname": r[1], "file_path": r[2], "kind": r[3]} for r in rows]


def lookup_symbols(
    conn: psycopg.Connection,
    query: str,
    limit: int,
) -> list[dict]:
    """Search symbols by qualname or name."""
    pattern = f"%{query}%"
    rows = conn.execute(
        """SELECT id, qualname, name, file_path, kind, summary_short FROM symbols
           WHERE qualname ILIKE %s OR name ILIKE %s LIMIT %s""",
        (pattern, pattern, limit),
    ).fetchall()
    return [{"id": r[0], "qualname": r[1], "name": r[2],
             "file_path": r[3], "kind": r[4], "summary_short": r[5]} for r in rows]


def symbols_in_file(conn: psycopg.Connection, path: str) -> list[dict]:
    """Get all symbols in a file."""
    rows = conn.execute(
        """SELECT id, kind, qualname, name, start_line, end_line, summary_short
           FROM symbols WHERE file_path = %s ORDER BY start_line""",
        (path,),
    ).fetchall()
    return [{"id": r[0], "kind": r[1], "qualname": r[2], "name": r[3],
             "start_line": r[4], "end_line": r[5], "summary_short": r[6]} for r in rows]


def get_symbol(conn: psycopg.Connection, id: str) -> dict | None:
    """Get a single symbol by id."""
    row = conn.execute("SELECT * FROM symbols WHERE id = %s", (id,)).fetchone()
    if row is None:
        return None
    columns = ["id", "file_path", "kind", "qualname", "name", "parent_id", "signature",
               "decorators", "docstring", "start_line", "end_line", "raw_code", "code_hash",
               "token_est", "pagerank", "summary_short", "summary_json", "summary_long",
               "ctx_hash", "status", "attempts", "error", "search_text", "search", "embed_hash",
               "call_sites", "extra", "is_async", "is_entrypoint", "entry_kind", "scc_id", "layer"]
    return dict(zip(columns, row))


def get_symbols(conn: psycopg.Connection, ids: list[str]) -> list[dict]:
    """Get multiple symbols by ids."""
    rows = conn.execute(
        "SELECT * FROM symbols WHERE id = ANY(%s)",
        (ids,),
    ).fetchall()
    columns = ["id", "file_path", "kind", "qualname", "name", "parent_id", "signature",
               "decorators", "docstring", "start_line", "end_line", "raw_code", "code_hash",
               "token_est", "pagerank", "summary_short", "summary_json", "summary_long",
               "ctx_hash", "status", "attempts", "error", "search_text", "search", "embed_hash",
               "call_sites", "extra", "is_async", "is_entrypoint", "entry_kind", "scc_id", "layer"]
    return [dict(zip(columns, row)) for row in rows]


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def save_symbol_summary(
    conn: psycopg.Connection,
    id: str,
    summary_json: dict,
    summary_short: str,
    summary_long: str | None,
    ctx_hash: str,
) -> None:
    """Save a completed symbol summary."""
    conn.execute(
        """UPDATE symbols SET summary_json = %s, summary_short = %s, summary_long = %s,
           ctx_hash = %s, status = 'done', attempts = attempts + 1, error = NULL
           WHERE id = %s""",
        (json.dumps(summary_json), summary_short, summary_long, ctx_hash, id),
    )


def save_trivial_summary(conn: psycopg.Connection, id: str, summary_short: str) -> None:
    """Mark a symbol as having a trivial (skipped) summary."""
    conn.execute(
        "UPDATE symbols SET summary_short = %s, status = 'skipped_trivial' WHERE id = %s",
        (summary_short, id),
    )


def quarantine_symbol(
    conn: psycopg.Connection,
    id: str,
    raw_output: str,
    error: str,
) -> None:
    """Mark a symbol as failed with error details."""
    error_msg = f"{error}\n---\nRAW---\n{raw_output[:2000]}"
    conn.execute(
        "UPDATE symbols SET status = 'failed', error = %s, attempts = attempts + 1 WHERE id = %s",
        (error_msg, id),
    )


def set_summary_long(conn: psycopg.Connection, id: str, text: str) -> None:
    """Set the long summary for a symbol."""
    conn.execute(
        "UPDATE symbols SET summary_long = %s WHERE id = %s",
        (text, id),
    )


def status_counts(conn: psycopg.Connection) -> dict:
    """Get counts of symbols by status."""
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM symbols GROUP BY status"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def ctx_inputs(conn: psycopg.Connection) -> list[dict]:
    """Get symbols pending or failed that need context for summarization."""
    rows = conn.execute(
        """SELECT id, code_hash, ctx_hash FROM symbols
           WHERE status IN ('pending', 'failed') ORDER BY layer ASC NULLS LAST, id"""
    ).fetchall()
    return [{"id": r[0], "code_hash": r[1], "ctx_hash": r[2]} for r in rows]


# -----------------------------------------------------------------------------
# Search / Vectors
# -----------------------------------------------------------------------------


def set_search_text(conn: psycopg.Connection, rows: list[dict]) -> None:
    """Update search_text for multiple symbols."""
    for row in rows:
        conn.execute(
            "UPDATE symbols SET search_text = %s WHERE id = %s",
            (row.get("search_text"), row.get("id")),
        )


def lexical_search(
    conn: psycopg.Connection,
    terms: list[str],
    limit: int,
) -> list[dict]:
    """Full-text search using PostgreSQL tsvector."""
    tsquery = " & ".join(terms)
    rows = conn.execute(
        """SELECT id, qualname, file_path, summary_short,
           ts_rank(search, to_tsquery('simple', %s)) as rank
           FROM symbols WHERE search @@ to_tsquery('simple', %s)
           ORDER BY rank DESC LIMIT %s""",
        (tsquery, tsquery, limit),
    ).fetchall()
    return [{"id": r[0], "qualname": r[1], "file_path": r[2],
             "summary_short": r[3], "rank": r[4]} for r in rows]


def trigram_search(
    conn: psycopg.Connection,
    q: str,
    limit: int,
) -> list[dict]:
    """Search using PostgreSQL trigram similarity."""
    rows = conn.execute(
        """SELECT id, qualname, file_path, summary_short,
           similarity(qualname, %s) as sim
           FROM symbols WHERE qualname %% %s ORDER BY sim DESC LIMIT %s""",
        (q, q, limit),
    ).fetchall()
    return [{"id": r[0], "qualname": r[1], "file_path": r[2],
             "summary_short": r[3], "sim": r[4]} for r in rows]


def existing_embedding_hashes(conn: psycopg.Connection, hashes: list[str]) -> set[str]:
    """Check which embedding hashes already exist in DB."""
    rows = conn.execute(
        "SELECT hash FROM embeddings WHERE hash = ANY(%s)",
        (hashes,),
    ).fetchall()
    return {row[0] for row in rows}


def upsert_embeddings(conn: psycopg.Connection, rows: list[dict]) -> None:
    """Insert or update embeddings."""
    for row in rows:
        conn.execute(
            "INSERT INTO embeddings (hash, vec) VALUES (%s, %s) ON CONFLICT(hash) DO NOTHING",
            (row.get("hash"), row.get("vec")),
        )


def dense_search(
    conn: psycopg.Connection,
    vec: list[float],
    limit: int,
    source: str,
) -> list[dict]:
    """Vector similarity search."""
    rows = conn.execute(
        """SELECT s.id, s.qualname, s.file_path, s.summary_short,
           e.vec <-> %s::vector as dist
           FROM symbols s JOIN embeddings e ON e.hash = s.embed_hash
           WHERE s.embed_hash IS NOT NULL ORDER BY dist LIMIT %s""",
        (vec, limit),
    ).fetchall()
    return [{"id": r[0], "qualname": r[1], "file_path": r[2],
             "summary_short": r[3], "dist": r[4]} for r in rows]


def set_view_status(
    conn: psycopg.Connection,
    view: str,
    status: str,
    config: dict | None,
) -> None:
    """Update or insert view status."""
    conn.execute(
        """INSERT INTO view_status (view, status, built_at, config)
           VALUES (%s, %s, now(), %s)
           ON CONFLICT(view) DO UPDATE SET status = EXCLUDED.status,
           built_at = EXCLUDED.built_at, config = EXCLUDED.config""",
        (view, status, json.dumps(config) if config else None),
    )


def get_view_status(conn: psycopg.Connection) -> dict:
    """Get all view statuses."""
    rows = conn.execute(
        "SELECT view, status, built_at, config FROM view_status"
    ).fetchall()
    return {row[0]: {"status": row[1], "built_at": row[2], "config": row[3]} for row in rows}


# -----------------------------------------------------------------------------
# Modules / Features
# -----------------------------------------------------------------------------


def load_module_tree(conn: psycopg.Connection) -> list[dict]:
    """Load the entire module tree."""
    rows = conn.execute("SELECT * FROM module_tree ORDER BY id").fetchall()
    columns = ["id", "parent_id", "name", "dir_hint", "member_hash", "summary", "aggregate_hash"]
    return [dict(zip(columns, row)) for row in rows]


def replace_module_subtree(
    conn: psycopg.Connection,
    module_id: int | None,
    parent_id: int | None,
    name: str,
    dir_hint: str | None,
    member_hash: str,
    summary: str | None,
) -> int:
    """Insert or update a module subtree."""
    if module_id is not None:
        conn.execute(
            "UPDATE module_tree SET name = %s, dir_hint = %s, member_hash = %s, summary = %s WHERE id = %s",
            (name, dir_hint, member_hash, summary, module_id),
        )
        return module_id
    else:
        row = conn.execute(
            """INSERT INTO module_tree (parent_id, name, dir_hint, member_hash, summary)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (parent_id, name, dir_hint, member_hash, summary),
        ).fetchone()
        return row[0]


def save_module_summary(
    conn: psycopg.Connection,
    id: int,
    summary: str,
    aggregate_hash: str,
) -> None:
    """Save summary for a module."""
    conn.execute(
        "UPDATE module_tree SET summary = %s, aggregate_hash = %s WHERE id = %s",
        (summary, aggregate_hash, id),
    )


def save_features(conn: psycopg.Connection, features: list[dict]) -> None:
    """Save features and their members."""
    if not features:
        return

    feature_names = [f["name"] for f in features]

    with conn.transaction():
        # Delete existing features
        conn.execute(
            "DELETE FROM features WHERE name = ANY(%s)",
            (feature_names,),
        )

        # Insert features
        for feature in features:
            row = conn.execute(
                """INSERT INTO features (name, entry_symbol_id, summary, aggregate_hash)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (feature["name"], feature.get("entry_symbol_id"),
                 feature.get("summary"), feature.get("aggregate_hash")),
            ).fetchone()
            feature_id = row[0]

            # Insert feature members
            for member in feature.get("members", []):
                conn.execute(
                    "INSERT INTO feature_members (feature_id, symbol_id, role) VALUES (%s, %s, %s)",
                    (feature_id, member.get("symbol_id"), member.get("role")),
                )


def feature_members(conn: psycopg.Connection, feature_id: int) -> list[dict]:
    """Get members of a feature."""
    rows = conn.execute(
        """SELECT fm.symbol_id, fm.role, s.qualname, s.file_path, s.summary_short
           FROM feature_members fm JOIN symbols s ON s.id = fm.symbol_id
           WHERE fm.feature_id = %s""",
        (feature_id,),
    ).fetchall()
    return [{"symbol_id": r[0], "role": r[1], "qualname": r[2],
             "file_path": r[3], "summary_short": r[4]} for r in rows]


# -----------------------------------------------------------------------------
# Cache / Eval
# -----------------------------------------------------------------------------


def get_cached_answer(
    conn: psycopg.Connection,
    query_hash: str,
    index_version: str,
) -> str | None:
    """Get cached answer for query."""
    row = conn.execute(
        "SELECT answer FROM answer_cache WHERE query_hash = %s AND index_version = %s",
        (query_hash, index_version),
    ).fetchone()
    return row[0] if row else None


def put_cached_answer(
    conn: psycopg.Connection,
    query_hash: str,
    mode: str,
    answer: str,
    index_version: str,
) -> None:
    """Cache an answer."""
    conn.execute(
        """INSERT INTO answer_cache (query_hash, mode, answer, index_version)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT(query_hash) DO UPDATE SET mode = EXCLUDED.mode,
           answer = EXCLUDED.answer, index_version = EXCLUDED.index_version""",
        (query_hash, mode, answer, index_version),
    )


def save_eval_metrics(
    conn: psycopg.Connection,
    run_id: str,
    question_id: str,
    metrics: dict,
    config: dict,
) -> None:
    """Save evaluation metrics for a run."""
    conn.execute(
        """INSERT INTO eval_runs (run_id, question_id, metrics, config)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT(run_id, question_id) DO UPDATE SET metrics = EXCLUDED.metrics,
           config = EXCLUDED.config""",
        (run_id, question_id, json.dumps(metrics), json.dumps(config)),
    )


def get_all_symbols(conn: psycopg.Connection) -> list[dict]:
    """Load symbols through the central DB access layer."""
    ids = [row[0] for row in conn.execute("SELECT id FROM symbols ORDER BY id").fetchall()]
    return get_symbols(conn, ids)


def set_entrypoints(conn: psycopg.Connection, rows: list[dict]) -> None:
    """Replace detected entrypoints, clearing flags that no longer apply."""
    conn.execute("UPDATE symbols SET is_entrypoint = FALSE, entry_kind = NULL")
    for row in rows:
        conn.execute(
            "UPDATE symbols SET is_entrypoint = TRUE, entry_kind = %s WHERE id = %s",
            (row["entry_kind"], row["id"]),
        )
