-- Core schema for CodeFlowLens
-- Extensions must be created first
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Meta table for key-value configuration
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Files table (includes spec columns + D2 additions)
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    language TEXT,
    size_bytes INT,
    token_est INT,
    parsed_at TIMESTAMPTZ,
    summary TEXT,
    summary_ctx_hash TEXT,
    parse_status TEXT,
    parse_error TEXT,
    imports JSONB,
    exports JSONB,
    embed_hash TEXT
);

-- Symbols table (includes spec columns + D2 additions)
CREATE TABLE IF NOT EXISTS symbols (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('function', 'method', 'class', 'nested', 'module')),
    qualname TEXT NOT NULL,
    name TEXT NOT NULL,
    parent_id TEXT,
    signature TEXT,
    decorators TEXT,
    docstring TEXT,
    start_line INT NOT NULL,
    end_line INT NOT NULL,
    raw_code TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    token_est INT,
    pagerank DOUBLE PRECISION,
    summary_short TEXT,
    summary_json JSONB,
    summary_long TEXT,
    ctx_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'done', 'failed', 'skipped_trivial')),
    attempts INT NOT NULL DEFAULT 0,
    error TEXT,
    search_text TEXT,
    search TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', coalesce(search_text, ''))) STORED,
    embed_hash TEXT,
    call_sites JSONB,
    extra JSONB,
    is_async BOOLEAN,
    is_entrypoint BOOLEAN DEFAULT FALSE,
    entry_kind TEXT,
    scc_id TEXT,
    layer INT
);

-- Symbols indexes (from spec)
CREATE INDEX IF NOT EXISTS symbols_search_idx ON symbols USING GIN (search);
CREATE INDEX IF NOT EXISTS symbols_qualname_trgm ON symbols USING GIN (qualname gin_trgm_ops);
CREATE INDEX IF NOT EXISTS symbols_status_idx ON symbols (status);
CREATE INDEX IF NOT EXISTS symbols_file_idx ON symbols (file_path);

-- Edges table (with D1: surrogate PK)
CREATE TABLE IF NOT EXISTS edges (
    id BIGSERIAL PRIMARY KEY,
    caller_id TEXT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    callee_id TEXT,
    callee_expr TEXT NOT NULL,
    line INT NOT NULL,
    kind TEXT,
    resolution TEXT,
    source TEXT NOT NULL DEFAULT 'ast' CHECK (source IN ('ast', 'scip', 'dynamic')),
    confidence REAL NOT NULL DEFAULT 0.5,
    control_ctx TEXT
);

-- Edges unique index (uses COALESCE which is not supported in table constraint)
CREATE UNIQUE INDEX IF NOT EXISTS edges_unique ON edges (caller_id, callee_expr, line, source, COALESCE(callee_id, ''));

-- Edges indexes
CREATE INDEX IF NOT EXISTS edges_callee_idx ON edges (callee_id);
CREATE INDEX IF NOT EXISTS edges_caller_idx ON edges (caller_id);

-- Module tree table (from spec)
CREATE TABLE IF NOT EXISTS module_tree (
    id BIGSERIAL PRIMARY KEY,
    parent_id BIGINT REFERENCES module_tree(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    dir_hint TEXT,
    member_hash TEXT NOT NULL,
    summary TEXT,
    aggregate_hash TEXT
);

-- Module members table
CREATE TABLE IF NOT EXISTS module_members (
    module_id BIGINT REFERENCES module_tree(id) ON DELETE CASCADE,
    symbol_id TEXT REFERENCES symbols(id) ON DELETE CASCADE,
    PRIMARY KEY (module_id, symbol_id)
);

-- D14: Module files table
CREATE TABLE IF NOT EXISTS module_files (
    module_id BIGINT REFERENCES module_tree(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    PRIMARY KEY (module_id, file_path)
);

-- Features table
CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    entry_symbol_id TEXT REFERENCES symbols(id) ON DELETE SET NULL,
    summary TEXT,
    aggregate_hash TEXT
);

-- Feature members table
CREATE TABLE IF NOT EXISTS feature_members (
    feature_id BIGINT REFERENCES features(id) ON DELETE CASCADE,
    symbol_id TEXT REFERENCES symbols(id) ON DELETE CASCADE,
    role TEXT,
    PRIMARY KEY (feature_id, symbol_id)
);

-- View status table
CREATE TABLE IF NOT EXISTS view_status (
    view TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    built_at TIMESTAMPTZ,
    config JSONB
);

-- Answer cache table
CREATE TABLE IF NOT EXISTS answer_cache (
    query_hash TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    answer TEXT NOT NULL,
    index_version TEXT NOT NULL
);

-- Eval runs table
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT,
    question_id TEXT,
    metrics JSONB,
    config JSONB,
    PRIMARY KEY (run_id, question_id)
);