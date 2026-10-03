-- Embeddings table for vector search
-- Dimension will be substituted by run_migrations()
CREATE TABLE IF NOT EXISTS embeddings (
    hash TEXT PRIMARY KEY,
    vec vector(:EMBED_DIM) NOT NULL
);