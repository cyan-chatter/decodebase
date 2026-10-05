-- Draft knowledge is never eligible for RAG until source-only review succeeds.
CREATE TABLE IF NOT EXISTS knowledge_drafts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('symbol','file','feature')),
    title TEXT NOT NULL,
    draft TEXT NOT NULL,
    draft_hash TEXT NOT NULL,
    evidence JSONB NOT NULL,
    generator_digest TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','complete','partial','rejected')),
    published TEXT,
    review JSONB,
    verifier_digest TEXT,
    guard_version TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE knowledge_drafts ADD COLUMN IF NOT EXISTS embed_hash TEXT;
