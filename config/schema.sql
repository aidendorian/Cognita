CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chats (
    id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    chat_id INT  REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    project_id INT  REFERENCES projects(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    source TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
    ON embeddings USING hnsw (embedding vector_cosine_ops);

ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS chunk_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;

CREATE INDEX IF NOT EXISTS embeddings_fts_idx
    ON embeddings USING gin (chunk_tsv);

CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    project_id INT  REFERENCES projects(id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    content TEXT NOT NULL,
    raw_length INT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE,

    project_id INTEGER NOT NULL
        REFERENCES projects(id)
        ON DELETE CASCADE,

    task TEXT NOT NULL,

    current_agent TEXT NOT NULL DEFAULT 'initializing',
    config JSONB NOT NULL DEFAULT '{}'::jsonb,

    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (
            status IN (
                'pending',
                'running',
                'completed',
                'failed',
                'cancelled',
                'interrupted',
                'needs_review'
            )
        ),

    final_output TEXT,

    error TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_runs_project_created
ON runs(project_id, created_at DESC);

CREATE INDEX idx_runs_status_created
ON runs(status, created_at ASC)
WHERE status IN ('pending', 'running');