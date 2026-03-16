-- =============================================================================
-- Migration 047: pgvector extension + GDELT embeddings table
-- =============================================================================
-- Adds vector similarity search for news RAG (Retrieval Augmented Generation).
-- Used by NewsIntelligenceEngine.search_historical_context()
--
-- Prerequisites:
--   - PostgreSQL 15+ with pgvector extension available
--   - Run: CREATE EXTENSION IF NOT EXISTS vector;
-- =============================================================================

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Embeddings table for GDELT articles (384-dim from MiniLM-L12-v2)
CREATE TABLE IF NOT EXISTS gdelt_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    url_hash        VARCHAR(64) NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    url             TEXT,
    source          VARCHAR(100),
    fecha           DATE NOT NULL,
    language        VARCHAR(10) DEFAULT 'en',
    category        VARCHAR(50),
    bias_label      VARCHAR(30),
    tone            FLOAT,
    relevance_score FLOAT,
    embedding       vector(384) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Metadata indexes
CREATE INDEX IF NOT EXISTS idx_gdelt_emb_fecha
    ON gdelt_embeddings (fecha);

CREATE INDEX IF NOT EXISTS idx_gdelt_emb_category
    ON gdelt_embeddings (category);

CREATE INDEX IF NOT EXISTS idx_gdelt_emb_language
    ON gdelt_embeddings (language);

CREATE INDEX IF NOT EXISTS idx_gdelt_emb_source
    ON gdelt_embeddings (source);

-- Vector similarity index (IVFFlat for approximate nearest neighbor)
-- Requires ~1000+ rows for the index to be effective
-- Will be created after initial data load
CREATE INDEX IF NOT EXISTS idx_gdelt_emb_vector
    ON gdelt_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_gdelt_emb_fecha_cat
    ON gdelt_embeddings (fecha, category);

COMMENT ON TABLE gdelt_embeddings IS
    'Sentence-transformer embeddings (384d) for GDELT articles. '
    'Used by NewsIntelligenceEngine for semantic search via pgvector.';
