-- =============================================================================
-- Migration 045: NewsEngine — Initial Tables (8 tables)
-- =============================================================================
--
-- Creates the NewsEngine storage layer for article ingestion, enrichment,
-- cross-referencing, and feature export.
--
-- Group A tables per SDD-03. No macro_data table — macro comes from existing
-- macro_indicators_daily (L0 DAGs).
--
-- Contract: CTR-NEWS-STORAGE-001
-- Date: 2026-02-25
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- A1: sources — Registry of news sources
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_sources (
    source_id       VARCHAR(50)     PRIMARY KEY,
    source_name     VARCHAR(200)    NOT NULL,
    source_type     VARCHAR(20)     NOT NULL CHECK (source_type IN ('api', 'scraper', 'rss')),
    base_url        TEXT,
    enabled         BOOLEAN         DEFAULT TRUE,
    rate_limit_rpm  INTEGER         DEFAULT 60,
    config_json     JSONB           DEFAULT '{}',
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- Seed default sources
INSERT INTO news_sources (source_id, source_name, source_type, base_url, rate_limit_rpm) VALUES
    ('gdelt_doc',       'GDELT DOC 2.0',       'api',     'https://api.gdeltproject.org/api/v2/doc/doc',     60),
    ('gdelt_context',   'GDELT Context 2.0',   'api',     'https://api.gdeltproject.org/api/v2/context/context', 60),
    ('newsapi',         'NewsAPI.org',          'api',     'https://newsapi.org/v2',                           100),
    ('investing',       'Investing.com',        'rss',     'https://www.investing.com',                        30),
    ('larepublica',     'La Republica',         'scraper', 'https://www.larepublica.co',                       20),
    ('portafolio',      'Portafolio',           'scraper', 'https://www.portafolio.co',                        20)
ON CONFLICT (source_id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- A2: articles — Unified article storage (raw + enriched)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_articles (
    id              BIGSERIAL       PRIMARY KEY,
    source_id       VARCHAR(50)     NOT NULL REFERENCES news_sources(source_id),
    url             TEXT            NOT NULL,
    url_hash        VARCHAR(64)     NOT NULL,   -- SHA256 for dedup
    title           TEXT            NOT NULL,
    content         TEXT,
    summary         TEXT,
    published_at    TIMESTAMPTZ     NOT NULL,
    fetched_at      TIMESTAMPTZ     DEFAULT NOW(),

    -- Enrichment fields (populated by enrichment pipeline)
    category        VARCHAR(50),    -- 'monetary_policy', 'fx_market', 'commodities', etc.
    subcategory     VARCHAR(50),
    relevance_score DOUBLE PRECISION DEFAULT 0.0,
    sentiment_score DOUBLE PRECISION,           -- -1.0 to 1.0
    sentiment_label VARCHAR(20),                -- 'positive', 'negative', 'neutral'
    gdelt_tone      DOUBLE PRECISION,           -- GDELT native tone (-100 to 100)
    keywords        TEXT[],                      -- Extracted keywords array
    entities        TEXT[],                      -- Named entities
    language        VARCHAR(10)     DEFAULT 'es',
    country_focus   VARCHAR(10)     DEFAULT 'CO',
    is_breaking     BOOLEAN         DEFAULT FALSE,
    is_weekly_relevant BOOLEAN      DEFAULT FALSE,

    -- Metadata
    image_url       TEXT,
    author          VARCHAR(200),
    raw_json        JSONB,                      -- Original API response
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- Dedup index: unique URL per source
CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_hash
    ON news_articles(source_id, url_hash);

-- Query indexes
CREATE INDEX IF NOT EXISTS idx_articles_published
    ON news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_category
    ON news_articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_relevance
    ON news_articles(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source_date
    ON news_articles(source_id, published_at DESC);

-- ---------------------------------------------------------------------------
-- A3: keywords — Tracking keywords with priorities
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_keywords (
    id              SERIAL          PRIMARY KEY,
    keyword         VARCHAR(200)    NOT NULL UNIQUE,
    category        VARCHAR(50),
    priority        INTEGER         DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
    language        VARCHAR(10)     DEFAULT 'es',
    is_active       BOOLEAN         DEFAULT TRUE,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- Seed core keywords for USDCOP trading
INSERT INTO news_keywords (keyword, category, priority) VALUES
    ('dolar colombia',      'fx_market',        5),
    ('tasa de cambio',      'fx_market',        5),
    ('USD COP',             'fx_market',        5),
    ('banco de la republica', 'monetary_policy', 5),
    ('tasa de interes',     'monetary_policy',  5),
    ('fed rate',            'monetary_policy',  4),
    ('petroleo',            'commodities',      4),
    ('inflacion colombia',  'inflation',        4),
    ('EMBI',                'risk_premium',     4),
    ('devaluacion',         'fx_market',        4),
    ('remesas colombia',    'balance_payments', 3),
    ('inversion extranjera', 'capital_flows',   3),
    ('reforma tributaria',  'fiscal_policy',    3),
    ('TES',                 'fixed_income',     3),
    ('riesgo pais',         'risk_premium',     4)
ON CONFLICT (keyword) DO NOTHING;

-- ---------------------------------------------------------------------------
-- A4: cross_references — Topic clusters across sources
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_cross_references (
    id              SERIAL          PRIMARY KEY,
    topic           VARCHAR(500)    NOT NULL,
    cluster_date    DATE            NOT NULL,
    article_count   INTEGER         DEFAULT 0,
    avg_sentiment   DOUBLE PRECISION,
    dominant_category VARCHAR(50),
    sources_involved TEXT[],         -- Array of source_ids
    summary         TEXT,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crossref_date
    ON news_cross_references(cluster_date DESC);

-- ---------------------------------------------------------------------------
-- A5: cross_reference_articles — M:N join table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_cross_reference_articles (
    cross_reference_id INTEGER     NOT NULL REFERENCES news_cross_references(id) ON DELETE CASCADE,
    article_id         BIGINT      NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    similarity_score   DOUBLE PRECISION DEFAULT 0.0,
    PRIMARY KEY (cross_reference_id, article_id)
);

-- ---------------------------------------------------------------------------
-- A6: daily_digests — Raw statistical digests
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_daily_digests (
    id              SERIAL          PRIMARY KEY,
    digest_date     DATE            NOT NULL UNIQUE,
    digest_type     VARCHAR(20)     DEFAULT 'daily' CHECK (digest_type IN ('daily', 'weekly')),
    total_articles  INTEGER         DEFAULT 0,
    by_source       JSONB           DEFAULT '{}',   -- {source_id: count}
    by_category     JSONB           DEFAULT '{}',   -- {category: count}
    avg_sentiment   DOUBLE PRECISION,
    top_keywords    JSONB           DEFAULT '[]',   -- [{keyword, count}]
    top_articles    JSONB           DEFAULT '[]',   -- [{article_id, title, relevance}]
    cross_ref_count INTEGER         DEFAULT 0,
    summary_text    TEXT,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- A7: ingestion_log — Fetch audit trail
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_ingestion_log (
    id              BIGSERIAL       PRIMARY KEY,
    source_id       VARCHAR(50)     NOT NULL REFERENCES news_sources(source_id),
    run_type        VARCHAR(20)     DEFAULT 'scheduled' CHECK (run_type IN ('scheduled', 'manual', 'backfill')),
    started_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    articles_fetched INTEGER        DEFAULT 0,
    articles_new    INTEGER         DEFAULT 0,
    articles_updated INTEGER        DEFAULT 0,
    errors          INTEGER         DEFAULT 0,
    error_details   TEXT,
    status          VARCHAR(20)     DEFAULT 'running' CHECK (status IN ('running', 'success', 'partial', 'failed')),
    metadata        JSONB           DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_ingestion_source_date
    ON news_ingestion_log(source_id, started_at DESC);

-- ---------------------------------------------------------------------------
-- A8: feature_snapshots — Daily news-feature vectors
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_feature_snapshots (
    id              SERIAL          PRIMARY KEY,
    snapshot_date   DATE            NOT NULL UNIQUE,
    features        JSONB           NOT NULL,       -- {feature_name: value} ~60 features
    feature_version VARCHAR(20)     DEFAULT 'v1.0',
    article_count   INTEGER         DEFAULT 0,
    source_counts   JSONB           DEFAULT '{}',
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Monitoring view
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_news_daily_stats AS
SELECT
    DATE(published_at AT TIME ZONE 'America/Bogota') AS pub_date,
    source_id,
    COUNT(*) AS article_count,
    AVG(sentiment_score) AS avg_sentiment,
    AVG(relevance_score) AS avg_relevance,
    COUNT(*) FILTER (WHERE is_breaking) AS breaking_count
FROM news_articles
WHERE published_at > NOW() - INTERVAL '30 days'
GROUP BY pub_date, source_id
ORDER BY pub_date DESC, source_id;

COMMIT;
