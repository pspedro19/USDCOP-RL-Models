# SDD-03: Unified Storage Schema

| Campo | Valor |
|-------|-------|
| **Documento** | SDD-03 |
| **Título** | Unified Storage Schema |
| **Versión** | 2.0.0 |
| **Fecha** | 2026-02-25 |
| **Status** | 🔄 MERGED from NewsEngine SDD-03 v1 + Analysis Module DB spec |
| **Depende de** | SDD-00, SDD-01, SDD-02 |
| **Requerido por** | SDD-04, SDD-05, SDD-06, SDD-07, SDD-08 |

---

## 1. Engine & Strategy

| Entorno | Engine | Justificación |
|---------|--------|---------------|
| Desarrollo | SQLite | Zero-config, portable (NewsEngine dev) |
| Producción | PostgreSQL 16 | Concurrent writes, JSONB, FTS, partitioning |
| ORM | SQLAlchemy 2.0 + Alembic | NewsEngine tables |
| Raw SQL | Direct migrations | Analysis tables (aligns with existing trading DB) |

**Conventions:** `snake_case` tables (plural), `snake_case` columns, UUIDs for PKs (Postgres), timestamps always UTC.

---

## 2. Table Registry

### Group A — NewsEngine Tables (8 tables)

| # | Table | SDD | Purpose |
|---|-------|-----|---------|
| A1 | `sources` | SDD-02 | Registry of all data sources |
| A2 | `articles` | SDD-02/04 | Unified article storage (raw + enriched) |
| A3 | `macro_data` | SDD-02 | FRED/BanRep numeric time series |
| A4 | `keywords` | SDD-04 | Tracking keywords with priorities |
| A5 | `cross_references` | SDD-05 | Topic clusters across sources |
| A6 | `cross_reference_articles` | SDD-05 | M:N join table |
| A7 | `daily_digests` | SDD-06 | NewsEngine text digests |
| A8 | `ingestion_log` | SDD-02 | Fetch execution audit trail |
| A9 | `feature_snapshots` | SDD-06 | Daily feature vectors for reproducibility |

### Group B — Analysis Module Tables (4 tables)

| # | Table | SDD | Purpose |
|---|-------|-----|---------|
| B1 | `weekly_analysis` | SDD-07 | AI-generated weekly reports |
| B2 | `daily_analysis` | SDD-07 | AI-generated daily analysis entries |
| B3 | `macro_variable_snapshots` | SDD-07 | SMA + trend snapshots per variable per day |
| B4 | `analysis_chat_history` | SDD-09 | Chat conversation log |

### Shared Data (read by both subsystems)

| Table | Written by | Read by |
|-------|-----------|---------|
| `articles` | NewsEngine (ingestion + enrichment) | Analysis Engine (prompt context) |
| `macro_data` | NewsEngine (FRED/BanRep adapters) | Analysis Engine (SMA computation) |
| `cross_references` | NewsEngine (cross-ref engine) | Analysis Engine (prompt context) |

---

## 3. Group A — NewsEngine Tables

### A1: `sources`

```sql
CREATE TABLE sources (
    id            SERIAL PRIMARY KEY,
    source_id     VARCHAR(50) UNIQUE NOT NULL,
    name          VARCHAR(100) NOT NULL,
    source_type   VARCHAR(20) NOT NULL,            -- "api" | "scraper" | "macro"
    base_url      VARCHAR(500) NOT NULL,
    is_enabled    BOOLEAN DEFAULT TRUE,
    config        JSONB DEFAULT '{}',
    last_fetched_at TIMESTAMPTZ,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_source_type CHECK (source_type IN ('api', 'scraper', 'macro'))
);
```

### A2: `articles`

```sql
CREATE TABLE articles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       VARCHAR(50) NOT NULL REFERENCES sources(source_id),
    ingestion_id    UUID,

    -- Content
    url             VARCHAR(2000) UNIQUE NOT NULL,
    title           VARCHAR(1000) NOT NULL,
    summary         TEXT,
    content         TEXT,
    author          VARCHAR(500),
    image_url       VARCHAR(2000),

    -- Temporal
    published_at    TIMESTAMPTZ,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),

    -- Source metadata
    source_domain   VARCHAR(200),
    source_country  VARCHAR(10),
    language        VARCHAR(10),

    -- Enrichment (SDD-04, initially NULL)
    category        VARCHAR(50),
    tags            JSONB DEFAULT '[]',
    relevance_score FLOAT,
    sentiment_score FLOAT,
    sentiment_source VARCHAR(20),
    is_weekly_analysis BOOLEAN DEFAULT FALSE,

    -- GDELT-specific
    external_tone   FLOAT,

    -- Raw
    raw_data        JSONB,

    CONSTRAINT chk_title_length CHECK (char_length(title) >= 10)
);

CREATE INDEX idx_articles_published    ON articles(published_at DESC);
CREATE INDEX idx_articles_source       ON articles(source_id);
CREATE INDEX idx_articles_category     ON articles(category);
CREATE INDEX idx_articles_ingested     ON articles(ingested_at DESC);
CREATE INDEX idx_articles_relevance    ON articles(relevance_score DESC);
CREATE INDEX idx_articles_pub_date     ON articles(DATE(published_at));
CREATE INDEX idx_articles_fts ON articles
    USING GIN (to_tsvector('spanish', coalesce(title,'') || ' ' || coalesce(summary,'')));
```

### A3: `macro_data`

```sql
CREATE TABLE macro_data (
    id              SERIAL PRIMARY KEY,
    source_id       VARCHAR(50) NOT NULL REFERENCES sources(source_id),
    series_id       VARCHAR(50) NOT NULL,
    series_name     VARCHAR(200) NOT NULL,
    observation_date DATE NOT NULL,
    value           FLOAT,
    unit            VARCHAR(50),
    fetched_at      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_macro_series_date UNIQUE (series_id, observation_date)
);

CREATE INDEX idx_macro_date ON macro_data(observation_date DESC);
CREATE INDEX idx_macro_series ON macro_data(series_id, observation_date DESC);
```

### A4: `keywords`

```sql
CREATE TABLE keywords (
    id          SERIAL PRIMARY KEY,
    keyword     VARCHAR(100) UNIQUE NOT NULL,
    category    VARCHAR(50),
    priority    INTEGER DEFAULT 5,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

### A5-A6: `cross_references` + join table

```sql
CREATE TABLE cross_references (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic           VARCHAR(500) NOT NULL,
    match_score     FLOAT NOT NULL DEFAULT 0.0,
    sources_count   INTEGER NOT NULL DEFAULT 2,
    reference_date  DATE NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE cross_reference_articles (
    cross_ref_id    UUID NOT NULL REFERENCES cross_references(id) ON DELETE CASCADE,
    article_id      UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    PRIMARY KEY (cross_ref_id, article_id)
);

CREATE INDEX idx_crossref_date ON cross_references(reference_date DESC);
```

### A7: `daily_digests` (NewsEngine digests — distinct from `daily_analysis`)

```sql
CREATE TABLE daily_digests (
    id              SERIAL PRIMARY KEY,
    digest_date     DATE NOT NULL,
    digest_type     VARCHAR(10) NOT NULL,           -- "daily" | "weekly"
    total_articles  INTEGER DEFAULT 0,
    top_story_ids   JSONB DEFAULT '[]',
    key_topics      JSONB DEFAULT '[]',
    market_indicators JSONB DEFAULT '{}',
    category_breakdown JSONB DEFAULT '{}',
    source_breakdown JSONB DEFAULT '{}',
    generated_at    TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_digest_date_type UNIQUE (digest_date, digest_type)
);
```

**Note:** `daily_digests` (NewsEngine, SDD-06) is a raw statistical summary. `daily_analysis` (Analysis Module, SDD-07) is an AI-generated narrative. They serve different purposes and coexist.

### A8: `ingestion_log`

```sql
CREATE TABLE ingestion_log (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id           VARCHAR(50) NOT NULL REFERENCES sources(source_id),
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    status              VARCHAR(20) NOT NULL,
    articles_fetched    INTEGER DEFAULT 0,
    articles_new        INTEGER DEFAULT 0,
    articles_skipped    INTEGER DEFAULT 0,
    error_message       TEXT,
    execution_time_sec  FLOAT,
    CONSTRAINT chk_status CHECK (status IN ('success', 'partial', 'failed'))
);

CREATE INDEX idx_ingestion_log_source ON ingestion_log(source_id, started_at DESC);
```

### A9: `feature_snapshots`

```sql
CREATE TABLE feature_snapshots (
    id              SERIAL PRIMARY KEY,
    snapshot_date   DATE UNIQUE NOT NULL,
    features        JSONB NOT NULL,
    feature_version VARCHAR(20) NOT NULL,
    generated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feature_date ON feature_snapshots(snapshot_date DESC);
```

---

## 4. Group B — Analysis Module Tables

### B1: `weekly_analysis`

```sql
CREATE TABLE weekly_analysis (
    id                    SERIAL PRIMARY KEY,
    iso_year              SMALLINT NOT NULL,
    iso_week              SMALLINT NOT NULL,
    week_start_date       DATE NOT NULL,
    week_end_date         DATE NOT NULL,

    -- AI-generated content
    summary_markdown      TEXT,
    key_themes            JSONB DEFAULT '[]',
    sentiment             VARCHAR(20),              -- bullish_cop | bearish_cop | neutral | mixed
    sentiment_score       NUMERIC(4,2),

    -- Model signals context (from existing trading system)
    h5_signal_direction   VARCHAR(10),
    h5_confidence_tier    VARCHAR(10),
    h5_ensemble_return    NUMERIC(8,4),
    h1_signal_summary     JSONB DEFAULT '{}',       -- {long: N, short: N, hold: N}

    -- Weekly OHLCV
    week_open_price       NUMERIC(10,2),
    week_close_price      NUMERIC(10,2),
    week_high             NUMERIC(10,2),
    week_low              NUMERIC(10,2),
    week_return_pct       NUMERIC(8,4),

    -- Key macro snapshots (denormalized for fast reads)
    dxy_close             NUMERIC(10,4),
    dxy_sma20             NUMERIC(10,4),
    vix_close             NUMERIC(10,4),
    vix_sma20             NUMERIC(10,4),
    oil_close             NUMERIC(10,4),
    oil_sma20             NUMERIC(10,4),
    embi_close            NUMERIC(10,4),
    embi_sma20            NUMERIC(10,4),

    -- NewsEngine integration (cross-system link)
    news_article_count    INTEGER,                  -- Total articles from NewsEngine that week
    news_top_categories   JSONB DEFAULT '{}',       -- {forex: N, oil: N, ...} from articles table
    news_crossref_count   INTEGER,                  -- Cross-references from NewsEngine

    -- Generation metadata
    llm_model             VARCHAR(100),
    llm_tokens_used       INTEGER,
    generation_time_s     NUMERIC(6,2),
    generator_version     VARCHAR(20) DEFAULT '1.0.0',

    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_weekly_analysis_week UNIQUE (iso_year, iso_week)
);

CREATE INDEX idx_weekly_analysis_date ON weekly_analysis(week_start_date DESC);
```

### B2: `daily_analysis`

```sql
CREATE TABLE daily_analysis (
    id                    SERIAL PRIMARY KEY,
    analysis_date         DATE NOT NULL UNIQUE,
    iso_year              SMALLINT NOT NULL,
    iso_week              SMALLINT NOT NULL,
    day_of_week           SMALLINT NOT NULL,

    -- AI-generated content
    headline              VARCHAR(200) NOT NULL,
    daily_markdown        TEXT NOT NULL,
    key_events            JSONB DEFAULT '[]',
    sentiment             VARCHAR(20),
    sentiment_score       NUMERIC(4,2),

    -- Daily OHLCV
    daily_open            NUMERIC(10,2),
    daily_close           NUMERIC(10,2),
    daily_high            NUMERIC(10,2),
    daily_low             NUMERIC(10,2),
    daily_return_pct      NUMERIC(8,4),

    -- H1 model signal
    h1_direction          VARCHAR(10),
    h1_prediction         NUMERIC(10,2),
    h1_confidence         VARCHAR(10),

    -- Macro publications that day (from EconomicCalendar + macro_data)
    macro_publications    JSONB DEFAULT '[]',

    -- NewsEngine integration
    news_article_count    INTEGER,                  -- Articles from articles table for this date
    news_top_stories      JSONB DEFAULT '[]',       -- Top N by relevance_score from articles
    news_sentiment_avg    NUMERIC(4,2),             -- AVG(sentiment_score) from articles table

    -- Generation metadata
    llm_model             VARCHAR(100),
    llm_tokens_used       INTEGER,

    created_at            TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fk_daily_weekly FOREIGN KEY (iso_year, iso_week)
        REFERENCES weekly_analysis (iso_year, iso_week)
        ON DELETE CASCADE
);

CREATE INDEX idx_daily_analysis_week ON daily_analysis(iso_year, iso_week);
CREATE INDEX idx_daily_analysis_date ON daily_analysis(analysis_date DESC);
```

### B3: `macro_variable_snapshots`

```sql
CREATE TABLE macro_variable_snapshots (
    id                    SERIAL PRIMARY KEY,
    snapshot_date         DATE NOT NULL,
    variable_key          VARCHAR(80) NOT NULL,      -- DB column name from SSOT
    friendly_name         VARCHAR(30) NOT NULL,
    display_name          VARCHAR(60) NOT NULL,
    category              VARCHAR(20) NOT NULL,      -- key_features | rates | commodities | fx_peers

    -- Values (sourced from macro_data + macro_indicators_daily)
    current_value         NUMERIC(14,4),
    previous_close        NUMERIC(14,4),
    daily_change_pct      NUMERIC(8,4),
    weekly_change_pct     NUMERIC(8,4),

    -- Computed SMAs
    sma_5                 NUMERIC(14,4),
    sma_10                NUMERIC(14,4),
    sma_20                NUMERIC(14,4),
    sma_50                NUMERIC(14,4),

    -- Derived signals
    trend_vs_sma20        VARCHAR(20),              -- above | below | crossing_up | crossing_down
    sma_cross_signal      VARCHAR(20),              -- golden_cross | death_cross | none
    trend_direction       VARCHAR(10),              -- up | down | sideways
    z_score_20d           NUMERIC(8,4),

    -- Impact
    impact_on_usdcop      VARCHAR(20),
    impact_level          VARCHAR(10),

    created_at            TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_macro_snapshot UNIQUE (snapshot_date, variable_key)
);

CREATE INDEX idx_macro_snap_date ON macro_variable_snapshots(snapshot_date DESC);
CREATE INDEX idx_macro_snap_var ON macro_variable_snapshots(variable_key, snapshot_date DESC);
```

### B4: `analysis_chat_history`

```sql
CREATE TABLE analysis_chat_history (
    id                    SERIAL PRIMARY KEY,
    session_id            UUID NOT NULL DEFAULT gen_random_uuid(),
    role                  VARCHAR(15) NOT NULL,
    content               TEXT NOT NULL,
    context_week          VARCHAR(10),
    context_date          DATE,
    llm_model             VARCHAR(100),
    tokens_used           INTEGER,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_session ON analysis_chat_history(session_id, created_at);
```

---

## 5. Entity-Relationship Diagram

```
                    GROUP A: NewsEngine
                    ───────────────────

sources ──────┬───── articles ──── cross_reference_articles ──── cross_references
    │         │         │
    │    ingestion_log  │
    │                   │
    ├───── macro_data ──┤
    │                   │
    │              keywords
    │
    ├── daily_digests
    └── feature_snapshots


                    GROUP B: Analysis Module
                    ────────────────────────

weekly_analysis ──1:N── daily_analysis
                            │
                            │ context reference
                            ▼
                    analysis_chat_history

macro_variable_snapshots (independent, date-keyed)


                    CROSS-SYSTEM READS
                    ──────────────────

articles ─────────────────────▶ daily_analysis.news_* fields
                                weekly_analysis.news_* fields
                                prompt context (LLM)

macro_data ───────────────────▶ macro_variable_snapshots (SMA source)
                                prompt context (LLM)

cross_references ─────────────▶ prompt context (LLM)

macro_indicators_daily ───────▶ macro_variable_snapshots (SMA source)
(existing trading DB table)     [read via UnifiedMacroLoader]
```

---

## 6. Naming Disambiguation

Two pairs of similarly-named tables serve **different purposes**:

| NewsEngine Table | Analysis Table | Difference |
|------------------|----------------|------------|
| `daily_digests` | `daily_analysis` | Digests = raw stats (article counts, category breakdown). Analysis = AI-generated narrative with headline, markdown, sentiment interpretation. |
| `macro_data` | `macro_variable_snapshots` | `macro_data` = raw time series from FRED/BanRep. `macro_variable_snapshots` = computed SMAs, trend signals, z-scores for dashboard display. |

---

## 7. Data Retention

| Table | Growth Rate | Retention | Cleanup |
|-------|------------|-----------|---------|
| `articles` (full) | ~200K rows/year | 1 year full → metadata only | Nullify `content` column |
| `articles` (meta) | — | Indefinite | — |
| `macro_data` | ~8K rows/year | Indefinite | — |
| `ingestion_log` | ~2K rows/year | 90 days | DELETE |
| `feature_snapshots` | ~260 rows/year | Indefinite | — |
| `cross_references` | ~5K rows/year | 1 year | DELETE CASCADE |
| `weekly_analysis` | ~52 rows/year | Indefinite | — |
| `daily_analysis` | ~260 rows/year | Indefinite | — |
| `macro_variable_snapshots` | ~3.4K rows/year | 2 years | DELETE |
| `analysis_chat_history` | Variable | 6 months | DELETE |

---

## 8. Migration Files

| # | File | Tables |
|---|------|--------|
| 001 | `001_newsengine_initial.sql` | sources, articles, macro_data, keywords, cross_references, cross_reference_articles, daily_digests, ingestion_log, feature_snapshots |
| 045 | `045_weekly_analysis_tables.sql` | weekly_analysis, daily_analysis, macro_variable_snapshots, analysis_chat_history |

Both migrations are idempotent (`CREATE TABLE IF NOT EXISTS`).
