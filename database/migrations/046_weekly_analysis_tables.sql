-- =============================================================================
-- Migration 046: Analysis Module — 4 Tables
-- =============================================================================
--
-- Creates the Analysis Module storage layer for AI-generated weekly/daily
-- analysis, macro variable snapshots (SMA/BB/RSI/MACD), and chat history.
--
-- Group B tables per SDD-03. Enhanced with Bollinger bands, RSI, MACD, ROC
-- columns on macro_variable_snapshots (beyond original spec).
--
-- Contract: CTR-ANALYSIS-STORAGE-001
-- Date: 2026-02-25
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- B1: weekly_analysis — AI weekly reports
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weekly_analysis (
    id              SERIAL          PRIMARY KEY,
    iso_year        INTEGER         NOT NULL,
    iso_week        INTEGER         NOT NULL,
    week_start      DATE            NOT NULL,
    week_end        DATE            NOT NULL,

    -- AI-generated content
    summary_markdown TEXT,                      -- Full weekly analysis in markdown (Spanish)
    headline        VARCHAR(500),               -- One-line summary
    sentiment       VARCHAR(20),                -- 'bullish', 'bearish', 'neutral', 'mixed'
    themes          JSONB           DEFAULT '[]', -- [{theme, description, impact}]

    -- OHLCV context
    ohlcv_summary   JSONB           DEFAULT '{}', -- {open, high, low, close, change_pct, range_pct}

    -- Signals context
    h5_signal       JSONB           DEFAULT '{}', -- {direction, confidence, predicted_return, leverage}
    h1_signals      JSONB           DEFAULT '[]', -- [{date, direction, signal_strength}]

    -- News context (from NewsEngine)
    news_summary    JSONB           DEFAULT '{}', -- {article_count, top_categories, avg_sentiment, cross_refs}

    -- Metadata
    llm_model       VARCHAR(100),
    llm_tokens_used INTEGER,
    llm_cost_usd    DOUBLE PRECISION,
    generation_time_s DOUBLE PRECISION,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW(),

    CONSTRAINT uq_weekly_analysis_year_week UNIQUE (iso_year, iso_week)
);

-- ---------------------------------------------------------------------------
-- B2: daily_analysis — AI daily analysis entries
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_analysis (
    id              SERIAL          PRIMARY KEY,
    analysis_date   DATE            NOT NULL UNIQUE,
    iso_year        INTEGER         NOT NULL,
    iso_week        INTEGER         NOT NULL,
    day_of_week     INTEGER         NOT NULL,     -- 0=Mon, 4=Fri

    -- AI-generated content
    headline        VARCHAR(500),
    summary_markdown TEXT,
    sentiment       VARCHAR(20),

    -- Market data
    usdcop_close    DOUBLE PRECISION,
    usdcop_change_pct DOUBLE PRECISION,
    usdcop_high     DOUBLE PRECISION,
    usdcop_low      DOUBLE PRECISION,

    -- Signals
    h1_signal       JSONB           DEFAULT '{}', -- {direction, magnitude, confidence}
    h5_status       JSONB           DEFAULT '{}', -- {active_trade, direction, pnl_pct, exit_reason}

    -- Events & publications
    macro_publications JSONB        DEFAULT '[]', -- [{variable, value, previous, change_pct}]
    economic_events JSONB           DEFAULT '[]', -- [{event, impact_level, actual, forecast}]
    news_highlights JSONB           DEFAULT '[]', -- [{title, source, sentiment, url}]

    -- Metadata
    llm_model       VARCHAR(100),
    llm_tokens_used INTEGER,
    llm_cost_usd    DOUBLE PRECISION,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_daily_analysis_week
    ON daily_analysis(iso_year, iso_week);

-- ---------------------------------------------------------------------------
-- B3: macro_variable_snapshots — SMA/BB/RSI/MACD per variable per day
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS macro_variable_snapshots (
    id              SERIAL          PRIMARY KEY,
    snapshot_date   DATE            NOT NULL,
    variable_key    VARCHAR(50)     NOT NULL,       -- e.g., 'dxy', 'vix', 'wti', 'embi_col'
    variable_name   VARCHAR(200),                   -- Human-readable name

    -- Raw value
    value           DOUBLE PRECISION NOT NULL,

    -- Simple Moving Averages
    sma_5           DOUBLE PRECISION,
    sma_10          DOUBLE PRECISION,
    sma_20          DOUBLE PRECISION,
    sma_50          DOUBLE PRECISION,

    -- Bollinger Bands (20-period, 2 std)
    bollinger_upper_20 DOUBLE PRECISION,            -- SMA_20 + 2*std_20
    bollinger_lower_20 DOUBLE PRECISION,            -- SMA_20 - 2*std_20
    bollinger_width_20 DOUBLE PRECISION,            -- (upper - lower) / SMA_20

    -- RSI (Wilder's, 14-period)
    rsi_14          DOUBLE PRECISION,

    -- MACD (12, 26, 9)
    macd_line       DOUBLE PRECISION,               -- EMA_12 - EMA_26
    macd_signal     DOUBLE PRECISION,               -- EMA_9 of MACD line
    macd_histogram  DOUBLE PRECISION,               -- MACD - Signal

    -- Rate of Change
    roc_5           DOUBLE PRECISION,               -- 5-period ROC (%)
    roc_20          DOUBLE PRECISION,               -- 20-period ROC (%)

    -- Derived signals
    z_score_20      DOUBLE PRECISION,               -- (value - SMA_20) / std_20
    trend           VARCHAR(20),                    -- 'above_sma20', 'below_sma20', 'golden_cross', 'death_cross'
    signal          VARCHAR(30),                    -- 'overbought', 'oversold', 'neutral', 'bb_squeeze', etc.

    -- Metadata
    created_at      TIMESTAMPTZ     DEFAULT NOW(),

    CONSTRAINT uq_macro_snapshot_date_var UNIQUE (snapshot_date, variable_key)
);

CREATE INDEX IF NOT EXISTS idx_macro_snapshot_variable
    ON macro_variable_snapshots(variable_key, snapshot_date DESC);

-- ---------------------------------------------------------------------------
-- B4: analysis_chat_history — Chat sessions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analysis_chat_history (
    id              BIGSERIAL       PRIMARY KEY,
    session_id      VARCHAR(64)     NOT NULL,       -- UUID or hash
    role            VARCHAR(20)     NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT            NOT NULL,
    context_year    INTEGER,
    context_week    INTEGER,
    tokens_used     INTEGER,
    llm_model       VARCHAR(100),
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_session
    ON analysis_chat_history(session_id, created_at);

-- ---------------------------------------------------------------------------
-- Monitoring view for analysis
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_analysis_coverage AS
SELECT
    wa.iso_year,
    wa.iso_week,
    wa.week_start,
    wa.sentiment AS weekly_sentiment,
    COUNT(DISTINCT da.id) AS daily_entries,
    COUNT(DISTINCT mvs.variable_key) AS macro_vars_tracked,
    wa.llm_cost_usd AS weekly_cost,
    SUM(da.llm_cost_usd) AS daily_total_cost
FROM weekly_analysis wa
LEFT JOIN daily_analysis da
    ON da.iso_year = wa.iso_year AND da.iso_week = wa.iso_week
LEFT JOIN macro_variable_snapshots mvs
    ON mvs.snapshot_date BETWEEN wa.week_start AND wa.week_end
GROUP BY wa.id
ORDER BY wa.iso_year DESC, wa.iso_week DESC;

COMMIT;
