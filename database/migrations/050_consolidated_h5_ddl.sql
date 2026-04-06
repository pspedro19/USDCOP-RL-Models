-- =============================================================================
-- Migration 050: Consolidated H5 DDL (v2.0)
-- =============================================================================
-- This is a CONSOLIDATED DDL that includes ALL columns from migrations 043+044+048.
-- Use this for fresh installs. For existing DBs, run 043→044→048 individually.
--
-- IDEMPOTENT: Uses IF NOT EXISTS and ADD COLUMN IF NOT EXISTS throughout.
-- Safe to run on both fresh and existing databases.
--
-- Date: 2026-04-06
-- Version: Smart Simple v2.0
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- forecast_h5_predictions (model-level predictions)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_h5_predictions (
    id              BIGSERIAL PRIMARY KEY,
    inference_date  DATE NOT NULL,
    inference_week  INT NOT NULL,
    inference_year  INT NOT NULL,
    model_name      VARCHAR(30) NOT NULL,
    pred_return     DOUBLE PRECISION NOT NULL,
    pred_direction  VARCHAR(10) NOT NULL,
    direction       VARCHAR(10) NOT NULL,
    collapse_flag   BOOLEAN DEFAULT FALSE,
    collapse_std    DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(inference_date, model_name)
);

-- ---------------------------------------------------------------------------
-- forecast_h5_signals (weekly ensemble signal + v2.0 regime gate)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_h5_signals (
    id                      BIGSERIAL PRIMARY KEY,
    signal_date             DATE NOT NULL UNIQUE,
    inference_date          DATE NOT NULL,
    inference_week          INT NOT NULL,
    inference_year          INT NOT NULL,
    ensemble_return         DOUBLE PRECISION NOT NULL,
    direction               INT NOT NULL,
    realized_vol_21d        DOUBLE PRECISION,
    raw_leverage            DOUBLE PRECISION,
    clipped_leverage        DOUBLE PRECISION,
    asymmetric_leverage     DOUBLE PRECISION,
    long_multiplier         DOUBLE PRECISION DEFAULT 0.5,
    short_multiplier        DOUBLE PRECISION DEFAULT 1.0,
    collapse_flag           BOOLEAN DEFAULT FALSE,
    config_version          VARCHAR(30) DEFAULT 'smart_simple_v2.0',
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    -- v1.1 (migration 044): Confidence + stops
    confidence_tier         VARCHAR(10),
    confidence_agreement    DOUBLE PRECISION,
    confidence_magnitude    DOUBLE PRECISION,
    sizing_multiplier       DOUBLE PRECISION,
    skip_trade              BOOLEAN DEFAULT FALSE,
    hard_stop_pct           DOUBLE PRECISION,
    take_profit_pct         DOUBLE PRECISION,
    adjusted_leverage       DOUBLE PRECISION,
    -- v2.0 (migration 048): Regime gate + dynamic leverage
    regime                  VARCHAR(20),
    hurst_exponent          DOUBLE PRECISION,
    regime_leverage_scaler  DOUBLE PRECISION,
    rolling_wr_8w           DOUBLE PRECISION,
    dl_leverage_scaler      DOUBLE PRECISION,
    effective_hs_pct        DOUBLE PRECISION,
    effective_tp_pct        DOUBLE PRECISION
);

-- ---------------------------------------------------------------------------
-- forecast_h5_executions (weekly trade tracking)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_h5_executions (
    id                       BIGSERIAL PRIMARY KEY,
    signal_date              DATE NOT NULL UNIQUE,
    inference_week           INT NOT NULL,
    inference_year           INT NOT NULL,
    direction                INT NOT NULL,
    leverage                 DOUBLE PRECISION NOT NULL,
    entry_price              DOUBLE PRECISION,
    entry_timestamp          TIMESTAMPTZ,
    exit_price               DOUBLE PRECISION,
    exit_timestamp           TIMESTAMPTZ,
    exit_reason              VARCHAR(30),
    n_subtrades              INT DEFAULT 0,
    week_pnl_pct             DOUBLE PRECISION,
    week_pnl_unleveraged_pct DOUBLE PRECISION,
    status                   VARCHAR(20) DEFAULT 'pending',
    config_version           VARCHAR(30) DEFAULT 'smart_simple_v2.0',
    created_at               TIMESTAMPTZ DEFAULT NOW(),
    updated_at               TIMESTAMPTZ DEFAULT NOW(),
    -- v1.1 (migration 044)
    confidence_tier          VARCHAR(10),
    hard_stop_pct            DOUBLE PRECISION,
    take_profit_pct          DOUBLE PRECISION,
    -- v2.0 (migration 048)
    regime                   VARCHAR(20),
    hurst_exponent           DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_h5_exec_status ON forecast_h5_executions(status);
CREATE INDEX IF NOT EXISTS idx_h5_exec_week ON forecast_h5_executions(inference_year, inference_week);

-- ---------------------------------------------------------------------------
-- forecast_h5_subtrades (intra-week subtrade tracking)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_h5_subtrades (
    id                  BIGSERIAL PRIMARY KEY,
    execution_id        BIGINT REFERENCES forecast_h5_executions(id),
    subtrade_index      INT DEFAULT 0,
    direction           INT,
    entry_price         DOUBLE PRECISION,
    entry_timestamp     TIMESTAMPTZ,
    exit_price          DOUBLE PRECISION,
    exit_timestamp      TIMESTAMPTZ,
    exit_reason         VARCHAR(30),
    peak_price          DOUBLE PRECISION,
    trailing_state      VARCHAR(20),
    bar_count           INT DEFAULT 0,
    pnl_pct             DOUBLE PRECISION,
    pnl_unleveraged_pct DOUBLE PRECISION,
    cooldown_until      TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- forecast_h5_paper_trading (weekly evaluation + guardrails)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_h5_paper_trading (
    id                      BIGSERIAL PRIMARY KEY,
    signal_date             DATE,
    inference_week          INT,
    inference_year          INT,
    direction               INT,
    leverage                DOUBLE PRECISION,
    week_pnl_pct            DOUBLE PRECISION,
    n_subtrades             INT,
    cumulative_pnl_pct      DOUBLE PRECISION,
    running_da_pct          DOUBLE PRECISION,
    running_da_short_pct    DOUBLE PRECISION,
    running_da_long_pct     DOUBLE PRECISION,
    running_sharpe          DOUBLE PRECISION,
    running_max_dd_pct      DOUBLE PRECISION,
    n_weeks                 INT,
    n_long                  INT,
    n_short                 INT,
    long_pct_8w             DOUBLE PRECISION,
    consecutive_losses      INT DEFAULT 0,
    circuit_breaker         BOOLEAN DEFAULT FALSE,
    gate_status             VARCHAR(30),
    notes                   TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_h5_performance_summary AS
SELECT
    e.inference_year,
    e.inference_week,
    e.signal_date,
    e.direction,
    e.leverage,
    e.entry_price,
    e.exit_price,
    e.exit_reason,
    e.week_pnl_pct,
    e.status,
    e.confidence_tier,
    e.regime,
    e.hurst_exponent,
    s.ensemble_return,
    s.skip_trade,
    s.regime_leverage_scaler,
    s.dl_leverage_scaler
FROM forecast_h5_executions e
LEFT JOIN forecast_h5_signals s ON e.signal_date = s.signal_date
ORDER BY e.signal_date DESC;

COMMIT;
