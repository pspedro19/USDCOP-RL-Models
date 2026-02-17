-- =============================================================================
-- Migration 043: Track B — H=5 Linear-Only Paper Trading Tables
-- =============================================================================
--
-- 5 tables for H=5 weekly forecasting paper trading (Track B).
-- Completely isolated from Track A (H=1 daily) tables.
--
-- Tables:
--   forecast_h5_predictions   — Per-model weekly predictions
--   forecast_h5_signals       — Ensemble output: direction + leverage
--   forecast_h5_executions    — Weekly parent execution: entry/exit/PnL
--   forecast_h5_subtrades     — Sub-trades within a week (re-entry after trailing)
--   forecast_h5_paper_trading — Weekly summary for decision gates
--
-- Views:
--   v_h5_performance_summary  — Running performance metrics
--   v_h5_collapse_monitor     — Model collapse detection
--
-- Contract: FC-H5-TABLES-001
-- Date: 2026-02-16
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. forecast_h5_predictions
-- ---------------------------------------------------------------------------
-- Per-model weekly predictions. 2 rows per week (Ridge + BayesianRidge).
-- collapse_flag = TRUE when rolling 12-week prediction std < threshold.

CREATE TABLE IF NOT EXISTS forecast_h5_predictions (
    id              BIGSERIAL PRIMARY KEY,
    inference_date  DATE NOT NULL,              -- Sunday training date
    inference_week  INT NOT NULL,               -- ISO week number
    inference_year  INT NOT NULL,               -- ISO year
    target_date     DATE NOT NULL,              -- Target date (inference_date + 5 bdays)
    model_id        VARCHAR(30) NOT NULL,       -- 'ridge' or 'bayesian_ridge'
    horizon_id      INT NOT NULL DEFAULT 5,     -- Always 5 for Track B
    base_price      DOUBLE PRECISION NOT NULL,  -- Friday close used for prediction
    predicted_price DOUBLE PRECISION NOT NULL,
    predicted_return_pct DOUBLE PRECISION NOT NULL,  -- ln(pred/base) * 100
    direction       VARCHAR(10) NOT NULL,       -- 'UP' or 'DOWN'
    collapse_flag   BOOLEAN DEFAULT FALSE,      -- Rolling std < collapse_threshold
    collapse_std    DOUBLE PRECISION,           -- Rolling 12-week prediction std
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_h5_pred UNIQUE (inference_date, model_id, horizon_id)
);

CREATE INDEX IF NOT EXISTS idx_h5_pred_date ON forecast_h5_predictions (inference_date);
CREATE INDEX IF NOT EXISTS idx_h5_pred_week ON forecast_h5_predictions (inference_year, inference_week);


-- ---------------------------------------------------------------------------
-- 2. forecast_h5_signals
-- ---------------------------------------------------------------------------
-- Ensemble output: mean of 2 linear models -> direction + leverage.

CREATE TABLE IF NOT EXISTS forecast_h5_signals (
    id                  BIGSERIAL PRIMARY KEY,
    signal_date         DATE NOT NULL UNIQUE,       -- Monday signal date
    inference_date      DATE NOT NULL,              -- Sunday training date
    inference_week      INT NOT NULL,
    inference_year      INT NOT NULL,
    ensemble_return     DOUBLE PRECISION NOT NULL,  -- Mean of Ridge + BR predictions
    direction           INT NOT NULL,               -- +1 long, -1 short
    realized_vol_21d    DOUBLE PRECISION,           -- 21-day annualized realized vol
    raw_leverage        DOUBLE PRECISION,           -- Before clipping
    clipped_leverage    DOUBLE PRECISION,           -- After [0.5, 2.0] clip
    asymmetric_leverage DOUBLE PRECISION,           -- After direction-dependent scaling
    long_multiplier     DOUBLE PRECISION DEFAULT 0.5,
    short_multiplier    DOUBLE PRECISION DEFAULT 1.0,
    collapse_flag       BOOLEAN DEFAULT FALSE,      -- Any model collapsed?
    config_version      VARCHAR(30) DEFAULT 'smart_executor_h5_v1',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_h5_signal_week ON forecast_h5_signals (inference_year, inference_week);


-- ---------------------------------------------------------------------------
-- 3. forecast_h5_executions
-- ---------------------------------------------------------------------------
-- Weekly parent execution. One row per trading week.

CREATE TABLE IF NOT EXISTS forecast_h5_executions (
    id                  BIGSERIAL PRIMARY KEY,
    signal_date         DATE NOT NULL UNIQUE,       -- Monday signal date
    inference_week      INT NOT NULL,
    inference_year      INT NOT NULL,
    direction           INT NOT NULL,               -- +1 long, -1 short
    leverage            DOUBLE PRECISION NOT NULL,  -- Asymmetric leverage used
    entry_price         DOUBLE PRECISION,
    entry_timestamp     TIMESTAMPTZ,
    exit_price          DOUBLE PRECISION,
    exit_timestamp      TIMESTAMPTZ,
    exit_reason         VARCHAR(30),                -- 'week_end', 'hard_stop', 'circuit_breaker'
    n_subtrades         INT DEFAULT 0,              -- Total subtrades in this week
    week_pnl_pct        DOUBLE PRECISION,           -- Total week PnL (leveraged)
    week_pnl_unleveraged_pct DOUBLE PRECISION,      -- Total week PnL (unleveraged)
    status              VARCHAR(20) DEFAULT 'pending',  -- pending/positioned/closed/error/paused
    config_version      VARCHAR(30) DEFAULT 'smart_executor_h5_v1',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_h5_exec_week ON forecast_h5_executions (inference_year, inference_week);
CREATE INDEX IF NOT EXISTS idx_h5_exec_status ON forecast_h5_executions (status);


-- ---------------------------------------------------------------------------
-- 4. forecast_h5_subtrades
-- ---------------------------------------------------------------------------
-- Sub-trades within a week. Re-entry after trailing exit creates new subtrades.

CREATE TABLE IF NOT EXISTS forecast_h5_subtrades (
    id                  BIGSERIAL PRIMARY KEY,
    execution_id        BIGINT NOT NULL REFERENCES forecast_h5_executions(id),
    subtrade_index      INT NOT NULL DEFAULT 0,     -- 0 = initial entry, 1+ = re-entries
    direction           INT NOT NULL,               -- Same as parent execution direction
    entry_price         DOUBLE PRECISION NOT NULL,
    entry_timestamp     TIMESTAMPTZ NOT NULL,
    exit_price          DOUBLE PRECISION,
    exit_timestamp      TIMESTAMPTZ,
    exit_reason         VARCHAR(30),                -- 'trailing_stop', 'hard_stop', 'week_end', 'circuit_breaker'
    peak_price          DOUBLE PRECISION,           -- Best favorable price for trailing
    trailing_state      VARCHAR(20) DEFAULT 'waiting',  -- waiting/active/triggered/expired
    bar_count           INT DEFAULT 0,              -- Bars monitored in this subtrade
    pnl_pct             DOUBLE PRECISION,           -- Subtrade PnL (leveraged)
    pnl_unleveraged_pct DOUBLE PRECISION,           -- Subtrade PnL (unleveraged)
    cooldown_until      TIMESTAMPTZ,                -- No re-entry before this time
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_h5_sub_exec ON forecast_h5_subtrades (execution_id);
CREATE INDEX IF NOT EXISTS idx_h5_sub_state ON forecast_h5_subtrades (trailing_state);


-- ---------------------------------------------------------------------------
-- 5. forecast_h5_paper_trading
-- ---------------------------------------------------------------------------
-- Weekly summary for decision gates. One row per completed week.

CREATE TABLE IF NOT EXISTS forecast_h5_paper_trading (
    id                  BIGSERIAL PRIMARY KEY,
    signal_date         DATE NOT NULL UNIQUE,       -- Monday signal date
    inference_week      INT NOT NULL,
    inference_year      INT NOT NULL,
    direction           INT NOT NULL,               -- Signal direction for this week
    leverage            DOUBLE PRECISION NOT NULL,
    week_pnl_pct        DOUBLE PRECISION NOT NULL,  -- Week PnL (leveraged)
    n_subtrades         INT NOT NULL,
    cumulative_pnl_pct  DOUBLE PRECISION NOT NULL,  -- Running cumulative PnL
    running_da_pct      DOUBLE PRECISION,           -- Running direction accuracy %
    running_da_short_pct DOUBLE PRECISION,           -- Running SHORT DA %
    running_da_long_pct  DOUBLE PRECISION,           -- Running LONG DA %
    running_sharpe      DOUBLE PRECISION,           -- Running annualized Sharpe
    running_max_dd_pct  DOUBLE PRECISION,           -- Running max drawdown
    n_weeks             INT NOT NULL,               -- Total weeks so far
    n_long              INT DEFAULT 0,              -- Total LONG weeks
    n_short             INT DEFAULT 0,              -- Total SHORT weeks
    long_pct_8w         DOUBLE PRECISION,           -- LONG % in last 8 weeks (alarm)
    consecutive_losses  INT DEFAULT 0,              -- Current streak of losing weeks
    circuit_breaker     BOOLEAN DEFAULT FALSE,      -- TRUE if CB triggered
    gate_status         VARCHAR(30),                -- NULL (too early), 'promote', 'keep', 'switch_short', 'discard'
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_h5_pt_week ON forecast_h5_paper_trading (inference_year, inference_week);


-- ---------------------------------------------------------------------------
-- VIEW: v_h5_performance_summary
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW v_h5_performance_summary AS
SELECT
    COUNT(*) AS n_weeks,
    SUM(CASE WHEN direction = 1 THEN 1 ELSE 0 END) AS n_long,
    SUM(CASE WHEN direction = -1 THEN 1 ELSE 0 END) AS n_short,
    ROUND(AVG(week_pnl_pct)::numeric, 4) AS avg_week_pnl_pct,
    ROUND(SUM(week_pnl_pct)::numeric, 4) AS total_pnl_pct,
    ROUND((SUM(CASE WHEN week_pnl_pct > 0 THEN 1 ELSE 0 END)::float
           / NULLIF(COUNT(*), 0) * 100)::numeric, 1) AS win_rate_pct,
    ROUND((SUM(CASE WHEN direction = -1 AND week_pnl_pct > 0 THEN 1 ELSE 0 END)::float
           / NULLIF(SUM(CASE WHEN direction = -1 THEN 1 ELSE 0 END), 0) * 100)::numeric, 1) AS short_wr_pct,
    ROUND((SUM(CASE WHEN direction = 1 AND week_pnl_pct > 0 THEN 1 ELSE 0 END)::float
           / NULLIF(SUM(CASE WHEN direction = 1 THEN 1 ELSE 0 END), 0) * 100)::numeric, 1) AS long_wr_pct,
    MAX(signal_date) AS latest_week,
    MAX(cumulative_pnl_pct) AS peak_pnl_pct,
    MIN(running_max_dd_pct) AS worst_drawdown_pct
FROM forecast_h5_paper_trading;


-- ---------------------------------------------------------------------------
-- VIEW: v_h5_collapse_monitor
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW v_h5_collapse_monitor AS
SELECT
    p.inference_date,
    p.model_id,
    p.predicted_return_pct,
    p.collapse_flag,
    p.collapse_std,
    s.collapse_flag AS signal_collapse
FROM forecast_h5_predictions p
LEFT JOIN forecast_h5_signals s
    ON p.inference_date = s.inference_date
WHERE p.inference_date >= (CURRENT_DATE - INTERVAL '90 days')
ORDER BY p.inference_date DESC, p.model_id;


COMMIT;
