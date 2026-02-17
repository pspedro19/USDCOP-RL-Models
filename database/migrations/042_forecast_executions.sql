-- Migration 042: Forecast Execution Tracking (Smart Executor)
-- Created: 2026-02-15
--
-- Purpose: Track intraday execution of daily forecasting signals via the
-- Smart Executor (trailing stop + broker adapter). One row per signal_date,
-- updated as 5-min bars are processed during the T+1 session.
--
-- Architecture:
--   forecast_vol_targeting_signals (041, parent)
--       │ FK: signal_date
--       ▼
--   forecast_executions (this table)
--       │
--       ▼
--   v_execution_performance (aggregate view)
--
-- Contract: SMART-EXECUTOR-V1

-- =============================================================================
-- TABLE: FORECAST EXECUTIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS forecast_executions (
    id                  BIGSERIAL PRIMARY KEY,

    -- Signal identity (FK to forecast_vol_targeting_signals)
    signal_date         DATE NOT NULL UNIQUE,

    -- Execution state
    status              VARCHAR(20) NOT NULL DEFAULT 'idle',
    direction           SMALLINT NOT NULL,
    leverage            DOUBLE PRECISION NOT NULL,

    -- Entry
    entry_price         DOUBLE PRECISION,
    entry_timestamp     TIMESTAMPTZ,

    -- Exit
    exit_price          DOUBLE PRECISION,
    exit_timestamp      TIMESTAMPTZ,
    exit_reason         VARCHAR(30),       -- trailing_stop, hard_stop, session_close

    -- Trailing stop state (persisted for stateless reconstruction)
    peak_price          DOUBLE PRECISION,
    trailing_state      VARCHAR(20) DEFAULT 'waiting',
    bar_count           INTEGER DEFAULT 0,

    -- PnL
    pnl_pct             DOUBLE PRECISION,           -- leveraged PnL %
    pnl_unleveraged_pct DOUBLE PRECISION,           -- raw PnL %

    -- Comparison fields (filled by L6 monitor after session close)
    daily_close_price   DOUBLE PRECISION,           -- close[T+1]
    hold_to_close_pnl   DOUBLE PRECISION,           -- hold strategy PnL
    execution_alpha_pct DOUBLE PRECISION,           -- pnl_pct - hold_to_close_pnl

    -- Config snapshot for audit
    config_version      VARCHAR(30) DEFAULT 'smart_executor_v1',
    activation_pct      DOUBLE PRECISION,
    trail_pct           DOUBLE PRECISION,
    hard_stop_pct       DOUBLE PRECISION,

    -- Timestamps
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_exec_direction CHECK (direction IN (-1, 1)),
    CONSTRAINT chk_exec_status CHECK (status IN ('idle', 'positioned', 'monitoring', 'closed', 'error')),
    CONSTRAINT chk_exec_trailing_state CHECK (trailing_state IN ('waiting', 'active', 'triggered', 'expired')),
    CONSTRAINT chk_exec_leverage_positive CHECK (leverage > 0)
);

COMMENT ON TABLE forecast_executions IS
    'Intraday execution tracking for daily forecasting signals via Smart Executor.';
COMMENT ON COLUMN forecast_executions.signal_date IS
    'Date of the signal (day T). Execution happens during T+1 session.';
COMMENT ON COLUMN forecast_executions.trailing_state IS
    'TrailingStopTracker state: waiting -> active -> triggered/expired. Persisted for stateless reconstruction.';
COMMENT ON COLUMN forecast_executions.peak_price IS
    'Best favorable price seen since entry. Used to reconstruct TrailingStopTracker.';
COMMENT ON COLUMN forecast_executions.execution_alpha_pct IS
    'Execution alpha = smart executor PnL - hold-to-close PnL. Positive = trailing stop added value.';

-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_exec_signal_date
    ON forecast_executions (signal_date DESC);

-- Partial index for active positions (fast lookup for monitor task)
CREATE INDEX IF NOT EXISTS idx_exec_status_active
    ON forecast_executions (status)
    WHERE status IN ('positioned', 'monitoring');

-- =============================================================================
-- FK TO SIGNALS TABLE
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE constraint_name = 'fk_exec_signal') THEN
        ALTER TABLE forecast_executions
            ADD CONSTRAINT fk_exec_signal
            FOREIGN KEY (signal_date)
            REFERENCES forecast_vol_targeting_signals(signal_date)
            ON DELETE CASCADE;
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not add FK constraint fk_exec_signal: %', SQLERRM;
END $$;

-- =============================================================================
-- TRIGGER: Auto-update updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_exec_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_exec_updated_at ON forecast_executions;
CREATE TRIGGER trg_exec_updated_at
    BEFORE UPDATE ON forecast_executions
    FOR EACH ROW
    EXECUTE FUNCTION update_exec_timestamp();

-- =============================================================================
-- TRIGGER: Notify on execution closed
-- =============================================================================

CREATE OR REPLACE FUNCTION notify_execution_closed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'closed' AND (OLD.status IS NULL OR OLD.status != 'closed') THEN
        PERFORM pg_notify(
            'execution_closed',
            json_build_object(
                'signal_date', NEW.signal_date,
                'direction', NEW.direction,
                'exit_reason', NEW.exit_reason,
                'pnl_pct', NEW.pnl_pct,
                'bar_count', NEW.bar_count
            )::text
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_notify_exec_closed ON forecast_executions;
CREATE TRIGGER trg_notify_exec_closed
    AFTER INSERT OR UPDATE ON forecast_executions
    FOR EACH ROW
    EXECUTE FUNCTION notify_execution_closed();

-- =============================================================================
-- VIEW: Execution Performance (aggregate)
-- =============================================================================

CREATE OR REPLACE VIEW v_execution_performance AS
WITH closed AS (
    SELECT * FROM forecast_executions WHERE status = 'closed'
)
SELECT
    COUNT(*) AS total_executions,

    -- Exit reason breakdown
    COUNT(*) FILTER (WHERE exit_reason = 'trailing_stop') AS trailing_stop_exits,
    COUNT(*) FILTER (WHERE exit_reason = 'hard_stop') AS hard_stop_exits,
    COUNT(*) FILTER (WHERE exit_reason = 'session_close') AS session_close_exits,

    -- PnL
    ROUND(AVG(pnl_pct)::NUMERIC * 100, 4) AS avg_pnl_pct,
    ROUND(STDDEV_SAMP(pnl_pct)::NUMERIC * 100, 4) AS std_pnl_pct,
    ROUND(SUM(pnl_pct)::NUMERIC * 100, 4) AS total_pnl_pct,

    -- Win rate
    COUNT(*) FILTER (WHERE pnl_pct > 0) AS wins,
    COUNT(*) FILTER (WHERE pnl_pct <= 0) AS losses,
    CASE WHEN COUNT(*) > 0
        THEN ROUND(COUNT(*) FILTER (WHERE pnl_pct > 0)::NUMERIC / COUNT(*) * 100, 2)
        ELSE 0
    END AS win_rate_pct,

    -- Execution alpha
    ROUND(AVG(execution_alpha_pct)::NUMERIC * 100, 4) AS avg_alpha_pct,
    COUNT(*) FILTER (WHERE execution_alpha_pct > 0) AS alpha_positive_count,
    CASE WHEN COUNT(*) FILTER (WHERE execution_alpha_pct IS NOT NULL) > 0
        THEN ROUND(
            COUNT(*) FILTER (WHERE execution_alpha_pct > 0)::NUMERIC /
            COUNT(*) FILTER (WHERE execution_alpha_pct IS NOT NULL) * 100, 2
        )
        ELSE 0
    END AS alpha_hit_rate_pct,

    -- Bars
    ROUND(AVG(bar_count)::NUMERIC, 1) AS avg_bars_per_trade,

    -- Trailing stop activation rate
    CASE WHEN COUNT(*) > 0
        THEN ROUND(
            COUNT(*) FILTER (WHERE exit_reason = 'trailing_stop')::NUMERIC / COUNT(*) * 100, 2
        )
        ELSE 0
    END AS activation_rate_pct,

    -- Date range
    MIN(signal_date) AS first_date,
    MAX(signal_date) AS last_date

FROM closed;

COMMENT ON VIEW v_execution_performance IS
    'Aggregated Smart Executor performance: PnL, win rate, alpha, activation rate.';

-- =============================================================================
-- CLEANUP FUNCTION
-- =============================================================================

CREATE OR REPLACE FUNCTION cleanup_old_executions(days_to_keep INTEGER DEFAULT 365)
RETURNS BIGINT AS $$
DECLARE
    deleted_count BIGINT;
BEGIN
    DELETE FROM forecast_executions
    WHERE signal_date < CURRENT_DATE - days_to_keep
      AND status = 'closed';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_executions(INTEGER) IS
    'Remove closed executions older than N days. Only deletes closed rows for safety.';
