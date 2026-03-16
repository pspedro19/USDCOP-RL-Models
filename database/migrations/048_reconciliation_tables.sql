-- =============================================================================
-- Migration 048: Reconciliation Tables
-- =============================================================================
-- Tracks reconciliation between internal signals/executions and exchange fills.
-- Used by the daily reconciliation DAG to detect missed fills, slippage, and
-- quantity mismatches.
--
-- Tables:
--   reconciliation_runs    — One row per reconciliation execution
--   reconciliation_items   — Per-signal comparison results
--
-- Author: Trading Team
-- Date: 2026-03-15
-- =============================================================================

-- Reconciliation run summary
CREATE TABLE IF NOT EXISTS reconciliation_runs (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE NOT NULL,
    pipeline            VARCHAR(20) NOT NULL,          -- 'h1' or 'h5'
    exchange            VARCHAR(20) NOT NULL DEFAULT 'mexc',
    status              VARCHAR(20) NOT NULL DEFAULT 'running',  -- running, completed, failed
    signals_checked     INTEGER NOT NULL DEFAULT 0,
    matches             INTEGER NOT NULL DEFAULT 0,
    discrepancies       INTEGER NOT NULL DEFAULT 0,
    missed_fills        INTEGER NOT NULL DEFAULT 0,
    extra_fills         INTEGER NOT NULL DEFAULT 0,
    max_slippage_bps    DOUBLE PRECISION,
    avg_slippage_bps    DOUBLE PRECISION,
    notes               TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_recon_run UNIQUE (run_date, pipeline, exchange)
);

-- Per-signal reconciliation detail
CREATE TABLE IF NOT EXISTS reconciliation_items (
    id                  SERIAL PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES reconciliation_runs(id) ON DELETE CASCADE,
    signal_date         DATE NOT NULL,
    pipeline            VARCHAR(20) NOT NULL,
    direction           VARCHAR(10),                    -- 'LONG', 'SHORT'

    -- Internal (from DB signals/executions)
    internal_entry_price    DOUBLE PRECISION,
    internal_exit_price     DOUBLE PRECISION,
    internal_leverage       DOUBLE PRECISION,
    internal_pnl_pct        DOUBLE PRECISION,
    internal_exit_reason    VARCHAR(30),

    -- Exchange (from MEXC/Binance fills)
    exchange_entry_price    DOUBLE PRECISION,
    exchange_exit_price     DOUBLE PRECISION,
    exchange_quantity        DOUBLE PRECISION,
    exchange_commission     DOUBLE PRECISION,

    -- Comparison
    match_status        VARCHAR(20) NOT NULL,           -- 'match', 'slippage', 'missed_fill', 'extra_fill', 'qty_mismatch'
    entry_slippage_bps  DOUBLE PRECISION,
    exit_slippage_bps   DOUBLE PRECISION,
    pnl_diff_pct        DOUBLE PRECISION,
    notes               TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_recon_items_run ON reconciliation_items(run_id);
CREATE INDEX IF NOT EXISTS idx_recon_items_status ON reconciliation_items(match_status);
CREATE INDEX IF NOT EXISTS idx_recon_runs_date ON reconciliation_runs(run_date);
