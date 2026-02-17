-- ============================================================================
-- Migration 040: Multi-Pair FX Support
-- ============================================================================
-- Adds index on symbol column and convenience views for multi-pair monitoring.
-- Table name remains usdcop_m5_ohlcv (renaming would break 50+ files).
-- Table already has PRIMARY KEY (time, symbol) — no schema change needed.
--
-- Author: Pedro @ Lean Tech Solutions
-- Date: 2026-02-12
-- ============================================================================

-- Index for symbol-filtered queries (backfill, loader, monitoring)
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol
    ON usdcop_m5_ohlcv (symbol);

-- Composite index for common query pattern: symbol + time range
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
    ON usdcop_m5_ohlcv (symbol, time DESC);

-- Convenience view: latest bar per symbol
CREATE OR REPLACE VIEW fx_latest_bar AS
SELECT
    symbol,
    MAX(time) AS last_bar,
    COUNT(*) AS total_bars,
    MIN(time) AS first_bar
FROM usdcop_m5_ohlcv
GROUP BY symbol;

-- Convenience view: daily bar counts per symbol (for gap monitoring)
CREATE OR REPLACE VIEW fx_daily_bar_counts AS
SELECT
    symbol,
    DATE(time AT TIME ZONE 'America/Bogota') AS trading_date,
    COUNT(*) AS bar_count,
    MIN(time) AS first_bar,
    MAX(time) AS last_bar
FROM usdcop_m5_ohlcv
GROUP BY symbol, DATE(time AT TIME ZONE 'America/Bogota')
ORDER BY symbol, trading_date DESC;
