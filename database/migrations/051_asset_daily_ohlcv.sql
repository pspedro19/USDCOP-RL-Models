-- Migration 051: Multi-asset DAILY OHLCV table (scalable, additive)
-- ============================================================================
-- Purpose:
--   Provide a scalable, multi-asset home for DAILY bars (deep history 2004+),
--   keyed by (time, symbol) exactly like the existing 5-min table usdcop_m5_ohlcv.
--   Gold (XAU/USD) daily lives here; BTC/USD and any future asset plug in by symbol.
--
-- Scalability decision (aligned to sdd-multi-asset-onboarding + SPEC-12):
--   * 5-MIN bars reuse the existing multi-pair table `usdcop_m5_ohlcv`
--     (PK (time, symbol), migration 040). XAU/USD inserts there as symbol='XAU/USD'.
--     NO new 5-min table — that would be a silo.
--   * DAILY bars had no multi-asset table (only file-based usdcop_daily seed).
--     This migration adds ONE generic daily table for ALL assets.
--   * Timestamps are TIMESTAMPTZ (instant-based): COP stored at COT instants,
--     Gold at UTC instants — both unambiguous, mixable in one table.
--   * Idempotent UPSERT on (time, symbol) — safe to re-run ingestion.
--
-- Additive: does not touch usdcop_m5_ohlcv or any existing object.
-- Date: 2026-07-03  ·  Contract: CTR-L0-ASSET-DAILY-001
-- ============================================================================

CREATE TABLE IF NOT EXISTS asset_daily_ohlcv (
    time        TIMESTAMPTZ       NOT NULL,   -- daily bar timestamp (asset-native close instant)
    symbol      VARCHAR(20)       NOT NULL,   -- 'XAU/USD', 'USD/COP', 'BTC/USD', ...
    open        DOUBLE PRECISION  NOT NULL,
    high        DOUBLE PRECISION  NOT NULL,
    low         DOUBLE PRECISION  NOT NULL,
    close       DOUBLE PRECISION  NOT NULL,
    volume      DOUBLE PRECISION  DEFAULT 0,  -- 0 for metals/FX without volume
    source      VARCHAR(50)       DEFAULT 'twelvedata',  -- provenance: twelvedata | investing | seed
    updated_at  TIMESTAMPTZ       DEFAULT NOW(),
    PRIMARY KEY (time, symbol)
);

-- Multi-asset access patterns
CREATE INDEX IF NOT EXISTS idx_asset_daily_symbol      ON asset_daily_ohlcv (symbol);
CREATE INDEX IF NOT EXISTS idx_asset_daily_symbol_time ON asset_daily_ohlcv (symbol, time DESC);

-- TimescaleDB hypertable (best-effort; ignored if extension/policy unavailable)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('asset_daily_ohlcv', 'time',
                                  if_not_exists => TRUE, migrate_data => TRUE);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'asset_daily_ohlcv: hypertable step skipped (%).', SQLERRM;
END $$;

-- Monitoring view: latest daily bar + coverage per asset
CREATE OR REPLACE VIEW asset_daily_coverage AS
SELECT symbol,
       COUNT(*)             AS rows,
       MIN(time)            AS earliest,
       MAX(time)            AS latest,
       MAX(source)          AS last_source
FROM asset_daily_ohlcv
GROUP BY symbol
ORDER BY symbol;

COMMENT ON TABLE asset_daily_ohlcv IS
    'Multi-asset daily OHLCV (deep history). Keyed by (time,symbol). Gold/BTC/... plug in by symbol. 5-min bars stay in usdcop_m5_ohlcv.';
