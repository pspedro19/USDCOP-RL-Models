-- ============================================================================
-- Migration 063: native multi-timeframe OHLCV + native-first 1h/4h wide views
-- Contract: CTR-MKT-CANON-001 (operator 2026-07-22: "cobertura total desde la
--           maxima cantidad de dias posibles con TwelveData")
-- ============================================================================
-- 060 declared the rule: "if a provider ever delivers native 1h, it lands in
-- its own table and WINS in the contract view; the aggregates are the declared
-- fallback, never silently mixed". This migration is that table arriving.
--
--   tf IN ('1h','4h','1month') — daily stays in asset_daily_ohlcv, M5 stays in
--   usdcop_m5_ohlcv; one bar has exactly ONE home per timeframe.
--
-- Provenance probed live against the API (earliest_timestamp, 2026-07-21):
--   USD/COP 1h/4h desde 2019-09 · XAU 2020-01 · SPY 2020-02 · monthly decadas.
-- available_at is stamped at ingestion: for a historic backfill "now" is the
-- honest availability bound — it is when the bar became knowable to THIS system.
-- ============================================================================

CREATE TABLE IF NOT EXISTS asset_native_ohlcv (
    time         TIMESTAMPTZ NOT NULL,
    symbol       TEXT        NOT NULL,
    tf           TEXT        NOT NULL CHECK (tf IN ('1h', '4h', '1month')),
    open         NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC,
    volume       NUMERIC,
    source       TEXT        NOT NULL,
    available_at TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (time, symbol, tf)
);

-- ---------------------------------------------------------------------------
-- Native-first 1h wide: native bar wins, M5 aggregate is the declared fallback.
-- bar_origin says WHICH one you are reading — never silently mixed.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW market_ohlcv_1h_wide AS
WITH grid AS (
    SELECT bar_start_utc FROM market_ohlcv_1h_agg
    UNION SELECT time FROM asset_native_ohlcv WHERE tf = '1h'
)
SELECT
    g.bar_start_utc,
    g.bar_start_utc + interval '1 hour'             AS bar_end_utc,
    g.bar_start_utc AT TIME ZONE 'America/Bogota'   AS timestamp_cot,
    (g.bar_start_utc AT TIME ZONE 'America/Bogota')::date AS session_date_cot,

    COALESCE(n1.open,  a1.open)  AS open_usdcop,
    COALESCE(n1.high,  a1.high)  AS high_usdcop,
    COALESCE(n1.low,   a1.low)   AS low_usdcop,
    COALESCE(n1.close, a1.close) AS close_usdcop,
    COALESCE(n1.volume, a1.volume) AS volume_usdcop,
    COALESCE(n1.source, a1.source) AS source_usdcop,
    CASE WHEN n1.close IS NOT NULL THEN 'native'
         WHEN a1.close IS NOT NULL THEN 'derived_m5' END AS bar_origin_usdcop,
    a1.n_m5_bars AS n_m5_bars_usdcop,
    CASE WHEN n1.close IS NOT NULL THEN 'ok'
         WHEN a1.close IS NULL THEN 'closed_or_missing'
         WHEN a1.n_m5_bars < 12 THEN 'partial' ELSE 'ok' END AS status_usdcop,

    COALESCE(n3.open,  a3.open)  AS open_btcusdt,
    COALESCE(n3.close, a3.close) AS close_btcusdt,
    COALESCE(n3.volume, a3.volume) AS volume_btcusdt,
    CASE WHEN n3.close IS NOT NULL THEN 'native'
         WHEN a3.close IS NOT NULL THEN 'derived_m5' END AS bar_origin_btcusdt,
    a3.n_m5_bars AS n_m5_bars_btcusdt,
    CASE WHEN n3.close IS NOT NULL THEN 'ok'
         WHEN a3.close IS NULL THEN 'missing'
         WHEN a3.n_m5_bars < 12 THEN 'partial' ELSE 'ok' END AS status_btcusdt,

    COALESCE(n2.open,  a2.open)  AS open_xauusd,
    COALESCE(n2.close, a2.close) AS close_xauusd,
    CASE WHEN n2.close IS NOT NULL THEN 'native'
         WHEN a2.close IS NOT NULL THEN 'derived_m5' END AS bar_origin_xauusd,
    a2.n_m5_bars AS n_m5_bars_xauusd,
    CASE WHEN n2.close IS NOT NULL THEN 'ok'
         WHEN a2.close IS NULL THEN 'closed_or_missing'
         WHEN a2.n_m5_bars < 12 THEN 'partial' ELSE 'ok' END AS status_xauusd,

    'no_native_data'::text AS status_spx500

FROM grid g
LEFT JOIN market_ohlcv_1h_agg a1 ON a1.symbol='USD/COP'  AND a1.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_1h_agg a2 ON a2.symbol='XAU/USD'  AND a2.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_1h_agg a3 ON a3.symbol='BTC/USDT' AND a3.bar_start_utc=g.bar_start_utc
LEFT JOIN asset_native_ohlcv n1 ON n1.tf='1h' AND n1.symbol='USD/COP'  AND n1.time=g.bar_start_utc
LEFT JOIN asset_native_ohlcv n2 ON n2.tf='1h' AND n2.symbol='XAU/USD'  AND n2.time=g.bar_start_utc
LEFT JOIN asset_native_ohlcv n3 ON n3.tf='1h' AND n3.symbol='BTC/USDT' AND n3.time=g.bar_start_utc;

CREATE OR REPLACE VIEW market_ohlcv_4h_wide AS
WITH grid AS (
    SELECT bar_start_utc FROM market_ohlcv_4h_agg
    UNION SELECT time FROM asset_native_ohlcv WHERE tf = '4h'
)
SELECT
    g.bar_start_utc,
    g.bar_start_utc + interval '4 hours'            AS bar_end_utc,
    g.bar_start_utc AT TIME ZONE 'America/Bogota'   AS timestamp_cot,
    (g.bar_start_utc AT TIME ZONE 'America/Bogota')::date AS session_date_cot,
    COALESCE(n1.open,  a1.open)  AS open_usdcop,
    COALESCE(n1.close, a1.close) AS close_usdcop,
    a1.n_m5_bars AS n_m5_bars_usdcop,
    CASE WHEN n1.close IS NOT NULL THEN 'native'
         WHEN a1.close IS NOT NULL THEN 'derived_m5' END AS bar_origin_usdcop,
    CASE WHEN n1.close IS NOT NULL THEN 'ok'
         WHEN a1.close IS NULL THEN 'closed_or_missing'
         WHEN a1.n_m5_bars < 48 THEN 'partial' ELSE 'ok' END AS status_usdcop,
    COALESCE(n3.open,  a3.open)  AS open_btcusdt,
    COALESCE(n3.close, a3.close) AS close_btcusdt,
    a3.n_m5_bars AS n_m5_bars_btcusdt,
    CASE WHEN n3.close IS NOT NULL THEN 'native'
         WHEN a3.close IS NOT NULL THEN 'derived_m5' END AS bar_origin_btcusdt,
    CASE WHEN n3.close IS NOT NULL THEN 'ok'
         WHEN a3.close IS NULL THEN 'missing'
         WHEN a3.n_m5_bars < 48 THEN 'partial' ELSE 'ok' END AS status_btcusdt,
    COALESCE(n2.open,  a2.open)  AS open_xauusd,
    COALESCE(n2.close, a2.close) AS close_xauusd,
    a2.n_m5_bars AS n_m5_bars_xauusd,
    CASE WHEN n2.close IS NOT NULL THEN 'native'
         WHEN a2.close IS NOT NULL THEN 'derived_m5' END AS bar_origin_xauusd,
    CASE WHEN n2.close IS NOT NULL THEN 'ok'
         WHEN a2.close IS NULL THEN 'closed_or_missing'
         WHEN a2.n_m5_bars < 48 THEN 'partial' ELSE 'ok' END AS status_xauusd,
    'no_native_data'::text AS status_spx500
FROM grid g
LEFT JOIN market_ohlcv_4h_agg a1 ON a1.symbol='USD/COP'  AND a1.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_4h_agg a2 ON a2.symbol='XAU/USD'  AND a2.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_4h_agg a3 ON a3.symbol='BTC/USDT' AND a3.bar_start_utc=g.bar_start_utc
LEFT JOIN asset_native_ohlcv n1 ON n1.tf='4h' AND n1.symbol='USD/COP'  AND n1.time=g.bar_start_utc
LEFT JOIN asset_native_ohlcv n2 ON n2.tf='4h' AND n2.symbol='XAU/USD'  AND n2.time=g.bar_start_utc
LEFT JOIN asset_native_ohlcv n3 ON n3.tf='4h' AND n3.symbol='BTC/USDT' AND n3.time=g.bar_start_utc;
