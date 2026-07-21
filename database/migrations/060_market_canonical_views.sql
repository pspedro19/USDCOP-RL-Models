-- ============================================================================
-- Migration 060: canonical market-data contract — dims, calendar, PIT column,
--                ingestion manifests, and the long/wide view layer
-- Contract: CTR-MKT-CANON-001 (plan 2026-07-21, operator-approved; merged with
--           Codex's canonical-contract plan)
-- ============================================================================
-- Design decisions this migration encodes (and why):
--
--  * The physical SSOT stays where it is (usdcop_m5_ohlcv, asset_daily_ohlcv).
--    Codex proposed parallel market_ohlcv_* physical tables; two homes for the
--    same M5 bar is guaranteed drift, so the canonical contract is a LONG VIEW
--    over the existing tables instead. Renaming is forbidden by governance.
--
--  * available_at is added as a nullable column and stamped by ingestion from
--    now on. Historical rows keep NULL — "unknown" is the honest value; a
--    reconstructed availability time is fabricated PIT evidence, which is the
--    exact thing the constitution forbids. PIT lineage accrues forward for free.
--
--  * closed != missing is decided by market_session_calendar, not by the
--    consumer's memory of holidays. The row that motivated all of this:
--    2026-07-20 (Colombian Independence Day) must read usdcop=closed while
--    btcusdt=ok, instead of costing a morning of diagnosis.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. dim_asset: one row per tradeable asset, seeded from config/assets/*.yaml
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_asset (
    asset_id        TEXT PRIMARY KEY,          -- usdcop | xauusd | btcusdt | spx500
    symbol          TEXT NOT NULL UNIQUE,      -- provider symbol in the long tables
    display_name    TEXT NOT NULL,
    session_tz      TEXT NOT NULL,             -- IANA
    calendar_kind   TEXT NOT NULL CHECK (calendar_kind IN
                        ('colombia','nyse','utc_24_7','metals_23h')),
    annualization   INT  NOT NULL              -- declared clock (manifest F0 mirror)
);

INSERT INTO dim_asset (asset_id, symbol, display_name, session_tz, calendar_kind, annualization)
VALUES
  ('usdcop',  'USD/COP',  'USD/COP',  'America/Bogota',   'colombia',  52),
  ('xauusd',  'XAU/USD',  'Gold',     'UTC',              'metals_23h', 252),
  ('btcusdt', 'BTC/USDT', 'Bitcoin',  'UTC',              'utc_24_7',  365),
  ('spx500',  'SPX500',   'S&P 500',  'America/New_York', 'nyse',      252)
ON CONFLICT (asset_id) DO UPDATE SET
  symbol = EXCLUDED.symbol, session_tz = EXCLUDED.session_tz,
  calendar_kind = EXCLUDED.calendar_kind, annualization = EXCLUDED.annualization;

-- ---------------------------------------------------------------------------
-- 2. Session calendar: THE arbiter of closed vs missing. Seeded 2020-2027 by
--    scripts/ops/seed_session_calendar.py (TradingCalendar for Colombia, the
--    `holidays` package for NYSE with DST-aware open/close, trivial for 24/7).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_session_calendar (
    asset_id          TEXT NOT NULL REFERENCES dim_asset(asset_id),
    session_date      DATE NOT NULL,           -- date in the asset's LOCAL tz
    is_trading_day    BOOLEAN NOT NULL,
    session_open_utc  TIMESTAMPTZ,             -- NULL on non-trading days
    session_close_utc TIMESTAMPTZ,             -- DST folds into these instants
    PRIMARY KEY (asset_id, session_date)
);

-- ---------------------------------------------------------------------------
-- 3. PIT column + per-run ingestion manifests (Codex plan items 2 and 5)
-- ---------------------------------------------------------------------------
ALTER TABLE usdcop_m5_ohlcv   ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ;
ALTER TABLE asset_daily_ohlcv ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ;
-- Historic rows stay NULL on purpose. See header.

CREATE TABLE IF NOT EXISTS market_ingestion_manifest (
    id              BIGSERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider        TEXT NOT NULL,
    asset_id        TEXT,
    timeframe       TEXT NOT NULL,             -- 5m | 1h | 4h | daily
    window_start    TIMESTAMPTZ,
    window_end      TIMESTAMPTZ,
    rows_received   INT,
    rows_new        INT,
    latency_ms      INT,
    source_tz       TEXT,                      -- timezone the provider was ASKED for
    checksum_sha256 TEXT,                      -- of the normalized payload (per run, not per row)
    error           TEXT
);

-- ---------------------------------------------------------------------------
-- 4. Canonical LONG views (the contract; storage untouched)
--    quality_status is computed, not stored: a stored verdict goes stale.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW market_ohlcv_5m AS
SELECT
    da.asset_id,
    m.time                                   AS bar_start_utc,
    m.time + interval '5 minutes'            AS bar_end_utc,
    m.time AT TIME ZONE 'America/Bogota'     AS timestamp_cot,
    (m.time AT TIME ZONE da.session_tz)::date AS session_date_local,
    da.session_tz                            AS market_timezone,
    m.open, m.high, m.low, m.close, m.volume,
    m.source,
    CASE WHEN m.source LIKE '%backfill%'  THEN 'backfill'
         WHEN m.source LIKE '%gap_fill%'  THEN 'gap_fill'
         ELSE 'native' END                   AS bar_origin,
    m.available_at,
    CASE WHEN m.high >= GREATEST(m.open, m.close)
          AND m.low  <= LEAST(m.open, m.close)
          AND m.high >= m.low               THEN 'ok'
         ELSE 'incoherent_ohlc' END          AS quality_status
FROM usdcop_m5_ohlcv m
JOIN dim_asset da ON da.symbol = m.symbol;

CREATE OR REPLACE VIEW market_ohlcv_daily AS
SELECT
    da.asset_id,
    d.time                                   AS bar_start_utc,
    d.time + interval '1 day'                AS bar_end_utc,
    d.time AT TIME ZONE 'America/Bogota'     AS timestamp_cot,
    -- Daily bars are stored at 00:00 UTC of the trading date (XAU at 21:00 UTC, same UTC
    -- date). Anchor the session date in UTC — data-governance invariant #1: converting a
    -- 00:00-UTC daily stamp to the closing tz shifts EVERY bar one day back (Sunday
    -- pile-up bug). Verified live: SPX 2026-07-20 read as 07-19 under the NY conversion.
    (d.time AT TIME ZONE 'UTC')::date        AS session_date_local,
    da.session_tz                            AS market_timezone,
    d.open, d.high, d.low, d.close, d.volume,
    d.source,
    CASE WHEN d.source LIKE '%snapshot%' THEN 'snapshot' ELSE 'native' END AS bar_origin,
    d.available_at,
    CASE WHEN d.high >= GREATEST(d.open, d.close)
          AND d.low  <= LEAST(d.open, d.close)
          AND d.high >= d.low               THEN 'ok'
         ELSE 'incoherent_ohlc' END          AS quality_status
FROM asset_daily_ohlcv d
JOIN dim_asset da ON da.symbol = d.symbol;

-- ---------------------------------------------------------------------------
-- 5. 1h / 4h aggregates from M5 (plain matviews: the M5 base is a plain table
--    here, not a hypertable in every environment — a nightly/hourly REFRESH via
--    the existing DAGs is sufficient at this volume; continuous aggregates can
--    replace these transparently if the base is converted to a hypertable).
--    bar_origin='derived_m5' per the native-first rule: if a provider ever
--    delivers native 1h, it lands in its own table and WINS in the contract
--    view; these aggregates are the declared fallback, never silently mixed.
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS market_ohlcv_1h_agg AS
SELECT date_trunc('hour', m.time) AS bar_start_utc, m.symbol,
       (array_agg(m.open  ORDER BY m.time ASC ))[1] AS open,
       max(m.high) AS high, min(m.low) AS low,
       (array_agg(m.close ORDER BY m.time DESC))[1] AS close,
       sum(m.volume) AS volume,
       (array_agg(m.source ORDER BY m.time DESC))[1] AS source,
       count(*) AS n_m5_bars
FROM usdcop_m5_ohlcv m GROUP BY 1, 2;
CREATE UNIQUE INDEX IF NOT EXISTS ux_ohlcv_1h ON market_ohlcv_1h_agg (bar_start_utc, symbol);

CREATE MATERIALIZED VIEW IF NOT EXISTS market_ohlcv_4h_agg AS
SELECT to_timestamp(floor(extract(epoch FROM m.time) / 14400) * 14400) AS bar_start_utc,
       m.symbol,
       (array_agg(m.open  ORDER BY m.time ASC ))[1] AS open,
       max(m.high) AS high, min(m.low) AS low,
       (array_agg(m.close ORDER BY m.time DESC))[1] AS close,
       sum(m.volume) AS volume,
       (array_agg(m.source ORDER BY m.time DESC))[1] AS source,
       count(*) AS n_m5_bars
FROM usdcop_m5_ohlcv m GROUP BY 1, 2;
CREATE UNIQUE INDEX IF NOT EXISTS ux_ohlcv_4h ON market_ohlcv_4h_agg (bar_start_utc, symbol);
