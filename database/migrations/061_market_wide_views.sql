-- ============================================================================
-- Migration 061: the four wide consumption views (5m / 1h / 4h / daily)
-- Contract: CTR-MKT-CANON-001 — wide is a VIEW, never a table: closed-market
-- NULLs are not stored, staleness is never frozen, and adding an asset is an
-- ALTER VIEW instead of a backfill.
-- ============================================================================
-- status_<asset> semantics (the whole point of this layer):
--   closed    calendar says no session -> NULL is legitimate
--   ok        bar present and coherent
--   partial   aggregate built from fewer M5 bars than the session implies
--   missing   calendar says the session was OPEN and the bar is absent  << incident
--   no_native_data  the asset has no data at this granularity at all (SPX intraday)
-- staleness_seconds is computed at read time; a materialized staleness lies by
-- construction.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- DAILY (the one the operator asked to see spelled out), with the 17 verified
-- CLEAN macros — each in TWO forms: as-of (reporting) and _t1 (the ONLY form a
-- model may consume; T-1 rule). USDMXN/USDCLP were excluded while corrupted x10^4;
-- REINSTATED 2026-07-22 after the twelvedata repair (guard is now a hard test).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW market_ohlcv_daily_wide AS
WITH grid AS (
    -- session_date_local is UTC-anchored in the long view (governance inv. #1);
    -- never cast bar_start_utc::date here — that cast obeys the CLIENT's TimeZone GUC.
    SELECT DISTINCT session_date_local AS d FROM market_ohlcv_daily
), px AS (
    SELECT * FROM market_ohlcv_daily
), cal AS (
    SELECT * FROM market_session_calendar
), firsts AS (
    -- Before an asset's first ingested bar there is no incident to report:
    -- BTC in 2004 is 'no_native_data' (the grid reaches 2004 via XAU), not 'missing'.
    SELECT asset_id, min(session_date_local) AS first_bar FROM market_ohlcv_daily GROUP BY asset_id
), macro AS (
    SELECT fecha,
           fxrt_index_dxy_usa_d_dxy          AS dxy,
           volt_vix_usa_d_vix                AS vix,
           crsk_spread_embi_col_d_embi       AS embi,
           comm_oil_brent_glb_d_brent        AS brent,
           comm_oil_wti_glb_d_wti            AS wti,
           comm_metal_gold_glb_d_gold        AS gold,
           comm_agri_coffee_glb_d_coffee     AS coffee,
           eqty_index_colcap_col_d_colcap    AS colcap,
           finc_bond_yield10y_usa_d_ust10y   AS ust10y,
           finc_bond_yield2y_usa_d_dgs2      AS dgs2,
           finc_bond_yield5y_col_d_col5y     AS col5y,
           finc_bond_yield10y_col_d_col10y   AS col10y,
           finc_rate_ibr_overnight_col_d_ibr AS ibr,
           polr_policy_rate_col_d_tpm        AS tpm,
           polr_prime_rate_usa_d_prime       AS prime,
           fxrt_spot_usdcop_col_d_usdcop     AS usdcop_spot,
           -- reincorporadas 2026-07-22: corrupcion x10^4 reparada (twelvedata primaria,
           -- 111 filas corregidas + 147 gaps rellenados; guard duro test_macro_clean_fx_scale)
           fxrt_spot_usdmxn_mex_d_usdmxn     AS usdmxn,
           fxrt_spot_usdclp_chl_d_usdclp     AS usdclp
    FROM macro_indicators_daily
)
SELECT
    -- Pin midnight-UTC instants explicitly: date::timestamptz obeys the session
    -- TimeZone GUC and would move the key per client.
    (g.d::timestamp AT TIME ZONE 'UTC')                AS bar_start_utc,
    ((g.d + 1)::timestamp AT TIME ZONE 'UTC')          AS bar_end_utc,
    g.d::timestamp                                     AS timestamp_cot,
    g.d                                                AS session_date_cot,
    -- row-level status: worst per-asset status among assets whose market was open.
    -- 'pending' = the session date is today-or-future in UTC; the daily bar cannot
    -- exist yet, so its absence is NOT an incident (BTC 07-21 before 00:00 UTC+1).
    (SELECT CASE
        WHEN bool_or(s.st = 'missing') THEN 'missing'
        WHEN bool_or(s.st = 'pending') THEN 'pending'
        WHEN bool_or(s.st = 'partial') THEN 'partial'
        ELSE 'ok' END
     FROM (VALUES
        (CASE WHEN g.d < f1.first_bar THEN 'closed'
              WHEN c1.is_trading_day IS NOT TRUE THEN 'closed'
              WHEN p1.close IS NOT NULL THEN 'ok'
              WHEN g.d >= (now() AT TIME ZONE 'UTC')::date THEN 'pending'
              ELSE 'missing' END),
        (CASE WHEN g.d < f2.first_bar THEN 'closed'
              WHEN extract(dow FROM g.d) = 6 THEN 'closed'
              WHEN p2.close IS NOT NULL THEN 'ok'
              WHEN g.d >= (now() AT TIME ZONE 'UTC')::date THEN 'pending'
              ELSE 'missing' END),
        (CASE WHEN g.d < f3.first_bar THEN 'closed'
              WHEN p3.close IS NOT NULL THEN 'ok'
              WHEN g.d >= (now() AT TIME ZONE 'UTC')::date THEN 'pending'
              ELSE 'missing' END),
        (CASE WHEN g.d < f4.first_bar THEN 'closed'
              WHEN c4.is_trading_day IS NOT TRUE THEN 'closed'
              WHEN p4.close IS NOT NULL THEN 'ok'
              WHEN g.d >= (now() AT TIME ZONE 'UTC')::date THEN 'pending'
              ELSE 'missing' END)
     ) AS s(st) WHERE s.st <> 'closed')                AS availability_status,

    -- USDCOP -----------------------------------------------------------------
    p1.open  AS open_usdcop,  p1.high AS high_usdcop, p1.low AS low_usdcop,
    p1.close AS close_usdcop, p1.volume AS volume_usdcop,
    p1.source AS source_usdcop, p1.bar_origin AS bar_origin_usdcop,
    COALESCE(c1.is_trading_day, false) AS is_session_bar_usdcop,
    EXTRACT(epoch FROM now() - p1.bar_end_utc)::bigint AS staleness_seconds_usdcop,
    CASE WHEN g.d < f1.first_bar THEN 'no_native_data'
         WHEN c1.is_trading_day IS NOT TRUE THEN
              CASE WHEN p1.close IS NULL THEN 'closed' ELSE 'off_session' END
         WHEN p1.close IS NULL THEN
              CASE WHEN g.d >= (now() AT TIME ZONE 'UTC')::date THEN 'pending' ELSE 'missing' END
         WHEN p1.quality_status <> 'ok' THEN 'incoherent'
         ELSE 'ok' END AS status_usdcop,

    -- XAUUSD -----------------------------------------------------------------
    p2.open  AS open_xauusd,  p2.high AS high_xauusd, p2.low AS low_xauusd,
    p2.close AS close_xauusd, p2.volume AS volume_xauusd,
    p2.source AS source_xauusd, p2.bar_origin AS bar_origin_xauusd,
    (p2.close IS NOT NULL) AS is_session_bar_xauusd,
    EXTRACT(epoch FROM now() - p2.bar_end_utc)::bigint AS staleness_seconds_xauusd,
    CASE WHEN g.d < f2.first_bar THEN 'no_native_data'
         WHEN extract(dow FROM g.d) = 6 THEN
              CASE WHEN p2.close IS NULL THEN 'closed' ELSE 'off_session' END
         WHEN p2.close IS NULL THEN
              CASE WHEN g.d >= (now() AT TIME ZONE 'UTC')::date THEN 'pending' ELSE 'missing' END
         WHEN p2.quality_status <> 'ok' THEN 'incoherent'
         ELSE 'ok' END AS status_xauusd,

    -- BTCUSDT (24/7: never 'closed') ------------------------------------------
    p3.open  AS open_btcusdt,  p3.high AS high_btcusdt, p3.low AS low_btcusdt,
    p3.close AS close_btcusdt, p3.volume AS volume_btcusdt,
    p3.source AS source_btcusdt, p3.bar_origin AS bar_origin_btcusdt,
    true AS is_session_bar_btcusdt,
    EXTRACT(epoch FROM now() - p3.bar_end_utc)::bigint AS staleness_seconds_btcusdt,
    CASE WHEN g.d < f3.first_bar THEN 'no_native_data'
         WHEN p3.close IS NULL THEN
              CASE WHEN g.d >= (now() AT TIME ZONE 'UTC')::date THEN 'pending' ELSE 'missing' END
         WHEN p3.quality_status <> 'ok' THEN 'incoherent'
         ELSE 'ok' END AS status_btcusdt,

    -- SP500 (NYSE calendar; close = adj_close total-return, SDD-000 §4) --------
    p4.open  AS open_spx500,  p4.high AS high_spx500, p4.low AS low_spx500,
    p4.close AS close_spx500, p4.volume AS volume_spx500,
    p4.source AS source_spx500, p4.bar_origin AS bar_origin_spx500,
    COALESCE(c4.is_trading_day, false) AS is_session_bar_spx500,
    EXTRACT(epoch FROM now() - p4.bar_end_utc)::bigint AS staleness_seconds_spx500,
    CASE WHEN g.d < f4.first_bar THEN 'no_native_data'
         WHEN c4.is_trading_day IS NOT TRUE THEN
              CASE WHEN p4.close IS NULL THEN 'closed' ELSE 'off_session' END
         WHEN p4.close IS NULL THEN
              CASE WHEN g.d >= (now() AT TIME ZONE 'UTC')::date THEN 'pending' ELSE 'missing' END
         WHEN p4.quality_status <> 'ok' THEN 'incoherent'
         ELSE 'ok' END AS status_spx500,

    -- MACROS: as-of for reporting, _t1 (LAG) for models — the only legal model input
    ma.dxy AS macro_dxy,   LAG(ma.dxy)   OVER w AS macro_dxy_t1,
    ma.vix AS macro_vix,   LAG(ma.vix)   OVER w AS macro_vix_t1,
    ma.embi AS macro_embi, LAG(ma.embi)  OVER w AS macro_embi_t1,
    ma.brent AS macro_brent, LAG(ma.brent) OVER w AS macro_brent_t1,
    ma.wti AS macro_wti,   LAG(ma.wti)   OVER w AS macro_wti_t1,
    ma.gold AS macro_gold, LAG(ma.gold)  OVER w AS macro_gold_t1,
    ma.coffee AS macro_coffee, LAG(ma.coffee) OVER w AS macro_coffee_t1,
    ma.colcap AS macro_colcap, LAG(ma.colcap) OVER w AS macro_colcap_t1,
    ma.ust10y AS macro_ust10y, LAG(ma.ust10y) OVER w AS macro_ust10y_t1,
    ma.dgs2 AS macro_dgs2, LAG(ma.dgs2)  OVER w AS macro_dgs2_t1,
    ma.col5y AS macro_col5y, LAG(ma.col5y) OVER w AS macro_col5y_t1,
    ma.col10y AS macro_col10y, LAG(ma.col10y) OVER w AS macro_col10y_t1,
    ma.ibr AS macro_ibr,   LAG(ma.ibr)   OVER w AS macro_ibr_t1,
    ma.tpm AS macro_tpm,   LAG(ma.tpm)   OVER w AS macro_tpm_t1,
    ma.prime AS macro_prime, LAG(ma.prime) OVER w AS macro_prime_t1,
    ma.usdcop_spot AS macro_usdcop_spot, LAG(ma.usdcop_spot) OVER w AS macro_usdcop_spot_t1,
    ma.usdmxn AS macro_usdmxn, LAG(ma.usdmxn) OVER w AS macro_usdmxn_t1,
    ma.usdclp AS macro_usdclp, LAG(ma.usdclp) OVER w AS macro_usdclp_t1

FROM grid g
LEFT JOIN px p1 ON p1.asset_id='usdcop'  AND p1.session_date_local = g.d
LEFT JOIN px p2 ON p2.asset_id='xauusd'  AND p2.session_date_local = g.d
LEFT JOIN px p3 ON p3.asset_id='btcusdt' AND p3.session_date_local = g.d
LEFT JOIN px p4 ON p4.asset_id='spx500'  AND p4.session_date_local = g.d
LEFT JOIN cal c1 ON c1.asset_id='usdcop' AND c1.session_date = g.d
LEFT JOIN cal c4 ON c4.asset_id='spx500' AND c4.session_date = g.d
LEFT JOIN firsts f1 ON f1.asset_id='usdcop'
LEFT JOIN firsts f2 ON f2.asset_id='xauusd'
LEFT JOIN firsts f3 ON f3.asset_id='btcusdt'
LEFT JOIN firsts f4 ON f4.asset_id='spx500'
LEFT JOIN macro ma ON ma.fecha = g.d
WINDOW w AS (ORDER BY g.d);

-- ---------------------------------------------------------------------------
-- 5m wide: session pairs only (SPX has no native intraday -> status fixed at
-- 'no_native_data'; fabricating intraday bars from a daily close would invent
-- microstructure). Same 10-column block per asset.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW market_ohlcv_5m_wide AS
WITH grid AS (SELECT DISTINCT bar_start_utc FROM market_ohlcv_5m)
SELECT
    g.bar_start_utc,
    g.bar_start_utc + interval '5 minutes'          AS bar_end_utc,
    g.bar_start_utc AT TIME ZONE 'America/Bogota'   AS timestamp_cot,
    (g.bar_start_utc AT TIME ZONE 'America/Bogota')::date AS session_date_cot,
    CASE WHEN p1.close IS NULL AND c1.is_trading_day IS TRUE
              AND g.bar_start_utc >= c1.session_open_utc
              AND g.bar_start_utc < c1.session_close_utc
         THEN 'missing' ELSE 'ok' END               AS availability_status,

    p1.open AS open_usdcop, p1.high AS high_usdcop, p1.low AS low_usdcop,
    p1.close AS close_usdcop, p1.volume AS volume_usdcop,
    p1.source AS source_usdcop, p1.bar_origin AS bar_origin_usdcop,
    (c1.is_trading_day IS TRUE AND g.bar_start_utc >= c1.session_open_utc
       AND g.bar_start_utc < c1.session_close_utc)  AS is_session_bar_usdcop,
    EXTRACT(epoch FROM now() - (g.bar_start_utc + interval '5 minutes'))::bigint
                                                    AS staleness_seconds_usdcop,
    -- closed = non-trading day OR out-of-session hour on a trading day: a BTC bar at
    -- 03:00 UTC creates a grid row where COP's NULL is legitimate, not an incident.
    -- The 12:55 closing bar is IN session (8:00-12:55 inclusive, 60 bars) — hence
    -- strictly-greater on session_close_utc.
    -- off_session = bar PRESENT while the calendar says closed (provider quotes
    -- USD/COP offshore on Colombian holidays, ~12 days/yr): real data, but the
    -- session-calendar arbiter says the local market did not trade — surfaced
    -- distinctly instead of a contradictory 'closed'-with-a-price.
    CASE WHEN c1.is_trading_day IS NOT TRUE
           OR g.bar_start_utc < c1.session_open_utc
           OR g.bar_start_utc > c1.session_close_utc THEN
              CASE WHEN p1.close IS NULL THEN 'closed' ELSE 'off_session' END
         WHEN p1.close IS NULL THEN 'missing'
         WHEN p1.quality_status <> 'ok' THEN 'incoherent' ELSE 'ok' END AS status_usdcop,

    p3.open AS open_btcusdt, p3.high AS high_btcusdt, p3.low AS low_btcusdt,
    p3.close AS close_btcusdt, p3.volume AS volume_btcusdt,
    p3.source AS source_btcusdt, p3.bar_origin AS bar_origin_btcusdt,
    true AS is_session_bar_btcusdt,
    EXTRACT(epoch FROM now() - (g.bar_start_utc + interval '5 minutes'))::bigint
                                                    AS staleness_seconds_btcusdt,
    CASE WHEN p3.close IS NULL THEN 'missing'
         WHEN p3.quality_status <> 'ok' THEN 'incoherent' ELSE 'ok' END AS status_btcusdt,

    p2.open AS open_xauusd, p2.close AS close_xauusd, p2.volume AS volume_xauusd,
    p2.source AS source_xauusd, p2.bar_origin AS bar_origin_xauusd,
    (p2.close IS NOT NULL) AS is_session_bar_xauusd,
    EXTRACT(epoch FROM now() - (g.bar_start_utc + interval '5 minutes'))::bigint
                                                    AS staleness_seconds_xauusd,
    CASE WHEN p2.close IS NULL THEN 'closed_or_missing' ELSE 'ok' END AS status_xauusd,

    NULL::numeric AS open_spx500, NULL::numeric AS close_spx500,
    'no_native_data'::text AS status_spx500

FROM grid g
LEFT JOIN market_ohlcv_5m p1 ON p1.asset_id='usdcop'  AND p1.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_5m p2 ON p2.asset_id='xauusd'  AND p2.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_5m p3 ON p3.asset_id='btcusdt' AND p3.bar_start_utc=g.bar_start_utc
LEFT JOIN market_session_calendar c1
       ON c1.asset_id='usdcop'
      AND c1.session_date=(g.bar_start_utc AT TIME ZONE 'America/Bogota')::date;

-- ---------------------------------------------------------------------------
-- 1h / 4h wide over the aggregates. partial = fewer M5 bars than the bucket
-- implies for an in-session hour (COP: 12 per full hour).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW market_ohlcv_1h_wide AS
WITH grid AS (SELECT DISTINCT bar_start_utc FROM market_ohlcv_1h_agg)
SELECT
    g.bar_start_utc,
    g.bar_start_utc + interval '1 hour'             AS bar_end_utc,
    g.bar_start_utc AT TIME ZONE 'America/Bogota'   AS timestamp_cot,
    (g.bar_start_utc AT TIME ZONE 'America/Bogota')::date AS session_date_cot,
    a1.open AS open_usdcop, a1.high AS high_usdcop, a1.low AS low_usdcop,
    a1.close AS close_usdcop, a1.volume AS volume_usdcop,
    a1.source AS source_usdcop, 'derived_m5'::text AS bar_origin_usdcop,
    a1.n_m5_bars AS n_m5_bars_usdcop,
    CASE WHEN a1.close IS NULL THEN 'closed_or_missing'
         WHEN a1.n_m5_bars < 12 THEN 'partial' ELSE 'ok' END AS status_usdcop,
    a3.open AS open_btcusdt, a3.close AS close_btcusdt, a3.volume AS volume_btcusdt,
    'derived_m5'::text AS bar_origin_btcusdt, a3.n_m5_bars AS n_m5_bars_btcusdt,
    CASE WHEN a3.close IS NULL THEN 'missing'
         WHEN a3.n_m5_bars < 12 THEN 'partial' ELSE 'ok' END AS status_btcusdt,
    a2.open AS open_xauusd, a2.close AS close_xauusd,
    'derived_m5'::text AS bar_origin_xauusd, a2.n_m5_bars AS n_m5_bars_xauusd,
    CASE WHEN a2.close IS NULL THEN 'closed_or_missing'
         WHEN a2.n_m5_bars < 12 THEN 'partial' ELSE 'ok' END AS status_xauusd,
    'no_native_data'::text AS status_spx500
FROM grid g
LEFT JOIN market_ohlcv_1h_agg a1 ON a1.symbol='USD/COP'  AND a1.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_1h_agg a2 ON a2.symbol='XAU/USD'  AND a2.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_1h_agg a3 ON a3.symbol='BTC/USDT' AND a3.bar_start_utc=g.bar_start_utc;

CREATE OR REPLACE VIEW market_ohlcv_4h_wide AS
WITH grid AS (SELECT DISTINCT bar_start_utc FROM market_ohlcv_4h_agg)
SELECT
    g.bar_start_utc,
    g.bar_start_utc + interval '4 hours'            AS bar_end_utc,
    g.bar_start_utc AT TIME ZONE 'America/Bogota'   AS timestamp_cot,
    (g.bar_start_utc AT TIME ZONE 'America/Bogota')::date AS session_date_cot,
    a1.open AS open_usdcop, a1.close AS close_usdcop, a1.n_m5_bars AS n_m5_bars_usdcop,
    'derived_m5'::text AS bar_origin_usdcop,
    CASE WHEN a1.close IS NULL THEN 'closed_or_missing'
         WHEN a1.n_m5_bars < 48 THEN 'partial' ELSE 'ok' END AS status_usdcop,
    a3.open AS open_btcusdt, a3.close AS close_btcusdt, a3.n_m5_bars AS n_m5_bars_btcusdt,
    'derived_m5'::text AS bar_origin_btcusdt,
    CASE WHEN a3.close IS NULL THEN 'missing'
         WHEN a3.n_m5_bars < 48 THEN 'partial' ELSE 'ok' END AS status_btcusdt,
    a2.open AS open_xauusd, a2.close AS close_xauusd, a2.n_m5_bars AS n_m5_bars_xauusd,
    'derived_m5'::text AS bar_origin_xauusd,
    CASE WHEN a2.close IS NULL THEN 'closed_or_missing'
         WHEN a2.n_m5_bars < 48 THEN 'partial' ELSE 'ok' END AS status_xauusd,
    'no_native_data'::text AS status_spx500
FROM grid g
LEFT JOIN market_ohlcv_4h_agg a1 ON a1.symbol='USD/COP'  AND a1.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_4h_agg a2 ON a2.symbol='XAU/USD'  AND a2.bar_start_utc=g.bar_start_utc
LEFT JOIN market_ohlcv_4h_agg a3 ON a3.symbol='BTC/USDT' AND a3.bar_start_utc=g.bar_start_utc;
