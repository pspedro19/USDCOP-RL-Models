-- Migration 052: Crypto-native data layer for BTC/USDT (scalable, additive)
-- ============================================================================
-- Purpose:
--   BTC/USDT (the 3rd tradeable asset) needs signal inputs that DO NOT EXIST in
--   the repo: on-chain valuation, perp funding/derivatives, ETF flows, stablecoin
--   supply, and a discrete event calendar. FX/Gold reuse `macro_indicators_daily`;
--   crypto-native series have no home. This migration adds that home — additively.
--
-- What reuses existing tables (NO new table — avoid silos):
--   * 5-min bars   -> usdcop_m5_ohlcv           (PK (time,symbol), symbol='BTC/USDT')
--   * daily bars   -> asset_daily_ohlcv         (migration 051, symbol='BTC/USDT')
--   * FRED macro   -> macro_indicators_daily    (DFII10/DTWEXBGS/VIXCLS/M2SL already ingested)
--
-- What is genuinely new (this migration):
--   1. crypto_onchain_daily     — BGeometrics on-chain valuation (MVRV, NUPL, SOPR, ...)
--   2. crypto_derivatives_daily — Binance perp funding rate, OI, basis (DATA-ONLY, spot-executed)
--   3. crypto_flows_daily       — Farside ETF net flows (D+1 lag) + DefiLlama stablecoin supply
--   4. crypto_event_calendar    — discrete event gate G in {1,0.5,0.25,0} (LLM + vol-spike tagged)
--   5. crypto_exposure_signals  — engine output: exposure in [0,1] + component breakdown (audit trail)
--
-- Design rules (mirror 051 + strategy design in .claude/specs/assets/btcusdt/design/):
--   * All series keyed by (date/time, symbol) so a 2nd crypto asset (ETH) plugs in by symbol.
--   * TIMESTAMPTZ / DATE at the UTC instant — crypto is UTC-native (ADR-0008).
--   * Idempotent UPSERT on the PK — safe to re-run ingestion.
--   * source column carries provenance for every row (audit / DSR reproducibility).
--   * value_asof / published_at separate the *event* date from the *availability* date so
--     backfills never leak future data (ETF flows publish D+1; on-chain settles at UTC close).
--
-- Additive: touches no existing object. Date: 2026-07-05 · Contract: CTR-L0-CRYPTO-001
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. ON-CHAIN VALUATION (BGeometrics free API; frozen-fit HMM input — SPEC design 01/04)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crypto_onchain_daily (
    date          DATE              NOT NULL,   -- UTC calendar day the metric settles
    symbol        VARCHAR(20)       NOT NULL,   -- 'BTC/USDT' (BTC on-chain applies to the pair)
    mvrv_zscore   DOUBLE PRECISION,             -- market-value / realized-value z-score (valuation)
    nupl          DOUBLE PRECISION,             -- net unrealized profit/loss
    sopr          DOUBLE PRECISION,             -- spent-output profit ratio
    rhodl_ratio   DOUBLE PRECISION,             -- RHODL band ratio
    reserve_risk  DOUBLE PRECISION,             -- reserve risk (conviction vs price)
    puell_multiple DOUBLE PRECISION,            -- miner-revenue Puell multiple
    realized_price DOUBLE PRECISION,            -- aggregate realized price (USD)
    extra         JSONB             DEFAULT '{}'::jsonb,  -- forward-compat: any extra BGeo metric
    source        VARCHAR(50)       DEFAULT 'bgeometrics',
    published_at  TIMESTAMPTZ,                  -- when the value became available (leak guard)
    updated_at    TIMESTAMPTZ       DEFAULT NOW(),
    PRIMARY KEY (date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_crypto_onchain_symbol_date ON crypto_onchain_daily (symbol, date DESC);

-- ---------------------------------------------------------------------------
-- 2. DERIVATIVES / FUNDING (Binance perp — DATA ONLY; strategy executes spot, ADR-0008)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crypto_derivatives_daily (
    date            DATE            NOT NULL,
    symbol          VARCHAR(20)     NOT NULL,   -- 'BTC/USDT'
    funding_rate    DOUBLE PRECISION,           -- daily-aggregated perp funding (raw)
    funding_zscore  DOUBLE PRECISION,           -- rolling z (positioning brake input, sign '-')
    open_interest   DOUBLE PRECISION,           -- perp OI (USD)
    basis_annualized DOUBLE PRECISION,          -- (perp - spot) annualized basis
    long_short_ratio DOUBLE PRECISION,          -- top-trader long/short ratio
    liquidations_usd DOUBLE PRECISION,          -- 24h liquidations
    source          VARCHAR(50)     DEFAULT 'binance',
    published_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ     DEFAULT NOW(),
    PRIMARY KEY (date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_crypto_deriv_symbol_date ON crypto_derivatives_daily (symbol, date DESC);

-- ---------------------------------------------------------------------------
-- 3. FLOWS (Farside ETF net flows, D+1 published; DefiLlama stablecoin supply)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crypto_flows_daily (
    date              DATE          NOT NULL,   -- flow event date (UTC)
    symbol            VARCHAR(20)   NOT NULL,   -- 'BTC/USDT'
    etf_net_flow_usd  DOUBLE PRECISION,         -- spot-BTC-ETF aggregate net flow (Farside)
    etf_cum_flow_usd  DOUBLE PRECISION,         -- cumulative net flow
    stablecoin_supply_usd DOUBLE PRECISION,     -- aggregate stablecoin supply (DefiLlama)
    stablecoin_delta_usd  DOUBLE PRECISION,     -- day-over-day supply change (liquidity proxy)
    source            VARCHAR(50)   DEFAULT 'farside',
    published_at      TIMESTAMPTZ,              -- Farside publishes D+1 — leak guard critical here
    updated_at        TIMESTAMPTZ   DEFAULT NOW(),
    PRIMARY KEY (date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_crypto_flows_symbol_date ON crypto_flows_daily (symbol, date DESC);

-- ---------------------------------------------------------------------------
-- 4. EVENT CALENDAR (discrete event gate G — SPEC design 05; LLM + vol-spike tagged)
--    G multiplies exposure: 1.0 normal | 0.5 elevated | 0.25 high | 0.0 halt.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crypto_event_calendar (
    event_id      BIGSERIAL         PRIMARY KEY,
    symbol        VARCHAR(20)       NOT NULL DEFAULT 'BTC/USDT',
    event_time    TIMESTAMPTZ       NOT NULL,   -- when the event takes effect (UTC)
    event_type    VARCHAR(40)       NOT NULL,   -- 'macro_fomc'|'cpi'|'halving'|'unlock'|'hack'|'regulatory'|'vol_spike'
    severity      VARCHAR(20)       NOT NULL,   -- 'normal'|'elevated'|'high'|'halt'
    gate_value    DOUBLE PRECISION  NOT NULL,   -- G in {1.0, 0.5, 0.25, 0.0}
    title         TEXT,
    detected_by   VARCHAR(30)       DEFAULT 'llm',  -- 'llm'|'vol_spike'|'manual'|'calendar'
    confidence    DOUBLE PRECISION,             -- LLM confidence (contamination guard, ADR-0011)
    window_end    TIMESTAMPTZ,                  -- gate auto-clears at this instant
    source        VARCHAR(50)       DEFAULT 'analysis_llm',
    created_at    TIMESTAMPTZ       DEFAULT NOW(),
    UNIQUE (symbol, event_time, event_type)
);
CREATE INDEX IF NOT EXISTS idx_crypto_event_symbol_time ON crypto_event_calendar (symbol, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_crypto_event_active ON crypto_event_calendar (symbol, window_end);

-- ---------------------------------------------------------------------------
-- 5. EXPOSURE SIGNALS (engine output — full component audit trail, SPEC design 06)
--    One row per decision bar. exposure = clip( vol_target * risk_score * M_liq * G, 0, 1 ).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crypto_exposure_signals (
    signal_time     TIMESTAMPTZ     NOT NULL,   -- daily decision instant (UTC 00:00 close)
    symbol          VARCHAR(20)     NOT NULL,   -- 'BTC/USDT'
    strategy_id     VARCHAR(40)     NOT NULL DEFAULT 'btc_exposure_s3',
    target_exposure DOUBLE PRECISION NOT NULL,  -- final exposure in [0,1]
    cycle_regime    VARCHAR(20),                -- HMM state label (accumulation|markup|distribution|markdown)
    cycle_z         DOUBLE PRECISION,           -- z_ciclo (on-chain valuation component)
    funding_z       DOUBLE PRECISION,           -- z_funding (positioning brake component)
    risk_score      DOUBLE PRECISION,           -- R = 0.7*z_ciclo + 0.3*z_funding (additive, ADR-0009)
    vol_target_mult DOUBLE PRECISION,           -- vol-targeting multiplier (sigma_target/realized)
    liquidity_mult  DOUBLE PRECISION,           -- M_liquidez in [0.5,1.25]
    event_gate      DOUBLE PRECISION,           -- G in {1,0.5,0.25,0}
    meta_label_pass BOOLEAN,                    -- meta-labeling brake (XGBoost, SPEC design 08)
    realized_vol    DOUBLE PRECISION,           -- realized vol used for targeting
    rebalanced      BOOLEAN         DEFAULT FALSE,  -- did exposure cross the +/-12.5% band?
    source          VARCHAR(50)     DEFAULT 'btc_exposure_engine',
    updated_at      TIMESTAMPTZ     DEFAULT NOW(),
    PRIMARY KEY (signal_time, symbol, strategy_id)
);
CREATE INDEX IF NOT EXISTS idx_crypto_exposure_symbol_time ON crypto_exposure_signals (symbol, signal_time DESC);

-- ---------------------------------------------------------------------------
-- TimescaleDB hypertables (best-effort; ignored if extension unavailable)
-- ---------------------------------------------------------------------------
DO $$
DECLARE t TEXT;
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- time-series tables keyed on a time/date dimension
        PERFORM create_hypertable('crypto_exposure_signals', 'signal_time',
                                  if_not_exists => TRUE, migrate_data => TRUE);
        -- date-keyed tables: convert only if the timescaledb version accepts DATE partitioning;
        -- otherwise they remain plain tables (indexes above keep them fast).
        BEGIN
            PERFORM create_hypertable('crypto_onchain_daily', 'date',
                                      if_not_exists => TRUE, migrate_data => TRUE,
                                      chunk_time_interval => INTERVAL '365 days');
            PERFORM create_hypertable('crypto_derivatives_daily', 'date',
                                      if_not_exists => TRUE, migrate_data => TRUE,
                                      chunk_time_interval => INTERVAL '365 days');
            PERFORM create_hypertable('crypto_flows_daily', 'date',
                                      if_not_exists => TRUE, migrate_data => TRUE,
                                      chunk_time_interval => INTERVAL '365 days');
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'crypto date-keyed hypertables skipped (%); plain tables retained.', SQLERRM;
        END;
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'crypto hypertable step skipped (%).', SQLERRM;
END $$;

-- ---------------------------------------------------------------------------
-- Monitoring view: crypto data coverage per source (freshness dashboard)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW crypto_data_coverage AS
SELECT 'onchain'     AS dataset, symbol, COUNT(*) AS rows, MIN(date)::timestamptz AS earliest, MAX(date)::timestamptz AS latest FROM crypto_onchain_daily     GROUP BY symbol
UNION ALL
SELECT 'derivatives' AS dataset, symbol, COUNT(*) AS rows, MIN(date)::timestamptz AS earliest, MAX(date)::timestamptz AS latest FROM crypto_derivatives_daily GROUP BY symbol
UNION ALL
SELECT 'flows'       AS dataset, symbol, COUNT(*) AS rows, MIN(date)::timestamptz AS earliest, MAX(date)::timestamptz AS latest FROM crypto_flows_daily       GROUP BY symbol
UNION ALL
SELECT 'exposure'    AS dataset, symbol, COUNT(*) AS rows, MIN(signal_time)       AS earliest, MAX(signal_time)       AS latest FROM crypto_exposure_signals  GROUP BY symbol
ORDER BY dataset, symbol;

COMMENT ON TABLE crypto_onchain_daily IS
    'BTC on-chain valuation (BGeometrics). HMM cycle-regime input. Keyed (date,symbol); ETH/... plug in by symbol.';
COMMENT ON TABLE crypto_derivatives_daily IS
    'Perp funding/OI/basis (Binance). DATA-ONLY — strategy executes spot (ADR-0008). funding_zscore is the positioning brake.';
COMMENT ON TABLE crypto_flows_daily IS
    'Spot-ETF net flows (Farside, D+1) + stablecoin supply (DefiLlama). published_at guards D+1 leakage.';
COMMENT ON TABLE crypto_event_calendar IS
    'Discrete event gate G in {1,0.5,0.25,0}. LLM + vol-spike tagged; confidence guards LLM contamination (ADR-0011).';
COMMENT ON TABLE crypto_exposure_signals IS
    'Exposure engine output in [0,1] with full component breakdown (cycle/funding/liquidity/event/meta). Audit trail for DSR reproducibility.';
