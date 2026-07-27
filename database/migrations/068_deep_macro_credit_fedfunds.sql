-- ============================================================================
-- Migration 068: credito profundo (BAA10Y) + fed funds DIARIO (DFF) — 0 trials
-- ============================================================================
-- Contexto (directiva operador 2026-07-27, "variables que realmente impactan"):
-- HY-OAS (BAMLH0A0HYM2) quedo capado a 2023-07 por licencia ICE. El sustituto
-- profundo SIN licencia es el spread Moody's Baa - UST10y (FRED BAA10Y, diario
-- 1986->): mismo eje economico (stress de credito corporativo lidera drawdowns
-- de equity), correlacion historica alta con HY-OAS, y cubre 2000-02/2008.
-- Fed funds: la tabla mensual (FEDFUNDS) es lenta; DFF es la tasa EFECTIVA
-- DIARIA (FRED, 1954->) — el instrumento correcto para regimenes de politica.

ALTER TABLE macro_indicators_daily
    ADD COLUMN IF NOT EXISTS finc_spread_baa10y_usa_d_baa10y NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS polr_fed_funds_usa_d_dff        NUMERIC(8,4);

COMMENT ON COLUMN macro_indicators_daily.finc_spread_baa10y_usa_d_baa10y
    IS 'Moody''s Baa - UST10Y spread (FRED BAA10Y, pp, 1986->; sustituto profundo de HY-OAS capado por ICE)';
COMMENT ON COLUMN macro_indicators_daily.polr_fed_funds_usa_d_dff
    IS 'Effective Federal Funds Rate diaria (FRED DFF, %, 1954->)';
