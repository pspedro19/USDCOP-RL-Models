-- ============================================================================
-- Migration 067: SPX500 regime-gate macro variables (FRED, 0 trials)
-- ============================================================================
-- Contexto: PLAN-RENTABILIDAD-2026-07.md (spx500) pausa el regime-gated "hasta
-- tener variables reales bajo un contrato separado". Este es ese contrato de
-- DATOS: 8 series diarias/semanales + 1 mensual, todas FRED (fuente ya
-- operativa 8/8), verificadas en vivo 2026-07-27. La ingesta corre por el
-- mismo L0 (SSOT macro_variables_ssot.yaml -> registry -> FredExtractor ->
-- FrequencyRoutedUpsertService). El consumo por cualquier estrategia exige
-- shift T-1 / available_at <= decision (anti-leakage capa datos) y CUALQUIER
-- uso en señal se pre-registra como trial en el registry del activo.
--
-- Series (economic prior por variable):
--   BAMLH0A0HYM2  HY OAS            - stress de credito lidera drawdowns equity
--                                      (FRED solo publica desde 2023-07: ICE
--                                      restringio redistribucion; caveat).
--   T10Y2Y        curva 2s10s       - inversion/steepening = ciclo
--   T10Y3M        curva 3m10y       - senal recesiva lider
--   T10YIE        breakeven 10y     - regimen inflacionario
--   DFII10        real yield 10y    - regimen de tasas reales (tambien frontera Gold)
--   NFCI          cond. financieras - composite Chicago Fed (semanal, 1971->)
--   STLFSI4       stress STL Fed    - composite de stress (semanal, 1993->)
--   ICSA          initial claims    - deterioro laboral (semanal, 1967->)
--   SAHMREALTIME  regla de Sahm     - trigger recesivo (mensual, 1959->)
-- ============================================================================

-- Diarias / semanales-sparse (semanal se guarda en su fecha de observacion;
-- el ffill acotado vive rio abajo en el consumo, no aqui)
ALTER TABLE macro_indicators_daily
    ADD COLUMN IF NOT EXISTS finc_spread_hyoas_usa_d_hyoas   NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS finc_curve_t10y2y_usa_d_t10y2y  NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS finc_curve_t10y3m_usa_d_t10y3m  NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS infl_breakeven10y_usa_d_t10yie  NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS finc_realyield10y_usa_d_dfii10  NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS volt_nfci_usa_d_nfci            NUMERIC(10,5),
    ADD COLUMN IF NOT EXISTS volt_stress_stlfsi_usa_d_stlfsi4 NUMERIC(10,5),
    ADD COLUMN IF NOT EXISTS labr_claims_icsa_usa_d_icsa     NUMERIC(12,0);

-- Mensual
ALTER TABLE macro_indicators_monthly
    ADD COLUMN IF NOT EXISTS labr_sahm_usa_m_sahmrt          NUMERIC(6,2);

COMMENT ON COLUMN macro_indicators_daily.finc_spread_hyoas_usa_d_hyoas
    IS 'ICE BofA US High Yield OAS (FRED BAMLH0A0HYM2, %; historia solo 2023-07-> por licencia ICE)';
COMMENT ON COLUMN macro_indicators_daily.finc_curve_t10y2y_usa_d_t10y2y
    IS 'UST 10Y-2Y spread (FRED T10Y2Y, pp, 1976->)';
COMMENT ON COLUMN macro_indicators_daily.finc_curve_t10y3m_usa_d_t10y3m
    IS 'UST 10Y-3M spread (FRED T10Y3M, pp, 1982->)';
COMMENT ON COLUMN macro_indicators_daily.infl_breakeven10y_usa_d_t10yie
    IS '10Y breakeven inflation (FRED T10YIE, %, 2003->)';
COMMENT ON COLUMN macro_indicators_daily.finc_realyield10y_usa_d_dfii10
    IS '10Y TIPS real yield (FRED DFII10, %, 2003->)';
COMMENT ON COLUMN macro_indicators_daily.volt_nfci_usa_d_nfci
    IS 'Chicago Fed NFCI (FRED NFCI, semanal viernes-obs, 1971->)';
COMMENT ON COLUMN macro_indicators_daily.volt_stress_stlfsi_usa_d_stlfsi4
    IS 'St. Louis Fed Financial Stress Index v4 (FRED STLFSI4, semanal, 1993->)';
COMMENT ON COLUMN macro_indicators_daily.labr_claims_icsa_usa_d_icsa
    IS 'Initial jobless claims SA (FRED ICSA, semanal sabado-obs, 1967->)';
COMMENT ON COLUMN macro_indicators_monthly.labr_sahm_usa_m_sahmrt
    IS 'Sahm rule realtime (FRED SAHMREALTIME, pp, 1959->)';
