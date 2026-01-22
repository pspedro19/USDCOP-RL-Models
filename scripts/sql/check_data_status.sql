-- =============================================================================
-- DATA STATUS QUERIES - Validación para lanzar experimentos
-- =============================================================================
-- Ejecutar con:
--   docker exec -it usdcop-postgres-timescale psql -U postgres -d usdcop_trading -f /path/to/check_data_status.sql
-- O copiar queries individuales al cliente SQL
-- =============================================================================

-- #############################################################################
-- QUERY 1: Estado de OHLCV 5min
-- #############################################################################
SELECT
    '=== OHLCV 5min STATUS ===' as section;

SELECT
    MIN(time) as primera_barra,
    MAX(time) as ultima_barra,
    COUNT(*) as total_barras,
    COUNT(DISTINCT DATE(time)) as dias_trading,
    ROUND(COUNT(*) / NULLIF(COUNT(DISTINCT DATE(time)), 0), 0) as barras_por_dia_avg
FROM usdcop_m5_ohlcv;

-- Verificar cobertura del rango requerido (2023-01-01 a 2024-12-31)
SELECT
    '=== COBERTURA REQUERIDA ===' as section;

SELECT
    CASE WHEN MIN(time)::DATE <= '2023-01-01' THEN '✓' ELSE '✗' END as inicio_ok,
    CASE WHEN MAX(time)::DATE >= '2024-12-31' THEN '✓' ELSE '✗' END as fin_ok,
    MIN(time)::DATE as tiene_desde,
    MAX(time)::DATE as tiene_hasta,
    '2023-01-01'::DATE as necesita_desde,
    '2024-12-31'::DATE as necesita_hasta
FROM usdcop_m5_ohlcv;

-- Gaps por año
SELECT
    '=== BARRAS POR AÑO ===' as section;

SELECT
    EXTRACT(YEAR FROM time) as año,
    COUNT(*) as barras,
    MIN(time)::DATE as desde,
    MAX(time)::DATE as hasta
FROM usdcop_m5_ohlcv
GROUP BY EXTRACT(YEAR FROM time)
ORDER BY año;


-- #############################################################################
-- QUERY 2: Estado de Macro Indicators
-- #############################################################################
SELECT
    '=== MACRO INDICATORS STATUS ===' as section;

SELECT
    MIN(fecha) as primera_fecha,
    MAX(fecha) as ultima_fecha,
    COUNT(*) as total_dias
FROM macro_indicators_daily;

-- Conteo de columnas en la tabla
SELECT
    '=== COLUMNAS EN macro_indicators_daily ===' as section;

SELECT COUNT(*) as total_columnas
FROM information_schema.columns
WHERE table_name = 'macro_indicators_daily';

-- Variables críticas para RL (7 features)
SELECT
    '=== VARIABLES CRÍTICAS PARA RL ===' as section;

SELECT
    COUNT(*) as total_rows,
    COUNT(fxrt_index_dxy_usa_d_dxy) as dxy_populated,
    COUNT(volt_vix_usa_d_vix) as vix_populated,
    COUNT(crsk_spread_embi_col_d_embi) as embi_populated,
    COUNT(fxrt_spot_usdmxn_mex_d_usdmxn) as usdmxn_populated,
    COUNT(comm_oil_brent_glb_d_brent) as brent_populated,
    COUNT(finc_bond_yield10y_usa_d_ust10y) as ust10y_populated,
    COUNT(polr_fed_funds_usa_m_fedfunds) as fedfunds_populated
FROM macro_indicators_daily;

-- Porcentaje de llenado de variables críticas
SELECT
    '=== % LLENADO VARIABLES CRÍTICAS ===' as section;

SELECT
    ROUND(100.0 * COUNT(fxrt_index_dxy_usa_d_dxy) / NULLIF(COUNT(*), 0), 1) as dxy_pct,
    ROUND(100.0 * COUNT(volt_vix_usa_d_vix) / NULLIF(COUNT(*), 0), 1) as vix_pct,
    ROUND(100.0 * COUNT(crsk_spread_embi_col_d_embi) / NULLIF(COUNT(*), 0), 1) as embi_pct,
    ROUND(100.0 * COUNT(fxrt_spot_usdmxn_mex_d_usdmxn) / NULLIF(COUNT(*), 0), 1) as usdmxn_pct,
    ROUND(100.0 * COUNT(comm_oil_brent_glb_d_brent) / NULLIF(COUNT(*), 0), 1) as brent_pct,
    ROUND(100.0 * COUNT(finc_bond_yield10y_usa_d_ust10y) / NULLIF(COUNT(*), 0), 1) as ust10y_pct,
    ROUND(100.0 * COUNT(polr_fed_funds_usa_m_fedfunds) / NULLIF(COUNT(*), 0), 1) as fedfunds_pct
FROM macro_indicators_daily;

-- Últimos 5 registros de macro (para ver si hay datos recientes)
SELECT
    '=== ÚLTIMOS 5 REGISTROS MACRO ===' as section;

SELECT
    fecha,
    fxrt_index_dxy_usa_d_dxy as dxy,
    volt_vix_usa_d_vix as vix,
    crsk_spread_embi_col_d_embi as embi,
    comm_oil_brent_glb_d_brent as brent
FROM macro_indicators_daily
ORDER BY fecha DESC
LIMIT 5;


-- #############################################################################
-- QUERY 3: Resumen de decisión
-- #############################################################################
SELECT
    '=== DECISIÓN: ¿NECESITO CORRER PIPELINES? ===' as section;

SELECT
    CASE
        WHEN ohlcv_ok AND macro_ok THEN '✓ LISTO - Puedes lanzar el experimento'
        WHEN NOT ohlcv_ok AND NOT macro_ok THEN '✗ Correr L0_ohlcv_backfill + L0_macro_unified'
        WHEN NOT ohlcv_ok THEN '✗ Correr L0_ohlcv_backfill (faltan barras OHLCV)'
        WHEN NOT macro_ok THEN '✗ Correr L0_macro_unified (faltan datos macro)'
    END as accion_requerida,
    ohlcv_ok,
    macro_ok,
    ohlcv_desde,
    ohlcv_hasta,
    ohlcv_barras,
    macro_desde,
    macro_hasta,
    macro_dias
FROM (
    SELECT
        (SELECT MIN(time)::DATE <= '2023-01-01' AND MAX(time)::DATE >= '2024-12-31' FROM usdcop_m5_ohlcv) as ohlcv_ok,
        (SELECT MIN(fecha) <= '2023-01-01' AND MAX(fecha) >= '2024-12-31'
            AND COUNT(fxrt_index_dxy_usa_d_dxy) > COUNT(*) * 0.9 FROM macro_indicators_daily) as macro_ok,
        (SELECT MIN(time)::DATE FROM usdcop_m5_ohlcv) as ohlcv_desde,
        (SELECT MAX(time)::DATE FROM usdcop_m5_ohlcv) as ohlcv_hasta,
        (SELECT COUNT(*) FROM usdcop_m5_ohlcv) as ohlcv_barras,
        (SELECT MIN(fecha) FROM macro_indicators_daily) as macro_desde,
        (SELECT MAX(fecha) FROM macro_indicators_daily) as macro_hasta,
        (SELECT COUNT(*) FROM macro_indicators_daily) as macro_dias
) as status;
