-- ========================================
-- USDCOP TRADING SYSTEM - DIAGNOSTIC QUERIES
-- Fecha: 2026-01-08
-- ========================================

-- ========================================
-- 1. CONTEO DE TABLAS CRÍTICAS
-- ========================================
SELECT 'equity_snapshots' as tabla, COUNT(*) as rows FROM equity_snapshots
UNION ALL SELECT 'trades_history', COUNT(*) FROM trades_history
UNION ALL SELECT 'trading_state', COUNT(*) FROM trading_state
UNION ALL SELECT 'dw.fact_rl_inference', COUNT(*) FROM dw.fact_rl_inference
UNION ALL SELECT 'config.models', COUNT(*) FROM config.models
UNION ALL SELECT 'usdcop_m5_ohlcv', COUNT(*) FROM usdcop_m5_ohlcv
UNION ALL SELECT 'macro_indicators_daily', COUNT(*) FROM macro_indicators_daily;

-- ========================================
-- 2. DISTRIBUCIÓN DE SEÑALES (últimos 7 días)
-- ========================================
SELECT
    action_discretized,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
FROM dw.fact_rl_inference
WHERE timestamp_utc > NOW() - INTERVAL '7 days'
GROUP BY action_discretized
ORDER BY count DESC;

-- ========================================
-- 3. RANGO DE RAW_ACTION VALUES (CRÍTICO para threshold)
-- ========================================
SELECT
    MIN(action_raw) as min_action,
    MAX(action_raw) as max_action,
    AVG(action_raw)::numeric(10,4) as avg_action,
    STDDEV(action_raw)::numeric(10,4) as std_action,
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY action_raw)::numeric(10,4) as p10,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY action_raw)::numeric(10,4) as p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY action_raw)::numeric(10,4) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY action_raw)::numeric(10,4) as p75,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY action_raw)::numeric(10,4) as p90
FROM dw.fact_rl_inference
WHERE timestamp_utc > NOW() - INTERVAL '7 days'
  AND action_raw IS NOT NULL;

-- ========================================
-- 4. SEÑALES "PERDIDAS" POR THRESHOLD INCORRECTO
-- ========================================
SELECT
    COUNT(*) FILTER (WHERE action_raw > 0.10 AND action_raw <= 0.30) as would_be_long_now_hold,
    COUNT(*) FILTER (WHERE action_raw < -0.10 AND action_raw >= -0.30) as would_be_short_now_hold,
    COUNT(*) as total_inferences,
    ROUND(100.0 * COUNT(*) FILTER (WHERE action_raw > 0.10 AND action_raw <= 0.30) / NULLIF(COUNT(*), 0), 2) as pct_lost_long,
    ROUND(100.0 * COUNT(*) FILTER (WHERE action_raw < -0.10 AND action_raw >= -0.30) / NULLIF(COUNT(*), 0), 2) as pct_lost_short
FROM dw.fact_rl_inference
WHERE timestamp_utc > NOW() - INTERVAL '7 days'
  AND action_raw IS NOT NULL;

-- ========================================
-- 5. ESTADO ACTUAL DE POSICIONES (trading_state)
-- ========================================
SELECT * FROM trading_state ORDER BY last_updated DESC;

-- ========================================
-- 6. ÚLTIMOS 10 TRADES
-- ========================================
SELECT
    model_id,
    side as direction,
    entry_price,
    exit_price,
    pnl_usd,
    entry_time,
    exit_time
FROM trades_history
ORDER BY entry_time DESC
LIMIT 10;

-- ========================================
-- 7. MACRO DATA - ¿Hay NULLs recientes?
-- ========================================
SELECT
    fecha,
    fxrt_index_dxy_usa_d_dxy as dxy,
    volt_vix_usa_d_vix as vix,
    crsk_spread_embi_col_d_embi as embi,
    comm_oil_brent_glb_d_brent as brent
FROM macro_indicators_daily
WHERE fecha > CURRENT_DATE - INTERVAL '7 days'
ORDER BY fecha DESC;

-- ========================================
-- 8. GAPS EN OHLCV (últimos 7 días)
-- ========================================
WITH bars AS (
    SELECT
        time as timestamp_cot,
        LAG(time) OVER (ORDER BY time) as prev_ts
    FROM usdcop_m5_ohlcv
    WHERE time > NOW() - INTERVAL '7 days'
)
SELECT
    prev_ts as gap_start,
    timestamp_cot as gap_end,
    EXTRACT(EPOCH FROM (timestamp_cot - prev_ts))/60 as gap_minutes
FROM bars
WHERE timestamp_cot - prev_ts > INTERVAL '10 minutes'
ORDER BY gap_minutes DESC
LIMIT 20;

-- ========================================
-- 9. CONFIG.MODELS - Ver thresholds actuales
-- ========================================
SELECT
    model_id,
    model_name,
    threshold_long,
    threshold_short,
    enabled as is_active
FROM config.models;

-- ========================================
-- 10. VERIFICAR SCHEMA DW EXISTS
-- ========================================
SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('dw', 'config', 'public');

-- ========================================
-- 11. LISTAR TODAS LAS TABLAS
-- ========================================
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema IN ('public', 'dw', 'config', 'trading')
ORDER BY table_schema, table_name;
