-- ========================================================================
-- DATA MARTS - Materialized Views for BI
-- Sistema USDCOP Trading - Pre-aggregated views for dashboards
-- ========================================================================
-- Version: 1.0
-- Date: 2025-10-22
-- Description: Creates materialized views in dm schema for fast BI queries
-- ========================================================================

-- ========================================================================
-- PIPELINE HEALTH OVERVIEW
-- ========================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS dm.mv_pipeline_health AS
SELECT
    'L0' as layer,
    'Acquisition' as process,
    COUNT(*) as total_runs,
    SUM(CASE WHEN quality_passed THEN 1 ELSE 0 END) as passed_runs,
    ROUND(AVG(coverage_pct), 2) as avg_coverage_pct,
    ROUND(AVG(stale_rate_pct), 2) as avg_stale_rate_pct,
    SUM(rows_fetched) as total_rows_fetched,
    SUM(rows_inserted) as total_rows_inserted,
    ROUND(AVG(duration_sec), 1) as avg_duration_sec,
    MAX(execution_date) as last_execution_date
FROM dw.fact_l0_acquisition
WHERE execution_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY layer, process

UNION ALL

SELECT
    'L1' as layer,
    'Standardization' as process,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status_passed THEN 1 ELSE 0 END) as passed_runs,
    ROUND(AVG(coverage_pct), 2) as avg_coverage_pct,
    ROUND(AVG(repeated_ohlc_rate_pct), 2) as avg_repeated_ohlc_rate_pct,
    SUM(total_episodes) as total_episodes,
    SUM(accepted_episodes) as accepted_episodes,
    NULL as avg_duration_sec,
    MAX(date_cot) as last_execution_date
FROM dw.fact_l1_quality
WHERE date_cot >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY layer, process;

CREATE UNIQUE INDEX idx_mv_pipeline_health_layer ON dm.mv_pipeline_health(layer, process);
COMMENT ON MATERIALIZED VIEW dm.mv_pipeline_health IS 'Pipeline health summary for last 30 days';

-- ========================================================================
-- DAILY COVERAGE REPORT
-- ========================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS dm.mv_daily_coverage AS
SELECT
    dt.date_cot,
    dt.day_of_week,
    ds.symbol_code,
    COUNT(*) as bars_received,
    72 as bars_expected,  -- 6 hours * 12 bars/hour
    ROUND(COUNT(*) * 100.0 / 72, 2) as coverage_pct,
    MIN(dt.ts_cot) as first_bar_time,
    MAX(dt.ts_cot) as last_bar_time
FROM dw.fact_bar_5m fb
JOIN dw.dim_time_5m dt ON fb.time_id = dt.time_id
JOIN dw.dim_symbol ds ON fb.symbol_id = ds.symbol_id
WHERE dt.is_trading_hour = TRUE
AND dt.date_cot >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY dt.date_cot, dt.day_of_week, ds.symbol_code
ORDER BY dt.date_cot DESC, ds.symbol_code;

CREATE UNIQUE INDEX idx_mv_daily_coverage_date_symbol ON dm.mv_daily_coverage(date_cot, symbol_code);
COMMENT ON MATERIALIZED VIEW dm.mv_daily_coverage IS 'Daily data coverage report by symbol';

-- ========================================================================
-- INDICATOR STATISTICS
-- ========================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS dm.mv_indicator_stats AS
SELECT
    di.indicator_name,
    di.indicator_family,
    ds.symbol_code,
    COUNT(*) as data_points,
    ROUND(AVG(fi.indicator_value)::numeric, 4) as avg_value,
    ROUND(STDDEV(fi.indicator_value)::numeric, 4) as std_value,
    ROUND(MIN(fi.indicator_value)::numeric, 4) as min_value,
    ROUND(MAX(fi.indicator_value)::numeric, 4) as max_value,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fi.indicator_value)::numeric, 4) as median_value,
    MAX(fi.ts_utc) as last_updated
FROM dw.fact_indicator_5m fi
JOIN dw.dim_indicator di ON fi.indicator_id = di.indicator_id
JOIN dw.dim_symbol ds ON fi.symbol_id = ds.symbol_id
WHERE fi.ts_utc >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY di.indicator_name, di.indicator_family, ds.symbol_code;

CREATE UNIQUE INDEX idx_mv_indicator_stats_name_symbol ON dm.mv_indicator_stats(indicator_name, symbol_code);
COMMENT ON MATERIALIZED VIEW dm.mv_indicator_stats IS 'Technical indicator statistics for last 30 days';

-- ========================================================================
-- BACKTEST LEADERBOARD
-- ========================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS dm.mv_backtest_leaderboard AS
SELECT
    dr.run_id,
    dr.date_range_start,
    dr.date_range_end,
    dm.model_name,
    dm.algorithm,
    dm.version,
    ds.symbol_code,
    fps.split,
    fps.total_return,
    fps.cagr,
    fps.sharpe_ratio,
    fps.sortino_ratio,
    fps.calmar_ratio,
    fps.max_drawdown,
    fps.total_trades,
    fps.win_rate,
    fps.profit_factor,
    fps.is_production_ready,
    dr.execution_date,
    RANK() OVER (PARTITION BY fps.split ORDER BY fps.sharpe_ratio DESC NULLS LAST) as sharpe_rank,
    RANK() OVER (PARTITION BY fps.split ORDER BY fps.sortino_ratio DESC NULLS LAST) as sortino_rank,
    RANK() OVER (PARTITION BY fps.split ORDER BY fps.calmar_ratio DESC NULLS LAST) as calmar_rank
FROM dw.fact_perf_summary fps
JOIN dw.dim_backtest_run dr ON fps.run_sk = dr.run_sk
JOIN dw.dim_model dm ON dr.model_sk = dm.model_sk
JOIN dw.dim_symbol ds ON dr.symbol_id = ds.symbol_id
WHERE dr.execution_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY fps.split, fps.sharpe_ratio DESC NULLS LAST;

CREATE UNIQUE INDEX idx_mv_backtest_leaderboard ON dm.mv_backtest_leaderboard(run_id, split);
COMMENT ON MATERIALIZED VIEW dm.mv_backtest_leaderboard IS 'Backtest leaderboard ranked by performance metrics';

-- ========================================================================
-- FEATURE QUALITY SCORECARD
-- ========================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS dm.mv_feature_quality AS
SELECT
    df.feature_name,
    df.feature_type,
    df.tier,

    -- Forward IC metrics
    ROUND(AVG(CASE WHEN fic.split = 'train' THEN fic.ic END)::numeric, 4) as train_ic,
    ROUND(AVG(CASE WHEN fic.split = 'val' THEN fic.ic END)::numeric, 4) as val_ic,
    ROUND(AVG(CASE WHEN fic.split = 'test' THEN fic.ic END)::numeric, 4) as test_ic,

    -- Leakage tests
    MAX(CASE WHEN flt.status_passed THEN 1 ELSE 0 END) as leakage_pass,

    -- Observation stats (test split)
    ROUND(AVG(CASE WHEN fro.split = 'test' THEN fro.clip_rate END)::numeric, 6) as test_clip_rate,
    ROUND(AVG(CASE WHEN fro.split = 'test' THEN fro.abs_max END)::numeric, 4) as test_abs_max,

    -- Overall quality score (0-100)
    ROUND(
        (
            (1 - COALESCE(AVG(CASE WHEN fro.split = 'test' THEN fro.clip_rate END), 0)) * 30 +  -- Clip rate score
            (CASE WHEN MAX(CASE WHEN flt.status_passed THEN 1 ELSE 0 END) = 1 THEN 30 ELSE 0 END) +  -- Leakage score
            (ABS(COALESCE(AVG(CASE WHEN fic.split = 'test' THEN fic.ic END), 0)) * 100 * 40)  -- IC score
        )::numeric,
        2
    ) as quality_score,

    COUNT(DISTINCT fic.date_cot) as days_tracked,
    MAX(fic.date_cot) as last_updated

FROM dw.dim_feature df
LEFT JOIN dw.fact_forward_ic fic ON df.feature_id = fic.feature_id
LEFT JOIN dw.fact_leakage_tests flt ON df.feature_id = flt.feature_id
LEFT JOIN dw.fact_rl_obs_stats fro ON df.feature_id = fro.feature_id
WHERE df.is_trainable = TRUE
GROUP BY df.feature_name, df.feature_type, df.tier
ORDER BY quality_score DESC NULLS LAST;

CREATE UNIQUE INDEX idx_mv_feature_quality_name ON dm.mv_feature_quality(feature_name);
COMMENT ON MATERIALIZED VIEW dm.mv_feature_quality IS 'Feature quality scorecard with IC, leakage, and clip metrics';

-- ========================================================================
-- MODEL PERFORMANCE TIMELINE
-- ========================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS dm.mv_model_performance AS
SELECT
    dm.model_id,
    dm.model_name,
    dm.algorithm,
    dm.version,
    dm.is_production,
    dm.valid_from,
    dm.valid_to,
    dm.is_current,

    -- Inference latency (from fact_inference_latency)
    ROUND(AVG(fil.latency_p50_ms)::numeric, 2) as avg_latency_p50_ms,
    ROUND(AVG(fil.latency_p95_ms)::numeric, 2) as avg_latency_p95_ms,
    ROUND(AVG(fil.latency_p99_ms)::numeric, 2) as avg_latency_p99_ms,
    ROUND(AVG(fil.throughput_eps)::numeric, 2) as avg_throughput_eps,

    -- Signal generation stats (from fact_signal_5m)
    COUNT(DISTINCT fs.ts_utc) as total_signals,
    COUNT(DISTINCT CASE WHEN fs.action = 'BUY' THEN fs.ts_utc END) as buy_signals,
    COUNT(DISTINCT CASE WHEN fs.action = 'SELL' THEN fs.ts_utc END) as sell_signals,
    COUNT(DISTINCT CASE WHEN fs.action = 'HOLD' THEN fs.ts_utc END) as hold_signals,
    ROUND(AVG(fs.confidence)::numeric, 4) as avg_confidence,

    -- Backtest summary (best test split result)
    MAX(fps.sharpe_ratio) as best_sharpe_ratio,
    MAX(fps.sortino_ratio) as best_sortino_ratio,
    MAX(fps.total_return) as best_total_return,
    MIN(fps.max_drawdown) as min_max_drawdown,

    MAX(COALESCE(fil.created_at, fs.created_at)) as last_updated

FROM dw.dim_model dm
LEFT JOIN dw.fact_inference_latency fil ON dm.model_sk = fil.model_sk
LEFT JOIN dw.fact_signal_5m fs ON dm.model_sk = fs.model_sk
LEFT JOIN dw.dim_backtest_run dr ON dm.model_sk = dr.model_sk
LEFT JOIN dw.fact_perf_summary fps ON dr.run_sk = fps.run_sk AND fps.split = 'test'
WHERE dm.created_at >= CURRENT_DATE - INTERVAL '180 days'
GROUP BY dm.model_sk, dm.model_id, dm.model_name, dm.algorithm, dm.version,
         dm.is_production, dm.valid_from, dm.valid_to, dm.is_current
ORDER BY dm.is_production DESC, dm.is_current DESC, dm.valid_from DESC;

CREATE UNIQUE INDEX idx_mv_model_performance_sk ON dm.mv_model_performance(model_id, valid_from);
COMMENT ON MATERIALIZED VIEW dm.mv_model_performance IS 'Model performance timeline with latency, signals, and backtest metrics';

-- ========================================================================
-- TRADING SESSIONS SUMMARY
-- ========================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS dm.mv_trading_sessions AS
SELECT
    dt.date_cot,
    dt.day_of_week,
    ds.symbol_code,

    -- Price action
    (SELECT fb2.open FROM dw.fact_bar_5m fb2
     JOIN dw.dim_time_5m dt2 ON fb2.time_id = dt2.time_id
     WHERE fb2.symbol_id = fb.symbol_id AND dt2.date_cot = dt.date_cot
     ORDER BY dt2.ts_utc LIMIT 1) as session_open,

    (SELECT fb2.close FROM dw.fact_bar_5m fb2
     JOIN dw.dim_time_5m dt2 ON fb2.time_id = dt2.time_id
     WHERE fb2.symbol_id = fb.symbol_id AND dt2.date_cot = dt.date_cot
     ORDER BY dt2.ts_utc DESC LIMIT 1) as session_close,

    MAX(fb.high) as session_high,
    MIN(fb.low) as session_low,
    SUM(fb.volume) as session_volume,

    -- Session metrics
    COUNT(*) as bars_count,
    ROUND(
        ((SELECT fb2.close FROM dw.fact_bar_5m fb2
          JOIN dw.dim_time_5m dt2 ON fb2.time_id = dt2.time_id
          WHERE fb2.symbol_id = fb.symbol_id AND dt2.date_cot = dt.date_cot
          ORDER BY dt2.ts_utc DESC LIMIT 1) -
         (SELECT fb2.open FROM dw.fact_bar_5m fb2
          JOIN dw.dim_time_5m dt2 ON fb2.time_id = dt2.time_id
          WHERE fb2.symbol_id = fb.symbol_id AND dt2.date_cot = dt.date_cot
          ORDER BY dt2.ts_utc LIMIT 1)) * 100.0 /
        (SELECT fb2.open FROM dw.fact_bar_5m fb2
         JOIN dw.dim_time_5m dt2 ON fb2.time_id = dt2.time_id
         WHERE fb2.symbol_id = fb.symbol_id AND dt2.date_cot = dt.date_cot
         ORDER BY dt2.ts_utc LIMIT 1),
        4
    ) as session_return_pct,

    MIN(dt.ts_cot) as session_start,
    MAX(dt.ts_cot) as session_end

FROM dw.fact_bar_5m fb
JOIN dw.dim_time_5m dt ON fb.time_id = dt.time_id
JOIN dw.dim_symbol ds ON fb.symbol_id = ds.symbol_id
WHERE dt.is_trading_hour = TRUE
AND dt.date_cot >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY dt.date_cot, dt.day_of_week, ds.symbol_code, fb.symbol_id
ORDER BY dt.date_cot DESC;

CREATE UNIQUE INDEX idx_mv_trading_sessions ON dm.mv_trading_sessions(date_cot, symbol_code);
COMMENT ON MATERIALIZED VIEW dm.mv_trading_sessions IS 'Daily trading session summary with OHLCV and returns';

-- ========================================================================
-- REFRESH FUNCTIONS
-- ========================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION dm.refresh_all_marts()
RETURNS TABLE(
    view_name TEXT,
    refresh_status TEXT,
    rows_affected BIGINT,
    duration_ms BIGINT
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    row_count BIGINT;
BEGIN
    -- mv_pipeline_health
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY dm.mv_pipeline_health;
    end_time := clock_timestamp();
    GET DIAGNOSTICS row_count = ROW_COUNT;
    view_name := 'mv_pipeline_health';
    refresh_status := 'SUCCESS';
    rows_affected := row_count;
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    RETURN NEXT;

    -- mv_daily_coverage
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY dm.mv_daily_coverage;
    end_time := clock_timestamp();
    GET DIAGNOSTICS row_count = ROW_COUNT;
    view_name := 'mv_daily_coverage';
    refresh_status := 'SUCCESS';
    rows_affected := row_count;
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    RETURN NEXT;

    -- mv_indicator_stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY dm.mv_indicator_stats;
    end_time := clock_timestamp();
    GET DIAGNOSTICS row_count = ROW_COUNT;
    view_name := 'mv_indicator_stats';
    refresh_status := 'SUCCESS';
    rows_affected := row_count;
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    RETURN NEXT;

    -- mv_backtest_leaderboard
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY dm.mv_backtest_leaderboard;
    end_time := clock_timestamp();
    GET DIAGNOSTICS row_count = ROW_COUNT;
    view_name := 'mv_backtest_leaderboard';
    refresh_status := 'SUCCESS';
    rows_affected := row_count;
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    RETURN NEXT;

    -- mv_feature_quality
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY dm.mv_feature_quality;
    end_time := clock_timestamp();
    GET DIAGNOSTICS row_count = ROW_COUNT;
    view_name := 'mv_feature_quality';
    refresh_status := 'SUCCESS';
    rows_affected := row_count;
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    RETURN NEXT;

    -- mv_model_performance
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY dm.mv_model_performance;
    end_time := clock_timestamp();
    GET DIAGNOSTICS row_count = ROW_COUNT;
    view_name := 'mv_model_performance';
    refresh_status := 'SUCCESS';
    rows_affected := row_count;
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    RETURN NEXT;

    -- mv_trading_sessions
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY dm.mv_trading_sessions;
    end_time := clock_timestamp();
    GET DIAGNOSTICS row_count = ROW_COUNT;
    view_name := 'mv_trading_sessions';
    refresh_status := 'SUCCESS';
    rows_affected := row_count;
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    RETURN NEXT;

EXCEPTION WHEN OTHERS THEN
    view_name := 'ERROR';
    refresh_status := SQLERRM;
    rows_affected := 0;
    duration_ms := 0;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION dm.refresh_all_marts IS 'Refresh all materialized views in dm schema';

-- ========================================================================
-- COMPLETION MESSAGE
-- ========================================================================

DO $$
DECLARE
    view_count INT;
BEGIN
    SELECT COUNT(*) INTO view_count
    FROM pg_matviews
    WHERE schemaname = 'dm';

    RAISE NOTICE '';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE '✅ Data Marts Created Successfully!';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE 'Created % materialized views:', view_count;
    RAISE NOTICE '  - mv_pipeline_health         (Pipeline health summary)';
    RAISE NOTICE '  - mv_daily_coverage          (Daily data coverage)';
    RAISE NOTICE '  - mv_indicator_stats         (Technical indicator statistics)';
    RAISE NOTICE '  - mv_backtest_leaderboard    (Backtest performance ranking)';
    RAISE NOTICE '  - mv_feature_quality         (Feature quality scorecard)';
    RAISE NOTICE '  - mv_model_performance       (Model performance timeline)';
    RAISE NOTICE '  - mv_trading_sessions        (Daily trading sessions)';
    RAISE NOTICE '';
    RAISE NOTICE 'Created refresh function:';
    RAISE NOTICE '  - dm.refresh_all_marts()     (Refresh all views)';
    RAISE NOTICE '';
    RAISE NOTICE 'Usage:';
    RAISE NOTICE '  SELECT * FROM dm.refresh_all_marts();';
    RAISE NOTICE '';
    RAISE NOTICE 'Note: Run refresh after populating fact tables with data';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE '';
END $$;
