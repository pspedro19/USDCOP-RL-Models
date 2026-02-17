-- =============================================================================
-- Paper Trading Status Check
-- =============================================================================
-- Run: docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -f scripts/check_paper_trading.sql
-- =============================================================================

\echo '=== PAPER TRADING PERFORMANCE ==='
SELECT * FROM v_paper_trading_performance;

\echo ''
\echo '=== LAST 10 TRADING DAYS ==='
SELECT
    signal_date,
    CASE WHEN signal_direction = 1 THEN 'LONG' ELSE 'SHORT' END AS dir,
    ROUND(signal_leverage::NUMERIC, 3) AS leverage,
    ROUND(entry_price::NUMERIC, 2) AS entry,
    ROUND(close_price::NUMERIC, 2) AS close,
    ROUND((actual_return_1d * 100)::NUMERIC, 4) AS actual_ret_pct,
    ROUND((strategy_return * 100)::NUMERIC, 4) AS strat_ret_pct,
    ROUND(running_da_pct::NUMERIC, 1) AS da_pct,
    ROUND((running_max_drawdown * 100)::NUMERIC, 2) AS maxdd_pct,
    n_days_traded
FROM forecast_paper_trading
ORDER BY signal_date DESC
LIMIT 10;

\echo ''
\echo '=== LATEST SIGNAL ==='
SELECT
    signal_date,
    CASE WHEN forecast_direction = 1 THEN 'LONG' ELSE 'SHORT' END AS direction,
    ROUND(clipped_leverage::NUMERIC, 3) AS leverage,
    ROUND(position_size::NUMERIC, 3) AS position,
    ROUND(realized_vol_21d::NUMERIC, 4) AS vol_21d,
    ROUND(forecast_return::NUMERIC, 6) AS forecast_ret,
    ensemble_models,
    config_version,
    created_at
FROM forecast_vol_targeting_signals
ORDER BY signal_date DESC
LIMIT 1;

\echo ''
\echo '=== STOP CRITERIA CHECK ==='
SELECT
    n_days_traded AS days,
    ROUND(running_da_pct::NUMERIC, 1) AS da_pct,
    ROUND((cumulative_return - 1) * 100::NUMERIC, 2) AS cum_ret_pct,
    ROUND(running_sharpe::NUMERIC, 3) AS sharpe,
    ROUND(ABS(running_max_drawdown) * 100::NUMERIC, 2) AS maxdd_pct,
    CASE
        WHEN ABS(running_max_drawdown) > 0.20 THEN 'STOP: MaxDD > 20%'
        WHEN n_days_traded >= 60 AND running_da_pct < 46 THEN 'STOP: DA < 46% @ 60d'
        WHEN ABS(running_max_drawdown) > 0.15 THEN 'PAUSE: MaxDD > 15%'
        WHEN n_days_traded >= 40 AND running_da_pct < 48 THEN 'PAUSE: DA < 48% @ 40d'
        WHEN running_da_pct > 50 AND cumulative_return > 1.0 THEN 'CONTINUE (DA > 50%, ret > 0)'
        WHEN n_days_traded < 20 THEN 'TOO EARLY (< 20 days)'
        ELSE 'MONITORING'
    END AS status
FROM forecast_paper_trading
ORDER BY signal_date DESC
LIMIT 1;
