-- ============================================================================
-- Migration 083: make the H5 performance view strategy-safe after migration 064
-- Contract: C-012
--
-- Migration 064 permits one signal and execution per (signal_date, strategy_id).
-- The view installed by 050 joined only on signal_date, which cross-associated
-- different strategies once more than one strategy existed on the same date.
-- Keep the existing view columns in their original order and append strategy_id
-- so CREATE OR REPLACE VIEW remains compatible with existing readers.
-- ============================================================================

BEGIN;

CREATE OR REPLACE VIEW v_h5_performance_summary AS
SELECT
    e.inference_year,
    e.inference_week,
    e.signal_date,
    e.direction,
    e.leverage,
    e.entry_price,
    e.exit_price,
    e.exit_reason,
    e.week_pnl_pct,
    e.status,
    e.confidence_tier,
    e.regime,
    e.hurst_exponent,
    s.ensemble_return,
    s.skip_trade,
    s.regime_leverage_scaler,
    s.dl_leverage_scaler,
    e.strategy_id
FROM forecast_h5_executions e
LEFT JOIN forecast_h5_signals s
    ON e.signal_date = s.signal_date
   AND e.strategy_id = s.strategy_id
ORDER BY e.signal_date DESC, e.strategy_id;

COMMIT;
