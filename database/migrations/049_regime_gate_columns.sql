-- Migration 048: Add regime gate + dynamic leverage columns to forecast_h5_signals
-- Date: 2026-04-06
-- Required for: Smart Simple v2.0 (regime gate + effective HS + dynamic leverage)

ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS regime VARCHAR(20);
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS hurst_exponent DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS regime_leverage_scaler DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS rolling_wr_8w DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS dl_leverage_scaler DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS effective_hs_pct DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS effective_tp_pct DOUBLE PRECISION;

-- Add regime column to executions for tracking
ALTER TABLE forecast_h5_executions ADD COLUMN IF NOT EXISTS regime VARCHAR(20);
ALTER TABLE forecast_h5_executions ADD COLUMN IF NOT EXISTS hurst_exponent DOUBLE PRECISION;
