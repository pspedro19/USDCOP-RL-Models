-- =============================================================================
-- Migration 044: Smart Simple v1.0 — Confidence + Adaptive Stops Columns
-- =============================================================================
--
-- Adds nullable columns to forecast_h5_signals and forecast_h5_executions
-- for confidence scoring and adaptive stop levels.
--
-- Backward-compatible: all new columns are nullable, existing data unaffected.
-- Existing trailing_state/cooldown_until columns in subtrades become unused
-- but are kept (harmless, avoids destructive migration).
--
-- Contract: FC-H5-SIMPLE-001
-- Date: 2026-02-16
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- forecast_h5_signals: confidence scoring columns
-- ---------------------------------------------------------------------------
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS confidence_tier VARCHAR(10);
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS confidence_agreement DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS confidence_magnitude DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS sizing_multiplier DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS skip_trade BOOLEAN DEFAULT FALSE;

-- forecast_h5_signals: adaptive stop levels
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS hard_stop_pct DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS take_profit_pct DOUBLE PRECISION;
ALTER TABLE forecast_h5_signals ADD COLUMN IF NOT EXISTS adjusted_leverage DOUBLE PRECISION;

-- ---------------------------------------------------------------------------
-- forecast_h5_executions: confidence context for audit
-- ---------------------------------------------------------------------------
ALTER TABLE forecast_h5_executions ADD COLUMN IF NOT EXISTS confidence_tier VARCHAR(10);
ALTER TABLE forecast_h5_executions ADD COLUMN IF NOT EXISTS hard_stop_pct DOUBLE PRECISION;
ALTER TABLE forecast_h5_executions ADD COLUMN IF NOT EXISTS take_profit_pct DOUBLE PRECISION;

COMMIT;
