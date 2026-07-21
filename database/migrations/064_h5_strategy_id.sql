-- ============================================================================
-- Migration 064: strategy_id en las tablas H5 (bloqueador R5a, plan COP Fase A1)
-- Contract: CTR-STRAT-REGISTRY-001 (extiende) · encontrado por la revision
-- adversarial de Codex 2026-07-21: la unicidad por signal_date sola hace que dos
-- estrategias (v11 y la candidata v12) COLISIONEN el mismo lunes.
-- ============================================================================
-- Diseno:
--   * DEFAULT 'smart_simple_v11' -> los writers actuales de v11 no cambian
--     (backfill implicito de todo el historico, que ES de v11).
--   * forecast_h5_predictions NO recibe strategy_id: las predicciones son output
--     de MODELOS (ridge/br), compartidas entre estrategias; v12 consume las mismas.
--   * forecast_h5_subtrades hereda via execution_id.

ALTER TABLE forecast_h5_signals
    ADD COLUMN IF NOT EXISTS strategy_id TEXT NOT NULL DEFAULT 'smart_simple_v11';
ALTER TABLE forecast_h5_executions
    ADD COLUMN IF NOT EXISTS strategy_id TEXT NOT NULL DEFAULT 'smart_simple_v11';
ALTER TABLE forecast_h5_paper_trading
    ADD COLUMN IF NOT EXISTS strategy_id TEXT NOT NULL DEFAULT 'smart_simple_v11';

-- unicidad: (signal_date) -> (signal_date, strategy_id)
ALTER TABLE forecast_h5_signals
    DROP CONSTRAINT IF EXISTS forecast_h5_signals_signal_date_key;
ALTER TABLE forecast_h5_signals
    ADD CONSTRAINT uq_h5_signal_date_strategy UNIQUE (signal_date, strategy_id);

ALTER TABLE forecast_h5_executions
    DROP CONSTRAINT IF EXISTS forecast_h5_executions_signal_date_key;
ALTER TABLE forecast_h5_executions
    ADD CONSTRAINT uq_h5_exec_date_strategy UNIQUE (signal_date, strategy_id);

ALTER TABLE forecast_h5_paper_trading
    DROP CONSTRAINT IF EXISTS forecast_h5_paper_trading_signal_date_key;
ALTER TABLE forecast_h5_paper_trading
    ADD CONSTRAINT uq_h5_paper_date_strategy UNIQUE (signal_date, strategy_id);

CREATE INDEX IF NOT EXISTS ix_h5_signals_strategy ON forecast_h5_signals (strategy_id);
CREATE INDEX IF NOT EXISTS ix_h5_exec_strategy ON forecast_h5_executions (strategy_id);
CREATE INDEX IF NOT EXISTS ix_h5_paper_strategy ON forecast_h5_paper_trading (strategy_id);
