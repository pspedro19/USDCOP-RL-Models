-- Paper Trading Tables for PPO V1 Dashboard
-- Created: 7 January 2026

-- ============================================
-- Table 1: trading_state (Estado actual del modelo)
-- ============================================
CREATE TABLE IF NOT EXISTS trading_state (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL UNIQUE,

    -- Posición actual
    position VARCHAR(10) DEFAULT 'FLAT' CHECK (position IN ('LONG', 'SHORT', 'FLAT')),
    entry_price DECIMAL(12,4),
    entry_time TIMESTAMPTZ,
    bars_in_position INT DEFAULT 0,

    -- PnL
    unrealized_pnl DECIMAL(12,4) DEFAULT 0,
    realized_pnl DECIMAL(12,4) DEFAULT 0,

    -- Equity tracking
    equity DECIMAL(14,4) DEFAULT 10000,
    peak_equity DECIMAL(14,4) DEFAULT 10000,
    drawdown_pct DECIMAL(6,4) DEFAULT 0,

    -- Estadísticas
    trade_count INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,

    -- Metadata
    last_signal VARCHAR(10),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índice para queries rápidos
CREATE INDEX IF NOT EXISTS idx_trading_state_model ON trading_state(model_id);

-- ============================================
-- Table 2: trades_history (Historial de operaciones)
-- ============================================
CREATE TABLE IF NOT EXISTS trades_history (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,

    -- Detalles del trade
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    entry_price DECIMAL(12,4) NOT NULL,
    exit_price DECIMAL(12,4),
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    duration_bars INT,

    -- Resultado
    pnl_usd DECIMAL(12,4),
    pnl_pct DECIMAL(8,4),
    exit_reason VARCHAR(20),  -- SIGNAL_CHANGE, STOP_LOSS, END_OF_DAY, KILL_SWITCH

    -- Estado al momento del trade
    equity_at_entry DECIMAL(14,4),
    equity_at_exit DECIMAL(14,4),
    drawdown_at_entry DECIMAL(6,4),

    -- Metadata
    bar_number INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_trades_model ON trades_history(model_id);
CREATE INDEX IF NOT EXISTS idx_trades_time ON trades_history(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_exit ON trades_history(exit_time DESC);

-- ============================================
-- Table 3: equity_snapshots (Para gráfico de equity)
-- ============================================
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    equity DECIMAL(14,4) NOT NULL,
    drawdown_pct DECIMAL(6,4),
    position VARCHAR(10),
    bar_close_price DECIMAL(12,4),

    UNIQUE(model_id, timestamp)
);

-- Índice para queries de curva
CREATE INDEX IF NOT EXISTS idx_equity_model_time ON equity_snapshots(model_id, timestamp DESC);

-- Try to create hypertable if TimescaleDB is available (will fail gracefully if not)
DO $$
BEGIN
    PERFORM create_hypertable('equity_snapshots', 'timestamp', if_not_exists => TRUE);
EXCEPTION
    WHEN others THEN
        RAISE NOTICE 'TimescaleDB hypertable creation skipped (extension may not be available)';
END;
$$;

-- ============================================
-- Initial data for PPO V1
-- ============================================
INSERT INTO trading_state (model_id, equity, peak_equity, position, last_signal)
VALUES ('ppo_v1', 10000, 10000, 'FLAT', 'HOLD')
ON CONFLICT (model_id) DO UPDATE SET
    last_updated = NOW();

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Paper trading tables created successfully!';
END;
$$;
