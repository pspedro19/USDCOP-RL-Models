-- =============================================================================
-- Migration: 013_trades_audit.sql
-- =============================================================================
-- Creates the trades audit table for comprehensive trade logging and compliance.
--
-- P1: Trades Audit Table
--
-- Features:
-- - Complete trade lifecycle logging
-- - Model version tracking
-- - Risk metrics capture
-- - Feature snapshot at trade time
-- - Compliance-ready audit trail
--
-- Author: Trading Team
-- Date: 2026-01-17
-- =============================================================================

-- Create audit schema if not exists
CREATE SCHEMA IF NOT EXISTS audit;

-- =============================================================================
-- Trades Audit Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit.trades_audit (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- Trade identification
    trade_id UUID NOT NULL,
    correlation_id UUID,  -- Links to request that initiated trade

    -- Trade details
    action VARCHAR(10) NOT NULL CHECK (action IN ('LONG', 'SHORT', 'HOLD', 'CLOSE')),
    symbol VARCHAR(20) NOT NULL DEFAULT 'USDCOP',
    entry_price DECIMAL(15, 6),
    exit_price DECIMAL(15, 6),
    quantity DECIMAL(15, 6),
    position_size_usd DECIMAL(15, 2),

    -- P&L and metrics
    pnl DECIMAL(15, 6),
    pnl_percentage DECIMAL(10, 6),
    realized_pnl DECIMAL(15, 6),
    unrealized_pnl DECIMAL(15, 6),
    commission DECIMAL(10, 6),
    slippage DECIMAL(10, 6),

    -- Model information
    model_id VARCHAR(100),
    model_version VARCHAR(50),
    model_confidence DECIMAL(5, 4),  -- 0.0000 to 1.0000
    model_action_probabilities JSONB,  -- {"LONG": 0.7, "SHORT": 0.2, "HOLD": 0.1}

    -- Risk metrics at trade time
    drawdown_at_trade DECIMAL(10, 6),
    volatility_at_trade DECIMAL(10, 6),
    sharpe_ratio_at_trade DECIMAL(10, 6),
    risk_score DECIMAL(5, 2),

    -- Feature snapshot (for debugging/analysis)
    feature_snapshot JSONB,
    feature_hash VARCHAR(64),  -- SHA256 of features for verification

    -- Timing
    signal_timestamp TIMESTAMPTZ,  -- When signal was generated
    execution_timestamp TIMESTAMPTZ,  -- When trade was executed
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Source and environment
    source VARCHAR(50) DEFAULT 'inference_api',  -- Where trade originated
    environment VARCHAR(20) DEFAULT 'paper',  -- paper, staging, production
    is_paper_trade BOOLEAN DEFAULT TRUE,

    -- Additional metadata
    metadata JSONB,
    notes TEXT
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Primary lookup indexes
CREATE INDEX IF NOT EXISTS idx_trades_audit_trade_id ON audit.trades_audit(trade_id);
CREATE INDEX IF NOT EXISTS idx_trades_audit_created_at ON audit.trades_audit(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_audit_symbol ON audit.trades_audit(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_audit_action ON audit.trades_audit(action);

-- Model tracking indexes
CREATE INDEX IF NOT EXISTS idx_trades_audit_model_id ON audit.trades_audit(model_id);
CREATE INDEX IF NOT EXISTS idx_trades_audit_model_version ON audit.trades_audit(model_version);

-- Time-based partitioning support
CREATE INDEX IF NOT EXISTS idx_trades_audit_execution_time ON audit.trades_audit(execution_timestamp DESC);

-- Environment filtering
CREATE INDEX IF NOT EXISTS idx_trades_audit_environment ON audit.trades_audit(environment);

-- Correlation tracking
CREATE INDEX IF NOT EXISTS idx_trades_audit_correlation ON audit.trades_audit(correlation_id);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trades_audit_symbol_time
    ON audit.trades_audit(symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_trades_audit_model_time
    ON audit.trades_audit(model_id, created_at DESC);

-- =============================================================================
-- Trade Events Table (for lifecycle tracking)
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit.trade_events (
    id SERIAL PRIMARY KEY,
    trade_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- SIGNAL_GENERATED, ORDER_PLACED, ORDER_FILLED, etc.
    event_timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_data JSONB,
    source VARCHAR(50),

    CONSTRAINT fk_trade_events_trade
        FOREIGN KEY (trade_id)
        REFERENCES audit.trades_audit(trade_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trade_events_trade_id ON audit.trade_events(trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_events_type ON audit.trade_events(event_type);
CREATE INDEX IF NOT EXISTS idx_trade_events_timestamp ON audit.trade_events(event_timestamp DESC);

-- =============================================================================
-- Daily Trade Summary (materialized for performance)
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit.daily_trade_summary (
    id SERIAL PRIMARY KEY,
    summary_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT 'USDCOP',
    model_id VARCHAR(100),
    environment VARCHAR(20),

    -- Trade counts
    total_trades INTEGER DEFAULT 0,
    long_trades INTEGER DEFAULT 0,
    short_trades INTEGER DEFAULT 0,
    hold_signals INTEGER DEFAULT 0,

    -- P&L summary
    total_pnl DECIMAL(15, 6) DEFAULT 0,
    total_realized_pnl DECIMAL(15, 6) DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4),  -- 0.0000 to 1.0000

    -- Risk metrics
    max_drawdown DECIMAL(10, 6),
    avg_trade_duration_minutes DECIMAL(10, 2),
    sharpe_ratio DECIMAL(10, 6),

    -- Timing
    first_trade_at TIMESTAMPTZ,
    last_trade_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uk_daily_summary UNIQUE (summary_date, symbol, model_id, environment)
);

CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON audit.daily_trade_summary(summary_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_summary_model ON audit.daily_trade_summary(model_id);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to log a trade
CREATE OR REPLACE FUNCTION audit.log_trade(
    p_trade_id UUID,
    p_action VARCHAR(10),
    p_symbol VARCHAR(20),
    p_entry_price DECIMAL(15, 6),
    p_quantity DECIMAL(15, 6),
    p_model_id VARCHAR(100),
    p_model_version VARCHAR(50),
    p_model_confidence DECIMAL(5, 4),
    p_feature_snapshot JSONB DEFAULT NULL,
    p_environment VARCHAR(20) DEFAULT 'paper',
    p_metadata JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_feature_hash VARCHAR(64);
BEGIN
    -- Compute feature hash if snapshot provided
    IF p_feature_snapshot IS NOT NULL THEN
        v_feature_hash := encode(sha256(p_feature_snapshot::text::bytea), 'hex');
    END IF;

    INSERT INTO audit.trades_audit (
        trade_id,
        action,
        symbol,
        entry_price,
        quantity,
        model_id,
        model_version,
        model_confidence,
        feature_snapshot,
        feature_hash,
        signal_timestamp,
        environment,
        is_paper_trade,
        metadata
    ) VALUES (
        p_trade_id,
        p_action,
        p_symbol,
        p_entry_price,
        p_quantity,
        p_model_id,
        p_model_version,
        p_model_confidence,
        p_feature_snapshot,
        v_feature_hash,
        NOW(),
        p_environment,
        p_environment != 'production',
        p_metadata
    );

    -- Log the trade event
    INSERT INTO audit.trade_events (trade_id, event_type, event_data, source)
    VALUES (p_trade_id, 'SIGNAL_GENERATED', jsonb_build_object('action', p_action, 'confidence', p_model_confidence), 'audit.log_trade');

    RETURN p_trade_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update trade with exit
CREATE OR REPLACE FUNCTION audit.close_trade(
    p_trade_id UUID,
    p_exit_price DECIMAL(15, 6),
    p_pnl DECIMAL(15, 6),
    p_commission DECIMAL(10, 6) DEFAULT 0,
    p_slippage DECIMAL(10, 6) DEFAULT 0
)
RETURNS VOID AS $$
BEGIN
    UPDATE audit.trades_audit
    SET
        exit_price = p_exit_price,
        pnl = p_pnl,
        realized_pnl = p_pnl,
        commission = p_commission,
        slippage = p_slippage,
        execution_timestamp = NOW()
    WHERE trade_id = p_trade_id;

    -- Log the close event
    INSERT INTO audit.trade_events (trade_id, event_type, event_data, source)
    VALUES (p_trade_id, 'TRADE_CLOSED', jsonb_build_object('exit_price', p_exit_price, 'pnl', p_pnl), 'audit.close_trade');
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON TABLE audit.trades_audit IS 'Comprehensive trade audit log for compliance and analysis';
COMMENT ON TABLE audit.trade_events IS 'Trade lifecycle events for detailed tracking';
COMMENT ON TABLE audit.daily_trade_summary IS 'Daily aggregated trade statistics';
COMMENT ON FUNCTION audit.log_trade IS 'Log a new trade with all relevant metadata';
COMMENT ON FUNCTION audit.close_trade IS 'Update trade record with exit information';

-- =============================================================================
-- Migration metadata
-- =============================================================================
INSERT INTO public.schema_migrations (version, description, applied_at)
VALUES ('013', 'Create trades audit tables', NOW())
ON CONFLICT (version) DO NOTHING;
