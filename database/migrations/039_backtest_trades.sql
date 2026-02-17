-- Migration 039: Backtest Trades Persistence for L4 Validation
-- Created: 2026-02-04
--
-- Purpose: Store individual trades from L4 backtest validation so that
-- the frontend can replay the EXACT same backtest that L4 evaluated.
--
-- This ensures 100% consistency between:
-- - The metrics shown in FloatingExperimentPanel (from promotion_proposals)
-- - The trades shown in the backtest replay chart (from this table)
--
-- Architecture:
--   L4 Backtest ──► backtest_trades (trades) + promotion_proposals (metrics)
--                          │
--                          ▼
--   Frontend ◄──── /api/experiments/[id]/trades ◄──── backtest_trades

-- =============================================================================
-- BACKTEST TRADES TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_trades (
    id SERIAL PRIMARY KEY,

    -- Link to promotion proposal (L4's output)
    proposal_id VARCHAR(255) NOT NULL,

    -- Trade identification
    trade_id INTEGER NOT NULL,
    model_id VARCHAR(255) NOT NULL,

    -- Timestamps
    timestamp TIMESTAMPTZ NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,

    -- Trade details
    side VARCHAR(10) NOT NULL,  -- 'LONG' or 'SHORT'
    entry_price DECIMAL(12,4) NOT NULL,
    exit_price DECIMAL(12,4),

    -- P&L
    pnl DECIMAL(12,4),           -- In price points
    pnl_usd DECIMAL(12,2),       -- In USD
    pnl_percent DECIMAL(8,4),    -- As percentage

    -- Status
    status VARCHAR(20) DEFAULT 'closed',  -- 'open', 'closed'
    duration_minutes INTEGER,
    exit_reason VARCHAR(50),     -- 'take_profit', 'stop_loss', 'signal', 'end_of_period'

    -- Equity tracking (for replay)
    equity_at_entry DECIMAL(14,2),
    equity_at_exit DECIMAL(14,2),

    -- Model confidence
    entry_confidence DECIMAL(5,4),
    exit_confidence DECIMAL(5,4),

    -- Raw model output (for debugging)
    raw_action DECIMAL(10,6),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT backtest_trades_side_check
        CHECK (side IN ('LONG', 'SHORT', 'BUY', 'SELL')),
    CONSTRAINT backtest_trades_status_check
        CHECK (status IN ('open', 'closed', 'pending'))
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Primary lookup: Get all trades for a proposal
CREATE INDEX IF NOT EXISTS idx_backtest_trades_proposal
    ON backtest_trades(proposal_id);

-- For ordered replay
CREATE INDEX IF NOT EXISTS idx_backtest_trades_proposal_time
    ON backtest_trades(proposal_id, entry_time);

-- For model-based queries
CREATE INDEX IF NOT EXISTS idx_backtest_trades_model
    ON backtest_trades(model_id);

-- For date range queries
CREATE INDEX IF NOT EXISTS idx_backtest_trades_timestamp
    ON backtest_trades(timestamp);

-- =============================================================================
-- FOREIGN KEY (if promotion_proposals exists)
-- =============================================================================

DO $$
BEGIN
    -- Only add FK if promotion_proposals table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_name = 'promotion_proposals') THEN

        -- Check if FK already exists
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                       WHERE constraint_name = 'fk_backtest_trades_proposal') THEN
            ALTER TABLE backtest_trades
                ADD CONSTRAINT fk_backtest_trades_proposal
                FOREIGN KEY (proposal_id)
                REFERENCES promotion_proposals(proposal_id)
                ON DELETE CASCADE;
        END IF;
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not add FK constraint: %', SQLERRM;
END $$;

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to get trade summary for a proposal
CREATE OR REPLACE FUNCTION get_backtest_trade_summary(p_proposal_id VARCHAR)
RETURNS TABLE (
    total_trades BIGINT,
    winning_trades BIGINT,
    losing_trades BIGINT,
    total_pnl_usd DECIMAL,
    win_rate DECIMAL,
    first_trade TIMESTAMPTZ,
    last_trade TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_trades,
        COUNT(*) FILTER (WHERE pnl_usd > 0)::BIGINT as winning_trades,
        COUNT(*) FILTER (WHERE pnl_usd <= 0)::BIGINT as losing_trades,
        COALESCE(SUM(pnl_usd), 0)::DECIMAL as total_pnl_usd,
        CASE
            WHEN COUNT(*) > 0
            THEN (COUNT(*) FILTER (WHERE pnl_usd > 0)::DECIMAL / COUNT(*)::DECIMAL * 100)
            ELSE 0
        END as win_rate,
        MIN(entry_time) as first_trade,
        MAX(exit_time) as last_trade
    FROM backtest_trades
    WHERE proposal_id = p_proposal_id;
END;
$$ LANGUAGE plpgsql;

-- Function to delete old backtest trades (cleanup)
CREATE OR REPLACE FUNCTION cleanup_old_backtest_trades(days_to_keep INTEGER DEFAULT 90)
RETURNS BIGINT AS $$
DECLARE
    deleted_count BIGINT;
BEGIN
    DELETE FROM backtest_trades
    WHERE created_at < NOW() - (days_to_keep || ' days')::INTERVAL
    AND proposal_id NOT IN (
        -- Keep trades for proposals that are still pending or recently approved
        SELECT proposal_id FROM promotion_proposals
        WHERE status IN ('PENDING_APPROVAL', 'APPROVED')
        OR updated_at > NOW() - INTERVAL '30 days'
    );

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE backtest_trades IS
    'Stores individual trades from L4 backtest validation for exact replay in frontend';

COMMENT ON COLUMN backtest_trades.proposal_id IS
    'Links to promotion_proposals.proposal_id - the L4 validation that generated these trades';

COMMENT ON COLUMN backtest_trades.equity_at_entry IS
    'Portfolio equity before this trade - enables equity curve reconstruction';

COMMENT ON COLUMN backtest_trades.equity_at_exit IS
    'Portfolio equity after this trade - enables equity curve reconstruction';

COMMENT ON COLUMN backtest_trades.raw_action IS
    'Raw continuous action from PPO model before discretization (for debugging)';

COMMENT ON FUNCTION get_backtest_trade_summary(VARCHAR) IS
    'Returns aggregate statistics for trades in a backtest proposal';

COMMENT ON FUNCTION cleanup_old_backtest_trades(INTEGER) IS
    'Removes old backtest trades while preserving pending/recent proposals';
