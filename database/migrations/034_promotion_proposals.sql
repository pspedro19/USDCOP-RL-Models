-- Migration 034: Promotion Proposals Table
-- Part of L4 Backtest + Promotion system (primer voto)
-- Created: 2026-01-31

-- Promotion proposals from L4 (primer voto)
-- Each proposal requires human approval in Dashboard (segundo voto)
CREATE TABLE IF NOT EXISTS promotion_proposals (
    id SERIAL PRIMARY KEY,

    -- Identity
    proposal_id VARCHAR(255) UNIQUE NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,

    -- L4 Recommendation (primer voto)
    recommendation VARCHAR(20) NOT NULL CHECK (
        recommendation IN ('PROMOTE', 'REJECT', 'REVIEW')
    ),
    confidence DECIMAL(5,4),
    reason TEXT,

    -- Backtest metrics
    metrics JSONB NOT NULL,

    -- Comparison vs baseline
    vs_baseline JSONB,
    baseline_model_id VARCHAR(255),

    -- Criteria evaluation results
    criteria_results JSONB NOT NULL,

    -- Complete lineage chain
    lineage JSONB NOT NULL,

    -- Status tracking
    status VARCHAR(30) DEFAULT 'PENDING_APPROVAL' CHECK (
        status IN ('PENDING_APPROVAL', 'APPROVED', 'REJECTED', 'EXPIRED')
    ),

    -- Human reviewer info (segundo voto)
    reviewer VARCHAR(255),
    reviewer_email VARCHAR(255),
    reviewer_notes TEXT,
    reviewed_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days')
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_promotion_proposals_status
    ON promotion_proposals (status);

CREATE INDEX IF NOT EXISTS idx_promotion_proposals_experiment
    ON promotion_proposals (experiment_name);

CREATE INDEX IF NOT EXISTS idx_promotion_proposals_created
    ON promotion_proposals (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_promotion_proposals_pending
    ON promotion_proposals (status, expires_at)
    WHERE status = 'PENDING_APPROVAL';

CREATE INDEX IF NOT EXISTS idx_promotion_proposals_model
    ON promotion_proposals (model_id);

-- Comment on table
COMMENT ON TABLE promotion_proposals IS
    'L4 Backtest + Promotion proposals requiring human approval (2-vote system)';

COMMENT ON COLUMN promotion_proposals.recommendation IS
    'L4 automatic recommendation: PROMOTE (all criteria passed), REJECT (failed), REVIEW (edge case)';

COMMENT ON COLUMN promotion_proposals.status IS
    'PENDING_APPROVAL: awaiting human review, APPROVED: promoted to production, REJECTED: denied, EXPIRED: timeout';
