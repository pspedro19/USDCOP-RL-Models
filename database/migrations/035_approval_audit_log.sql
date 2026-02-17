-- Migration 035: Approval Audit Log Table
-- Tracks all human approval/rejection decisions (segundo voto)
-- Created: 2026-01-31

-- Audit log for all approvals/rejections
CREATE TABLE IF NOT EXISTS approval_audit_log (
    id SERIAL PRIMARY KEY,

    -- Action taken
    action VARCHAR(50) NOT NULL CHECK (
        action IN ('APPROVE', 'REJECT', 'REQUEST_MORE_TESTS', 'EXPIRE', 'ARCHIVE', 'ROLLBACK')
    ),

    -- Model and proposal reference
    model_id VARCHAR(255) NOT NULL,
    proposal_id VARCHAR(255) REFERENCES promotion_proposals(proposal_id),

    -- Reviewer info
    reviewer VARCHAR(255) NOT NULL,
    reviewer_email VARCHAR(255),
    notes TEXT,

    -- Context
    previous_production_model VARCHAR(255),

    -- Client info for security audit
    client_ip VARCHAR(45),
    user_agent TEXT,

    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_model
    ON approval_audit_log (model_id);

CREATE INDEX IF NOT EXISTS idx_audit_log_created
    ON approval_audit_log (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_reviewer
    ON approval_audit_log (reviewer);

CREATE INDEX IF NOT EXISTS idx_audit_log_action
    ON approval_audit_log (action, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_proposal
    ON approval_audit_log (proposal_id);

-- Comments
COMMENT ON TABLE approval_audit_log IS
    'Audit trail for all model promotion decisions (segundo voto)';

COMMENT ON COLUMN approval_audit_log.action IS
    'APPROVE: promoted to production, REJECT: denied promotion, ROLLBACK: reverted, ARCHIVE: manually archived';

COMMENT ON COLUMN approval_audit_log.previous_production_model IS
    'Model ID that was in production before this action (for rollback reference)';
