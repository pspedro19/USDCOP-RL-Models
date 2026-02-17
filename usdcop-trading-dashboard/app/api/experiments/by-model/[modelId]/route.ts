/**
 * GET /api/experiments/by-model/[modelId] - Get pending experiment for a specific model
 *
 * Returns the most recent PENDING_APPROVAL proposal for the given model_id
 */
import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';
import { protectApiRoute } from '@/lib/auth/api-auth';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

export async function GET(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  // Validate authentication (SKIP_AUTH=true bypasses auth for development)
  const skipAuth = process.env.SKIP_AUTH === 'true';
  if (!skipAuth) {
    const authResult = await protectApiRoute(request, { rateLimit: false });
    if (!authResult.authenticated) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  const modelId = params.modelId;

  try {
    // Get the most recent pending proposal for this model
    const result = await pool.query(`
      SELECT
        pp.id,
        pp.proposal_id,
        pp.model_id,
        pp.experiment_name,
        pp.recommendation,
        pp.confidence,
        pp.reason,
        pp.metrics,
        pp.vs_baseline,
        pp.criteria_results,
        pp.lineage,
        pp.status,
        pp.reviewer,
        pp.reviewer_email,
        pp.reviewer_notes,
        pp.reviewed_at,
        pp.created_at,
        pp.expires_at,
        EXTRACT(EPOCH FROM (pp.expires_at - NOW())) / 3600 as hours_until_expiry
      FROM promotion_proposals pp
      WHERE pp.model_id = $1
        AND pp.status = 'PENDING_APPROVAL'
        AND (pp.expires_at IS NULL OR pp.expires_at > NOW())
      ORDER BY pp.created_at DESC
      LIMIT 1
    `, [modelId]);

    if (result.rows.length === 0) {
      return NextResponse.json({ experiment: null });
    }

    const row = result.rows[0];

    // Get audit log for this proposal
    const auditResult = await pool.query(`
      SELECT action, reviewer, notes, created_at
      FROM approval_audit_log
      WHERE proposal_id = $1
      ORDER BY created_at DESC
    `, [row.proposal_id]);

    const experiment = {
      id: row.id,
      proposalId: row.proposal_id,
      modelId: row.model_id,
      experimentName: row.experiment_name,
      recommendation: row.recommendation,
      confidence: parseFloat(row.confidence),
      reason: row.reason,
      metrics: row.metrics,
      vsBaseline: row.vs_baseline,
      criteriaResults: row.criteria_results,
      lineage: row.lineage,
      status: row.status,
      reviewer: row.reviewer,
      reviewerEmail: row.reviewer_email,
      reviewerNotes: row.reviewer_notes,
      reviewedAt: row.reviewed_at,
      createdAt: row.created_at,
      expiresAt: row.expires_at,
      hoursUntilExpiry: row.hours_until_expiry ? parseFloat(row.hours_until_expiry) : null,
      auditLog: auditResult.rows.map((audit) => ({
        action: audit.action,
        reviewer: audit.reviewer,
        notes: audit.notes,
        createdAt: audit.created_at,
      })),
    };

    return NextResponse.json({ experiment });
  } catch (error) {
    console.error('Error fetching experiment by model:', error);
    return NextResponse.json(
      { error: 'Failed to fetch experiment' },
      { status: 500 }
    );
  }
}
