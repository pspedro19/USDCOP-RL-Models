/**
 * GET /api/experiments/[id] - Get experiment details by proposal ID
 */
import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';
import { protectApiRoute } from '@/lib/auth/api-auth';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  // Validate authentication (SKIP_AUTH=true bypasses auth for development)
  const skipAuth = process.env.SKIP_AUTH === 'true';
  if (!skipAuth) {
    const authResult = await protectApiRoute(request, { rateLimit: false });
    if (!authResult.authenticated) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  const proposalId = params.id;

  try {
    // Get experiment details
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
        pp.expires_at
      FROM promotion_proposals pp
      WHERE pp.proposal_id = $1
    `, [proposalId]);

    if (result.rows.length === 0) {
      return NextResponse.json(
        { error: 'Experiment not found' },
        { status: 404 }
      );
    }

    const row = result.rows[0];

    // Get audit log for this proposal
    const auditResult = await pool.query(`
      SELECT action, reviewer, notes, created_at
      FROM approval_audit_log
      WHERE proposal_id = $1
      ORDER BY created_at DESC
    `, [proposalId]);

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
      auditLog: auditResult.rows.map((audit) => ({
        action: audit.action,
        reviewer: audit.reviewer,
        notes: audit.notes,
        createdAt: audit.created_at,
      })),
    };

    return NextResponse.json({ experiment });
  } catch (error) {
    console.error('Error fetching experiment:', error);
    return NextResponse.json(
      { error: 'Failed to fetch experiment' },
      { status: 500 }
    );
  }
}
