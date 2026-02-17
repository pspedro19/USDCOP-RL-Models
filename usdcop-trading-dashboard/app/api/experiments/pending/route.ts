/**
 * GET /api/experiments/pending - Get pending experiments requiring approval
 *
 * Returns experiments with status='PENDING_APPROVAL' that haven't expired
 */
import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';
import { protectApiRoute } from '@/lib/auth/api-auth';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

export async function GET(request: NextRequest) {
  // Validate authentication (SKIP_AUTH=true bypasses auth for development)
  const skipAuth = process.env.SKIP_AUTH === 'true';
  if (!skipAuth) {
    const authResult = await protectApiRoute(request, { rateLimit: false });
    if (!authResult.authenticated) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
  }

  try {
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
        pp.created_at,
        pp.expires_at,
        EXTRACT(EPOCH FROM (pp.expires_at - NOW())) / 3600 as hours_until_expiry
      FROM promotion_proposals pp
      WHERE pp.status = 'PENDING_APPROVAL'
        AND (pp.expires_at IS NULL OR pp.expires_at > NOW())
      ORDER BY pp.created_at DESC
    `);

    const pending = result.rows.map((row) => ({
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
      createdAt: row.created_at,
      expiresAt: row.expires_at,
      hoursUntilExpiry: row.hours_until_expiry ? parseFloat(row.hours_until_expiry) : null,
    }));

    return NextResponse.json({
      pending,
      count: result.rowCount,
    });
  } catch (error) {
    console.error('Error fetching pending experiments:', error);
    return NextResponse.json(
      { error: 'Failed to fetch pending experiments' },
      { status: 500 }
    );
  }
}
