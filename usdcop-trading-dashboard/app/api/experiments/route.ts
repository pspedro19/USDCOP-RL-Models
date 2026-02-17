/**
 * GET /api/experiments - List all experiments with their status
 *
 * Query params:
 * - status: Filter by status ('PENDING_APPROVAL', 'APPROVED', 'REJECTED')
 * - limit: Max number of results (default 50)
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

  const searchParams = request.nextUrl.searchParams;
  const status = searchParams.get('status');
  const limit = parseInt(searchParams.get('limit') || '50', 10);

  try {
    let query = `
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
        pp.reviewer_notes,
        pp.reviewed_at,
        pp.created_at,
        pp.expires_at
      FROM promotion_proposals pp
    `;

    const params: (string | number)[] = [];
    let paramIndex = 1;

    if (status) {
      query += ` WHERE pp.status = $${paramIndex}`;
      params.push(status);
      paramIndex++;
    }

    query += ` ORDER BY pp.created_at DESC LIMIT $${paramIndex}`;
    params.push(limit);

    const result = await pool.query(query, params);

    // Transform rows for API response
    const experiments = result.rows.map((row) => ({
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
      reviewerNotes: row.reviewer_notes,
      reviewedAt: row.reviewed_at,
      createdAt: row.created_at,
      expiresAt: row.expires_at,
    }));

    // Get counts by status
    const countResult = await pool.query(`
      SELECT status, COUNT(*) as count
      FROM promotion_proposals
      GROUP BY status
    `);

    const statusCounts: Record<string, number> = {};
    countResult.rows.forEach((row) => {
      statusCounts[row.status] = parseInt(row.count, 10);
    });

    return NextResponse.json({
      experiments,
      total: result.rowCount,
      statusCounts,
    });
  } catch (error) {
    console.error('Error fetching experiments:', error);
    return NextResponse.json(
      { error: 'Failed to fetch experiments' },
      { status: 500 }
    );
  }
}
