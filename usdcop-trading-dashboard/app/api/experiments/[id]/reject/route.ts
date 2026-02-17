/**
 * POST /api/experiments/[id]/reject - Reject experiment
 *
 * Marks the proposal as rejected and logs the decision.
 */
import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';
import { protectApiRoute } from '@/lib/auth/api-auth';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

interface RejectRequest {
  notes?: string;
  reason?: string;
}

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  // Validate authentication (SKIP_AUTH=true uses mock user)
  const skipAuth = process.env.SKIP_AUTH === 'true';
  let authResult;
  if (skipAuth) {
    authResult = {
      authenticated: true,
      user: { email: 'admin@dev.local', name: 'Admin (Dev)', role: 'admin', id: 'dev-admin' },
    };
  } else {
    authResult = await protectApiRoute(request, { rateLimit: false });
  }

  if (!authResult.authenticated || !authResult.user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const proposalId = params.id;
  const body: RejectRequest = await request.json();
  const reviewer = authResult.user.email || authResult.user.name || 'unknown';

  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    // 1. Get proposal
    const proposalResult = await client.query(
      'SELECT * FROM promotion_proposals WHERE proposal_id = $1',
      [proposalId]
    );

    if (proposalResult.rows.length === 0) {
      await client.query('ROLLBACK');
      return NextResponse.json(
        { error: 'Proposal not found' },
        { status: 404 }
      );
    }

    const proposal = proposalResult.rows[0];

    if (proposal.status !== 'PENDING_APPROVAL') {
      await client.query('ROLLBACK');
      return NextResponse.json(
        { error: `Cannot reject: status is ${proposal.status}` },
        { status: 400 }
      );
    }

    // 2. Update proposal status
    const notes = body.notes || body.reason || 'Rejected by reviewer';
    await client.query(`
      UPDATE promotion_proposals
      SET status = 'REJECTED',
          reviewer = $1,
          reviewer_email = $2,
          reviewer_notes = $3,
          reviewed_at = NOW()
      WHERE proposal_id = $4
    `, [reviewer, authResult.user.email, notes, proposalId]);

    // 3. Insert audit log
    await client.query(`
      INSERT INTO approval_audit_log
      (action, model_id, proposal_id, reviewer, reviewer_email, notes)
      VALUES ('REJECT', $1, $2, $3, $4, $5)
    `, [
      proposal.model_id,
      proposalId,
      reviewer,
      authResult.user.email,
      notes,
    ]);

    await client.query('COMMIT');

    return NextResponse.json({
      success: true,
      modelId: proposal.model_id,
      status: 'REJECTED',
      rejectedBy: reviewer,
      message: 'Experiment rejected. Model will not be promoted.',
    });

  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Error rejecting experiment:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Rejection failed' },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
