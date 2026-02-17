/**
 * POST /api/experiments/[id]/approve - Approve experiment (SEGUNDO VOTO)
 *
 * This endpoint implements the second vote in the 2-vote promotion system.
 * When approved:
 * 1. Current production model is archived
 * 2. New model is promoted to production
 * 3. Audit log is created
 * 4. L5 will automatically pick up the new model via ProductionContract
 */
import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';
import { protectApiRoute } from '@/lib/auth/api-auth';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

interface ApproveRequest {
  notes?: string;
  promoteToProduction?: boolean;
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
  const body: ApproveRequest = await request.json();
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
        { error: `Cannot approve: status is ${proposal.status}` },
        { status: 400 }
      );
    }

    // 2. Get current production model (to archive)
    const currentProdResult = await client.query(`
      SELECT model_id FROM model_registry
      WHERE stage = 'production' AND is_active = TRUE
    `);
    const previousProdModelId = currentProdResult.rows[0]?.model_id;

    // 3. Archive previous production model
    if (previousProdModelId) {
      await client.query(`
        UPDATE model_registry
        SET stage = 'archived', is_active = FALSE, archived_at = NOW()
        WHERE model_id = $1
      `, [previousProdModelId]);
    }

    // 4. Update proposal status
    await client.query(`
      UPDATE promotion_proposals
      SET status = 'APPROVED',
          reviewer = $1,
          reviewer_email = $2,
          reviewer_notes = $3,
          reviewed_at = NOW()
      WHERE proposal_id = $4
    `, [reviewer, authResult.user.email, body.notes, proposalId]);

    // 5. Update model in registry to production (model was already registered by L4)
    await client.query(`
      UPDATE model_registry
      SET stage = 'production',
          is_active = TRUE,
          approved_by = $1,
          approved_at = NOW(),
          promoted_at = NOW(),
          metrics = $2,
          lineage = $3
      WHERE model_id = $4
    `, [
      reviewer,
      JSON.stringify(proposal.metrics),
      JSON.stringify(proposal.lineage),
      proposal.model_id,
    ]);

    // 6. Insert audit log
    await client.query(`
      INSERT INTO approval_audit_log
      (action, model_id, proposal_id, reviewer, reviewer_email, notes, previous_production_model)
      VALUES ('APPROVE', $1, $2, $3, $4, $5, $6)
    `, [
      proposal.model_id,
      proposalId,
      reviewer,
      authResult.user.email,
      body.notes,
      previousProdModelId,
    ]);

    await client.query('COMMIT');

    return NextResponse.json({
      success: true,
      modelId: proposal.model_id,
      newStage: 'production',
      previousModelArchived: previousProdModelId,
      approvedBy: reviewer,
      message: 'Model promoted to production. L5 will automatically load the new model.',
    });

  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Error approving experiment:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Approval failed' },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
