/**
 * POST /api/admin/users/:id/flag-test — toggle the is_test flag (admin:all).
 *
 * Spec C4: manual override over the email heuristic. Privileged action ⇒ audit row
 * (action 'user_flag_test', append-only INSERT recording the REAL acting admin).
 */
import { NextResponse } from 'next/server';

import { adminActor, requireAdminRole } from '@/lib/admin/guard';
import { query } from '@/lib/db/postgres-client';

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const denied = requireAdminRole(req);
  if (denied) return denied;
  const { id } = await ctx.params;

  let isTest: boolean;
  try {
    const body = await req.json();
    if (typeof body?.is_test !== 'boolean') throw new Error('is_test boolean required');
    isTest = body.is_test;
  } catch (e) {
    return NextResponse.json({ error: `invalid body: ${String((e as Error)?.message ?? e)}` }, { status: 400 });
  }

  try {
    const updated = await query(
      `UPDATE sb_users SET is_test = $1 WHERE id = $2 RETURNING id, email, is_test`,
      [isTest, id],
    );
    if (updated.rowCount === 0) {
      return NextResponse.json({ error: 'user not found' }, { status: 404 });
    }
    const actor = adminActor(req);
    // Audit is best-effort but loud: the flag change is legit even if the insert fails.
    try {
      await query(
        `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
         VALUES ($1, 'user_flag_test', 'admin_console', $2, $3::jsonb, $4)`,
        [actor.id, id, JSON.stringify({ target_email: updated.rows[0].email, is_test: isTest }), actor.ip],
      );
    } catch (auditErr) {
      console.error('[flag-test] audit insert failed:', auditErr);
    }
    return NextResponse.json(updated.rows[0]);
  } catch (e) {
    return NextResponse.json({ error: 'db unavailable', detail: String(e) }, { status: 503 });
  }
}
