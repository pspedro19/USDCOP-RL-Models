/**
 * GET /api/admin/overview — users + recent audit log + plan mix (admin:all via matrix).
 * Read-only aggregation for the /admin console; entitlement edits stay manual/webhook.
 */
import { NextResponse } from 'next/server';

import { query } from '@/lib/db/postgres-client';

export async function GET(req: Request) {
  // Defense in depth: middleware already enforces admin:all; re-check the stamped role.
  if (req.headers.get('x-user-role') !== 'admin') {
    return NextResponse.json({ error: 'admin only' }, { status: 403 });
  }
  try {
    const [users, audit] = await Promise.all([
      query(`SELECT id, email, role, status, entitlements->>'plan' AS plan,
                    entitlements->>'expires_at' AS expires_at, created_at
             FROM sb_users ORDER BY created_at DESC LIMIT 200`),
      query(`SELECT user_id, action, object_type, detail, ip, created_at
             FROM audit_log ORDER BY created_at DESC LIMIT 100`),
    ]);
    const planMix: Record<string, number> = {};
    for (const u of users.rows) planMix[u.plan ?? 'free'] = (planMix[u.plan ?? 'free'] ?? 0) + 1;
    return NextResponse.json({ users: users.rows, audit: audit.rows, plan_mix: planMix });
  } catch (e) {
    return NextResponse.json({ error: 'db unavailable', detail: String(e) }, { status: 503 });
  }
}
