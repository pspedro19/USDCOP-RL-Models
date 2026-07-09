/**
 * GET /api/admin/queue — approval queue (admin:all via matrix + in-handler re-check).
 *
 * Reads sb_users DIRECTLY (same Postgres SignalBridge writes to) so the queue never
 * 401s on an expired SignalBridge bearer — only the approve/reject ACTIONS need the
 * relayed admin token (SignalBridge stays the authority for state transitions).
 * Counter and table share this one query (spec C1).
 */
import { NextResponse } from 'next/server';

import { requireAdminRole } from '@/lib/admin/guard';
import { PENDING_QUEUE_SELECT } from '@/lib/admin/queue-sql';
import { normalizeRole, type QueueItem, type QueueResponse } from '@/lib/contracts/admin-console.contract';
import { query } from '@/lib/db/postgres-client';

export async function GET(req: Request) {
  const denied = requireAdminRole(req);
  if (denied) return denied;
  try {
    const res = await query(PENDING_QUEUE_SELECT);
    const items: QueueItem[] = res.rows.map((r) => ({
      id: r.id,
      email: r.email,
      name: r.name ?? null,
      status: r.status,
      role: normalizeRole(r.role),
      created_at: r.created_at,
      waiting_hours: Math.round(Number(r.waiting_hours) * 10) / 10,
      is_test: !!r.is_test,
    }));
    const testHidden = items.filter((i) => i.is_test).length;
    const body: QueueResponse = {
      items,
      count: items.length - testHidden,
      test_hidden: testHidden,
    };
    return NextResponse.json(body);
  } catch (e) {
    return NextResponse.json({ error: 'db unavailable', detail: String(e) }, { status: 503 });
  }
}
