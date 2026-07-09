/**
 * GET /api/admin/users/list — full users table for the console (admin:all).
 *
 * Enriches sb_users with execution mode/kill-switch (user_risk_limits_v2 via LEFT
 * JOIN). If the RBAC tables are missing (older DB), degrades to the plain users
 * query — the widget renders with execution columns as '—' instead of failing (C5).
 * Roles are normalized to the RBAC contract enum at this read boundary (C2);
 * staff plan is nulled server-side (C3).
 */
import { NextResponse } from 'next/server';

import { requireAdminRole } from '@/lib/admin/guard';
import {
  normalizeRole, planForDisplay,
  type AdminUserRow, type UsersListResponse,
} from '@/lib/contracts/admin-console.contract';
import { query } from '@/lib/db/postgres-client';

const BASE_SELECT = `
  SELECT u.id, u.email, u.name, u.role, u.status,
         u.entitlements->>'plan'       AS plan,
         u.entitlements->>'expires_at' AS expires_at,
         u.created_at, u.last_login,
         COALESCE(u.is_test, FALSE)    AS is_test`;

const WITH_EXECUTION = `${BASE_SELECT},
         rl.mode        AS execution_mode,
         rl.kill_switch AS kill_switch
  FROM sb_users u
  LEFT JOIN user_risk_limits_v2 rl ON rl.user_id = u.id
  ORDER BY u.created_at DESC
  LIMIT 500`;

const WITHOUT_EXECUTION = `${BASE_SELECT},
         NULL AS execution_mode, NULL AS kill_switch
  FROM sb_users u
  ORDER BY u.created_at DESC
  LIMIT 500`;

export async function GET(req: Request) {
  const denied = requireAdminRole(req);
  if (denied) return denied;
  try {
    let res;
    try {
      res = await query(WITH_EXECUTION);
    } catch {
      res = await query(WITHOUT_EXECUTION); // degraded: no user_risk_limits_v2 yet
    }
    const users: AdminUserRow[] = res.rows.map((r) => {
      const role = normalizeRole(r.role);
      return {
        id: r.id,
        email: r.email,
        name: r.name ?? null,
        role,
        raw_role: r.role,
        status: r.status,
        plan: planForDisplay(role, r.plan),
        expires_at: r.expires_at ?? null,
        created_at: r.created_at,
        last_login: r.last_login ?? null,
        is_test: !!r.is_test,
        execution_mode: r.execution_mode === 'live' || r.execution_mode === 'paper' ? r.execution_mode : null,
        kill_switch: r.kill_switch ?? null,
      };
    });
    const body: UsersListResponse = { users, total: users.length };
    return NextResponse.json(body);
  } catch (e) {
    return NextResponse.json({ error: 'db unavailable', detail: String(e) }, { status: 503 });
  }
}
