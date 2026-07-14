/**
 * GET /api/admin/kpis — business KPIs for the Overview (admin:all).
 *
 * Server-side aggregation over is_test = FALSE (spec C4: test traffic never pollutes
 * business metrics). The "Pendientes" figure reuses the SAME queue predicate as
 * /api/admin/queue (spec C1). MRR/conversion/churn stay null until Fase 6 (billing):
 * the provider — not this dashboard — is the source of truth for money.
 */
import { NextResponse } from 'next/server';

import { requireAdminRole } from '@/lib/admin/guard';
import { PENDING_QUEUE_WHERE } from '@/lib/admin/queue-sql';
import { computeRevenue } from '@/lib/admin/revenue';
import { STAFF_ROLES, type BusinessKpis } from '@/lib/contracts/admin-console.contract';
import { query } from '@/lib/db/postgres-client';

export async function GET(req: Request) {
  const denied = requireAdminRole(req);
  if (denied) return denied;
  try {
    const staffList = STAFF_ROLES.map((r) => `'${r}'`).join(',');
    const [counts, planMix, revenue] = await Promise.all([
      query(`
        SELECT
          COUNT(*) FILTER (WHERE status = 'approved' AND NOT COALESCE(is_test, FALSE))            AS total_users,
          COUNT(*) FILTER (WHERE status = 'approved' AND NOT COALESCE(is_test, FALSE)
                             AND created_at > NOW() - INTERVAL '7 days')                          AS new_7d,
          COUNT(*) FILTER (WHERE NOT COALESCE(is_test, FALSE)
                             AND last_login > NOW() - INTERVAL '7 days')                          AS active_7d,
          COUNT(*) FILTER (WHERE NOT COALESCE(is_test, FALSE)
                             AND last_login > NOW() - INTERVAL '30 days')                         AS active_30d,
          COUNT(*) FILTER (WHERE ${PENDING_QUEUE_WHERE} AND NOT COALESCE(is_test, FALSE))         AS pending_queue,
          COUNT(*) FILTER (WHERE ${PENDING_QUEUE_WHERE} AND COALESCE(is_test, FALSE))             AS pending_test_hidden
        FROM sb_users`),
      // Plan mix over CUSTOMERS only — staff has no plan (spec C3).
      query(`
        SELECT COALESCE(entitlements->>'plan', 'free') AS plan, COUNT(*) AS n
        FROM sb_users
        WHERE status = 'approved' AND NOT COALESCE(is_test, FALSE)
          AND role NOT IN (${staffList})
        GROUP BY 1`),
      // Real revenue KPIs (MRR/conversion/churn) from entitlements + audit_log —
      // same aggregator the Ingresos tab uses, so both surfaces always agree.
      computeRevenue(),
    ]);

    const c = counts.rows[0];
    const mix: Record<string, number> = {};
    for (const row of planMix.rows) mix[row.plan] = Number(row.n);

    const body: BusinessKpis = {
      total_users: Number(c.total_users),
      new_7d: Number(c.new_7d),
      active_7d: Number(c.active_7d),
      active_30d: Number(c.active_30d),
      pending_queue: Number(c.pending_queue),
      pending_test_hidden: Number(c.pending_test_hidden),
      plan_mix: mix,
      mrr_cop: revenue.mrr_cop,
      conversion_30d_pct: revenue.conversion_30d_pct,
      churn_monthly_pct: revenue.churn_monthly_pct,
    };
    return NextResponse.json(body);
  } catch (e) {
    return NextResponse.json({ error: 'db unavailable', detail: String(e) }, { status: 503 });
  }
}
