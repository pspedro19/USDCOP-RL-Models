/**
 * Real admin revenue aggregation (CTR-ADMIN-CONSOLE-001 · Ingresos).
 *
 * Computes MRR/ARR/ARPU/LTV, per-plan & per-asset breakdown, 30-day MRR movement,
 * payment health and dunning from REAL data only:
 *   - sb_users.entitlements  → active paid subscriptions (plan + expiry)
 *   - audit_log              → plan_change / plan_payment_failed events (webhook-written)
 *   - lib/billing/prices     → plan prices SSOT (shared with Wompi checkout)
 *
 * No mocks. With no paying customers yet the figures are genuine zeros (money not
 * invented); metrics that aren't measured (refunds, chargebacks, up/down-grade split)
 * stay null → the UI renders "—", never a fake 0 (checklist 10.1).
 *
 * Excludes staff (admin/developer) and is_test accounts (spec C3/C4).
 */
import { planPricesCents, planPriceCop, PLAN_LABELS, PAID_PLANS } from '@/lib/billing/prices';
import { STAFF_ROLES } from '@/lib/contracts/admin-console.contract';
import { PLAN_DEFAULTS, type PlanId } from '@/lib/contracts/rbac.contract';
import type { AdminRevenueResponse } from '@/lib/contracts/admin-console.contract';
import { query } from '@/lib/db/postgres-client';

const STAFF_SQL = STAFF_ROLES.map((r) => `'${r}'`).join(',');
const CUSTOMER_WHERE = `status='approved' AND NOT COALESCE(is_test,FALSE) AND role NOT IN (${STAFF_SQL})`;
const ASSET_SYMBOLS: Record<string, string> = { usdcop: 'USD/COP', xauusd: 'XAU/USD', btcusdt: 'BTC/USDT' };

export interface RevenueAggregate {
  revenue: AdminRevenueResponse;
  /** KPI subset reused by /api/admin/kpis so both surfaces agree. */
  mrr_cop: number;
  conversion_30d_pct: number | null;
  churn_monthly_pct: number | null;
}

function round(n: number, d = 0): number {
  const f = 10 ** d;
  return Math.round(n * f) / f;
}

export async function computeRevenue(): Promise<RevenueAggregate> {
  const [activeMix, totals, changeByPlan, failed, dunningRows] = await Promise.all([
    // Active paid subscriptions by plan (not expired).
    query(`SELECT COALESCE(entitlements->>'plan','free') AS plan, COUNT(*) AS n
           FROM sb_users
           WHERE ${CUSTOMER_WHERE}
             AND (entitlements->>'expires_at' IS NULL
                  OR (entitlements->>'expires_at')::timestamptz > NOW())
           GROUP BY 1`),
    query(`SELECT COUNT(*) AS customers FROM sb_users WHERE ${CUSTOMER_WHERE}`),
    // New/renewed subscriptions in the last 30d, by plan (from the webhook audit trail).
    query(`SELECT COALESCE(detail->>'plan','') AS plan, COUNT(*) AS n
           FROM audit_log
           WHERE action='plan_change' AND created_at > NOW() - INTERVAL '30 days'
           GROUP BY 1`),
    // Failed payments in the last 30d (churn signal).
    query(`SELECT COUNT(*) AS n, COUNT(DISTINCT user_id) AS users
           FROM audit_log
           WHERE action='plan_payment_failed' AND created_at > NOW() - INTERVAL '30 days'`),
    query(`SELECT u.email, COALESCE(u.entitlements->>'plan','free') AS plan, a.created_at
           FROM audit_log a JOIN sb_users u ON u.id = a.user_id
           WHERE a.action='plan_payment_failed' AND a.created_at > NOW() - INTERVAL '30 days'
           ORDER BY a.created_at DESC LIMIT 20`),
  ]);

  const pricesCents = planPricesCents();

  // Active paid subscribers per plan.
  const paidCounts: Record<string, number> = {};
  let paidSubscribers = 0;
  for (const row of activeMix.rows) {
    const plan = row.plan as string;
    if (!PAID_PLANS.includes(plan as PlanId)) continue;
    paidCounts[plan] = Number(row.n);
    paidSubscribers += Number(row.n);
  }

  // MRR (COP) from active paid subscriptions.
  let mrrCents = 0;
  for (const [plan, n] of Object.entries(paidCounts)) mrrCents += n * (pricesCents[plan] ?? 0);
  const mrr = round(mrrCents / 100);
  const arr = mrr * 12;
  const arpu = paidSubscribers > 0 ? round(mrr / paidSubscribers) : null;

  // Per-plan revenue breakdown (only paid plans; add-ons row kept for shape parity).
  const porPlan = PAID_PLANS.map((plan) => {
    const amount = round(((paidCounts[plan] ?? 0) * (pricesCents[plan] ?? 0)) / 100);
    return { plan: PLAN_LABELS[plan] ?? plan, amount, pct: mrr > 0 ? round((amount / mrr) * 100, 1) : 0 };
  });
  porPlan.push({ plan: 'Add-ons por activo', amount: 0, pct: 0 }); // add-ons not sold yet (real 0)

  // Per-asset attribution: each plan's MRR split across the assets it entitles.
  const perAsset: Record<string, number> = { usdcop: 0, xauusd: 0, btcusdt: 0 };
  for (const [plan, n] of Object.entries(paidCounts)) {
    const assets = PLAN_DEFAULTS[plan as PlanId]?.assets ?? [];
    if (!assets.length) continue;
    const per = (n * (pricesCents[plan] ?? 0)) / 100 / assets.length;
    for (const a of assets) if (a in perAsset) perAsset[a] += per;
  }
  const porActivo = Object.entries(ASSET_SYMBOLS).map(([id, symbol]) => ({
    symbol,
    amount: round(perAsset[id] ?? 0),
  }));

  // 30-day MRR movement from the audit trail.
  let nuevo = 0;
  for (const row of changeByPlan.rows) nuevo += Number(row.n) * planPriceCop(row.plan);
  const failedCount = Number(failed.rows[0]?.n ?? 0);
  const churnedUsers = Number(failed.rows[0]?.users ?? 0);
  // Churn revenue: attribute failed payments at the average paid price we can see.
  const avgPaidPrice = paidSubscribers > 0 ? mrr / paidSubscribers : 0;
  const churnCop = round(churnedUsers * avgPaidPrice);
  const movimiento = {
    nuevo: round(nuevo),
    expansion: null, // upgrade/downgrade split not measured yet
    contraccion: null,
    churn: churnCop,
    neto: round(nuevo - churnCop),
  };

  const cobros = {
    exitosos: changeByPlan.rows.reduce((s, r) => s + Number(r.n), 0),
    fallidos: failedCount,
    reembolsos: null, // not tracked as events yet → "—", never a fake 0
    contracargos: null,
  };

  const dunning = dunningRows.rows.map((r) => ({
    user: r.email as string,
    plan: PLAN_LABELS[r.plan as string] ?? (r.plan as string),
    amount: planPriceCop(r.plan as string),
    attempts: null, // retry count not tracked yet
    reason: 'Pago rechazado',
  }));

  // LTV = ARPU / churn-rate (only meaningful once there is churn to divide by).
  const denom = paidSubscribers + churnedUsers;
  const churnRate = denom > 0 && churnedUsers > 0 ? round((churnedUsers / denom) * 100, 1) : null;
  const ltv = arpu != null && churnRate != null && churnRate > 0 ? round(arpu / (churnRate / 100)) : null;

  const customers = Number(totals.rows[0]?.customers ?? 0);
  const conversion = customers > 0 ? round((paidSubscribers / customers) * 100, 1) : null;

  const revenue: AdminRevenueResponse = {
    kpis: { mrr, arr, arpu, ltv },
    por_plan: porPlan,
    movimiento,
    cobros,
    por_activo: porActivo,
    dunning,
    phase_note:
      paidSubscribers > 0
        ? 'Ingresos en vivo sobre suscripciones reales (sb_users.entitlements + audit_log).'
        : 'En vivo · aún sin suscriptores de pago — se poblará al activar billing (Wompi) y cerrar cobros.',
  };

  return { revenue, mrr_cop: mrr, conversion_30d_pct: conversion, churn_monthly_pct: churnRate };
}
