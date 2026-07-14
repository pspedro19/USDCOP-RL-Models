/**
 * GET /api/billing/prices — PUBLIC plan + add-on prices (COP).
 *
 * The buyer-facing counterpart to the server-only lib/billing/prices SSOT: it
 * surfaces the SAME numbers used for the Wompi charge and admin revenue so the
 * Pricing page and Cart can show real amounts instead of "—". Readable pre-login
 * (public in the RBAC matrix) — a price list is marketing, not monetized IP.
 * Zero price literals here: everything is read from lib/billing/prices.
 */
import { ok } from '@/lib/api/envelope';
import { PLAN_LABELS, addonPricesCop, planPriceCents, planPriceCop } from '@/lib/billing/prices';
import type { BillingPricesResponse } from '@/lib/contracts/catalog.contract';
import type { PlanId } from '@/lib/contracts/rbac.contract';

export const dynamic = 'force-dynamic';

const PLAN_ORDER: readonly PlanId[] = ['free', 'signals', 'auto'];

export async function GET() {
  const body: BillingPricesResponse = {
    currency: 'COP',
    plans: PLAN_ORDER.map((plan) => ({
      plan,
      label: PLAN_LABELS[plan] ?? plan,
      price_month_cop: planPriceCop(plan),
      price_month_cents: planPriceCents(plan),
    })),
    addons: addonPricesCop(),
  };
  return ok(body, { meta: { asOf: new Date().toISOString() } });
}
