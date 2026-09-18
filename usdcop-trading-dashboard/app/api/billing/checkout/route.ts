/**
 * POST /api/billing/checkout — create a hosted-checkout session for the logged-in user.
 * Body: { plan, addOnAssets?: string[], interval?: 'month'|'year', currency?: 'COP'|'USD' }
 * Auth: middleware guarantees a session ('authenticated') and stamps x-user-id.
 *
 * Used by the Pricing page. Like /api/cart/checkout it MUST seal the server-side
 * quote in `checkout_orders` before returning a URL: the webhook credits the sealed
 * quote and nothing else, so an unsealed checkout would take the user's money and
 * grant nothing. Add-ons requested by the client are priced server-side from the
 * SSOT (unknown/unpriced add-ons make the provider throw) — the client never sets
 * the amount (CTR-RBAC-001 rule 7).
 *
 * CHANNELS. The quote now carries a currency and a billing interval, because the annual
 * multi-asset bundle is priced in USD while the local plans settle in COP. Only
 * combinations the price table actually publishes may be bought: an unpriced combination
 * is refused rather than defaulted, so an unsold plan can never be charged at 0.
 *
 * The USD channel has NO registered provider yet — Wompi settles COP. A USD checkout
 * therefore fails closed with an explicit reason instead of quietly creating a COP charge
 * for a dollar price, which would take the wrong amount from the customer.
 */
import { NextResponse } from 'next/server';

import { query } from '@/lib/db/postgres-client';
import { getBillingProvider } from '@/lib/billing';
import { addonPricesCop, isPlanSold, planPrice } from '@/lib/billing/prices';
import type { BillingInterval, Currency } from '@/lib/contracts/rbac.contract';
import { logServerError } from '@/lib/api/envelope';

export async function POST(req: Request) {
  const userId = req.headers.get('x-user-id');
  if (!userId) return NextResponse.json({ error: 'unauthenticated' }, { status: 401 });

  let body: { plan?: string; addOnAssets?: string[]; interval?: string; currency?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'invalid JSON' }, { status: 400 });
  }
  // Narrowed to the union so the provider's typed contract keeps checking this call site:
  // a plan added to the list without being added to the type must not compile.
  const SELLABLE_PLANS = ['signals', 'auto', 'desk'] as const;
  type SellablePlan = (typeof SELLABLE_PLANS)[number];
  const isSellablePlan = (v: unknown): v is SellablePlan =>
    typeof v === 'string' && (SELLABLE_PLANS as readonly string[]).includes(v);
  if (!isSellablePlan(body.plan)) {
    return NextResponse.json(
      { error: `plan must be one of: ${SELLABLE_PLANS.join(', ')}` }, { status: 400 });
  }
  const plan: SellablePlan = body.plan;
  const interval: BillingInterval = body.interval === 'year' ? 'year' : 'month';
  if (body.currency !== undefined && body.currency !== 'COP' && body.currency !== 'USD') {
    return NextResponse.json({ error: "currency must be 'COP' or 'USD'" }, { status: 400 });
  }
  const currency: Currency = (body.currency as Currency) ?? 'COP';

  // Refuse an unpublished combination instead of defaulting it to zero.
  if (!isPlanSold(plan, currency, interval)) {
    return NextResponse.json(
      { error: 'plan not sold in that currency/interval', plan: plan, currency, interval },
      { status: 400 },
    );
  }
  // Fail closed rather than charging COP for a USD price.
  if (currency !== 'COP') {
    return NextResponse.json(
      { error: 'no payment provider registered for that currency', currency }, { status: 503 });
  }

  const res = await query<{ email: string }>('SELECT email FROM sb_users WHERE id = $1', [userId]);
  const email = res.rows[0]?.email;
  if (!email) return NextResponse.json({ error: 'user not found' }, { status: 404 });

  // Add-ons are a COP-channel concept: the USD bundle already covers every asset, so
  // letting add-ons ride along would double-charge for what the plan includes.
  const addOnAssets = currency === 'COP' ? (body.addOnAssets ?? []).map(String) : [];

  let session;
  try {
    session = await getBillingProvider().createCheckout({
      userId, email, plan, addOnAssets,
    });
  } catch (e) {
    // Provider not configured / no published price ⇒ fail closed, no URL. The reason
    // stays in the server log; the body never carries paths or driver text.
    logServerError('billing/checkout provider', e);
    return NextResponse.json({ error: 'billing provider not configured' }, { status: 503 });
  }

  try {
    // `isPlanSold` above guarantees this is a number; the SSOT stays the only place a
    // price is decided.
    const base = planPrice(plan, currency, interval) ?? 0;
    const amountCents = Math.round(base +
      addOnAssets.reduce((sum, id) => sum + Math.round((addonPricesCop()[id] ?? 0) * 100), 0));
    await query(
      `INSERT INTO checkout_orders
         (user_id, plan, addon_assets, amount_cents, currency, billing_interval, reference, status)
       VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7,'pending')
       ON CONFLICT (reference) DO NOTHING`,
      [userId, plan, JSON.stringify(addOnAssets), amountCents, currency, interval,
       session.reference],
    );
  } catch (e) {
    logServerError('billing/checkout seal-quote', e);
    return NextResponse.json({ error: 'could not start payment' }, { status: 502 });
  }

  return NextResponse.json(session);
}
