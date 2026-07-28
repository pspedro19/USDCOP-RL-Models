/**
 * POST /api/billing/checkout — create a hosted-checkout session for the logged-in user.
 * Body: { plan: 'signals' | 'auto', addOnAssets?: string[] }
 * Auth: middleware guarantees a session ('authenticated') and stamps x-user-id.
 *
 * Used by the Pricing page. Like /api/cart/checkout it MUST seal the server-side
 * quote in `checkout_orders` before returning a URL: the webhook credits the sealed
 * quote and nothing else, so an unsealed checkout would take the user's money and
 * grant nothing. Add-ons requested by the client are priced server-side from the
 * SSOT (unknown/unpriced add-ons make the provider throw) — the client never sets
 * the amount (CTR-RBAC-001 rule 7).
 */
import { NextResponse } from 'next/server';

import { query } from '@/lib/db/postgres-client';
import { getBillingProvider } from '@/lib/billing';
import { planPriceCents, addonPricesCop } from '@/lib/billing/prices';
import { logServerError } from '@/lib/api/envelope';

export async function POST(req: Request) {
  const userId = req.headers.get('x-user-id');
  if (!userId) return NextResponse.json({ error: 'unauthenticated' }, { status: 401 });

  let body: { plan?: string; addOnAssets?: string[] };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'invalid JSON' }, { status: 400 });
  }
  if (body.plan !== 'signals' && body.plan !== 'auto') {
    return NextResponse.json({ error: "plan must be 'signals' or 'auto'" }, { status: 400 });
  }

  const res = await query<{ email: string }>('SELECT email FROM sb_users WHERE id = $1', [userId]);
  const email = res.rows[0]?.email;
  if (!email) return NextResponse.json({ error: 'user not found' }, { status: 404 });

  const addOnAssets = (body.addOnAssets ?? []).map(String);

  let session;
  try {
    session = await getBillingProvider().createCheckout({
      userId, email, plan: body.plan, addOnAssets,
    });
  } catch (e) {
    // Provider not configured / no published price ⇒ fail closed, no URL. The reason
    // stays in the server log; the body never carries paths or driver text.
    logServerError('billing/checkout provider', e);
    return NextResponse.json({ error: 'billing provider not configured' }, { status: 503 });
  }

  try {
    const amountCents = Math.round(planPriceCents(body.plan) +
      addOnAssets.reduce((sum, id) => sum + Math.round((addonPricesCop()[id] ?? 0) * 100), 0));
    await query(
      `INSERT INTO checkout_orders (user_id, plan, addon_assets, amount_cents, currency, reference, status)
       VALUES ($1,$2,$3::jsonb,$4,'COP',$5,'pending')
       ON CONFLICT (reference) DO NOTHING`,
      [userId, body.plan, JSON.stringify(addOnAssets), amountCents, session.reference],
    );
  } catch (e) {
    logServerError('billing/checkout seal-quote', e);
    return NextResponse.json({ error: 'could not start payment' }, { status: 502 });
  }

  return NextResponse.json(session);
}
