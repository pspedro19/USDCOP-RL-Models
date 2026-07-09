/**
 * POST /api/billing/webhook — payment provider events (CTR-RBAC-001 rule 9).
 *
 * Public route (providers can't log in) but SIGNATURE-VERIFIED inside: an event that
 * fails verification is dropped with 401. On payment.approved the user's
 * `sb_users.entitlements` is upgraded to the paid plan (+30 days) and the change is
 * written to the append-only audit_log. The provider is the source of truth — the
 * client NEVER updates entitlements directly.
 */
import { NextResponse } from 'next/server';

import { query } from '@/lib/db/postgres-client';
import { getBillingProvider, decodeReference } from '@/lib/billing';
import { PLAN_DEFAULTS } from '@/lib/contracts/rbac.contract';

export async function POST(req: Request) {
  const rawBody = await req.text();
  const provider = getBillingProvider();

  const verification = await provider.verifyWebhook(rawBody, req.headers);
  if (!verification.valid || !verification.event) {
    return NextResponse.json({ error: verification.error ?? 'invalid signature' }, { status: 401 });
  }

  const { type, reference } = verification.event;
  const decoded = decodeReference(reference);
  if (!decoded) {
    return NextResponse.json({ error: 'unknown reference format' }, { status: 400 });
  }

  if (type === 'payment.approved') {
    const base = PLAN_DEFAULTS[decoded.plan];
    const entitlements = {
      ...base,
      assets: [...new Set([...base.assets, ...decoded.addOns])],
      expires_at: new Date(Date.now() + 30 * 86_400_000).toISOString(),
    };
    await query(
      'UPDATE sb_users SET entitlements = $1::jsonb WHERE id = $2',
      [JSON.stringify(entitlements), decoded.userId],
    );
    await audit(decoded.userId, 'plan_change', {
      provider: provider.name, plan: decoded.plan, addOns: decoded.addOns, reference,
    });
  } else {
    // declined / cancelled: record only — expiry-based degradation handles access.
    await audit(decoded.userId, 'plan_payment_failed', { provider: provider.name, type, reference });
  }

  return NextResponse.json({ received: true });
}

async function audit(userId: string, action: string, detail: Record<string, unknown>) {
  try {
    await query(
      'INSERT INTO audit_log (user_id, action, object_type, detail) VALUES ($1,$2,$3,$4::jsonb)',
      [userId, action, 'entitlements', JSON.stringify(detail)],
    );
  } catch (e) {
    console.error('[billing/webhook] audit insert failed:', e);
  }
}
