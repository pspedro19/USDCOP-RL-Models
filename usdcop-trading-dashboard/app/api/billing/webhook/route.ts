/**
 * POST /api/billing/webhook — payment provider events (CTR-RBAC-001 rule 9).
 *
 * Public route (providers can't log in) but SIGNATURE-VERIFIED inside: an event that
 * fails verification is dropped with 401. The provider is the source of truth — the
 * client NEVER updates entitlements directly.
 *
 * MONEY INVARIANTS (CODEX P0-3):
 *  1. The economic terms come from the SEALED QUOTE in `checkout_orders`
 *     (`amount_cents`, `currency`, `plan`, `addon_assets`), never from the current
 *     price list: if a price changes between checkout and confirmation, we credit
 *     what the user accepted. `decodeReference()` is only a cross-check of the
 *     correlation id — NEVER the economic source of truth.
 *  2. Idempotency + order transition + entitlement + audit happen in ONE
 *     transaction. A mid-flight failure rolls the whole thing back, so a retry
 *     (the provider always retries) is still able to credit the user.
 *  3. Order transitions are applied only from states migration 059 declares legal;
 *     a cancellation NEVER mutates an already-paid order.
 * Fail-closed everywhere: when in doubt we do not credit.
 */
import { NextResponse } from 'next/server';
import type { PoolClient } from 'pg';

import { getClient } from '@/lib/db/postgres-client';
import { getBillingProvider, decodeReference } from '@/lib/billing';
import type { BillingEventType, NormalizedBillingEvent } from '@/lib/billing';
import { PLAN_DEFAULTS } from '@/lib/contracts/rbac.contract';
import { addonPricesCop } from '@/lib/billing/prices';
import { logServerError } from '@/lib/api/envelope';

/** Terminal order status per normalized event + the states it may come FROM (mig. 059). */
const ORDER_TRANSITION: Record<Exclude<BillingEventType, 'payment.approved'>,
  { to: string; from: string[] }> = {
  'payment.declined': { to: 'failed', from: ['created', 'pending'] },
  'subscription.cancelled': { to: 'cancelled', from: ['created', 'pending'] },
  'payment.refunded': { to: 'refunded', from: ['paid'] },
  'payment.charged_back': { to: 'charged_back', from: ['paid'] },
};

/** Rejection with a client-safe code; rolls the transaction back. */
class WebhookReject extends Error {
  constructor(public code: string, public status = 400) { super(code); }
}

interface OrderRow {
  user_id: string; plan: string; addon_assets: unknown;
  amount_cents: string | number; currency: string; status: string;
}

export async function POST(req: Request) {
  const rawBody = await req.text();

  let provider;
  try {
    provider = getBillingProvider();
  } catch (e) {
    logServerError('billing/webhook provider', e);
    return NextResponse.json({ error: 'billing provider unavailable' }, { status: 503 });
  }

  const verification = await provider.verifyWebhook(rawBody, req.headers);
  if (!verification.valid) {
    return NextResponse.json({ error: verification.error ?? 'invalid signature' }, { status: 401 });
  }
  if (!verification.event) {
    // Signature valid, no transition expressed (PENDING / unknown status / unknown
    // event). ACK so the provider stops retrying — and mutate NOTHING (P0-1).
    console.warn('[billing/webhook] ignored event:', verification.ignored?.reason ?? 'no event');
    return NextResponse.json({ received: true, ignored: true });
  }

  const event = verification.event;
  if (!event.reference) {
    return NextResponse.json({ error: 'missing reference' }, { status: 400 });
  }

  let client: PoolClient;
  try {
    client = await getClient();
  } catch (e) {
    logServerError('billing/webhook db', e);
    return NextResponse.json({ error: 'temporarily unavailable' }, { status: 503 });
  }

  try {
    await client.query('BEGIN');
    // Idempotency INSIDE the transaction: if anything below fails, this row is rolled
    // back too, so the provider's retry can still credit the user (otherwise a
    // transient DB error would permanently deny a paid entitlement).
    await client.query(
      'INSERT INTO billing_webhook_events (reference,event_type) VALUES ($1,$2)',
      [event.reference, event.type],
    );
    await client.query(
      `INSERT INTO billing_events (provider_event_id, order_reference, event_type, payload)
       VALUES ($1,$2,$3,$4::jsonb) ON CONFLICT (provider_event_id) DO NOTHING`,
      [event.providerEventId, event.reference, event.type, JSON.stringify(event.raw ?? {})],
    );

    // The sealed quote is the only economic authority. Locked for the transaction.
    const orderRes = await client.query<OrderRow>(
      `SELECT user_id, plan, addon_assets, amount_cents, currency, status
         FROM checkout_orders WHERE reference = $1 FOR UPDATE`,
      [event.reference],
    );
    const order = orderRes.rows[0];
    if (!order) throw new WebhookReject('unknown order', 400);

    const addOns = normalizeAddOns(order.addon_assets);
    assertReferenceMatchesQuote(event.reference, order, addOns);

    const applied = event.type === 'payment.approved'
      ? await applyApproval(client, event, order, addOns, provider.name)
      : await applyNonApproval(client, event, order, provider.name);

    await client.query('COMMIT');
    return NextResponse.json({ received: true, applied });
  } catch (e) {
    await client.query('ROLLBACK').catch(() => undefined);
    if ((e as { code?: string }).code === '23505') {
      // Same reference+type already processed — provider retry, nothing to do.
      return NextResponse.json({ received: true, duplicate: true });
    }
    if (e instanceof WebhookReject) {
      console.warn('[billing/webhook] rejected:', e.code, 'ref:', event.reference);
      return NextResponse.json({ error: e.code }, { status: e.status });
    }
    logServerError('billing/webhook apply', e);
    return NextResponse.json({ error: 'internal error' }, { status: 500 });
  } finally {
    client.release();
  }
}

// ─────────────────────────────────────────────────────────────────── helpers

function normalizeAddOns(value: unknown): string[] {
  const arr = typeof value === 'string' ? safeParse(value) : value;
  return Array.isArray(arr) ? arr.map(String) : [];
}

function safeParse(s: string): unknown {
  try { return JSON.parse(s); } catch { return []; }
}

/**
 * The reference is a correlation id, not a price tag: it must AGREE with the sealed
 * quote or we refuse. (Prevents a forged/replayed reference from redirecting a
 * payment to another user or plan.)
 */
function assertReferenceMatchesQuote(reference: string, order: OrderRow, addOns: string[]) {
  const decoded = decodeReference(reference);
  if (!decoded) throw new WebhookReject('unknown reference format', 400);
  if (decoded.userId !== order.user_id) throw new WebhookReject('reference/quote user mismatch', 400);
  if (decoded.plan !== order.plan) throw new WebhookReject('reference/quote plan mismatch', 400);
  const a = [...decoded.addOns].sort().join('|');
  const b = [...addOns].sort().join('|');
  if (a !== b) throw new WebhookReject('reference/quote add-on mismatch', 400);
}

async function applyApproval(
  client: PoolClient, event: NormalizedBillingEvent, order: OrderRow,
  addOns: string[], providerName: string,
): Promise<boolean> {
  const base = PLAN_DEFAULTS[order.plan as keyof typeof PLAN_DEFAULTS];
  if (!base || order.plan === 'free') throw new WebhookReject('unknown plan in quote', 400);
  if (addOns.some((id) => typeof addonPricesCop()[id] !== 'number')) {
    throw new WebhookReject('unknown add-on', 400);
  }

  // Currency and amount are checked against the SEALED quote, not the price list.
  const sealedAmount = Number(order.amount_cents);
  if (!Number.isFinite(sealedAmount) || sealedAmount <= 0) throw new WebhookReject('invalid sealed amount', 400);
  if (!event.currency || event.currency !== order.currency) throw new WebhookReject('currency mismatch', 400);
  if (event.amountInCents == null || event.amountInCents !== sealedAmount) {
    throw new WebhookReject('amount mismatch', 400);
  }

  // Only an order that is still awaiting payment may become paid (mig. 059).
  const upd = await client.query(
    `UPDATE checkout_orders SET status='paid' WHERE reference=$1 AND status = ANY($2::text[])`,
    [event.reference, ['created', 'pending']],
  );
  if (upd.rowCount === 0) throw new WebhookReject('order not payable in its current state', 409);

  const entitlements = {
    ...base,
    assets: [...new Set([...base.assets, ...addOns])],
    expires_at: new Date(Date.now() + 30 * 86_400_000).toISOString(),
  };
  await client.query(
    'UPDATE sb_users SET entitlements = $1::jsonb WHERE id = $2',
    [JSON.stringify(entitlements), order.user_id],
  );
  await audit(client, order.user_id, 'plan_change', {
    provider: providerName, plan: order.plan, addOns, reference: event.reference,
    amount_cents: sealedAmount, currency: order.currency,
    provider_event_id: event.providerEventId,
  });
  return true;
}

async function applyNonApproval(
  client: PoolClient, event: NormalizedBillingEvent, order: OrderRow, providerName: string,
): Promise<boolean> {
  const rule = ORDER_TRANSITION[event.type as Exclude<BillingEventType, 'payment.approved'>];
  if (!rule) throw new WebhookReject('unsupported event type', 400);

  // `from` encodes migration 059's legal lifecycle: a cancellation/decline can only
  // affect an order that was never paid; refund/chargeback only a paid one.
  const upd = await client.query(
    `UPDATE checkout_orders SET status=$1 WHERE reference=$2 AND status = ANY($3::text[])`,
    [rule.to, event.reference, rule.from],
  );
  const applied = (upd.rowCount ?? 0) > 0;

  if (applied && (rule.to === 'refunded' || rule.to === 'charged_back')) {
    // Money came back ⇒ the paid-for add-ons go away. (Plan-level revocation is an
    // OPERATOR DECISION — see report; not invented here.)
    await client.query(
      "UPDATE sb_users SET entitlements = jsonb_set(entitlements, '{assets}', '[]'::jsonb) WHERE id=$1",
      [order.user_id],
    );
  }
  await audit(client, order.user_id, applied ? 'plan_payment_failed' : 'plan_payment_event_noop', {
    provider: providerName, type: event.type, reference: event.reference,
    order_status: order.status, applied, provider_event_id: event.providerEventId,
  });
  return applied;
}

/**
 * Audit is part of the money transaction: if it cannot be written, NOTHING is
 * applied (append-only audit_log, CTR-RBAC-001 rule 4). Never swallowed.
 */
async function audit(client: PoolClient, userId: string, action: string, detail: Record<string, unknown>) {
  await client.query(
    'INSERT INTO audit_log (user_id, action, object_type, detail) VALUES ($1,$2,$3,$4::jsonb)',
    [userId, action, 'entitlements', JSON.stringify(detail)],
  );
}
