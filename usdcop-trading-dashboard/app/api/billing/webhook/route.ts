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
 *
 * MONEY INVARIANTS (CODEX P0 round 3 — CXD-059):
 *  0. **The event is confirmed SERVER-TO-SERVER before anything is mutated.** The
 *     ledger of invariant 4 stops the SECOND use of a provider event; it cannot stop
 *     the FIRST. An attacker with any validly-signed APPROVED body opens his own
 *     checkout for the same plan (same amount, same currency) and replays that body
 *     under HIS reference: signature valid, ledger clean, sealed quote matching —
 *     because it is his own quote. `GET /transactions/:id` is the only account of the
 *     payment he cannot author. Timeout/5xx ⇒ 503 and NOTHING written (a postponed
 *     credit is recovered by the retry); contradiction ⇒ 4xx + incident.
 *     See `lib/billing/confirmation.ts`.
 *
 * MONEY INVARIANTS (CODEX P0 round 2 — CXD-056):
 *  4. `billing_events.provider_event_id` is the AUTHORITATIVE global idempotency key,
 *     asserted inside the same transaction with `ON CONFLICT DO NOTHING RETURNING`.
 *     It also stores the reference the event was first seen with, which is what closes
 *     the replay: the provider's checksum does NOT cover `transaction.reference`
 *     (see `lib/billing/event-ledger.ts`), so a signed event can be replayed with
 *     another user's reference and still verify. Same id + same everything ⇒ retry,
 *     no-op. Same id + a different reference/payload ⇒ SECURITY INCIDENT, rejected and
 *     recorded; never credited. `billing_webhook_events` (reference,event_type) does
 *     NOT catch this — its key changes with the forged reference.
 *  5. An approval UNIONS entitlements onto the CURRENT row read `FOR UPDATE`; it never
 *     replaces them. Replacing lost every asset bought in an earlier order and made
 *     two concurrent checkouts last-write-wins. A refund subtracts ONLY that order's
 *     `addon_assets`. Revoking the PLAN itself is an OPERATOR DECISION, not ours.
 * Fail-closed everywhere: when in doubt we do not credit.
 */
import { NextResponse } from 'next/server';
import type { PoolClient } from 'pg';

import { getClient } from '@/lib/db/postgres-client';
import { getBillingProvider, decodeReference } from '@/lib/billing';
import type {
  BillingEventType, BillingProvider, NormalizedBillingEvent, TransactionLookup,
} from '@/lib/billing';
import { classifyLedgerConflict, type LedgerRow } from '@/lib/billing/event-ledger';
import { confirmAgainstProvider, type ConfirmationVerdict } from '@/lib/billing/confirmation';
import { PLAN_DEFAULTS, effectiveEntitlements } from '@/lib/contracts/rbac.contract';
import type { Entitlements } from '@/lib/contracts/rbac.contract';
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
  constructor(
    public code: string,
    public status = 400,
    /** When set, an audit row is written AFTER the rollback (it must outlive it). */
    public incident?: { userId: string | null; detail: Record<string, unknown> },
  ) { super(code); }
}

/** A genuine provider retry: nothing to apply, nothing wrong. */
class WebhookDuplicate extends Error {}

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
    // ── STEP 0 (CXD-059): confirm the event against the provider's OWN API, BEFORE
    // `BEGIN` and before a single row moves. Deliberately outside the transaction: it
    // is a network call (no lock may be held across it) and, on doubt, it must leave
    // the database — INCLUDING the idempotency ledger — completely untouched, so the
    // provider's retry is a fresh attempt and not a swallowed "duplicate".
    const verdict = await confirmWithProvider(provider, event);
    if (verdict.kind === 'unavailable') {
      // We could not reach a verdict. 503 ⇒ the provider retries and the legitimate
      // payment is credited then (proven by test "THE RETRY AFTER THE OUTAGE
      // CREDITS"). An outage postpones money; it never grants it.
      logServerError('billing/webhook confirm', new Error(verdict.reason));
      return NextResponse.json({ error: 'temporarily unavailable' }, { status: 503 });
    }
    if (verdict.kind === 'mismatch') {
      // The provider's record contradicts the body. Never retryable, never credited.
      // The authoritative snapshot names ANOTHER customer's reference, so it goes to
      // the audit row only — never to the response (that would confirm to the attacker
      // whose payment he just tried to steal).
      throw new WebhookReject('transaction not confirmed by provider', 409, {
        userId: decodeReference(verdict.authoritative.reference ?? '')?.userId ?? null,
        detail: {
          reason: 'authoritative confirmation mismatch',
          mismatched_field: verdict.field,
          provider_event_id: event.providerEventId,
          claimed_reference: event.reference,
          claimed_user_id: decodeReference(event.reference)?.userId ?? null,
          authoritative_reference: verdict.authoritative.reference ?? null,
          authoritative_status: verdict.authoritative.status ?? null,
          authoritative_amount_cents: verdict.authoritative.amountInCents ?? null,
          authoritative_currency: verdict.authoritative.currency ?? null,
          unauthenticated_fields: event.unauthenticatedFields ?? [],
        },
      });
    }

    await client.query('BEGIN');
    // ── AUTHORITATIVE idempotency: the provider event id, INSIDE the transaction.
    // It binds this provider event to the reference it first arrived with. The
    // reference is NOT signed by the provider, so this binding — not the checksum —
    // is what stops a signed event from being replayed onto another user's order.
    // If anything below fails, this row rolls back too, so the provider's retry can
    // still credit the user (a transient DB error must not deny a paid entitlement).
    await assertFirstSightingOfProviderEvent(client, event);
    // Secondary ledger (migration 058, UNIQUE(reference,event_type)): catches a retry
    // that the provider re-sends under a NEW transaction id for the same order.
    await client.query(
      'INSERT INTO billing_webhook_events (reference,event_type) VALUES ($1,$2)',
      [event.reference, event.type],
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
      : await applyNonApproval(client, event, order, addOns, provider.name);

    await client.query('COMMIT');
    return NextResponse.json({ received: true, applied });
  } catch (e) {
    await client.query('ROLLBACK').catch(() => undefined);
    if (e instanceof WebhookDuplicate || (e as { code?: string }).code === '23505') {
      // Already processed — provider retry, nothing to do, nothing credited again.
      return NextResponse.json({ received: true, duplicate: true });
    }
    if (e instanceof WebhookReject) {
      if (e.incident) {
        // The incident must OUTLIVE the rolled-back money transaction: it is written
        // on a fresh implicit transaction on the same connection. Best effort — a
        // failure to record must not turn a rejection into an acceptance.
        console.warn('[billing/webhook] SECURITY', e.code, e.incident.detail);
        await audit(client, e.incident.userId, 'billing_replay_blocked', e.incident.detail)
          .catch((err) => logServerError('billing/webhook incident', err));
      }
      console.warn('[billing/webhook] rejected:', e.code, 'ref:', event.reference);
      return NextResponse.json({ error: e.code }, { status: e.status });
    }
    logServerError('billing/webhook apply', e);
    return NextResponse.json({ error: 'internal error' }, { status: 500 });
  } finally {
    client.release();
  }
}

/**
 * Ask the provider what it actually recorded, then compare (CXD-059).
 *
 * Two fail-closed guarantees live here:
 *  - a provider that VIOLATES its contract by throwing is still only `unavailable`;
 *    an exception must never become an approval;
 *  - an event carrying NO provider transaction id cannot be confirmed at all. If it
 *    could CREATE value it is refused outright. Non-crediting events (declines,
 *    cancellations, refunds) are allowed through — they cannot grant anything, and
 *    Wompi's query API has no vocabulary for a refund, so demanding a confirmation it
 *    cannot express would reject legitimate revocations. Every event Wompi actually
 *    emits carries an id and IS confirmed; this branch is only reachable for a future
 *    provider (see the residual-risk note in the handover report).
 */
async function confirmWithProvider(
  provider: BillingProvider, event: NormalizedBillingEvent,
): Promise<ConfirmationVerdict> {
  if (!event.providerTransactionId) {
    if (event.type === 'payment.approved') {
      return { kind: 'mismatch', field: 'transaction_id', authoritative: {} };
    }
    console.warn('[billing/webhook] unconfirmable non-crediting event:', event.type, event.providerEventId);
    return { kind: 'confirmed' };
  }

  let lookup: TransactionLookup;
  try {
    lookup = await provider.fetchTransaction(event.providerTransactionId);
  } catch (e) {
    logServerError('billing/webhook fetchTransaction', e);
    lookup = { kind: 'unavailable', reason: 'confirmation port threw' };
  }
  return confirmAgainstProvider(event, lookup);
}

/**
 * Claim `event.providerEventId` in the append-only ledger, or explain the collision.
 *
 * `ON CONFLICT DO NOTHING RETURNING` yields zero rows when the id already exists —
 * the previous code used the same statement WITHOUT `RETURNING` and ignored the
 * conflict, so a signed event replayed with a second reference sailed past it and
 * credited that second reference (CODEX CXD-056). Fail-closed: any collision that is
 * not identical in reference, type and payload is a security incident.
 */
async function assertFirstSightingOfProviderEvent(
  client: PoolClient, event: NormalizedBillingEvent,
) {
  const claimed = await client.query(
    `INSERT INTO billing_events (provider_event_id, order_reference, event_type, payload)
     VALUES ($1,$2,$3,$4::jsonb)
     ON CONFLICT (provider_event_id) DO NOTHING
     RETURNING provider_event_id`,
    [event.providerEventId, event.reference, event.type, JSON.stringify(event.raw ?? {})],
  );
  if ((claimed.rowCount ?? 0) > 0) return;

  const prior = await client.query<LedgerRow>(
    'SELECT order_reference, event_type, payload FROM billing_events WHERE provider_event_id = $1',
    [event.providerEventId],
  );
  const row = prior.rows[0];
  // A conflict with no readable prior row is unexplained ⇒ we do not credit.
  if (!row) throw new WebhookReject('ledger conflict without a prior row', 409);

  const verdict = classifyLedgerConflict(row, {
    reference: event.reference, type: event.type, raw: event.raw,
  });
  if (verdict.kind === 'duplicate') throw new WebhookDuplicate();

  throw new WebhookReject('provider event already bound to another reference', 409, {
    // Attributed to the account whose real payment is being reused.
    userId: decodeReference(verdict.boundReference)?.userId ?? null,
    detail: {
      provider_event_id: event.providerEventId,
      bound_reference: verdict.boundReference,
      claimed_reference: event.reference,
      claimed_user_id: decodeReference(event.reference)?.userId ?? null,
      changed: verdict.changed,
      event_type: event.type,
      // Why this is possible at all — see lib/billing/event-ledger.ts.
      unauthenticated_fields: event.unauthenticatedFields ?? [],
    },
  });
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

  // ── UNION, never replace (CODEX CXD-056). `FOR UPDATE` holds the row for the whole
  // transaction: without it two approvals landing at once are last-write-wins and one
  // paid order silently disappears. `effectiveEntitlements` is the SSOT for "what does
  // this user actually hold right now" — an EXPIRED row degrades to free, so expired
  // rights are correctly NOT carried over.
  const currentRow = await client.query<{ entitlements: unknown }>(
    'SELECT entitlements FROM sb_users WHERE id = $1 FOR UPDATE',
    [order.user_id],
  );
  if (currentRow.rowCount === 0) throw new WebhookReject('unknown user for the sealed quote', 400);
  const current = effectiveEntitlements(parseEntitlements(currentRow.rows[0].entitlements));

  // Time already paid for is a purchased right too: never shorten it.
  const grantedUntil = Date.now() + 30 * 86_400_000;
  const ownedUntil = current.expires_at ? Date.parse(current.expires_at) : NaN;
  const expiresAt = new Date(
    Number.isFinite(ownedUntil) && ownedUntil > grantedUntil ? ownedUntil : grantedUntil,
  ).toISOString();

  const entitlements: Entitlements = {
    ...base,
    assets: [...new Set([...current.assets, ...base.assets, ...addOns])],
    expires_at: expiresAt,
  };
  await client.query(
    'UPDATE sb_users SET entitlements = $1::jsonb WHERE id = $2',
    [JSON.stringify(entitlements), order.user_id],
  );
  await audit(client, order.user_id, 'plan_change', {
    provider: providerName, plan: order.plan, addOns, reference: event.reference,
    amount_cents: sealedAmount, currency: order.currency,
    provider_event_id: event.providerEventId,
    // Recoverable trail: what the user held before this purchase. A purchase of a
    // DIFFERENT plan applies the plan of the sealed quote; ranking plans (and thus
    // refusing a "downgrade") is an OPERATOR DECISION, not invented here.
    previous_plan: current.plan, previous_assets: current.assets,
    previous_expires_at: current.expires_at,
  });
  return true;
}

async function applyNonApproval(
  client: PoolClient, event: NormalizedBillingEvent, order: OrderRow,
  addOns: string[], providerName: string,
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
    // Money came back ⇒ ONLY the add-ons THIS order paid for go away. Wiping
    // `assets` to `[]` also deleted assets bought in other, un-refunded orders
    // (CODEX CXD-056). Same row lock as the approval path, same reason.
    const currentRow = await client.query<{ entitlements: unknown }>(
      'SELECT entitlements FROM sb_users WHERE id = $1 FOR UPDATE',
      [order.user_id],
    );
    // Operate on the STORED row, not the effective one: a refund must not silently
    // rewrite the plan of an expired user. (Plan-level revocation is an OPERATOR
    // DECISION — see report; not invented here.)
    const stored = parseEntitlements(currentRow.rows[0]?.entitlements) ?? { ...PLAN_DEFAULTS.free };
    const refunded = new Set(addOns);
    const next = {
      ...stored,
      assets: (Array.isArray(stored.assets) ? stored.assets : []).filter((a) => !refunded.has(a)),
    };
    await client.query(
      'UPDATE sb_users SET entitlements = $1::jsonb WHERE id = $2',
      [JSON.stringify(next), order.user_id],
    );
  }
  await audit(client, order.user_id, applied ? 'plan_payment_failed' : 'plan_payment_event_noop', {
    provider: providerName, type: event.type, reference: event.reference,
    order_status: order.status, applied, provider_event_id: event.providerEventId,
    revoked_assets: applied && (rule.to === 'refunded' || rule.to === 'charged_back') ? addOns : [],
  });
  return applied;
}

/** `jsonb` may arrive already parsed (pg) or as text. Anything else ⇒ null (fail-closed). */
function parseEntitlements(value: unknown): Entitlements | null {
  const raw = typeof value === 'string' ? safeParse(value) : value;
  return raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Entitlements) : null;
}

/**
 * Audit is part of the money transaction: if it cannot be written, NOTHING is
 * applied (append-only audit_log, CTR-RBAC-001 rule 4). Never swallowed.
 */
async function audit(client: PoolClient, userId: string | null, action: string, detail: Record<string, unknown>) {
  await client.query(
    'INSERT INTO audit_log (user_id, action, object_type, detail) VALUES ($1,$2,$3,$4::jsonb)',
    [userId, action, 'entitlements', JSON.stringify(detail)],
  );
}
