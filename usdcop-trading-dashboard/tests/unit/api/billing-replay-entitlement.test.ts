/**
 * P0 money-safety, second round — CODEX finding **CXD-056** (`.claude/coordination/
 * INBOX-CLAUDE.md`), reproduced by his probe
 * `.claude/coordination/integration/probes/billing_reference_replay_probe.test.ts`.
 *
 * This file is the PERMANENT protection of the ROUTE that his loose probe asked for
 * ("probe replay convertido en protección de route"). Origin of both findings: CODEX.
 *
 * ── P0-A · REPLAY CROSS-REFERENCE ────────────────────────────────────────────────
 * Wompi's webhook checksum is `sha256(<values of signature.properties> + timestamp +
 * events_secret)` and its `signature.properties` are `transaction.id`,
 * `transaction.status`, `transaction.amount_in_cents`. **`transaction.reference` — the
 * field that decides WHO gets credited — is NOT covered by the signature.** Therefore
 * two events with the same transaction id/status/amount and DIFFERENT references carry
 * the byte-identical, genuinely valid checksum. The provider cannot tell them apart:
 * that guarantee does not exist and we do not claim it (K-031). The replay is closed
 * where it CAN be closed — the append-only ledger keyed by `provider_event_id`.
 *
 * ── P0-B · LOSS OF ALREADY-PURCHASED RIGHTS ──────────────────────────────────────
 * `applyApproval` replaced entitlements with `PLAN_DEFAULTS + addOns` without reading
 * or locking the current row: buying BTC deleted an owned XAU, concurrent checkouts
 * were last-write-wins, and a refund wiped `assets` to `[]` including assets from
 * other paid orders.
 *
 * Mutations that MUST turn these red again:
 *  A1  `ON CONFLICT (provider_event_id) DO NOTHING` without `RETURNING` / without
 *      acting on `rowCount === 0`            → replay credits the second reference
 *  A2  treating a ledger conflict as a duplicate no-op regardless of the reference
 *      → forged reference silently accepted (200 instead of 409)
 *  B1  `assets: [...base.assets, ...addOns]` (no union with the current row)
 *  B2  dropping `FOR UPDATE` from the sb_users read → last-write-wins returns
 *  B3  `jsonb_set(entitlements,'{assets}','[]')` on refund → other assets vanish
 */
import { createHash } from 'node:crypto';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/db/postgres-client', async () => {
  const { scriptedPg } = await import('../../mocks/scripted-postgres');
  return {
    query: scriptedPg.query,
    pgQuery: scriptedPg.query,
    getClient: scriptedPg.getClient,
    getPool: () => { throw new Error('pool disabled in tests'); },
  };
});

import { scriptedPg as pg } from '../../mocks/scripted-postgres';
import { POST as webhookPOST } from '@/app/api/billing/webhook/route';

const EVENTS_SECRET = 'events-secret';
const USER_A = '11111111-1111-4111-8111-111111111111';
const USER_B = '22222222-2222-4222-8222-222222222222';

const PLAN_PRICE = 9_900_000;

/**
 * A genuinely signed Wompi event. The checksum covers ONLY id/status/amount — pass a
 * different `reference` with the same id/status/amount and the checksum is unchanged
 * and still verifies. That is the attack, not a test shortcut.
 */
function signedEvent(opts: {
  txId?: string; status?: string; reference: string; amountInCents?: number; currency?: string;
}) {
  const tx = {
    id: opts.txId ?? 'tx_replay_1',
    status: opts.status ?? 'APPROVED',
    reference: opts.reference,
    amount_in_cents: opts.amountInCents ?? PLAN_PRICE,
    currency: opts.currency ?? 'COP',
  };
  const timestamp = 1_700_000_000;
  const properties = ['transaction.id', 'transaction.status', 'transaction.amount_in_cents'];
  const checksum = createHash('sha256')
    .update(`${tx.id}${tx.status}${tx.amount_in_cents}${timestamp}${EVENTS_SECRET}`)
    .digest('hex');
  return JSON.stringify({
    event: 'transaction.updated',
    data: { transaction: tx },
    signature: { properties, checksum },
    timestamp,
  });
}

const post = (body: string) =>
  webhookPOST(new Request('http://t/api/billing/webhook', { method: 'POST', body }));

const refFor = (user: string, addOns: string[] = [], ts = 1700000000000) =>
  `sub_signals_${user}_${addOns.join('+') || 'base'}_${ts}`;

function seedUser(id: string, entitlements: Record<string, unknown>) {
  pg.state.users.set(id, { email: `${id}@example.com`, role: 'free', entitlements });
}

function seedOrder(reference: string, user: string, opts: {
  addOns?: string[]; amount?: number; status?: string; plan?: string;
} = {}) {
  pg.state.orders.set(reference, {
    user_id: user, plan: opts.plan ?? 'signals', addon_assets: opts.addOns ?? [],
    amount_cents: opts.amount ?? PLAN_PRICE, currency: 'COP', reference,
    status: opts.status ?? 'pending',
  });
}

const freeEnt = { plan: 'free', assets: ['usdcop'], expires_at: null };
const ent = (u: string) => pg.state.users.get(u)!.entitlements as { plan: string; assets: string[]; expires_at: string | null };

beforeEach(() => {
  pg.reset();
  vi.restoreAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  process.env.WOMPI_EVENTS_SECRET = EVENTS_SECRET;
  delete process.env.BILLING_PRICES_COP;
  delete process.env.BILLING_ADDON_PRICES_COP;
});

// ══════════════════════════════════════════════════ P0-A · replay cross-reference
describe('P0-A the same provider event id can never credit a second reference (CODEX CXD-056)', () => {
  const REF_A = refFor(USER_A);
  const REF_B = refFor(USER_B);

  function seedBothVictims() {
    seedUser(USER_A, { ...freeEnt });
    seedUser(USER_B, { ...freeEnt });
    seedOrder(REF_A, USER_A);
    seedOrder(REF_B, USER_B); // same plan, same amount ⇒ same checksum
  }

  it('the forged reference has a VALID signature — this is not a signature bug', async () => {
    const { WompiProvider } = await import('@/lib/billing/wompi');
    const provider = new WompiProvider();
    const a = await provider.verifyWebhook(signedEvent({ reference: REF_A }), new Headers());
    const b = await provider.verifyWebhook(signedEvent({ reference: REF_B }), new Headers());
    // Both verify. The checksum genuinely does not cover the reference; the defence
    // therefore CANNOT live here (documented, not claimed).
    expect(a.valid).toBe(true);
    expect(b.valid).toBe(true);
    expect(a.event?.providerEventId).toBe(b.event?.providerEventId);
    // ...and the provider must SAY that the reference is unauthenticated.
    expect(a.event?.unauthenticatedFields).toContain('reference');
  });

  it('credits user A, then REFUSES to credit user B with the same transaction', async () => {
    seedBothVictims();

    const first = await post(signedEvent({ reference: REF_A }));
    expect(first.status).toBe(200);
    expect(ent(USER_A).plan).toBe('signals');

    // Snapshot everything the attacker wants to move.
    const beforeOrderB = JSON.stringify(pg.state.orders.get(REF_B));
    const beforeEntB = JSON.stringify(ent(USER_B));
    const beforeAudit = JSON.stringify(pg.state.audit);

    const replay = await post(signedEvent({ reference: REF_B }));

    expect(replay.status).toBeGreaterThanOrEqual(400);
    expect(replay.status).toBeLessThan(500);
    // Order, entitlement and audit are byte-identical (CODEX's mutation test).
    expect(JSON.stringify(pg.state.orders.get(REF_B))).toBe(beforeOrderB);
    expect(JSON.stringify(ent(USER_B))).toBe(beforeEntB);
    expect(ent(USER_B).plan).toBe('free');
    // The money audit trail of the LEGITIMATE payment is untouched too.
    expect(JSON.stringify(pg.state.audit.filter((r) => r.action === 'plan_change'))).toBe(
      JSON.stringify(JSON.parse(beforeAudit).filter((r: Record<string, unknown>) => r.action === 'plan_change')),
    );
  });

  it('records the blocked replay as a security incident that survives the rollback', async () => {
    seedBothVictims();
    await post(signedEvent({ reference: REF_A }));
    await post(signedEvent({ reference: REF_B }));

    const incidents = pg.state.audit.filter((r) => String(r.action).includes('replay'));
    expect(incidents.length, 'a blocked replay must leave a trace').toBe(1);
    const detail = JSON.parse(String(incidents[0].detail));
    expect(detail.provider_event_id).toContain('tx_replay_1');
    expect(detail.bound_reference).toBe(REF_A);
    expect(detail.claimed_reference).toBe(REF_B);
  });

  it('a genuine provider retry (identical event) is a duplicate no-op, not an incident', async () => {
    seedBothVictims();
    const evt = signedEvent({ reference: REF_A });
    await post(evt);
    const afterFirst = JSON.stringify(ent(USER_A));
    const auditCount = pg.state.audit.length;

    const retry = await post(evt);
    expect(retry.status).toBe(200);
    expect(await retry.json()).toMatchObject({ duplicate: true });
    expect(JSON.stringify(ent(USER_A))).toBe(afterFirst);
    expect(pg.state.audit.length).toBe(auditCount);
    expect(pg.state.audit.filter((r) => String(r.action).includes('replay')).length).toBe(0);
  });

  it('the ledger — not `billing_webhook_events` — is what stops it', async () => {
    seedBothVictims();
    await post(signedEvent({ reference: REF_A }));
    // The (reference,event_type) key of migration 058 does NOT collide across
    // references: this is why provider_event_id has to be the authoritative key.
    expect(pg.state.webhookEvents.has(`${REF_B}:payment.approved`)).toBe(false);
    await post(signedEvent({ reference: REF_B }));
    expect(pg.state.billingEvents.get('wompi:tx_replay_1:APPROVED')!.order_reference).toBe(REF_A);
  });
});

// ══════════════════════════════════════════════ P0-B · already-purchased rights
describe('P0-B an approval never destroys rights the user already paid for (CODEX CXD-056)', () => {
  it('a user who owns XAU and buys BTC keeps BOTH', async () => {
    seedUser(USER_A, {
      plan: 'signals', assets: ['usdcop', 'xauusd'],
      expires_at: new Date(Date.now() + 20 * 86_400_000).toISOString(),
    });
    const ref = refFor(USER_A, ['btcusdt']);
    seedOrder(ref, USER_A, { addOns: ['btcusdt'], amount: PLAN_PRICE + 3_900_000 });

    const res = await post(signedEvent({ reference: ref, amountInCents: PLAN_PRICE + 3_900_000 }));
    expect(res.status).toBe(200);
    expect(ent(USER_A).assets).toContain('btcusdt');
    expect(ent(USER_A).assets, 'XAU was already paid for').toContain('xauusd');
  });

  it('two approvals in a row union — neither payment disappears', async () => {
    seedUser(USER_A, { ...freeEnt });
    const refGold = refFor(USER_A, ['xauusd'], 1700000000001);
    const refBtc = refFor(USER_A, ['btcusdt'], 1700000000002);
    seedOrder(refGold, USER_A, { addOns: ['xauusd'], amount: PLAN_PRICE + 3_900_000 });
    seedOrder(refBtc, USER_A, { addOns: ['btcusdt'], amount: PLAN_PRICE + 3_900_000 });

    await post(signedEvent({ txId: 'tx_gold', reference: refGold, amountInCents: PLAN_PRICE + 3_900_000 }));
    await post(signedEvent({ txId: 'tx_btc', reference: refBtc, amountInCents: PLAN_PRICE + 3_900_000 }));

    expect(ent(USER_A).assets).toEqual(expect.arrayContaining(['usdcop', 'xauusd', 'btcusdt']));
  });

  it('serializes concurrent approvals with a row lock (SELECT ... FOR UPDATE)', async () => {
    seedUser(USER_A, { ...freeEnt });
    const ref = refFor(USER_A, ['xauusd']);
    seedOrder(ref, USER_A, { addOns: ['xauusd'], amount: PLAN_PRICE + 3_900_000 });
    await post(signedEvent({ reference: ref, amountInCents: PLAN_PRICE + 3_900_000 }));

    const locked = pg.state.log.filter(
      (s) => /SELECT entitlements FROM sb_users/i.test(s) && /FOR UPDATE/i.test(s),
    );
    expect(locked.length, 'the union is only safe under a row lock').toBeGreaterThan(0);
  });

  it('never shortens an entitlement the user already paid for', async () => {
    const farFuture = new Date(Date.now() + 300 * 86_400_000).toISOString();
    seedUser(USER_A, { plan: 'signals', assets: ['usdcop'], expires_at: farFuture });
    const ref = refFor(USER_A);
    seedOrder(ref, USER_A);

    await post(signedEvent({ reference: ref }));
    expect(new Date(ent(USER_A).expires_at!).getTime()).toBeGreaterThanOrEqual(
      new Date(farFuture).getTime(),
    );
  });

  it('does not carry over rights that already EXPIRED', async () => {
    seedUser(USER_A, {
      plan: 'signals', assets: ['usdcop', 'xauusd'],
      expires_at: new Date(Date.now() - 86_400_000).toISOString(),
    });
    const ref = refFor(USER_A);
    seedOrder(ref, USER_A);

    await post(signedEvent({ reference: ref }));
    expect(ent(USER_A).assets).not.toContain('xauusd');
  });
});

describe('P0-B a refund removes ONLY what that order paid for (CODEX CXD-056)', () => {
  const refBtc = refFor(USER_A, ['btcusdt']);

  async function refund(reference: string) {
    const billing = await import('@/lib/billing');
    const provider = {
      name: 'wompi',
      createCheckout: async () => { throw new Error('n/a'); },
      verifyWebhook: async () => ({
        valid: true,
        event: {
          type: 'payment.refunded' as const, reference,
          providerEventId: `wompi:refund_${reference.slice(-6)}`, raw: {},
        },
      }),
    };
    const spy = vi.spyOn(billing, 'getBillingProvider').mockReturnValue(provider as never);
    try {
      return await post('{}');
    } finally {
      spy.mockRestore();
    }
  }

  it('refunding BTC leaves the separately purchased XAU in place', async () => {
    seedUser(USER_A, {
      plan: 'signals', assets: ['usdcop', 'xauusd', 'btcusdt'],
      expires_at: new Date(Date.now() + 20 * 86_400_000).toISOString(),
    });
    seedOrder(refBtc, USER_A, { addOns: ['btcusdt'], status: 'paid', amount: PLAN_PRICE + 3_900_000 });

    const res = await refund(refBtc);
    expect(res.status).toBeLessThan(500);
    expect(pg.state.orders.get(refBtc)!.status).toBe('refunded');
    expect(ent(USER_A).assets).not.toContain('btcusdt');
    expect(ent(USER_A).assets, 'XAU was NOT refunded').toContain('xauusd');
    expect(ent(USER_A).assets).toContain('usdcop');
  });

  it('does not revoke the PLAN (operator decision, not the webhook’s)', async () => {
    seedUser(USER_A, {
      plan: 'signals', assets: ['usdcop', 'btcusdt'],
      expires_at: new Date(Date.now() + 20 * 86_400_000).toISOString(),
    });
    seedOrder(refBtc, USER_A, { addOns: ['btcusdt'], status: 'paid', amount: PLAN_PRICE + 3_900_000 });

    await refund(refBtc);
    expect(ent(USER_A).plan).toBe('signals');
  });
});
