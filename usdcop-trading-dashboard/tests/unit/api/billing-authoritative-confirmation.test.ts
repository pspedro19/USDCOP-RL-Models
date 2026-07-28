/**
 * P0 money-safety, third round — CODEX finding **CXD-059**.
 *
 * ── THE HOLE THE LEDGER DOES NOT CLOSE ───────────────────────────────────────────
 * `billing_events.provider_event_id` (CXD-056) stops the SECOND use of a provider
 * event. It does NOT stop the FIRST. Wompi's checksum covers
 * `transaction.id|status|amount_in_cents` and **not `transaction.reference`**, so an
 * attacker holding ANY validly-signed APPROVED body can:
 *   1. open his own checkout for the same plan (⇒ same `amount_in_cents`, same `COP`),
 *   2. POST that body to our webhook with `reference` swapped for HIS reference.
 * Signature: valid. Ledger: first sighting, no conflict. Sealed quote: user, plan,
 * add-ons, amount and currency all match — because it is HIS OWN quote. He is credited
 * without paying, and nothing in the body betrays it.
 *
 * ── THE ONLY FIX: ASK THE PROVIDER ──────────────────────────────────────────────
 * Before mutating an order or an entitlement, confirm the transaction
 * server-to-server (`GET /v1/transactions/:id`) through an injectable port and require
 * the AUTHORITATIVE `id`/`reference`/`status`/`amount`/`currency` to match the event.
 *
 * This was previously recorded as an OPERATOR DECISION on the grounds that failing
 * closed would let a provider outage block legitimate payments. **CODEX rejected that
 * and is right**: an outage does not LOSE a legitimate payment, it postpones it — the
 * provider retries and the retry credits (test 3 below proves exactly that). Failing
 * open gives money away. It is not a trade-off.
 *
 * Mutations that MUST turn these red again:
 *  C1  skip the confirmation call for `payment.approved`      → test 1 credits a forgery
 *  C2  treat `unavailable` (timeout/5xx) as "confirmed"       → test 3 credits blind
 *  C3  compare only the amount and not the reference          → test 1 credits a forgery
 *  C4  return the provider/transport error message verbatim   → test 5 leaks
 */
import { createHash } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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
import { createWompiTransactionsApi } from '../../mocks/wompi-transactions-api';
import { POST as webhookPOST } from '@/app/api/billing/webhook/route';

const EVENTS_SECRET = 'events-secret';
/** The real payer. */
const VICTIM = '11111111-1111-4111-8111-111111111111';
/** The attacker: a real account, with a real (unpaid) pending order of his own. */
const ATTACKER = '22222222-2222-4222-8222-222222222222';
const PLAN_PRICE = 9_900_000;

/**
 * The provider's authoritative API. STRICT here: a recorded transaction is immutable,
 * so no test can accidentally make a forgery true. `vi.stubGlobal` replaces only the
 * transport — the real `WompiProvider.fetchTransaction` (URL, status classification,
 * parsing) is what executes.
 */
const providerApi = createWompiTransactionsApi();

// ─────────────────────────────────────────────────────────── signed webhook bodies
/**
 * A genuinely signed Wompi event. The checksum covers ONLY id/status/amount: pass a
 * different `reference` with the same id/status/amount and the checksum is UNCHANGED
 * and still verifies. That is the attack, not a test shortcut.
 */
function signedEvent(opts: {
  txId?: string; status?: string; reference: string; amountInCents?: number; currency?: string;
}) {
  const tx = {
    id: opts.txId ?? 'tx_victim_1',
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

const refFor = (user: string, ts: number) => `sub_signals_${user}_base_${ts}`;
const REF_VICTIM = refFor(VICTIM, 1_700_000_000_001);
const REF_ATTACKER = refFor(ATTACKER, 1_700_000_000_002);

const freeEnt = { plan: 'free', assets: ['usdcop'], expires_at: null };
const ent = (u: string) =>
  pg.state.users.get(u)!.entitlements as { plan: string; assets: string[]; expires_at: string | null };

function seedUser(id: string) {
  pg.state.users.set(id, { email: `${id}@example.com`, role: 'free', entitlements: { ...freeEnt } });
}

function seedOrder(reference: string, user: string) {
  pg.state.orders.set(reference, {
    user_id: user, plan: 'signals', addon_assets: [],
    amount_cents: PLAN_PRICE, currency: 'COP', reference, status: 'pending',
  });
}

/** The payment that ACTUALLY happened, as the provider's API reports it. */
function seedRealPayment(txId: string, reference: string) {
  providerApi.record({
    id: txId, status: 'APPROVED', reference, amount_in_cents: PLAN_PRICE, currency: 'COP',
  });
}

beforeEach(() => {
  pg.reset();
  providerApi.reset();
  vi.restoreAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.stubGlobal('fetch', providerApi.fetch);
  process.env.WOMPI_EVENTS_SECRET = EVENTS_SECRET;
  process.env.WOMPI_PUBLIC_KEY = 'pub_test_key';
  delete process.env.BILLING_PRICES_COP;
  delete process.env.BILLING_ADDON_PRICES_COP;

  // Both accounts exist; both have a pending order for the same plan ⇒ same amount,
  // same currency. Only ONE of them was actually paid.
  seedUser(VICTIM);
  seedUser(ATTACKER);
  seedOrder(REF_VICTIM, VICTIM);
  seedOrder(REF_ATTACKER, ATTACKER);
  seedRealPayment('tx_victim_1', REF_VICTIM);
});

afterEach(() => { vi.unstubAllGlobals(); });

// ═══════════════════════════════════════════════════════ 1 · FIRST use, forged ref
describe('CXD-059 · a FORGED reference is refused on its FIRST use', () => {
  it('does not credit the attacker even though signature, ledger and sealed quote all agree', async () => {
    // The attacker replays the victim's signed body under HIS OWN reference. Nothing
    // local can tell: the amount/currency/user/plan match his own sealed quote, and
    // the provider event id has never been seen before (no ledger conflict).
    const res = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_ATTACKER }));

    expect(res.status, 'a forged first use must be refused').toBeGreaterThanOrEqual(400);
    expect(ent(ATTACKER).plan, 'the attacker paid nothing').toBe('free');
    expect(pg.state.orders.get(REF_ATTACKER)!.status).toBe('pending');
    // ...and it must have asked the provider rather than trusted the body.
    expect(providerApi.calls.some((u) => u.includes('tx_victim_1')), 'must confirm server-to-server').toBe(true);
  });

  it('leaves a security incident naming both references', async () => {
    await post(signedEvent({ txId: 'tx_victim_1', reference: REF_ATTACKER }));
    const incidents = pg.state.audit.filter((r) => /replay|confirm/i.test(String(r.action)));
    expect(incidents.length, 'a refused forgery must leave a trace').toBeGreaterThan(0);
    const detail = JSON.parse(String(incidents[0].detail));
    expect(detail.claimed_reference).toBe(REF_ATTACKER);
    expect(detail.authoritative_reference).toBe(REF_VICTIM);
  });

  it('a transaction the provider has never heard of is refused', async () => {
    const res = await post(signedEvent({ txId: 'tx_invented', reference: REF_ATTACKER }));
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
    expect(ent(ATTACKER).plan).toBe('free');
  });

  it('an amount/currency the provider does not confirm is refused', async () => {
    // Consistent local story (order sealed at PLAN_PRICE), provider says otherwise.
    providerApi.record({
      id: 'tx_odd', status: 'APPROVED', reference: REF_ATTACKER,
      amount_in_cents: 1_000, currency: 'COP',
    });
    const res = await post(signedEvent({ txId: 'tx_odd', reference: REF_ATTACKER }));
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(ent(ATTACKER).plan).toBe('free');
  });
});

// ═══════════════════════════════════════════════════════════ 2 · the happy path
describe('CXD-059 · an authoritatively confirmed payment credits exactly once', () => {
  it('credits the real payer', async () => {
    const res = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM }));
    expect(res.status).toBe(200);
    expect(ent(VICTIM).plan).toBe('signals');
    expect(pg.state.orders.get(REF_VICTIM)!.status).toBe('paid');
  });

  it('a provider retry of the confirmed event is a duplicate no-op — credited ONCE', async () => {
    const evt = signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM });
    expect((await post(evt)).status).toBe(200);
    const after = JSON.stringify(ent(VICTIM));
    const plans = () => pg.state.audit.filter((r) => r.action === 'plan_change').length;
    const credits = plans();

    const retry = await post(evt);
    expect(retry.status).toBe(200);
    expect(await retry.json()).toMatchObject({ duplicate: true });
    expect(JSON.stringify(ent(VICTIM))).toBe(after);
    expect(plans(), 'exactly one credit').toBe(credits);
  });
});

// ═════════════════════════════════════════════════════ 3 · provider outage ⇒ 503
describe('CXD-059 · a provider outage postpones the credit, it never grants it', () => {
  it('network failure: nothing is mutated and the answer is retryable', async () => {
    providerApi.setMode('down');
    const res = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM }));

    expect(res.status, 'retryable, not a rejection').toBe(503);
    expect(ent(VICTIM).plan).toBe('free');
    expect(pg.state.orders.get(REF_VICTIM)!.status).toBe('pending');
    // Nothing was written to the ledger either — otherwise the retry would be
    // swallowed as a "duplicate" and the paid user would NEVER be credited.
    expect(pg.state.billingEvents.size).toBe(0);
    expect(pg.state.webhookEvents.size).toBe(0);
  });

  it('5xx from the provider behaves the same way', async () => {
    providerApi.setMode('http5xx');
    const res = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM }));
    expect(res.status).toBe(503);
    expect(ent(VICTIM).plan).toBe('free');
    expect(pg.state.billingEvents.size).toBe(0);
  });

  /**
   * A misconfigured key or a rate limit must be RETRYABLE, not a rejection: only an
   * explicit 404 is the provider denying the transaction. Classifying 401/429 as
   * "forged" would permanently refuse payments that really happened — fail-closed
   * must not become fail-destructive.
   */
  it.each([401, 403, 429, 502])('http %i is "cannot tell", never a verdict', async (status) => {
    providerApi.setMode('up');
    vi.stubGlobal('fetch', async () => new Response('nope', { status }));
    const res = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM }));
    expect(res.status).toBe(503);
    expect(ent(VICTIM).plan).toBe('free');
    expect(pg.state.billingEvents.size, 'the retry must not be swallowed as a duplicate').toBe(0);
  });

  it('an unparseable provider answer is NOT a confirmation', async () => {
    providerApi.setMode('garbage');
    const res = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM }));
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(ent(VICTIM).plan).toBe('free');
  });

  it('THE RETRY AFTER THE OUTAGE CREDITS — no legitimate payment is lost', async () => {
    providerApi.setMode('down');
    expect((await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM }))).status).toBe(503);
    expect(ent(VICTIM).plan).toBe('free');

    providerApi.setMode('up');
    const retry = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM }));
    expect(retry.status).toBe(200);
    expect(ent(VICTIM).plan, 'the postponed payment is credited on retry').toBe('signals');
    expect(pg.state.orders.get(REF_VICTIM)!.status).toBe('paid');
  });
});

// ══════════════════════════════════════════════════ 4 · global replay still no-op
describe('CXD-059 · the CXD-056 ledger defence is intact', () => {
  it('the same provider event can never credit a second reference', async () => {
    // The attacker gets his OWN payment confirmed first...
    seedRealPayment('tx_attacker_1', REF_ATTACKER);
    expect((await post(signedEvent({ txId: 'tx_attacker_1', reference: REF_ATTACKER }))).status).toBe(200);

    // ...then replays that same event onto the victim's still-pending order. Now the
    // ledger fires (second use), and it must still refuse.
    const before = JSON.stringify(pg.state.orders.get(REF_VICTIM));
    const replay = await post(signedEvent({ txId: 'tx_attacker_1', reference: REF_VICTIM }));

    expect(replay.status).toBeGreaterThanOrEqual(400);
    expect(replay.status).toBeLessThan(500);
    expect(JSON.stringify(pg.state.orders.get(REF_VICTIM))).toBe(before);
    expect(ent(VICTIM).plan).toBe('free');
    expect(pg.state.billingEvents.get('wompi:tx_attacker_1:APPROVED')!.order_reference).toBe(REF_ATTACKER);
  });

  it('a byte-identical retry is a no-op, not an incident', async () => {
    const evt = signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM });
    await post(evt);
    const auditCount = pg.state.audit.length;
    const retry = await post(evt);
    expect(await retry.json()).toMatchObject({ duplicate: true });
    expect(pg.state.audit.length, 'a retry is not an incident').toBe(auditCount);
  });
});

// ══════════════════════════════════════════════════════════ 5 · no leakage at all
describe('CXD-059 · the confirmation never leaks a secret or an internal path', () => {
  const forbidden = [
    'hunter2', 'pub_prod_', 'Bearer', 'WOMPI_', 'wompi.ts', 'postgres-client.ts',
    'C:\\srv', 'ETIMEDOUT', 'checkout.wompi.co', 'production.wompi.co', 'sandbox.wompi.co',
  ];

  const assertClean = async (res: Response) => {
    const text = await res.text();
    for (const needle of forbidden) {
      expect(text, `response body must not contain ${needle}`).not.toContain(needle);
    }
    expect(text).not.toMatch(/\bstack\b/i);
  };

  it('outage response is generic', async () => {
    providerApi.setMode('down');
    await assertClean(await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM })));
  });

  it('5xx response is generic', async () => {
    providerApi.setMode('http5xx');
    await assertClean(await post(signedEvent({ txId: 'tx_victim_1', reference: REF_VICTIM })));
  });

  it('mismatch response is generic (it must not tell the attacker what it saw)', async () => {
    const res = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_ATTACKER }));
    await assertClean(res);
  });

  it('the victim reference is NEVER echoed back to the attacker', async () => {
    const res = await post(signedEvent({ txId: 'tx_victim_1', reference: REF_ATTACKER }));
    const text = await res.text();
    expect(text, 'the forged answer must not disclose whose payment it was').not.toContain(VICTIM);
    expect(text).not.toContain(REF_VICTIM);
  });
});
