/**
 * P1 AVAILABILITY — CODEX finding **CXD-062**.
 *
 * ── THE FINDING ─────────────────────────────────────────────────────────────────
 * `route.ts` PROMISED, in a comment, that the server-to-server confirmation happens
 * "BEFORE `BEGIN` and before a single row moves"… while the code called `getClient()`
 * FIRST and only then awaited the provider. The money invariants were intact; the
 * RESOURCE order was not. A guarantee written in a comment and not enforced by the
 * code is exactly the failure mode this repo keeps finding — the comment was not a
 * cosmetic detail, it was a false promise.
 *
 * ── THE CONSEQUENCE, MEASURED HERE ──────────────────────────────────────────────
 * The confirmation is a NETWORK call with a multi-second budget. Holding a pooled
 * connection across it means a provider that hangs converts webhook traffic into
 * pool exhaustion — and the pool is shared, so the pages that die are `/api/market`,
 * `/api/analysis`, auth… APIs with nothing to do with billing.
 *
 * The pool is modelled in `tests/mocks/scripted-postgres.ts` the way `pg.Pool`
 * behaves: `size` connections exist, `getClient()` leases one and BLOCKS when they
 * are all out. So "the pool is exhausted" is not asserted through a proxy metric —
 * an unrelated caller simply does not get a connection.
 *
 * Mutations that MUST turn these red again:
 *  A1  acquire the client before confirming            → tests 1 and 2 (leases held)
 *  A2  keep the mismatch incident on the held client   → test 3 (leak of a lease)
 *  A3  take a connection during a provider outage      → test 4 (peak > 0)
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
const PAYER_A = '11111111-1111-4111-8111-111111111111';
const PAYER_B = '22222222-2222-4222-8222-222222222222';
const OUTSIDER = '33333333-3333-4333-8333-333333333333';
const PLAN_PRICE = 9_900_000;

const providerApi = createWompiTransactionsApi();

/**
 * A GATE in front of the provider's transport: every confirmation blocks here until
 * the test opens it. This is the hung provider — the exact condition under which the
 * old ordering starves the pool.
 */
function createHungTransport(inner: (input: unknown) => Promise<Response>) {
  let open!: () => void;
  const opened = new Promise<void>((resolve) => { open = resolve; });
  let inFlight = 0;
  let hang = true;
  return {
    get inFlight() { return inFlight; },
    /** Let every blocked confirmation through. */
    release() { hang = false; open(); },
    fetch: async (input: unknown): Promise<Response> => {
      inFlight += 1;
      if (hang) await opened;
      inFlight -= 1;
      return inner(input);
    },
  };
}

/**
 * Genuinely signed Wompi body (checksum over id|status|amount + timestamp + secret).
 * Same shape as the CXD-056/059 suites — kept local on purpose: each suite owns the
 * bytes it posts, so a change to one attack model cannot silently rewrite another.
 */
function signedEvent(opts: { txId: string; reference: string; status?: string }) {
  const tx = {
    id: opts.txId,
    status: opts.status ?? 'APPROVED',
    reference: opts.reference,
    amount_in_cents: PLAN_PRICE,
    currency: 'COP',
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
const REF_A = refFor(PAYER_A, 1_700_000_000_001);
const REF_B = refFor(PAYER_B, 1_700_000_000_002);

const freeEnt = { plan: 'free', assets: ['usdcop'], expires_at: null };
const ent = (u: string) =>
  pg.state.users.get(u)!.entitlements as { plan: string; assets: string[] };

function seedUser(id: string) {
  pg.state.users.set(id, { email: `${id}@example.com`, role: 'free', entitlements: { ...freeEnt } });
}
function seedOrder(reference: string, user: string) {
  pg.state.orders.set(reference, {
    user_id: user, plan: 'signals', addon_assets: [],
    amount_cents: PLAN_PRICE, currency: 'COP', reference, status: 'pending',
  });
}

/** Let every already-resolved continuation run, without waiting on the hung gate. */
const settle = async (rounds = 5) => {
  for (let i = 0; i < rounds; i += 1) await new Promise((r) => setTimeout(r, 0));
};

let transport: ReturnType<typeof createHungTransport>;

beforeEach(() => {
  pg.reset();
  providerApi.reset();
  vi.restoreAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  transport = createHungTransport(providerApi.fetch);
  vi.stubGlobal('fetch', transport.fetch);
  process.env.WOMPI_EVENTS_SECRET = EVENTS_SECRET;
  process.env.WOMPI_PUBLIC_KEY = 'pub_test_key';
  delete process.env.BILLING_PRICES_COP;
  delete process.env.BILLING_ADDON_PRICES_COP;

  seedUser(PAYER_A); seedUser(PAYER_B); seedUser(OUTSIDER);
  seedOrder(REF_A, PAYER_A); seedOrder(REF_B, PAYER_B);
  providerApi.record({
    id: 'tx_a', status: 'APPROVED', reference: REF_A, amount_in_cents: PLAN_PRICE, currency: 'COP',
  });
  providerApi.record({
    id: 'tx_b', status: 'APPROVED', reference: REF_B, amount_in_cents: PLAN_PRICE, currency: 'COP',
  });
});

afterEach(async () => {
  transport.release();
  vi.unstubAllGlobals();
});

// ════════════════════════════ 1 · no connection is held across the network wait
describe('CXD-062 · the provider is confirmed BEFORE any pooled connection is taken', () => {
  it('a pool of 1 is still 1/1 available while the confirmation is in flight', async () => {
    pg.state.pool.size = 1;

    let settled = false;
    const pending = post(signedEvent({ txId: 'tx_a', reference: REF_A })).then((r) => {
      settled = true; return r;
    });
    await settle();

    // The request really is parked on the provider, not finished early.
    expect(transport.inFlight, 'the confirmation must be in flight').toBe(1);
    expect(settled, 'the webhook must still be waiting on the provider').toBe(false);

    // THE ASSERTION: the wait costs zero database connections.
    expect(pg.state.pool.leased, 'no connection may be held across the provider wait').toBe(0);
    expect(pg.available(), 'the whole pool stays usable').toBe(1);
    expect(pg.state.pool.peak, 'not even briefly, before the wait').toBe(0);

    // …and the credit still happens once the provider answers.
    transport.release();
    const res = await pending;
    expect(res.status).toBe(200);
    expect(ent(PAYER_A).plan).toBe('signals');
    expect(pg.state.orders.get(REF_A)!.status).toBe('paid');
    expect(pg.state.pool.leased, 'the connection is returned').toBe(0);
    expect(pg.state.pool.peak, 'exactly one connection, only for the transaction').toBe(1);
  });

  it('N hung webhooks do NOT starve a pool of N: an unrelated API still gets a connection', async () => {
    pg.state.pool.size = 2;

    const a = post(signedEvent({ txId: 'tx_a', reference: REF_A }));
    const b = post(signedEvent({ txId: 'tx_b', reference: REF_B }));
    await settle();
    expect(transport.inFlight, 'both are parked on the provider').toBe(2);

    // An API with nothing to do with billing asks for a connection. This is the
    // collateral damage: with the old ordering it queues behind two network waits.
    let served = false;
    const outsider = pg.getClient().then((c) => { served = true; return c; });
    await settle();
    expect(served, 'a non-billing caller must not queue behind a provider timeout').toBe(true);
    (await outsider).release();

    transport.release();
    expect((await a).status).toBe(200);
    expect((await b).status).toBe(200);
    expect(ent(PAYER_A).plan).toBe('signals');
    expect(ent(PAYER_B).plan).toBe('signals');
  });
});

// ═══════════════════════════════ 2 · the security incident survives the reorder
describe('CXD-062 · a mismatch still leaves its audit incident', () => {
  it('records the forged/authoritative pair without ever holding a connection across the wait', async () => {
    pg.state.pool.size = 1;

    // PAYER_B replays PAYER_A's signed body under his own reference.
    let settled = false;
    const pending = post(signedEvent({ txId: 'tx_a', reference: REF_B })).then((r) => {
      settled = true; return r;
    });
    await settle();
    expect(settled).toBe(false);
    expect(pg.state.pool.leased, 'the mismatch path waits on the network too').toBe(0);

    transport.release();
    const res = await pending;

    expect(res.status, 'unchanged money semantics: 4xx, never retryable').toBe(409);
    expect(ent(PAYER_B).plan, 'nothing credited').toBe('free');
    expect(pg.state.orders.get(REF_B)!.status).toBe('pending');

    const incidents = pg.state.audit.filter((r) => /replay|confirm/i.test(String(r.action)));
    expect(incidents.length, 'the incident must NOT be lost to the refactor').toBe(1);
    const detail = JSON.parse(String(incidents[0].detail));
    expect(detail.claimed_reference).toBe(REF_B);
    expect(detail.authoritative_reference).toBe(REF_A);

    // The incident is a SHORT, SEPARATE operation — the connection it uses is given
    // back, so a burst of forgeries cannot pin the pool either.
    expect(pg.state.pool.leased, 'the incident releases its connection').toBe(0);
    expect(pg.available()).toBe(1);
  });
});

// ═════════════════════════════════ 3 · an outage costs no database resource at all
describe('CXD-062 · a provider outage never touches the pool', () => {
  it('503 without having leased a single connection (and nothing written)', async () => {
    pg.state.pool.size = 1;
    transport.release();            // no hang: fail fast instead
    providerApi.setMode('down');

    const res = await post(signedEvent({ txId: 'tx_a', reference: REF_A }));

    expect(res.status, 'retryable, exactly as before').toBe(503);
    expect(pg.state.pool.peak, 'an outage must not cost a connection').toBe(0);
    // The pre-existing money invariant, restated here so this suite cannot be
    // "fixed" by writing the ledger earlier: not even the ledger is touched.
    expect(pg.state.billingEvents.size).toBe(0);
    expect(pg.state.webhookEvents.size).toBe(0);
    expect(ent(PAYER_A).plan).toBe('free');
  });
});
