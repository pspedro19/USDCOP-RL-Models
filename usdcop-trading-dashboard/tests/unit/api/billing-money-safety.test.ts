/**
 * P0 money-safety tests for billing (CODEX review findings P0-1..P0-4).
 *
 * Rules under test (`.claude/rules/rbac.md` §7 — the provider is the truth, the client
 * never; fail-closed):
 *  P0-1  no fabricated transitions: PENDING/unknown provider events must NEVER be
 *        normalized into `subscription.cancelled`.
 *  P0-2  no checkout URL signed with an empty integrity secret.
 *  P0-3  the webhook credits the SEALED quote (checkout_orders), not the current price;
 *        idempotency + entitlement + order + audit commit atomically; a cancellation
 *        never mutates an already-paid order.
 *  P0-4  internal errors never leak paths/stacks to the client.
 *
 * The DB is a scripted fake that honours the two invariants migration 058/059 encode:
 * UNIQUE(reference,event_type) on billing_webhook_events and the legal-transition
 * trigger on checkout_orders (paid -> cancelled/failed raises).
 */
import { createHash } from 'node:crypto';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// ── scripted postgres (shared by `query` and the transactional `getClient`) ─────
const pg = vi.hoisted(() => {
  type Row = Record<string, unknown>;
  const state = {
    orders: new Map<string, Row>(),
    users: new Map<string, Row>(),
    webhookEvents: new Set<string>(),
    billingEvents: new Set<string>(),
    audit: [] as Row[],
    cart: [] as string[],
    log: [] as string[],
    failOn: null as null | ((sql: string) => boolean),
  };

  const LEGAL: Record<string, string[]> = {
    created: ['pending', 'paid'],
    pending: ['paid', 'failed', 'cancelled', 'expired'],
    paid: ['refunded', 'charged_back'],
  };

  const norm = (s: string) => s.replace(/\s+/g, ' ').trim();

  function statusFilter(sql: string, params: unknown[]): string[] | null {
    const anyM = sql.match(/status\s*=\s*ANY\(\$(\d+)(?:::text\[\])?\)/i);
    if (anyM) return params[Number(anyM[1]) - 1] as string[];
    const inM = sql.match(/status\s+IN\s*\(([^)]*)\)/i);
    if (inM) return inM[1].split(',').map((s) => s.trim().replace(/'/g, ''));
    return null;
  }

  function exec(text: string, params: unknown[] = [], stage: (() => void)[] | null) {
    const sql = norm(text);
    state.log.push(sql);
    if (state.failOn?.(sql)) {
      const e = new Error(
        'error: relation blew up at C:\\srv\\usdcop\\lib\\db\\postgres-client.ts:117 (password=hunter2)',
      );
      throw e;
    }
    const apply = (fn: () => void) => (stage ? stage.push(fn) : fn());
    const res = (rows: Row[] = [], rowCount = rows.length) => ({ rows, rowCount });

    // ── idempotency ledger (migration 058: UNIQUE(reference,event_type))
    if (/INSERT INTO billing_webhook_events/i.test(sql)) {
      const key = `${params[0]}:${params[1]}`;
      if (state.webhookEvents.has(key)) {
        const e = new Error('duplicate key value violates unique constraint') as Error & { code?: string };
        e.code = '23505';
        throw e;
      }
      apply(() => state.webhookEvents.add(key));
      return res([], 1);
    }
    // ── append-only provider event ledger
    if (/INSERT INTO billing_events/i.test(sql)) {
      const id = String(params[0]);
      if (state.billingEvents.has(id)) return res([], 0);
      apply(() => state.billingEvents.add(id));
      return res([], 1);
    }
    if (/INSERT INTO audit_log/i.test(sql)) {
      apply(() => state.audit.push({ user_id: params[0], action: params[1], detail: params[3] }));
      return res([], 1);
    }
    // ── sealed quote
    if (/FROM checkout_orders/i.test(sql)) {
      const o = state.orders.get(String(params[0]));
      return res(o ? [{ ...o }] : []);
    }
    if (/INSERT INTO checkout_orders/i.test(sql)) {
      const reference = String(params[4]);
      if (!state.orders.has(reference)) {
        apply(() => state.orders.set(reference, {
          user_id: params[0], plan: params[1],
          addon_assets: JSON.parse(String(params[2])),
          amount_cents: Number(params[3]), currency: 'COP',
          reference, status: 'pending',
        }));
      }
      return res([], 1);
    }
    if (/UPDATE checkout_orders/i.test(sql)) {
      const setM = sql.match(/SET status\s*=\s*(?:'([a-z_]+)'|\$(\d+))/i);
      const refM = sql.match(/reference\s*=\s*\$(\d+)/i);
      if (!setM || !refM) throw new Error(`unparseable UPDATE in test harness: ${sql}`);
      const next = setM[1] ?? String(params[Number(setM[2]) - 1]);
      const reference = String(params[Number(refM[1]) - 1]);
      const order = state.orders.get(reference);
      if (!order) return res([], 0);
      const allowed = statusFilter(sql, params);
      if (allowed && !allowed.includes(String(order.status))) return res([], 0);
      const from = String(order.status);
      if (from !== next && !(LEGAL[from] ?? []).includes(next)) {
        throw new Error(`illegal checkout order transition: ${from} -> ${next}`);
      }
      apply(() => { order.status = next; });
      return res([], 1);
    }
    // ── users
    if (/UPDATE sb_users SET entitlements/i.test(sql)) {
      const jsonbSet = /jsonb_set/i.test(sql);
      const userId = String(params[jsonbSet ? 0 : 1]);
      const u = state.users.get(userId);
      if (!u) return res([], 0);
      apply(() => {
        if (jsonbSet) u.entitlements = { ...(u.entitlements as Row), assets: [] };
        else u.entitlements = JSON.parse(String(params[0]));
      });
      return res([], 1);
    }
    if (/SELECT role, entitlements FROM sb_users/i.test(sql)) {
      const u = state.users.get(String(params[0]));
      return res(u ? [{ role: u.role, entitlements: u.entitlements }] : []);
    }
    if (/SELECT email FROM sb_users/i.test(sql)) {
      const u = state.users.get(String(params[0]));
      return res(u ? [{ email: u.email }] : []);
    }
    if (/FROM user_cart/i.test(sql)) return res(state.cart.map((a) => ({ asset_id: a })));

    return res([]);
  }

  return {
    state,
    reset() {
      state.orders.clear(); state.users.clear(); state.webhookEvents.clear();
      state.billingEvents.clear(); state.audit.length = 0; state.cart.length = 0;
      state.log.length = 0; state.failOn = null;
    },
    query: (text: string, params?: unknown[]) => Promise.resolve(exec(text, params, null)),
    getClient: () => {
      let pending: (() => void)[] | null = null;
      return Promise.resolve({
        query: (text: string, params?: unknown[]) => {
          const sql = text.replace(/\s+/g, ' ').trim().toUpperCase();
          if (sql === 'BEGIN') { pending = []; state.log.push('BEGIN'); return Promise.resolve({ rows: [], rowCount: 0 }); }
          if (sql === 'COMMIT') {
            state.log.push('COMMIT');
            (pending ?? []).forEach((fn) => fn()); pending = null;
            return Promise.resolve({ rows: [], rowCount: 0 });
          }
          if (sql === 'ROLLBACK') { state.log.push('ROLLBACK'); pending = null; return Promise.resolve({ rows: [], rowCount: 0 }); }
          return Promise.resolve(exec(text, params, pending));
        },
        release: () => {},
      });
    },
  };
});

vi.mock('@/lib/db/postgres-client', () => ({
  query: pg.query,
  pgQuery: pg.query,
  getClient: pg.getClient,
  getPool: () => { throw new Error('pool disabled in tests'); },
}));

import { POST as webhookPOST } from '@/app/api/billing/webhook/route';
import { POST as cartCheckoutPOST } from '@/app/api/cart/checkout/route';
import { POST as billingCheckoutPOST } from '@/app/api/billing/checkout/route';
import { WompiProvider } from '@/lib/billing/wompi';

const EVENTS_SECRET = 'events-secret';
const USER = '11111111-1111-4111-8111-111111111111';
const REF = `sub_signals_${USER}_base_1700000000000`;

function signedEvent(opts: {
  status?: string;
  event?: string;
  reference?: string;
  amountInCents?: number;
  currency?: string;
  txId?: string;
}) {
  const tx = {
    id: opts.txId ?? 'tx_1',
    status: opts.status ?? 'APPROVED',
    reference: opts.reference ?? REF,
    amount_in_cents: opts.amountInCents ?? 9_900_000,
    currency: opts.currency ?? 'COP',
  };
  const timestamp = 1_700_000_000;
  const properties = ['transaction.id', 'transaction.status', 'transaction.amount_in_cents'];
  const concatenated = [tx.id, tx.status, tx.amount_in_cents].join('');
  const checksum = createHash('sha256')
    .update(`${concatenated}${timestamp}${EVENTS_SECRET}`)
    .digest('hex');
  return JSON.stringify({
    event: opts.event ?? 'transaction.updated',
    data: { transaction: tx },
    signature: { properties, checksum },
    timestamp,
  });
}

const post = (body: string) =>
  webhookPOST(new Request('http://t/api/billing/webhook', { method: 'POST', body }));

function seedPaidPath(overrides: Record<string, unknown> = {}) {
  pg.state.users.set(USER, {
    email: 'u@example.com', role: 'free',
    entitlements: { plan: 'free', assets: ['usdcop'], expires_at: null },
  });
  pg.state.orders.set(REF, {
    user_id: USER, plan: 'signals', addon_assets: [],
    amount_cents: 9_900_000, currency: 'COP', reference: REF, status: 'pending',
    ...overrides,
  });
}

beforeEach(() => {
  pg.reset();
  vi.restoreAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  process.env.WOMPI_EVENTS_SECRET = EVENTS_SECRET;
  process.env.WOMPI_PUBLIC_KEY = 'pub_test_key';
  process.env.WOMPI_INTEGRITY_SECRET = 'integrity-secret';
  process.env.NEXTAUTH_URL = 'http://t';
  delete process.env.BILLING_PRICES_COP;
  delete process.env.BILLING_ADDON_PRICES_COP;
});

// ───────────────────────────────────────────────────────── P0-3 (a) sealed quote
describe('P0-3(a) the webhook credits the SEALED quote, not the current price', () => {
  it('credits the amount the user accepted after a price change', async () => {
    seedPaidPath();
    // Price list changes AFTER the checkout was sealed (9.900.000 cents).
    process.env.BILLING_PRICES_COP = JSON.stringify({ signals: 19_900_000 });

    const res = await post(signedEvent({ amountInCents: 9_900_000 }));
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body).toMatchObject({ received: true });
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('signals');
    expect(pg.state.orders.get(REF)!.status).toBe('paid');
  });

  it('rejects when the provider amount does not match the sealed quote', async () => {
    seedPaidPath();
    const res = await post(signedEvent({ amountInCents: 1_000 }));
    expect(res.status).toBe(400);
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('free');
    expect(pg.state.orders.get(REF)!.status).toBe('pending');
  });

  it('fails closed when there is no sealed order for the reference', async () => {
    pg.state.users.set(USER, { email: 'u@example.com', role: 'free', entitlements: { plan: 'free', assets: [] } });
    const res = await post(signedEvent({}));
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('free');
  });

  it('rejects when the sealed quote disagrees with the reference tuple (plan)', async () => {
    seedPaidPath({ plan: 'auto' }); // reference says `signals`
    const res = await post(signedEvent({}));
    expect(res.status).toBe(400);
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('free');
  });

  it('credits the add-ons sealed in the order', async () => {
    pg.state.users.set(USER, {
      email: 'u@example.com', role: 'free',
      entitlements: { plan: 'free', assets: ['usdcop'], expires_at: null },
    });
    const ref = `sub_signals_${USER}_xauusd_1700000000000`;
    pg.state.orders.set(ref, {
      user_id: USER, plan: 'signals', addon_assets: ['xauusd'],
      amount_cents: 9_900_000 + 3_900_000, currency: 'COP', reference: ref, status: 'pending',
    });
    const res = await post(signedEvent({ reference: ref, amountInCents: 9_900_000 + 3_900_000 }));
    expect(res.status).toBe(200);
    expect((pg.state.users.get(USER)!.entitlements as { assets: string[] }).assets).toContain('xauusd');
  });
});

// ─────────────────────────────────────────────────── P0-3 (b) one transaction
describe('P0-3(b) atomicity — a mid-flight failure leaves NO split state', () => {
  it('rolls back the entitlement when a later write fails', async () => {
    seedPaidPath();
    pg.state.failOn = (sql) => /INSERT INTO audit_log/i.test(sql);

    const res = await post(signedEvent({})).catch((e) => e as Error);
    expect(res).toBeInstanceOf(Response);
    expect((res as Response).status).toBeGreaterThanOrEqual(500);

    // nothing partially applied
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('free');
    expect(pg.state.orders.get(REF)!.status).toBe('pending');
    expect(pg.state.webhookEvents.size).toBe(0);
  });

  it('rolls back when the order transition fails after the entitlement write', async () => {
    seedPaidPath();
    pg.state.failOn = (sql) => /UPDATE checkout_orders/i.test(sql);

    const res = await post(signedEvent({})).catch((e) => e as Error);
    expect(res).toBeInstanceOf(Response);
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('free');
  });
});

// ─────────────────────────────────────────────────────── P0-3 idempotency
describe('idempotency — retries never double-credit, failures never poison the ledger', () => {
  it('a retried event is a no-op (no second entitlement extension)', async () => {
    seedPaidPath();
    const evt = signedEvent({});
    const first = await post(evt);
    expect(first.status).toBe(200);
    const after1 = { ...(pg.state.users.get(USER)!.entitlements as Record<string, unknown>) };
    const auditCount = pg.state.audit.length;

    const second = await post(evt);
    expect(second.status).toBe(200);
    expect(await second.json()).toMatchObject({ duplicate: true });
    expect(pg.state.users.get(USER)!.entitlements).toEqual(after1);
    expect(pg.state.audit.length).toBe(auditCount);
  });

  it('after a failed attempt the retry still credits the user', async () => {
    seedPaidPath();
    pg.state.failOn = (sql) => /INSERT INTO audit_log/i.test(sql);
    await post(signedEvent({})).catch(() => undefined);

    pg.state.failOn = null;
    const res = await post(signedEvent({}));
    expect(res.status).toBe(200);
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('signals');
    expect(pg.state.orders.get(REF)!.status).toBe('paid');
  });
});

// ───────────────────────────────────── P0-3(c) cancellations never touch paid
describe('P0-3(c) a cancellation never mutates an already-paid order', () => {
  it('leaves a paid order paid and the entitlement intact', async () => {
    seedPaidPath({ status: 'paid' });
    pg.state.users.set(USER, {
      email: 'u@example.com', role: 'free',
      entitlements: { plan: 'signals', assets: ['usdcop'], expires_at: '2999-01-01T00:00:00.000Z' },
    });

    const raw = JSON.stringify({ type: 'subscription.cancelled', reference: REF });
    const provider = { verifyWebhook: async () => ({
      valid: true,
      event: { type: 'subscription.cancelled' as const, reference: REF, providerEventId: 'evt_cancel_1', raw: {} },
    }), name: 'wompi', createCheckout: async () => { throw new Error('n/a'); } };

    const billing = await import('@/lib/billing');
    const spy = vi.spyOn(billing, 'getBillingProvider').mockReturnValue(provider as never);
    try {
      const res = await post(raw).catch((e) => e as Error);
      expect(res).toBeInstanceOf(Response);
      expect((res as Response).status).toBeLessThan(500);
    } finally {
      spy.mockRestore();
    }

    expect(pg.state.orders.get(REF)!.status).toBe('paid');
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('signals');
  });
});

// ─────────────────────────────────────── P0-1 no fabricated cancellations
describe('P0-1 exhaustive status mapping — no fabricated transitions', () => {
  const provider = new WompiProvider();
  const verify = (body: string) => provider.verifyWebhook(body, new Headers());

  it('PENDING is ignored, never a cancellation', async () => {
    const v = await verify(signedEvent({ status: 'PENDING' }));
    expect(v.valid).toBe(true);
    expect(v.event?.type).not.toBe('subscription.cancelled');
    expect(v.event).toBeUndefined();
  });

  it('an unknown transaction status is ignored, never a cancellation', async () => {
    const v = await verify(signedEvent({ status: 'SOME_FUTURE_STATUS' }));
    expect(v.event?.type).not.toBe('subscription.cancelled');
    expect(v.event).toBeUndefined();
  });

  it('an unknown event name is ignored, never a cancellation', async () => {
    const v = await verify(signedEvent({ event: 'nequi_token.updated', status: 'APPROVED' }));
    expect(v.event?.type).not.toBe('subscription.cancelled');
    expect(v.event).toBeUndefined();
  });

  it('maps the known statuses explicitly', async () => {
    expect((await verify(signedEvent({ status: 'APPROVED' }))).event?.type).toBe('payment.approved');
    for (const s of ['DECLINED', 'ERROR', 'VOIDED']) {
      expect((await verify(signedEvent({ status: s }))).event?.type).toBe('payment.declined');
    }
  });

  it('normalizes currency and a provider event id', async () => {
    const v = await verify(signedEvent({ status: 'APPROVED', txId: 'tx_42' }));
    expect(v.event?.currency).toBe('COP');
    expect(v.event?.providerEventId).toContain('tx_42');
  });

  it('a PENDING event through the route mutates nothing', async () => {
    seedPaidPath();
    const res = await post(signedEvent({ status: 'PENDING' }));
    expect(res.status).toBeLessThan(500);
    expect(pg.state.orders.get(REF)!.status).toBe('pending');
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('free');
  });
});

// ───────────────────────────────────────────────── P0-2 fail-closed checkout
describe('P0-2 checkout fails closed on incomplete configuration', () => {
  it('never produces a URL when WOMPI_INTEGRITY_SECRET is missing', async () => {
    delete process.env.WOMPI_INTEGRITY_SECRET;
    await expect(new WompiProvider().createCheckout({
      userId: USER, email: 'u@example.com', plan: 'signals',
    })).rejects.toThrow(/WOMPI_INTEGRITY_SECRET/);
  });

  it('never produces a URL when the integrity secret is blank', async () => {
    process.env.WOMPI_INTEGRITY_SECRET = '   ';
    await expect(new WompiProvider().createCheckout({
      userId: USER, email: 'u@example.com', plan: 'signals',
    })).rejects.toThrow(/WOMPI_INTEGRITY_SECRET/);
  });

  it('never produces a zero-amount checkout for a paid plan', async () => {
    process.env.BILLING_PRICES_COP = JSON.stringify({ signals: 0 });
    await expect(new WompiProvider().createCheckout({
      userId: USER, email: 'u@example.com', plan: 'signals',
    })).rejects.toThrow();
  });

  it('never produces a checkout for an add-on with no published price', async () => {
    await expect(new WompiProvider().createCheckout({
      userId: USER, email: 'u@example.com', plan: 'signals', addOnAssets: ['not_a_real_asset'],
    })).rejects.toThrow();
  });
});

// ──────────────────────────── every checkout entry point seals a quote (E2E)
describe('every checkout entry point seals a quote the webhook can credit', () => {
  const seedUser = () => pg.state.users.set(USER, {
    email: 'u@example.com', role: 'free',
    entitlements: { plan: 'free', assets: ['usdcop'], expires_at: null },
  });

  const approve = async (reference: string, amountInCents: number) =>
    post(signedEvent({ reference, amountInCents, txId: `tx_${reference.slice(-6)}` }));

  it('POST /api/cart/checkout → webhook credits the user', async () => {
    seedUser();
    pg.state.cart.push('xauusd');
    const res = await cartCheckoutPOST(new Request('http://t/api/cart/checkout', {
      method: 'POST',
      headers: { 'x-user-role': 'free', 'x-user-id': USER, 'content-type': 'application/json' },
      body: JSON.stringify({ plan: 'signals' }),
    }));
    const { data } = await res.json();
    const order = pg.state.orders.get(data.reference)!;
    expect(order).toBeTruthy();

    const wh = await approve(data.reference, Number(order.amount_cents));
    expect(wh.status).toBe(200);
    expect((pg.state.users.get(USER)!.entitlements as { plan: string; assets: string[] }).plan).toBe('signals');
    expect((pg.state.users.get(USER)!.entitlements as { assets: string[] }).assets).toContain('xauusd');
  });

  it('POST /api/billing/checkout (Pricing page) → webhook credits the user', async () => {
    seedUser();
    const res = await billingCheckoutPOST(new Request('http://t/api/billing/checkout', {
      method: 'POST',
      headers: { 'x-user-id': USER, 'content-type': 'application/json' },
      body: JSON.stringify({ plan: 'auto' }),
    }));
    expect(res.status).toBe(200);
    const session = await res.json();
    const order = pg.state.orders.get(session.reference);
    expect(order, 'the pricing-page checkout MUST seal a quote').toBeTruthy();

    const wh = await approve(session.reference, Number(order!.amount_cents));
    expect(wh.status).toBe(200);
    expect((pg.state.users.get(USER)!.entitlements as { plan: string }).plan).toBe('auto');
  });

  it('POST /api/billing/checkout does not leak internal detail', async () => {
    seedUser();
    pg.state.failOn = (sql) => /INSERT INTO checkout_orders/i.test(sql);
    const res = await billingCheckoutPOST(new Request('http://t/api/billing/checkout', {
      method: 'POST',
      headers: { 'x-user-id': USER, 'content-type': 'application/json' },
      body: JSON.stringify({ plan: 'signals' }),
    }));
    const text = await res.text();
    expect(res.ok).toBe(false);
    expect(text).not.toContain('postgres-client.ts');
    expect(text).not.toContain('hunter2');
    expect(text).not.toMatch(/\bdetail\b/);
  });
});

// ─────────────────────────────────────────────────────── P0-3 currency check
describe('currency mismatch is rejected', () => {
  it('does not credit when the provider currency differs from the sealed quote', async () => {
    seedPaidPath();
    const res = await post(signedEvent({ currency: 'USD' }));
    expect(res.status).toBe(400);
    expect((pg.state.users.get(USER)!.entitlements as Record<string, unknown>).plan).toBe('free');
    expect(pg.state.orders.get(REF)!.status).toBe('pending');
  });
});

// ───────────────────────────────────────────────────── P0-4 no error leakage
describe('P0-4 internal errors never leak paths or stacks', () => {
  it('cart checkout returns a generic error body', async () => {
    pg.state.users.set(USER, {
      email: 'u@example.com', role: 'free',
      entitlements: { plan: 'free', assets: ['usdcop'], expires_at: null },
    });
    pg.state.failOn = (sql) => /INSERT INTO checkout_orders/i.test(sql);

    const res = await cartCheckoutPOST(new Request('http://t/api/cart/checkout', {
      method: 'POST',
      headers: { 'x-user-role': 'free', 'x-user-id': USER, 'content-type': 'application/json' },
      body: JSON.stringify({ plan: 'signals' }),
    }));
    const text = await res.text();

    expect(res.ok).toBe(false);
    expect(text).not.toContain('postgres-client.ts');
    expect(text).not.toContain('C:\\srv');
    expect(text).not.toContain('hunter2');
    expect(text).not.toMatch(/\bdetail\b/);
  });

  it('the webhook returns a generic body when the DB blows up', async () => {
    seedPaidPath();
    pg.state.failOn = (sql) => /UPDATE sb_users/i.test(sql);
    const res = await post(signedEvent({})).catch((e) => e as Error);
    expect(res).toBeInstanceOf(Response);
    const text = await (res as Response).text();
    expect(text).not.toContain('postgres-client.ts');
    expect(text).not.toContain('hunter2');
  });
});
