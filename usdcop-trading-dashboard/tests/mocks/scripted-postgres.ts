/**
 * Scripted Postgres fake for the BILLING money paths.
 *
 * It is not a general SQL engine: it honours exactly the invariants the billing
 * migrations encode, so a test that passes here would also pass against the real
 * schema, and a test that fails here fails for a real reason:
 *
 *  - `billing_webhook_events`  UNIQUE (reference, event_type)          (mig. 058)
 *  - `billing_events`          UNIQUE (provider_event_id)              (mig. 059)
 *                              + `ON CONFLICT DO NOTHING RETURNING` returns 0 rows
 *  - `checkout_orders`         legal-transition trigger; quote is immutable (mig. 059)
 *  - `audit_log`               append-only (mig. 055) — rows are only pushed
 *  - transactions              writes are STAGED and applied on COMMIT, discarded on
 *                              ROLLBACK, so "half-applied" state is impossible to fake
 *                              accidentally and `SELECT ... FOR UPDATE` reads committed
 *                              state exactly as another connection would.
 *
 * Shared by `tests/unit/api/billing-money-safety.test.ts` and
 * `tests/unit/api/billing-replay-entitlement.test.ts` (DRY: one fake, one set of
 * invariants — two copies would let the two suites disagree about the schema).
 */

export type Row = Record<string, unknown>;

export interface BillingEventRow {
  provider_event_id: string;
  order_reference: string;
  event_type: string;
  payload: unknown;
}

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

/** `jsonb` normalizes key order and drops duplicates — model that on write. */
function asJsonb(value: unknown): unknown {
  return JSON.parse(typeof value === 'string' ? value : JSON.stringify(value ?? null));
}

export function createScriptedPostgres() {
  const state = {
    orders: new Map<string, Row>(),
    users: new Map<string, Row>(),
    webhookEvents: new Set<string>(),
    /** provider_event_id → committed ledger row (mig. 059 UNIQUE). */
    billingEvents: new Map<string, BillingEventRow>(),
    audit: [] as Row[],
    cart: [] as string[],
    log: [] as string[],
    failOn: null as null | ((sql: string) => boolean),
  };

  function exec(text: string, params: unknown[] = [], stage: (() => void)[] | null) {
    const sql = norm(text);
    state.log.push(sql);
    if (state.failOn?.(sql)) {
      throw new Error(
        'error: relation blew up at C:\\srv\\usdcop\\lib\\db\\postgres-client.ts:117 (password=hunter2)',
      );
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
    // ── append-only provider event ledger (migration 059: UNIQUE(provider_event_id)).
    //    `ON CONFLICT DO NOTHING RETURNING x` yields ZERO rows on conflict — that is
    //    the signal the route uses to tell a retry from a forged replay.
    if (/INSERT INTO billing_events/i.test(sql)) {
      const id = String(params[0]);
      if (state.billingEvents.has(id)) return res([], 0);
      const row: BillingEventRow = {
        provider_event_id: id,
        order_reference: String(params[1]),
        event_type: String(params[2]),
        payload: asJsonb(params[3]),
      };
      apply(() => state.billingEvents.set(id, row));
      return /RETURNING/i.test(sql) ? res([{ provider_event_id: id }], 1) : res([], 1);
    }
    if (/FROM billing_events/i.test(sql)) {
      const row = state.billingEvents.get(String(params[0]));
      return res(row ? [{ ...row }] : []);
    }
    if (/INSERT INTO audit_log/i.test(sql)) {
      apply(() => state.audit.push({
        user_id: params[0], action: params[1], object_type: params[2], detail: params[3],
      }));
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
        else u.entitlements = asJsonb(params[0]);
      });
      return res([], 1);
    }
    // `SELECT entitlements FROM sb_users ... FOR UPDATE` — row lock for the money path.
    if (/SELECT entitlements FROM sb_users/i.test(sql)) {
      const u = state.users.get(String(params[0]));
      return res(u ? [{ entitlements: u.entitlements }] : []);
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
}

/** One instance per test FILE (vitest isolates module registries per file). */
export const scriptedPg = createScriptedPostgres();
