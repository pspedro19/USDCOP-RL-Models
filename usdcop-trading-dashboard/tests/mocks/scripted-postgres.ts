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
 *  - the CONNECTION POOL       `getClient()` leases one of `state.pool.size` slots and
 *                              BLOCKS when they are all out, exactly like `pg.Pool`
 *                              (CXD-062). Default size is unbounded, so suites that do
 *                              not care are unaffected; a suite that does sets
 *                              `state.pool.size = N` and can then observe how long a
 *                              route holds a connection. `query()` (the pool-level
 *                              helper) is a SHORT checkout: it never occupies a slot
 *                              across an await the caller controls.
 *
 * Shared by `tests/unit/api/billing-money-safety.test.ts` and
 * `tests/unit/api/billing-replay-entitlement.test.ts` (DRY: one fake, one set of
 * invariants — two copies would let the two suites disagree about the schema).
 */

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

export type Row = Record<string, unknown>;

export interface BillingEventRow {
  provider_event_id: string;
  order_reference: string;
  event_type: string;
  payload: unknown;
}

/**
 * The legal order lifecycle is **not restated here** — it is PARSED from the migration
 * that owns `checkout_orders` (CXD-063).
 *
 * A hand-copied table is a second source of truth for a MONEY invariant: the day the
 * trigger and the copy disagree, every billing suite goes green against a schema that
 * would reject the write in production — which is exactly the failure mode CXD-063 is
 * (the application said `failed -> paid` was fine and the database did not). Parsing
 * the real DDL makes the two impossible to desynchronise, and makes a pending schema
 * change show up as a red test instead of as a production 500.
 *
 * The file is located by SUFFIX, not by number, so renumbering a migration does not
 * silently fall back to a stale default. There is deliberately no fallback at all: an
 * unparseable lifecycle must fail loudly, never default to something permissive.
 *
 * The file is found by walking UP from the working directory (`import.meta.url` is not
 * a `file:` URL once vite has transformed the module), so it resolves whether vitest is
 * invoked from the dashboard or from the repository root. Several `database/migrations`
 * directories exist in the tree, so the walk keeps going until one actually CONTAINS
 * the ledger migration rather than stopping at the first directory of that name.
 */
function findOrderLedgerMigration(): { dir: string; file: string } {
  const searched: string[] = [];
  for (let dir = process.cwd(); ; dir = dirname(dir)) {
    const candidate = join(dir, 'database', 'migrations');
    if (existsSync(candidate)) {
      searched.push(candidate);
      const file = readdirSync(candidate).find((f) => /checkout_order_ledger\.sql$/i.test(f));
      if (file) return { dir: candidate, file };
    }
    if (dirname(dir) === dir) {
      throw new Error(`billing fake: no *_checkout_order_ledger.sql in ${searched.join(', ') || '(no database/migrations found)'}`);
    }
  }
}

/**
 * `status -> statuses it may become`, read out of ANY piece of DDL that declares the
 * `enforce_checkout_order_transition` body.
 *
 * Exported because the lifecycle is declared in TWO places for two different audiences
 * (see `billing-payment-retry.test.ts`): the baseline ledger migration, which only ever
 * runs on a BRAND-NEW database, and a versioned `CREATE OR REPLACE FUNCTION` migration,
 * which is what an ALREADY DEPLOYED database gets. Both must be judged by the SAME
 * parser, or "the baseline says X and the compensation says Y" becomes invisible again.
 *
 * It reads SEMANTICS, NOT SPELLING. The first version of this parser only understood
 * `IN (...)`, and CODEX's migration — which said `= 'paid'`, an exactly equivalent
 * predicate — was reported as "does not permit failed -> paid". A lock that forces
 * somebody else's SQL into the shape its author happened to choose is a false alarm
 * waiting to happen and pressure to write worse DDL, so every equivalent spelling of
 * "the set of statuses NEW.status may take" is accepted:
 *
 *   NEW.status IN ('paid')            NEW.status IN ( 'paid' , 'refunded' )
 *   NEW.status = 'paid'               NEW.status = ANY(ARRAY['paid','refunded'])
 *   NEW.status = ANY('{paid}'::text[])
 *
 * with arbitrary whitespace/newlines throughout. What it must keep saying NO to is the
 * transition being genuinely absent — that is the only thing it is allowed to be red about.
 */
export function parseOrderTransitions(sql: string): Record<string, string[]> {
  const table: Record<string, string[]> = {};
  const rule = new RegExp(
    String.raw`OLD\.status\s*=\s*'(\w+)'\s+AND\s+NEW\.status\s*(?:` +
      String.raw`IN\s*\(([^)]*)\)` +                              // IN ('a','b')
      String.raw`|=\s*ANY\s*\(\s*ARRAY\s*\[([^\]]*)\]` +          // = ANY(ARRAY['a','b'])
      String.raw`|=\s*ANY\s*\(\s*'\{([^}]*)\}'` +                 // = ANY('{a,b}'::text[])
      String.raw`|=\s*'([^']*)'` +                                // = 'a'
      String.raw`)`,
    'gi',
  );
  for (let m = rule.exec(sql); m !== null; m = rule.exec(sql)) {
    const set = m[2] ?? m[3] ?? m[4] ?? m[5] ?? '';
    const targets = set.split(',').map((s) => s.trim().replace(/'/g, '')).filter(Boolean);
    table[m[1]] = [...new Set([...(table[m[1]] ?? []), ...targets])];
  }
  return table;
}

function loadOrderLifecycle(): { dir: string; file: string; table: Record<string, string[]> } {
  const { dir, file } = findOrderLedgerMigration();
  const table = parseOrderTransitions(readFileSync(join(dir, file), 'utf8'));
  if (Object.keys(table).length === 0) {
    throw new Error(`billing fake: could not parse the transition trigger out of ${file}`);
  }
  return { dir, file, table };
}

/**
 * The fake enforces the BASELINE lifecycle only. That is deliberate: the baseline is the
 * complete DDL of the table, whereas a compensating migration is a delta aimed at
 * already-deployed databases. Folding the delta in here would let a fix that landed ONLY
 * as a compensation turn the baseline assertion green, i.e. it would hide exactly the
 * asymmetry between a fresh install and a deployed one that CXD-063 is about.
 */
const lifecycle = loadOrderLifecycle();

/** Name of the migration the lifecycle below was read from (for failure messages). */
export const ORDER_LEDGER_MIGRATION = lifecycle.file;

/** Directory the billing DDL lives in — resolved by walking up, never hardcoded. */
export const ORDER_MIGRATIONS_DIR = lifecycle.dir;

/** `status -> statuses it may legally become`, exactly as the trigger enforces it. */
export const LEGAL_ORDER_TRANSITIONS: Readonly<Record<string, readonly string[]>> = lifecycle.table;

/** A fresh copy, so a test that overrides `state.lifecycle` cannot corrupt the parsed truth. */
const migrationLifecycle = () =>
  Object.fromEntries(Object.entries(lifecycle.table).map(([k, v]) => [k, [...v]]));

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
    /**
     * The order lifecycle this fake enforces. Initialised from the REAL migration on
     * every `reset()`, so the default is always the schema that actually ships.
     *
     * It is writable for exactly one purpose: a change that needs BOTH an application
     * edit and a migration edit is owned by two different people, so the application
     * half must be provable while the schema half is still pending. A test that writes
     * here is asserting behaviour against a schema THAT DOES NOT EXIST YET and must say
     * so in its name — it is not evidence that production works
     * (`billing-payment-retry.test.ts`, CXD-063).
     */
    lifecycle: migrationLifecycle() as Record<string, string[]>,
    /**
     * Bounded connection pool (CXD-062). `size` = how many connections exist;
     * `leased` = how many are checked out RIGHT NOW; `peak` = the high-water mark;
     * `waiters` = callers queued because every connection is out.
     */
    pool: {
      size: Number.POSITIVE_INFINITY,
      leased: 0,
      peak: 0,
      waiters: [] as (() => void)[],
      /**
       * Bumped by `reset()`. A client leased before a reset belongs to a pool that no
       * longer exists (a request the previous test left in flight), so releasing it
       * must not credit the new one — otherwise `leased` drifts negative and a later
       * test reads an exhausted pool as available.
       */
      gen: 0,
    },
  };

  /** Connections currently available to any caller — billing or not. */
  const available = () => state.pool.size - state.pool.leased;

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
      if (from !== next && !(state.lifecycle[from] ?? []).includes(next)) {
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
    available,
    reset() {
      state.orders.clear(); state.users.clear(); state.webhookEvents.clear();
      state.billingEvents.clear(); state.audit.length = 0; state.cart.length = 0;
      state.log.length = 0; state.failOn = null;
      state.lifecycle = migrationLifecycle();
      state.pool.size = Number.POSITIVE_INFINITY;
      state.pool.leased = 0; state.pool.peak = 0; state.pool.waiters.length = 0;
      state.pool.gen += 1;
    },
    query: (text: string, params?: unknown[]) => Promise.resolve(exec(text, params, null)),
    getClient: async () => {
      const gen = state.pool.gen;
      // Lease a slot, or queue until someone releases one — `pg.Pool.connect()`.
      if (state.pool.leased >= state.pool.size) {
        await new Promise<void>((resolve) => { state.pool.waiters.push(resolve); });
      } else {
        state.pool.leased += 1;
      }
      state.pool.peak = Math.max(state.pool.peak, state.pool.leased);

      let pending: (() => void)[] | null = null;
      let released = false;
      return {
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
        release: () => {
          if (released) return;
          released = true;
          if (gen !== state.pool.gen) return;   // stale lease from a previous test
          // Hand the slot straight to the next waiter (no window where a late caller
          // can jump the queue) or give it back to the pool.
          const next = state.pool.waiters.shift();
          if (next) next(); else state.pool.leased -= 1;
        },
      };
    },
  };
}

/** One instance per test FILE (vitest isolates module registries per file). */
export const scriptedPg = createScriptedPostgres();
