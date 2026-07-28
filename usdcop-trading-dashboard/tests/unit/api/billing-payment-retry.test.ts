/**
 * P0 money-safety, fourth round — CODEX finding **CXD-063** (payment retry).
 *
 * ── THE HOLE ────────────────────────────────────────────────────────────────────
 * Wompi's own documentation (https://docs.wompi.co/docs/colombia/reintento-de-pago/)
 * states that when a customer retries a failed payment, the SAME reference ends up
 * with TWO transactions — one DECLINED and one APPROVED — and that the second webhook
 * carries a DIFFERENT transaction id.
 *
 * That sequence slips past every defence we built, and correctly so: it is not a
 * replay. The two events have different `provider_event_id`s (different tx ids) and
 * different `event_type`s, so neither `billing_events` (CXD-056) nor
 * `billing_webhook_events` (mig. 058) sees a conflict, and BOTH are confirmed
 * server-to-server (CXD-059) because both really happened.
 *
 * But the first one leaves `checkout_orders.status = 'failed'`, and `applyApproval`
 * only accepted `created|pending`. So the authoritative APPROVED that follows hits
 *   `order not payable in its current state` → 409 → ROLLBACK,
 * for ever, on every provider retry. **The customer is charged and is NEVER credited,
 * and no retry can fix it.**
 *
 * ── THE REMEDY (minimum safe, as prescribed by CODEX) ────────────────────────────
 *  a) `failed -> paid` is legal **only inside `applyApproval`**, i.e. only after a
 *     MATCHING server-to-server confirmation, inside the same transaction.
 *  b) `payment.declined` stays restricted to `created|pending`, so a LATE decline can
 *     never degrade an order that is already `paid`. (b) is not decoration: without
 *     it, opening (a) just moves the hole to the opposite direction.
 *
 * ── THE OTHER HALF OF THE FIX IS NOT OURS, AND IT HAS TWO LAYERS ────────────────
 * The lifecycle is enforced twice: here, and by the `checkout_order_transition`
 * trigger that owns `checkout_orders` (CODEX's files — we do not touch them). The
 * fake in `tests/mocks/scripted-postgres.ts` PARSES that trigger rather than
 * restating it, so section 0 below is a live precondition and not a copy.
 *
 * The schema half is NOT one edit, it is two, because this repository installs DDL
 * through two different doors (CODEX's own observation, and he is right):
 *
 *   LAYER 1 — the baseline `*_checkout_order_ledger.sql`. It is executed exactly once,
 *     by `init-scripts/26-restore-features.sh`, and ONLY while Postgres initialises an
 *     EMPTY volume. Fixing it fixes every database created from tomorrow on, and no
 *     database that exists today. It is also untracked and un-checksummed: nothing
 *     records that it ran or which version of it ran.
 *
 *   LAYER 2 — an additive, VERSIONED migration that does `CREATE OR REPLACE FUNCTION`
 *     on the same trigger function and is enrolled in an explicit plan of
 *     `scripts/ops/db_migrate.py`. That runner is the only path that touches a
 *     database which is already deployed, and the only one that leaves a tracked,
 *     checksummed record in `_migrations` (a file that is not in a plan is inert:
 *     "adding a file does not make it deployable").
 *
 * Fixing only layer 1 would turn this suite green while every already-deployed
 * database kept burning retried payments — green in one lane is not green in the
 * repository. So section 0 demands BOTH, and says which one is missing.
 *
 * ── HOW TO MAKE THIS FILE GREEN ─────────────────────────────────────────────────
 * By shipping the two migrations. Nothing in this file needs to be touched for it to
 * go green: it reads the real SQL and the real migration plan from disk. Deleting,
 * skipping or `.todo`-ing section 0 is FORBIDDEN — a red here is the only signal that
 * a charged customer is still uncreditable somewhere, and silencing it re-creates
 * CXD-063 with the alarm disconnected.
 *
 * Mutations that MUST turn these red again:
 *  R1  drop `'failed'` from the payable-from set        → scenario 1 never credits
 *  R2  add `'paid'` to `payment.declined`'s from-set    → scenario 2 revokes a paid plan
 *  R3  move the `failed -> paid` opening before the S2S → scenario 4 credits a forgery
 *  R4  widen it in `applyNonApproval` instead           → scenario 2 revokes a paid plan
 *  R5  fix only the baseline, leave deployed DBs broken → section 0 "layer 2" is red
 *  R6  ship the compensation but never add it to a plan → section 0 "deployable" is red
 *
 * What section 0 must NEVER be red about: the STYLE of somebody else's SQL. See 0a.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
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

import {
  scriptedPg as pg,
  LEGAL_ORDER_TRANSITIONS,
  ORDER_LEDGER_MIGRATION,
  ORDER_MIGRATIONS_DIR,
  parseOrderTransitions,
} from '../../mocks/scripted-postgres';
import { createWompiTransactionsApi } from '../../mocks/wompi-transactions-api';
import { POST as webhookPOST } from '@/app/api/billing/webhook/route';

const EVENTS_SECRET = 'events-secret';
const PAYER = '11111111-1111-4111-8111-111111111111';
const PLAN_PRICE = 9_900_000;

/** STRICT: a recorded transaction is immutable, so no test can make a forgery true. */
const providerApi = createWompiTransactionsApi();

/** The two transactions Wompi's retry flow produces for ONE reference. */
const TX_DECLINED = 'tx_attempt_1_declined';
const TX_APPROVED = 'tx_attempt_2_approved';

function signedEvent(opts: {
  txId: string; status: string; reference: string; amountInCents?: number; currency?: string;
}) {
  const tx = {
    id: opts.txId,
    status: opts.status,
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

const REF = `sub_signals_${PAYER}_base_1700000000001`;

const freeEnt = { plan: 'free', assets: ['usdcop'], expires_at: null };
const ent = () =>
  pg.state.users.get(PAYER)!.entitlements as { plan: string; assets: string[]; expires_at: string | null };
const orderStatus = () => String(pg.state.orders.get(REF)!.status);
const credits = () => pg.state.audit.filter((r) => r.action === 'plan_change').length;

/**
 * Model the schema AS IT WILL BE once CODEX lands `failed -> paid` in the trigger.
 *
 * The application half of CXD-063 and the schema half are owned by two different
 * people; without this, the application half would be unprovable until the other
 * landed, and "unprovable" is how a money fix quietly rots. Everything under this flag
 * asserts the APPLICATION behaviour; the DATABASE half is asserted separately by the
 * schema precondition above, which reads the real migration and is red until it ships.
 * Nothing here is evidence that production works today.
 */
function assumeSchemaAllowsRecovery() {
  pg.state.lifecycle.failed = [...new Set([...(pg.state.lifecycle.failed ?? []), 'paid'])];
}

/** Both attempts really happened, and the provider's API says so. */
function seedProviderRetryHistory() {
  providerApi.record({
    id: TX_DECLINED, status: 'DECLINED', reference: REF,
    amount_in_cents: PLAN_PRICE, currency: 'COP',
  });
  providerApi.record({
    id: TX_APPROVED, status: 'APPROVED', reference: REF,
    amount_in_cents: PLAN_PRICE, currency: 'COP',
  });
}

const declinedEvent = () => signedEvent({ txId: TX_DECLINED, status: 'DECLINED', reference: REF });
const approvedEvent = () => signedEvent({ txId: TX_APPROVED, status: 'APPROVED', reference: REF });

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

  pg.state.users.set(PAYER, { email: 'payer@example.com', role: 'free', entitlements: { ...freeEnt } });
  pg.state.orders.set(REF, {
    user_id: PAYER, plan: 'signals', addon_assets: [],
    amount_cents: PLAN_PRICE, currency: 'COP', reference: REF, status: 'pending',
  });
  seedProviderRetryHistory();
});

afterEach(() => { vi.unstubAllGlobals(); });

// ═══════════════════════════════════════════ 0 · locating the SECOND schema layer
/**
 * The compensating migration is found BY WHAT IT DOES, never by its number: the number
 * is CODEX's to choose, and a test that hardcoded `060_*.sql` would either block his
 * numbering or go quietly green against the wrong file. The signature we look for is
 * the only one a fix for existing databases can have:
 *
 *   - it is a versioned `.sql` in the same migrations directory,
 *   - it is NOT the baseline ledger file (that one only runs on an empty volume),
 *   - it does `CREATE OR REPLACE FUNCTION` (the only way to change a live database's
 *     trigger without dropping the table), and
 *   - it names the order-status transition it is replacing.
 *
 * Everything is derived from where the baseline was found, so the whole check moves
 * with the repository instead of pinning absolute paths.
 */
const REPO_ROOT = dirname(dirname(ORDER_MIGRATIONS_DIR));
/** The runner that is the ONLY door into an already-deployed database. */
const MIGRATION_RUNNER = join(REPO_ROOT, 'scripts', 'ops', 'db_migrate.py');

const IS_BASELINE = /checkout_order_ledger\.sql$/i;
const REPLACES_A_FUNCTION = /CREATE\s+OR\s+REPLACE\s+FUNCTION/i;
const TOUCHES_ORDER_TRANSITION = /enforce_checkout_order_transition|illegal checkout order transition/i;

interface Compensation { file: string; transitions: Record<string, readonly string[]> }

function findOrderTransitionCompensations(): Compensation[] {
  return readdirSync(ORDER_MIGRATIONS_DIR)
    .filter((f) => /\.sql$/i.test(f) && !IS_BASELINE.test(f))
    .sort()
    .map((file) => ({ file, sql: readFileSync(join(ORDER_MIGRATIONS_DIR, file), 'utf8') }))
    .filter(({ sql }) => REPLACES_A_FUNCTION.test(sql) && TOUCHES_ORDER_TRANSITION.test(sql))
    .map(({ file, sql }) => ({ file, transitions: parseOrderTransitions(sql) }));
}

/**
 * A migration file that is not listed in an explicit plan is INERT — `db_migrate.py`
 * says so itself: "New migrations are never discovered by a broad production glob:
 * adding a file does not make it deployable". Enrolment is therefore part of the fix,
 * not paperwork, and it is also what produces the tracked+checksummed `_migrations`
 * row that the untracked baseline never produces.
 */
function isEnrolledInAMigrationPlan(file: string): boolean {
  if (!existsSync(MIGRATION_RUNNER)) return false;
  // A mention inside a `#` comment is documentation, not enrolment: it does not put
  // the file in any plan tuple, so it would never run.
  return readFileSync(MIGRATION_RUNNER, 'utf8')
    .split(/\r?\n/)
    .some((line) => line.includes(file) && !line.trim().startsWith('#'));
}

const COMPENSATIONS = findOrderTransitionCompensations();
const FOUND = COMPENSATIONS.length
  ? COMPENSATIONS.map((c) => c.file).join(', ')
  : '(none)';

// ════════════════════════════════════ 0 · the half of the fix that lives in the schema
describe('CXD-063 · schema precondition (owned by CODEX, migration *_checkout_order_ledger.sql)', () => {
  /**
   * The application may only open a transition the database also permits. This
   * assertion reads the REAL migration, so it cannot drift: it is red while the
   * trigger still refuses the recovery, and green the moment CODEX adds `failed` to
   * the from-states of `paid`:
   *
   *   (OLD.status='failed'  AND NEW.status IN ('paid'))
   *
   * Note the asymmetry that must be preserved: `paid` must NOT gain `failed` as a
   * destination — that is the half that stops a late decline (scenario 2).
   */
  it('LAYER 1/2 — the baseline permits the payment-retry recovery failed -> paid', () => {
    expect(
      LEGAL_ORDER_TRANSITIONS.failed ?? [],
      `LAYER 1 of 2 MISSING (fresh installs). ${ORDER_LEDGER_MIGRATION} still refuses ` +
      'failed -> paid, so any database created from now on is born unable to credit a ' +
      'retried payment. Fix: add (OLD.status=\'failed\' AND NEW.status IN (\'paid\')) to ' +
      'enforce_checkout_order_transition() in that file. This is NOT the same defect as a ' +
      'red LAYER 2 — see the next describe block.',
    ).toContain('paid');
  });

  it('still refuses the reverse, paid -> failed (a late decline must never degrade a sale)', () => {
    expect(LEGAL_ORDER_TRANSITIONS.paid ?? []).not.toContain('failed');
    expect(LEGAL_ORDER_TRANSITIONS.paid ?? []).not.toContain('cancelled');
  });

  /**
   * Localises the remaining gap. The application no longer refuses the recovery with
   * its own 409 — it issues the UPDATE, and only the trigger stands in the way. This
   * test therefore PASSES both before and after CODEX's migration lands (afterwards
   * the recovery simply succeeds), while pinning today's blame where it belongs.
   */
  it('the application no longer refuses the recovery on its own', async () => {
    await post(declinedEvent());
    const res = await post(approvedEvent());

    expect(
      await res.clone().json().catch(() => ({})),
      'a 409 "not payable" here would mean the application half regressed',
    ).not.toMatchObject({ error: 'order not payable in its current state' });
  });
});

// ═════════════════════════ 0a · the lock reads SEMANTICS, not somebody else's style
/**
 * Regression for a defect in THIS harness, not in the schema.
 *
 * The first version of the parser only recognised `NEW.status IN ('paid')`. CODEX's
 * migration expressed the very same rule as `NEW.status = 'paid'` and was reported as
 * "does not permit failed -> paid" — a red that was purely about spelling, which forced
 * him to rewrite valid DDL to satisfy a test. Next time it would be `ANY(ARRAY[...])`
 * and the alarm would fire again with nothing broken.
 *
 * The contract of the parser is therefore: RED when the transition is absent, never when
 * it is written differently.
 */
describe('CXD-063 · the transition parser accepts equivalent SQL spellings', () => {
  const permitsFailedToPaid = (sql: string) => (parseOrderTransitions(sql).failed ?? []).includes('paid');

  it.each([
    ["IN, single",        "(OLD.status='failed' AND NEW.status IN ('paid'))"],
    ["IN, multi",         "(OLD.status='failed' AND NEW.status IN ('paid','refunded'))"],
    ["IN, loose spacing", "( OLD.status  =  'failed'   AND   NEW.status   IN (  'paid' ,  'refunded' ) )"],
    ["IN, newlines",      "(OLD.status = 'failed'\n       AND NEW.status IN (\n         'paid'\n       ))"],
    ["equality",          "(OLD.status = 'failed' AND NEW.status = 'paid')"],
    ["equality, tight",   "(OLD.status='failed' AND NEW.status='paid')"],
    ["ANY(ARRAY[...])",   "(OLD.status='failed' AND NEW.status = ANY(ARRAY['paid','refunded']))"],
    ["ANY('{...}')",      "(OLD.status='failed' AND NEW.status = ANY('{paid,refunded}'::text[]))"],
    ["lowercase keywords","(old.status='failed' and new.status in ('paid'))"],
  ])('reads %s as "failed may become paid"', (_form, sql) => {
    expect(permitsFailedToPaid(sql), `spelling rejected, but it says the same thing: ${sql}`).toBe(true);
  });

  it('still says NO when the transition is genuinely absent, in every spelling', () => {
    expect(permitsFailedToPaid("(OLD.status='failed' AND NEW.status IN ('cancelled','expired'))")).toBe(false);
    expect(permitsFailedToPaid("(OLD.status='failed' AND NEW.status = 'cancelled')")).toBe(false);
    expect(permitsFailedToPaid("(OLD.status='failed' AND NEW.status = ANY(ARRAY['cancelled']))")).toBe(false);
    // ...and it does not confuse the ORIGIN either: pending -> paid is not failed -> paid.
    expect(permitsFailedToPaid("(OLD.status='pending' AND NEW.status = 'paid')")).toBe(false);
  });

  it('keeps the destinations of each origin separate', () => {
    const t = parseOrderTransitions(
      "(OLD.status='pending' AND NEW.status IN ('paid','failed')) OR " +
      "(OLD.status='failed' AND NEW.status = 'paid') OR " +
      "(OLD.status='paid' AND NEW.status = ANY(ARRAY['refunded','charged_back']))",
    );
    expect(t.pending).toEqual(expect.arrayContaining(['paid', 'failed']));
    expect(t.failed).toEqual(['paid']);
    expect(t.paid).toEqual(expect.arrayContaining(['refunded', 'charged_back']));
    expect(t.paid).not.toContain('failed');
  });
});

// ══════════════════════════ 0b · the schema layer that reaches the databases in use
/**
 * Why this block exists even though the baseline is already asserted above.
 *
 * The baseline is applied by `init-scripts/26-restore-features.sh`, which Postgres only
 * runs while it initialises an EMPTY data directory. Editing it is a fix for databases
 * that do not exist yet. Every database already running — including the one holding the
 * customer of CXD-063 — keeps the function it was created with, for ever, unless a
 * versioned migration replaces it.
 *
 * The baseline is, on top of that, untracked and un-checksummed: nothing records that it
 * ran, in which version, or whether the running function still matches the file. The
 * versioned migration is the ONLY layer that leaves an auditable `_migrations` row, so it
 * is also the only evidence an operator can ever produce that the money path was repaired.
 *
 * Hence: one fix, two mandatory layers, two distinct failure messages.
 */
describe('CXD-063 · schema half, LAYER 2 — the fix for databases that ALREADY exist', () => {
  const LAYER2_MISSING =
    'LAYER 2 of 2 MISSING (deployed databases). No versioned migration in ' +
    `${ORDER_MIGRATIONS_DIR} does CREATE OR REPLACE FUNCTION on the checkout-order ` +
    'transition. Repairing only the baseline repairs only databases created from now ' +
    'on: every deployed database keeps the old trigger and keeps refusing failed -> ' +
    `paid, so charged customers stay uncreditable. Scanned candidates: ${FOUND}.`;

  it('a versioned migration re-declares the transition function for existing databases', () => {
    expect(COMPENSATIONS.map((c) => c.file), LAYER2_MISSING).not.toHaveLength(0);
  });

  it('that migration actually opens the retry recovery failed -> paid', () => {
    const opens = COMPENSATIONS.filter((c) => (c.transitions.failed ?? []).includes('paid'));
    expect(
      opens.map((c) => c.file),
      COMPENSATIONS.length === 0
        ? LAYER2_MISSING
        : `LAYER 2 INCOMPLETE. ${FOUND} replaces the transition function but its new body ` +
          'still does not allow failed -> paid, so running it changes nothing for the ' +
          'customer who retried a declined payment.',
    ).not.toHaveLength(0);
  });

  it('and preserves the asymmetry: no compensation re-opens paid -> failed/cancelled', () => {
    for (const c of COMPENSATIONS) {
      expect(
        c.transitions.paid ?? [],
        `LAYER 2 UNSAFE. ${c.file} lets a paid order move backwards, which is how a late ` +
        'decline (scenario 2 below) would revoke a plan the customer paid for.',
      ).not.toContain('failed');
      expect(c.transitions.paid ?? [], `LAYER 2 UNSAFE: ${c.file}`).not.toContain('cancelled');
    }
  });

  /**
   * `db_migrate.py` deliberately refuses to discover migrations by glob, so a file that
   * is not named in a plan never runs anywhere. A compensation that is not enrolled is
   * indistinguishable, in production, from a compensation that was never written.
   */
  it('the compensation is deployable: enrolled in an explicit plan of db_migrate.py', () => {
    expect(existsSync(MIGRATION_RUNNER), `migration runner not found at ${MIGRATION_RUNNER}`).toBe(true);

    const enrolled = COMPENSATIONS.filter((c) => isEnrolledInAMigrationPlan(c.file));
    expect(
      enrolled.map((c) => c.file),
      COMPENSATIONS.length === 0
        ? `${LAYER2_MISSING} (nothing to enrol yet)`
        : `LAYER 2 INERT. ${FOUND} exists but is not listed in any MIGRATION_PLANS entry of ` +
          `${MIGRATION_RUNNER}, and that runner never picks files up by glob ("adding a file ` +
          'does not make it deployable"). It will therefore never run against a deployed ' +
          'database, and will never leave the tracked+checksummed `_migrations` row that the ' +
          'untracked baseline cannot leave either.',
    ).not.toHaveLength(0);
  });
});

// ═════════════════════════════════════════════════════ 1 · the documented retry flow
describe('CXD-063 · a declined attempt followed by an approved one credits exactly once', () => {
  beforeEach(assumeSchemaAllowsRecovery);

  it('the first attempt leaves the order failed and grants nothing', async () => {
    const res = await post(declinedEvent());

    expect(res.status).toBe(200);
    expect(orderStatus()).toBe('failed');
    expect(ent().plan).toBe('free');
    expect(credits()).toBe(0);
    // It was confirmed against the provider like every other event.
    expect(providerApi.calls.some((u) => u.includes(TX_DECLINED))).toBe(true);
  });

  it('the second, AUTHORITATIVE attempt recovers the order and credits the payer', async () => {
    expect((await post(declinedEvent())).status).toBe(200);
    expect(orderStatus()).toBe('failed');

    const res = await post(approvedEvent());

    expect(res.status, 'the payer was charged: he MUST be credited').toBe(200);
    expect(orderStatus()).toBe('paid');
    expect(ent().plan).toBe('signals');
    expect(credits(), 'exactly one credit').toBe(1);
    expect(providerApi.calls.some((u) => u.includes(TX_APPROVED))).toBe(true);
  });

  it('neither ledger mistakes the retry for a replay (different id, different type)', async () => {
    await post(declinedEvent());
    await post(approvedEvent());

    expect(pg.state.billingEvents.size, 'two distinct provider events').toBe(2);
    expect(pg.state.webhookEvents.has(`${REF}:payment.declined`)).toBe(true);
    expect(pg.state.webhookEvents.has(`${REF}:payment.approved`)).toBe(true);
    // No forgery was suspected: a retry is not an incident.
    expect(pg.state.audit.filter((r) => String(r.action).includes('replay')).length).toBe(0);
  });

  it('the recovery is auditable: the trail says which state the order came from', async () => {
    await post(declinedEvent());
    await post(approvedEvent());

    const credit = pg.state.audit.filter((r) => r.action === 'plan_change').at(-1)!;
    const detail = JSON.parse(String(credit.detail));
    expect(detail.reference).toBe(REF);
    expect(detail.provider_event_id).toContain(TX_APPROVED);
    expect(detail.previous_order_status, 'a failed->paid recovery must be visible').toBe('failed');
  });
});

// ══════════════════════════════════════════════════ 2 · a LATE decline never degrades
describe('CXD-063 · an approval is never undone by a decline that arrives afterwards', () => {
  it('the order stays paid and the entitlements are untouched', async () => {
    expect((await post(approvedEvent())).status).toBe(200);
    expect(orderStatus()).toBe('paid');
    const granted = JSON.stringify(ent());

    // The declined attempt of the SAME reference is delivered late (provider queues
    // are not ordered). It is authoritative and genuinely declined — and irrelevant.
    const late = await post(declinedEvent());

    expect(late.status).toBeLessThan(500);
    expect(orderStatus(), 'a paid order is never degraded').toBe('paid');
    expect(JSON.stringify(ent()), 'paid rights are not revoked by an old failure').toBe(granted);
    expect(ent().plan).toBe('signals');
    expect(credits()).toBe(1);
  });

  it('records the late decline as an explicit no-op rather than swallowing it', async () => {
    await post(approvedEvent());
    await post(declinedEvent());

    const noop = pg.state.audit.filter((r) => r.action === 'plan_payment_event_noop');
    expect(noop.length, 'the late decline must leave a trail').toBe(1);
    const detail = JSON.parse(String(noop[0].detail));
    expect(detail.applied).toBe(false);
    expect(detail.order_status).toBe('paid');
  });
});

// ═══════════════════════════════════════════════════════ 3 · replay is still a no-op
describe('CXD-063 · the idempotency ledger still holds through the retry flow', () => {
  beforeEach(assumeSchemaAllowsRecovery);

  it('replaying the approval credits nothing a second time', async () => {
    await post(declinedEvent());
    await post(approvedEvent());
    const granted = JSON.stringify(ent());

    const replay = await post(approvedEvent());

    expect(replay.status).toBe(200);
    expect(await replay.json()).toMatchObject({ duplicate: true });
    expect(JSON.stringify(ent())).toBe(granted);
    expect(credits(), 'still exactly one credit').toBe(1);
    expect(orderStatus()).toBe('paid');
  });

  it('replaying the DECLINE after the recovery does not re-fail the order', async () => {
    await post(declinedEvent());
    await post(approvedEvent());

    const replay = await post(declinedEvent());
    expect(replay.status).toBeLessThan(500);
    expect(orderStatus()).toBe('paid');
    expect(ent().plan).toBe('signals');
  });
});

// ════════════════════════════════ 4 · the opening is only for CONFIRMED payments
describe('CXD-063 · a failed order is only recoverable by an authoritatively confirmed payment', () => {
  // Deliberately assumes the PERMISSIVE schema: the point is that even where the
  // database would allow `failed -> paid`, an unconfirmed body still cannot trigger it.
  beforeEach(assumeSchemaAllowsRecovery);

  it('an APPROVED the provider has never heard of does not resurrect it', async () => {
    await post(declinedEvent());
    expect(orderStatus()).toBe('failed');

    const res = await post(signedEvent({ txId: 'tx_invented', status: 'APPROVED', reference: REF }));

    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
    expect(orderStatus(), 'an unconfirmed body must not reopen a failed order').toBe('failed');
    expect(ent().plan).toBe('free');
    expect(credits()).toBe(0);
  });

  it('an APPROVED whose authoritative record names ANOTHER reference does not resurrect it', async () => {
    await post(declinedEvent());
    // A real payment of somebody else, replayed onto this failed order.
    providerApi.record({
      id: 'tx_someone_else', status: 'APPROVED',
      reference: `sub_signals_22222222-2222-4222-8222-222222222222_base_1700000000002`,
      amount_in_cents: PLAN_PRICE, currency: 'COP',
    });

    const res = await post(signedEvent({ txId: 'tx_someone_else', status: 'APPROVED', reference: REF }));

    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(orderStatus()).toBe('failed');
    expect(ent().plan).toBe('free');
  });

  it('a provider outage leaves the failed order recoverable by the later retry', async () => {
    await post(declinedEvent());
    providerApi.setMode('down');

    expect((await post(approvedEvent())).status).toBe(503);
    expect(orderStatus()).toBe('failed');
    expect(ent().plan).toBe('free');
    // Nothing was written, so the retry is a fresh attempt and not a "duplicate".
    expect(pg.state.billingEvents.has(`wompi:${TX_APPROVED}:APPROVED`)).toBe(false);

    providerApi.setMode('up');
    expect((await post(approvedEvent())).status).toBe(200);
    expect(orderStatus()).toBe('paid');
    expect(ent().plan).toBe('signals');
    expect(credits()).toBe(1);
  });
});
