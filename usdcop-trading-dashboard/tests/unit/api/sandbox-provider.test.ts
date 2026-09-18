/**
 * The sandbox billing provider must be a TEST DOUBLE, never a bypass.
 *
 * It exists because the Wompi integration is complete but inert without merchant keys, so
 * the purchase path — checkout, signed webhook, server-to-server confirmation, entitlement
 * union, cart clear — could not be run end to end at all. What makes that legitimate is
 * that it runs the SAME checks: a forged signature is refused here exactly as it would be
 * by the real adapter, and the fields the signature does not cover are declared so the
 * webhook still confirms them against the sealed quote.
 *
 * And it must be impossible to switch on by accident. Three independent switches are
 * required; any one missing and construction throws.
 *
 * Mutations that MUST turn these red again:
 *   M1  constructing without BILLING_SANDBOX_ENABLED                → silent free purchases
 *   M2  constructing on a deployment that calls itself production   → same, in the worst place
 *   M3  accepting a webhook whose signature does not verify         → forged payments credited
 *   M4  dropping `reference` from `unauthenticatedFields`           → webhook stops confirming
 *                                                                     who gets credited
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { SandboxProvider, signSandboxEvent } from '@/lib/billing/sandbox';

const ENV_KEYS = ['APP_ENV', 'BILLING_SANDBOX_ENABLED', 'BILLING_SANDBOX_SECRET'] as const;
const saved: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const k of ENV_KEYS) saved[k] = process.env[k];
  process.env.BILLING_SANDBOX_ENABLED = 'true';
  process.env.BILLING_SANDBOX_SECRET = 'test-secret';
  delete process.env.APP_ENV;
});
afterEach(() => {
  for (const k of ENV_KEYS) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

const USER = 'aa8552d0-cd68-4c3b-9967-8d67a3b73b5b';

function signedBody(over: Record<string, unknown> = {}) {
  const id = 'sbx_1';
  const status = 'APPROVED';
  const amount = 13_800_000;
  return JSON.stringify({
    id,
    reference: `sub_signals_${USER}_xauusd_1`,
    status,
    amount_in_cents: amount,
    currency: 'COP',
    signature: signSandboxEvent(id, status, amount),
    ...over,
  });
}

describe('it cannot be switched on by accident', () => {
  it('refuses without the explicit enable flag', () => {
    delete process.env.BILLING_SANDBOX_ENABLED;
    expect(() => new SandboxProvider()).toThrow(/BILLING_SANDBOX_ENABLED/);
  });

  it('refuses on a deployment that declares itself production', () => {
    process.env.APP_ENV = 'production';
    expect(() => new SandboxProvider()).toThrow(/production/);
  });

  it('refuses without its own secret, at construction rather than at the first charge', () => {
    delete process.env.BILLING_SANDBOX_SECRET;
    expect(() => new SandboxProvider()).toThrow(/BILLING_SANDBOX_SECRET/);
  });

  it('constructs when all three switches are set', () => {
    expect(() => new SandboxProvider()).not.toThrow();
  });
});

describe('it runs the real webhook gauntlet, not a shortcut', () => {
  it('refuses a forged signature', async () => {
    const p = new SandboxProvider();
    const v = await p.verifyWebhook(signedBody({ signature: 'deadbeef' }), new Headers());
    expect(v.valid).toBe(false);
    expect(v.event).toBeUndefined();
  });

  it('refuses a body whose amount was tampered with after signing', async () => {
    const p = new SandboxProvider();
    const v = await p.verifyWebhook(signedBody({ amount_in_cents: 1 }), new Headers());
    expect(v.valid).toBe(false);
  });

  it('accepts a correctly signed event and normalizes it', async () => {
    const p = new SandboxProvider();
    const v = await p.verifyWebhook(signedBody(), new Headers());
    expect(v.valid).toBe(true);
    expect(v.event?.type).toBe('payment.approved');
    expect(v.event?.providerTransactionId).toBe('sbx_1');
  });

  it('declares `reference` as unauthenticated, so the webhook still confirms who gets credited', async () => {
    const p = new SandboxProvider();
    const v = await p.verifyWebhook(signedBody(), new Headers());
    // The signature covers id|status|amount — NOT the reference. Hiding that would let a
    // replay with a swapped reference credit the wrong account.
    expect(v.event?.unauthenticatedFields).toContain('reference');
  });

  it('acknowledges and drops a status it does not model, inventing no transition', async () => {
    const p = new SandboxProvider();
    const v = await p.verifyWebhook(signedBody({
      status: 'PENDING', signature: signSandboxEvent('sbx_1', 'PENDING', 13_800_000),
    }), new Headers());
    expect(v.valid).toBe(true);
    expect(v.event).toBeUndefined();
    expect(v.ignored?.providerStatus).toBe('PENDING');
  });

  it('reports a transaction it never issued as not_found, never as unavailable', async () => {
    const p = new SandboxProvider();
    const r = await p.fetchTransaction('sbx_never');
    // "I have no such payment" and "I cannot tell right now" get opposite HTTP answers
    // upstream; collapsing them would accept a forgery during an outage.
    expect(r.kind).toBe('not_found');
  });
});

describe('checkout produces a reference the codec round-trips', () => {
  it('encodes user, plan and add-ons', async () => {
    const p = new SandboxProvider();
    const s = await p.createCheckout({
      userId: USER, email: 'x@y.co', plan: 'signals', addOnAssets: ['xauusd'],
    });
    expect(s.provider).toBe('sandbox');
    expect(s.reference).toContain(USER);
    expect(s.checkoutUrl).toContain(encodeURIComponent(s.reference));
  });

  it('round-trips the multi-asset bundle plan too', async () => {
    const p = new SandboxProvider();
    // `desk` was added to the contract after the reference regex was written; encoding
    // asserts its own round-trip, so a plan the codec does not know cannot be sold.
    const s = await p.createCheckout({ userId: USER, email: 'x@y.co', plan: 'desk' });
    expect(s.reference).toMatch(/^sub_desk_/);
  });
});
