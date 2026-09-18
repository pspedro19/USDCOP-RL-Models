/**
 * Two commercial mechanisms that must not fail open.
 *
 * ── 1 · EXPIRING GRANTS (the NDA data room) ──────────────────────────────────────
 * Selling research access means handing a prospect `research:read` for the length of a
 * deal. `rbac_user_overrides` could express the grant but never its END, so every data-room
 * access was permanent until a human remembered to revoke it — the documented failure mode
 * of every data room. Expiry is enforced at RESOLUTION time, so a lapsed grant is inert the
 * instant it lapses rather than whenever a sweeper next runs.
 *
 * A DENY may never expire: a revocation that silently lapses would restore a permission
 * somebody deliberately took away. That is a CHECK in migration 091 and a rule here.
 *
 * ── 2 · BILLING INTERVAL ─────────────────────────────────────────────────────────
 * Every purchase granted a hardcoded 30 days. An annual plan would have been charged for a
 * year and credited for a month. The grant window now comes from the sealed quote.
 *
 * Mutations that MUST turn these red again:
 *   M1  `isOverrideActive` returning true for a past `expires_at`  → dead grant still works
 *   M2  letting a deny carry an expiry                             → revocation self-reverts
 *   M3  INTERVAL_DAYS.year set to 30                               → annual credited as monthly
 *   M4  a plan/currency/interval combo priced 0 instead of null    → unsold plan looks free
 */
import { describe, expect, it } from 'vitest';

import { INTERVAL_DAYS, PLAN_DEFAULTS } from '@/lib/contracts/rbac.contract';
import { isOverrideActive, type UserOverride } from '@/lib/auth/rbac-resolver';
import { isPlanSold, planOffers, planPrice } from '@/lib/billing/prices';

const DAY = 86_400_000;
const grant = (over: Partial<UserOverride> = {}): UserOverride => ({
  permission: 'research:read', effect: 'grant', expires_at: null, nda_reference: null, ...over,
});

describe('expiring grants close themselves', () => {
  it('applies a grant with no expiry', () => {
    expect(isOverrideActive(grant())).toBe(true);
  });

  it('applies a grant whose window is still open', () => {
    expect(isOverrideActive(grant({ expires_at: new Date(Date.now() + 5 * DAY).toISOString() }))).toBe(true);
  });

  it('stops applying the moment the window closes', () => {
    expect(isOverrideActive(grant({ expires_at: new Date(Date.now() - 1000).toISOString() }))).toBe(false);
  });

  it('treats an unparseable expiry as closed, never as permanent', () => {
    expect(isOverrideActive(grant({ expires_at: 'whenever' }))).toBe(false);
  });

  it('never lets a deny lapse, even if a row somehow carried an expiry', () => {
    const denied: UserOverride = {
      permission: 'research:read', effect: 'deny',
      expires_at: new Date(Date.now() - 30 * DAY).toISOString(), nda_reference: null,
    };
    expect(isOverrideActive(denied)).toBe(true); // still denying
  });
});

describe('the billing interval decides the grant window', () => {
  it('credits a year for a year, not a month', () => {
    expect(INTERVAL_DAYS.year).toBe(365);
    expect(INTERVAL_DAYS.month).toBe(30);
    expect(INTERVAL_DAYS.year).toBeGreaterThan(INTERVAL_DAYS.month * 11);
  });
});

describe('the price table distinguishes "free" from "not sold"', () => {
  it('prices the four-asset bundle annually in USD', () => {
    expect(planPrice('desk', 'USD', 'year')).toBe(10_000_00);
    expect(isPlanSold('desk', 'USD', 'year')).toBe(true);
  });

  it('does not sell the bundle monthly, or in the local channel', () => {
    expect(planPrice('desk', 'USD', 'month')).toBeNull();
    expect(planPrice('desk', 'COP', 'year')).toBeNull();
    expect(isPlanSold('desk', 'COP', 'month')).toBe(false);
  });

  it('keeps the ladder monotonic: a year of one asset never costs more than the bundle', () => {
    const oneAssetYear = planPrice('signals', 'USD', 'year');
    const bundleYear = planPrice('desk', 'USD', 'year');
    expect(oneAssetYear).not.toBeNull();
    expect(bundleYear).not.toBeNull();
    expect(oneAssetYear as number).toBeLessThan(bundleYear as number);
  });

  it('charges a premium for month-to-month over committing for a year', () => {
    const monthly = planPrice('signals', 'USD', 'month') as number;
    const yearly = planPrice('signals', 'USD', 'year') as number;
    expect(monthly * 12).toBeGreaterThan(yearly);
  });

  it('lists only the combinations a plan is actually sold in', () => {
    const offers = planOffers('desk');
    expect(offers).toHaveLength(1);
    expect(offers[0]).toMatchObject({ currency: 'USD', interval: 'year' });
  });
});

describe('the bundle plan carries every onboarded asset', () => {
  it('covers all four, and grants no automatic execution', () => {
    expect(PLAN_DEFAULTS.desk.assets).toEqual(['usdcop', 'xauusd', 'btcusdt', 'spx500']);
    // Automatic execution is a separate legal gate (rbac.md §9) — a bundle must not
    // hand it out as a side effect of covering more assets.
    expect(PLAN_DEFAULTS.desk.execution.enabled).toBe(false);
    expect(PLAN_DEFAULTS.desk.signals_realtime).toBe(true);
  });
});
