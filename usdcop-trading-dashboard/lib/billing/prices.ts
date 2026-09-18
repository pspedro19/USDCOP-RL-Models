/**
 * Plan price SSOT (COP). Single source consumed by the Wompi provider (checkout
 * amount) AND the admin revenue aggregator (MRR/ARR). Override the placeholders
 * via BILLING_PRICES_COP env (JSON, cents): {"signals":<cents>,"auto":<cents>}.
 *
 * Prices are a business decision (spec §B.5) — these are placeholders until set.
 */
import type { BillingInterval, Currency, PlanId } from '@/lib/contracts/rbac.contract';

/** Monthly price per plan, in COP CENTS. */
export const DEFAULT_PRICES_COP_CENTS: Record<string, number> = {
  free: 0,
  signals: 990_000_00 / 10, // 99.000 COP/mes
  auto: 2_990_000_00 / 10, // 299.000 COP/mes
  desk: 0, // local channel does not sell the bundle monthly — see PRICE_TABLE
};

// ── multi-currency, multi-interval price table (SSOT) ─────────────────────────
//
// The original table could express exactly one thing: a monthly COP price. The commercial
// plan needs two channels (Wompi settles COP locally, USD internationally) and two periods
// (month-to-month carries a flexibility premium; the annual commitment is what unlocks the
// low rate). Both dimensions live HERE so checkout, the webhook's amount comparison and the
// revenue aggregator can never disagree about what a plan costs.
//
// A missing combination is `null`, not zero: "we do not sell this" and "this is free" are
// different statements, and rendering an unpublished price as $0 would be a live offer.

/** Minor units per whole unit. COP is quoted in cents by Wompi; USD in cents likewise. */
export const CENTS_PER_UNIT = 100;

type PriceCell = number | null; // minor units (cents), or null when not sold in that combo

const PRICE_TABLE: Record<Currency, Record<BillingInterval, Record<string, PriceCell>>> = {
  COP: {
    month: { free: 0, signals: 99_000_00, auto: 299_000_00, desk: null },
    // Annual COP = ten months for twelve: the same commitment discount as the USD ladder,
    // expressed in the currency the local gateway actually settles.
    year: { free: 0, signals: 990_000_00, auto: 2_990_000_00, desk: null },
  },
  USD: {
    // Month-to-month is the flexibility premium, not the headline: US$1.000 per month for a
    // single asset is deliberately more expensive than committing for a year.
    month: { free: 0, signals: 1_000_00, auto: null, desk: null },
    // The ladder: 1 asset US$4.000 → the four-asset bundle US$10.000. Monotonic — more
    // assets always cost more, longer commitment always costs less per unit, and no customer
    // can ever pay less by buying more.
    year: { free: 0, signals: 4_000_00, auto: null, desk: 10_000_00 },
  },
};

/** Price of a plan for one (currency, interval), in MINOR UNITS, or null if not sold. */
export function planPrice(plan: string, currency: Currency, interval: BillingInterval): PriceCell {
  const cell = PRICE_TABLE[currency]?.[interval]?.[plan];
  return typeof cell === 'number' ? cell : null;
}

/** True when the plan is actually purchasable in that channel — never infer it from a 0. */
export function isPlanSold(plan: string, currency: Currency, interval: BillingInterval): boolean {
  return planPrice(plan, currency, interval) !== null;
}

/** Every (currency, interval) combination a plan is sold in — drives the pricing page. */
export function planOffers(plan: string): Array<{
  currency: Currency; interval: BillingInterval; price_minor: number;
}> {
  const out: Array<{ currency: Currency; interval: BillingInterval; price_minor: number }> = [];
  for (const currency of ['COP', 'USD'] as Currency[]) {
    for (const interval of ['month', 'year'] as BillingInterval[]) {
      const v = planPrice(plan, currency, interval);
      if (v !== null && v > 0) out.push({ currency, interval, price_minor: v });
    }
  }
  return out;
}

/** Effective monthly prices (COP cents), with env override merged over defaults. */
export function planPricesCents(): Record<string, number> {
  try {
    return { ...DEFAULT_PRICES_COP_CENTS, ...JSON.parse(process.env.BILLING_PRICES_COP ?? '{}') };
  } catch {
    return DEFAULT_PRICES_COP_CENTS;
  }
}

/** Monthly price of one plan, in COP cents (0 for unknown/free). */
export function planPriceCents(plan: string): number {
  return planPricesCents()[plan] ?? 0;
}

/** Monthly price of one plan, in whole COP. */
export function planPriceCop(plan: string): number {
  return Math.round(planPriceCents(plan) / 100);
}

/** Human labels for the paid plans (admin revenue breakdown). */
export const PLAN_LABELS: Record<string, string> = {
  free: 'Free',
  signals: 'Señales Pro',
  auto: 'Auto Premium',
  desk: 'Mesa · 4 activos',
};

/** Plans that generate recurring revenue. */
export const PAID_PLANS: PlanId[] = ['signals', 'auto', 'desk'];

// ── per-asset add-on prices (SSOT) ─────────────────────────────────────────────

/**
 * Per-asset add-on monthly price, in WHOLE COP. Single source consumed by the
 * catalog service (/api/catalog), Pricing and Cart. Only assets with a real
 * published price appear here — everything else resolves to `null`, which the UI
 * renders as an honest "—" (never invent a price). Coming-soon assets stay out.
 * Override via BILLING_ADDON_PRICES_COP env (JSON, whole COP): {"xauusd":39000}.
 */
export const DEFAULT_ADDON_PRICES_COP: Record<string, number> = {
  // XAU/USD is deliberately ABSENT: Gold is bundled into every plan (PLAN_DEFAULTS), so it
  // is not an add-on any more. Leaving a price here would publish something for sale that
  // the buyer already owns — the cart would refuse it as ALREADY_ENTITLED after showing a
  // price, which reads as a broken catalogue rather than as an included benefit.
  btcusdt: 39_000, // Bitcoin add-on
  spx500: 39_000, // S&P 500 add-on; final price remains a business decision
};

/** Effective add-on prices (whole COP), with env override merged over defaults. */
export function addonPricesCop(): Record<string, number> {
  try {
    return { ...DEFAULT_ADDON_PRICES_COP, ...JSON.parse(process.env.BILLING_ADDON_PRICES_COP ?? '{}') };
  } catch {
    return DEFAULT_ADDON_PRICES_COP;
  }
}

/** Add-on monthly price for one asset in whole COP, or `null` if none is published. */
export function addonPriceCop(assetId: string): number | null {
  const v = addonPricesCop()[assetId];
  return typeof v === 'number' ? v : null;
}
