/**
 * Plan price SSOT (COP). Single source consumed by the Wompi provider (checkout
 * amount) AND the admin revenue aggregator (MRR/ARR). Override the placeholders
 * via BILLING_PRICES_COP env (JSON, cents): {"signals":<cents>,"auto":<cents>}.
 *
 * Prices are a business decision (spec §B.5) — these are placeholders until set.
 */
import type { PlanId } from '@/lib/contracts/rbac.contract';

/** Monthly price per plan, in COP CENTS. */
export const DEFAULT_PRICES_COP_CENTS: Record<string, number> = {
  free: 0,
  signals: 990_000_00 / 10, // 99.000 COP/mes
  auto: 2_990_000_00 / 10, // 299.000 COP/mes
};

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
};

/** Plans that generate recurring revenue. */
export const PAID_PLANS: PlanId[] = ['signals', 'auto'];

// ── per-asset add-on prices (SSOT) ─────────────────────────────────────────────

/**
 * Per-asset add-on monthly price, in WHOLE COP. Single source consumed by the
 * catalog service (/api/catalog), Pricing and Cart. Only assets with a real
 * published price appear here — everything else resolves to `null`, which the UI
 * renders as an honest "—" (never invent a price). Coming-soon assets stay out.
 * Override via BILLING_ADDON_PRICES_COP env (JSON, whole COP): {"xauusd":39000}.
 */
export const DEFAULT_ADDON_PRICES_COP: Record<string, number> = {
  xauusd: 39_000, // Gold add-on
  btcusdt: 39_000, // Bitcoin add-on
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
