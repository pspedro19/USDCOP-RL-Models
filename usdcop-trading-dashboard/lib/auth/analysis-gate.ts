/**
 * Per-asset + freshness gate for analysis content (CTR-RBAC-001 R3).
 *
 * WHY THIS EXISTS: the gate was implemented once, inline, in `/api/data/[...path]` — a route
 * the product does not actually call. The analysis UI (`hooks/useWeeklyAnalysis.ts`) fetches
 * `/api/analysis/weeks` and `/api/analysis/week/<year>/<week>?asset=…`, and those routes
 * applied neither the asset scoping nor the plan delay. A `free` user (whose plan covers
 * USD/COP only) could read the full, undelayed Gold / BTC / S&P weekly analysis simply by
 * changing a query parameter, so every per-asset add-on was being given away.
 *
 * The permission matrix in `middleware.ts` cannot close this: it answers "may this role reach
 * this route", never "does this PLAN include this asset". Entitlements are per-request data,
 * so the check belongs in the handler — which is exactly where this helper is used.
 *
 * Fails closed: `getEntitlements` already degrades to the free plan on a DB error, and an
 * unknown asset id resolves to the default rather than bypassing the check.
 */
import { NextResponse } from 'next/server';

import { getEntitlements, isFresherThanAllowed } from '@/lib/auth/entitlements';

/** Assets that can appear in a request; anything else collapses to the default. */
const KNOWN_ASSETS = ['usdcop', 'xauusd', 'btcusdt', 'spx500'] as const;
export const DEFAULT_ASSET = 'usdcop';

/** Normalize an arbitrary `?asset=` value to a known id (never trust the query string). */
export function resolveAssetId(raw: string | null | undefined): string {
  if (!raw) return DEFAULT_ASSET;
  const v = String(raw).toLowerCase();
  return (KNOWN_ASSETS as readonly string[]).includes(v) ? v : DEFAULT_ASSET;
}

/**
 * Enforce the plan on one analysis read.
 *
 * @param userId   identity stamped on the request by the middleware (`x-user-id`)
 * @param asset    raw asset id from the request (normalized internally)
 * @param fileName artifact name used for the freshness check; omit when the response
 *                 carries no single dated artifact (e.g. an index listing)
 * @returns a 403 `NextResponse` to return as-is, or `null` when the caller may proceed.
 */
export async function gateAnalysisAccess(
  userId: string | null | undefined,
  asset: string | null | undefined,
  fileName?: string,
): Promise<NextResponse | null> {
  const assetId = resolveAssetId(asset);
  const ent = await getEntitlements(userId);

  if (!ent.assets.includes(assetId)) {
    return NextResponse.json(
      { error: 'asset not in plan', asset: assetId, plan: ent.plan, upgrade: true },
      { status: 403 },
    );
  }
  if (fileName && isFresherThanAllowed(fileName, ent.analysis_delay_days)) {
    return NextResponse.json(
      {
        error: 'content delayed for your plan',
        delay_days: ent.analysis_delay_days,
        plan: ent.plan,
        upgrade: true,
      },
      { status: 403 },
    );
  }
  return null;
}

/**
 * Drop entries a plan may not see yet from an index listing, so the week picker never
 * offers a week the reader would get a 403 on. Entries are matched by their artifact name.
 */
export async function filterDelayedWeeks<T>(
  userId: string | null | undefined,
  weeks: readonly T[],
  nameOf: (entry: T) => string,
): Promise<T[]> {
  const ent = await getEntitlements(userId);
  if (ent.analysis_delay_days <= 0) return [...weeks];
  return weeks.filter((w) => !isFresherThanAllowed(nameOf(w), ent.analysis_delay_days));
}
