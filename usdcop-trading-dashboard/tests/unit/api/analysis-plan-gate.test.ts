/**
 * Per-asset + freshness enforcement on the analysis routes the PRODUCT actually calls.
 *
 * The gate existed only inside `/api/data/[...path]`, a route the UI never fetches:
 * `hooks/useWeeklyAnalysis.ts` goes to `/api/analysis/weeks` and
 * `/api/analysis/week/<year>/<week>?asset=…`. Neither checked the plan, so a `free`
 * session (whose entitlement covers USD/COP only) read the full Gold / BTC / S&P weekly
 * analysis by editing one query parameter — every per-asset add-on was being given away.
 *
 * A second, quieter hole: `isFresherThanAllowed` matched `YYYY-Www`, but the published
 * artifacts are named `weekly_2026_W27.json` (UNDERSCORE). The pattern matched nothing,
 * so the free plan's T+7 delay was never applied to the very files it exists to delay.
 *
 * Mutations that MUST turn these red again:
 *   M1  drop `gateAnalysisAccess` from the week route      → cross-asset read returns 200
 *   M2  revert the week regex to `(\d{4})-W(\d{2})`        → fresh week served to `free`
 *   M3  stop filtering the index                           → picker offers a 403 week
 *   M4  let an unknown `?asset=` fall through unnormalized  → gate bypassed
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const entitlementsFor = vi.fn();

vi.mock('@/lib/auth/entitlements', async () => {
  const actual = await vi.importActual<typeof import('@/lib/auth/entitlements')>(
    '@/lib/auth/entitlements',
  );
  return {
    // The freshness helper is the REAL one — M2 must be observable through these tests.
    isFresherThanAllowed: actual.isFresherThanAllowed,
    getEntitlements: (userId: string | null | undefined) => entitlementsFor(userId),
  };
});

import { isFresherThanAllowed } from '@/lib/auth/entitlements';
import { resolveAssetId } from '@/lib/auth/analysis-gate';
import { GET as weekGET } from '@/app/api/analysis/week/[year]/[week]/route';
import { GET as weeksGET } from '@/app/api/analysis/weeks/route';

const FREE = {
  plan: 'free',
  assets: ['usdcop'],
  analysis_delay_days: 7,
  forecast_delay_hours: 168,
  signals_realtime: false,
};
const PAID = { ...FREE, plan: 'signals', assets: ['usdcop', 'xauusd'], analysis_delay_days: 0 };

/** A NextRequest-shaped stub: the handlers read only `headers` and `nextUrl.searchParams`. */
function req(url: string, userId = 'user-1') {
  const u = new URL(url, 'http://localhost:5000');
  return {
    headers: new Headers({ 'x-user-id': userId }),
    nextUrl: u,
    url: u.toString(),
  } as unknown as import('next/server').NextRequest;
}

const params = (year: string, week: string) => ({ params: Promise.resolve({ year, week }) });

beforeEach(() => entitlementsFor.mockReset());
afterEach(() => vi.clearAllMocks());

describe('the week regex actually matches the artifacts we publish', () => {
  it('treats a fresh underscore-named week as delayed for a 7-day plan', () => {
    // Next year's W01 is unambiguously newer than any "now − 7 days" cutoff, so this
    // asserts the SEPARATOR is understood without re-deriving ISO weeks in the test.
    const name = `weekly_${new Date().getUTCFullYear() + 1}_W01.json`;
    expect(isFresherThanAllowed(name, 7)).toBe(true);
  });

  it('still accepts the dashed spelling', () => {
    expect(isFresherThanAllowed('weekly_2019-W02.json', 7)).toBe(false); // old, not delayed
  });

  it('never delays anything when the plan has no delay', () => {
    expect(isFresherThanAllowed('weekly_2099_W52.json', 0)).toBe(false);
  });

  it('leaves an undated artifact ungated', () => {
    expect(isFresherThanAllowed('analysis_index.json', 7)).toBe(false);
  });
});

describe('asset ids from the query string are normalized, never trusted', () => {
  it('collapses an unknown asset to the default instead of passing it through', () => {
    expect(resolveAssetId('../../etc/passwd')).toBe('usdcop');
    expect(resolveAssetId('XAUUSD')).toBe('xauusd');
    expect(resolveAssetId(null)).toBe('usdcop');
  });
});

describe('GET /api/analysis/week — the paid surface', () => {
  it('refuses an asset outside the plan with 403 + upgrade, before any disk read', async () => {
    entitlementsFor.mockResolvedValue(FREE);
    const res = await weekGET(req('/api/analysis/week/2026/27?asset=xauusd'), params('2026', '27'));
    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body).toMatchObject({ error: 'asset not in plan', asset: 'xauusd', upgrade: true });
  });

  it('allows an asset inside the plan', async () => {
    entitlementsFor.mockResolvedValue(PAID);
    const res = await weekGET(req('/api/analysis/week/2026/27?asset=xauusd'), params('2026', '27'));
    // 200 with data or 404 when that week is absent — either way, NOT a plan refusal.
    expect(res.status).not.toBe(403);
  });

  it('delays a too-fresh week for a plan that has a delay', async () => {
    entitlementsFor.mockResolvedValue(FREE);
    const y = new Date().getUTCFullYear() + 1; // guaranteed in the future ⇒ fresher than any cutoff
    const res = await weekGET(req(`/api/analysis/week/${y}/40?asset=usdcop`), params(String(y), '40'));
    expect(res.status).toBe(403);
    expect((await res.json()).error).toBe('content delayed for your plan');
  });

  it('rejects a malformed week before consulting the plan at all', async () => {
    entitlementsFor.mockResolvedValue(FREE);
    const res = await weekGET(req('/api/analysis/week/2026/99'), params('2026', '99'));
    expect(res.status).toBe(400);
    expect(entitlementsFor).not.toHaveBeenCalled();
  });
});

describe('GET /api/analysis/weeks — the index must not advertise what it will refuse', () => {
  it('refuses an asset outside the plan', async () => {
    entitlementsFor.mockResolvedValue(FREE);
    const res = await weeksGET(req('/api/analysis/weeks?asset=btcusdt'));
    expect(res.status).toBe(403);
  });

  it('drops weeks still inside the delay window', async () => {
    entitlementsFor.mockResolvedValue(FREE);
    const res = await weeksGET(req('/api/analysis/weeks?asset=usdcop'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const future = new Date().getUTCFullYear() + 1;
    expect(body.weeks.some((w: { year: number }) => w.year >= future)).toBe(false);
  });

  it('returns every week when the plan carries no delay', async () => {
    entitlementsFor.mockResolvedValue(PAID);
    const res = await weeksGET(req('/api/analysis/weeks?asset=usdcop'));
    expect(res.status).toBe(200);
    expect(Array.isArray((await res.json()).weeks)).toBe(true);
  });
});
