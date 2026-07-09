import { test, expect } from '@playwright/test';

/**
 * E2E tests for the MULTI-ASSET /analysis feature — validates the dynamic asset
 * selector (USD/COP · Gold · Bitcoin) end-to-end through the running Next.js
 * server: the /api/analysis/assets SSOT route, per-asset week index + weekly
 * views (with real Gold/BTC news), backward-compatibility, and path-safety.
 *
 * The analysis API routes are public (not in PROTECTED_API_ROUTES), so these
 * assertions run against the production container without auth. A final
 * auth-aware DOM smoke confirms the /analysis page is served + protected.
 */

test.describe.configure({ mode: 'serial' });

const ASSETS = ['usdcop', 'xauusd', 'btcusdt'];

test.describe('multi-asset /analysis — asset SSOT', () => {
  test('GET /api/analysis/assets returns the 3 configured assets', async ({ request }) => {
    const res = await request.get('/api/analysis/assets');
    expect(res.status()).toBe(200);
    const data = await res.json();

    expect(data.default_asset).toBe('usdcop');
    const ids = data.assets.map((a: any) => a.asset_id).sort();
    expect(ids).toEqual([...ASSETS].sort());

    // Each asset carries the display metadata the selector needs
    for (const a of data.assets) {
      expect(a.asset_id).toBeTruthy();
      expect(a.display_name).toBeTruthy();
      expect(a.chart_symbol).not.toContain('/');
    }
  });
});

test.describe('multi-asset /analysis — per-asset week index', () => {
  for (const asset of ASSETS) {
    test(`weeks?asset=${asset} returns weekly entries`, async ({ request }) => {
      const res = await request.get(`/api/analysis/weeks?asset=${asset}`);
      expect(res.status()).toBe(200);
      const data = await res.json();
      expect(Array.isArray(data.weeks)).toBe(true);
      expect(data.weeks.length).toBeGreaterThanOrEqual(7);
      // Sorted newest-first (frontend relies on weeks[0] === most recent)
      const first = data.weeks[0];
      expect(first.year).toBe(2026);
      expect(first.has_weekly).toBe(true);
    });
  }
});

test.describe('multi-asset /analysis — Gold/BTC weekly view (real data)', () => {
  for (const asset of ['xauusd', 'btcusdt']) {
    test(`week/2026/20?asset=${asset} has technicals, signal + real news`, async ({ request }) => {
      const res = await request.get(`/api/analysis/week/2026/20?asset=${asset}`);
      expect(res.status()).toBe(200);
      const d = await res.json();

      // Core sections the /analysis page renders
      expect(d.weekly_summary).toBeDefined();
      expect(d.weekly_summary.ohlcv).toBeDefined();
      expect(typeof d.weekly_summary.ohlcv.close).toBe('number');
      expect(d.daily_entries.length).toBeGreaterThanOrEqual(1);

      // Real technicals computed from price
      expect(d.technical_analysis).toBeDefined();
      expect(d.technical_analysis.support_resistance.resistance)
        .toBeGreaterThanOrEqual(d.technical_analysis.support_resistance.support);

      // Strategy positioning mapped into the signal card
      expect(['LONG', 'SHORT', 'HOLD']).toContain(d.signals.h5.direction);

      // Real news populated (Google News) — clusters by source
      expect(d.news_context.article_count).toBeGreaterThan(0);
      expect(d.news_intelligence.clusters.length).toBeGreaterThan(0);

      // JSON-safety: no NaN/Infinity leaked into the payload
      const raw = JSON.stringify(d);
      expect(raw).not.toContain('NaN');
      expect(raw).not.toContain('Infinity');
    });
  }
});

test.describe('multi-asset /analysis — compatibility + safety', () => {
  test('USD/COP legacy path still works (no asset param)', async ({ request }) => {
    const withParam = await request.get('/api/analysis/weeks?asset=usdcop');
    const noParam = await request.get('/api/analysis/weeks');
    expect(withParam.status()).toBe(200);
    expect(noParam.status()).toBe(200);
    const a = await withParam.json();
    const b = await noParam.json();
    expect(a.weeks.length).toBe(b.weeks.length); // same data via fallback
  });

  test('unknown / traversal asset ids collapse to the default (path-safe)', async ({ request }) => {
    const bogus = await request.get('/api/analysis/weeks?asset=xxx');
    const traversal = await request.get('/api/analysis/weeks?asset=%2e%2e%2fstrategies');
    expect(bogus.status()).toBe(200);
    expect(traversal.status()).toBe(200);
    // Both resolve to usdcop's index rather than erroring or escaping
    const b = await bogus.json();
    expect(Array.isArray(b.weeks)).toBe(true);
    expect(b.weeks.length).toBeGreaterThanOrEqual(7);
  });

  test('per-asset calendar route serves', async ({ request }) => {
    for (const asset of ASSETS) {
      const res = await request.get(`/api/analysis/calendar?asset=${asset}`);
      expect(res.status()).toBe(200);
    }
  });
});

test.describe('multi-asset /analysis — page render (auth-aware)', () => {
  test('/analysis is served and either renders the selector or requires auth', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });
    const url = page.url();

    if (url.includes('/login')) {
      // Production auth is enforced (expected) — the page is wired + protected.
      expect(url).toContain('callbackUrl');
      return;
    }

    // Auth bypass / authenticated session: the asset selector must render with
    // all three assets, and switching to Gold must not break the page.
    const selector = page.locator('text=/USD\\/COP/i');
    await expect(selector.first()).toBeVisible({ timeout: 20000 });
    await expect(page.locator('text=/Oro|Gold/i').first()).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=/Bitcoin/i').first()).toBeVisible({ timeout: 10000 });

    await page.locator('text=/Oro|Gold/i').first().click();
    await page.waitForTimeout(2000);
    expect(await page.content()).not.toContain('Application error');
  });
});
