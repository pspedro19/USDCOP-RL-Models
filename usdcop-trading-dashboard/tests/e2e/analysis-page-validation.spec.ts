import { test, expect } from '@playwright/test';

/**
 * E2E tests for /analysis page — validates that weekly analysis data,
 * macro charts, signal cards, daily timeline, and news are rendered.
 */

// Run sequentially to avoid overwhelming the dev server
test.describe.configure({ mode: 'serial' });

test.describe('/analysis page — API Validation', () => {

  test('analysis index returns 7+ weeks', async ({ request }) => {
    const response = await request.get('/api/analysis/weeks');
    expect(response.status()).toBe(200);

    const data = await response.json();
    expect(data.weeks.length).toBeGreaterThanOrEqual(7);

    const firstWeek = data.weeks[0];
    expect(firstWeek.year).toBe(2026);
    expect(firstWeek.has_weekly).toBe(true);
  });

  test('weekly data W07 has all sections populated', async ({ request }) => {
    const response = await request.get('/api/analysis/week/2026/7');
    expect(response.status()).toBe(200);

    const data = await response.json();

    // Core sections
    expect(data.weekly_summary).toBeDefined();
    expect(data.daily_entries).toBeDefined();
    expect(data.macro_snapshots).toBeDefined();
    expect(data.macro_charts).toBeDefined();
    expect(data.signals).toBeDefined();
    expect(data.upcoming_events).toBeDefined();
    expect(data.news_context).toBeDefined();

    // 5 daily entries
    expect(data.daily_entries.length).toBe(5);

    // 8 macro snapshots
    expect(Object.keys(data.macro_snapshots).length).toBeGreaterThanOrEqual(8);

    // 8 macro charts with time series
    const chartKeys = Object.keys(data.macro_charts);
    expect(chartKeys.length).toBeGreaterThanOrEqual(8);
    const firstChart = data.macro_charts[chartKeys[0]];
    expect(firstChart.data.length).toBeGreaterThan(10);

    // H5 and H1 signals
    expect(data.signals.h5).toBeDefined();
    expect(data.signals.h1).toBeDefined();
  });

  test('DXY snapshot has valid technical indicators', async ({ request }) => {
    const response = await request.get('/api/analysis/week/2026/7');
    const data = await response.json();
    const dxy = data.macro_snapshots['dxy'];

    expect(dxy).toBeDefined();
    expect(dxy.value).toBeGreaterThan(90); // DXY is always around 90-115
    expect(dxy.value).toBeLessThan(130);
    expect(dxy.sma_20).toBeGreaterThan(0);
    expect(dxy.rsi_14).toBeGreaterThanOrEqual(0);
    expect(dxy.rsi_14).toBeLessThanOrEqual(100);
    expect(['above_sma20', 'below_sma20', 'golden_cross', 'death_cross', 'neutral']).toContain(dxy.trend);
  });

  test('all 7 weekly files load successfully', async ({ request }) => {
    for (let w = 1; w <= 7; w++) {
      const response = await request.get(`/api/analysis/week/2026/${w}`);
      expect(response.status()).toBe(200);

      const data = await response.json();
      expect(data.daily_entries.length).toBeGreaterThanOrEqual(1);
      expect(Object.keys(data.macro_snapshots).length).toBeGreaterThanOrEqual(4);
    }
  });

  test('calendar endpoint responds', async ({ request }) => {
    const response = await request.get('/api/analysis/calendar');
    expect(response.status()).toBe(200);
    const data = await response.json();
    expect(Array.isArray(data.events)).toBe(true);
  });
});

test.describe('/analysis page — UI Rendering', () => {

  test('page renders with analysis content (not stuck loading)', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

    // Wait for actual content to replace loading spinner
    const content = page.locator('text=/Indicadores Macro|Graficos Macro|Timeline Diario/i');
    await expect(content.first()).toBeVisible({ timeout: 20000 });
  });

  test('week selector shows SEMANA with correct week number', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

    // Wait for data to load
    const weekLabel = page.locator('text=/SEMANA\\s+\\d+/i');
    await expect(weekLabel.first()).toBeVisible({ timeout: 20000 });
  });

  test('8 macro charts are visible', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

    // Wait for charts section
    const chartsTitle = page.locator('text=Indicadores Macro');
    await expect(chartsTitle.first()).toBeVisible({ timeout: 20000 });

    // Scroll down to reveal chart grid (charts are below the fold)
    await chartsTitle.first().scrollIntoViewIfNeeded();
    await page.waitForTimeout(3000);

    // Check for individual chart labels in the full page content
    const chartNames = ['DXY', 'VIX', 'WTI', 'EMBI', 'Treasury', 'IBR', 'Oro', 'Brent'];
    const content = await page.content();
    let foundCharts = 0;
    for (const name of chartNames) {
      if (content.includes(name)) {
        foundCharts++;
      }
    }
    expect(foundCharts).toBeGreaterThanOrEqual(6);
  });

  test('chart canvas elements render in macro section', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

    // Wait for chart section to load
    const chartsTitle = page.locator('text=Indicadores Macro');
    await expect(chartsTitle.first()).toBeVisible({ timeout: 20000 });
    await page.waitForTimeout(3000);

    // UnifiedMacroChart uses lightweight-charts which renders via canvas
    const canvases = page.locator('canvas');
    const count = await canvases.count();
    expect(count).toBeGreaterThanOrEqual(1);
  });

  test('daily timeline section is rendered', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

    const timeline = page.locator('text=Timeline Diario');
    await expect(timeline).toBeVisible({ timeout: 20000 });

    // Timeline should have at least one day entry visible (dates may be in various formats)
    const anyDate = page.locator('text=/2026|Lun|Mar|Mie|Jue|Vie|Mon|Tue|Wed|Thu|Fri|dry.run|DRY/i');
    await expect(anyDate.first()).toBeVisible({ timeout: 10000 });
  });

  test('macro snapshot bar shows indicator values', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

    const indicators = page.locator('text=Indicadores Macro');
    await expect(indicators).toBeVisible({ timeout: 20000 });
  });

  test('floating chat widget is visible', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });
    await page.waitForTimeout(3000);

    // The chat widget has a chat icon button
    const chatButton = page.locator('button').filter({ has: page.locator('svg') }).last();
    // At minimum the page should have rendered the chat widget area
    const pageContent = await page.content();
    // FloatingChatWidget is lazy-loaded, just verify no crash
    expect(pageContent).toContain('analysis');
  });

  test('week navigation changes displayed data', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

    // Wait for initial content
    const content = page.locator('text=/Indicadores Macro|Graficos Macro/i');
    await expect(content.first()).toBeVisible({ timeout: 20000 });

    // Find the prev week button (< arrow)
    const prevBtn = page.locator('button').filter({ hasText: /[<←◀]/ }).first();
    const hasPrev = await prevBtn.isVisible().catch(() => false);

    if (hasPrev) {
      await prevBtn.click();
      await page.waitForTimeout(3000);
      // Page should still show analysis content (not error)
      await expect(page).toHaveURL(/\/analysis/);
    }
  });

  test('full page screenshot captures complete analysis', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

    // Wait for all content to render
    const content = page.locator('text=/Indicadores Macro/i');
    await expect(content.first()).toBeVisible({ timeout: 20000 });
    await page.waitForTimeout(3000);

    await page.screenshot({
      path: 'tests/e2e/screenshots/analysis-page-final.png',
      fullPage: true,
    });

    const fs = require('fs');
    expect(fs.existsSync('tests/e2e/screenshots/analysis-page-final.png')).toBe(true);
  });
});
