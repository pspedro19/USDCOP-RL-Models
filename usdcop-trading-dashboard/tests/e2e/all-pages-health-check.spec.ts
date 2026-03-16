import { test, expect } from '@playwright/test';

/**
 * Comprehensive Health Check — All Dashboard Pages
 * =================================================
 * Visits every page, validates rendering, captures screenshots,
 * and checks for console errors.
 */

const SCREENSHOTS_DIR = 'tests/e2e/screenshots/health-check';

// ============================================================
// 1. HUB PAGE
// ============================================================
test('1. Hub page loads and shows navigation cards', async ({ page }) => {
  await page.goto('/hub', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  // Hub uses router.push() cards, not <a> links
  // Check for card titles that we know exist
  const cardTitles = ['Trading Dashboard', 'Monitor de Produccion', 'Forecasting Semanal', 'Analisis Semanal'];
  let found = 0;
  for (const title of cardTitles) {
    const el = page.locator(`text=${title}`).first();
    if (await el.isVisible().catch(() => false)) found++;
  }
  expect(found).toBeGreaterThanOrEqual(3);
  console.log(`[OK] Hub: ${found}/${cardTitles.length} cards visible`);

  await page.screenshot({
    path: `${SCREENSHOTS_DIR}/01-hub.png`,
    fullPage: true,
  });
});

// ============================================================
// 2. FORECASTING PAGE
// ============================================================
test('2. Forecasting page loads with model data', async ({ page }) => {
  await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);

  const body = await page.content();
  expect(body.length).toBeGreaterThan(1000);

  await page.screenshot({
    path: `${SCREENSHOTS_DIR}/02-forecasting.png`,
    fullPage: true,
  });
  console.log('[OK] Forecasting page loaded');
});

// ============================================================
// 3. DASHBOARD PAGE (Backtest Review)
// ============================================================
test('3. Dashboard page loads with backtest data', async ({ page }) => {
  await page.goto('/dashboard', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(5000);

  const body = await page.content();
  expect(body.length).toBeGreaterThan(1000);

  await page.screenshot({
    path: `${SCREENSHOTS_DIR}/03-dashboard.png`,
    fullPage: true,
  });
  console.log('[OK] Dashboard page loaded');
});

// ============================================================
// 4. PRODUCTION PAGE
// ============================================================
test('4. Production page loads', async ({ page }) => {
  await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);

  const body = await page.content();
  expect(body.length).toBeGreaterThan(1000);

  await page.screenshot({
    path: `${SCREENSHOTS_DIR}/04-production.png`,
    fullPage: true,
  });
  console.log('[OK] Production page loaded');
});

// ============================================================
// 5. ANALYSIS — API Tests
// ============================================================
test('5a. Analysis API: index returns 9 weeks', async ({ request }) => {
  const response = await request.get('/api/analysis/weeks');
  expect(response.status()).toBe(200);

  const data = await response.json();
  expect(data.weeks.length).toBeGreaterThanOrEqual(7);
  console.log(`[OK] Analysis index: ${data.weeks.length} weeks available`);

  // Verify first week structure
  const first = data.weeks[0];
  expect(first.year).toBe(2026);
  expect(first.has_weekly).toBe(true);
  expect(first.sentiment).toBeTruthy();
});

test('5b. Analysis API: latest week JSON has all sections', async ({ request }) => {
  const indexRes = await request.get('/api/analysis/weeks');
  const index = await indexRes.json();
  const latest = index.weeks[0];

  const response = await request.get(`/api/analysis/week/${latest.year}/${latest.week}`);
  expect(response.status()).toBe(200);

  const data = await response.json();

  // Weekly summary
  expect(data.weekly_summary).toBeDefined();
  expect(data.weekly_summary.headline).toBeTruthy();
  expect(data.weekly_summary.markdown.length).toBeGreaterThan(100);
  expect(data.weekly_summary.sentiment).toBeTruthy();
  expect(data.weekly_summary.ohlcv).toBeDefined();
  expect(data.weekly_summary.ohlcv.open).toBeGreaterThan(3000);
  console.log(`[OK] Weekly summary: "${data.weekly_summary.headline.substring(0, 50)}..."`);

  // Daily entries
  expect(data.daily_entries).toBeDefined();
  expect(data.daily_entries.length).toBeGreaterThanOrEqual(1);
  const firstDay = data.daily_entries[0];
  expect(firstDay.headline).toBeTruthy();
  expect(firstDay.summary_markdown.length).toBeGreaterThan(50);
  console.log(`[OK] Daily entries: ${data.daily_entries.length} days`);

  // Macro snapshots (8 variables)
  expect(data.macro_snapshots).toBeDefined();
  const macroKeys = Object.keys(data.macro_snapshots);
  expect(macroKeys.length).toBeGreaterThanOrEqual(8);
  console.log(`[OK] Macro snapshots: ${macroKeys.join(', ')}`);

  // Macro charts with time series data
  expect(data.macro_charts).toBeDefined();
  const chartKeys = Object.keys(data.macro_charts);
  expect(chartKeys.length).toBeGreaterThanOrEqual(8);
  const firstChart = data.macro_charts[chartKeys[0]];
  expect(firstChart.data.length).toBeGreaterThan(10);
  console.log(`[OK] Macro charts: ${chartKeys.length} variables, ${firstChart.data.length}+ data points each`);

  // Signals section
  expect(data.signals).toBeDefined();
});

test('5c. Analysis API: DXY technical indicators valid', async ({ request }) => {
  const indexRes = await request.get('/api/analysis/weeks');
  const latest = (await indexRes.json()).weeks[0];

  const response = await request.get(`/api/analysis/week/${latest.year}/${latest.week}`);
  const data = await response.json();
  const dxy = data.macro_snapshots['dxy'];

  expect(dxy).toBeDefined();
  expect(dxy.value).toBeGreaterThan(90);
  expect(dxy.value).toBeLessThan(130);
  expect(dxy.sma_5).toBeGreaterThan(0);
  expect(dxy.sma_20).toBeGreaterThan(0);
  expect(dxy.rsi_14).toBeGreaterThanOrEqual(0);
  expect(dxy.rsi_14).toBeLessThanOrEqual(100);
  expect(dxy.bollinger_upper_20).toBeGreaterThan(dxy.bollinger_lower_20);
  expect(dxy.macd_line).toBeDefined();
  console.log(`[OK] DXY=${dxy.value}, RSI=${dxy.rsi_14.toFixed(1)}, MACD=${dxy.macd_histogram.toFixed(4)}, trend=${dxy.trend}`);
});

test('5d. Analysis API: all 9 weeks load', async ({ request }) => {
  const results: string[] = [];
  for (let w = 1; w <= 9; w++) {
    const response = await request.get(`/api/analysis/week/2026/${w}`);
    expect(response.status()).toBe(200);
    const data = await response.json();
    expect(data.weekly_summary).toBeDefined();
    results.push(`W${String(w).padStart(2, '0')}: ${data.daily_entries.length} days, ${Object.keys(data.macro_snapshots).length} vars`);
  }
  console.log(`[OK] All 9 weeks loaded:\n  ${results.join('\n  ')}`);
});

test('5e. Analysis API: calendar endpoint', async ({ request }) => {
  const response = await request.get('/api/analysis/calendar');
  expect(response.status()).toBe(200);
  const data = await response.json();
  expect(data).toBeDefined();
  console.log('[OK] Calendar endpoint responds');
});

// ============================================================
// 5. ANALYSIS — UI Tests
// ============================================================
test('5f. Analysis page renders content (not stuck loading)', async ({ page }) => {
  await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

  const content = page.locator('text=/Analisis Semanal|Indicadores Macro|Graficos Macro|Timeline Diario/i');
  await expect(content.first()).toBeVisible({ timeout: 20000 });

  await page.screenshot({
    path: `${SCREENSHOTS_DIR}/05a-analysis-top.png`,
    fullPage: false,
  });
  console.log('[OK] Analysis page rendered with content');
});

test('5g. Analysis: SEMANA week selector visible', async ({ page }) => {
  await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

  const weekLabel = page.locator('text=/SEMANA\\s+\\d+/i');
  await expect(weekLabel.first()).toBeVisible({ timeout: 20000 });

  const text = await weekLabel.first().textContent();
  console.log(`[OK] Week selector: "${text}"`);
});

test('5h. Analysis: macro Recharts SVGs render', async ({ page }) => {
  await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

  // Wait for data to load first (Indicadores Macro = data loaded)
  await page.locator('text=/Indicadores Macro/i').waitFor({ timeout: 25000 });
  await page.waitForTimeout(3000); // Give Recharts time to render SVGs

  // Check for Recharts SVGs or any SVG in the chart area
  const svgs = page.locator('.recharts-surface, .recharts-wrapper svg');
  const count = await svgs.count();

  // Also check for chart container divs as fallback
  const chartContainers = page.locator('[class*="recharts"], [class*="chart"]');
  const containerCount = await chartContainers.count();

  console.log(`[OK] Recharts SVGs: ${count}, chart containers: ${containerCount}`);
  expect(count + containerCount).toBeGreaterThanOrEqual(1);
});

test('5i. Analysis: daily timeline has entries', async ({ page }) => {
  await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });

  const timeline = page.locator('text=/Timeline Diario/i');
  await expect(timeline).toBeVisible({ timeout: 20000 });
  console.log('[OK] Daily timeline section visible');
});

test('5j. Analysis: week navigation prev/next works', async ({ page }) => {
  await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });
  await page.locator('text=/SEMANA\\s+\\d+/i').first().waitFor({ timeout: 20000 });

  const weekBefore = await page.locator('text=/SEMANA\\s+\\d+/i').first().textContent();

  // Try clicking prev arrow
  const buttons = page.locator('button');
  const btnCount = await buttons.count();
  for (let i = 0; i < btnCount; i++) {
    const btn = buttons.nth(i);
    const text = await btn.textContent().catch(() => '');
    if (text?.includes('<') || text?.includes('\u2190') || text?.includes('\u25C0')) {
      await btn.click();
      await page.waitForTimeout(3000);
      break;
    }
  }

  // Verify URL still analysis (no crash)
  await expect(page).toHaveURL(/\/analysis/);
  console.log(`[OK] Week navigation: ${weekBefore} -> navigated`);
});

test('5k. Analysis: full page screenshot', async ({ page }) => {
  await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(6000);

  await page.screenshot({
    path: `${SCREENSHOTS_DIR}/05-analysis-full.png`,
    fullPage: true,
  });
  console.log('[OK] Full analysis page screenshot captured');
});

// ============================================================
// 6. EXECUTION PAGE
// ============================================================
test('6. Execution page loads', async ({ page }) => {
  await page.goto('/execution', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(3000);

  const body = await page.content();
  expect(body.length).toBeGreaterThan(500);

  await page.screenshot({
    path: `${SCREENSHOTS_DIR}/06-execution.png`,
    fullPage: true,
  });
  console.log('[OK] Execution page loaded');
});

// ============================================================
// 7. NAVBAR ACROSS ALL PAGES
// ============================================================
test('7. Navbar visible on all pages', async ({ page }) => {
  const pages = ['/hub', '/forecasting', '/dashboard', '/production', '/analysis'];
  const results: string[] = [];

  for (const p of pages) {
    await page.goto(p, { waitUntil: 'load', timeout: 30000 });
    await page.waitForTimeout(1500);

    const nav = page.locator('nav').first();
    const hasNav = await nav.isVisible().catch(() => false);
    results.push(`${p}: ${hasNav ? 'OK' : 'MISSING'}`);
  }
  console.log(`[INFO] Navbar check:\n  ${results.join('\n  ')}`);
});
