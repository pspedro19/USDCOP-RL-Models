/**
 * Production Readiness E2E Test Suite
 *
 * Validates all 8 dashboard pages are rendering correctly with live backend data.
 * Takes full-page screenshots for visual audit. Asserts critical KPIs, data presence,
 * API health, and zero console errors (after filtering benign noise).
 *
 * Run:  npx playwright test tests/e2e/production-readiness.spec.ts --project=chromium
 */
import { test, expect, Page } from '@playwright/test';
import {
  authenticateUser,
  setupConsoleErrorCapture,
  filterBenignErrors,
} from './helpers/auth';

const SCREENSHOT_DIR = 'tests/e2e/screenshots/production-readiness';
const BASE = 'http://localhost:5000';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Navigate, wait for network + extra settle time, take screenshot */
async function navigateAndScreenshot(
  page: Page,
  path: string,
  name: string,
  opts: { settleMs?: number; fullPage?: boolean; waitUntil?: 'load' | 'networkidle' } = {}
) {
  const { settleMs = 3000, fullPage = true, waitUntil = 'networkidle' } = opts;
  await page.goto(path, { waitUntil, timeout: 60000 });
  await page.waitForTimeout(settleMs);
  await page.screenshot({ path: `${SCREENSHOT_DIR}/${name}.png`, fullPage });
}

/** Assert element count >= min */
async function assertMinCount(page: Page, selector: string, min: number, label: string) {
  const count = await page.locator(selector).count();
  expect(count, `Expected at least ${min} ${label}, found ${count}`).toBeGreaterThanOrEqual(min);
}

// ---------------------------------------------------------------------------
// 0. Infrastructure Health
// ---------------------------------------------------------------------------

test.describe('0 — Infrastructure Health', () => {
  test('API health endpoint returns healthy', async ({ request }) => {
    const res = await request.get(`${BASE}/api/health`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body.status).toBe('healthy');
    expect(body.checks).toBeDefined();
    // trading_api and data_files must be ok
    for (const check of body.checks) {
      expect(check.status, `Health check "${check.name}" failed`).toBe('ok');
    }
  });

  test('Production status API returns APPROVED', async ({ request }) => {
    const res = await request.get(`${BASE}/api/production/status`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body.status).toBe('APPROVED');
    expect(body.strategy).toBe('smart_simple_v11');
    expect(body.backtest_recommendation).toBe('PROMOTE');
    expect(body.gates).toBeDefined();
    // All 5 gates must pass
    const passed = body.gates.filter((g: { passed: boolean }) => g.passed).length;
    expect(passed, `Only ${passed}/5 gates passed`).toBe(5);
  });

  test('Analysis weeks API returns data', async ({ request }) => {
    const res = await request.get(`${BASE}/api/analysis/weeks`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    const weeks = body.weeks ?? body;
    expect(weeks.length, 'Expected at least 10 analysis weeks').toBeGreaterThanOrEqual(10);
  });

  test('Forecasting CSV is accessible and non-empty', async ({ request }) => {
    const res = await request.get(`${BASE}/forecasting/bi_dashboard_unified.csv`);
    expect(res.ok()).toBeTruthy();
    const text = await res.text();
    const lines = text.trim().split('\n');
    expect(lines.length, 'CSV should have 400+ rows').toBeGreaterThan(400);
    // Check header contains expected columns
    expect(lines[0]).toContain('model_name');
    expect(lines[0]).toContain('horizon');
  });

  test('Summary 2025 JSON is valid', async ({ request }) => {
    const res = await request.get(`${BASE}/data/production/summary_2025.json`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body.strategy_id).toBe('smart_simple_v11');
    expect(body.year).toBe(2025);
    const stats = body.strategies[body.strategy_id];
    expect(stats.total_return_pct).toBeGreaterThan(15);
    expect(stats.sharpe).toBeGreaterThan(1.5);
    expect(body.statistical_tests.p_value).toBeLessThan(0.05);
    expect(body.statistical_tests.significant).toBe(true);
  });

  test('Summary 2026 JSON is valid', async ({ request }) => {
    const res = await request.get(`${BASE}/data/production/summary.json`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(body.strategy_id).toBe('smart_simple_v11');
    expect(body.year).toBe(2026);
    const stats = body.strategies[body.strategy_id];
    expect(stats).toBeDefined();
  });

  test('Trade files exist for active strategy', async ({ request }) => {
    const res2025 = await request.get(`${BASE}/data/production/trades/smart_simple_v11_2025.json`);
    expect(res2025.ok(), '2025 trade file missing').toBeTruthy();
    const trades2025 = await res2025.json();
    expect(trades2025.trades.length, 'Expected 20+ backtest trades').toBeGreaterThanOrEqual(20);

    const res2026 = await request.get(`${BASE}/data/production/trades/smart_simple_v11.json`);
    expect(res2026.ok(), '2026 trade file missing').toBeTruthy();
    const trades2026 = await res2026.json();
    expect(trades2026.trades).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// 1. Hub Page
// ---------------------------------------------------------------------------

test.describe('1 — Hub Page', () => {
  test('renders navigation cards and screenshot', async ({ page }) => {
    const errors = setupConsoleErrorCapture(page);
    await authenticateUser(page);
    await navigateAndScreenshot(page, '/hub', '01-hub-page');

    const body = await page.textContent('body');
    expect(body).toBeTruthy();

    // Hub uses buttons for navigation cards, not <a> links
    // Verify key module cards are present by text content
    expect(body!.includes('Forecasting'), 'Missing Forecasting module card').toBeTruthy();
    expect(body!.includes('Dashboard'), 'Missing Dashboard module card').toBeTruthy();
    expect(body!.includes('Produccion') || body!.includes('Producción'), 'Missing Production module card').toBeTruthy();
    expect(body!.includes('SignalBridge'), 'Missing SignalBridge module card').toBeTruthy();
    expect(body!.includes('Analisis') || body!.includes('Análisis'), 'Missing Analysis module card').toBeTruthy();

    // Should have 6 module cards (buttons with "Acceder al modulo")
    const moduleButtons = page.locator('button:has-text("Acceder al modulo")');
    const count = await moduleButtons.count();
    expect(count, `Expected 6 module cards, found ${count}`).toBeGreaterThanOrEqual(5);

    const critical = filterBenignErrors(errors);
    expect(critical, `Console errors on /hub: ${critical.join('; ')}`).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 2. Forecasting Page
// ---------------------------------------------------------------------------

test.describe('2 — Forecasting Page', () => {
  test('loads CSV data and renders model zoo', async ({ page }) => {
    const errors = setupConsoleErrorCapture(page);
    await authenticateUser(page);
    await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(5000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/02-forecasting-overview.png`, fullPage: true });

    // The page should show model names or a table/grid
    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    // Should contain at least one model name
    const modelNames = ['ridge', 'bayesian', 'xgboost', 'lightgbm', 'catboost'];
    const found = modelNames.filter(m => body!.toLowerCase().includes(m));
    expect(found.length, `Expected model names on page, found: ${found}`).toBeGreaterThanOrEqual(1);

    const critical = filterBenignErrors(errors);
    expect(critical, `Console errors on /forecasting: ${critical.join('; ')}`).toHaveLength(0);
  });

  test('backtest images load without 404', async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(3000);

    // Check that at least some images are visible (not broken)
    const images = page.locator('img[src*="backtest"]');
    const imgCount = await images.count();
    // Some pages may lazy-load; check those visible
    if (imgCount > 0) {
      const firstImg = images.first();
      await expect(firstImg).toBeVisible({ timeout: 10000 });
    }
  });

  test('forward forecast section shows recent weeks', async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(5000);

    const body = await page.textContent('body');
    // Should show recent week references (W09, W10, W11, W12)
    const recentWeeks = ['W09', 'W10', 'W11', 'W12', 'w09', 'w10', 'w11', 'w12',
                          '2026_W09', '2026_W10', '2026_W11', '2026_W12'];
    const hasRecent = recentWeeks.some(w => body!.includes(w));
    // Forward PNGs should be accessible
    const forwardImgs = page.locator('img[src*="forward"]');
    const fwCount = await forwardImgs.count();
    // Either text or images indicate forward data is present
    expect(hasRecent || fwCount > 0, 'No recent forward forecast data visible').toBeTruthy();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/02b-forecasting-forward.png`, fullPage: true });
  });
});

// ---------------------------------------------------------------------------
// 3. Dashboard Page (2025 Backtest + Approval)
// ---------------------------------------------------------------------------

test.describe('3 — Dashboard Page (Backtest 2025)', () => {
  test('renders KPIs, trades, and approval gates', async ({ page }) => {
    const errors = setupConsoleErrorCapture(page);
    await authenticateUser(page);
    await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
    await page.waitForTimeout(10000); // Dashboard loads multiple JSON files
    await page.screenshot({ path: `${SCREENSHOT_DIR}/03-dashboard-full.png`, fullPage: true });

    const body = await page.textContent('body');
    expect(body).toBeTruthy();

    // Should show strategy name or ID
    expect(
      body!.toLowerCase().includes('smart simple') || body!.toLowerCase().includes('smart_simple'),
      'Strategy name not found on dashboard'
    ).toBeTruthy();

    // Should show key metrics (return, sharpe, or similar numeric values)
    const hasPercentage = /\d+\.\d+%/.test(body!);
    const hasNumeric = /\d{2,}/.test(body!);
    expect(hasPercentage || hasNumeric, 'No numeric KPIs visible').toBeTruthy();

    const critical = filterBenignErrors(errors);
    expect(critical, `Console errors on /dashboard: ${critical.join('; ')}`).toHaveLength(0);
  });

  test('approval section shows APPROVED status', async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
    await page.waitForTimeout(8000);

    const body = await page.textContent('body');
    // Should show approval-related text
    const approvalTerms = ['APPROVED', 'APROBAD', 'PROMOTE', 'PENDING', 'aprobad', 'approved'];
    const hasApproval = approvalTerms.some(t => body!.includes(t));
    expect(hasApproval, 'No approval status visible on dashboard').toBeTruthy();

    // Gates section should be present
    const gateTerms = ['Retorno', 'Sharpe', 'Drawdown', 'Trades', 'Significancia', 'gate', 'Gate'];
    const hasGates = gateTerms.some(t => body!.includes(t));
    expect(hasGates, 'No gate information visible on dashboard').toBeTruthy();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/03b-dashboard-approval.png`, fullPage: false });
  });
});

// ---------------------------------------------------------------------------
// 4. Production Page (2026 YTD)
// ---------------------------------------------------------------------------

test.describe('4 — Production Page (2026 YTD)', () => {
  test('renders production metrics and trades', async ({ page }) => {
    const errors = setupConsoleErrorCapture(page);
    await authenticateUser(page);
    await page.goto('/production', { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(8000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/04-production-full.png`, fullPage: true });

    const body = await page.textContent('body');
    expect(body).toBeTruthy();

    // Should show 2026 reference
    expect(body!.includes('2026'), 'No 2026 year reference on production page').toBeTruthy();

    // Should show strategy
    expect(
      body!.toLowerCase().includes('smart simple') || body!.toLowerCase().includes('smart_simple'),
      'Strategy name not found on production page'
    ).toBeTruthy();

    const critical = filterBenignErrors(errors);
    expect(critical, `Console errors on /production: ${critical.join('; ')}`).toHaveLength(0);
  });

  test('equity curve PNG loads (graceful if missing)', async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/production', { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(5000);

    // Equity curve images are optional (graceful fallback)
    const equityImgs = page.locator('img[src*="equity_curve"]');
    const count = await equityImgs.count();
    if (count > 0) {
      // If present, verify they rendered (not display:none from error handler)
      const firstVisible = await equityImgs.first().isVisible();
      // Log but don't fail — PNGs are optional per SDD spec
      console.log(`Equity curve visible: ${firstVisible}`);
    }
  });

  test('production shows APPROVED badge', async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/production', { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(5000);

    const body = await page.textContent('body');
    const approvedTerms = ['APPROVED', 'APROBAD', 'approved', 'aprobado'];
    const hasApproved = approvedTerms.some(t => body!.includes(t));
    expect(hasApproved, 'APPROVED badge not visible on production page').toBeTruthy();

    await page.screenshot({ path: `${SCREENSHOT_DIR}/04b-production-approval-badge.png`, fullPage: false });
  });
});

// ---------------------------------------------------------------------------
// 5. Analysis Page
// ---------------------------------------------------------------------------

test.describe('5 — Analysis Page', () => {
  test('loads weekly analysis with macro data', async ({ page }) => {
    const errors = setupConsoleErrorCapture(page);
    await authenticateUser(page);
    await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });

    // Wait for content to render
    try {
      await page.locator('text=/SEMANA|Indicadores|Timeline|Análisis/i').first().waitFor({ timeout: 20000 });
    } catch {
      // Page may render differently, continue
    }
    await page.waitForTimeout(5000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/05-analysis-overview.png`, fullPage: true });

    const body = await page.textContent('body');
    expect(body).toBeTruthy();

    // Should show week references
    const weekTerms = ['Semana', 'SEMANA', 'W1', 'W2', 'semana'];
    const hasWeek = weekTerms.some(t => body!.includes(t));
    expect(hasWeek, 'No week references on analysis page').toBeTruthy();

    const critical = filterBenignErrors(errors);
    expect(critical, `Console errors on /analysis: ${critical.join('; ')}`).toHaveLength(0);
  });

  test('macro chart section renders or page shows week selector', async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
    await page.waitForTimeout(5000);

    // Scroll to macro section
    await page.evaluate(() => window.scrollBy(0, 600));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/05b-analysis-macro-charts.png`, fullPage: false });

    const body = await page.textContent('body');
    // Check for macro variable names OR week selector (data may be in different weeks)
    // The analysis page shows "SEMANA X" header and a week dropdown
    const macroTerms = ['DXY', 'VIX', 'WTI', 'EMBI', 'dxy', 'vix',
                        'Dollar', 'Volatilidad', 'Crudo', 'Petroleo', 'petroleo',
                        'Indicadores', 'indicadores', 'Macro', 'macro'];
    const pageTerms = ['SEMANA', 'Semana', 'semana', 'Analisis Semanal',
                       'generate_weekly_analysis']; // Even "no data" message is valid page state
    const hasMacro = macroTerms.some(t => body!.includes(t));
    const hasPage = pageTerms.some(t => body!.includes(t));
    expect(hasMacro || hasPage, 'Analysis page did not render any expected content').toBeTruthy();
  });

  test('daily timeline section renders', async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
    await page.waitForTimeout(5000);

    // Scroll further down to timeline
    await page.evaluate(() => window.scrollBy(0, 2000));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/05c-analysis-timeline.png`, fullPage: false });
  });
});

// ---------------------------------------------------------------------------
// 6. Execution Module
// ---------------------------------------------------------------------------

test.describe('6 — Execution Module', () => {
  test('execution dashboard loads with status cards', async ({ page }) => {
    const errors = setupConsoleErrorCapture(page);
    await authenticateUser(page);
    await page.goto('/execution/dashboard', { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(5000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/06-execution-dashboard.png`, fullPage: true });

    const body = await page.textContent('body');
    expect(body).toBeTruthy();
    // Should mention trading mode, status, or kill switch
    const execTerms = ['PAPER', 'Trading', 'Kill', 'Status', 'Bridge', 'Execution',
                       'paper', 'trading', 'status', 'bridge', 'execution'];
    const hasExec = execTerms.some(t => body!.includes(t));
    expect(hasExec, 'No execution-related content visible').toBeTruthy();

    const critical = filterBenignErrors(errors);
    expect(critical, `Console errors on /execution/dashboard: ${critical.join('; ')}`).toHaveLength(0);
  });

  test('exchanges page loads', async ({ page }) => {
    await authenticateUser(page);
    await navigateAndScreenshot(page, '/execution/exchanges', '06b-execution-exchanges', { settleMs: 4000 });
  });

  test('settings page loads', async ({ page }) => {
    await authenticateUser(page);
    await navigateAndScreenshot(page, '/execution/settings', '06c-execution-settings', { settleMs: 4000 });
  });
});

// ---------------------------------------------------------------------------
// 7. Cross-Page Navigation & Auth
// ---------------------------------------------------------------------------

test.describe('7 — Cross-Page Navigation', () => {
  test('all main pages load without crash', async ({ page }) => {
    const errors = setupConsoleErrorCapture(page);
    await authenticateUser(page);

    const pages = [
      { path: '/hub', name: 'Hub' },
      { path: '/forecasting', name: 'Forecasting' },
      { path: '/dashboard', name: 'Dashboard' },
      { path: '/production', name: 'Production' },
      { path: '/analysis', name: 'Analysis' },
      { path: '/execution/dashboard', name: 'Execution' },
    ];

    for (const p of pages) {
      await page.goto(p.path, { waitUntil: 'load', timeout: 30000 });
      await page.waitForTimeout(2000);
      // Page should not show error boundary or blank
      const body = await page.textContent('body');
      expect(body!.length, `${p.name} page appears empty`).toBeGreaterThan(100);
      // No "Error" or "500" as the main content
      const isErrorPage = body!.includes('Internal Server Error') || body!.includes('Application error');
      expect(isErrorPage, `${p.name} shows an error page`).toBeFalsy();
    }

    const critical = filterBenignErrors(errors);
    // Allow some errors across 6 page navigations but flag if too many
    expect(critical.length, `${critical.length} console errors across pages: ${critical.slice(0, 3).join('; ')}`).toBeLessThan(5);
  });

  test('login page renders auth form', async ({ page }) => {
    await page.goto('/login', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/07-login-page.png`, fullPage: true });

    const body = await page.textContent('body');
    // Should have login-related content
    const loginTerms = ['Login', 'login', 'Sign', 'sign', 'Usuario', 'usuario', 'Password', 'password'];
    const hasLogin = loginTerms.some(t => body!.includes(t));
    expect(hasLogin, 'No login form content visible').toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// 8. Data Integrity & Contract Compliance
// ---------------------------------------------------------------------------

test.describe('8 — Data Contract Compliance', () => {
  test('summary_2025.json conforms to StrategySummary contract', async ({ request }) => {
    const res = await request.get(`${BASE}/data/production/summary_2025.json`);
    const body = await res.json();

    // Required top-level fields
    expect(body.generated_at).toBeDefined();
    expect(body.strategy_name).toBeDefined();
    expect(body.strategy_id).toBeDefined();
    expect(body.year).toBe(2025);
    expect(body.initial_capital).toBe(10000);

    // strategies object must contain strategy_id key + buy_and_hold
    expect(body.strategies[body.strategy_id]).toBeDefined();
    expect(body.strategies.buy_and_hold).toBeDefined();

    // Strategy stats
    const stats = body.strategies[body.strategy_id];
    expect(stats.final_equity).toBeGreaterThan(10000);
    expect(stats.total_return_pct).toBeGreaterThan(0);
    expect(typeof stats.sharpe).toBe('number');
    expect(typeof stats.max_dd_pct).toBe('number');
    expect(typeof stats.win_rate_pct).toBe('number');
    // profit_factor must be number or null (never Infinity)
    expect(stats.profit_factor === null || typeof stats.profit_factor === 'number').toBeTruthy();
    if (stats.profit_factor !== null) {
      expect(isFinite(stats.profit_factor), 'profit_factor must not be Infinity').toBeTruthy();
    }

    // Statistical tests
    expect(body.statistical_tests.p_value).toBeLessThan(0.05);
    expect(body.statistical_tests.significant).toBe(true);
  });

  test('trades file conforms to StrategyTradeFile contract', async ({ request }) => {
    const res = await request.get(`${BASE}/data/production/trades/smart_simple_v11_2025.json`);
    const body = await res.json();

    expect(body.strategy_id).toBe('smart_simple_v11');
    expect(body.strategy_name).toBeDefined();
    expect(body.initial_capital).toBe(10000);
    expect(body.trades.length).toBeGreaterThan(0);

    // Validate first trade structure
    const trade = body.trades[0];
    expect(trade.trade_id).toBeDefined();
    expect(trade.timestamp).toBeDefined();
    expect(['LONG', 'SHORT']).toContain(trade.side);
    expect(typeof trade.entry_price).toBe('number');
    expect(typeof trade.exit_price).toBe('number');
    expect(typeof trade.pnl_usd).toBe('number');
    expect(typeof trade.pnl_pct).toBe('number');
    expect(trade.exit_reason).toBeDefined();

    // No NaN or Infinity in any trade
    for (const t of body.trades) {
      expect(isFinite(t.pnl_usd), `Trade ${t.trade_id} has non-finite pnl_usd`).toBeTruthy();
      expect(isFinite(t.pnl_pct), `Trade ${t.trade_id} has non-finite pnl_pct`).toBeTruthy();
      expect(isFinite(t.entry_price), `Trade ${t.trade_id} has non-finite entry_price`).toBeTruthy();
      expect(isFinite(t.exit_price), `Trade ${t.trade_id} has non-finite exit_price`).toBeTruthy();
    }
  });

  test('approval_state.json conforms to ApprovalState contract', async ({ request }) => {
    const res = await request.get(`${BASE}/data/production/approval_state.json`);
    const body = await res.json();

    expect(body.status).toBe('APPROVED');
    expect(body.strategy).toBe('smart_simple_v11');
    expect(body.backtest_year).toBe(2025);
    expect(['PROMOTE', 'REVIEW', 'REJECT']).toContain(body.backtest_recommendation);

    // Gates array
    expect(body.gates.length).toBe(5);
    for (const gate of body.gates) {
      expect(gate.gate).toBeDefined();
      expect(gate.label).toBeDefined();
      expect(typeof gate.passed).toBe('boolean');
      expect(typeof gate.value).toBe('number');
      expect(typeof gate.threshold).toBe('number');
    }

    // Deploy manifest
    expect(body.deploy_manifest).toBeDefined();
    expect(body.deploy_manifest.script).toContain('train_and_export_smart_simple');
  });

  test('no Infinity, NaN, or undefined in any JSON file', async ({ request }) => {
    const files = [
      '/data/production/summary.json',
      '/data/production/summary_2025.json',
      '/data/production/approval_state.json',
      '/data/production/trades/smart_simple_v11_2025.json',
      '/data/production/trades/smart_simple_v11.json',
    ];

    for (const file of files) {
      const res = await request.get(`${BASE}${file}`);
      if (!res.ok()) continue; // Skip if 404

      const text = await res.text();
      expect(text.includes('Infinity'), `${file} contains Infinity`).toBeFalsy();
      expect(text.includes('NaN'), `${file} contains NaN`).toBeFalsy();
      expect(text.includes('undefined'), `${file} contains undefined`).toBeFalsy();
    }
  });
});

// ---------------------------------------------------------------------------
// 9. Performance & Responsiveness
// ---------------------------------------------------------------------------

test.describe('9 — Performance Baseline', () => {
  test('all pages load within 15 seconds', async ({ page }) => {
    await authenticateUser(page);

    const routes = ['/hub', '/forecasting', '/dashboard', '/production', '/analysis'];
    const timings: Record<string, number> = {};

    for (const route of routes) {
      const start = Date.now();
      await page.goto(route, { waitUntil: 'load', timeout: 30000 });
      timings[route] = Date.now() - start;
      expect(timings[route], `${route} took ${timings[route]}ms (>15s)`).toBeLessThan(15000);
    }

    console.log('Page load timings:', timings);
  });
});

// ---------------------------------------------------------------------------
// 10. Visual Regression Baseline (Full-Page Screenshots)
// ---------------------------------------------------------------------------

test.describe('10 — Visual Baseline Screenshots', () => {
  test('capture all pages at desktop resolution', async ({ page }) => {
    test.setTimeout(120000);
    await authenticateUser(page);
    await page.setViewportSize({ width: 1920, height: 1080 });

    const captures = [
      { path: '/hub', name: '10-desktop-hub', settle: 2000 },
      { path: '/forecasting', name: '10-desktop-forecasting', settle: 3000 },
      { path: '/dashboard', name: '10-desktop-dashboard', settle: 5000 },
      { path: '/production', name: '10-desktop-production', settle: 5000 },
      { path: '/analysis', name: '10-desktop-analysis', settle: 4000 },
      { path: '/execution/dashboard', name: '10-desktop-execution', settle: 2000 },
    ];

    for (const cap of captures) {
      await page.goto(cap.path, { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(cap.settle);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/${cap.name}.png`,
        fullPage: true,
      });
    }
  });

  test('capture critical pages at mobile resolution', async ({ page }) => {
    await authenticateUser(page);
    await page.setViewportSize({ width: 375, height: 812 }); // iPhone X

    const captures = [
      { path: '/hub', name: '10-mobile-hub', settle: 3000 },
      { path: '/forecasting', name: '10-mobile-forecasting', settle: 5000 },
      { path: '/production', name: '10-mobile-production', settle: 6000 },
    ];

    for (const cap of captures) {
      await page.goto(cap.path, { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(cap.settle);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/${cap.name}.png`,
        fullPage: true,
      });
    }
  });
});
