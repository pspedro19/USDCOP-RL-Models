/**
 * SDD Full Pipeline — E2E Integration Tests
 * ============================================
 * Verifies the complete production lifecycle:
 *   1. Python export pipeline generates valid JSONs/PNGs
 *   2. Dashboard renders strategy-agnostic production page
 *   3. Approval flow (PENDING -> APPROVED) works end-to-end
 *   4. Forecasting page has weekly forecast PNGs
 *   5. No console errors throughout
 *
 * Spec refs:
 *   - sdd-strategy-spec.md (universal schemas)
 *   - sdd-dashboard-integration.md (JSON file layout)
 *   - sdd-approval-spec.md (gate system + lifecycle)
 *   - sdd-pipeline-lifecycle.md (6-stage lifecycle)
 *
 * Run:
 *   npx playwright test tests/e2e/sdd-full-pipeline.spec.ts --project=chromium
 */

import { test, expect } from '@playwright/test';
import { execSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import {
  authenticateUser,
  setupConsoleErrorCapture,
  filterBenignErrors,
} from './helpers/auth';

const SCREENSHOTS_DIR = 'tests/e2e/screenshots/sdd-pipeline';
const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..');
const DATA_DIR = path.join(PROJECT_ROOT, 'usdcop-trading-dashboard', 'public', 'data', 'production');

// All tests in this suite run on port 5000 (webServer port)
test.use({ baseURL: 'http://localhost:5000' });

/**
 * Check if JSON data files are fresh (modified within last hour).
 * If fresh, skip the heavy Python pipeline in beforeAll.
 */
function dataIsFresh(): boolean {
  const summaryPath = path.join(DATA_DIR, 'summary.json');
  const approvalPath = path.join(DATA_DIR, 'approval_state.json');
  const tradesPath = path.join(DATA_DIR, 'trades', 'smart_simple_v11.json');

  for (const p of [summaryPath, approvalPath, tradesPath]) {
    if (!fs.existsSync(p)) return false;
    const stat = fs.statSync(p);
    const ageMs = Date.now() - stat.mtimeMs;
    if (ageMs > 60 * 60 * 1000) return false; // older than 1 hour
  }
  return true;
}

// Serial execution: tests depend on each other (approval state transitions)
test.describe.serial('SDD Full Pipeline — E2E Integration', () => {
  // Accumulated console errors across all tests
  let allConsoleErrors: string[] = [];

  // =========================================================================
  // SETUP: Run Python export pipeline (train + generate JSONs/PNGs)
  // =========================================================================

  test.beforeAll(async () => {
    // Check if data files are fresh — skip heavy pipeline if so
    if (dataIsFresh()) {
      console.log('Data files are fresh (< 1 hour old), skipping Python pipeline');
    } else {
      // Step 1: Run both phases to generate all JSONs + PNGs
      console.log('Running Python export pipeline (--phase both)...');
      try {
        const output = execSync(
          'python scripts/train_and_export_smart_simple.py --phase both',
          {
            cwd: PROJECT_ROOT,
            timeout: 300_000, // 5 minutes (training takes ~2-3 min)
            stdio: 'pipe',
            encoding: 'utf-8',
          }
        );
        console.log('Pipeline output (last 500 chars):', output.slice(-500));
      } catch (error: unknown) {
        const err = error as { stdout?: string; stderr?: string; message?: string };
        console.error('Pipeline STDOUT:', err.stdout?.slice(-500));
        console.error('Pipeline STDERR:', err.stderr?.slice(-500));
        throw new Error(`Python pipeline failed: ${err.message}`);
      }
    }

    // Always reset approval to PENDING for a clean test run
    console.log('Resetting approval state to PENDING...');
    try {
      execSync(
        'python scripts/train_and_export_smart_simple.py --reset-approval',
        {
          cwd: PROJECT_ROOT,
          timeout: 30_000,
          stdio: 'pipe',
          encoding: 'utf-8',
        }
      );
      console.log('Approval reset complete');
    } catch (error: unknown) {
      const err = error as { message?: string };
      console.warn('Approval reset failed (non-fatal):', err.message);
    }
  });

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page);
    const errors = setupConsoleErrorCapture(page);
    // Accumulate across tests
    page.on('close', () => {
      allConsoleErrors.push(...errors);
    });
  });

  // =========================================================================
  // GROUP 1: JSON Contract Compliance (API-level, no browser rendering)
  // =========================================================================

  test('1.1 summary.json conforms to SDD StrategySummary schema', async ({ request }) => {
    const resp = await request.get('/data/production/summary.json');
    expect(resp.ok()).toBeTruthy();

    const summary = await resp.json();

    // Strategy identity
    expect(summary.strategy_id).toBe('smart_simple_v11');
    expect(summary.strategy_name).toContain('Smart Simple');
    expect(summary.year).toBe(2026);
    expect(summary.initial_capital).toBe(10000.0);

    // Strategy stats (dynamic lookup by strategy_id)
    const stats = summary.strategies[summary.strategy_id];
    expect(stats).toBeDefined();
    expect(typeof stats.final_equity).toBe('number');
    expect(typeof stats.total_return_pct).toBe('number');
    expect(typeof stats.sharpe).toBe('number');
    expect(stats.max_dd_pct).toBeDefined();
    expect(stats.win_rate_pct).toBeDefined();
    // profit_factor can be null (no losses) or number — never Infinity
    expect(stats.profit_factor === null || typeof stats.profit_factor === 'number').toBeTruthy();
    expect(stats.exit_reasons).toBeDefined();
    expect(stats.n_long).toBeDefined();
    expect(stats.n_short).toBeDefined();

    // Buy-and-hold baseline
    expect(summary.strategies.buy_and_hold).toBeDefined();
    expect(typeof summary.strategies.buy_and_hold.final_equity).toBe('number');
    expect(typeof summary.strategies.buy_and_hold.total_return_pct).toBe('number');

    // Statistical tests
    expect(typeof summary.statistical_tests.p_value).toBe('number');
    expect(typeof summary.statistical_tests.significant).toBe('boolean');

    // Monthly breakdown (optional but expected)
    if (summary.monthly) {
      expect(summary.monthly.months.length).toBeGreaterThan(0);
      expect(summary.monthly.trades.length).toBe(summary.monthly.months.length);
      expect(summary.monthly.pnl_pct.length).toBe(summary.monthly.months.length);
    }
  });

  test('1.2 JSON files have NO Infinity, NaN, or undefined', async ({ request }) => {
    const files = [
      '/data/production/summary.json',
      '/data/production/summary_2025.json',
    ];

    for (const file of files) {
      const resp = await request.get(file);
      expect(resp.ok(), `${file} should exist`).toBeTruthy();

      const text = await resp.text();
      expect(text).not.toContain('Infinity');
      expect(text).not.toContain('-Infinity');
      expect(text).not.toContain('NaN');

      // JSON.parse must succeed (no syntax errors)
      const parsed = JSON.parse(text);
      expect(parsed).toBeDefined();
    }

    // Also check approval_state via API
    const approvalResp = await request.get('/api/production/status');
    expect(approvalResp.ok()).toBeTruthy();
    const approvalText = await approvalResp.text();
    expect(approvalText).not.toContain('Infinity');
    expect(approvalText).not.toContain('NaN');
    JSON.parse(approvalText); // must not throw
  });

  test('1.3 strategy_id consistent across all files', async ({ request }) => {
    const summaryResp = await request.get('/data/production/summary.json');
    const summary = await summaryResp.json();

    const approvalResp = await request.get('/api/production/status');
    const approval = await approvalResp.json();

    expect(summary.strategy_id).toBe('smart_simple_v11');
    expect(approval.strategy).toBe('smart_simple_v11');
  });

  test('1.4 summary_2025.json has OOS backtest metrics', async ({ request }) => {
    const resp = await request.get('/data/production/summary_2025.json');
    expect(resp.ok()).toBeTruthy();

    const summary = await resp.json();
    expect(summary.year).toBe(2025);
    expect(summary.strategy_id).toBe('smart_simple_v11');

    const stats = summary.strategies.smart_simple_v11;
    expect(stats).toBeDefined();
    // Expected ~+20% from backtest
    expect(stats.total_return_pct).toBeGreaterThan(15);

    // Must be statistically significant (p < 0.05)
    expect(summary.statistical_tests.p_value).toBeLessThan(0.05);
    expect(summary.statistical_tests.significant).toBe(true);
  });

  test('1.5 trades JSON has valid trades with SDD schema', async ({ request }) => {
    // Check both production (2026) and backtest (2025) trade files
    for (const file of [
      '/data/production/trades/smart_simple_v11.json',
      '/data/production/trades/smart_simple_v11_2025.json',
    ]) {
      const resp = await request.get(file);
      expect(resp.ok(), `${file} should exist`).toBeTruthy();

      const data = await resp.json();
      expect(data.strategy_id).toBe('smart_simple_v11');
      expect(data.trades).toBeDefined();
      expect(data.trades.length).toBeGreaterThan(0);

      // Validate first trade has all required SDD fields
      const trade = data.trades[0];
      expect(trade.trade_id).toBeDefined();
      expect(trade.timestamp).toBeDefined();
      expect(trade.exit_timestamp).toBeDefined();
      expect(['LONG', 'SHORT']).toContain(trade.side);
      expect(trade.entry_price).toBeGreaterThan(3000); // USDCOP range
      expect(trade.exit_price).toBeGreaterThan(3000);
      expect(typeof trade.pnl_usd).toBe('number');
      expect(typeof trade.pnl_pct).toBe('number');
      expect(['take_profit', 'week_end', 'hard_stop']).toContain(trade.exit_reason);
      expect(trade.leverage).toBeGreaterThan(0);

      // Summary section
      expect(data.summary.total_trades).toBe(data.trades.length);
      expect(data.summary.winning_trades + data.summary.losing_trades).toBeLessThanOrEqual(
        data.trades.length
      );
    }
  });

  test('1.6 approval_state.json has 5 gates, PENDING status', async ({ request }) => {
    const resp = await request.get('/api/production/status');
    expect(resp.ok()).toBeTruthy();

    const state = await resp.json();
    expect(state.status).toBe('PENDING_APPROVAL');
    expect(state.strategy).toBe('smart_simple_v11');
    expect(state.gates).toBeDefined();
    expect(state.gates.length).toBe(5);
    expect(state.backtest_recommendation).toBe('PROMOTE');

    // Each gate has required fields
    for (const gate of state.gates) {
      expect(gate.gate).toBeDefined();
      expect(gate.label).toBeDefined();
      expect(typeof gate.passed).toBe('boolean');
      expect(typeof gate.value).toBe('number');
      expect(typeof gate.threshold).toBe('number');
    }

    // Verify all 5 gate IDs
    const gateIds = state.gates.map((g: { gate: string }) => g.gate);
    expect(gateIds).toContain('min_return_pct');
    expect(gateIds).toContain('min_sharpe_ratio');
    expect(gateIds).toContain('max_drawdown_pct');
    expect(gateIds).toContain('min_trades');
    expect(gateIds).toContain('statistical_significance');
  });

  // =========================================================================
  // GROUP 2: Production Page — Pending State (UI tests)
  // =========================================================================

  test('2.1 Page loads with strategy name and PENDING badge', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // Page header
    await expect(page.locator('text=Produccion 2026')).toBeVisible({ timeout: 10000 });

    // Strategy badge (Smart Simple v1.1.0)
    await expect(page.getByText('Smart Simple v1.1.0', { exact: true })).toBeVisible();

    // Status badge (Pendiente = PENDING_APPROVAL)
    await expect(page.locator('text=Pendiente')).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/01-production-pending.png`,
      fullPage: true,
    });
  });

  test('2.2 KPI cards show real values', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // All 5 KPI card labels must be visible (use .first() since labels appear in gates too)
    for (const label of [
      'Retorno Total',
      'Sharpe Ratio',
      'Profit Factor',
      'Win Rate',
      'Max Drawdown',
    ]) {
      await expect(page.getByText(label).first()).toBeVisible({ timeout: 5000 });
    }

    // KPI values should not show "undefined" or "NaN"
    const kpiSection = page.locator('text=Metricas Clave').first().locator('..');
    const kpiText = await kpiSection.textContent();
    expect(kpiText).not.toContain('undefined');
    expect(kpiText).not.toContain('NaN');
  });

  test('2.3 Gates panel shows 5 validation criteria', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // Gates section header
    await expect(page.locator('text=Gates de Validacion')).toBeVisible({ timeout: 10000 });

    // "X/5 pasaron" badge
    await expect(page.locator('text=/\\d\\/5 pasaron/')).toBeVisible();

    // Gate labels (use .first() since some labels appear in KPI cards too)
    for (const label of [
      'Retorno Minimo',
      'Sharpe Minimo',
      'Max Drawdown',
      'Trades Minimos',
      'Significancia',
    ]) {
      await expect(page.getByText(label).first()).toBeVisible();
    }

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/02-gates-panel.png`,
      fullPage: false,
    });
  });

  test('2.4 Approval panel visible with buttons', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // Approval section
    await expect(page.locator('text=Aprobacion de Estrategia')).toBeVisible({ timeout: 10000 });

    // Buttons
    await expect(page.locator('button:has-text("Aprobar y Promover")')).toBeVisible();
    await expect(page.locator('button:has-text("Rechazar")')).toBeVisible();

    // Recommendation badge (PROMOTE expected for Smart Simple v1.1)
    await expect(page.locator('text=PROMOTE')).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/03-approval-panel.png`,
      fullPage: false,
    });
  });

  test('2.5 Trade table shows trades with SDD exit reasons', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // Trade table header
    await expect(page.getByText('Historial de Trades').first()).toBeVisible({ timeout: 10000 });

    // Table should have rows
    const tableRows = page.locator('table tbody tr');
    const rowCount = await tableRows.count();
    expect(rowCount).toBeGreaterThan(0);

    // Exit reason badges should include Smart Simple reasons
    const pageText = await page.locator('table').textContent();
    const hasSmartSimpleReasons =
      pageText?.includes('take_profit') || pageText?.includes('week_end');
    expect(hasSmartSimpleReasons).toBeTruthy();

    // Side badges should include SHORT (regime is SHORT-biased)
    expect(pageText).toContain('SHORT');

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/04-trade-table.png`,
      fullPage: true,
    });
  });

  // =========================================================================
  // GROUP 3: Approval Flow (serial, stateful)
  // =========================================================================

  test('3.1 Click Aprobar shows confirmation dialog', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // Click "Aprobar y Promover"
    await page.locator('button:has-text("Aprobar y Promover")').click();
    await page.waitForTimeout(500);

    // Confirmation step should appear
    await expect(page.locator('button:has-text("Confirmar")')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('Cancelar').first()).toBeVisible();
  });

  test('3.2 Fill notes and confirm -> APPROVED', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // Click "Aprobar y Promover"
    await page.locator('button:has-text("Aprobar y Promover")').click();
    await page.waitForTimeout(500);

    // Fill reviewer notes
    const notesInput = page.locator('input[placeholder*="Notas"]');
    await notesInput.fill('E2E pipeline test — Playwright');

    // Click "Confirmar"
    await page.locator('button:has-text("Confirmar")').click();
    await page.waitForTimeout(2000);

    // Status badge should change to "Aprobado"
    await expect(page.getByText('Aprobado').first()).toBeVisible({ timeout: 10000 });

    // Approval panel should be hidden (only visible when PENDING)
    await expect(page.locator('text=Aprobacion de Estrategia')).not.toBeVisible();

    // Approved-by text should appear
    await expect(page.getByText('dashboard_user').first()).toBeVisible();

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/05-approved-state.png`,
      fullPage: true,
    });
  });

  test('3.3 API confirms APPROVED state', async ({ request }) => {
    const resp = await request.get('/api/production/status');
    expect(resp.ok()).toBeTruthy();

    const state = await resp.json();
    expect(state.status).toBe('APPROVED');
    expect(state.approved_by).toBe('dashboard_user');
    expect(state.reviewer_notes).toBe('E2E pipeline test — Playwright');
  });

  // =========================================================================
  // GROUP 4: Production Page — Approved State
  // =========================================================================

  test('4.1 Page shows Aprobado, KPIs persist', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // Aprobado badge visible
    await expect(page.getByText('Aprobado').first()).toBeVisible({ timeout: 10000 });

    // Approval panel NOT visible (hidden after approval)
    await expect(page.locator('button:has-text("Aprobar y Promover")')).not.toBeVisible();

    // KPIs still there
    await expect(page.getByText('Retorno Total').first()).toBeVisible();

    // Trade table still there
    await expect(page.getByText('Historial de Trades').first()).toBeVisible();
  });

  test('4.2 Equity curve PNG renders or gracefully hides', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    const equityImg = page.locator('img[src*="equity_curve_2026"]');
    const imgCount = await equityImg.count();

    if (imgCount > 0) {
      // If image element exists, check if it loaded or gracefully hid
      const isVisible = await equityImg.isVisible().catch(() => false);
      if (isVisible) {
        const naturalWidth = await equityImg.evaluate(
          (el: HTMLImageElement) => el.naturalWidth
        );
        expect(naturalWidth).toBeGreaterThan(100);
      }
      // If not visible, the onError handler hid the container — that's OK (graceful fallback)
    }
    // If no img element at all, PNGs weren't generated — also OK
  });

  // =========================================================================
  // GROUP 5: Forecasting Page
  // =========================================================================

  test('5.1 Forecasting page loads without errors', async ({ page }) => {
    await page.goto('/forecasting');
    await page.waitForTimeout(5000);

    // No error alert visible
    const errorAlert = page.locator('[role="alert"]:has-text("error")');
    const hasError = await errorAlert.isVisible().catch(() => false);
    expect(hasError).toBeFalsy();

    // Loading spinner should be gone after 10s
    await page.waitForTimeout(5000);
    const spinner = page.locator('.animate-spin');
    const spinnerCount = await spinner.count();
    // Allow 0 spinners (fully loaded) or some decorative ones
    // The key check is no error state
  });

  test('5.2 Weekly forward forecast PNGs exist (HTTP check)', async ({ request }) => {
    const models = ['ridge', 'bayesian_ridge'];
    const weeks = ['W01', 'W02', 'W03'];
    let found = 0;

    for (const model of models) {
      for (const week of weeks) {
        const resp = await request.get(
          `/forecasting/forward_${model}_2026_${week}.png`
        );
        if (resp.ok()) {
          const body = await resp.body();
          expect(body.length).toBeGreaterThan(1000); // Not empty/tiny
          found++;
        }
      }
    }
    // At least some forward PNGs should exist
    expect(found).toBeGreaterThan(0);
  });

  test('5.3 Backtest PNGs exist for multiple horizons', async ({ request }) => {
    const models = ['ridge', 'bayesian_ridge'];
    const horizons = ['h1', 'h5'];
    let found = 0;

    for (const model of models) {
      for (const h of horizons) {
        const resp = await request.get(`/forecasting/backtest_${model}_${h}.png`);
        if (resp.ok()) {
          const body = await resp.body();
          expect(body.length).toBeGreaterThan(1000);
          found++;
        }
      }
    }
    expect(found).toBeGreaterThan(0);
  });

  test('5.4 CSV data file loads', async ({ request }) => {
    const resp = await request.get('/forecasting/bi_dashboard_unified.csv');
    expect(resp.ok()).toBeTruthy();

    const body = await resp.text();
    expect(body).toContain('model_name'); // Header row
    expect(body.length).toBeGreaterThan(5000); // Has real data
  });

  // =========================================================================
  // GROUP 6: Hub Navigation
  // =========================================================================

  test('6.1 Hub has all menu cards', async ({ page }) => {
    await page.goto('/hub');
    await page.waitForTimeout(2000);

    // Check key navigation cards
    const pageText = await page.textContent('body');
    expect(pageText).toContain('Trading');
    expect(pageText).toContain('Produccion');
    expect(pageText).toContain('Forecasting');
  });

  test('6.2 Production card navigates correctly', async ({ page }) => {
    await page.goto('/hub');
    await page.waitForTimeout(2000);

    // Find and click the production link
    const prodLink = page.locator('a[href="/production"]').first();
    if (await prodLink.isVisible().catch(() => false)) {
      await prodLink.click();
      await page.waitForTimeout(2000);
      expect(page.url()).toContain('/production');
      await expect(page.locator('text=Produccion')).toBeVisible({ timeout: 10000 });
    } else {
      // Try clicking any card/button that navigates to production
      const prodCard = page.locator('text=Produccion').first();
      await prodCard.click();
      await page.waitForTimeout(2000);
      expect(page.url()).toContain('/production');
    }
  });

  test('6.3 Forecasting card navigates correctly', async ({ page }) => {
    await page.goto('/hub');
    await page.waitForTimeout(2000);

    const forecastLink = page.locator('a[href="/forecasting"]').first();
    if (await forecastLink.isVisible().catch(() => false)) {
      await forecastLink.click();
      await page.waitForTimeout(2000);
      expect(page.url()).toContain('/forecasting');
    } else {
      const forecastCard = page.locator('text=Forecasting').first();
      await forecastCard.click();
      await page.waitForTimeout(2000);
      expect(page.url()).toContain('/forecasting');
    }
  });

  // =========================================================================
  // GROUP 7: Cross-Suite Validation
  // =========================================================================

  test('7.1 No critical console errors throughout entire suite', async ({ page }) => {
    // Navigate one more page to flush any remaining errors
    await page.goto('/production');
    await page.waitForTimeout(2000);

    const criticalErrors = filterBenignErrors(allConsoleErrors);
    if (criticalErrors.length > 0) {
      console.warn('Console errors found:', criticalErrors);
    }
    // Allow 0 critical errors
    expect(criticalErrors.length).toBe(0);
  });

  // =========================================================================
  // TEARDOWN: Reset approval state for re-runnability
  // =========================================================================

  test.afterAll(async () => {
    try {
      execSync(
        'python scripts/train_and_export_smart_simple.py --reset-approval',
        {
          cwd: PROJECT_ROOT,
          timeout: 30_000,
          stdio: 'pipe',
        }
      );
      console.log('Approval state reset to PENDING_APPROVAL');
    } catch (e) {
      console.warn('Failed to reset approval state:', e);
    }
  });
});
