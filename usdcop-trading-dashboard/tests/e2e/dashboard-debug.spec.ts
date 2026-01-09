import { test, expect } from '@playwright/test';
import * as fs from 'fs';

/**
 * Dashboard Debug Test
 * Captures all console logs, network requests, and screenshots
 * for comprehensive debugging analysis
 */

test.describe('Dashboard Debug Analysis', () => {
  test('capture full dashboard state with logs', async ({ page }) => {
    const logs: string[] = [];
    const errors: string[] = [];
    const networkRequests: { url: string; status: number; method: string }[] = [];

    // Capture ALL console messages
    page.on('console', msg => {
      const logEntry = `[${msg.type().toUpperCase()}] ${msg.text()}`;
      logs.push(logEntry);
      console.log(logEntry);
    });

    // Capture page errors
    page.on('pageerror', error => {
      const errorEntry = `[PAGE ERROR] ${error.message}`;
      errors.push(errorEntry);
      console.error(errorEntry);
    });

    // Capture network requests
    page.on('response', response => {
      const entry = {
        url: response.url(),
        status: response.status(),
        method: response.request().method()
      };
      networkRequests.push(entry);

      // Log failed requests
      if (response.status() >= 400) {
        console.log(`[NETWORK ${response.status()}] ${response.request().method()} ${response.url()}`);
      }
    });

    // Step 1: Login first
    console.log('\n========== STEP 1: LOGIN ==========\n');
    await page.goto('/login');
    await page.screenshot({ path: 'tests/e2e/screenshots/debug-01-login.png', fullPage: true });

    await page.locator('input[autocomplete="username"]').fill('admin');
    await page.locator('input[type="password"]').fill('admin123');
    await page.getByRole('button', { name: /iniciar sesión/i }).click();

    // Wait for redirect
    await page.waitForURL('**/dashboard', { timeout: 15000 });
    console.log('\n========== STEP 2: DASHBOARD LOADED ==========\n');

    // Step 2: Wait and capture dashboard state
    await page.waitForTimeout(3000); // Let APIs attempt to load
    await page.screenshot({ path: 'tests/e2e/screenshots/debug-02-dashboard-initial.png', fullPage: true });

    // Step 3: Wait more for all network requests
    console.log('\n========== STEP 3: WAITING FOR DATA ==========\n');
    await page.waitForTimeout(5000);
    await page.screenshot({ path: 'tests/e2e/screenshots/debug-03-dashboard-after-5s.png', fullPage: true });

    // Step 4: Check specific elements
    console.log('\n========== STEP 4: ELEMENT ANALYSIS ==========\n');

    // Check KPI cards
    const kpiCards = await page.locator('[class*="KPI"], [class*="kpi"], [class*="metric"]').count();
    console.log(`[ANALYSIS] KPI Cards found: ${kpiCards}`);

    // Check for loading states
    const loadingElements = await page.locator('text=Loading, text=Cargando, [class*="loading"], [class*="skeleton"]').count();
    console.log(`[ANALYSIS] Loading elements: ${loadingElements}`);

    // Check for error states
    const errorElements = await page.locator('text=Error, text=error, text=failed').count();
    console.log(`[ANALYSIS] Error elements: ${errorElements}`);

    // Check equity curve
    const equityCurve = await page.locator('text=EQUITY CURVE').count();
    console.log(`[ANALYSIS] Equity Curve section: ${equityCurve > 0 ? 'Found' : 'Not found'}`);

    // Step 5: Get page content for analysis
    const pageContent = await page.content();
    const hasRealData = !pageContent.includes('Loading') && !pageContent.includes('Fetching');
    console.log(`[ANALYSIS] Has real data loaded: ${hasRealData}`);

    // Step 6: Final screenshot
    await page.screenshot({ path: 'tests/e2e/screenshots/debug-04-dashboard-final.png', fullPage: true });

    // Save debug report
    const report = {
      timestamp: new Date().toISOString(),
      url: page.url(),
      consoleLogs: logs,
      errors: errors,
      networkRequests: networkRequests.filter(r => r.status >= 400 || r.url.includes('/api/')),
      failedRequests: networkRequests.filter(r => r.status >= 400),
      analysis: {
        kpiCards,
        loadingElements,
        errorElements,
        hasRealData
      }
    };

    fs.writeFileSync(
      'tests/e2e/screenshots/debug-report.json',
      JSON.stringify(report, null, 2)
    );

    console.log('\n========== DEBUG REPORT SAVED ==========\n');
    console.log(`Total console logs: ${logs.length}`);
    console.log(`Total errors: ${errors.length}`);
    console.log(`Failed network requests: ${networkRequests.filter(r => r.status >= 400).length}`);

    // Print failed requests summary
    console.log('\n========== FAILED REQUESTS ==========\n');
    networkRequests
      .filter(r => r.status >= 400)
      .forEach(r => console.log(`${r.status} ${r.method} ${r.url}`));
  });
});
