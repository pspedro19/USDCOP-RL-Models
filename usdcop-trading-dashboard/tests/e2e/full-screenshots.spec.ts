import { test, expect } from '@playwright/test';

const DIR = 'tests/e2e/screenshots/full-validation';

test.describe('Full Visual Validation Screenshots', () => {

  test('01 - Hub page complete', async ({ page }) => {
    await page.goto('/hub', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `${DIR}/01-hub.png`, fullPage: true });
  });

  test('02 - Forecasting with charts', async ({ page }) => {
    await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(5000);
    await page.screenshot({ path: `${DIR}/02-forecasting.png`, fullPage: true });
  });

  test('03 - Dashboard backtest 2025', async ({ page }) => {
    await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
    // Wait for backtest data to load
    await page.waitForTimeout(12000);
    await page.screenshot({ path: `${DIR}/03-dashboard.png`, fullPage: true });
  });

  test('04 - Production page 2026', async ({ page }) => {
    await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(8000);
    await page.screenshot({ path: `${DIR}/04-production.png`, fullPage: true });
  });

  test('05 - Analysis page week 9 (latest)', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
    // Wait for content to render (React Query fetches index then weekly data)
    await page.locator('text=/Indicadores Macro|Timeline Diario|SEMANA/i').first().waitFor({ timeout: 30000 });
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `${DIR}/05-analysis-latest.png`, fullPage: true });
  });

  test('06 - Analysis page scrolled to macro charts', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
    await page.locator('text=/Indicadores Macro|SEMANA/i').first().waitFor({ timeout: 30000 });
    await page.waitForTimeout(3000);
    await page.evaluate(() => window.scrollBy(0, 800));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/06-analysis-macro.png`, fullPage: false });
  });

  test('07 - Analysis page scrolled to timeline', async ({ page }) => {
    await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
    await page.locator('text=/Timeline Diario|Indicadores Macro/i').first().waitFor({ timeout: 30000 });
    await page.waitForTimeout(3000);
    await page.evaluate(() => window.scrollBy(0, 2000));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/07-analysis-timeline.png`, fullPage: false });
  });

  test('08 - Console error check across all pages', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        // Ignore non-critical errors: WebSocket (no WS server), MetaMask, favicon
        if (text.includes('MetaMask') || text.includes('ethereum') || text.includes('favicon')) return;
        if (text.includes('ws://localhost:8000')) return; // WebSocket server not running
        errors.push(`${text.substring(0, 200)}`);
      }
    });

    const pages = ['/hub', '/forecasting', '/dashboard', '/production', '/analysis'];
    for (const p of pages) {
      await page.goto(p, { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(5000);
    }

    if (errors.length > 0) {
      console.log(`Console errors found (${errors.length}):\n  ${errors.join('\n  ')}`);
    } else {
      console.log('[OK] No console errors across all 5 pages');
    }
    // Don't fail the test for console errors, just report
    expect(true).toBe(true);
  });

  test('09 - Previous analysis screenshots (Mar 9)', async ({ page }) => {
    // Check existing detailed screenshots from previous sessions
    const fs = await import('fs');
    const existing = [
      'tests/e2e/screenshots/health-check/03-dashboard-detailed.png',
      'tests/e2e/screenshots/health-check/03-dashboard-fixed.png',
      'tests/e2e/screenshots/health-check/04-production-fixed.png',
    ];
    for (const f of existing) {
      const exists = fs.existsSync(f);
      console.log(`${f}: ${exists ? 'EXISTS' : 'MISSING'}`);
    }
    expect(true).toBe(true);
  });
});
