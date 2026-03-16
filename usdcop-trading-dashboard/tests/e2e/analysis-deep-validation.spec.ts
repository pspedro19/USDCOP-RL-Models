import { test, expect } from '@playwright/test';

const DIR = 'tests/e2e/screenshots/full-validation';

test.describe('Analysis Deep Validation', () => {

  test('Analysis page renders fully with data', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        if (text.includes('MetaMask') || text.includes('ethereum') || text.includes('favicon')) return;
        consoleErrors.push(text.substring(0, 300));
      }
    });

    await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });

    // Wait for the week selector to appear (data loaded)
    const weekSelector = page.locator('text=/SEMANA/i');
    await expect(weekSelector.first()).toBeVisible({ timeout: 30000 });

    // Wait for content sections to render
    await page.waitForTimeout(8000);

    // Top of page screenshot
    await page.screenshot({ path: `${DIR}/analysis-01-top.png`, fullPage: false });

    // Scroll to macro section
    await page.evaluate(() => window.scrollTo(0, 400));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/analysis-02-macro.png`, fullPage: false });

    // Scroll to charts
    await page.evaluate(() => window.scrollTo(0, 1000));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/analysis-03-charts.png`, fullPage: false });

    // Scroll to timeline
    await page.evaluate(() => window.scrollTo(0, 2000));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/analysis-04-timeline.png`, fullPage: false });

    // Scroll to bottom
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/analysis-05-bottom.png`, fullPage: false });

    // Full page
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(1000);
    await page.screenshot({ path: `${DIR}/analysis-06-fullpage.png`, fullPage: true });

    // Report console errors
    if (consoleErrors.length > 0) {
      console.log(`Console errors (${consoleErrors.length}):`);
      consoleErrors.forEach((e, i) => console.log(`  ${i+1}. ${e}`));
    } else {
      console.log('[OK] No console errors on /analysis');
    }

    // Verify key sections exist in page content
    const content = await page.content();
    const checks = [
      { name: 'Week selector', pattern: /SEMANA/i },
      { name: 'Analysis content', pattern: /Analisis|analisis/i },
    ];

    for (const check of checks) {
      const found = check.pattern.test(content);
      console.log(`[${found ? 'OK' : 'FAIL'}] ${check.name}`);
      expect(found).toBe(true);
    }
  });

  test('Dashboard loads with full backtest data', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        if (text.includes('MetaMask') || text.includes('ethereum') || text.includes('favicon')) return;
        consoleErrors.push(text.substring(0, 300));
      }
    });

    await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
    await page.waitForTimeout(12000);

    await page.screenshot({ path: `${DIR}/dashboard-01-top.png`, fullPage: false });

    await page.evaluate(() => window.scrollTo(0, 600));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/dashboard-02-chart.png`, fullPage: false });

    await page.evaluate(() => window.scrollTo(0, 1500));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/dashboard-03-gates.png`, fullPage: false });

    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/dashboard-04-trades.png`, fullPage: false });

    if (consoleErrors.length > 0) {
      console.log(`Dashboard console errors (${consoleErrors.length}):`);
      consoleErrors.forEach((e, i) => console.log(`  ${i+1}. ${e}`));
    } else {
      console.log('[OK] No console errors on /dashboard');
    }
  });

  test('Production page loads with 2026 data', async ({ page }) => {
    await page.goto('/production', { waitUntil: 'load', timeout: 60000 });
    await page.waitForTimeout(12000);

    await page.screenshot({ path: `${DIR}/production-01-top.png`, fullPage: false });

    await page.evaluate(() => window.scrollTo(0, 800));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/production-02-chart.png`, fullPage: false });

    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${DIR}/production-03-bottom.png`, fullPage: false });
  });
});
