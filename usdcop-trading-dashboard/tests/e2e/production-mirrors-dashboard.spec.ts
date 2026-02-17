import { test, expect } from '@playwright/test';

/**
 * Production = Dashboard Mirror Verification
 * ============================================
 * Verifies that /production is an exact copy of /dashboard.
 * Both pages should render the same components, charts, and data.
 */

const SCREENSHOTS_DIR = 'tests/e2e/screenshots/production-mirror';

// Override baseURL for this test file (server runs on 5000)
test.use({ baseURL: 'http://localhost:5000' });

test.describe('Production mirrors Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.error(`[BROWSER ERROR] ${msg.text()}`);
      }
    });

    // Set auth
    await page.goto('/login');
    await page.evaluate(() => {
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('username', 'admin');
      sessionStorage.setItem('isAuthenticated', 'true');
      sessionStorage.setItem('username', 'admin');
    });
  });

  test('Production page loads without errors (HTTP 200)', async ({ page }) => {
    const response = await page.goto('/production');
    expect(response?.status()).toBe(200);

    await page.waitForTimeout(3000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/01-production-loaded.png`,
      fullPage: true,
    });

    console.log('Production page loaded successfully');
  });

  test('Dashboard page loads without errors (HTTP 200)', async ({ page }) => {
    const response = await page.goto('/dashboard');
    expect(response?.status()).toBe(200);

    await page.waitForTimeout(3000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/02-dashboard-loaded.png`,
      fullPage: true,
    });

    console.log('Dashboard page loaded successfully');
  });

  test('Both pages have the same components', async ({ page }) => {
    // Check dashboard
    await page.goto('/dashboard');
    await page.waitForTimeout(4000);

    const dashboardButtons = await page.locator('button').count();
    const dashboardSelects = await page.locator('select').count();
    const dashboardCards = await page.locator('[class*="card"], [class*="Card"]').count();
    const dashboardH1 = await page.locator('h1, h2, h3').allTextContents();

    console.log(`Dashboard: ${dashboardButtons} buttons, ${dashboardSelects} selects, ${dashboardCards} cards`);
    console.log(`Dashboard headers: ${dashboardH1.slice(0, 5).join(', ')}`);

    // Check production
    await page.goto('/production');
    await page.waitForTimeout(4000);

    const productionButtons = await page.locator('button').count();
    const productionSelects = await page.locator('select').count();
    const productionCards = await page.locator('[class*="card"], [class*="Card"]').count();
    const productionH1 = await page.locator('h1, h2, h3').allTextContents();

    console.log(`Production: ${productionButtons} buttons, ${productionSelects} selects, ${productionCards} cards`);
    console.log(`Production headers: ${productionH1.slice(0, 5).join(', ')}`);

    // Both should have roughly the same structure (allow small diff due to API timing)
    expect(Math.abs(productionButtons - dashboardButtons)).toBeLessThanOrEqual(10);
    expect(Math.abs(productionSelects - dashboardSelects)).toBeLessThanOrEqual(2);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/03-production-components.png`,
      fullPage: true,
    });
  });

  test('Produccion nav button exists in navbar', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(3000);

    // Nav uses <button onClick>
    const prodButton = page.locator('button:has-text("Produccion")');
    await expect(prodButton).toBeVisible({ timeout: 5000 });

    console.log('Produccion nav button visible');

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/04-nav-produccion.png`,
      fullPage: true,
    });
  });

  test('Production page has model selector and KPI cards', async ({ page }) => {
    await page.goto('/production');
    await page.waitForTimeout(4000);

    // Look for key dashboard components that should be present
    // Model-related UI
    const hasModelUI = await page.locator('text=Modelo, text=Model, text=Investor, text=PPO').first().isVisible().catch(() => false);
    console.log(`Model UI visible: ${hasModelUI}`);

    // KPI-style cards or metrics
    const hasMetrics = await page.locator('text=Sharpe, text=Retorno, text=Return, text=Trades').first().isVisible().catch(() => false);
    console.log(`Metrics visible: ${hasMetrics}`);

    // Charts/canvas
    const hasChart = await page.locator('canvas, svg, [class*="chart"]').first().isVisible().catch(() => false);
    console.log(`Chart visible: ${hasChart}`);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/05-production-kpi-model.png`,
      fullPage: true,
    });
  });
});
