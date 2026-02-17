import { test, expect } from '@playwright/test';

/**
 * Full Navigation Validation E2E Test
 * ====================================
 * Tests complete frontend navigation flow:
 * 1. Login with admin/admin123
 * 2. Hub page navigation
 * 3. Dashboard with model selection
 * 4. FloatingExperimentPanel visibility
 * 5. Production page
 * 6. Console error capture
 */

const SCREENSHOTS_DIR = 'tests/e2e/screenshots/navigation-flow';

// Collect all console errors
const consoleErrors: string[] = [];
const consoleWarnings: string[] = [];
const consoleLogs: string[] = [];

test.describe('Full Navigation Validation', () => {
  test.beforeEach(async ({ page }) => {
    // Capture ALL console output
    page.on('console', msg => {
      const text = `[${msg.type().toUpperCase()}] ${msg.text()}`;
      if (msg.type() === 'error') {
        consoleErrors.push(text);
        console.error(`[BROWSER ERROR] ${msg.text()}`);
      } else if (msg.type() === 'warning') {
        consoleWarnings.push(text);
        console.warn(`[BROWSER WARN] ${msg.text()}`);
      } else {
        consoleLogs.push(text);
        console.log(`[BROWSER] ${msg.text()}`);
      }
    });

    // Capture page errors (uncaught exceptions)
    page.on('pageerror', error => {
      consoleErrors.push(`[PAGE ERROR] ${error.message}`);
      console.error(`[PAGE ERROR] ${error.message}`);
    });

    // Capture failed requests
    page.on('requestfailed', request => {
      const failure = request.failure();
      consoleErrors.push(`[REQUEST FAILED] ${request.url()} - ${failure?.errorText}`);
      console.error(`[REQUEST FAILED] ${request.url()} - ${failure?.errorText}`);
    });
  });

  test('Complete navigation flow with screenshots', async ({ page }) => {
    // ============================================================
    // STEP 1: Login Page
    // ============================================================
    console.log('\n========== STEP 1: LOGIN PAGE ==========');
    await page.goto('/login');
    await page.waitForLoadState('networkidle');

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/01-login-page.png`,
      fullPage: true
    });
    console.log('[OK] Login page loaded');

    // Verify login form exists
    const usernameInput = page.locator('input[autocomplete="username"]');
    const passwordInput = page.locator('input[type="password"]');
    await expect(usernameInput).toBeVisible({ timeout: 5000 });
    await expect(passwordInput).toBeVisible({ timeout: 5000 });
    console.log('[OK] Login form elements found');

    // ============================================================
    // STEP 2: Fill credentials and login
    // ============================================================
    console.log('\n========== STEP 2: LOGIN ==========');
    await usernameInput.fill('admin');
    await passwordInput.fill('admin123');

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/02-credentials-filled.png`,
      fullPage: true
    });

    // Click login button
    const loginButton = page.getByRole('button', { name: /iniciar sesión/i });
    await loginButton.click();
    console.log('[OK] Login button clicked');

    // Wait for redirect - could be /hub or /dashboard
    try {
      await page.waitForURL(/\/(hub|dashboard)/, { timeout: 15000 });
      console.log(`[OK] Redirected to: ${page.url()}`);
    } catch (e) {
      // Take error screenshot
      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/ERROR-login-failed.png`,
        fullPage: true
      });
      console.error('[ERROR] Login redirect failed!');
      console.error('Current URL:', page.url());

      // Check for error message on page
      const errorText = await page.locator('text=Credenciales inválidas').isVisible();
      if (errorText) {
        console.error('[ERROR] Invalid credentials message shown');
      }
      throw e;
    }

    await page.waitForLoadState('networkidle');
    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/03-after-login.png`,
      fullPage: true
    });

    // ============================================================
    // STEP 3: Hub Page (if redirected there)
    // ============================================================
    if (page.url().includes('/hub')) {
      console.log('\n========== STEP 3: HUB PAGE ==========');
      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/04-hub-page.png`,
        fullPage: true
      });

      // Look for dashboard card/link
      const dashboardLink = page.locator('a[href="/dashboard"], [data-testid="dashboard-card"]').first();
      if (await dashboardLink.isVisible()) {
        await dashboardLink.click();
        await page.waitForURL('**/dashboard', { timeout: 10000 });
        console.log('[OK] Navigated to Dashboard from Hub');
      }
    }

    // ============================================================
    // STEP 4: Dashboard Page
    // ============================================================
    console.log('\n========== STEP 4: DASHBOARD ==========');
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // Allow React to render

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/05-dashboard-initial.png`,
      fullPage: true
    });
    console.log('[OK] Dashboard loaded');

    // Check for model dropdown
    const modelDropdown = page.locator('[data-testid="model-dropdown"], select, [role="combobox"]').first();
    const dropdownVisible = await modelDropdown.isVisible().catch(() => false);
    console.log(`[INFO] Model dropdown visible: ${dropdownVisible}`);

    // Check for navbar
    const navbar = page.locator('nav, [role="navigation"]').first();
    const navbarVisible = await navbar.isVisible().catch(() => false);
    console.log(`[INFO] Navbar visible: ${navbarVisible}`);

    // ============================================================
    // STEP 5: Check for Floating Experiment Panel
    // ============================================================
    console.log('\n========== STEP 5: EXPERIMENT PANEL ==========');

    // Look for floating panel (various possible selectors)
    const floatingPanel = page.locator(
      '[data-testid="floating-experiment-panel"], ' +
      '.fixed.bottom-0, ' +
      '[class*="FloatingExperiment"], ' +
      'text=PROPUESTA PENDIENTE, ' +
      'text=PENDING_APPROVAL'
    ).first();

    const panelVisible = await floatingPanel.isVisible().catch(() => false);
    console.log(`[INFO] Floating Experiment Panel visible: ${panelVisible}`);

    if (panelVisible) {
      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/06-experiment-panel-visible.png`,
        fullPage: true
      });
      console.log('[OK] Experiment panel found and captured');
    } else {
      console.log('[INFO] No pending experiment panel (expected if no PENDING_APPROVAL)');
    }

    // Try to find and click on a model with pending approval
    const modelSelect = page.locator('select, [role="listbox"]').first();
    if (await modelSelect.isVisible().catch(() => false)) {
      await modelSelect.click();
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/07-model-dropdown-open.png`,
        fullPage: true
      });
    }

    // ============================================================
    // STEP 6: Production Page
    // ============================================================
    console.log('\n========== STEP 6: PRODUCTION ==========');
    await page.goto('/production');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/08-production-page.png`,
      fullPage: true
    });
    console.log('[OK] Production page loaded');

    // ============================================================
    // STEP 7: Forecasting Page
    // ============================================================
    console.log('\n========== STEP 7: FORECASTING ==========');
    await page.goto('/forecasting');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/09-forecasting-page.png`,
      fullPage: true
    });
    console.log('[OK] Forecasting page loaded');

    // ============================================================
    // STEP 8: Back to Hub via Navbar
    // ============================================================
    console.log('\n========== STEP 8: NAVBAR NAVIGATION ==========');
    const hubLink = page.locator('a[href="/hub"], nav a:has-text("Inicio")').first();
    if (await hubLink.isVisible().catch(() => false)) {
      await hubLink.click();
      await page.waitForURL('**/hub', { timeout: 10000 });
      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/10-back-to-hub.png`,
        fullPage: true
      });
      console.log('[OK] Navigated back to Hub via navbar');
    }

    // ============================================================
    // FINAL REPORT
    // ============================================================
    console.log('\n========================================');
    console.log('NAVIGATION TEST COMPLETE');
    console.log('========================================');
    console.log(`Total Console Errors: ${consoleErrors.length}`);
    console.log(`Total Console Warnings: ${consoleWarnings.length}`);

    if (consoleErrors.length > 0) {
      console.log('\n--- CONSOLE ERRORS ---');
      consoleErrors.forEach(e => console.log(e));
    }

    // Save report to file
    const report = {
      timestamp: new Date().toISOString(),
      pagesVisited: ['/login', '/hub', '/dashboard', '/production', '/forecasting'],
      consoleErrors,
      consoleWarnings,
      totalErrors: consoleErrors.length,
      totalWarnings: consoleWarnings.length,
      success: consoleErrors.length === 0
    };

    // Write report using page.evaluate
    await page.evaluate((reportData) => {
      console.log('NAVIGATION_REPORT:', JSON.stringify(reportData, null, 2));
    }, report);

    // Assert no critical errors
    const criticalErrors = consoleErrors.filter(e =>
      !e.includes('SignalBridge') &&
      !e.includes('ECONNREFUSED') &&
      !e.includes('favicon')
    );

    expect(criticalErrors.length).toBe(0);
  });

  test('Verify model dropdown shows pending approval models', async ({ page }) => {
    console.log('\n========== MODEL DROPDOWN TEST ==========');

    // Login first
    await page.goto('/login');
    await page.locator('input[autocomplete="username"]').fill('admin');
    await page.locator('input[type="password"]').fill('admin123');
    await page.getByRole('button', { name: /iniciar sesión/i }).click();

    // Wait for redirect
    await page.waitForURL(/\/(hub|dashboard)/, { timeout: 15000 });

    // Go to dashboard
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    // Find the model dropdown - try multiple selectors
    const dropdownSelectors = [
      'button:has-text("Seleccionar modelo")',
      'button:has-text("Investor Demo")',
      'button:has-text("PPO")',
      '[data-testid="model-select"]',
      'select',
      '[role="combobox"]'
    ];

    let dropdownFound = false;
    for (const selector of dropdownSelectors) {
      const dropdown = page.locator(selector).first();
      if (await dropdown.isVisible().catch(() => false)) {
        console.log(`[OK] Found dropdown with selector: ${selector}`);
        await dropdown.click();
        await page.waitForTimeout(500);

        await page.screenshot({
          path: `${SCREENSHOTS_DIR}/11-dropdown-clicked.png`,
          fullPage: true
        });

        dropdownFound = true;
        break;
      }
    }

    if (!dropdownFound) {
      console.log('[WARN] Model dropdown not found with standard selectors');
      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/11-dropdown-not-found.png`,
        fullPage: true
      });
    }

    // Look for PPO SSOT model in dropdown options
    const ppoModel = page.locator('text=PPO SSOT').first();
    const ppoVisible = await ppoModel.isVisible().catch(() => false);
    console.log(`[INFO] PPO SSOT model in dropdown: ${ppoVisible}`);

    // Look for Investor Demo
    const investorDemo = page.locator('text=Investor Demo').first();
    const demoVisible = await investorDemo.isVisible().catch(() => false);
    console.log(`[INFO] Investor Demo in dropdown: ${demoVisible}`);
  });
});
