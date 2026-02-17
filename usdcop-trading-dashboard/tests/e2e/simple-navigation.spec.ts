import { test, expect } from '@playwright/test';

/**
 * Simple Navigation Test - Direct Access (Dev Mode)
 * Since NODE_ENV=development bypasses auth, we test direct navigation
 */

const SCREENSHOTS_DIR = 'tests/e2e/screenshots/navigation-flow';

test.describe('Simple Navigation (Dev Mode)', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.error(`[BROWSER ERROR] ${msg.text()}`);
      }
    });
    page.on('pageerror', error => {
      console.error(`[PAGE ERROR] ${error.message}`);
    });
  });

  test('Navigate all pages directly', async ({ page }) => {
    // Set auth in localStorage before navigating
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('username', 'admin');
      sessionStorage.setItem('isAuthenticated', 'true');
      sessionStorage.setItem('username', 'admin');
    });

    // ============ DASHBOARD ============
    console.log('\n=== TESTING DASHBOARD ===');
    await page.goto('/dashboard');
    await page.waitForTimeout(3000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/01-dashboard.png`,
      fullPage: true
    });

    const dashboardTitle = await page.locator('h1, h2').first().textContent();
    console.log(`Dashboard title: ${dashboardTitle}`);

    // Check for key dashboard elements
    const hasChart = await page.locator('canvas, [data-testid="chart"]').first().isVisible().catch(() => false);
    console.log(`Chart visible: ${hasChart}`);

    // ============ HUB ============
    console.log('\n=== TESTING HUB ===');
    await page.goto('/hub');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/02-hub.png`,
      fullPage: true
    });

    // ============ PRODUCTION ============
    console.log('\n=== TESTING PRODUCTION ===');
    await page.goto('/production');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/03-production.png`,
      fullPage: true
    });

    // ============ FORECASTING ============
    console.log('\n=== TESTING FORECASTING ===');
    await page.goto('/forecasting');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/04-forecasting.png`,
      fullPage: true
    });

    // ============ LOGIN ============
    console.log('\n=== TESTING LOGIN PAGE ===');
    await page.goto('/login');
    await page.waitForTimeout(2000);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/05-login.png`,
      fullPage: true
    });

    console.log('\n=== ALL PAGES CAPTURED ===');
  });

  test('Test login flow manually', async ({ page }) => {
    console.log('\n=== MANUAL LOGIN TEST ===');

    // Go to login
    await page.goto('/login');
    await page.waitForTimeout(2000);

    // Check what inputs exist
    const inputs = await page.locator('input').all();
    console.log(`Found ${inputs.length} inputs on login page`);

    for (let i = 0; i < inputs.length; i++) {
      const type = await inputs[i].getAttribute('type');
      const name = await inputs[i].getAttribute('name');
      const placeholder = await inputs[i].getAttribute('placeholder');
      const autocomplete = await inputs[i].getAttribute('autocomplete');
      console.log(`  Input ${i}: type=${type}, name=${name}, placeholder=${placeholder}, autocomplete=${autocomplete}`);
    }

    // Try to fill the first text input with admin
    const usernameInput = page.locator('input').first();
    await usernameInput.fill('admin');

    // Find password input
    const passwordInput = page.locator('input[type="password"]').first();
    if (await passwordInput.isVisible()) {
      await passwordInput.fill('admin123');
    }

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/06-login-filled.png`,
      fullPage: true
    });

    // Find and click submit button
    const buttons = await page.locator('button').all();
    console.log(`Found ${buttons.length} buttons`);
    for (let i = 0; i < buttons.length; i++) {
      const text = await buttons[i].textContent();
      console.log(`  Button ${i}: "${text}"`);
    }

    // Click the login button
    const loginBtn = page.locator('button:has-text("Iniciar"), button:has-text("Login"), button[type="submit"]').first();
    if (await loginBtn.isVisible()) {
      console.log('Clicking login button...');
      await loginBtn.click();
      await page.waitForTimeout(3000);

      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/07-after-login-click.png`,
        fullPage: true
      });

      console.log(`Current URL after login: ${page.url()}`);
    }
  });

  test('Check model dropdown in dashboard', async ({ page }) => {
    console.log('\n=== MODEL DROPDOWN TEST ===');

    // Set auth
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('username', 'admin');
    });

    // Go to dashboard
    await page.goto('/dashboard');
    await page.waitForTimeout(3000);

    // Look for any dropdown/select elements
    const selects = await page.locator('select').all();
    console.log(`Found ${selects.length} select elements`);

    const buttons = await page.locator('button').all();
    console.log(`Found ${buttons.length} buttons`);

    // Look for model-related text
    const modelText = await page.locator('text=Investor Demo, text=PPO, text=modelo, text=Model').first().isVisible().catch(() => false);
    console.log(`Model-related text visible: ${modelText}`);

    // Try to find the floating experiment panel
    const floatingPanel = await page.locator('[class*="fixed"][class*="bottom"], [class*="floating"]').first().isVisible().catch(() => false);
    console.log(`Floating panel visible: ${floatingPanel}`);

    // Take screenshot of current state
    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/08-dashboard-elements.png`,
      fullPage: true
    });

    // Click on anything that looks like a model selector
    const modelButton = page.locator('button:has-text("Seleccionar"), button:has-text("Investor"), button:has-text("PPO")').first();
    if (await modelButton.isVisible().catch(() => false)) {
      await modelButton.click();
      await page.waitForTimeout(1000);
      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/09-model-dropdown-open.png`,
        fullPage: true
      });
    }
  });
});
