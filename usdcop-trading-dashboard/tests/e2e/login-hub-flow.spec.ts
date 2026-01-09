import { test, expect } from '@playwright/test';

/**
 * Login -> Hub -> Navigation Flow Test
 * Tests the complete user flow after authentication
 */
test('Complete Login to Hub Navigation Flow', async ({ page }) => {
  // Storage for all logs
  const consoleLogs: string[] = [];
  const networkErrors: string[] = [];
  const apiCalls: { url: string; status: number; method: string }[] = [];

  // Capture console messages
  page.on('console', (msg) => {
    const type = msg.type().toUpperCase();
    const text = msg.text();
    consoleLogs.push(`[${type}] ${text}`);
  });

  page.on('pageerror', (error) => {
    consoleLogs.push(`[PAGE_ERROR] ${error.message}`);
  });

  page.on('response', async (response) => {
    const url = response.url();
    const status = response.status();
    const method = response.request().method();

    if (url.includes('/api/') || url.includes('localhost:5000')) {
      apiCalls.push({ url, status, method });
    }
  });

  page.on('requestfailed', (request) => {
    networkErrors.push(`[FAILED] ${request.method()} ${request.url()} - ${request.failure()?.errorText}`);
  });

  await page.setViewportSize({ width: 1920, height: 1080 });

  // === STEP 1: Go to Login ===
  console.log('\n=== STEP 1: NAVIGATING TO LOGIN ===');
  await page.goto('http://localhost:5000/login', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  await page.screenshot({
    path: 'tests/e2e/screenshots/01-login-page.png',
    fullPage: true
  });
  console.log('Screenshot: 01-login-page.png');

  // === STEP 2: Fill Login Form ===
  console.log('\n=== STEP 2: FILLING LOGIN FORM ===');

  // Find username field by autocomplete attribute
  const usernameInput = page.locator('input[autocomplete="username"]');
  if (await usernameInput.count() === 0) {
    console.log('Could not find username input, trying alternative...');
  }
  await usernameInput.fill('admin');
  console.log('Filled username field with: admin');

  // Find password field
  const passwordInput = page.locator('input[type="password"]');
  await passwordInput.fill('admin123');
  console.log('Filled password field with: admin123');

  // Verify fields are filled
  const usernameValue = await usernameInput.inputValue();
  const passwordValue = await passwordInput.inputValue();
  console.log(`Username value: ${usernameValue}`);
  console.log(`Password value: ${passwordValue ? '********' : '(empty)'}`);

  await page.screenshot({
    path: 'tests/e2e/screenshots/02-login-filled.png',
    fullPage: true
  });
  console.log('Screenshot: 02-login-filled.png');

  // === STEP 3: Submit Login ===
  console.log('\n=== STEP 3: SUBMITTING LOGIN ===');

  // Find and click submit button
  const submitButton = page.locator('button[type="submit"], button:has-text("Sign In"), button:has-text("Login"), button:has-text("Iniciar")').first();
  await submitButton.click();

  // Wait for navigation to /hub
  try {
    await page.waitForURL('**/hub', { timeout: 10000 });
    console.log('Successfully navigated to /hub');
  } catch (e) {
    console.log('Navigation to /hub timed out, checking current URL...');
  }

  await page.waitForTimeout(1000);

  // Check current URL
  const urlAfterLogin = page.url();
  console.log(`URL after login: ${urlAfterLogin}`);

  await page.screenshot({
    path: 'tests/e2e/screenshots/03-after-login.png',
    fullPage: true
  });
  console.log('Screenshot: 03-after-login.png');

  // === STEP 4: Check if we're at Hub ===
  console.log('\n=== STEP 4: CHECKING HUB PAGE ===');

  if (urlAfterLogin.includes('/hub')) {
    console.log('SUCCESS: Redirected to /hub');

    // Wait for hub page to fully load
    await page.waitForSelector('h1:has-text("Terminal USD/COP")', { timeout: 5000 });

    // Look for the menu cards using data-testid
    const dashboardCard = await page.locator('[data-testid="hub-card-dashboard"]').count();
    const forecastingCard = await page.locator('[data-testid="hub-card-forecasting"]').count();

    console.log(`Dashboard card found: ${dashboardCard > 0}`);
    console.log(`Forecasting card found: ${forecastingCard > 0}`);

    await page.screenshot({
      path: 'tests/e2e/screenshots/04-hub-page.png',
      fullPage: true
    });
    console.log('Screenshot: 04-hub-page.png');

    // === STEP 5: Navigate to Dashboard ===
    console.log('\n=== STEP 5: NAVIGATING TO DASHBOARD ===');

    // Use data-testid for reliable selection
    const dashboardButton = page.locator('[data-testid="hub-card-dashboard"]');
    await dashboardButton.waitFor({ state: 'visible', timeout: 5000 });
    const buttonCount = await dashboardButton.count();
    console.log(`Dashboard button found: ${buttonCount > 0}`);

    if (buttonCount > 0) {
      // Debug: Get button info
      const boundingBox = await dashboardButton.boundingBox();
      console.log(`Button bounding box: ${JSON.stringify(boundingBox)}`);

      // Scroll into view and click
      await dashboardButton.scrollIntoViewIfNeeded();
      await dashboardButton.click();

      // Wait for navigation
      await page.waitForURL('**/dashboard', { timeout: 5000 }).catch(() => {
        console.log('Navigation to /dashboard timed out');
      });
      await page.waitForTimeout(1000);

      console.log(`URL after clicking Dashboard: ${page.url()}`);

      await page.screenshot({
        path: 'tests/e2e/screenshots/05-dashboard.png',
        fullPage: true
      });
      console.log('Screenshot: 05-dashboard.png');
    }

    // === STEP 6: Use navbar to go to Forecasting ===
    console.log('\n=== STEP 6: NAVIGATING TO FORECASTING VIA NAVBAR ===');

    const forecastingNavButton = page.locator('button:has-text("Forecasting"), nav >> text=Forecasting').first();
    if (await forecastingNavButton.count() > 0) {
      await forecastingNavButton.click();
      await page.waitForTimeout(3000);

      console.log(`URL after clicking Forecasting: ${page.url()}`);

      await page.screenshot({
        path: 'tests/e2e/screenshots/06-forecasting.png',
        fullPage: true
      });
      console.log('Screenshot: 06-forecasting.png');
    }

    // === STEP 7: Check logout in profile menu ===
    console.log('\n=== STEP 7: CHECKING PROFILE MENU ===');

    const profileButton = page.locator('[class*="profile"], button:has([class*="User"]), button:has-text("admin")').first();
    if (await profileButton.count() > 0) {
      await profileButton.click();
      await page.waitForTimeout(1000);

      await page.screenshot({
        path: 'tests/e2e/screenshots/07-profile-menu.png',
        fullPage: true
      });
      console.log('Screenshot: 07-profile-menu.png');
    }

  } else if (urlAfterLogin.includes('/dashboard')) {
    console.log('NOTE: Redirected to /dashboard instead of /hub');
    console.log('The redirect might not be applied correctly');
  } else {
    console.log(`UNEXPECTED: Ended up at ${urlAfterLogin}`);
  }

  // === PRINT RESULTS ===
  console.log('\n' + '='.repeat(80));
  console.log('CONSOLE LOGS (' + consoleLogs.length + ' total)');
  console.log('='.repeat(80));
  consoleLogs.slice(-30).forEach(log => console.log(log));

  const errors = consoleLogs.filter(l =>
    l.includes('[ERROR]') || l.includes('[PAGE_ERROR]') || l.toLowerCase().includes('error')
  );

  console.log('\n' + '='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total Console Logs: ${consoleLogs.length}`);
  console.log(`Errors: ${errors.length}`);
  console.log(`Network Errors: ${networkErrors.length}`);
  console.log(`API Calls: ${apiCalls.length}`);

  if (errors.length > 0) {
    console.log('\n--- ERRORS ---');
    errors.forEach(e => console.log(e));
  }

  if (networkErrors.length > 0) {
    console.log('\n--- NETWORK ERRORS ---');
    networkErrors.forEach(e => console.log(e));
  }
});
