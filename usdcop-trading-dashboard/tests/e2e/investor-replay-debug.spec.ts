/**
 * Investor Mode Replay Debug Test
 * Captures console logs, network requests, and screenshots to debug
 * the "Authentication required" error when replaying investor mode.
 */

import { test, expect } from '@playwright/test';

test.use({
  browserName: 'chromium',
});

test.describe('Investor Mode Replay Debug', () => {
  test('debug investor mode replay authentication error', async ({ page }) => {
    const consoleLogs: string[] = [];
    const networkRequests: { url: string; status: number | null; error?: string }[] = [];
    const errors: string[] = [];

    // Capture all console messages
    page.on('console', msg => {
      const text = `[${msg.type().toUpperCase()}] ${msg.text()}`;
      consoleLogs.push(text);
      console.log(text);
    });

    // Capture page errors
    page.on('pageerror', error => {
      const errorMsg = `[PAGE ERROR] ${error.message}`;
      errors.push(errorMsg);
      console.log(errorMsg);
    });

    // Capture network requests and responses
    page.on('request', request => {
      const url = request.url();
      if (url.includes('api') || url.includes('backtest') || url.includes('models') || url.includes('replay')) {
        console.log(`[REQUEST] ${request.method()} ${url}`);
      }
    });

    page.on('response', response => {
      const url = response.url();
      if (url.includes('api') || url.includes('backtest') || url.includes('models') || url.includes('replay')) {
        const status = response.status();
        networkRequests.push({ url, status });
        console.log(`[RESPONSE] ${status} ${url}`);

        // Log response body for errors
        if (status >= 400) {
          response.text().then(text => {
            console.log(`[ERROR RESPONSE BODY] ${text}`);
          }).catch(() => {});
        }
      }
    });

    page.on('requestfailed', request => {
      const url = request.url();
      const failure = request.failure();
      if (url.includes('api') || url.includes('backtest') || url.includes('models') || url.includes('replay')) {
        const errorMsg = `[REQUEST FAILED] ${url} - ${failure?.errorText || 'Unknown error'}`;
        networkRequests.push({ url, status: null, error: failure?.errorText });
        errors.push(errorMsg);
        console.log(errorMsg);
      }
    });

    // Step 1: Login
    console.log('\n=== STEP 1: LOGIN ===');
    await page.goto('http://localhost:5000/login', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: 'tests/e2e/screenshots/investor-debug-01-login.png',
      fullPage: true
    });

    const usernameInput = page.locator('input[autocomplete="username"]');
    const passwordInput = page.locator('input[type="password"]');
    const loginButton = page.locator('button[type="submit"]');

    if (await usernameInput.isVisible()) {
      await usernameInput.fill('admin');
      await passwordInput.fill('admin123');
      await loginButton.click();
      console.log('Submitted login form');

      await page.waitForTimeout(3000);
      console.log(`Current URL after login: ${page.url()}`);
    }

    await page.screenshot({
      path: 'tests/e2e/screenshots/investor-debug-02-after-login.png',
      fullPage: true
    });

    // Step 2: Navigate to Dashboard
    console.log('\n=== STEP 2: NAVIGATE TO DASHBOARD ===');
    await page.goto('http://localhost:5000/dashboard', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(3000);

    await page.screenshot({
      path: 'tests/e2e/screenshots/investor-debug-03-dashboard.png',
      fullPage: true
    });

    // Step 3: Check models API
    console.log('\n=== STEP 3: CHECK MODELS API ===');
    try {
      const modelsResponse = await page.request.get('http://localhost:5000/api/models');
      const modelsData = await modelsResponse.json();
      console.log('Models API Response:', JSON.stringify(modelsData, null, 2));
    } catch (e) {
      console.log('Models API Error:', e);
    }

    // Step 4: Find and select Investor Demo model
    console.log('\n=== STEP 4: SELECT INVESTOR MODEL ===');

    // Look for model dropdown
    const modelButtons = await page.locator('button').filter({ hasText: /model|PPO|investor/i }).all();
    console.log(`Found ${modelButtons.length} model-related buttons`);

    for (const btn of modelButtons) {
      const text = await btn.textContent();
      console.log(`Button: ${text}`);
    }

    // Try to click dropdown and select investor_demo
    const dropdownTrigger = page.locator('[class*="dropdown"], button:has-text("PPO"), button:has-text("Model")').first();
    if (await dropdownTrigger.isVisible()) {
      await dropdownTrigger.click();
      await page.waitForTimeout(500);

      await page.screenshot({
        path: 'tests/e2e/screenshots/investor-debug-04-dropdown-open.png',
        fullPage: true
      });

      // Look for investor option
      const investorOption = page.locator('[role="option"], li, button').filter({ hasText: /investor|demo/i }).first();
      if (await investorOption.isVisible()) {
        await investorOption.click();
        console.log('Selected investor demo model');
        await page.waitForTimeout(1000);
      }
    }

    await page.screenshot({
      path: 'tests/e2e/screenshots/investor-debug-05-model-selected.png',
      fullPage: true
    });

    // Step 5: Click Replay button
    console.log('\n=== STEP 5: START REPLAY ===');
    const replayButton = page.locator('button').filter({ hasText: /Replay/i }).first();

    if (await replayButton.isVisible()) {
      console.log('Replay button found, clicking...');
      await replayButton.click();
      await page.waitForTimeout(5000);

      await page.screenshot({
        path: 'tests/e2e/screenshots/investor-debug-06-replay-clicked.png',
        fullPage: true
      });
    } else {
      console.log('Replay button NOT visible');

      // Take screenshot of current state
      await page.screenshot({
        path: 'tests/e2e/screenshots/investor-debug-06-no-replay-button.png',
        fullPage: true
      });
    }

    // Step 6: Wait and capture any errors
    console.log('\n=== STEP 6: WAITING FOR ERRORS ===');
    await page.waitForTimeout(5000);

    await page.screenshot({
      path: 'tests/e2e/screenshots/investor-debug-07-final-state.png',
      fullPage: true
    });

    // Check for error messages on the page
    const pageContent = await page.textContent('body');
    if (pageContent?.toLowerCase().includes('authentication') ||
        pageContent?.toLowerCase().includes('error') ||
        pageContent?.toLowerCase().includes('unauthorized')) {
      console.log('FOUND ERROR TEXT ON PAGE');

      // Screenshot any visible error dialog
      const errorDialog = page.locator('[role="dialog"], [class*="error"], [class*="alert"]').first();
      if (await errorDialog.isVisible()) {
        await errorDialog.screenshot({
          path: 'tests/e2e/screenshots/investor-debug-08-error-dialog.png'
        });
      }
    }

    // Final summary
    console.log('\n========================================');
    console.log('=== DEBUG SUMMARY ===');
    console.log('========================================');
    console.log(`Total console logs: ${consoleLogs.length}`);
    console.log(`Total network requests: ${networkRequests.length}`);
    console.log(`Total errors: ${errors.length}`);

    // Print failed requests
    const failedRequests = networkRequests.filter(r => r.status === null || r.status >= 400);
    if (failedRequests.length > 0) {
      console.log('\n=== FAILED REQUESTS ===');
      failedRequests.forEach(r => {
        console.log(`${r.status || 'FAILED'} ${r.url} ${r.error || ''}`);
      });
    }

    // Print authentication-related logs
    const authLogs = consoleLogs.filter(log =>
      log.toLowerCase().includes('auth') ||
      log.toLowerCase().includes('401') ||
      log.toLowerCase().includes('403') ||
      log.toLowerCase().includes('network error')
    );
    if (authLogs.length > 0) {
      console.log('\n=== AUTH-RELATED LOGS ===');
      authLogs.forEach(log => console.log(log));
    }

    // Print all errors
    if (errors.length > 0) {
      console.log('\n=== ALL ERRORS ===');
      errors.forEach(err => console.log(err));
    }

    // The test passes - we're just debugging
    expect(true).toBe(true);
  });
});
