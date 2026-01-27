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
  test.setTimeout(120000);
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
      // Use native input setter to trigger React's onChange
      await usernameInput.focus();
      await page.evaluate(() => {
        const input = document.querySelector('input[autocomplete="username"]') as HTMLInputElement;
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;
        nativeInputValueSetter.call(input, 'admin');
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      });

      await passwordInput.focus();
      await page.evaluate(() => {
        const input = document.querySelector('input[type="password"]') as HTMLInputElement;
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;
        nativeInputValueSetter.call(input, 'admin123');
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      });

      await page.waitForTimeout(500);
      console.log('Button disabled?', await loginButton.isDisabled());

      if (await loginButton.isDisabled()) {
        // Fallback: set auth directly and navigate
        console.log('Button still disabled, setting auth directly');
        await page.evaluate(() => {
          localStorage.setItem('isAuthenticated', 'true');
          sessionStorage.setItem('isAuthenticated', 'true');
          localStorage.setItem('username', 'admin');
          sessionStorage.setItem('username', 'admin');
        });
        await page.goto('http://localhost:5000/hub', { waitUntil: 'domcontentloaded' });
      } else {
        await loginButton.click();
      }
      console.log('Login completed');

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

    // Wait for the ModelDropdown button to appear (it shows a skeleton while loading)
    // The button contains badge text like DEPLOYED, DEMO, TESTING
    console.log('Waiting for model dropdown button...');

    // The dropdown button has badge text - wait for it to appear (up to 30s for HMR in dev mode)
    const dropdownButton = page.locator('button').filter({ hasText: /DEPLOYED|DEMO|TESTING|En Producción|Select Model/i }).first();
    try {
      await dropdownButton.waitFor({ state: 'visible', timeout: 30000 });
      console.log('Model dropdown button appeared');
    } catch {
      console.log('Model dropdown button did NOT appear after 30s');

      // Debug: list all buttons
      const allButtons = await page.locator('button').all();
      console.log(`Total buttons: ${allButtons.length}`);
      for (const btn of allButtons) {
        const text = (await btn.textContent())?.trim().substring(0, 80);
        console.log(`  Button: "${text}"`);
      }

      // Check body for model text
      const bodyText = await page.textContent('body') || '';
      console.log(`Page has DEPLOYED: ${bodyText.includes('DEPLOYED')}`);
      console.log(`Page has PPO: ${bodyText.includes('PPO')}`);
      console.log(`Page has Select Model: ${bodyText.includes('Select Model')}`);
      console.log(`Page has No models: ${bodyText.includes('No models')}`);
      console.log(`Page has animate-pulse: ${(await page.locator('.animate-pulse').count()) > 0}`);

      await page.screenshot({
        path: 'tests/e2e/screenshots/investor-debug-04-no-dropdown.png',
        fullPage: true
      });
    }

    const dropdownVisible = await dropdownButton.isVisible().catch(() => false);
    if (dropdownVisible) {
      const btnText = await dropdownButton.textContent();
      console.log(`Dropdown button text: "${btnText?.trim()}"`);
      await dropdownButton.click();
      await page.waitForTimeout(1000);

      await page.screenshot({
        path: 'tests/e2e/screenshots/investor-debug-04-dropdown-open.png',
        fullPage: true
      });

      // Dropdown items are divs with cursor-pointer class
      const investorOption = page.locator('div[class*="cursor-pointer"]').filter({ hasText: /investor|demo/i }).first();
      if (await investorOption.isVisible({ timeout: 3000 }).catch(() => false)) {
        await investorOption.click();
        console.log('Selected investor demo model');
        await page.waitForTimeout(2000);
      } else {
        console.log('Investor option not found in dropdown');
        const dropdownItems = await page.locator('div[class*="cursor-pointer"]').all();
        console.log(`Found ${dropdownItems.length} cursor-pointer divs`);
        for (const item of dropdownItems) {
          const t = (await item.textContent())?.trim().substring(0, 60);
          console.log(`  Dropdown item: "${t}"`);
        }
      }
    }

    await page.screenshot({
      path: 'tests/e2e/screenshots/investor-debug-05-model-selected.png',
      fullPage: true
    });

    // Step 5: Click "Iniciar" (Start Backtest/Replay) button
    console.log('\n=== STEP 5: START BACKTEST/REPLAY ===');

    // The start button is labeled "Iniciar" - Backtest panel is expanded by default
    await page.waitForTimeout(2000);
    // Scroll to top to ensure backtest controls are visible
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(500);

    const iniciarButton = page.locator('button').filter({ hasText: /Iniciar/i }).first();

    if (await iniciarButton.isVisible({ timeout: 10000 }).catch(() => false)) {
      console.log('Iniciar button found, clicking...');
      await iniciarButton.click();
      await page.waitForTimeout(10000); // Wait for backtest to run

      await page.screenshot({
        path: 'tests/e2e/screenshots/investor-debug-06-replay-started.png',
        fullPage: true
      });
    } else {
      console.log('Iniciar button NOT visible');
      await page.screenshot({
        path: 'tests/e2e/screenshots/investor-debug-06-no-iniciar-button.png',
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
      try {
        const errorDialog = page.locator('[role="dialog"], [class*="error"], [class*="alert"]').first();
        if (await errorDialog.isVisible({ timeout: 2000 }).catch(() => false)) {
          await errorDialog.screenshot({
            path: 'tests/e2e/screenshots/investor-debug-08-error-dialog.png'
          });
        }
      } catch {
        console.log('Error dialog screenshot failed (element may have detached)');
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
