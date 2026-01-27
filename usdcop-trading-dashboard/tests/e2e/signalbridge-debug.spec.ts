import { test, expect } from '@playwright/test';

test.describe('SignalBridge Debug - Console & Network Monitoring', () => {
  test('Monitor SignalBridge pages for errors', async ({ page }) => {
    const consoleLogs: string[] = [];
    const networkErrors: string[] = [];
    const apiCalls: { url: string; status: number; method: string }[] = [];

    // Capture console logs
    page.on('console', (msg) => {
      const text = `[${msg.type().toUpperCase()}] ${msg.text()}`;
      consoleLogs.push(text);
      console.log(text);
    });

    // Capture page errors
    page.on('pageerror', (error) => {
      const text = `[PAGE ERROR] ${error.message}`;
      consoleLogs.push(text);
      console.log(text);
    });

    // Capture network requests and responses
    page.on('response', async (response) => {
      const url = response.url();
      const status = response.status();
      const method = response.request().method();

      // Log all API calls
      if (url.includes('/api/')) {
        apiCalls.push({ url, status, method });
        console.log(`[API ${method}] ${url} -> ${status}`);

        if (status >= 400) {
          let body = '';
          try {
            body = await response.text();
          } catch (e) {
            body = 'Could not read body';
          }
          networkErrors.push(`${method} ${url} -> ${status}: ${body}`);
          console.log(`[ERROR RESPONSE] ${body}`);
        }
      }
    });

    // Step 1: Login
    console.log('\n=== Step 1: Login ===');
    await page.goto('http://localhost:5000/login');
    await page.waitForLoadState('networkidle');

    // Fill login form
    await page.fill('input[name="username"], input[type="text"]', 'admin');
    await page.fill('input[name="password"], input[type="password"]', 'admin123');

    // Click login button
    await page.click('button[type="submit"]');

    // Wait for redirect
    await page.waitForURL('**/hub**', { timeout: 10000 }).catch(() => {
      console.log('Did not redirect to hub, checking current URL...');
    });

    console.log(`Current URL after login: ${page.url()}`);
    await page.waitForTimeout(2000);

    // Step 2: Navigate to SignalBridge
    console.log('\n=== Step 2: Navigate to SignalBridge ===');

    // Try clicking SignalBridge link
    const signalbridgeLink = page.locator('a[href*="execution"], a:has-text("SignalBridge"), a:has-text("Execution")').first();
    if (await signalbridgeLink.isVisible()) {
      await signalbridgeLink.click();
    } else {
      // Direct navigation
      await page.goto('http://localhost:5000/execution');
    }

    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    console.log(`Current URL: ${page.url()}`);

    // Step 3: Test Dashboard page
    console.log('\n=== Step 3: Testing Dashboard ===');
    await page.goto('http://localhost:5000/execution/dashboard');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    // Check for errors on page
    const dashboardContent = await page.textContent('body');
    if (dashboardContent?.includes('Connection Error') || dashboardContent?.includes('404')) {
      console.log('[DASHBOARD ERROR] Found error message on page');
    }

    // Step 4: Test Exchanges page
    console.log('\n=== Step 4: Testing Exchanges ===');
    await page.goto('http://localhost:5000/execution/exchanges');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    const exchangesContent = await page.textContent('body');
    if (exchangesContent?.includes('404') || exchangesContent?.includes('Error')) {
      console.log('[EXCHANGES ERROR] Found error message on page');
    }

    // Step 5: Test Executions page
    console.log('\n=== Step 5: Testing Executions ===');
    await page.goto('http://localhost:5000/execution/executions');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    const executionsContent = await page.textContent('body');
    if (executionsContent?.includes('Unauthorized')) {
      console.log('[EXECUTIONS ERROR] Found Unauthorized message');
    }

    // Step 6: Test Settings/Risk page
    console.log('\n=== Step 6: Testing Settings ===');
    await page.goto('http://localhost:5000/execution/settings');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    const settingsContent = await page.textContent('body');
    if (settingsContent?.includes('Failed to load')) {
      console.log('[SETTINGS ERROR] Found "Failed to load" message');
    }

    // Summary
    console.log('\n========== SUMMARY ==========');
    console.log(`Total API calls: ${apiCalls.length}`);
    console.log(`Network errors: ${networkErrors.length}`);
    console.log(`Console logs: ${consoleLogs.length}`);

    console.log('\n--- API Calls ---');
    apiCalls.forEach(call => {
      const status = call.status >= 400 ? `❌ ${call.status}` : `✓ ${call.status}`;
      console.log(`${status} ${call.method} ${call.url}`);
    });

    console.log('\n--- Network Errors ---');
    networkErrors.forEach(err => console.log(err));

    console.log('\n--- Console Errors ---');
    consoleLogs.filter(log => log.includes('ERROR') || log.includes('error')).forEach(log => console.log(log));
  });
});
