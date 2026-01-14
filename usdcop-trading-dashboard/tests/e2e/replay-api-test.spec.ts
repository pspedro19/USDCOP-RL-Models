/**
 * Replay API Test
 * Tests that the replay functionality correctly loads data and plays
 */

import { test, expect } from '@playwright/test';

test.describe('Replay Mode API', () => {
  test('should load replay data when clicking Iniciar button', async ({ page }) => {
    // Collect all console messages
    const consoleMessages: string[] = [];
    page.on('console', (msg) => {
      consoleMessages.push(`[${msg.type()}] ${msg.text()}`);
    });

    // Collect network requests
    const apiRequests: string[] = [];
    page.on('request', (request) => {
      if (request.url().includes('/api/')) {
        apiRequests.push(`${request.method()} ${request.url()}`);
      }
    });

    const apiResponses: { url: string; status: number }[] = [];
    page.on('response', (response) => {
      if (response.url().includes('/api/')) {
        apiResponses.push({ url: response.url(), status: response.status() });
      }
    });

    // Navigate to dashboard
    await page.goto('http://localhost:3001/dashboard');
    await page.waitForLoadState('networkidle');

    // Take initial screenshot
    await page.screenshot({ path: 'tests/e2e/screenshots/replay-test-1-initial.png', fullPage: true });

    // Click on "Replay Mode" button to enable replay mode
    const replayModeButton = page.locator('button:has-text("Replay Mode")');
    await expect(replayModeButton).toBeVisible({ timeout: 10000 });
    await replayModeButton.click();

    // Wait for replay control bar to appear
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'tests/e2e/screenshots/replay-test-2-replay-mode.png', fullPage: true });

    // Find and click the "Iniciar" button
    const iniciarButton = page.locator('button:has-text("Iniciar")');
    await expect(iniciarButton).toBeVisible({ timeout: 5000 });

    // Check if button is enabled
    const isDisabled = await iniciarButton.isDisabled();
    console.log('Iniciar button disabled:', isDisabled);

    // Click the button
    await iniciarButton.click();

    // Wait for API calls to complete
    await page.waitForTimeout(5000);

    // Take screenshot after clicking
    await page.screenshot({ path: 'tests/e2e/screenshots/replay-test-3-after-iniciar.png', fullPage: true });

    // Log all API requests and responses
    console.log('\n=== API Requests ===');
    apiRequests.forEach(req => console.log(req));

    console.log('\n=== API Responses ===');
    apiResponses.forEach(res => console.log(`${res.status} ${res.url}`));

    // Check for 404 errors
    const failed404 = apiResponses.filter(r => r.status === 404);
    console.log('\n=== 404 Errors ===');
    failed404.forEach(res => console.log(res.url));

    // Log console messages
    console.log('\n=== Console Messages ===');
    consoleMessages.forEach(msg => console.log(msg));

    // Verify no 404 errors from replay-specific APIs
    const replayApiFailed = failed404.filter(r =>
      r.url.includes('/trades/history') ||
      r.url.includes('/equity-curve') ||
      r.url.includes('/price-data') ||
      r.url.includes('/candlesticks')
    );

    expect(replayApiFailed.length, `Found ${replayApiFailed.length} failed replay API calls: ${replayApiFailed.map(r => r.url).join(', ')}`).toBe(0);

    // Verify at least one successful trades API call
    const successfulTradesCall = apiResponses.find(r =>
      r.url.includes('/trading/trades/history') && r.status === 200
    );
    expect(successfulTradesCall, 'Expected at least one successful trades API call').toBeDefined();
  });
});
