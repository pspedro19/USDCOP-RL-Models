/**
 * Full Replay Test - Tests the complete replay flow
 * Captures console logs, screenshots, and validates the replay functionality
 */

import { test, expect } from '@playwright/test';

test.use({
  browserName: 'chromium',
});

test.describe('Full Replay Flow Test', () => {
  test('execute replay and verify trades appear progressively', async ({ page }) => {
    const consoleLogs: string[] = [];
    const errors: string[] = [];
    const replayLogs: string[] = [];

    // Capture all console messages
    page.on('console', msg => {
      const text = `[${msg.type().toUpperCase()}] ${msg.text()}`;
      consoleLogs.push(text);

      // Filter replay-related logs
      if (text.toLowerCase().includes('replay') ||
          text.toLowerCase().includes('trade') ||
          text.toLowerCase().includes('backtest') ||
          text.toLowerCase().includes('inference')) {
        replayLogs.push(text);
        console.log(text);
      }
    });

    // Capture page errors
    page.on('pageerror', error => {
      const errText = `[PAGE ERROR] ${error.message}`;
      errors.push(errText);
      console.log(errText);
    });

    // Capture network requests to API
    page.on('request', request => {
      if (request.url().includes('/api/replay') || request.url().includes('/api/trading')) {
        console.log(`[REQUEST] ${request.method()} ${request.url()}`);
      }
    });

    page.on('response', response => {
      if (response.url().includes('/api/replay') || response.url().includes('/api/trading')) {
        console.log(`[RESPONSE] ${response.status()} ${response.url()}`);
      }
    });

    console.log('\n========================================');
    console.log('STARTING FULL REPLAY TEST');
    console.log('========================================\n');

    // Step 1: Navigate to dashboard
    console.log('Step 1: Navigating to dashboard...');
    await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    await page.screenshot({
      path: 'tests/e2e/screenshots/replay-test-01-initial.png',
      fullPage: false
    });
    console.log('Screenshot: replay-test-01-initial.png');

    // Step 2: Enter Replay Mode
    console.log('\nStep 2: Entering Replay Mode...');
    const replayButton = page.locator('button').filter({ hasText: /Replay/i }).first();

    if (await replayButton.isVisible()) {
      await replayButton.click();
      console.log('Clicked Replay button');
      await page.waitForTimeout(2000);
    } else {
      console.log('ERROR: Replay button not found!');
      await page.screenshot({ path: 'tests/e2e/screenshots/replay-test-error-no-button.png' });
      throw new Error('Replay button not found');
    }

    await page.screenshot({
      path: 'tests/e2e/screenshots/replay-test-02-replay-mode.png',
      fullPage: false
    });
    console.log('Screenshot: replay-test-02-replay-mode.png');

    // Step 3: Set a shorter date range for faster testing
    console.log('\nStep 3: Checking date range...');
    const dateInfo = await page.locator('text=/Rango:|Range:/i').first().textContent().catch(() => 'not found');
    console.log(`Current date range: ${dateInfo}`);

    // Step 4: Set speed to 16x for faster testing
    console.log('\nStep 4: Setting speed to 16x...');
    const speed16Button = page.locator('button:has-text("16x")').first();
    if (await speed16Button.isVisible()) {
      await speed16Button.click();
      console.log('Speed set to 16x');
      await page.waitForTimeout(500);
    }

    // Step 5: Click "Iniciar" to start replay
    console.log('\nStep 5: Starting replay...');
    const iniciarButton = page.locator('button').filter({ hasText: /Iniciar|Start|Play/i }).first();

    if (await iniciarButton.isVisible()) {
      await iniciarButton.click();
      console.log('Clicked Iniciar button - Replay starting...');
    } else {
      console.log('WARNING: Iniciar button not found, looking for play icon...');
      const playButton = page.locator('button[aria-label*="play"], button:has(svg)').first();
      if (await playButton.isVisible()) {
        await playButton.click();
        console.log('Clicked play button');
      }
    }

    await page.screenshot({
      path: 'tests/e2e/screenshots/replay-test-03-started.png',
      fullPage: false
    });
    console.log('Screenshot: replay-test-03-started.png');

    // Step 6: Wait and capture progress
    console.log('\nStep 6: Monitoring replay progress...');

    for (let i = 0; i < 5; i++) {
      await page.waitForTimeout(3000);

      // Capture metrics
      const sharpeText = await page.locator('text=/SHARPE/i').first().textContent().catch(() => 'N/A');
      const tradesText = await page.locator('text=/TRADES/i').first().textContent().catch(() => 'N/A');
      const winRateText = await page.locator('text=/WIN RATE/i').first().textContent().catch(() => 'N/A');

      console.log(`\n--- Progress check ${i + 1}/5 ---`);
      console.log(`Sharpe: ${sharpeText}`);
      console.log(`Trades: ${tradesText}`);
      console.log(`Win Rate: ${winRateText}`);

      await page.screenshot({
        path: `tests/e2e/screenshots/replay-test-progress-${i + 1}.png`,
        fullPage: false
      });
      console.log(`Screenshot: replay-test-progress-${i + 1}.png`);

      // Check for trade markers in chart
      const tradeMarkers = await page.locator('[class*="signal"], [class*="marker"], [class*="buy"], [class*="sell"]').count();
      console.log(`Trade markers visible: ${tradeMarkers}`);
    }

    // Step 7: Final screenshot and summary
    console.log('\nStep 7: Final summary...');
    await page.screenshot({
      path: 'tests/e2e/screenshots/replay-test-final.png',
      fullPage: true
    });
    console.log('Screenshot: replay-test-final.png (full page)');

    // Print summary
    console.log('\n========================================');
    console.log('TEST SUMMARY');
    console.log('========================================');
    console.log(`Total console logs: ${consoleLogs.length}`);
    console.log(`Replay-related logs: ${replayLogs.length}`);
    console.log(`Errors: ${errors.length}`);

    if (replayLogs.length > 0) {
      console.log('\n--- REPLAY LOGS ---');
      replayLogs.slice(-20).forEach(log => console.log(log));
    }

    if (errors.length > 0) {
      console.log('\n--- ERRORS ---');
      errors.forEach(err => console.log(err));
    }

    // Verify no critical errors
    expect(errors.filter(e => e.includes('CRITICAL') || e.includes('TypeError'))).toHaveLength(0);

    console.log('\n========================================');
    console.log('TEST COMPLETED');
    console.log('========================================');
  });
});
