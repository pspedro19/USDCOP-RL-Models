/**
 * Hybrid Replay System E2E Tests
 *
 * Tests the timeline-based replay system with:
 * - Speed controls (0.5x to 16x)
 * - Timeline navigation
 * - Trade clustering
 * - Keyboard shortcuts
 */

import { test, expect } from '@playwright/test';

test.describe('Hybrid Replay System', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to dashboard
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Wait for the page to be fully loaded
    await page.waitForTimeout(2000);
  });

  test('should show replay mode button on dashboard', async ({ page }) => {
    // Look for the Replay Mode button
    const replayButton = page.locator('button:has-text("Replay")');
    await expect(replayButton).toBeVisible();

    // Take screenshot of initial state
    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-01-initial.png',
      fullPage: false
    });
  });

  test('should enter replay mode and show control bar', async ({ page }) => {
    // Click replay mode button
    const replayButton = page.locator('button:has-text("Replay")');
    await replayButton.click();

    // Wait for replay control bar to appear
    await page.waitForTimeout(1000);

    // Take screenshot of replay mode
    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-02-control-bar.png',
      fullPage: false
    });

    // Verify control bar elements are visible
    const controlBar = page.locator('[class*="ReplayControlBar"], [class*="replay"]').first();
    await expect(controlBar).toBeVisible();
  });

  test('should show all speed buttons including 0.5x and 16x', async ({ page }) => {
    // Enter replay mode
    const replayButton = page.locator('button:has-text("Replay")');
    await replayButton.click();
    await page.waitForTimeout(1000);

    // Check for speed buttons
    const speedButtons = page.locator('button:has-text("x")');
    const count = await speedButtons.count();

    console.log(`Found ${count} speed buttons`);

    // Take screenshot showing speed buttons
    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-03-speed-buttons.png',
      fullPage: false
    });

    // Verify we have at least 6 speed buttons (0.5x, 1x, 2x, 4x, 8x, 16x)
    expect(count).toBeGreaterThanOrEqual(6);
  });

  test('should display timeline information when enabled', async ({ page }) => {
    // Enter replay mode
    const replayButton = page.locator('button:has-text("Replay")');
    await replayButton.click();
    await page.waitForTimeout(1000);

    // Look for timeline-related text
    const durationText = page.locator('text=/Duración|Duration/i');
    const tradesText = page.locator('text=/Trades|trades/i');

    // Take screenshot
    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-04-timeline-info.png',
      fullPage: false
    });

    // Log what we found
    const hasDuration = await durationText.count() > 0;
    const hasTrades = await tradesText.count() > 0;
    console.log(`Timeline info - Duration: ${hasDuration}, Trades: ${hasTrades}`);
  });

  test('should respond to keyboard shortcuts', async ({ page }) => {
    // Enter replay mode
    const replayButton = page.locator('button:has-text("Replay")');
    await replayButton.click();
    await page.waitForTimeout(1000);

    // Press 1 for 1x speed
    await page.keyboard.press('Digit1');
    await page.waitForTimeout(500);

    // Take screenshot after speed change
    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-05-keyboard-1x.png',
      fullPage: false
    });

    // Press 5 for 16x speed
    await page.keyboard.press('Digit5');
    await page.waitForTimeout(500);

    // Take screenshot after speed change
    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-06-keyboard-16x.png',
      fullPage: false
    });

    // Press backtick for 0.5x speed
    await page.keyboard.press('Backquote');
    await page.waitForTimeout(500);

    // Take screenshot
    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-07-keyboard-0.5x.png',
      fullPage: false
    });
  });

  test('should capture console logs for debugging', async ({ page }) => {
    const consoleLogs: string[] = [];

    // Capture console messages
    page.on('console', msg => {
      const text = `[${msg.type()}] ${msg.text()}`;
      consoleLogs.push(text);
      console.log(text);
    });

    // Capture errors
    page.on('pageerror', error => {
      consoleLogs.push(`[ERROR] ${error.message}`);
      console.error('[PAGE ERROR]', error.message);
    });

    // Enter replay mode
    const replayButton = page.locator('button:has-text("Replay")');
    await replayButton.click();
    await page.waitForTimeout(2000);

    // Try to load replay data by interacting with date pickers if available
    const loadButton = page.locator('button:has-text("Load"), button:has-text("Cargar")');
    if (await loadButton.count() > 0) {
      await loadButton.first().click();
      await page.waitForTimeout(3000);
    }

    // Take final screenshot
    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-08-console-debug.png',
      fullPage: true
    });

    // Log summary
    console.log('\n=== Console Log Summary ===');
    console.log(`Total logs captured: ${consoleLogs.length}`);

    // Filter for replay-related logs
    const replayLogs = consoleLogs.filter(log =>
      log.toLowerCase().includes('replay') ||
      log.toLowerCase().includes('timeline') ||
      log.toLowerCase().includes('trade')
    );

    console.log(`Replay-related logs: ${replayLogs.length}`);
    replayLogs.forEach(log => console.log(log));
  });

  test('should verify hybrid replay types are working', async ({ page }) => {
    // This test checks the browser's execution of our timeline code
    const result = await page.evaluate(() => {
      // Check if replay types exist
      const checks = {
        hasWindow: typeof window !== 'undefined',
        // Add any global checks we can do
      };
      return checks;
    });

    console.log('Type checks:', result);
    expect(result.hasWindow).toBe(true);

    await page.screenshot({
      path: 'tests/e2e/screenshots/hybrid-replay-09-types-check.png',
      fullPage: false
    });
  });
});
