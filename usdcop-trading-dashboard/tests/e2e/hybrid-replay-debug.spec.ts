/**
 * Hybrid Replay Debug Test
 * Simple test to capture screenshots and console logs
 */

import { test, expect } from '@playwright/test';

test.use({
  // Only use chromium for faster testing
  browserName: 'chromium',
});

test.describe('Hybrid Replay Debug', () => {
  test('capture replay mode state and console logs', async ({ page }) => {
    const consoleLogs: string[] = [];
    const errors: string[] = [];

    // Capture all console messages
    page.on('console', msg => {
      const text = `[${msg.type().toUpperCase()}] ${msg.text()}`;
      consoleLogs.push(text);
    });

    // Capture page errors
    page.on('pageerror', error => {
      errors.push(`[PAGE ERROR] ${error.message}`);
    });

    // Navigate to dashboard
    console.log('Navigating to dashboard...');
    await page.goto('http://localhost:5000/dashboard', { waitUntil: 'domcontentloaded' });

    // Wait for initial load
    await page.waitForTimeout(3000);

    // Screenshot 1: Initial dashboard state
    await page.screenshot({
      path: 'tests/e2e/screenshots/debug-01-dashboard-initial.png',
      fullPage: false
    });
    console.log('Screenshot 1: Initial dashboard');

    // Find and click the Replay Mode button
    console.log('Looking for Replay button...');
    const replayButton = page.locator('button').filter({ hasText: /Replay/i }).first();

    const buttonVisible = await replayButton.isVisible().catch(() => false);
    console.log(`Replay button visible: ${buttonVisible}`);

    if (buttonVisible) {
      await replayButton.click();
      console.log('Clicked Replay button');
      await page.waitForTimeout(2000);

      // Screenshot 2: After entering replay mode
      await page.screenshot({
        path: 'tests/e2e/screenshots/debug-02-replay-mode.png',
        fullPage: false
      });
      console.log('Screenshot 2: Replay mode active');

      // Look for speed buttons
      const allButtons = await page.locator('button').all();
      console.log(`Total buttons on page: ${allButtons.length}`);

      // Check for specific speed values
      for (const speed of ['0.5', '1', '2', '4', '8', '16']) {
        const speedBtn = page.locator(`button:has-text("${speed}x")`);
        const count = await speedBtn.count();
        console.log(`Speed ${speed}x button count: ${count}`);
      }

      // Screenshot 3: Focus on control area
      await page.screenshot({
        path: 'tests/e2e/screenshots/debug-03-controls.png',
        fullPage: false
      });

      // Try clicking 16x speed
      const speed16 = page.locator('button:has-text("16x")').first();
      if (await speed16.isVisible().catch(() => false)) {
        await speed16.click();
        console.log('Clicked 16x speed button');
        await page.waitForTimeout(500);
      }

      // Screenshot 4: After speed change
      await page.screenshot({
        path: 'tests/e2e/screenshots/debug-04-speed-16x.png',
        fullPage: false
      });

      // Try clicking 0.5x speed
      const speed05 = page.locator('button:has-text("0.5x")').first();
      if (await speed05.isVisible().catch(() => false)) {
        await speed05.click();
        console.log('Clicked 0.5x speed button');
        await page.waitForTimeout(500);
      }

      // Screenshot 5: After 0.5x speed
      await page.screenshot({
        path: 'tests/e2e/screenshots/debug-05-speed-0.5x.png',
        fullPage: false
      });

      // Check for timeline info
      const pageContent = await page.content();
      const hasEstimatedDuration = pageContent.includes('Duración') || pageContent.includes('Duration');
      const hasGroups = pageContent.includes('grupo') || pageContent.includes('group');
      console.log(`Has estimated duration text: ${hasEstimatedDuration}`);
      console.log(`Has groups text: ${hasGroups}`);

    } else {
      console.log('Replay button not found, taking diagnostic screenshot');
      await page.screenshot({
        path: 'tests/e2e/screenshots/debug-error-no-button.png',
        fullPage: true
      });

      // Log page HTML for debugging
      const bodyHTML = await page.locator('body').innerHTML();
      console.log('Page body preview:', bodyHTML.substring(0, 500));
    }

    // Final summary
    console.log('\n=== TEST SUMMARY ===');
    console.log(`Console logs captured: ${consoleLogs.length}`);
    console.log(`Errors captured: ${errors.length}`);

    // Show replay-related logs
    const replayLogs = consoleLogs.filter(log =>
      log.toLowerCase().includes('replay') ||
      log.toLowerCase().includes('timeline') ||
      log.toLowerCase().includes('hybrid')
    );

    if (replayLogs.length > 0) {
      console.log('\n=== REPLAY-RELATED LOGS ===');
      replayLogs.forEach(log => console.log(log));
    }

    if (errors.length > 0) {
      console.log('\n=== ERRORS ===');
      errors.forEach(err => console.log(err));
    }

    // Test passes if we got this far
    expect(true).toBe(true);
  });
});
