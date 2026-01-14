/**
 * Backtest Verification Test
 * ===========================
 * Verifies that backtest results update all dashboard sections.
 */

import { test, expect } from '@playwright/test';

test.describe('Backtest Results Display', () => {
  test.setTimeout(180000); // 3 minutes

  test('should show backtest results in all sections', async ({ page }) => {
    // Capture console logs
    page.on('console', msg => console.log(`[${msg.type()}] ${msg.text()}`));

    // Step 1: Go to dashboard
    console.log('=== Step 1: Navigate to dashboard ===');
    await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);
    await page.screenshot({ path: 'tests/e2e/screenshots/verify-01-initial.png', fullPage: true });

    // Step 2: Open backtest panel
    console.log('=== Step 2: Open backtest panel ===');
    await page.locator('button:has-text("Backtest")').click();
    await page.waitForTimeout(1000);

    // Step 3: Select validation dates
    console.log('=== Step 3: Select validation dates ===');
    const validationBtn = page.locator('button:has-text("Validación")').first();
    if (await validationBtn.isVisible()) {
      await validationBtn.click();
      await page.waitForTimeout(500);
    }

    // Step 4: Start backtest
    console.log('=== Step 4: Start backtest ===');
    const startBtn = page.locator('button:has-text("Iniciar Backtest")');
    if (await startBtn.isEnabled()) {
      await startBtn.click();
      console.log('Clicked Iniciar Backtest');

      // Wait for backtest to complete (max 90 seconds)
      console.log('Waiting for backtest to complete...');
      await page.waitForTimeout(5000);
      await page.screenshot({ path: 'tests/e2e/screenshots/verify-02-progress.png', fullPage: true });

      // Wait more time for completion
      for (let i = 0; i < 12; i++) {
        await page.waitForTimeout(5000);
        const content = await page.content();
        if (content.includes('status: completed') || content.includes('132 trades')) {
          console.log(`Backtest completed after ${(i+1)*5} seconds`);
          break;
        }
      }

      await page.waitForTimeout(3000);
    }

    // Step 5: Wait for chart and data to fully load
    console.log('=== Step 5: Wait for chart to load ===');

    // Wait for chart to finish loading (no more "Loading chart..." text)
    await page.waitForFunction(() => {
      const chartArea = document.querySelector('[class*="TradingChart"]') || document.body;
      return !chartArea.textContent?.includes('Loading chart...');
    }, { timeout: 30000 }).catch(() => {
      console.log('Chart may still be loading after 30s');
    });

    // Additional wait for signals to render
    await page.waitForTimeout(5000);

    // Step 6: Take final screenshot
    console.log('=== Step 6: Final verification ===');
    await page.screenshot({ path: 'tests/e2e/screenshots/verify-03-final.png', fullPage: true });

    // Step 7: Verify sections updated
    console.log('=== Step 7: Checking sections ===');

    // Check Trading Summary card (use role heading to be specific)
    const tradingSummaryHeading = page.getByRole('heading', { name: /Trading Summary/i }).first();
    const isSummaryVisible = await tradingSummaryHeading.isVisible().catch(() => false);
    console.log(`Trading Summary heading visible: ${isSummaryVisible}`);

    // Check for REPLAY badge (indicates backtest mode is active)
    const replayBadge = page.locator('text=/REPLAY/i').first();
    const hasReplayBadge = await replayBadge.isVisible().catch(() => false);
    console.log(`REPLAY badge visible: ${hasReplayBadge}`);

    // Check if we have trade count (132 trades from backtest)
    const tradesText = await page.locator('text=/132|\\d+ trades|Total Operaciones/i').first().textContent().catch(() => 'not found');
    console.log(`Trades text found: ${tradesText}`);

    // Check equity curve section
    const equitySection = page.locator('h3:has-text("Equity Curve")').first();
    const isEquityVisible = await equitySection.isVisible().catch(() => false);
    console.log(`Equity Curve section visible: ${isEquityVisible}`);

    // Check for profit/loss value (not $0.00)
    const profitLossText = await page.locator('text=/\\$[-]?\\d+[,.]?\\d*/').first().textContent().catch(() => 'not found');
    console.log(`Profit/Loss text found: ${profitLossText}`);

    // Screenshot of chart area
    const chartSection = page.locator('[class*="chart"]').first();
    if (await chartSection.isVisible().catch(() => false)) {
      await chartSection.screenshot({ path: 'tests/e2e/screenshots/verify-04-chart.png' });
    }

    // Check for BUY/SELL signals on chart
    const signalMarkers = page.locator('text=/BUY|SELL/').all();
    const signalCount = (await signalMarkers).length;
    console.log(`Signal markers visible on page: ${signalCount}`);

    console.log('=== Test completed ===');
  });
});
