import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5000';

test.describe('Dropdown menus and trade markers', () => {
  test('Dashboard has strategy selector dropdown', async ({ page }) => {
    await page.goto(`${BASE_URL}/dashboard`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(5000);

    await page.screenshot({ path: 'test-results/dashboard-dropdown.png', fullPage: true });

    const body = await page.textContent('body');

    // Check for strategy selector content
    const hasStrategyName = body?.includes('Smart Simple v1.1.0') || body?.includes('Smart Simple');
    const hasPipeline = body?.includes('ml_forecasting');
    const hasReturnPct = body?.includes('23.') || body?.includes('+23');
    const hasSharpe = body?.includes('3.82') || body?.includes('3.822');
    const hasApproved = body?.includes('APPROVED') || body?.includes('Aprobado');

    console.log('\n=== DASHBOARD DROPDOWN CHECK ===');
    console.log('Has strategy name:', hasStrategyName);
    console.log('Has pipeline:', hasPipeline);
    console.log('Has return %:', hasReturnPct);
    console.log('Has Sharpe:', hasSharpe);
    console.log('Has APPROVED:', hasApproved);

    // Look for the dropdown button
    const dropdownBtns = await page.$$('button:has-text("Smart Simple")');
    console.log('Strategy dropdown buttons found:', dropdownBtns.length);

    // Check for the green pulse dot indicator
    const pulseDots = await page.$$('.animate-pulse');
    console.log('Pulse indicators:', pulseDots.length);

    expect(hasStrategyName).toBe(true);
  });

  test('Production has strategy dropdown with metrics', async ({ page }) => {
    await page.goto(`${BASE_URL}/production`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(5000);

    await page.screenshot({ path: 'test-results/production-dropdown.png', fullPage: true });

    const body = await page.textContent('body');

    console.log('\n=== PRODUCTION DROPDOWN CHECK ===');
    const hasStrategyName = body?.includes('Smart Simple');
    const hasApproved = body?.includes('APPROVED') || body?.includes('Aprobado');
    console.log('Has strategy name:', hasStrategyName);
    console.log('Has APPROVED:', hasApproved);

    // Click the strategy dropdown to open it
    const stratDropdown = await page.$('button:has-text("Smart Simple")');
    if (stratDropdown) {
      await stratDropdown.click();
      await page.waitForTimeout(500);
      await page.screenshot({ path: 'test-results/production-dropdown-open.png', fullPage: true });

      const expandedBody = await page.textContent('body');
      const hasRetorno = expandedBody?.includes('Retorno');
      const hasSharpeLabel = expandedBody?.includes('Sharpe');
      const hasWinRate = expandedBody?.includes('Win Rate');
      const hasMaxDD = expandedBody?.includes('Max DD');
      console.log('Dropdown expanded:');
      console.log('  Has Retorno:', hasRetorno);
      console.log('  Has Sharpe:', hasSharpeLabel);
      console.log('  Has Win Rate:', hasWinRate);
      console.log('  Has Max DD:', hasMaxDD);
    } else {
      console.log('Strategy dropdown button NOT found');
    }

    expect(hasStrategyName).toBe(true);
  });

  test('Production chart has trade markers', async ({ page }) => {
    await page.goto(`${BASE_URL}/production`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(8000);

    await page.screenshot({ path: 'test-results/production-chart-markers.png', fullPage: true });

    const body = await page.textContent('body');

    // Check for trade-related content
    const hasTrades = body?.includes('trade') || body?.includes('Trade') || body?.includes('SHORT') || body?.includes('LONG');
    const has2026 = body?.includes('2026');
    console.log('\n=== PRODUCTION TRADE MARKERS ===');
    console.log('Has trades content:', hasTrades);
    console.log('Has 2026:', has2026);

    // Check for chart canvas elements (lightweight-charts renders on canvas)
    const canvases = await page.$$('canvas');
    console.log('Canvas elements (charts):', canvases.length);

    // Check for marker overlay elements
    const markerOverlays = await page.$$('[class*="marker"], [class*="signal"], [data-signal]');
    console.log('Marker overlay elements:', markerOverlays.length);

    expect(hasTrades).toBe(true);
  });
});
