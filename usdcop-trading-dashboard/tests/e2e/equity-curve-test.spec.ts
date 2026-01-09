import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5000';

test('Capture redesigned equity curve', async ({ page }) => {
  // Login first
  await page.goto(`${BASE_URL}/login`);
  await page.waitForLoadState('networkidle');

  // Fill login credentials
  await page.fill('input[type="text"], input[placeholder*="user" i], input[name="username"]', 'admin');
  await page.fill('input[type="password"]', 'admin123');
  await page.click('button[type="submit"]');

  // Wait for redirect to dashboard
  await page.waitForURL('**/dashboard**', { timeout: 15000 });
  await page.waitForLoadState('networkidle');

  // Wait for the page to fully load
  await page.waitForTimeout(5000);

  // Scroll to the Performance section
  const performanceSection = page.locator('text=Performance').first();
  await performanceSection.scrollIntoViewIfNeeded();

  // Wait a bit for the chart to render
  await page.waitForTimeout(2000);

  // Take a full page screenshot
  await page.screenshot({
    path: 'tests/e2e/screenshots/equity-curve-full.png',
    fullPage: true
  });

  // Find the Performance section and take a focused screenshot
  const equityCurveCard = page.locator('.bg-slate-900\\/40').filter({ hasText: 'Equity Curve' }).first();

  if (await equityCurveCard.isVisible()) {
    await equityCurveCard.screenshot({
      path: 'tests/e2e/screenshots/equity-curve-section.png'
    });
    console.log('✅ Equity curve section captured');
  } else {
    console.log('⚠️ Equity curve card not found');
  }

  // Also scroll to see all metrics
  await page.evaluate(() => {
    const element = document.querySelector('[class*="grid grid-cols-4"]');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });

  await page.waitForTimeout(1000);

  // Take screenshot of metrics row area
  await page.screenshot({
    path: 'tests/e2e/screenshots/equity-curve-metrics.png'
  });

  console.log('✅ Screenshots captured successfully');
});
