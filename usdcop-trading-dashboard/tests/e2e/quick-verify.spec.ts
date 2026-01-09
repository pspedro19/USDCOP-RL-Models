import { test, expect } from '@playwright/test';

test('Quick verify Dashboard and Forecasting', async ({ page }) => {
  // Login first
  await page.goto('http://localhost:5000/login');
  await page.locator('input[autocomplete="username"]').fill('admin');
  await page.locator('input[type="password"]').fill('admin123');
  await page.locator('button[type="submit"]').click();
  await page.waitForURL('**/hub', { timeout: 10000 });

  // Take Hub screenshot
  await page.screenshot({ path: 'tests/e2e/screenshots/verify-hub.png', fullPage: true });

  // Go to Dashboard
  await page.goto('http://localhost:5000/dashboard');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'tests/e2e/screenshots/verify-dashboard.png', fullPage: true });

  // Go to Forecasting
  await page.goto('http://localhost:5000/forecasting');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'tests/e2e/screenshots/verify-forecasting.png', fullPage: true });

  // Mobile view - Dashboard
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('http://localhost:5000/dashboard');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'tests/e2e/screenshots/verify-dashboard-mobile.png', fullPage: true });

  // Mobile view - Forecasting
  await page.goto('http://localhost:5000/forecasting');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'tests/e2e/screenshots/verify-forecasting-mobile.png', fullPage: true });

  console.log('Screenshots captured successfully');
});
