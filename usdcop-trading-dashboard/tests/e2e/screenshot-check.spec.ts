import { test } from '@playwright/test';

test('Screenshot /production page', async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto('http://localhost:5000/production', { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForTimeout(5000);
  await page.screenshot({ path: 'tests/e2e/screenshots/production-check.png', fullPage: true });
});

test('Screenshot /dashboard page', async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForTimeout(5000);
  await page.screenshot({ path: 'tests/e2e/screenshots/dashboard-check.png', fullPage: true });
});
