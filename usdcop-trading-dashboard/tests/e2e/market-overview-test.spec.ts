import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5000';

test('Capture Market Overview section', async ({ page }) => {
  // Login first
  await page.goto(`${BASE_URL}/login`);
  await page.waitForLoadState('networkidle');

  await page.fill('input[type="text"], input[placeholder*="user" i], input[name="username"]', 'admin');
  await page.fill('input[type="password"]', 'admin123');
  await page.click('button[type="submit"]');

  await page.waitForURL('**/dashboard**', { timeout: 15000 });
  await page.waitForLoadState('networkidle');

  // Wait for price to load
  await page.waitForTimeout(5000);

  // Scroll to Market Overview section
  const marketSection = page.locator('text=Market Overview').first();
  await marketSection.scrollIntoViewIfNeeded();

  await page.waitForTimeout(2000);

  // Take a screenshot of the viewport at Market Overview
  await page.screenshot({
    path: 'tests/e2e/screenshots/market-overview-detail.png'
  });

  console.log('✅ Market Overview screenshot captured');
});
