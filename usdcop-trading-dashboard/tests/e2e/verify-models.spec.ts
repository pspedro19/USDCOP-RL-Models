import { test, expect } from '@playwright/test';

test('Verify v19 and v20 models appear', async ({ page }) => {
  // Login
  await page.goto('http://localhost:5000/login');
  await page.locator('input[autocomplete="username"]').fill('admin');
  await page.locator('input[type="password"]').fill('admin123');
  await page.locator('button[type="submit"]').click();
  await page.waitForURL('**/hub', { timeout: 10000 });

  // Go to Dashboard
  await page.goto('http://localhost:5000/dashboard');
  await page.waitForTimeout(4000);

  // Screenshot initial
  await page.screenshot({ path: 'tests/e2e/screenshots/models-initial.png', fullPage: true });

  // Find model selector dropdown
  const pageContent = await page.textContent('body');
  console.log('Page contains V19:', pageContent?.includes('V19'));
  console.log('Page contains V20:', pageContent?.includes('V20'));
  console.log('Page contains PPO:', pageContent?.includes('PPO'));

  // Click on model dropdown to open it
  const dropdownButton = page.locator('button:has-text("PPO"), button:has-text("Model"), [class*="dropdown"]').first();
  if (await dropdownButton.count() > 0) {
    await dropdownButton.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: 'tests/e2e/screenshots/models-dropdown-open.png', fullPage: true });

    // Check for v19 and v20 in dropdown
    const dropdownContent = await page.textContent('body');
    console.log('Dropdown contains V19:', dropdownContent?.includes('V19'));
    console.log('Dropdown contains V20:', dropdownContent?.includes('V20'));
  }

  // Call API directly
  const response = await page.request.get('http://localhost:5000/api/models');
  const data = await response.json();
  console.log('API Models:', JSON.stringify(data.models?.map((m: any) => ({ id: m.id, name: m.name })), null, 2));
});
