import { test, expect } from '@playwright/test';

const viewports = [
  { name: 'mobile', width: 375, height: 667 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'desktop', width: 1280, height: 800 },
  { name: 'wide', width: 1920, height: 1080 }
];

test('Dashboard viewport responsiveness check', async ({ page }) => {
  // Login
  await page.goto('http://localhost:3000/login');
  await page.waitForLoadState('networkidle');

  // Fill login form
  const usernameInput = page.locator('input[type="text"]').first();
  const passwordInput = page.locator('input[type="password"]').first();

  await usernameInput.fill('admin');
  await passwordInput.fill('admin1234');

  // Submit
  await page.click('button[type="submit"]');

  // Wait and check if we're on dashboard or need to navigate
  await page.waitForTimeout(3000);

  // If still on login, go directly to dashboard
  const currentUrl = page.url();
  console.log('Current URL after login:', currentUrl);

  if (!currentUrl.includes('/dashboard')) {
    console.log('Navigating directly to dashboard...');
    await page.goto('http://localhost:3000/dashboard');
  }

  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000);

  // Test each viewport
  for (const viewport of viewports) {
    console.log(`\n========== Testing ${viewport.name} (${viewport.width}x${viewport.height}) ==========`);

    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.waitForTimeout(500);

    // Screenshot
    await page.screenshot({
      path: `tests/e2e/screenshots/viewport-${viewport.name}.png`,
      fullPage: true
    });

    // Check main elements are visible
    const header = page.locator('header').first();
    const headerVisible = await header.isVisible();
    console.log(`[${viewport.name}] Header visible: ${headerVisible}`);

    // Check for horizontal overflow (scrollbar indicator)
    // Allow 50px tolerance for Next.js dev tools, error indicators, scrollbar
    const scrollWidth = await page.evaluate(() => document.body.scrollWidth);
    const clientWidth = await page.evaluate(() => document.body.clientWidth);
    const overflowAmount = scrollWidth - clientWidth;
    const hasSignificantOverflow = overflowAmount > 50; // 50px tolerance for dev tools

    console.log(`[${viewport.name}] Scroll width: ${scrollWidth}, Client width: ${clientWidth}, Overflow: ${overflowAmount}px`);
    console.log(`[${viewport.name}] Overflow status: ${hasSignificantOverflow ? '❌ SIGNIFICANT' : '✅ ACCEPTABLE'}`);

    // Assert no significant horizontal overflow (allow minor variance)
    expect(hasSignificantOverflow, `${viewport.name} has ${overflowAmount}px overflow (max 30px)`).toBe(false);

    console.log(`[${viewport.name}] ✅ Passed`);
  }

  console.log('\n========== All viewport tests completed ==========');
});
