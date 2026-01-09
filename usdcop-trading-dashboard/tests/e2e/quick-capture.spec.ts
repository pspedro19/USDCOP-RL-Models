import { test, expect } from '@playwright/test';

test('Quick dashboard capture with logs', async ({ page }) => {
  // Collect console messages
  const logs: string[] = [];
  page.on('console', (msg) => {
    logs.push(`[${msg.type()}] ${msg.text()}`);
  });
  page.on('pageerror', (error) => {
    logs.push(`[ERROR] ${error.message}`);
  });

  // Set viewport
  await page.setViewportSize({ width: 1920, height: 1080 });

  // Login
  console.log('=== STEP 1: LOGIN ===');
  await page.goto('http://localhost:5000/login');
  await page.waitForLoadState('networkidle');

  await page.fill('input[type="text"]', 'admin');
  await page.fill('input[type="password"]', 'admin1234');
  await page.click('button[type="submit"]');

  // Wait for navigation
  await page.waitForTimeout(3000);

  // Go to dashboard
  console.log('=== STEP 2: NAVIGATE TO DASHBOARD ===');
  await page.goto('http://localhost:5000/dashboard');
  await page.waitForLoadState('networkidle');
  console.log('Network idle, waiting for content...');
  await page.waitForTimeout(8000);

  // Check what's visible
  console.log('=== STEP 3: CHECK CONTENT ===');
  const body = await page.locator('body').innerHTML();
  console.log('Body length:', body.length);

  // Check for specific elements
  const hasHeader = await page.locator('header').count();
  const hasMain = await page.locator('main').count();
  const hasModel = await page.locator('text=USD/COP').count();
  console.log(`Elements: header=${hasHeader}, main=${hasMain}, USDCOP=${hasModel}`);

  // Screenshot
  await page.screenshot({
    path: 'tests/e2e/screenshots/quick-dashboard.png',
    fullPage: true
  });

  // Print console logs
  console.log('=== CONSOLE LOGS ===');
  logs.forEach(log => console.log(log));

  // Print errors only
  const errors = logs.filter(l => l.includes('[error]') || l.includes('[ERROR]'));
  if (errors.length > 0) {
    console.log('=== ERRORS FOUND ===');
    errors.forEach(e => console.log(e));
  }

  console.log('=== SCREENSHOT SAVED ===');
});
