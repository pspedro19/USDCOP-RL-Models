import { test, expect } from '@playwright/test';

/**
 * Dashboard Production Check
 * Verifies v19 and v20 models are properly configured
 * Captures console logs and API responses
 */
test('Dashboard Production Check - v19 & v20 Models', async ({ page }) => {
  const consoleLogs: string[] = [];
  const apiCalls: { url: string; status: number; method: string; body?: string }[] = [];
  const errors: string[] = [];

  // Capture console
  page.on('console', (msg) => {
    const text = `[${msg.type().toUpperCase()}] ${msg.text()}`;
    consoleLogs.push(text);
    if (msg.type() === 'error') errors.push(text);
  });

  page.on('pageerror', (error) => {
    errors.push(`[PAGE_ERROR] ${error.message}`);
  });

  // Capture API responses
  page.on('response', async (response) => {
    const url = response.url();
    if (url.includes('/api/')) {
      let body = '';
      try {
        const contentType = response.headers()['content-type'] || '';
        if (contentType.includes('json')) {
          body = await response.text();
        }
      } catch (e) {}
      apiCalls.push({
        url,
        status: response.status(),
        method: response.request().method(),
        body: body.substring(0, 500)
      });
    }
  });

  // Login
  await page.goto('http://localhost:5000/login');
  await page.locator('input[autocomplete="username"]').fill('admin');
  await page.locator('input[type="password"]').fill('admin123');
  await page.locator('button[type="submit"]').click();
  await page.waitForURL('**/hub', { timeout: 10000 });

  // Go to Dashboard
  console.log('\n=== DASHBOARD ANALYSIS ===');
  await page.goto('http://localhost:5000/dashboard');
  await page.waitForTimeout(5000);

  // Screenshot initial state
  await page.screenshot({ path: 'tests/e2e/screenshots/prod-dashboard-initial.png', fullPage: true });

  // Check for model selector
  const modelSelector = page.locator('[class*="model"], select, [data-testid*="model"]').first();
  const modelSelectorExists = await modelSelector.count() > 0;
  console.log(`Model selector found: ${modelSelectorExists}`);

  // Find all dropdown/select elements
  const dropdowns = await page.locator('button, select').all();
  console.log(`Found ${dropdowns.length} interactive elements`);

  // Look for model-related text
  const pageContent = await page.textContent('body');
  const hasV19 = pageContent?.includes('v19') || pageContent?.includes('V19');
  const hasV20 = pageContent?.includes('v20') || pageContent?.includes('V20');
  const hasPPO = pageContent?.includes('PPO') || pageContent?.includes('ppo');
  console.log(`Page mentions v19: ${hasV19}`);
  console.log(`Page mentions v20: ${hasV20}`);
  console.log(`Page mentions PPO: ${hasPPO}`);

  // Wait for more data to load
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'tests/e2e/screenshots/prod-dashboard-loaded.png', fullPage: true });

  // Print API calls
  console.log('\n=== API CALLS ===');
  apiCalls.forEach(call => {
    console.log(`${call.method} ${call.url} -> ${call.status}`);
    if (call.body && call.status !== 200) {
      console.log(`  Response: ${call.body.substring(0, 200)}`);
    }
  });

  // Print errors
  console.log('\n=== ERRORS ===');
  errors.forEach(e => console.log(e));

  // Print relevant console logs
  console.log('\n=== CONSOLE LOGS (last 30) ===');
  consoleLogs.slice(-30).forEach(log => console.log(log));

  // Check specific endpoints
  console.log('\n=== ENDPOINT ANALYSIS ===');
  const modelEndpoints = apiCalls.filter(c => c.url.includes('/models/'));
  const tradingEndpoints = apiCalls.filter(c => c.url.includes('/trading/'));
  const marketEndpoints = apiCalls.filter(c => c.url.includes('/market/'));

  console.log(`Model endpoints: ${modelEndpoints.length}`);
  modelEndpoints.forEach(e => console.log(`  ${e.method} ${e.url} -> ${e.status}`));

  console.log(`Trading endpoints: ${tradingEndpoints.length}`);
  tradingEndpoints.forEach(e => console.log(`  ${e.method} ${e.url} -> ${e.status}`));

  console.log(`Market endpoints: ${marketEndpoints.length}`);
  marketEndpoints.forEach(e => console.log(`  ${e.method} ${e.url} -> ${e.status}`));
});
