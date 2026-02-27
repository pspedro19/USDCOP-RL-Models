import { test, expect } from '@playwright/test';

test('Production page console errors', async ({ page }) => {
  const errors: string[] = [];
  const logs: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
    else logs.push(`[${msg.type()}] ${msg.text()}`);
  });
  page.on('pageerror', (err) => errors.push(`PAGE_ERROR: ${err.message}`));

  // Intercept network to see what fails
  const failedRequests: string[] = [];
  page.on('response', (res) => {
    if (res.status() >= 400) {
      failedRequests.push(`${res.status()} ${res.url()}`);
    }
  });

  await page.goto('http://localhost:5000/production', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(8000);

  console.log('\n=== CONSOLE ERRORS ===');
  errors.forEach(e => console.log('  ERR:', e));
  console.log('\n=== FAILED REQUESTS ===');
  failedRequests.forEach(r => console.log('  FAIL:', r));

  const body = await page.textContent('body');
  console.log('\nBody length:', body?.length);
  console.log('Body snippet:', body?.substring(0, 300));

  await page.screenshot({ path: 'test-results/production-debug.png', fullPage: true });
});
