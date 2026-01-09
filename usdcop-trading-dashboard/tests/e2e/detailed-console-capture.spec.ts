import { test, expect } from '@playwright/test';

/**
 * Detailed Console Log Capture
 * Captures ALL console messages, network errors, and failed requests
 */
test('Detailed PPO Dashboard Analysis', async ({ page }) => {
  // Storage for all logs
  const consoleLogs: string[] = [];
  const networkErrors: string[] = [];
  const failedRequests: string[] = [];
  const apiCalls: { url: string; status: number; method: string }[] = [];

  // Capture console messages
  page.on('console', (msg) => {
    const type = msg.type().toUpperCase();
    const text = msg.text();
    consoleLogs.push(`[${type}] ${text}`);
  });

  // Capture page errors
  page.on('pageerror', (error) => {
    consoleLogs.push(`[PAGE_ERROR] ${error.message}`);
  });

  // Capture network requests and responses
  page.on('response', async (response) => {
    const url = response.url();
    const status = response.status();
    const method = response.request().method();

    // Log API calls
    if (url.includes('/api/')) {
      apiCalls.push({ url, status, method });

      if (status >= 400) {
        let body = '';
        try {
          body = await response.text();
          body = body.substring(0, 200);
        } catch {}
        failedRequests.push(`[${status}] ${method} ${url} - ${body}`);
      }
    }
  });

  page.on('requestfailed', (request) => {
    networkErrors.push(`[FAILED] ${request.method()} ${request.url()} - ${request.failure()?.errorText}`);
  });

  // Set viewport
  await page.setViewportSize({ width: 1920, height: 1080 });

  // Go directly to dashboard (skip login for faster testing)
  console.log('=== NAVIGATING TO DASHBOARD ===');
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'domcontentloaded' });

  // Wait for initial load
  await page.waitForTimeout(3000);

  // Wait for API calls to complete
  console.log('=== WAITING FOR API CALLS ===');
  await page.waitForTimeout(10000);

  // Take screenshot
  await page.screenshot({
    path: 'tests/e2e/screenshots/detailed-ppo-analysis.png',
    fullPage: true
  });

  // Print results
  console.log('\n' + '='.repeat(80));
  console.log('CONSOLE LOGS (' + consoleLogs.length + ' total)');
  console.log('='.repeat(80));
  consoleLogs.forEach(log => console.log(log));

  console.log('\n' + '='.repeat(80));
  console.log('API CALLS (' + apiCalls.length + ' total)');
  console.log('='.repeat(80));
  apiCalls.forEach(call => {
    const statusIcon = call.status < 400 ? '✓' : '✗';
    console.log(`${statusIcon} [${call.status}] ${call.method} ${call.url}`);
  });

  console.log('\n' + '='.repeat(80));
  console.log('FAILED REQUESTS (' + failedRequests.length + ' total)');
  console.log('='.repeat(80));
  failedRequests.forEach(req => console.log(req));

  console.log('\n' + '='.repeat(80));
  console.log('NETWORK ERRORS (' + networkErrors.length + ' total)');
  console.log('='.repeat(80));
  networkErrors.forEach(err => console.log(err));

  // Summary
  const errors = consoleLogs.filter(l =>
    l.includes('[ERROR]') || l.includes('[PAGE_ERROR]') || l.includes('error')
  );
  const warnings = consoleLogs.filter(l => l.includes('[WARNING]') || l.includes('[WARN]'));

  console.log('\n' + '='.repeat(80));
  console.log('SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total Console Logs: ${consoleLogs.length}`);
  console.log(`Errors: ${errors.length}`);
  console.log(`Warnings: ${warnings.length}`);
  console.log(`Failed API Calls: ${failedRequests.length}`);
  console.log(`Network Errors: ${networkErrors.length}`);

  if (errors.length > 0) {
    console.log('\n--- ERRORS DETAIL ---');
    errors.forEach(e => console.log(e));
  }
});
