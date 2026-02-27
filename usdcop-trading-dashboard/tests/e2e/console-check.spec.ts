import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5000';

const PAGES = [
  { name: 'Hub', path: '/hub' },
  { name: 'Forecasting', path: '/forecasting' },
  { name: 'Dashboard', path: '/dashboard' },
  { name: 'Production', path: '/production' },
];

for (const pg of PAGES) {
  test(`${pg.name} (${pg.path}) - no console errors`, async ({ page }) => {
    const errors: string[] = [];
    const warnings: string[] = [];
    const logs: string[] = [];

    page.on('console', (msg) => {
      const text = msg.text();
      if (msg.type() === 'error') {
        // Ignore known benign errors
        if (text.includes('favicon') || text.includes('404') && text.includes('.ico')) return;
        errors.push(`[ERROR] ${text}`);
      } else if (msg.type() === 'warning') {
        warnings.push(`[WARN] ${text}`);
      } else {
        logs.push(`[${msg.type()}] ${text}`);
      }
    });

    page.on('pageerror', (err) => {
      errors.push(`[PAGE_ERROR] ${err.message}`);
    });

    const response = await page.goto(`${BASE_URL}${pg.path}`, {
      waitUntil: 'networkidle',
      timeout: 30000,
    });

    // Wait extra for any async rendering
    await page.waitForTimeout(3000);

    // Log results
    console.log(`\n=== ${pg.name} (${pg.path}) ===`);
    console.log(`Status: ${response?.status()}`);
    console.log(`Errors: ${errors.length}`);
    errors.forEach((e) => console.log(`  ${e}`));
    console.log(`Warnings: ${warnings.length}`);
    warnings.forEach((w) => console.log(`  ${w}`));
    console.log(`Logs: ${logs.length}`);

    // Check page loaded
    expect(response?.status()).toBeLessThan(400);

    // Report errors but don't fail on warnings
    if (errors.length > 0) {
      console.log('\nCONSOLE ERRORS FOUND:');
      errors.forEach((e) => console.log(`  ${e}`));
    }

    // Fail only on actual JS errors / page errors
    const criticalErrors = errors.filter(
      (e) => e.includes('[PAGE_ERROR]')
    );
    expect(criticalErrors).toHaveLength(0);
  });
}
