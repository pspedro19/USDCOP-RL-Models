import { test, expect } from '@playwright/test';

test('Screenshot both pages for dropdown verification', async ({ page }) => {
  // Dashboard
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(6000);
  await page.screenshot({ path: 'test-results/NOW-dashboard.png', fullPage: true });

  const dashBody = await page.textContent('body');
  console.log('\n=== DASHBOARD ===');
  console.log('Has ml_forecasting:', dashBody?.includes('ml_forecasting'));
  console.log('Has Smart Simple:', dashBody?.includes('Smart Simple'));
  console.log('Has +23:', dashBody?.includes('+23'));
  console.log('Has APPROVED:', dashBody?.includes('APPROVED'));

  // Find all buttons with strategy text
  const stratBtns = await page.$$('button');
  for (const btn of stratBtns) {
    const text = await btn.textContent();
    if (text && (text.includes('Smart') || text.includes('strategy') || text.includes('ml_'))) {
      console.log('Found strategy button:', text.trim().substring(0, 80));
    }
  }

  // Production
  await page.goto('http://localhost:5000/production', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(6000);
  await page.screenshot({ path: 'test-results/NOW-production.png', fullPage: true });

  const prodBody = await page.textContent('body');
  console.log('\n=== PRODUCTION ===');
  console.log('Has Smart Simple:', prodBody?.includes('Smart Simple'));
  console.log('Has APPROVED:', prodBody?.includes('APPROVED'));
  console.log('Body length:', prodBody?.length);
  console.log('First 400 chars:', prodBody?.substring(0, 400));

  const prodBtns = await page.$$('button');
  for (const btn of prodBtns) {
    const text = await btn.textContent();
    if (text && (text.includes('Smart') || text.includes('strategy') || text.includes('APPROVED'))) {
      console.log('Found strategy button:', text.trim().substring(0, 80));
    }
  }
});
