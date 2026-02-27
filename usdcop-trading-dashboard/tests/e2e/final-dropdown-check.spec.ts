import { test, expect } from '@playwright/test';

test('Final dropdown verification - both pages', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(`PAGE_ERROR: ${err.message}`));

  // === DASHBOARD ===
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(6000);
  await page.screenshot({ path: 'test-results/FINAL-dashboard.png', fullPage: true });

  const dashBody = await page.textContent('body');
  const dashHasDropdown = dashBody?.includes('ml_forecasting') && dashBody?.includes('Smart Simple');
  console.log('\n=== DASHBOARD HEADER ===');
  console.log('Has strategy dropdown content:', dashHasDropdown);

  // Click it to expand
  const dashDropBtn = await page.$('button:has-text("Smart Simple")');
  if (dashDropBtn) {
    console.log('Dashboard dropdown button: FOUND');
    await dashDropBtn.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'test-results/FINAL-dashboard-open.png' });
    const expanded = await page.textContent('body');
    console.log('Expanded has "Estrategia Activa":', expanded?.includes('Estrategia Activa'));
  } else {
    console.log('Dashboard dropdown button: NOT FOUND');
  }

  // === PRODUCTION ===
  await page.goto('http://localhost:5000/production', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(8000);
  await page.screenshot({ path: 'test-results/FINAL-production.png', fullPage: true });

  const prodBody = await page.textContent('body');
  console.log('\n=== PRODUCTION ===');
  console.log('Body length:', prodBody?.length);
  console.log('Has Smart Simple:', prodBody?.includes('Smart Simple'));
  console.log('Has Retorno:', prodBody?.includes('Retorno'));
  console.log('Has Sharpe:', prodBody?.includes('Sharpe'));

  const prodDropBtn = await page.$('button:has-text("Smart Simple")');
  if (prodDropBtn) {
    console.log('Production dropdown button: FOUND');
    await prodDropBtn.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: 'test-results/FINAL-production-open.png' });
    const expanded = await page.textContent('body');
    console.log('Expanded has Retorno:', expanded?.includes('Retorno'));
    console.log('Expanded has Win Rate:', expanded?.includes('Win Rate'));
    console.log('Expanded has Max DD:', expanded?.includes('Max DD'));
  } else {
    console.log('Production dropdown button: NOT FOUND');
  }

  console.log('\n=== ERRORS ===');
  errors.forEach(e => console.log('  ', e));
  console.log('Total errors:', errors.length);
});
