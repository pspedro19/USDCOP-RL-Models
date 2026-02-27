import { test, expect } from '@playwright/test';

test('Production trade table shows all trades including open', async ({ page }) => {
  await page.goto('http://localhost:5000/production', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(10000);

  // Scroll to trade table
  const tradeTable = await page.$('text=Historial de Trades');
  if (tradeTable) {
    await tradeTable.scrollIntoViewIfNeeded();
    await page.waitForTimeout(500);
  }

  await page.screenshot({ path: 'test-results/trade-table.png', fullPage: true });

  const body = await page.textContent('body');
  console.log('\n=== TRADE TABLE CHECK ===');
  console.log('Has "Historial de Trades":', body?.includes('Historial de Trades'));
  console.log('Has "ABIERTO":', body?.includes('ABIERTO'));
  console.log('Has "take_profit":', body?.includes('take_profit'));
  console.log('Has "SHORT":', body?.includes('SHORT'));
  console.log('Has "3,686" (trade 1 entry):', body?.includes('3,686'));
  console.log('Has "3,663" or "3,662" (trade 2 entry):', body?.includes('3,663') || body?.includes('3,662'));
  console.log('Has "2 operaciones":', body?.includes('2 operaciones'));

  // Count table rows
  const rows = await page.$$('table tbody tr');
  console.log('Trade table rows:', rows.length);

  expect(body?.includes('ABIERTO')).toBe(true);
});
