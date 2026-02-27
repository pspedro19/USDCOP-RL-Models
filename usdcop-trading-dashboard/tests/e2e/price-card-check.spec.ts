import { test, expect } from '@playwright/test';

test('Production page shows real-time USD/COP price card', async ({ page }) => {
  await page.goto('http://localhost:5000/production', { waitUntil: 'networkidle', timeout: 60000 });

  // Wait for the price card to appear (loads ~1-2s from investing.com)
  await page.waitForSelector('text=Rango Dia', { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(2000);

  await page.screenshot({ path: 'test-results/price-card.png', fullPage: true });

  const body = await page.textContent('body');
  console.log('\n=== PRICE CARD CHECK ===');
  console.log('Has "USD/COP":', body?.includes('USD/COP'));
  console.log('Has "Spot":', body?.includes('Spot'));
  console.log('Has "$3,6" (price):', body?.includes('$3,6'));
  console.log('Has "Rango Dia":', body?.includes('Rango Dia'));
  console.log('Has "Rango 52 Sem":', body?.includes('Rango 52 Sem'));
  console.log('Has "Fuente":', body?.includes('Fuente'));
  console.log('Has "Investing.com":', body?.includes('Investing.com'));
  console.log('Has "Mercado":', body?.includes('Mercado'));

  // Core assertions
  expect(body?.includes('USD/COP')).toBe(true);
  expect(body?.includes('Fuente')).toBe(true);
});
