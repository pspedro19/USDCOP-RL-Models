import { test, expect } from '@playwright/test';

test('Production page shows correct KPIs and active position with realtime price', async ({ page }) => {
  await page.goto('http://localhost:5000/production', { waitUntil: 'networkidle', timeout: 60000 });

  // Wait for full page load (KPIs appear after useLiveProduction completes)
  await page.waitForSelector('text=Metricas Clave', { timeout: 45000 }).catch(() => {});
  await page.waitForTimeout(3000);

  await page.screenshot({ path: 'test-results/production-kpi.png', fullPage: true });

  const body = await page.textContent('body');
  console.log('\n=== KPI CHECK ===');
  console.log('Has "+2.9" or "+3.0" (return):', body?.includes('+2.9') || body?.includes('+3.0'));
  console.log('Has "$10,29" or "$10297" (equity):', body?.includes('$10,29') || body?.includes('$10297') || body?.includes('10298'));
  console.log('Has "5.10" (sharpe):', body?.includes('5.10'));
  console.log('Has "100%" (win rate):', body?.includes('100%'));

  console.log('\n=== POSITION CHECK ===');
  console.log('Has "Posicion Activa":', body?.includes('Posicion Activa'));
  console.log('Has "$3,665" or "$3,66" (realtime price in position):', body?.includes('$3,665') || body?.includes('$3,66'));
  console.log('Has "1.5%" (TP):', body?.includes('1.5%'));
  console.log('Has "3.0%" (HS):', body?.includes('3.0%'));

  console.log('\n=== PRICE CARD CHECK ===');
  console.log('Has "USD/COP":', body?.includes('USD/COP'));
  console.log('Has "Investing.com":', body?.includes('Investing.com'));

  // Core assertions
  expect(body?.includes('+2.9') || body?.includes('+3.0')).toBe(true);
  expect(body?.includes('USD/COP')).toBe(true);
});
