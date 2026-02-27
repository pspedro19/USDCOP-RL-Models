/**
 * Debug Models Dropdown V3 - Específico para ModelDropdown
 */

import { test, expect } from '@playwright/test';

test.use({ browserName: 'chromium' });

test('debug ModelDropdown', async ({ page }) => {
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.log(`[ERROR] ${msg.text()}`);
    }
  });

  console.log('Navegando a dashboard...');
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);

  // Capturar el header completo
  console.log('\n=== BUSCANDO EN HEADER ===');
  const headerDiv = page.locator('header > div > div');
  const headerContent = await headerDiv.innerHTML();
  console.log('Header div innerHTML:');
  console.log(headerContent);

  // Buscar el contenedor del ModelDropdown (debería ser el tercer div hijo)
  console.log('\n=== TERCER DIV DEL HEADER (donde debería estar ModelDropdown) ===');
  const thirdDiv = page.locator('header > div > div > div').nth(2);
  if (await thirdDiv.count() > 0) {
    const thirdDivHtml = await thirdDiv.innerHTML();
    console.log('Tercer div:');
    console.log(thirdDivHtml);
  } else {
    console.log('NO HAY TERCER DIV');
  }

  // Buscar todos los divs en el header
  console.log('\n=== TODOS LOS DIVS HIJOS EN EL HEADER ===');
  const headerDivs = await page.locator('header > div > div > div').all();
  console.log(`Total divs hijos: ${headerDivs.length}`);
  for (let i = 0; i < headerDivs.length; i++) {
    const html = await headerDivs[i].innerHTML();
    console.log(`\nDiv ${i}:`);
    console.log(html.substring(0, 300));
  }

  // Buscar específicamente elementos que podrían ser el dropdown
  console.log('\n=== BUSCAR POR CLASES DEL MODELDROPDOWN ===');
  // El ModelDropdown tiene: "relative w-full sm:w-auto sm:min-w-[280px]"
  const modelDropdownContainer = page.locator('.sm\\:min-w-\\[280px\\]');
  const count = await modelDropdownContainer.count();
  console.log(`Elementos con clase min-w-[280px]: ${count}`);

  // Buscar por texto "Select Model" o "PPO Primary"
  const selectModelText = page.locator('text=Select Model');
  console.log(`Elementos con "Select Model": ${await selectModelText.count()}`);

  const ppoPrimaryText = page.locator('text=PPO Primary');
  console.log(`Elementos con "PPO Primary": ${await ppoPrimaryText.count()}`);

  // Screenshot
  await page.screenshot({ path: 'tests/e2e/screenshots/debug-v3-header.png', fullPage: false, clip: { x: 0, y: 0, width: 1280, height: 200 } });
  console.log('\nScreenshot guardado');

  expect(true).toBe(true);
});
