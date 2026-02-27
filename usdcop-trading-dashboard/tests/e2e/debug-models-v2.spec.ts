/**
 * Debug Models Dropdown V2
 */

import { test, expect } from '@playwright/test';

test.use({ browserName: 'chromium' });

test('debug models v2', async ({ page }) => {
  const logs: string[] = [];

  page.on('console', msg => {
    const text = `[${msg.type()}] ${msg.text()}`;
    logs.push(text);
    if (msg.type() === 'error' || msg.text().includes('model') || msg.text().includes('Model')) {
      console.log(text);
    }
  });

  page.on('pageerror', err => console.log('[PAGE ERROR]', err.message));

  // Interceptar la llamada a /api/models
  await page.route('**/api/models', async route => {
    console.log('[INTERCEPT] Request to /api/models');
    const response = await route.fetch();
    const body = await response.text();
    console.log('[INTERCEPT] Response:', body.substring(0, 300));
    await route.fulfill({ response });
  });

  console.log('Navegando a dashboard...');
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(5000);

  // Buscar header
  console.log('\n=== HEADER ===');
  const header = page.locator('header').first();
  if (await header.count() > 0) {
    const headerHtml = await header.innerHTML();
    console.log('Header HTML (primeros 500 chars):');
    console.log(headerHtml.substring(0, 500));
  }

  // Buscar todos los botones
  console.log('\n=== BOTONES ===');
  const buttons = await page.locator('button').all();
  console.log(`Total botones: ${buttons.length}`);
  for (let i = 0; i < Math.min(buttons.length, 15); i++) {
    const text = await buttons[i].textContent();
    const classes = await buttons[i].getAttribute('class');
    console.log(`${i}: "${text?.substring(0, 40)}" - classes: ${classes?.substring(0, 60)}`);
  }

  // Buscar específicamente el dropdown de modelo
  console.log('\n=== BUSCAR DROPDOWN MODELO ===');
  const modelDropdown = page.locator('button:has-text("Select Model"), button:has-text("PPO"), button:has-text("Model")');
  const count = await modelDropdown.count();
  console.log(`Encontrados ${count} elementos con texto de modelo`);

  // Buscar por estructura - div con w-3 h-3 rounded-full (el indicador de color)
  const colorIndicators = await page.locator('.rounded-full').all();
  console.log(`Indicadores de color (rounded-full): ${colorIndicators.length}`);

  // Buscar texto "No models available"
  const noModelsText = page.locator('text=No models available');
  if (await noModelsText.count() > 0) {
    console.log('ENCONTRADO: "No models available" - Los modelos no se cargaron!');
  }

  // Screenshot del header
  await page.screenshot({ path: 'tests/e2e/screenshots/debug-v2-full.png', fullPage: true });

  // Verificar si el ModelContext tiene modelos
  console.log('\n=== VERIFICAR ESTADO DE REACT ===');
  const reactState = await page.evaluate(() => {
    // Intentar encontrar el estado de React
    const root = document.getElementById('__next');
    return {
      hasRoot: !!root,
      bodyClasses: document.body.className,
      pageContent: document.body.innerText.substring(0, 500)
    };
  });
  console.log('React root:', reactState.hasRoot);
  console.log('Body classes:', reactState.bodyClasses);
  console.log('Page content:', reactState.pageContent);

  // Logs de error
  console.log('\n=== ERRORES EN CONSOLA ===');
  const errorLogs = logs.filter(l => l.includes('error') || l.includes('Error'));
  errorLogs.forEach(l => console.log(l));

  expect(true).toBe(true);
});
