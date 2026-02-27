/**
 * Debug Models Dropdown
 * Captura logs de consola para debuggear el dropdown de modelos
 */

import { test, expect } from '@playwright/test';

test.use({
  browserName: 'chromium',
});

test('debug models dropdown', async ({ page }) => {
  const consoleLogs: string[] = [];
  const networkRequests: { url: string; status: number | null; body?: string }[] = [];

  // Capturar todos los logs de consola
  page.on('console', msg => {
    const text = `[${msg.type().toUpperCase()}] ${msg.text()}`;
    consoleLogs.push(text);
    console.log(text);
  });

  // Capturar errores de página
  page.on('pageerror', error => {
    console.log(`[PAGE ERROR] ${error.message}`);
  });

  // Capturar requests de API
  page.on('response', async response => {
    const url = response.url();
    if (url.includes('/api/models')) {
      const status = response.status();
      let body = '';
      try {
        body = await response.text();
      } catch (e) {}
      networkRequests.push({ url, status, body });
      console.log(`[API RESPONSE] ${status} ${url}`);
      console.log(`[API BODY] ${body.substring(0, 500)}`);
    }
  });

  // Step 1: Ir al dashboard
  console.log('\n=== STEP 1: NAVEGANDO AL DASHBOARD ===');
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);

  // Step 2: Verificar llamada a /api/models
  console.log('\n=== STEP 2: VERIFICAR LLAMADA API MODELS ===');
  const modelsRequest = networkRequests.find(r => r.url.includes('/api/models'));
  if (modelsRequest) {
    console.log(`Models API llamada: ${modelsRequest.status}`);
    console.log(`Respuesta: ${modelsRequest.body?.substring(0, 500)}`);
  } else {
    console.log('NO SE ENCONTRO LLAMADA A /api/models');
  }

  // Step 3: Buscar dropdowns en la página
  console.log('\n=== STEP 3: BUSCAR DROPDOWNS ===');

  // Buscar todos los selects y dropdowns
  const selects = await page.locator('select').all();
  console.log(`Encontrados ${selects.length} elementos <select>`);

  const dropdownButtons = await page.locator('[class*="dropdown"], [role="combobox"], [role="listbox"], button[class*="select"]').all();
  console.log(`Encontrados ${dropdownButtons.length} elementos dropdown/combobox`);

  // Buscar botones con texto de modelo
  const modelButtons = await page.locator('button').filter({ hasText: /PPO|model|Model|investor/i }).all();
  console.log(`Encontrados ${modelButtons.length} botones relacionados con modelo:`);
  for (const btn of modelButtons) {
    const text = await btn.textContent();
    console.log(`  - Button: "${text?.substring(0, 50)}"`);
  }

  // Step 4: Buscar el componente BacktestControlPanel o ModelSelector
  console.log('\n=== STEP 4: BUSCAR COMPONENTES DE MODELO ===');

  // Buscar por data-testid o clases específicas
  const modelSelector = page.locator('[data-testid*="model"], [class*="model-select"], [class*="ModelSelect"]').first();
  if (await modelSelector.count() > 0) {
    console.log('Encontrado selector de modelo');
    const html = await modelSelector.innerHTML();
    console.log(`HTML: ${html.substring(0, 300)}`);
  }

  // Step 5: Tomar screenshot
  await page.screenshot({ path: 'tests/e2e/screenshots/debug-models-01.png', fullPage: true });
  console.log('Screenshot guardado: debug-models-01.png');

  // Step 6: Buscar en el HTML completo por "investor" o "ppo_primary"
  console.log('\n=== STEP 6: BUSCAR EN HTML ===');
  const pageContent = await page.content();

  if (pageContent.includes('investor_demo')) {
    console.log('ENCONTRADO "investor_demo" en el HTML');
  } else {
    console.log('NO SE ENCONTRO "investor_demo" en el HTML');
  }

  if (pageContent.includes('ppo_primary')) {
    console.log('ENCONTRADO "ppo_primary" en el HTML');
  } else {
    console.log('NO SE ENCONTRO "ppo_primary" en el HTML');
  }

  if (pageContent.includes('PPO Primary')) {
    console.log('ENCONTRADO "PPO Primary" en el HTML');
  }

  // Step 7: Verificar ModelContext
  console.log('\n=== STEP 7: LOGS DE MODELCONTEXT ===');
  const modelLogs = consoleLogs.filter(log =>
    log.toLowerCase().includes('model') ||
    log.toLowerCase().includes('investor') ||
    log.toLowerCase().includes('ppo')
  );
  console.log(`Logs relacionados con modelo (${modelLogs.length}):`);
  modelLogs.forEach(log => console.log(`  ${log}`));

  // Step 8: Resumen
  console.log('\n========================================');
  console.log('=== RESUMEN DEBUG ===');
  console.log('========================================');
  console.log(`Total console logs: ${consoleLogs.length}`);
  console.log(`Total network requests a /api/models: ${networkRequests.length}`);

  // Imprimir todos los logs de error
  const errorLogs = consoleLogs.filter(log => log.includes('ERROR') || log.includes('error'));
  if (errorLogs.length > 0) {
    console.log('\n=== ERRORES ===');
    errorLogs.forEach(log => console.log(log));
  }

  expect(true).toBe(true);
});
