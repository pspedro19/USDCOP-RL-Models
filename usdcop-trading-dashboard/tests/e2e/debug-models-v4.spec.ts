/**
 * Debug Models Dropdown V4 - Wait longer and capture all logs
 */

import { test, expect } from '@playwright/test';

test.use({ browserName: 'chromium' });

test('debug ModelDropdown v4', async ({ page }) => {
  const allLogs: string[] = [];

  page.on('console', msg => {
    const text = `[${msg.type()}] ${msg.text()}`;
    allLogs.push(text);
    console.log(text);
  });

  page.on('pageerror', err => console.log(`[PAGE ERROR] ${err.message}`));

  console.log('Navegando a dashboard...');
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle', timeout: 60000 });

  // Wait for potential async loading
  console.log('Esperando 10 segundos para carga completa...');
  await page.waitForTimeout(10000);

  // Check for skeleton or actual dropdown
  console.log('\n=== VERIFICAR SKELETON VS DROPDOWN ===');
  const skeleton = page.locator('.animate-pulse.rounded-xl.bg-slate-800');
  const skeletonCount = await skeleton.count();
  console.log(`Skeletons visibles: ${skeletonCount}`);

  if (skeletonCount > 0) {
    console.log('PROBLEMA: Todavía hay skeletons - los modelos no cargaron');
  }

  // Check for actual model dropdown button
  const dropdownButton = page.locator('button:has(.rounded-full)').filter({ hasText: /PPO|Model|Select|Investor/i });
  const dropdownCount = await dropdownButton.count();
  console.log(`Dropdown buttons encontrados: ${dropdownCount}`);

  // Check for "No models available" message
  const noModels = page.locator('text=No models available');
  if (await noModels.count() > 0) {
    console.log('ENCONTRADO: "No models available"');
  }

  // Check network requests
  console.log('\n=== VERIFICAR FETCH A /api/models ===');
  const modelLogs = allLogs.filter(l => l.includes('model') || l.includes('Model') || l.includes('fetch'));
  console.log(`Logs relacionados: ${modelLogs.length}`);
  modelLogs.forEach(l => console.log(l));

  // Make direct API call to verify endpoint works
  console.log('\n=== LLAMADA DIRECTA A API ===');
  const apiResponse = await page.request.get('http://localhost:5000/api/models');
  console.log(`Status: ${apiResponse.status()}`);
  const apiData = await apiResponse.json();
  console.log(`Modelos recibidos: ${apiData.models?.length || 0}`);
  if (apiData.models) {
    apiData.models.forEach((m: { id: string; name: string }) => {
      console.log(`  - ${m.id}: ${m.name}`);
    });
  }

  // Screenshot
  await page.screenshot({ path: 'tests/e2e/screenshots/debug-v4.png', fullPage: true });

  expect(true).toBe(true);
});
