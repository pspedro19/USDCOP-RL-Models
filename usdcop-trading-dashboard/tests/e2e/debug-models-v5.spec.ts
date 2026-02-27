/**
 * Debug Models Dropdown V5 - Find the actual dropdown
 */

import { test, expect } from '@playwright/test';

test.use({ browserName: 'chromium' });

test('find model dropdown', async ({ page }) => {
  page.on('console', msg => {
    if (msg.type() === 'error') console.log(`[ERROR] ${msg.text()}`);
  });

  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(5000);

  // Print ALL buttons text
  console.log('\n=== TODOS LOS BOTONES ===');
  const allButtons = await page.locator('button').all();
  for (let i = 0; i < allButtons.length; i++) {
    const text = (await allButtons[i].textContent())?.trim().replace(/\\s+/g, ' ');
    if (text) console.log(`${i}: "${text.substring(0, 80)}"`);
  }

  // Search for PPO text anywhere in page
  console.log('\n=== BUSCAR TEXTO PPO EN PAGINA ===');
  const ppoElements = await page.locator('*:has-text("PPO")').all();
  console.log(`Elementos con PPO: ${ppoElements.length}`);

  // Search for model-related divs with role=button or clickable
  console.log('\n=== BUSCAR ELEMENTOS CLICKABLES CON MODELO ===');
  const clickables = await page.locator('[role="button"], [class*="cursor-pointer"]').all();
  console.log(`Elementos clickables: ${clickables.length}`);

  // Get header HTML
  console.log('\n=== HEADER COMPLETO ===');
  const header = await page.locator('header').innerHTML();
  console.log(header.substring(0, 2000));

  // Take screenshot
  await page.screenshot({ path: 'tests/e2e/screenshots/debug-v5.png', fullPage: true });

  expect(true).toBe(true);
});
