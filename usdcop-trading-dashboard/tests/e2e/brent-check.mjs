import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/final-audit';
mkdirSync(DIR, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

console.log('Loading /analysis...');
await page.goto('http://localhost:3000/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });
for (let i = 0; i < 20; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(3000);

// Scroll to Brent chart area (bottom of chart grid)
await page.evaluate(() => window.scrollTo(0, 2400));
await page.waitForTimeout(1000);
await page.screenshot({ path: `${DIR}/brent-fix-check.png` });
console.log('Brent area screenshot taken');

// Also take W09 macro cards screenshot by navigating there
// Find and click the left button twice
const buttons = await page.locator('button').all();
for (const btn of buttons) {
  const box = await btn.boundingBox().catch(() => null);
  if (box && box.x < 30 && box.y < 80 && box.width < 50) {
    console.log('Found left nav button, clicking twice...');
    await btn.click();
    await page.waitForTimeout(3000);
    for (let j = 0; j < 15; j++) {
      const t = await page.textContent('body');
      if (!t.includes('Cargando analisis')) break;
      await page.waitForTimeout(500);
    }
    await btn.click();
    await page.waitForTimeout(3000);
    for (let j = 0; j < 15; j++) {
      const t = await page.textContent('body');
      if (!t.includes('Cargando analisis')) break;
      await page.waitForTimeout(500);
    }
    break;
  }
}

await page.waitForTimeout(2000);
const week = await page.evaluate(() => document.body.textContent?.match(/SEMANA\s+\d+/i)?.[0]);
console.log(`Now on: ${week}`);

// Screenshot macro cards with source labels
await page.evaluate(() => {
  const h2s = document.querySelectorAll('h2');
  for (const h of h2s) {
    if (h.textContent?.includes('Indicadores')) {
      h.scrollIntoView({ behavior: 'instant' });
      break;
    }
  }
});
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/w09-macro-cards-sources.png` });
console.log('W09 macro cards screenshot taken');

// Scroll to Brent chart area on W09
await page.evaluate(() => window.scrollTo(0, 2400));
await page.waitForTimeout(1000);
await page.screenshot({ path: `${DIR}/w09-brent-fix.png` });
console.log('W09 Brent chart screenshot taken');

await browser.close();
console.log('Done!');
