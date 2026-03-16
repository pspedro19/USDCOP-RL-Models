import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/sections-audit';
mkdirSync(DIR, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

await page.goto('http://localhost:3000/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });
for (let i = 0; i < 20; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(3000);

// Find and scroll to Brent chart
await page.evaluate(() => {
  const h3s = document.querySelectorAll('h3');
  for (const h of h3s) {
    if (h.textContent?.includes('Brent')) {
      h.scrollIntoView({ behavior: 'instant', block: 'center' });
      break;
    }
  }
});
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/brent-chart-focused.png` });
console.log('Brent chart screenshot taken');

// Also check which charts are showing and their data state
const chartAudit = await page.evaluate(() => {
  const charts = document.querySelectorAll('.recharts-responsive-container');
  const chartInfo = [];
  for (const c of charts) {
    const parent = c.closest('[class*="rounded-xl"]');
    const h3 = parent?.querySelector('h3');
    const name = h3?.textContent || 'unknown';
    const yTicks = c.querySelectorAll('.recharts-yAxis .recharts-cartesian-axis-tick-value');
    const yValues = Array.from(yTicks).map(t => t.textContent);
    chartInfo.push({ name, yValues });
  }
  return chartInfo;
});

console.log('\nChart Y-axis values:');
for (const c of chartAudit) {
  console.log(`  ${c.name}: ${c.yValues.join(', ')}`);
}

await browser.close();
console.log('\nDone!');
