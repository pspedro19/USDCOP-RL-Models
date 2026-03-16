import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/final-audit';
mkdirSync(DIR, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

// Load default landing page (latest week)
console.log('Loading /analysis (latest week)...');
await page.goto('http://localhost:3000/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });
for (let i = 0; i < 20; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(3000);

// Screenshot 1: Top (summary + cards)
await page.screenshot({ path: `${DIR}/latest-top.png` });
console.log('Screenshot 1: Top');

// Scroll to macro indicators section
await page.evaluate(() => {
  const el = document.querySelector('h2');
  if (el) el.scrollIntoView({ behavior: 'instant' });
});
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/latest-macro-cards.png` });
console.log('Screenshot 2: Macro indicator cards with source labels');

// Scroll to charts
await page.evaluate(() => window.scrollTo(0, 1500));
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/latest-charts-top.png` });
console.log('Screenshot 3: Charts top row');

await page.evaluate(() => window.scrollTo(0, 2300));
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/latest-charts-bottom.png` });
console.log('Screenshot 4: Charts bottom row');

// Scroll to daily timeline
await page.evaluate(() => window.scrollTo(0, 3200));
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/latest-daily-timeline.png` });
console.log('Screenshot 5: Daily timeline');

// Full page
await page.screenshot({ path: `${DIR}/latest-full.png`, fullPage: true });
console.log('Screenshot 6: Full page');

// Now navigate to W09 using URL param approach
console.log('\nLoading W09 directly via URL...');
await page.goto('http://localhost:3000/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });
for (let i = 0; i < 20; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(2000);

// Click left arrow button to navigate backwards
// Find the left chevron button specifically
const leftBtns = await page.locator('button svg').all();
for (const svg of leftBtns) {
  const parent = await svg.locator('..').first();
  try {
    const svgHtml = await svg.evaluate(el => el.outerHTML);
    if (svgHtml.includes('M15') || svgHtml.includes('polyline') || svgHtml.includes('chevron')) {
      // Try clicking the parent button
      const bbox = await parent.boundingBox();
      if (bbox && bbox.x < 100) { // Left-side button
        for (let i = 0; i < 2; i++) {
          await parent.click();
          console.log(`Clicked left arrow (${i + 1})`);
          await page.waitForTimeout(2000);
          for (let j = 0; j < 10; j++) {
            const t = await page.textContent('body');
            if (!t.includes('Cargando analisis')) break;
            await page.waitForTimeout(500);
          }
          await page.waitForTimeout(1500);
        }
        break;
      }
    }
  } catch { /* skip */ }
}

await page.waitForTimeout(2000);
const currentWeek = await page.evaluate(() =>
  document.body.textContent?.match(/SEMANA\s+\d+/i)?.[0] || 'unknown'
);
console.log(`Now on: ${currentWeek}`);

// Screenshots of W09 sections
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/w09-top.png` });

// Scroll to bias/clusters section
await page.evaluate(() => window.scrollTo(0, 2800));
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/w09-bias-clusters.png` });

// Full page
await page.screenshot({ path: `${DIR}/w09-full.png`, fullPage: true });

// Count source labels
const sources = await page.evaluate(() => {
  const text = document.body.textContent || '';
  const matches = text.match(/Fuente:\s*\S+/g) || [];
  return matches;
});
console.log(`\nSource attribution labels: ${sources.length}`);
sources.slice(0, 16).forEach(s => console.log(`  ${s}`));

await browser.close();
console.log('\nDone!');
