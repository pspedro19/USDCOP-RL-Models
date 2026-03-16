import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/analysis-audit';
mkdirSync(DIR, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

// Go directly to W09 via the dropdown
console.log('Loading /analysis...');
await page.goto('http://localhost:3000/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });

// Wait for load
for (let i = 0; i < 15; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(2000);

// Click the dropdown to select W09
const dropdown = page.locator('select').first();
if (await dropdown.isVisible({ timeout: 2000 }).catch(() => false)) {
  console.log('Found dropdown, selecting W09...');
  await dropdown.selectOption({ label: /W09|W9/ });
  await page.waitForTimeout(3000);
} else {
  // Try clicking the left arrow to navigate to W09 (2 clicks from W11)
  console.log('No dropdown, clicking left arrows...');
  for (let i = 0; i < 2; i++) {
    const leftBtn = page.locator('button').first();
    const allBtns = await page.locator('button').all();
    for (const btn of allBtns) {
      const html = await btn.innerHTML().catch(() => '');
      if (html.includes('chevron-left') || html.includes('ChevronLeft')) {
        await btn.click();
        console.log(`  Clicked left arrow (${i+1})`);
        break;
      }
    }
    await page.waitForTimeout(2000);
  }

  // Wait for data
  for (let i = 0; i < 10; i++) {
    const text = await page.textContent('body');
    if (!text.includes('Cargando analisis')) break;
    await page.waitForTimeout(1000);
  }
  await page.waitForTimeout(2000);
}

console.log('Taking W09 screenshots...');

// Full page
const totalHeight = await page.evaluate(() => document.body.scrollHeight);
console.log(`Page height: ${totalHeight}px`);

await page.screenshot({ path: `${DIR}/w09-full.png`, fullPage: true });

// Scroll screenshots
const vh = 900;
const steps = Math.ceil(totalHeight / vh);
for (let i = 0; i < steps; i++) {
  await page.evaluate((y) => window.scrollTo(0, y), i * vh);
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${DIR}/w09-scroll-${String(i).padStart(2, '0')}.png` });
}

// Audit
const audit = await page.evaluate(() => {
  const body = document.body.textContent || '';
  const links = document.querySelectorAll('a[href^="http"]');
  return {
    totalHeight: document.body.scrollHeight,
    externalLinks: links.length,
    linkSamples: Array.from(links).slice(0, 10).map(l => ({
      text: (l.textContent || '').trim().slice(0, 60),
      href: (l.getAttribute('href') || '').slice(0, 100),
    })),
    hasOHLCV: body.includes('Apertura') && body.includes('Cierre'),
    hasThemes: body.includes('Commodities') || body.includes('Riesgo'),
    hasArticleCount: !!body.match(/\d+ articulos/),
    hasSourceBreakdown: body.includes('colombia_scraper') || body.includes('gdelt'),
    hasClusters: body.includes('cluster'),
    hasEvents: body.includes('FOMC') || body.includes('BanRep'),
    currentWeek: body.match(/SEMANA\s+\d+/i)?.[0] || 'unknown',
  };
});

console.log('\n=== W09 AUDIT ===');
console.log(JSON.stringify(audit, null, 2));

await browser.close();
console.log('\nDone!');
