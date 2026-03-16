import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const BASE = 'http://localhost:3000/analysis';
const DIR = 'tests/e2e/screenshots/analysis-audit';
mkdirSync(DIR, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

console.log('Navigating to /analysis...');
await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 30000 });

// Wait for loading to finish
for (let i = 0; i < 20; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  console.log(`  waiting for data... (${i+1})`);
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(2000);

// Full page screenshot
console.log('Taking full-page screenshot...');
await page.screenshot({ path: `${DIR}/full-page.png`, fullPage: true });

// Get total height
const totalHeight = await page.evaluate(() => document.body.scrollHeight);
console.log(`Page height: ${totalHeight}px`);

// Scroll screenshots
const vh = 900;
const steps = Math.ceil(totalHeight / vh);
for (let i = 0; i < steps; i++) {
  await page.evaluate((y) => window.scrollTo(0, y), i * vh);
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${DIR}/scroll-${String(i).padStart(2, '0')}.png` });
  console.log(`  scroll-${String(i).padStart(2, '0')}.png (y=${i * vh})`);
}

// Audit
const audit = await page.evaluate(() => {
  const body = document.body.textContent || '';
  const links = document.querySelectorAll('a[href^="http"]');
  const headings = document.querySelectorAll('h2, h3');
  const charts = document.querySelectorAll('.recharts-wrapper, svg.recharts-surface');
  const cards = document.querySelectorAll('[class*="rounded-xl"]');

  return {
    totalHeight: document.body.scrollHeight,
    externalLinks: links.length,
    linkSamples: Array.from(links).slice(0, 15).map(l => ({
      text: (l.textContent || '').trim().slice(0, 80),
      href: (l.getAttribute('href') || '').slice(0, 100),
    })),
    headings: Array.from(headings).map(h => (h.textContent || '').trim().slice(0, 80)),
    chartElements: charts.length,
    cardElements: cards.length,
    hasFuentes: body.includes('Fuentes'),
    hasZeroSentiment: body.includes('0.000'),
    hasDiagnostico: body.includes('Diagnóstico') || body.includes('Diagnostico'),
    hasEscenarios: body.includes('Escenarios'),
    hasMacro: body.includes('Indicadores Macro'),
    hasTimeline: body.includes('Cronología') || body.includes('Timeline'),
  };
});

console.log('\n=== PAGE AUDIT ===');
console.log(JSON.stringify(audit, null, 2));

// Now test an older week (W05) with articles
console.log('\nNavigating to W05...');
// Click previous arrow 6 times
for (let i = 0; i < 6; i++) {
  const prevBtn = page.locator('button svg.lucide-chevron-left, button:has(svg[class*="chevron"])').first();
  const altBtn = page.locator('button').filter({ hasText: '<' }).first();

  try {
    if (await prevBtn.isVisible({ timeout: 1000 })) {
      await prevBtn.click();
    } else if (await altBtn.isVisible({ timeout: 1000 })) {
      await altBtn.click();
    }
  } catch(e) {}
  await page.waitForTimeout(1500);
}

// Wait for load
for (let i = 0; i < 10; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(2000);

await page.evaluate(() => window.scrollTo(0, 0));
await page.screenshot({ path: `${DIR}/week05-top.png` });
await page.screenshot({ path: `${DIR}/week05-full.png`, fullPage: true });
console.log('Saved W05 screenshots');

await browser.close();
console.log('\nDone!');
