import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/final-v2';
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

// Full page
await page.screenshot({ path: `${DIR}/full-page.png`, fullPage: true });
console.log('Full page screenshot');

// Top area (summary)
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/01-summary.png` });

// Charts area
await page.evaluate(() => window.scrollTo(0, 900));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/02-charts-top.png` });

await page.evaluate(() => window.scrollTo(0, 1800));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/03-charts-bottom.png` });

// Timeline
await page.evaluate(() => window.scrollTo(0, 2700));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/04-timeline.png` });

// Methodology section
await page.evaluate(() => {
  const h2s = document.querySelectorAll('h2');
  for (const h of h2s) {
    if (h.textContent?.includes('Metodologia')) {
      h.scrollIntoView({ behavior: 'instant', block: 'start' });
      break;
    }
  }
});
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/05-methodology.png` });

// References section
await page.evaluate(() => {
  const h2s = document.querySelectorAll('h2');
  for (const h of h2s) {
    if (h.textContent?.includes('Referencias')) {
      h.scrollIntoView({ behavior: 'instant', block: 'start' });
      break;
    }
  }
});
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/06-references.png` });

// Final audit
const audit = await page.evaluate(() => {
  const text = document.body.textContent || '';
  const links = document.querySelectorAll('a[href^="http"]');
  const linkDetails = Array.from(links).slice(0, 20).map(l => ({
    text: (l.textContent || '').trim().slice(0, 60),
    href: (l.getAttribute('href') || '').slice(0, 80),
  }));

  return {
    hasSummary: text.includes('SEMANA'),
    hasMacroCards: text.includes('Indicadores Macro'),
    hasCharts: text.includes('DXY (Dollar Index)'),
    hasFuentes: (text.match(/Fuente:/g) || []).length,
    hasTimeline: text.includes('Timeline Diario') || text.includes('Lunes') || text.includes('Martes'),
    hasMethodology: text.includes('Metodologia e Interpretabilidad'),
    hasDrivers: text.includes('Que Mueve el USD/COP'),
    hasReferences: text.includes('Referencias y Fuentes de Datos'),
    hasFRED: text.includes('FRED'),
    hasBanRep: text.includes('Banco de la Republica'),
    hasDisclaimer: text.includes('no constituyen consejo'),
    totalExternalLinks: links.length,
    linkDetails,
  };
});

console.log('\n========================================');
console.log('  FINAL AUDIT v2');
console.log('========================================');
console.log(`Weekly summary: ${audit.hasSummary ? 'YES' : 'NO'}`);
console.log(`Macro cards: ${audit.hasMacroCards ? 'YES' : 'NO'}`);
console.log(`Macro charts: ${audit.hasCharts ? 'YES' : 'NO'}`);
console.log(`"Fuente:" labels: ${audit.hasFuentes}`);
console.log(`Daily timeline: ${audit.hasTimeline ? 'YES' : 'NO'}`);
console.log(`Methodology section: ${audit.hasMethodology ? 'YES' : 'NO'}`);
console.log(`  Macro drivers: ${audit.hasDrivers ? 'YES' : 'NO'}`);
console.log(`References section: ${audit.hasReferences ? 'YES' : 'NO'}`);
console.log(`  FRED: ${audit.hasFRED ? 'YES' : 'NO'}`);
console.log(`  BanRep: ${audit.hasBanRep ? 'YES' : 'NO'}`);
console.log(`Disclaimer: ${audit.hasDisclaimer ? 'YES' : 'NO'}`);
console.log(`\nTotal external hyperlinks: ${audit.totalExternalLinks}`);
console.log('\nSample hyperlinks:');
for (const l of audit.linkDetails.slice(0, 15)) {
  console.log(`  [${l.text.slice(0,40)}] -> ${l.href}`);
}

await browser.close();
console.log('\nDone!');
