import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/sections-audit';
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

// Full page screenshot
await page.screenshot({ path: `${DIR}/full-page.png`, fullPage: true });
console.log('Full page screenshot taken');

// Check for new sections
const audit = await page.evaluate(() => {
  const text = document.body.textContent || '';
  return {
    hasMethodology: text.includes('Metodologia e Interpretabilidad'),
    hasReferences: text.includes('Referencias y Fuentes de Datos'),
    hasFREDLink: text.includes('FRED (Federal Reserve Economic Data)'),
    hasBanRepLink: text.includes('Banco de la Republica'),
    hasInvestingLink: text.includes('Investing.com'),
    hasDrivers: text.includes('Que Mueve el USD/COP'),
    hasPetroleo: text.includes('Petroleo (WTI / Brent)'),
    hasDXY: text.includes('DXY (Dollar Index)'),
    hasVIX: text.includes('VIX (Indice de Volatilidad)'),
    hasEMBI: text.includes('EMBI Colombia (Riesgo Pais)'),
    hasTasas: text.includes('Tasas BanRep'),
    hasIndicators: text.includes('Indicadores Tecnicos'),
    hasSMA: text.includes('SMA (Simple Moving Average)'),
    hasRSI: text.includes('RSI (Relative Strength Index)'),
    hasBollinger: text.includes('Bandas de Bollinger'),
    hasPipeline: text.includes('Pipeline Diario'),
    hasLimitaciones: text.includes('Limitaciones y Advertencias'),
    hasDisclaimer: text.includes('NO constituye consejo de inversion'),
    hasFuenteMacro: text.includes('Fuentes de Datos Macroeconomicos'),
    hasFuenteNoticias: text.includes('Fuentes de Noticias'),
    hasGDELT: text.includes('GDELT Project'),
    hasPortafolio: text.includes('Portafolio.co'),
    hasLaRepublica: text.includes('La Republica'),
    hasHowTo: text.includes('Como Leer Este Reporte'),
    hasArticleLinks: document.querySelectorAll('a[href^="http"]').length,
  };
});

console.log('\n========================================');
console.log('  NEW SECTIONS AUDIT');
console.log('========================================');

const checks = [
  ['Methodology section', audit.hasMethodology],
  ['References section', audit.hasReferences],
  ['FRED link', audit.hasFREDLink],
  ['BanRep link', audit.hasBanRepLink],
  ['Investing.com link', audit.hasInvestingLink],
  ['Macro drivers (Que Mueve)', audit.hasDrivers],
  ['Petroleo driver', audit.hasPetroleo],
  ['DXY driver', audit.hasDXY],
  ['VIX driver', audit.hasVIX],
  ['EMBI driver', audit.hasEMBI],
  ['Tasas BanRep driver', audit.hasTasas],
  ['Indicators section', audit.hasIndicators],
  ['SMA explainer', audit.hasSMA],
  ['RSI explainer', audit.hasRSI],
  ['Bollinger explainer', audit.hasBollinger],
  ['Pipeline explanation', audit.hasPipeline],
  ['Limitations & warnings', audit.hasLimitaciones],
  ['Investment disclaimer', audit.hasDisclaimer],
  ['Macro data sources header', audit.hasFuenteMacro],
  ['News sources header', audit.hasFuenteNoticias],
  ['GDELT source', audit.hasGDELT],
  ['Portafolio source', audit.hasPortafolio],
  ['La Republica source', audit.hasLaRepublica],
  ['How to read guide', audit.hasHowTo],
];

let passed = 0;
for (const [name, ok] of checks) {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}`);
  if (ok) passed++;
}
console.log(`\nScore: ${passed}/${checks.length} checks passed`);
console.log(`External links on page: ${audit.hasArticleLinks}`);

// Scroll to methodology section and screenshot
await page.evaluate(() => {
  const h2s = document.querySelectorAll('h2');
  for (const h of h2s) {
    if (h.textContent?.includes('Metodologia')) {
      h.scrollIntoView({ behavior: 'instant' });
      break;
    }
  }
});
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/methodology-section.png` });
console.log('\nMethodology screenshot taken');

// Scroll to references section and screenshot
await page.evaluate(() => {
  const h2s = document.querySelectorAll('h2');
  for (const h of h2s) {
    if (h.textContent?.includes('Referencias')) {
      h.scrollIntoView({ behavior: 'instant' });
      break;
    }
  }
});
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/references-section.png` });
console.log('References screenshot taken');

// Scroll to Brent chart to check if outlier filtering works
await page.evaluate(() => window.scrollTo(0, 2300));
await page.waitForTimeout(500);
await page.screenshot({ path: `${DIR}/charts-brent-area.png` });
console.log('Brent chart area screenshot taken');

await browser.close();
console.log('\nDone! Screenshots in', DIR);
