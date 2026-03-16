import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/professional-audit';
mkdirSync(DIR, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

console.log('Loading /analysis...');
await page.goto('http://localhost:3000/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });
for (let i = 0; i < 25; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(3000);

// Scroll-by-scroll screenshots of the ENTIRE page
const totalHeight = await page.evaluate(() => document.body.scrollHeight);
console.log(`Page height: ${totalHeight}px`);

const vh = 900;
const steps = Math.ceil(totalHeight / vh);
for (let i = 0; i < steps; i++) {
  await page.evaluate((y) => window.scrollTo(0, y), i * vh);
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${DIR}/scroll-${String(i).padStart(2, '0')}.png` });
}
console.log(`Took ${steps} scroll screenshots`);

// Full page
await page.screenshot({ path: `${DIR}/full-page.png`, fullPage: true });

// ---- COMPREHENSIVE TEXT AUDIT ----
const audit = await page.evaluate(() => {
  const body = document.body.textContent || '';
  const html = document.body.innerHTML || '';

  // Find all text issues
  const issues = [];

  // Check for broken/placeholder text
  if (body.includes('undefined')) issues.push('Found "undefined" in text');
  if (body.includes('null')) issues.push('Found "null" in text');
  if (body.includes('NaN')) issues.push('Found "NaN" in text');
  if (body.includes('[object')) issues.push('Found "[object" in text');
  if (body.includes('TODO')) issues.push('Found "TODO" in text');
  if (body.includes('FIXME')) issues.push('Found "FIXME" in text');
  if (body.includes('lorem')) issues.push('Found "lorem" placeholder text');
  if (body.includes('placeholder')) issues.push('Found "placeholder" text');

  // Check for empty sections
  const h2s = document.querySelectorAll('h2');
  const h3s = document.querySelectorAll('h3');

  // Check for broken images
  const imgs = document.querySelectorAll('img');
  let brokenImgs = 0;
  imgs.forEach(img => {
    if (!img.complete || img.naturalHeight === 0) brokenImgs++;
  });

  // Check all external links are valid (have href)
  const links = document.querySelectorAll('a[href^="http"]');
  let emptyLinks = 0;
  links.forEach(l => {
    if (!l.textContent?.trim()) emptyLinks++;
  });

  // Key sections present
  const sections = {
    weekSelector: body.includes('SEMANA'),
    weeklySummary: body.includes('Analisis Semanal') || body.includes('Resumen'),
    themes: body.includes('Commodities') || body.includes('Riesgo') || body.includes('Volatilidad'),
    ohlcv: body.includes('Apertura') || body.includes('Cierre') || body.includes('OHLCV'),
    macroCards: body.includes('Indicadores Macro'),
    macroCharts: body.includes('DXY (Dollar Index)'),
    chartSources: (body.match(/Fuente:/g) || []).length,
    timeline: body.includes('Timeline Diario'),
    methodology: body.includes('Metodologia e Interpretabilidad'),
    drivers: body.includes('Que Mueve el USD/COP'),
    petroleo: body.includes('Petroleo (WTI / Brent)'),
    dxy: body.includes('DXY (Dollar Index)'),
    vix: body.includes('VIX (Indice de Volatilidad)') || body.includes('VIX (Volatilidad)'),
    embi: body.includes('EMBI Colombia'),
    tasasBanRep: body.includes('Tasas BanRep'),
    indicators: body.includes('Indicadores Tecnicos'),
    aiExplainer: body.includes('Como se Genera el Analisis'),
    howToRead: body.includes('Como Leer Este Reporte'),
    references: body.includes('Referencias y Fuentes de Datos'),
    macroSources: body.includes('Fuentes de Datos Macroeconomicos'),
    newsSources: body.includes('Fuentes de Noticias'),
    disclaimer: body.includes('no constituyen consejo de inversion'),
    fred: body.includes('FRED (Federal Reserve'),
    banrep: body.includes('Banco de la Republica'),
    gdelt: body.includes('GDELT Project'),
    portafolio: body.includes('Portafolio.co'),
    larepublica: body.includes('La Republica'),
  };

  // Check for Spanish text consistency
  const spanishChecks = {
    hasSpanishDays: body.includes('Lunes') || body.includes('Martes'),
    hasSpanishHeaders: body.includes('Indicadores') && body.includes('Analisis'),
    noEnglishHeaders: !body.includes('Weekly Summary') && !body.includes('Daily Timeline'),
  };

  // Recharts rendered properly
  const rechartsContainers = document.querySelectorAll('.recharts-responsive-container');
  const emptyCharts = [];
  rechartsContainers.forEach((c, i) => {
    const lines = c.querySelectorAll('.recharts-line-curve');
    if (lines.length === 0) {
      const parent = c.closest('[class*="rounded"]');
      const name = parent?.querySelector('h3')?.textContent || `chart-${i}`;
      emptyCharts.push(name);
    }
  });

  return {
    issues,
    sections,
    spanishChecks,
    totalLinks: links.length,
    emptyLinks,
    brokenImgs,
    totalCharts: rechartsContainers.length,
    emptyCharts,
    h2Count: h2s.length,
    h3Count: h3s.length,
    pageHeight: document.body.scrollHeight,
  };
});

console.log('\n╔══════════════════════════════════════════╗');
console.log('║   PROFESSIONAL QUALITY AUDIT             ║');
console.log('╚══════════════════════════════════════════╝');

// Text issues
console.log('\n── TEXT QUALITY ──');
if (audit.issues.length === 0) {
  console.log('  [OK] No placeholder/broken text found');
} else {
  audit.issues.forEach(i => console.log(`  [WARN] ${i}`));
}
console.log(`  Broken images: ${audit.brokenImgs}`);
console.log(`  Empty links: ${audit.emptyLinks}`);

// Sections
console.log('\n── PAGE SECTIONS ──');
const sectionChecks = [
  ['Selector de Semana', audit.sections.weekSelector],
  ['Resumen Semanal', audit.sections.weeklySummary],
  ['Temas Principales', audit.sections.themes],
  ['OHLCV USD/COP', audit.sections.ohlcv],
  ['Tarjetas Macro', audit.sections.macroCards],
  ['Graficos Macro', audit.sections.macroCharts],
  ['Timeline Diario', audit.sections.timeline],
  ['Metodologia', audit.sections.methodology],
  ['Drivers USD/COP', audit.sections.drivers],
  ['  - Petroleo', audit.sections.petroleo],
  ['  - DXY', audit.sections.dxy],
  ['  - VIX', audit.sections.vix],
  ['  - EMBI', audit.sections.embi],
  ['  - Tasas BanRep', audit.sections.tasasBanRep],
  ['Indicadores Tecnicos', audit.sections.indicators],
  ['Explicacion IA', audit.sections.aiExplainer],
  ['Como Leer el Reporte', audit.sections.howToRead],
  ['Referencias', audit.sections.references],
  ['  - FRED', audit.sections.fred],
  ['  - BanRep', audit.sections.banrep],
  ['  - GDELT', audit.sections.gdelt],
  ['  - Portafolio', audit.sections.portafolio],
  ['  - La Republica', audit.sections.larepublica],
  ['Fuentes Macro', audit.sections.macroSources],
  ['Fuentes Noticias', audit.sections.newsSources],
  ['Disclaimer Legal', audit.sections.disclaimer],
];

let passed = 0;
for (const [name, ok] of sectionChecks) {
  const icon = ok ? 'PASS' : 'FAIL';
  console.log(`  [${icon}] ${name}`);
  if (ok) passed++;
}

console.log('\n── CHARTS ──');
console.log(`  Total charts rendered: ${audit.totalCharts}`);
console.log(`  "Fuente:" labels: ${audit.sections.chartSources}`);
if (audit.emptyCharts.length > 0) {
  console.log(`  Empty charts: ${audit.emptyCharts.join(', ')}`);
} else {
  console.log('  [OK] All charts have data');
}

console.log('\n── LINKS & REFS ──');
console.log(`  External hyperlinks: ${audit.totalLinks}`);

console.log('\n── SPANISH CONSISTENCY ──');
console.log(`  Spanish day names: ${audit.spanishChecks.hasSpanishDays ? 'YES' : 'NO'}`);
console.log(`  Spanish headers: ${audit.spanishChecks.hasSpanishHeaders ? 'YES' : 'NO'}`);
console.log(`  No English headers: ${audit.spanishChecks.noEnglishHeaders ? 'YES' : 'NO'}`);

console.log('\n── STRUCTURE ──');
console.log(`  H2 section headers: ${audit.h2Count}`);
console.log(`  H3 sub-headers: ${audit.h3Count}`);
console.log(`  Page height: ${audit.pageHeight}px`);

console.log(`\n══ SCORE: ${passed}/${sectionChecks.length} sections present ══`);

await browser.close();
console.log('\nScreenshots saved to', DIR);
