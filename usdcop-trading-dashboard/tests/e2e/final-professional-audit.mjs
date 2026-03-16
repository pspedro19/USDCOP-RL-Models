import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/final-professional';
mkdirSync(DIR, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto('http://localhost:3000/analysis', { waitUntil: 'load', timeout: 30000 });

// Wait for content to appear (not just network idle)
for (let i = 0; i < 30; i++) {
  const ready = await page.evaluate(() => document.querySelectorAll('h2').length > 0);
  if (ready) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(2000); // Extra settle time for charts

// Screenshots
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/01-header-summary.png` });

await page.evaluate(() => window.scrollTo(0, 900));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/02-macro-charts-1.png` });

await page.evaluate(() => window.scrollTo(0, 1800));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/03-macro-charts-2.png` });

await page.evaluate(() => window.scrollTo(0, 2700));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/04-timeline.png` });

// Methodology
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

// References
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

// Bottom
await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
await page.waitForTimeout(300);
await page.screenshot({ path: `${DIR}/07-disclaimer.png` });

// Full page
await page.screenshot({ path: `${DIR}/full-page.png`, fullPage: true });

// COMPREHENSIVE AUDIT
const audit = await page.evaluate(() => {
  const text = document.body.textContent || '';
  const links = document.querySelectorAll('a[href^="http"]');
  const charts = document.querySelectorAll('.recharts-responsive-container');
  const h2Count = document.querySelectorAll('h2').length;
  const h3Count = document.querySelectorAll('h3').length;

  // Visible text issues check
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      const el = node.parentElement;
      if (!el) return NodeFilter.FILTER_REJECT;
      const tag = el.tagName.toLowerCase();
      if (['script', 'style', 'noscript'].includes(tag)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  const visIssues = [];
  let n;
  while ((n = walker.nextNode())) {
    const t = n.textContent.trim();
    if (!t) continue;
    if (t === 'undefined') visIssues.push('undefined');
    if (t === 'null') visIssues.push('null');
    if (t === 'NaN') visIssues.push('NaN');
  }

  // Empty charts check
  const emptyCharts = [];
  charts.forEach((c, i) => {
    const lines = c.querySelectorAll('.recharts-line-curve, .recharts-area');
    if (lines.length === 0) {
      const parent = c.closest('[class*="rounded"]');
      const name = parent?.querySelector('h3')?.textContent || `chart-${i}`;
      emptyCharts.push(name);
    }
  });

  // Link audit
  const linkDetails = Array.from(links).slice(0, 25).map(l => ({
    text: (l.textContent || '').trim().slice(0, 50),
    href: (l.getAttribute('href') || '').slice(0, 80),
    empty: !(l.textContent || '').trim(),
  }));

  return {
    pageHeight: document.body.scrollHeight,
    h2Count, h3Count,
    totalCharts: charts.length,
    emptyCharts,
    totalExternalLinks: links.length,
    emptyLinks: linkDetails.filter(l => l.empty).length,
    linkDetails,
    fuenteLabels: (text.match(/Fuente:/g) || []).length,
    visIssues,
    sections: {
      weekSelector: text.includes('SEMANA'),
      weeklySummary: text.includes('Informe Semanal') || text.includes('Analisis Semanal') || text.includes('Resumen Semanal'),
      themes: text.includes('sesgo') || text.includes('Commodities') || text.includes('aversion'),
      ohlcv: text.includes('Apertura') || text.includes('Cierre') || text.includes('OHLCV'),
      macroCards: text.includes('Indicadores Macro'),
      macroCharts: text.includes('DXY (Dollar Index)'),
      timeline: text.includes('Timeline Diario'),
      methodology: text.includes('Metodologia e Interpretabilidad'),
      drivers: text.includes('Que Mueve el USD/COP'),
      petroleo: text.includes('Petroleo (WTI / Brent)'),
      dxy_driver: text.includes('DXY (Dollar Index)'),
      vix_driver: text.includes('VIX (Indice de Volatilidad)') || text.includes('VIX (Volatilidad)'),
      embi: text.includes('EMBI Colombia'),
      tasasBanRep: text.includes('Tasas BanRep'),
      indicators: text.includes('Indicadores Tecnicos'),
      aiExplain: text.includes('Como se Genera el Analisis'),
      howToRead: text.includes('Como Leer Este Reporte'),
      references: text.includes('Referencias y Fuentes de Datos'),
      fred: text.includes('FRED'),
      banrep: text.includes('Banco de la Republica'),
      gdelt: text.includes('GDELT'),
      portafolio: text.includes('Portafolio'),
      larepublica: text.includes('La Republica'),
      macroSources: text.includes('Fuentes de Datos Macroeconomicos'),
      newsSources: text.includes('Fuentes de Noticias'),
      disclaimer: text.includes('no constituyen consejo de inversion'),
      spanishHeaders: text.includes('Indicadores') && text.includes('Analisis'),
      noEnglishHeaders: !text.includes('Weekly Summary') && !text.includes('Daily Timeline'),
    }
  };
});

console.log('');
console.log('==================================================');
console.log('     FINAL PROFESSIONAL QUALITY AUDIT');
console.log('==================================================');
console.log('');
console.log('-- STRUCTURE --');
console.log(`  Page height: ${audit.pageHeight}px`);
console.log(`  H2 headers: ${audit.h2Count}`);
console.log(`  H3 sub-headers: ${audit.h3Count}`);
console.log(`  Charts rendered: ${audit.totalCharts}`);
console.log(`  External links: ${audit.totalExternalLinks}`);
console.log(`  Empty links: ${audit.emptyLinks}`);
console.log(`  "Fuente:" labels: ${audit.fuenteLabels}`);
console.log('');
console.log('-- TEXT QUALITY --');
if (audit.visIssues.length === 0) {
  console.log('  [OK] No visible undefined/null/NaN text');
} else {
  audit.visIssues.forEach(i => console.log(`  [WARN] ${i}`));
}
console.log('');
console.log('-- CHARTS --');
console.log(`  Total: ${audit.totalCharts}`);
if (audit.emptyCharts.length > 0) {
  console.log(`  Empty: ${audit.emptyCharts.join(', ')}`);
} else if (audit.totalCharts > 0) {
  console.log('  [OK] All charts have data');
}
console.log('');
console.log('-- SECTIONS --');
const checks = Object.entries(audit.sections);
let passed = 0;
for (const [key, ok] of checks) {
  const label = key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim();
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label}`);
  if (ok) passed++;
}
console.log('');
console.log('-- LINKS (sample) --');
audit.linkDetails.filter(l => !l.empty).slice(0, 15).forEach((l, i) => {
  console.log(`  ${i+1}. [${l.text.slice(0,35)}] -> ${l.href}`);
});
console.log('');
console.log(`== SCORE: ${passed}/${checks.length} sections ==`);
console.log(`== TEXT QUALITY: ${audit.visIssues.length === 0 ? 'CLEAN' : audit.visIssues.length + ' issues'} ==`);
console.log(`== STATUS: ${passed >= checks.length - 3 ? 'PROFESSIONAL QUALITY' : 'NEEDS WORK'} ==`);

await browser.close();
console.log(`\nScreenshots saved to ${DIR}`);
