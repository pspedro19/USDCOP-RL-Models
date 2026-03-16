import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const DIR = 'tests/e2e/screenshots/final-audit';
mkdirSync(DIR, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

// Navigate to analysis page
console.log('Loading /analysis...');
await page.goto('http://localhost:3000/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });

// Wait for data to load
for (let i = 0; i < 20; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(3000);

// Navigate to W09 (go left from latest week)
console.log('Navigating to W09...');
const allBtns = await page.locator('button').all();
for (const btn of allBtns) {
  const html = await btn.innerHTML().catch(() => '');
  if (html.includes('chevron-left') || html.includes('ChevronLeft') || html.includes('←')) {
    // Check if this is the week nav left button
    const parent = await btn.evaluate(el => el.parentElement?.textContent || '');
    if (parent.includes('SEMANA') || parent.includes('semana')) {
      // Click left to go back weeks — need to reach W09 from W11
      for (let i = 0; i < 2; i++) {
        await btn.click();
        console.log(`  Clicked left (${i + 1})`);
        await page.waitForTimeout(2000);
        // Wait for loading
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
}

// Verify we're on W09
const headerText = await page.textContent('body');
const weekMatch = headerText.match(/SEMANA\s+(\d+)/i);
console.log(`Current week: ${weekMatch?.[0] || 'unknown'}`);

// Wait for all content to render including charts
await page.waitForTimeout(3000);

// Take full-page screenshot
const totalHeight = await page.evaluate(() => document.body.scrollHeight);
console.log(`Page height: ${totalHeight}px`);
await page.screenshot({ path: `${DIR}/w09-full.png`, fullPage: true });

// Scroll-by-scroll screenshots
const vh = 900;
const steps = Math.ceil(totalHeight / vh);
for (let i = 0; i < steps; i++) {
  await page.evaluate((y) => window.scrollTo(0, y), i * vh);
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${DIR}/w09-scroll-${String(i).padStart(2, '0')}.png` });
}

// Run comprehensive audit
const audit = await page.evaluate(() => {
  const body = document.body.textContent || '';
  const html = document.body.innerHTML || '';

  // External links (hyperlinks with href)
  const links = document.querySelectorAll('a[href^="http"]');
  const linkSamples = Array.from(links).slice(0, 10).map(l => ({
    text: (l.textContent || '').trim().slice(0, 80),
    href: (l.getAttribute('href') || '').slice(0, 120),
  }));

  // Source attribution
  const fuenteMatches = body.match(/Fuente:\s*[^\n]+/g) || [];

  // Section detection
  return {
    totalHeight: document.body.scrollHeight,

    // Source attribution
    sourceLabels: fuenteMatches.slice(0, 20),
    sourceCount: fuenteMatches.length,

    // External links
    externalLinks: links.length,
    linkSamples,

    // Key sections present
    hasOHLCV: body.includes('Apertura') && body.includes('Cierre'),
    hasThemes: body.includes('Commodities') || body.includes('Riesgo') || body.includes('FX'),
    hasArticleCount: !!(body.match(/\d+ articulos/)),
    hasSourceBreakdown: body.includes('colombia_scraper') || body.includes('gdelt') || body.includes('investing'),
    hasClusters: body.includes('Clusters de Noticias'),
    hasClusterCount: !!(body.match(/\d+ clusters/)),
    hasBiasSection: body.includes('Sesgo') || body.includes('Diversidad'),
    hasEvents: body.includes('FOMC') || body.includes('BanRep') || body.includes('Eventos'),
    hasMacroCharts: body.includes('DXY (Dollar Index)') || body.includes('VIX (Volatilidad)'),
    hasRegime: body.includes('Regimen') || body.includes('risk_on') || body.includes('risk_off') || body.includes('transition'),
    hasTechnicalAnalysis: body.includes('Analisis Tecnico') || body.includes('Técnico') || body.includes('bullish') || body.includes('bearish'),
    hasTradingScenarios: body.includes('Escenarios') || body.includes('Scenarios'),
    hasSignalCards: body.includes('H5 Semanal') || body.includes('H1 Diario'),
    hasDailyTimeline: body.includes('Lunes') || body.includes('Martes') || body.includes('Miercoles'),
    hasExpandButtons: body.includes('Ver informe completo') || body.includes('Ver analisis completo'),
    hasMacroIndicators: body.includes('Indicadores Macro'),

    // Current week
    currentWeek: body.match(/SEMANA\s+\d+/i)?.[0] || 'unknown',
  };
});

console.log('\n========================================');
console.log('     FINAL VISUAL AUDIT: W09');
console.log('========================================');
console.log(`Week: ${audit.currentWeek}`);
console.log(`Page height: ${audit.totalHeight}px`);
console.log('');

console.log('--- SOURCE ATTRIBUTION ---');
console.log(`"Fuente:" labels found: ${audit.sourceCount}`);
audit.sourceLabels.forEach(s => console.log(`  ${s}`));
console.log('');

console.log('--- EXTERNAL LINKS ---');
console.log(`Clickable hyperlinks: ${audit.externalLinks}`);
audit.linkSamples.forEach(l => console.log(`  [${l.text.slice(0,50)}] -> ${l.href.slice(0,80)}`));
console.log('');

console.log('--- KEY SECTIONS ---');
const sections = [
  ['OHLCV Cards', audit.hasOHLCV],
  ['Theme Badges', audit.hasThemes],
  ['Article Count', audit.hasArticleCount],
  ['Source Breakdown', audit.hasSourceBreakdown],
  ['News Clusters', audit.hasClusters],
  ['Cluster Count', audit.hasClusterCount],
  ['Bias Section', audit.hasBiasSection],
  ['Events Panel', audit.hasEvents],
  ['Macro Charts', audit.hasMacroCharts],
  ['Macro Indicators', audit.hasMacroIndicators],
  ['Regime Indicator', audit.hasRegime],
  ['Technical Analysis', audit.hasTechnicalAnalysis],
  ['Trading Scenarios', audit.hasTradingScenarios],
  ['Signal Cards (H1/H5)', audit.hasSignalCards],
  ['Daily Timeline', audit.hasDailyTimeline],
  ['Expand/Collapse Buttons', audit.hasExpandButtons],
];

let passed = 0;
for (const [name, ok] of sections) {
  const icon = ok ? 'PASS' : 'FAIL';
  console.log(`  [${icon}] ${name}`);
  if (ok) passed++;
}
console.log(`\nScore: ${passed}/${sections.length} sections present`);

await browser.close();
console.log('\nDone! Screenshots saved to', DIR);
