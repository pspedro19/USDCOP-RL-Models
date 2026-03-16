import { test } from '@playwright/test';

const BASE = 'http://localhost:3000/analysis';
const DIR = 'tests/e2e/screenshots/analysis-audit';

test('Scroll through analysis page capturing every viewport', async ({ page }) => {
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(BASE, { waitUntil: 'networkidle' });

  // Wait for content to load (not "Cargando analisis...")
  await page.waitForFunction(() => {
    return !document.body.textContent?.includes('Cargando analisis');
  }, { timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(2000);

  // Get total page height
  const totalHeight = await page.evaluate(() => document.body.scrollHeight);
  console.log(`Total page height: ${totalHeight}px`);

  // Take screenshots at every viewport-height step
  const viewportHeight = 900;
  const steps = Math.ceil(totalHeight / viewportHeight);
  console.log(`Taking ${steps} scroll screenshots`);

  for (let i = 0; i < steps; i++) {
    await page.evaluate((y) => window.scrollTo(0, y), i * viewportHeight);
    await page.waitForTimeout(500);
    await page.screenshot({
      path: `${DIR}/scroll-${String(i).padStart(2, '0')}.png`,
      fullPage: false,
    });
  }

  // Also take one full-page screenshot
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(500);
  await page.screenshot({
    path: `${DIR}/full-page.png`,
    fullPage: true,
  });

  // Audit: count links, check sections, log issues
  const audit = await page.evaluate(() => {
    const body = document.body.textContent || '';
    const links = document.querySelectorAll('a[href^="http"]');
    const headings = document.querySelectorAll('h2, h3');
    const charts = document.querySelectorAll('svg, canvas, .recharts-wrapper');
    const cards = document.querySelectorAll('[class*="rounded-xl"]');

    return {
      totalHeight: document.body.scrollHeight,
      externalLinks: links.length,
      linkSamples: Array.from(links).slice(0, 10).map(l => ({
        text: l.textContent?.trim().slice(0, 60),
        href: l.getAttribute('href')?.slice(0, 80),
      })),
      headings: Array.from(headings).map(h => h.textContent?.trim().slice(0, 80)),
      chartElements: charts.length,
      cardElements: cards.length,
      hasFuentes: body.includes('Fuentes'),
      hasZeroSentiment: body.includes('0.000'),
      hasDiagnostico: body.includes('Diagnóstico'),
      hasEscenarios: body.includes('Escenarios'),
      hasMacro: body.includes('Indicadores Macro'),
      hasTimeline: body.includes('Timeline') || body.includes('Cronolog'),
    };
  });

  console.log('\n=== PAGE AUDIT ===');
  console.log(JSON.stringify(audit, null, 2));
});
