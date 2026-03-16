import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

// Log ALL network responses
page.on('response', async (response) => {
  const url = response.url();
  if (url.includes('localhost:3000') && !url.includes('_next/static') && !url.includes('.js') && !url.includes('.css') && !url.includes('webpack')) {
    console.log(`[${response.status()}] ${url.replace('http://localhost:3000', '')}`);
  }
});

await page.goto('http://localhost:3000/analysis', { waitUntil: 'load', timeout: 30000 });
console.log('\n--- Page loaded ---');

// Wait and check
await page.waitForTimeout(10000);

// Try to fetch from browser context
const apiResult = await page.evaluate(async () => {
  try {
    const res = await fetch('/api/analysis/weeks');
    const data = await res.json();
    return { status: res.status, weeks: data?.weeks?.length || 0, first: data?.weeks?.[0] };
  } catch (e) {
    return { error: e.message };
  }
});
console.log('In-browser fetch /api/analysis/weeks:', JSON.stringify(apiResult));

if (apiResult.first) {
  const weekRes = await page.evaluate(async (yr, wk) => {
    try {
      const res = await fetch(`/api/analysis/week/${yr}/${wk}`);
      const data = await res.json();
      return { status: res.status, hasWeekly: !!data?.weekly_summary, hasMacro: !!data?.macro_snapshots };
    } catch (e) {
      return { error: e.message };
    }
  }, apiResult.first.year, apiResult.first.week);
  console.log('In-browser fetch week data:', JSON.stringify(weekRes));
}

// Check the component state
const state = await page.evaluate(() => {
  const body = document.body.textContent || '';
  return {
    hasLoading: body.includes('Cargando analisis'),
    hasSEMANA: body.includes('SEMANA'),
    hasIndicadores: body.includes('Indicadores'),
    pageHeight: document.body.scrollHeight,
    h2Count: document.querySelectorAll('h2').length,
  };
});
console.log('Page state:', JSON.stringify(state));

await page.screenshot({ path: 'tests/e2e/screenshots/final-professional/debug2.png' });
await browser.close();
