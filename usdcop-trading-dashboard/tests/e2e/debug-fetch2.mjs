import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

// Log ALL API responses
page.on('response', async (response) => {
  const url = response.url();
  if (url.includes('/api/')) {
    let body = '';
    try { body = (await response.text()).slice(0, 300); } catch {}
    console.log(`[${response.status()}] ${url.replace('http://localhost:3000', '')} -> ${body.slice(0, 100)}`);
  }
});

await page.goto('http://localhost:3000/analysis', { waitUntil: 'load', timeout: 30000 });
console.log('Page loaded');

// Wait for requests to complete
for (let i = 0; i < 20; i++) {
  await page.waitForTimeout(1000);
  const state = await page.evaluate(() => ({
    loading: document.body.textContent?.includes('Cargando') || false,
    h2: document.querySelectorAll('h2').length,
    height: document.body.scrollHeight,
  }));
  if (i % 5 === 0 || !state.loading) {
    console.log(`  ${i}s: loading=${state.loading}, h2=${state.h2}, height=${state.height}`);
  }
  if (!state.loading && state.h2 > 0) {
    console.log('Content loaded!');
    break;
  }
}

// Try in-browser fetch
const weekResult = await page.evaluate(async () => {
  try {
    const res = await fetch('/api/analysis/week/2026/11');
    return { status: res.status, ok: res.ok, bodyLen: (await res.text()).length };
  } catch (e) {
    return { error: e.message };
  }
});
console.log('Direct fetch /api/analysis/week/2026/11:', JSON.stringify(weekResult));

await page.screenshot({ path: 'tests/e2e/screenshots/final-professional/debug3.png' });
await browser.close();
