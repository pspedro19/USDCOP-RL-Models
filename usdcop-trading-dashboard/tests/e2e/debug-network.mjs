import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

// Intercept network requests
page.on('response', async (response) => {
  const url = response.url();
  if (url.includes('/api/analysis/') || url.includes('/data/analysis/')) {
    const status = response.status();
    let body = '';
    try { body = (await response.text()).slice(0, 200); } catch {}
    console.log(`[${status}] ${url.replace('http://localhost:3000', '')}`);
    if (status >= 400) console.log(`  Body: ${body}`);
  }
});

page.on('console', msg => {
  const text = msg.text();
  if (text.includes('Error') || text.includes('error') || text.includes('fail')) {
    console.log(`[CONSOLE ${msg.type()}] ${text.slice(0, 200)}`);
  }
});

await page.goto('http://localhost:3000/analysis', { waitUntil: 'networkidle', timeout: 45000 });
console.log('Page loaded (networkidle)');

await page.waitForTimeout(8000);
console.log('After 8s extra wait');

const bodyLen = await page.evaluate(() => document.body.textContent?.length || 0);
const hasLoading = await page.evaluate(() => document.body.textContent?.includes('Cargando') || false);
const pageHeight = await page.evaluate(() => document.body.scrollHeight);

console.log(`Body length: ${bodyLen}, hasLoading: ${hasLoading}, pageHeight: ${pageHeight}`);

await page.screenshot({ path: 'tests/e2e/screenshots/final-professional/debug.png' });
await browser.close();
