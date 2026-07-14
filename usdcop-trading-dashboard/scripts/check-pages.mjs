// Smoke: key pages render without an error boundary after the wiring changes.
import { chromium } from 'playwright';
const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
async function solveCaptcha(page) {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) await page.fill('input[placeholder="respuesta"]', String(m[2] === '+' ? +m[1] + +m[3] : +m[1] * +m[3]));
  } catch {}
}
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 1200 } });
await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(2500);
await page.fill('input[name="username"], input[type="text"]', 'admin');
await page.fill('input[type="password"]', 'Admin2026!');
await solveCaptcha(page);
await page.click('button[type="submit"]');
await page.waitForTimeout(6000);
const pages = ['/dashboard', '/production', '/forecasting', '/execution/dashboard', '/execution/exchanges', '/execution/settings', '/admin'];
let pass = 0;
for (const p of pages) {
  await page.goto(`${BASE}${p}`, { waitUntil: 'commit', timeout: 90000 }).catch(() => {});
  await page.waitForTimeout(3500);
  const err = await page.getByText(/Something went wrong|Application error|Unhandled|500 -/i).count();
  const hasContent = (await page.locator('main, [role="main"], .gm-content').count()) > 0;
  const ok = err === 0 && hasContent;
  if (ok) pass++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${p}${err ? ' (error boundary)' : ''}${!hasContent ? ' (no content)' : ''}`);
}
console.log(`\n${pass}/${pages.length} pages render clean`);
await browser.close();
process.exit(pass === pages.length ? 0 : 1);
