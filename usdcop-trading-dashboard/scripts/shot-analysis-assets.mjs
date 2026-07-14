// Focused proof: login as admin, screenshot /analysis for Gold & BTC (macro charts + technical).
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const OUT = 'shots/analysis-parity';
mkdirSync(OUT, { recursive: true });

async function solveCaptcha(page) {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) {
      const ans = m[2] === '+' ? Number(m[1]) + Number(m[3]) : Number(m[1]) * Number(m[3]);
      await page.fill('input[placeholder="respuesta"]', String(ans));
    }
  } catch { /* no captcha */ }
}

const page = await (await chromium.launch()).newPage({ viewport: { width: 1600, height: 2200 } });
await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(2500);
await page.fill('input[name="username"], input[type="text"], input[name="email"]', 'admin');
await page.fill('input[type="password"]', 'Admin2026!');
await solveCaptcha(page);
await page.click('button[type="submit"]');
await page.waitForTimeout(6000);
console.log('logged in ->', page.url());

await page.goto(`${BASE}/analysis`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(4000);

for (const [asset, rx] of [['xauusd', /Oro|Gold|XAU/i], ['btcusdt', /Bitcoin|BTC/i]]) {
  // Click the asset selector chip (segmented control — only chips carry aria-pressed).
  await page.locator('button[aria-pressed]', { hasText: rx }).first().click({ timeout: 15000 });
  await page.waitForTimeout(8000); // index + weekly view refetch + recharts render
  const charts = await page.locator('.recharts-responsive-container').count();
  const tech = await page.getByText(/Análisis técnico|Technical analysis/).count();
  const macro = await page.getByText(/Indicadores macro|Macro indicators/).count();
  const scen = await page.getByText(/Escenarios de trading|Trading scenarios/).count();
  await page.screenshot({ path: `${OUT}/${asset}-analysis.png`, fullPage: true });
  console.log(`${asset}: recharts=${charts} technicalCard=${tech} macroSection=${macro} scenarios=${scen} -> ${OUT}/${asset}-analysis.png`);
}
await page.context().browser().close();
