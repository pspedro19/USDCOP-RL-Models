// Proof: login as admin, screenshot SignalBridge exchanges (copy-trading + best-practices) + executions.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const OUT = 'shots/execution-fidelity';
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

const page = await (await chromium.launch()).newPage({ viewport: { width: 1600, height: 2400 } });
await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(2500);
await page.fill('input[name="username"], input[type="text"], input[name="email"]', 'admin');
await page.fill('input[type="password"]', 'Admin2026!');
await solveCaptcha(page);
await page.click('button[type="submit"]');
await page.waitForTimeout(6000);
console.log('logged in ->', page.url());

for (const [name, url] of [['exchanges', '/execution/exchanges'], ['executions', '/execution/executions']]) {
  await page.goto(`${BASE}${url}`, { waitUntil: 'commit', timeout: 90000 });
  await page.waitForTimeout(5000);
  const copy = await page.getByText(/Copy trading/i).count();
  const best = await page.getByText(/Mejores prácticas|Best practices/i).count();
  const pnl = await page.getByRole('columnheader', { name: /P&L/i }).count();
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
  console.log(`${name}: copyTrading=${copy} bestPractices=${best} pnlHeader=${pnl} -> ${OUT}/${name}.png`);
}
await page.context().browser().close();
