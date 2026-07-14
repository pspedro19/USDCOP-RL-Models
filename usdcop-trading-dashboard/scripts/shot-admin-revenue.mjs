// Proof: admin console Resumen + Ingresos tabs render REAL revenue (not "Fase 6").
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const OUT = 'shots/admin-revenue';
mkdirSync(OUT, { recursive: true });

async function solveCaptcha(page) {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) await page.fill('input[placeholder="respuesta"]', String(m[2] === '+' ? +m[1] + +m[3] : +m[1] * +m[3]));
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

// Resumen (default admin tab)
await page.goto(`${BASE}/admin`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(5000);
const mrrTile = await page.getByText(/MRR/).count();
const subsTile = await page.getByText(/Suscriptores/).count();
const churnTile = await page.getByText(/Churn/).count();
await page.screenshot({ path: `${OUT}/resumen.png`, fullPage: true });
console.log(`resumen: MRR=${mrrTile} Suscriptores=${subsTile} Churn=${churnTile}`);

// Ingresos tab
await page.getByTestId('admin-tab-ingresos').click({ timeout: 15000 }).catch(() => {});
await page.waitForTimeout(4000);
const stillFase6 = await page.getByText(/Fase 6/).count();
const hasCop = await page.getByText(/\$\s?0|\$[0-9]/).count();
await page.screenshot({ path: `${OUT}/ingresos.png`, fullPage: true });
console.log(`ingresos: residual "Fase 6"=${stillFase6} COP-values=${hasCop}`);
await page.context().browser().close();
