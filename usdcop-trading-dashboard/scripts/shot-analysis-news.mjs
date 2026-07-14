// Proof: /analysis (USD/COP, W27) renders the news-intelligence + bias cards
// that were previously empty. Logs in, opens Analysis, ensures W27, screenshots.
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const OUT = 'shots/analysis-news';
mkdirSync(OUT, { recursive: true });

async function solveCaptcha(page) {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) await page.fill('input[placeholder="respuesta"]', String(m[2] === '+' ? +m[1] + +m[3] : +m[1] * +m[3]));
  } catch { /* no captcha */ }
}

const page = await (await chromium.launch()).newPage({ viewport: { width: 1600, height: 2600 } });
await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(2500);
await page.fill('input[name="username"], input[type="text"], input[name="email"]', 'admin');
await page.fill('input[type="password"]', 'Admin2026!');
await solveCaptcha(page);
await page.click('button[type="submit"]');
await page.waitForTimeout(6000);

await page.goto(`${BASE}/analysis`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(6000);

// Ensure USD/COP asset chip (aria-pressed selector chips only)
await page.locator('button[aria-pressed]', { hasText: /COP|Peso/i }).first().click({ timeout: 8000 }).catch(() => {});
await page.waitForTimeout(2500);

// Try to select week 27 via the WeekSelector if a control exposes it
await page.getByText(/W27|Semana 27|27/).first().click({ timeout: 4000 }).catch(() => {});
await page.waitForTimeout(3500);

const newsCard = await page.getByText(/Clusters de noticias|News clusters/i).count();
const biasCard = await page.getByText(/Sesgo mediático|Media bias/i).count();
const clusterLabels = await page.getByText(/General|Political|Monetary|Commodities/i).count();
await page.screenshot({ path: `${OUT}/analysis-usdcop.png`, fullPage: true });
console.log(`news-cluster card=${newsCard}  bias card=${biasCard}  cluster-labels=${clusterLabels}`);
await page.context().browser().close();
