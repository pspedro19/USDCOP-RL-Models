// Comprehensive verification of the wiring program:
// chatbot (real LLM), news bell, news theme selector, analysis synthesis/mtf/fx,
// admin Sistema observability, real prices (Pricing + /api/billing/prices).
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const OUT = 'shots/verify-program';
mkdirSync(OUT, { recursive: true });
const results = [];
const check = (name, ok, detail = '') => { results.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`); };

async function solveCaptcha(page) {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) await page.fill('input[placeholder="respuesta"]', String(m[2] === '+' ? +m[1] + +m[3] : +m[1] * +m[3]));
  } catch {}
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 2200 } });
await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(2500);
await page.fill('input[name="username"], input[type="text"], input[name="email"]', 'admin');
await page.fill('input[type="password"]', 'Admin2026!');
await solveCaptcha(page);
await page.click('button[type="submit"]');
await page.waitForTimeout(6000);

// 1) Public prices endpoint returns real COP
try {
  const prices = await page.evaluate(async () => (await fetch('/api/billing/prices')).json());
  const signals = prices?.data?.plans?.find((p) => p.plan === 'signals')?.price_month_cop;
  check('prices endpoint returns real COP', signals === 99000, `signals=${signals}`);
} catch (e) { check('prices endpoint returns real COP', false, String(e).slice(0, 80)); }

// 2) Chatbot returns a REAL LLM reply (not the no-key placeholder)
try {
  const chat = await page.evaluate(async () => {
    const r = await fetch('/api/analysis/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: '¿Resumen de la semana en una frase?', session_id: 'verify', asset: 'usdcop', year: 2026, week: 27 }) });
    const d = await r.json();
    return { status: r.status, provider: d.provider, tokens: d.tokens_used, reply: (d.reply || d.error || '').slice(0, 160) };
  });
  const real = chat.status === 200 && chat.provider && chat.provider !== 'none' && !/(sin LLM configurado)/i.test(chat.reply);
  check('chatbot real LLM reply', real, `provider=${chat.provider} tokens=${chat.tokens} · ${chat.reply.slice(0, 90)}`);
} catch (e) { check('chatbot real LLM reply', false, String(e).slice(0, 80)); }

// 3) Analysis page: news bell + theme selector + synthesis/mtf/fx
await page.goto(`${BASE}/analysis`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(6000);
check('news bell in topbar', (await page.locator('[data-testid="topbar-news"]').count()) > 0);
check('news theme selector', (await page.getByLabel(/Filtrar por tema|Filter by theme/i).count()) > 0);
check('news cluster card', (await page.getByText(/Clusters de noticias|News clusters/i).count()) > 0);
check('synthesis/mtf/fx cards', (await page.getByText(/Síntesis|Synthesis|Multi-?timeframe|Alineación|FX|Carry/i).count()) > 0);
await page.screenshot({ path: `${OUT}/analysis.png`, fullPage: true });

// 4) Admin Sistema observability
await page.goto(`${BASE}/admin`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(4000);
await page.getByTestId('admin-tab-sistema').click({ timeout: 12000 }).catch(async () => { await page.getByText(/Sistema/).first().click().catch(() => {}); });
await page.waitForTimeout(4000);
check('admin Sistema observability', (await page.getByText(/SLO|p95|Prometheus|Alertas|Pipeline|DAG|CPU|Memoria|targets/i).count()) > 0);
await page.screenshot({ path: `${OUT}/admin-sistema.png`, fullPage: true });

// 5) Pricing real prices
await page.goto(`${BASE}/pricing`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(4000);
check('pricing shows real COP', (await page.getByText(/99[.,]?000|299[.,]?000/).count()) > 0);
await page.screenshot({ path: `${OUT}/pricing.png`, fullPage: true });

console.log('\n=== SUMMARY ===');
console.log(`${results.filter((r) => r.ok).length}/${results.length} checks passed`);
await browser.close();
process.exit(results.every((r) => r.ok) ? 0 : 1);
