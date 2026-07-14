// Verify the analysis chatbot returns a REAL LLM reply (not the no-key placeholder).
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
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(2500);
await page.fill('input[name="username"], input[type="text"], input[name="email"]', 'admin');
await page.fill('input[type="password"]', 'Admin2026!');
await solveCaptcha(page);
await page.click('button[type="submit"]');
await page.waitForTimeout(6000);

await page.goto(`${BASE}/analysis`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(6000);

// Directly probe the API from the authenticated browser context (bypasses UI timing).
const result = await page.evaluate(async () => {
  const r = await fetch('/api/analysis/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: '¿Cuál es el resumen de la semana en una frase?', session_id: 'qa_probe', asset: 'usdcop', year: 2026, week: 27 }),
  });
  const d = await r.json();
  return { status: r.status, reply: (d.reply || d.error || '').slice(0, 400), tokens: d.tokens_used };
});
console.log('HTTP', result.status, '| tokens:', result.tokens);
console.log('REPLY:', result.reply);
console.log(result.reply.includes('Chat sin LLM configurado') ? '>>> PLACEHOLDER (no LLM)' : '>>> REAL LLM REPLY');
await browser.close();
