#!/usr/bin/env node
/**
 * HubView preview fix (operator bug: "veo todo lo del admin, debería ver solo lo de mi rol").
 * As a real admin, activating "Ver como <rol>" must downgrade the HUB CARDS + kicker to that
 * role's surface — not keep showing the admin cards. Verifies the gm-view-as-role cookie is
 * honored by HubView (effectiveRole/canSee).
 *
 *   node scripts/hub-preview-qa.mjs
 */
import { chromium } from '@playwright/test';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
let pass = 0, fail = 0;
const check = (name, ok, detail = '') => {
  if (ok) pass++; else fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
};

async function solveCaptcha(page) {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) await page.fill('input[placeholder="respuesta"]', String(m[2] === '+' ? +m[1] + +m[3] : +m[1] * +m[3]));
  } catch { /* none */ }
}

const b = await chromium.launch();
const page = await b.newPage();
await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(2500);
await page.fill('input[name="username"], input[type="text"], input[name="email"]', 'admin');
await page.fill('input[type="password"]', 'Admin2026!');
await solveCaptcha(page);
await page.click('button[type="submit"]');
await page.waitForTimeout(6000);

// Baseline: admin hub shows the Admin card + "Admin" kicker.
await page.goto(`${BASE}/hub`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(3000);
const adminCardBefore = await page.locator('text=Consola de administración').count();
const kickerBefore = await page.locator('text=/Bienvenido ·/').first().textContent().catch(() => '');
check('admin hub shows Admin card', adminCardBefore > 0, `count=${adminCardBefore}`);
check('admin hub kicker = Admin', /Admin/.test(kickerBefore || ''), `"${(kickerBefore || '').trim()}"`);

// Activate "Ver como subscriber" (server sets both gm-view-as cookies).
const imp = await page.evaluate(async () => {
  const r = await fetch('/api/admin/impersonate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role: 'subscriber', motivo: 'hub preview qa' }),
  });
  return r.status;
});
check('impersonate subscriber ok', imp === 200, `status=${imp}`);

// Reload hub: cards + kicker must downgrade to subscriber.
await page.goto(`${BASE}/hub`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(3000);
const adminCardAfter = await page.locator('text=Consola de administración').count();
const kickerAfter = await page.locator('text=/Bienvenido ·/').first().textContent().catch(() => '');
check('preview hides Admin card', adminCardAfter === 0, `count=${adminCardAfter}`);
check('preview kicker = Suscriptor', /Suscriptor/.test(kickerAfter || ''), `"${(kickerAfter || '').trim()}"`);

// Exit preview → admin surface restored.
await page.evaluate(async () => { await fetch('/api/admin/impersonate', { method: 'DELETE' }); });
await page.goto(`${BASE}/hub`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(3000);
const adminCardRestored = await page.locator('text=Consola de administración').count();
check('exit preview restores Admin card', adminCardRestored > 0, `count=${adminCardRestored}`);

console.log(`\nHUB PREVIEW: ${pass} PASS, ${fail} FAIL`);
await b.close();
process.exit(fail === 0 ? 0 : 1);
