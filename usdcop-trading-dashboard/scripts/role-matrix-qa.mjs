#!/usr/bin/env node
/**
 * FINAL role×surface truth-table validation (CTR-RBAC-001).
 *
 * The RBAC contract IS a matrix — so the final test asserts it cell by cell with real
 * sessions: 5 principals (anon, free, subscriber, developer, admin) × the critical
 * surfaces (research, signals, approvals, admin, per-asset forecasting, data delays).
 * Complements: rbac:check (route coverage), rbac:test (contract invariants),
 * functional-qa (flows), registration-qa (journey), promotion-e2e (Vote-2).
 *
 *   node scripts/role-matrix-qa.mjs
 */
import { chromium } from '@playwright/test';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const USERS = {
  free: ['free@test.com', 'Test2026!'],
  subscriber: ['pro@test.com', 'Test2026!'],
  developer: ['dev@test.com', 'Test2026!'],
  admin: ['admin', 'Admin2026!'],
};

// Each cell: [method, url, {anon, free, subscriber, developer, admin}] — expected[] = allowed statuses.
const CELLS = [
  ['GET', '/api/registry', { anon: [401], free: [403], subscriber: [403], developer: [200], admin: [200] }],
  ['GET', '/api/production/strategies', { anon: [401], free: [403], subscriber: [200], developer: [200], admin: [200] }],
  ['GET', '/api/production/live', { anon: [401], free: [403], subscriber: [200], developer: [200], admin: [200] }],
  // Vote-2: admin-only at the edge. Admin probes a nonexistent sid → 404 (never 401/403).
  ['POST', '/api/production/approve', { anon: [401], free: [403], subscriber: [403], developer: [403], admin: [404] },
    { action: 'APPROVE', strategy_id: 'matrix_probe_nonexistent' }],
  ['POST', '/api/production/deploy', { anon: [401], free: [403], subscriber: [403], developer: [403], admin: [404, 409] },
    { strategy_id: 'matrix_probe_nonexistent' }],
  ['GET', '/api/admin/users', { anon: [401], free: [403], subscriber: [403], developer: [403], admin: [200, 502] }],
  // Per-asset forecasting: internals see all; clients only their plan's assets.
  ['GET', '/api/forecasting/xauusd/weekly_inference_2026.json', { anon: [401], free: [403], subscriber: [403], developer: [200], admin: [200] }],
  ['GET', '/api/forecasting/btcusdt/weekly_inference_2026.json', { anon: [401], free: [403], subscriber: [403], developer: [200], admin: [200] }],
  ['GET', '/api/forecasting/bi_dashboard_unified.csv', { anon: [401], free: [200], subscriber: [200], developer: [200], admin: [200] }],
  // Current-week PNG: delayed plans blocked (403), internals pass through (404: no file yet).
  // subscriber plan `signals` is AL DÍA (delay 0) → passes the gate → 404 (file absent); only free (T-1) gets 403.
  ['GET', '/forecasting/forward_ridge_2099_W01.png', { anon: [401], free: [403], subscriber: [404], developer: [404], admin: [404] }],
  ['GET', '/api/billing/me', { anon: [401], free: [200], subscriber: [200], developer: [200], admin: [200] }],
  ['GET', '/api/captcha', { anon: [200], free: [200], subscriber: [200], developer: [200], admin: [200] }],
];

async function solveCaptcha(page) {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) await page.fill('input[placeholder="respuesta"]',
      String(m[2] === '+' ? Number(m[1]) + Number(m[3]) : Number(m[1]) * Number(m[3])));
  } catch { /* no captcha rendered */ }
}

const b = await chromium.launch();
let pass = 0, fail = 0;
const failures = [];

// Nav/hub visibility truth-table (DOM). Sidebar items are <button> (router.push),
// so match by label inside the sections <nav>. Backtest→research:read,
// Admin→admin:all. Locks in: a subscriber never sees Admin/Backtest; admin
// (superset) sees both; developer sees Backtest but not Admin.
const NAV_EXPECT = {
  free:       { Backtest: false, Admin: false },
  subscriber: { Backtest: false, Admin: false },
  developer:  { Backtest: true,  Admin: false },
  admin:      { Backtest: true,  Admin: true },
};

async function probeNav(role, page) {
  await page.goto(`${BASE}/hub`, { waitUntil: 'commit', timeout: 90000 }).catch(() => {});
  await page.waitForTimeout(3000);
  const want = NAV_EXPECT[role];
  for (const [label, shouldShow] of Object.entries(want)) {
    const count = await page.locator('nav button', { hasText: new RegExp(`^${label}$`) }).count().catch(() => -1);
    const shown = count > 0;
    const ok = shown === shouldShow;
    if (ok) pass++; else { fail++; failures.push(`${role} nav ${label}: ${shown ? 'visible' : 'hidden'}, want ${shouldShow ? 'visible' : 'hidden'}`); }
  }
}

async function probeAll(role, page) {
  for (const [method, url, expects, body] of CELLS) {
    const want = expects[role];
    const got = await page.evaluate(async ({ m, u, bd }) => {
      const r = await fetch(u, m === 'POST'
        ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(bd ?? {}) }
        : undefined);
      return r.status;
    }, { m: method, u: url, bd: body });
    const ok = want.includes(got);
    if (ok) pass++; else { fail++; failures.push(`${role} ${method} ${url}: got ${got}, want ${want}`); }
  }
}

// anon
{
  const ctx = await b.newContext(); const page = await ctx.newPage();
  await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
  await page.waitForTimeout(2000);
  await probeAll('anon', page);
  // page-level: /admin must not be reachable
  const r = await page.request.get(`${BASE}/admin`, { maxRedirects: 0 }).catch(() => null);
  if (r && [307, 302, 401].includes(r.status())) pass++; else { fail++; failures.push(`anon /admin page: ${r?.status()}`); }
  await ctx.close();
}
// authenticated roles
for (const [role, [u, p]] of Object.entries(USERS)) {
  const ctx = await b.newContext(); const page = await ctx.newPage();
  await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
  await page.waitForTimeout(2500);
  await page.fill('input[name="username"], input[type="text"], input[name="email"]', u);
  await page.fill('input[type="password"]', p);
  await solveCaptcha(page);
  await page.click('button[type="submit"]');
  await page.waitForTimeout(6000);
  if (page.url().includes('/login')) { fail++; failures.push(`${role}: login failed`); await ctx.close(); continue; }
  await probeAll(role, page);
  await probeNav(role, page);
  await ctx.close();
}
await b.close();

console.log(`\nROLE MATRIX: ${pass} PASS, ${fail} FAIL (${CELLS.length}×5 cells + page checks)`);
for (const f of failures) console.log('  FAIL:', f);
process.exit(fail === 0 ? 0 : 1);
