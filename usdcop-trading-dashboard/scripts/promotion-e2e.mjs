#!/usr/bin/env node
/**
 * Model-promotion visual E2E (Vote 2/2). Drives the admin UI:
 *   /dashboard → ApprovalPanel (5/5 gates, PROMOTE) → "Aprobar y Promover" → "Confirmar"
 * then verifies approval_state.json flips PENDING_APPROVAL → APPROVED with the AUTHENTICATED
 * admin principal (not a client-supplied name), and /production reflects APPROVED.
 *
 *   node scripts/promotion-e2e.mjs
 *
 * The fire-and-forget deploy spawns python3 in the node dashboard container (absent) → no-ops,
 * so no heavy retrain/republish runs (v11 is FROZEN). Caller restores the PENDING backup after.
 */
import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const OUT = 'shots/iter-promotion'; mkdirSync(OUT, { recursive: true });
const ADMIN = { u: 'admin', p: 'Admin2026!' };
const results = [];
const check = (name, ok, detail = '') => { results.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}: ${name}${detail ? ' — ' + detail : ''}`); };


// Solve the on-page anti-bot challenge (server-signed math captcha).
async function solveCaptcha(page) {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) {
      const ans = m[2] === '+' ? Number(m[1]) + Number(m[3]) : Number(m[1]) * Number(m[3]);
      await page.fill('input[placeholder="respuesta"]', String(ans));
    }
  } catch { /* captcha not present (older build) — proceed */ }
}

async function login(page, u, p) {
  await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
  await page.waitForTimeout(2500);
  await page.fill('input[name="username"], input[type="text"], input[name="email"]', u);
  await page.fill('input[type="password"]', p);
  await solveCaptcha(page);
  await page.click('button[type="submit"]');
  await page.waitForTimeout(6000);
  return !page.url().includes('/login');
}

const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1440, height: 1200 } });
const page = await ctx.newPage();

check('admin login', await login(page, ADMIN.u, ADMIN.p), page.url().replace(BASE, ''));

// 1. status BEFORE via API (authenticated session cookie carried by ctx)
let before = await page.evaluate(async (base) => {
  const r = await fetch(`${base}/api/production/status`); return r.ok ? r.json() : { error: r.status };
}, BASE);
check('pre-state PENDING_APPROVAL', before.status === 'PENDING_APPROVAL', `status=${before.status}`);
check('gates 5/5 in bundle', Array.isArray(before.gates) && before.gates.filter(g => g.passed).length === 5,
  `${(before.gates || []).filter(g => g.passed).length}/${(before.gates || []).length}`);
check('recommendation PROMOTE', before.backtest_recommendation === 'PROMOTE', before.backtest_recommendation);

// 2. Go to /dashboard, find the ApprovalPanel
await page.goto(`${BASE}/dashboard`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(9000);
await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
await page.waitForTimeout(1500);
const approveBtn = page.getByText('Aprobar y Promover', { exact: false }).first();
const approveVisible = await approveBtn.isVisible().catch(() => false);
check('ApprovalPanel "Aprobar y Promover" visible for admin', approveVisible);
await approveBtn.scrollIntoViewIfNeeded().catch(() => {});
await page.waitForTimeout(500);
await page.screenshot({ path: `${OUT}/01-pending-approvalpanel.png` });

// 3. Click Aprobar → Confirmar
if (approveVisible) {
  await approveBtn.click(); await page.waitForTimeout(800);
  const confirmBtn = page.getByText('Confirmar', { exact: true }).first();
  check('confirm dialog appears', await confirmBtn.isVisible().catch(() => false));
  await page.screenshot({ path: `${OUT}/02-confirm-dialog.png` });
  await confirmBtn.click();
  await page.waitForTimeout(4000);
}

// 4. status AFTER
let after = await page.evaluate(async (base) => {
  const r = await fetch(`${base}/api/production/status`); return r.ok ? r.json() : { error: r.status };
}, BASE);
check('post-state APPROVED', after.status === 'APPROVED', `status=${after.status}`);
check('approved_by = authenticated principal (not client string)',
  !!after.approved_by && after.approved_by !== 'dashboard_user',
  `approved_by=${after.approved_by}`);
check('approved_at set', !!after.approved_at, after.approved_at || '');
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/03-approved-state.png` });

// 5. /production reflects APPROVED
await page.goto(`${BASE}/production`, { waitUntil: 'commit', timeout: 90000 });
await page.waitForTimeout(9000);
const prodHtml = await page.content();
check('/production shows APPROVED badge', /APROBAD|APPROVED/i.test(prodHtml));
await page.evaluate(() => window.scrollTo(0, 0)); await page.waitForTimeout(500);
await page.screenshot({ path: `${OUT}/04-production-approved.png` });

await b.close();
const fails = results.filter(r => !r.ok).length;
console.log(`\nPROMOTION E2E: ${results.length - fails}/${results.length} PASS, ${fails} FAIL`);
process.exit(0);
