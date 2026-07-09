#!/usr/bin/env node
/**
 * Registration + admin-approval E2E (QA-100 §F1++) — TWO concurrent browser contexts.
 *
 *   Session A (anon)  : /login → "Solicitar acceso" → /register → submit → "pendiente"
 *   Session B (admin) : login → /admin → Cola de aprobación → Aprobar
 *   MailHog           : capture the temporary password emailed on approval
 *   Session A (cont.) : temp-pw login → FORCED /reset-password → clean /login → role hub
 *   Negatives         : approve-as-non-admin 403 · double-approve idempotent · pre-approval login blocked
 *
 * Best-practice product flow: nothing grants app access until the temp password is
 * consumed. Screenshots + per-context console logs are written to <outdir>.
 *
 * Run:  node scripts/registration-qa.mjs <outdir>
 * Env:  QA_BASE (http://localhost:5000) · QA_SB (http://localhost:8085) · QA_MAILHOG (http://localhost:8025)
 *       ADMIN_USER (admin) · ADMIN_PW (Admin2026!)  — the bootstrap admin (idempotent at SB startup)
 */
import { execSync } from 'node:child_process';
import { chromium } from '@playwright/test';

// The per-IP register throttle (audit A8-03) locks repeated harness runs (5/15min).
// Clear OUR OWN lockout bucket before starting — same pattern as the login-lockout reset.
function clearRegisterThrottle() {
  try {
    const pw = execSync('docker exec usdcop-redis sh -c "echo $REDIS_PASSWORD"').toString().trim()
      || execSync('docker exec usdcop-signalbridge sh -c "echo $REDIS_PASSWORD"').toString().trim();
    const keys = execSync(`docker exec usdcop-redis redis-cli -a "${pw}" --no-auth-warning --scan --pattern "auth:*register*"`).toString().split(String.fromCharCode(10)).map((s) => s.trim()).filter(Boolean);
    for (const k of keys) execSync(`docker exec usdcop-redis redis-cli -a "${pw}" --no-auth-warning del "${k}"`);
    console.log(`register throttle cleared (${keys.length} keys)`);
  } catch (e) { console.log('throttle clear skipped:', String(e).slice(0, 80)); }
}
clearRegisterThrottle();
import { mkdirSync, writeFileSync } from 'node:fs';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const SB = process.env.QA_SB ?? 'http://localhost:8085';
const MAILHOG = (process.env.QA_MAILHOG ?? 'http://localhost:8025').replace(/\/$/, '');
const ADMIN_USER = process.env.ADMIN_USER ?? 'admin';           // maps to admin@trading.usdcop.com
const ADMIN_PW = process.env.ADMIN_PW ?? 'Admin2026!';
const OUT = process.argv[2] ?? 'shots/reg';
mkdirSync(OUT, { recursive: true });

const R = [];
const ok = (name, cond, detail = '') => {
  R.push({ name, ok: !!cond, detail: String(detail).slice(0, 200) });
  console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${detail ? ' — ' + String(detail).slice(0, 140) : ''}`);
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const api = async (url, opts = {}) => {
  const r = await fetch(url, { ...opts, headers: { 'Content-Type': 'application/json', ...(opts.headers ?? {}) } });
  let body = null; try { body = await r.json(); } catch { /* non-json */ }
  return { status: r.status, body };
};

async function sbToken(email, password) {
  const r = await api(`${SB}/api/auth/login`, { method: 'POST', body: JSON.stringify({ email, password }) });
  return r.body?.access_token ?? null;
}

// Robust temp-password extraction from the approval email (HTML or text bodies).
async function captureTempPw(email, tries = 20) {
  const pats = [
    /[Tt]emporary password[:\s]*<[^>]*>\s*([A-Za-z0-9!@#$%^&*._-]{8,})/,
    /[Tt]emporary password[:\s]+([A-Za-z0-9!@#$%^&*._-]{8,})/,
    /<code[^>]*>([A-Za-z0-9!@#$%^&*._-]{8,})<\/code>/,
    /<b[^>]*>([A-Za-z0-9!@#$%^&*._-]{10,})<\/b>/,
  ];
  const unescape = (s) => s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'");
  for (let i = 0; i < tries; i++) {
    const r = await fetch(`${MAILHOG}/api/v2/messages?limit=50`).then((x) => x.json()).catch(() => null);
    for (const m of r?.items ?? []) {
      const to = (m.To ?? []).map((h) => `${h.Mailbox}@${h.Domain}`);
      const subj = (m.Content?.Headers?.Subject ?? [''])[0] ?? '';
      if (to.includes(email) && /approv/i.test(subj)) {
        const body = unescape(m.Content?.Body ?? '');
        for (const rx of pats) { const mm = rx.exec(body); if (mm) return mm[1]; }
        return null; // email found but no pw parsed
      }
    }
    await sleep(1000);
  }
  return null;
}

function attachConsole(page, sink, tag) {
  page.on('console', (m) => { if (m.type() === 'error') sink.push({ ctx: tag, url: page.url().replace(BASE, ''), text: m.text().slice(0, 200) }); });
  page.on('pageerror', (e) => sink.push({ ctx: tag, url: page.url().replace(BASE, ''), text: 'pageerror: ' + String(e).slice(0, 200) }));
}


// Solve the on-page anti-bot challenge (server-signed math captcha).
async function solveCaptcha(page, inputSel = 'input[placeholder="respuesta"]') {
  try {
    const q = await page.locator('text=/¿Cuánto es/').first().textContent({ timeout: 4000 });
    const m = q && q.match(/(\d+)\s*([+×])\s*(\d+)/);
    if (m) {
      const ans = m[2] === '+' ? Number(m[1]) + Number(m[3]) : Number(m[1]) * Number(m[3]);
      await page.fill(inputSel, String(ans));
    }
  } catch { /* captcha not present */ }
}

async function uiLogin(page, user, pw) {
  await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
  await page.waitForTimeout(2500);
  await page.fill('input[name="username"], input[type="text"], input[name="email"]', user);
  await page.fill('input[type="password"]', pw);
  await solveCaptcha(page);
  await page.click('button[type="submit"]');
  await page.waitForTimeout(6000);
}

const consoleLog = [];
const browser = await chromium.launch();
const email = `reguser_${Date.now()}@example.com`;
const NAME = 'QA Applicant';
const APPLICANT_PW = 'Applicant1!';
const NEW_PW = 'BrandNew!Pw24';
let userId = null;

// ── P0: bootstrap admin reachable (idempotent at SB startup) ────────────────
{
  const tok = await sbToken('admin@trading.usdcop.com', ADMIN_PW);
  ok('P0 bootstrap admin can authenticate (approver exists)', !!tok);
}

// ════════════ SESSION A — registration UI (anon) ════════════
const ctxA = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const pageA = await ctxA.newPage();
attachConsole(pageA, consoleLog, 'A');
{
  await pageA.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
  await pageA.waitForTimeout(2500);
  await pageA.screenshot({ path: `${OUT}/A1-login.png`, fullPage: true });
  const link = pageA.locator('[data-testid="login-to-register"]');
  ok('A1 login has "Solicitar acceso" link', await link.count() > 0);
  await link.first().click().catch(() => {});
  await pageA.waitForTimeout(1500);
  ok('A2 navigated to /register', pageA.url().includes('/register'), pageA.url().replace(BASE, ''));

  await pageA.fill('[data-testid="reg-name"]', NAME);
  await pageA.fill('[data-testid="reg-email"]', email);
  await pageA.fill('[data-testid="reg-password"]', APPLICANT_PW);
  await pageA.fill('[data-testid="reg-confirm"]', APPLICANT_PW);
  await pageA.waitForTimeout(500);
  await pageA.screenshot({ path: `${OUT}/A3-register-filled.png`, fullPage: true });

  await solveCaptcha(pageA, '[data-testid="reg-captcha"]');
  const submit = pageA.locator('[data-testid="reg-submit"]');
  ok('A3 submit enabled with valid form', await submit.isEnabled());
  await submit.click();
  const pending = pageA.locator('[data-testid="register-pending"]');
  await pending.waitFor({ state: 'visible', timeout: 15000 }).catch(() => {});
  ok('A4 registration → "pendiente de aprobación" panel', await pending.count() > 0);
  await pageA.screenshot({ path: `${OUT}/A4-register-pending.png`, fullPage: true });
}

// pre-approval negatives (API-level, deterministic)
{
  const dup = await api(`${SB}/api/auth/register`, { method: 'POST', body: JSON.stringify({ email, password: APPLICANT_PW, name: NAME }) });
  ok('N1 duplicate registration rejected (409)', dup.status === 409, `status=${dup.status}`);
  const early = await api(`${SB}/api/auth/login`, { method: 'POST', body: JSON.stringify({ email, password: APPLICANT_PW }) });
  ok('N2 login before approval is blocked (not 200)', early.status !== 200, `status=${early.status}`);
}

// ════════════ SESSION B — admin approval UI ════════════
const ctxB = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const pageB = await ctxB.newPage();
attachConsole(pageB, consoleLog, 'B');
{
  await uiLogin(pageB, ADMIN_USER, ADMIN_PW);
  ok('B1 admin logged in (left /login)', !pageB.url().includes('/login'), pageB.url().replace(BASE, ''));
  await pageB.goto(`${BASE}/admin`, { waitUntil: 'commit', timeout: 90000 });
  await pageB.waitForTimeout(4000);
  // Console v2: the queue lives under the "Registros" tab (Overview is the default).
  await pageB.locator('[data-testid="admin-tab-registros"]').click().catch(() => {});
  await pageB.waitForTimeout(1000);
  const queue = pageB.locator('[data-testid="approval-queue"]');
  ok('B2 admin sees approval queue', await queue.count() > 0);
  await pageB.screenshot({ path: `${OUT}/B2-admin-queue.png`, fullPage: true });

  const row = pageB.locator(`[data-testid="pending-row-${email}"]`);
  await row.waitFor({ state: 'visible', timeout: 8000 }).catch(() => {});
  ok('B3 new applicant appears PENDING in queue', await row.count() > 0, email);

  // Resolve the user id for the API negatives (via admin API).
  const atok = await sbToken('admin@trading.usdcop.com', ADMIN_PW);
  const list = await api(`${SB}/api/admin/users?status=pending`, { headers: { Authorization: `Bearer ${atok}` } });
  userId = (list.body ?? []).find((u) => u.email === email)?.id ?? null;

  // N3: approve as non-admin (free) → 403
  const freeTok = await sbToken('free@test.com', 'Test2026!');
  if (freeTok && userId) {
    const denied = await api(`${SB}/api/admin/users/${userId}/approve`, { method: 'POST', headers: { Authorization: `Bearer ${freeTok}` } });
    ok('N3 approve as non-admin DENIED (403)', denied.status === 403, `status=${denied.status}`);
  } else ok('N3 approve as non-admin DENIED (403)', false, 'missing free token or user id');

  // Approve through the real admin UI button.
  const approveBtn = pageB.locator(`[data-testid="approve-${email}"]`);
  ok('B4 Aprobar button present', await approveBtn.count() > 0);
  await approveBtn.first().click().catch(() => {});
  const flash = pageB.locator('[data-testid="admin-flash"]');
  await flash.waitFor({ state: 'visible', timeout: 15000 }).catch(() => {});
  const flashText = (await flash.textContent().catch(() => '')) ?? '';
  ok('B5 approval flash shows success + email sent', /Aprobado/.test(flashText) && /correo enviado/.test(flashText), flashText.slice(0, 120));
  await pageB.waitForTimeout(1500);
  ok('B6 approved row removed from queue', await pageB.locator(`[data-testid="pending-row-${email}"]`).count() === 0);
  await pageB.screenshot({ path: `${OUT}/B6-admin-approved.png`, fullPage: true });

  // N4: double approval is idempotent / clean (no 500)
  if (atok && userId) {
    const again = await api(`${SB}/api/admin/users/${userId}/approve`, { method: 'POST', headers: { Authorization: `Bearer ${atok}` } });
    ok('N4 double-approve does not 500', again.status < 500, `status=${again.status}`);
  }
}

// ════════════ MailHog — capture temp password ════════════
const tempPw = await captureTempPw(email);
ok('M1 temp password captured from approval email', !!tempPw, tempPw ? `len=${tempPw.length}` : 'not found');

// ════════════ SESSION A — temp login → forced reset → clean login → hub ════════════
{
  const pageC = await ctxA.newPage(); // reuse ctxA (anon origin); fresh page
  attachConsole(pageC, consoleLog, 'A2');
  if (tempPw) {
    await pageC.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
    await pageC.waitForTimeout(2500);
    await pageC.fill('input[name="username"], input[type="text"], input[name="email"]', email);
    await pageC.fill('input[type="password"]', tempPw);
    await solveCaptcha(pageC);
    await pageC.click('button[type="submit"]');
    await pageC.waitForTimeout(5000);
    ok('C1 temp-pw login forced to /reset-password', pageC.url().includes('/reset-password'), pageC.url().replace(BASE, ''));
    await pageC.screenshot({ path: `${OUT}/C1-forced-reset.png`, fullPage: true });

    // current temp pw is pre-filled from sessionStorage; set the new one.
    await pageC.fill('[data-testid="rst-new"]', NEW_PW);
    await pageC.fill('[data-testid="rst-confirm"]', NEW_PW);
    await pageC.waitForTimeout(400);
    await pageC.click('[data-testid="rst-submit"]');
    const done = pageC.locator('[data-testid="reset-done"]');
    await done.waitFor({ state: 'visible', timeout: 15000 }).catch(() => {});
    ok('C2 password reset succeeded', await done.count() > 0);
    await pageC.waitForTimeout(2500); // allow redirect to /login?reset=1
    ok('C3 redirected back to /login after reset', pageC.url().includes('/login'), pageC.url().replace(BASE, ''));

    // clean login with the new password → role hub
    await pageC.fill('input[name="username"], input[type="text"], input[name="email"]', email);
    await pageC.fill('input[type="password"]', NEW_PW);
    await solveCaptcha(pageC);
    await pageC.click('button[type="submit"]');
    await pageC.waitForTimeout(6000);
    ok('C4 clean login with new password reaches app (not /login, not /reset)',
      !pageC.url().includes('/login') && !pageC.url().includes('/reset-password'), pageC.url().replace(BASE, ''));
    await pageC.screenshot({ path: `${OUT}/C4-hub-after-reset.png`, fullPage: true });
  } else {
    ok('C1 temp-pw login forced to /reset-password', false, 'skipped: no temp pw');
  }
}

await browser.close();
writeFileSync(`${OUT}/registration-report.json`, JSON.stringify({ email, userId, results: R, console: consoleLog }, null, 1));
const fails = R.filter((x) => !x.ok).length;
console.log(`\nREGISTRATION E2E: ${R.length - fails}/${R.length} PASS, ${fails} FAIL, ${consoleLog.length} console errors -> ${OUT}/registration-report.json`);
process.exit(fails ? 1 : 0);
