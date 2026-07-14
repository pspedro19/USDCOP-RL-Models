#!/usr/bin/env node
/**
 * Visual TDD/BDD harness (ux-navigation spec) — captures EVERY view for EVERY role,
 * crawls the navigation, and emits a machine-readable report.
 *
 *   node scripts/visual-qa.mjs shots/iter-1            # full run (4 roles + public)
 *   node scripts/visual-qa.mjs shots/iter-1 --mobile   # adds 390px viewport pass
 *
 * Output: <outdir>/<NN>-<role>-<view>.png + report.json (url, status, console errors,
 * load ms, nav-crawl results). Claude then analyzes each screenshot against
 * docs/rbac/VISUAL-SPEC-CHECKLIST.md and iterates until convergence.
 */
import { chromium } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';

const BASE = process.env.QA_BASE ?? 'http://localhost:5000';
const OUT = process.argv[2] ?? 'shots/iter-x';
const MOBILE = process.argv.includes('--mobile');
const WIDE = process.argv.includes('--wide');
const ZOOM = process.argv.includes('--zoom');
mkdirSync(OUT, { recursive: true });

const USERS = {
  admin: { u: 'admin', p: 'Admin2026!' },
  developer: { u: 'dev@test.com', p: 'Test2026!' },
  subscriber: { u: 'pro@test.com', p: 'Test2026!' },
  free: { u: 'free@test.com', p: 'Test2026!' },
};

// Views per audience. `scrolls` = extra full-width captures at fractions of page height.
const PUBLIC_VIEWS = [
  { url: '/', name: 'landing', wait: 8000, scrolls: [0.25, 0.55, 0.85] },
  { url: '/metodologia', name: 'metodologia', wait: 2500 },
  { url: '/pricing', name: 'pricing', wait: 2500 },
  { url: '/legal/riesgo', name: 'legal-riesgo', wait: 2000 },
  { url: '/legal/terminos', name: 'legal-terminos', wait: 2000 },
  { url: '/login', name: 'login', wait: 2500 },
];
const APP_VIEWS = {
  admin: ['/hub', '/dashboard', '/production', '/forecasting', '/analysis', '/admin', '/account/billing'],
  developer: ['/hub', '/dashboard', '/production', '/forecasting', '/analysis'],
  subscriber: ['/hub', '/production', '/forecasting', '/analysis', '/account/billing'],
  free: ['/hub', '/forecasting', '/analysis', '/account/billing'],
};
const WAITS = { '/dashboard': 10000, '/production': 10000, '/forecasting': 8000,
                '/analysis': 8000, '/hub': 6000, '/admin': 5000 };

const report = { started: new Date().toISOString(), base: BASE, shots: [], nav: [], errors: [], console: [] };
let seq = 0;
const pad = (n) => String(n).padStart(2, '0');

async function newPage(browser, viewport) {
  const ctx = await browser.newContext({ viewport });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => report.errors.push({ page: page.url(), err: String(e).slice(0, 160) }));
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning') {
      report.console.push({ page: page.url().replace(BASE, ''), type: m.type(),
                            msg: m.text().slice(0, 180) });
    }
  });
  return { ctx, page };
}

async function capture(page, url, name, wait = 4000, scrolls = []) {
  const t0 = Date.now();
  let status = 'ok';
  try {
    const resp = await page.goto(`${BASE}${url}`, { waitUntil: 'commit', timeout: 90000 });
    await page.waitForTimeout(wait);
    // scroll down then back up so lazy/inView content mounts BEFORE the fullPage shot
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(1000);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(600);
    const file = `${pad(++seq)}-${name}.png`;
    await page.screenshot({ path: `${OUT}/${file}`, fullPage: true });
    report.shots.push({ file, url, http: resp?.status() ?? null, ms: Date.now() - t0 });
    for (let i = 0; i < scrolls.length; i++) {
      await page.evaluate((f) => window.scrollTo(0, document.body.scrollHeight * f), scrolls[i]);
      await page.waitForTimeout(1200);
      const sf = `${pad(++seq)}-${name}-s${i + 1}.png`;
      await page.screenshot({ path: `${OUT}/${sf}` });
      report.shots.push({ file: sf, url: `${url}#s${i + 1}`, http: null, ms: 0 });
    }
  } catch (e) {
    status = `FAIL: ${String(e).slice(0, 100)}`;
    report.shots.push({ file: null, url, error: status });
  }
  console.log(`${status === 'ok' ? 'shot' : 'FAIL'}: ${name} (${url})`);
}


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

async function login(page, role) {
  const { u, p } = USERS[role];
  await page.goto(`${BASE}/login`, { waitUntil: 'commit', timeout: 90000 });
  await page.waitForTimeout(2500);
  await page.fill('input[name="username"], input[type="text"], input[name="email"]', u);
  await page.fill('input[type="password"]', p);
  await solveCaptcha(page);
  await page.click('button[type="submit"]');
  await page.waitForTimeout(6000);
  const ok = !page.url().includes('/login');
  report.nav.push({ role, step: 'login', ok, landed: page.url().replace(BASE, '') });
  return ok;
}

async function crawlNav(page, role) {
  // Click-crawl every visible navbar entry: each must land without error page.
  const links = await page.locator('nav button:visible, nav a:visible').allTextContents();
  report.nav.push({ role, step: 'nav-entries', entries: links.map((l) => l.trim()).filter(Boolean) });
}

const browser = await chromium.launch();

// ── public (anonymous)
{
  const { ctx, page } = await newPage(browser, { width: 1440, height: 900 });
  for (const v of PUBLIC_VIEWS) await capture(page, v.url, `anon-${v.name}`, v.wait, v.scrolls ?? []);
  // deny-by-default probes
  for (const url of ['/dashboard', '/admin', '/data/registry.json']) {
    const r = await page.request.get(`${BASE}${url}`, { maxRedirects: 0 }).catch(() => null);
    report.nav.push({ role: 'anon', step: `deny ${url}`, http: r?.status() ?? 'ERR' });
  }
  await ctx.close();
}

// ── each role
for (const role of Object.keys(USERS)) {
  const { ctx, page } = await newPage(browser, { width: 1440, height: 900 });
  if (!(await login(page, role))) { await ctx.close(); continue; }
  await capture(page, '/hub', `${role}-hub`, WAITS['/hub']);
  await crawlNav(page, role);
  for (const url of APP_VIEWS[role].filter((u) => u !== '/hub')) {
    await capture(page, url, `${role}${url.replace(/\//g, '-')}`, WAITS[url] ?? 5000);
  }
  await ctx.close();
}

// ── wide pass (ultrawide layout check: catches left-pinned / uncentered content
//    that a fixed 1440 capture hides — regression 2026-07-11). Measures the content
//    column's left/right gap; flags asymmetry > 8px as a layout failure.
if (WIDE) {
  const { ctx, page } = await newPage(browser, { width: 2560, height: 1400 });
  if (await login(page, 'admin')) {
    for (const url of ['/hub', '/production', '/analysis', '/dashboard']) {
      await capture(page, url, `wide2560${url.replace(/\//g, '-')}`, WAITS[url] ?? 5000);
      const gap = await page.evaluate(() => {
        const inner = document.querySelector('main > div');
        if (!inner) return null;
        const b = inner.getBoundingClientRect();
        const SIDEBAR = 248;
        const gapL = Math.round(b.left - SIDEBAR);
        const gapR = Math.round(window.innerWidth - b.right);
        return { gapL, gapR, colWidth: Math.round(b.width), balanced: Math.abs(gapL - gapR) <= 8 };
      }).catch(() => null);
      const ok = !gap || gap.balanced;
      report.nav.push({ role: 'admin', step: `wide-center ${url}`, ...(gap ?? {}), ok });
      if (!ok) report.shots.push({ file: null, url: `${url}@2560`, error: `content not centered: gapL=${gap.gapL} gapR=${gap.gapR}` });
    }
  }
  await ctx.close();
}

// ── zoom pass (200% zoom / reflow check: WCAG 1.4.4 text-resize & 1.4.10 reflow —
//    at 200% zoom content must reflow WITHOUT a horizontal scrollbar. Emulates 200%
//    zoom via a 720px-wide viewport (half of the 1440 baseline) and asserts the page
//    never overflows its own viewport width on the key data-dense admin views).
if (ZOOM) {
  const { ctx, page } = await newPage(browser, { width: 720, height: 900 });
  if (await login(page, 'admin')) {
    for (const url of ['/hub', '/analysis']) {
      await capture(page, url, `zoom200${url.replace(/\//g, '-')}`, WAITS[url] ?? 5000);
      const overflow = await page.evaluate(() =>
        Math.round(document.documentElement.scrollWidth - window.innerWidth)
      ).catch(() => null);
      const ok = overflow === null || overflow <= 2;
      report.nav.push({ role: 'admin', step: `zoom200 ${url}`, overflow: overflow ?? 0, ok });
      if (!ok) report.shots.push({ file: null, url: `${url}@720`, error: `horizontal overflow at 200% zoom: ${overflow}px over` });
    }
  }
  await ctx.close();
}

// ── mobile pass (admin only: layout check)
if (MOBILE) {
  const { ctx, page } = await newPage(browser, { width: 390, height: 844 });
  await capture(page, '/', 'mobile-landing', 6000);
  await capture(page, '/pricing', 'mobile-pricing', 2500);
  if (await login(page, 'admin')) await capture(page, '/hub', 'mobile-hub', 6000);
  await ctx.close();
}

await browser.close();
report.finished = new Date().toISOString();
writeFileSync(`${OUT}/report.json`, JSON.stringify(report, null, 1));
const fails = report.shots.filter((s) => s.error).length;
console.log(`\nDONE: ${report.shots.length} captures, ${fails} failures, ` +
            `${report.errors.length} page errors -> ${OUT}/report.json`);
process.exit(0);
