#!/usr/bin/env node
/**
 * Dynamic RBAC smoke (CTR-RBAC-001 / migración 056). Verifies, with a real admin
 * session: the role matrix endpoint, a live permission toggle (+ revert), the
 * self-lockout guardrail, and server-enforced downgrade-only "Ver como" preview.
 *
 *   node scripts/rbac-dynamic-qa.mjs
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
  } catch { /* no captcha */ }
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

const api = (method, url, body) => page.evaluate(async ({ m, u, bd }) => {
  const r = await fetch(u, m === 'GET'
    ? undefined
    : { method: m, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(bd ?? {}) });
  let data = null; try { data = await r.json(); } catch { /* empty */ }
  return { status: r.status, data };
}, { m: method, u: url, bd: body });

// 1) matrix endpoint returns the seeded map
const mx = await api('GET', '/api/admin/roles');
const subHas = (perm) => (mx.data?.data?.matrix?.subscriber ?? []).includes(perm);
check('GET /api/admin/roles matrix', mx.status === 200 && subHas('signals:read'), `status=${mx.status}`);

// 2) toggle a subscriber permission off → verify → restore
const off = await api('PATCH', '/api/admin/roles', { role: 'subscriber', permission: 'signals:read', enabled: false });
const afterOff = await api('GET', '/api/admin/roles');
check('PATCH toggle off applies', off.status === 200 && !((afterOff.data?.data?.matrix?.subscriber ?? []).includes('signals:read')));
const on = await api('PATCH', '/api/admin/roles', { role: 'subscriber', permission: 'signals:read', enabled: true });
const afterOn = await api('GET', '/api/admin/roles');
check('PATCH toggle on restores', on.status === 200 && (afterOn.data?.data?.matrix?.subscriber ?? []).includes('signals:read'));

// 3) guardrail: cannot strip admin:all from admin
const guard = await api('PATCH', '/api/admin/roles', { role: 'admin', permission: 'admin:all', enabled: false });
check('guardrail admin:all locked', guard.status === 403, `status=${guard.status}`);

// 4) per-user override endpoint reachable (self)
const meId = await page.evaluate(async () => {
  const r = await fetch('/api/admin/users/list');
  const d = await r.json().catch(() => null);
  const rows = d?.data?.users ?? d?.users ?? [];
  const admin = rows.find((u) => u.role === 'admin') ?? rows[0];
  return admin?.id ?? null;
});
if (meId) {
  const ov = await api('GET', `/api/admin/users/${meId}/overrides`);
  check('GET user overrides', ov.status === 200 && Array.isArray(ov.data?.data?.overrides));
} else {
  check('GET user overrides', false, 'could not resolve an admin user id');
}

// 5) server-enforced downgrade preview: view-as free → admin API read is 403; exit → 200
const beforePreview = await api('GET', '/api/admin/roles');
const startView = await api('POST', '/api/admin/impersonate', { role: 'free', motivo: 'smoke de preview downgrade' });
const duringPreview = await api('GET', '/api/admin/roles');
const endView = await api('DELETE', '/api/admin/impersonate');
const afterPreview = await api('GET', '/api/admin/roles');
check('preview downgrade blocks admin read',
  beforePreview.status === 200 && startView.status === 200 && duringPreview.status === 403 && afterPreview.status === 200,
  `before=${beforePreview.status} during=${duringPreview.status} after=${afterPreview.status}`);

console.log(`\nDYNAMIC RBAC: ${pass} PASS, ${fail} FAIL`);
await b.close();
process.exit(fail === 0 ? 0 : 1);
