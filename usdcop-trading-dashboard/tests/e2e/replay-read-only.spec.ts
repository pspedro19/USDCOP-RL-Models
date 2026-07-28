/**
 * BL-34 remediation E2E — /replay es investigación READ-ONLY (approval-gates.md).
 * =================================================================================
 * Rechazo Codex (INBOX-CLAUDE 2026-07-27T22:27:30 + CXD-009): "/replay reexporta
 * /dashboard y screenshot 60d0af8 exhibe botones admin Aprobar/Rechazar; nav aun
 * enlaza /dashboard". ACK C-002: la entrada RBAC "no aprueba por si solo BL-34 ni
 * sustituye pruebas de autorizacion server-side/Playwright".
 *
 * Este spec prueba, contra el dev server real:
 *  1. AUTORIZACIÓN: anónimo en /replay → redirect a /login (edge middleware);
 *     anónimo en /api/registry (API del replay) → 401 JSON.
 *  2. READ-ONLY: admin logueado en /replay ve el replay alimentado por el bundle
 *     del registry (selector de estrategias + KPIs) pero NUNCA la superficie de
 *     Vote-2 (botones "Aprobar (Voto 2/2)"/"Rechazar" ni el panel de deploy) —
 *     esos viven SOLO en /dashboard (approval-gates.md invariante 3).
 *  3. PREVIEW: las cifras recomputadas por el replay se etiquetan como PREVIEW
 *     (el Voto 2 y los gates leen el bundle publicado — quant-constitution §7).
 *  4. NAV: la entrada de investigación "Backtest" navega a /replay (nav correcto);
 *     la superficie de aprobación /dashboard queda como entrada admin-only.
 *
 * Run: BASE_URL=http://localhost:3001 npx playwright test tests/e2e/replay-read-only.spec.ts --project=chromium
 */
import { test, expect, type Page } from '@playwright/test';

const ADMIN_USER = process.env.QA_ADMIN_USER ?? 'admin';
const ADMIN_PASS = process.env.QA_ADMIN_PASS ?? 'Admin2026!';

/** Resuelve el captcha aritmético del login (espera a que la pregunta cargue del API). */
async function solveCaptcha(page: Page): Promise<void> {
  const question = page.locator('text=/¿Cuánto es/').first();
  await question.waitFor({ state: 'visible', timeout: 30_000 });
  const q = (await question.textContent()) ?? '';
  const m = q.match(/(\d+)\s*([+×x*])\s*(\d+)/);
  if (!m) throw new Error(`Captcha ilegible: "${q}"`);
  const answer = m[2] === '+' ? Number(m[1]) + Number(m[3]) : Number(m[1]) * Number(m[3]);
  await page.fill('input[placeholder="respuesta"]', String(answer));
}

async function loginAs(page: Page, user: string, pass: string): Promise<void> {
  await page.goto('/login', { waitUntil: 'domcontentloaded' });
  // Espera a que la pregunta del captcha cargue ANTES de escribir: garantiza que React
  // ya hidrató el formulario (si no, la hidratación resetea los inputs controlados).
  // El fetch de montaje puede colgarse si el dev server está compilando — el botón
  // "Generar nueva operación" re-solicita el reto; reintenta hasta 4 veces.
  const question = page.locator('text=/¿Cuánto es/').first();
  for (let i = 0; i < 4; i++) {
    try {
      await question.waitFor({ state: 'visible', timeout: 20_000 });
      break;
    } catch {
      if (i === 3) throw new Error('captcha nunca cargó (4 intentos)');
      await page.getByRole('button', { name: /Generar nueva operación/i }).click().catch(() => {});
    }
  }
  await page.fill('input[name="username"], input[type="text"], input[name="email"]', user);
  await page.fill('input[type="password"]', pass);
  await solveCaptcha(page);
  const submit = page.locator('button[type="submit"]');
  await expect(submit).toBeEnabled({ timeout: 15_000 });
  await submit.click();
  // El proxy de login (SignalBridge + NextAuth) puede tardar bastante en dev.
  await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 120_000 });
  // Deja asentar la redirección client-side post-login antes del siguiente goto
  // (evita net::ERR_ABORTED por carrera con la navegación en curso).
  await page.waitForLoadState('load').catch(() => {});
}

test.describe('BL-34 — /replay read-only (autorización server-side + constitución)', () => {
  test('anon: /replay NO se sirve — el edge redirige a /login', async ({ request }) => {
    const res = await request.get('/replay', { maxRedirects: 0 });
    expect([302, 307]).toContain(res.status());
    expect(res.headers()['location'] ?? '').toContain('/login');
  });

  test('anon: /api/registry (API del replay) responde 401 JSON', async ({ request }) => {
    const res = await request.get('/api/registry', { maxRedirects: 0 });
    expect(res.status()).toBe(401);
    const body = await res.json();
    expect(body.error).toBeTruthy();
  });

  test('admin: /replay es read-only (sin Vote-2), etiqueta PREVIEW y la nav enlaza /replay', async ({ page }) => {
    // Un solo login (el proxy de auth en dev es lento); luego los 3 chequeos en secuencia.
    test.setTimeout(360_000);
    await loginAs(page, ADMIN_USER, ADMIN_PASS);

    // ── 1. /replay renderiza el replay del bundle SIN superficie de aprobación ──
    await page.goto('/replay', { waitUntil: 'domcontentloaded' });
    // Con estado PENDING_APPROVAL publicado, /dashboard SÍ muestra Aprobar/Rechazar
    // para admin — /replay no debe mostrarlos JAMÁS (este era el rechazo de Codex).
    // Espera a que la sección cargue (bundle del registry → selector con Sharpe).
    await expect(page.getByText(/Sharpe/).first()).toBeVisible({ timeout: 60_000 });

    // Datos del bundle del registry: el selector muestra la estrategia default del SSOT.
    const reg = await page.evaluate(() => fetch('/api/registry').then((r) => (r.ok ? r.json() : null)));
    expect(reg?.strategies?.length ?? 0).toBeGreaterThan(0);
    const defId: string = reg.default?.strategy_id ?? reg.strategies[0].strategy_id;
    const defName: string =
      reg.strategies.find((s: { strategy_id: string }) => s.strategy_id === defId)?.display_name ?? defId;
    await expect(page.getByText(defName).first()).toBeVisible({ timeout: 15_000 });

    // READ-ONLY: cero superficie de aprobación (ni Aprobar, ni Rechazar, ni deploy).
    await expect(page.getByRole('button', { name: /Aprobar/i })).toHaveCount(0);
    await expect(page.getByRole('button', { name: /^Rechazar/i })).toHaveCount(0);
    await expect(page.getByText(/Desplegar a producción/i)).toHaveCount(0);

    // Nota explícita de solo-lectura (el Voto 2 vive en /dashboard).
    await expect(page.getByTestId('replay-readonly-note')).toBeVisible();

    // ── 2. Las cifras recomputadas por el replay se etiquetan PREVIEW ──
    // (audit I-4 / quant-constitution §7: los gates y el Vote 2 leen el bundle publicado).
    await page.getByRole('button', { name: /Reproducir replay|Play replay/i }).click();
    await expect(page.getByText(/PREVIEW/).first()).toBeVisible({ timeout: 60_000 });

    // ── 3. Nav correcta (rechazo: "nav aun enlaza /dashboard") ──
    await page.goto('/hub', { waitUntil: 'domcontentloaded' });
    const backtestNav = page.locator('nav button', { hasText: /^Backtest$/ }).first();
    await expect(backtestNav).toBeVisible({ timeout: 30_000 });
    await backtestNav.click();
    await page.waitForURL(/\/replay/, { timeout: 30_000 });

    // La superficie de Vote-2 sigue existiendo para admin como entrada separada.
    const approvalNav = page.locator('nav button', { hasText: /^Aprobación$/ }).first();
    await expect(approvalNav).toBeVisible();
    await approvalNav.click();
    await page.waitForURL(/\/dashboard/, { timeout: 30_000 });
  });
});
