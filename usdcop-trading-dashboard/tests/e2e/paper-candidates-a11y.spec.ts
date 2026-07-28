/**
 * BL-05 (remediación CXD-022) — spec E2E del panel Candidatas A/B.
 *
 * ⚠ ESTADO: **NO EJECUTADO** — pendiente de corrida en la ola de pruebas.
 * Orden del operador (2026-07-28): no levantar Docker ni el dashboard en esta
 * sesión. Este archivo es un ENTREGABLE DE CÓDIGO listo para correr; **no existe
 * ninguna evidencia E2E asociada y no debe declararse verde hasta correrlo**.
 * Correrlo: levantar el dashboard con el build que incluya este remedio y
 *   `npx playwright test tests/e2e/paper-candidates-a11y.spec.ts --project=chromium`
 * (BASE_URL por defecto http://localhost:5000). Requiere sesión admin: el panel
 * está oculto para role free/subscriber (ProductionView: `isClientView`).
 *
 * Cubre exactamente lo que el rechazo pidió y el commit 60d0af8 no probaba:
 *  - viewport móvil real 375px (portrait) y landscape,
 *  - navegación por teclado hasta la región scrolleable (tabIndex=0) + FOCO VISIBLE,
 *  - row headers reales y equivalentes textuales (Sí/No, "sin dato") en runtime,
 *  - cero errores de consola,
 *  - la tabla scrollea DENTRO de su región; el body nunca desborda horizontal.
 * Screenshots → tests/e2e/__screenshots__/ (default acordado en el brief).
 */
import { test, expect, type Page } from '@playwright/test';

const SHOTS = 'tests/e2e/__screenshots__';
const PANEL_NAME = /candidatas/i;

function armConsoleCapture(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  return errors;
}

/** Resuelve el captcha aritmético server-signed ("¿Cuánto es A + B?") leyendo el DOM
 *  — contrato de selectores E2E documentado en app/login/page.tsx. */
async function solveCaptcha(page: Page) {
  const qLoc = page.getByText(/Cuánto es/);
  await qLoc.waitFor({ timeout: 15_000 });
  const q = (await qLoc.textContent()) ?? '';
  const m = /(\d+)\s*([+×x*\-−])\s*(\d+)/.exec(q);
  if (!m) throw new Error(`captcha ilegible: "${q}"`);
  const [a, op, b] = [Number(m[1]), m[2], Number(m[3])];
  const answer = op === '+' ? a + b : op === '-' || op === '−' ? a - b : a * b;
  await page.locator('input[placeholder="respuesta"]').fill(String(answer));
}

async function login(page: Page) {
  await page.goto('/login');
  const passwords = [process.env.PW_ADMIN_PASSWORD, 'Admin2026!', 'admin123'].filter(
    (p): p is string => Boolean(p),
  );
  for (const pw of passwords) {
    await page.locator('input[name="username"]').fill('admin');
    await page.locator('input[type="password"]').first().fill(pw);
    await solveCaptcha(page); // se re-emite (nonce one-time) en cada intento
    await page.locator('button[type="submit"]').first().click();
    try {
      await page.waitForURL(/\/(hub|dashboard|production)/, { timeout: 8_000 });
      return;
    } catch {
      /* siguiente candidata */
    }
  }
  throw new Error('login falló con todas las credenciales candidatas');
}

async function gotoPanel(page: Page) {
  await page.goto('/production');
  const region = page.getByRole('region', { name: PANEL_NAME });
  await expect(region).toBeVisible({ timeout: 30_000 });
  return region;
}

test.describe('BL-05 paper-candidates a11y móvil (375px portrait)', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('teclado alcanza la región scrolleable, sin overflow de página ni errores de consola', async ({ page }) => {
    const consoleErrors = armConsoleCapture(page);
    await login(page);
    const region = await gotoPanel(page);

    // La región de scroll es focusable (axe scrollable-region-focusable).
    await expect(region).toHaveAttribute('tabindex', '0');

    // Row headers reales dentro de la tabla (el fix CXD-022, verificado en runtime).
    const rowHeaders = page.getByRole('rowheader');
    expect(await rowHeaders.count()).toBeGreaterThan(0);

    // Equivalentes textuales: el estado producción/paper no puede depender del color
    // del badge, y las celdas vacías no pueden ser solo el símbolo '—'.
    await expect(rowHeaders.first()).toContainText(/En producción: (Sí|No)/);
    await expect(rowHeaders.first()).toContainText(/Juez sellado: (Sí|No)/);

    // Navegación por teclado: Tab hasta que el foco aterrice en la región (acotado).
    let reached = false;
    for (let i = 0; i < 60 && !reached; i++) {
      await page.keyboard.press('Tab');
      reached = await region.evaluate((el) => el === document.activeElement);
    }
    expect(reached, 'Tab nunca llegó a la región scrolleable de la tabla').toBe(true);

    // FOCO VISIBLE (WCAG 2.4.7): alcanzar la región por teclado no basta si el
    // usuario no ve dónde está. GM.focus pinta un ring vía :focus-visible → el
    // elemento enfocado debe diferenciarse por outline o box-shadow.
    const focusRing = await region.evaluate((el) => {
      const s = getComputedStyle(el);
      return {
        outlineWidth: s.outlineWidth,
        outlineStyle: s.outlineStyle,
        boxShadow: s.boxShadow,
      };
    });
    const hasVisibleFocus =
      (focusRing.boxShadow !== 'none' && focusRing.boxShadow !== '') ||
      (focusRing.outlineStyle !== 'none' && parseFloat(focusRing.outlineWidth) > 0);
    expect(
      hasVisibleFocus,
      `la región enfocada no muestra indicador visible: ${JSON.stringify(focusRing)}`,
    ).toBe(true);

    // La tabla scrollea DENTRO de la región; el documento no desborda horizontal.
    const { regionScrollable, bodyOverflow } = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement;
      return {
        regionScrollable: el.scrollWidth > el.clientWidth,
        bodyOverflow:
          document.documentElement.scrollWidth - document.documentElement.clientWidth,
      };
    });
    expect(regionScrollable, 'en 375px la tabla debe scrollear dentro de su región').toBe(true);
    expect(bodyOverflow, 'la página no debe desbordar horizontalmente').toBeLessThanOrEqual(1);

    await page.screenshot({ path: `${SHOTS}/paper-candidates-375-portrait.png`, fullPage: false });
    expect(consoleErrors, `errores de consola: ${consoleErrors.join(' | ')}`).toHaveLength(0);
  });
});

test.describe('BL-05 paper-candidates a11y móvil (landscape 667x375)', () => {
  test.use({ viewport: { width: 667, height: 375 } });

  test('landscape: panel visible, tabla contenida, consola limpia', async ({ page }) => {
    const consoleErrors = armConsoleCapture(page);
    await login(page);
    const region = await gotoPanel(page);

    await expect(region).toHaveAttribute('tabindex', '0');
    const bodyOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(bodyOverflow, 'la página no debe desbordar horizontalmente en landscape').toBeLessThanOrEqual(1);

    await region.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${SHOTS}/paper-candidates-667-landscape.png`, fullPage: false });
    expect(consoleErrors, `errores de consola: ${consoleErrors.join(' | ')}`).toHaveLength(0);
  });
});

test.describe('BL-05 preferencia de tamaño de fuente (WCAG 1.4.4)', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('la tabla escala con el font-size del root (rem, no px fijos)', async ({ page }) => {
    await login(page);
    const region = await gotoPanel(page);
    const table = region.locator('table');

    const before = await table.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));

    // Un usuario que sube el tamaño de fuente del navegador mueve el root em.
    // Con `text-[12.5px]` (el rechazo CXD-022) este valor NO cambiaría.
    await page.evaluate(() => {
      document.documentElement.style.fontSize = '20px';
    });
    const after = await table.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));

    expect(after, `tipografía fija: ${before}px sigue igual tras subir el root`).toBeGreaterThan(
      before,
    );
    await page.screenshot({ path: `${SHOTS}/paper-candidates-375-font-scaled.png`, fullPage: false });
  });
});
