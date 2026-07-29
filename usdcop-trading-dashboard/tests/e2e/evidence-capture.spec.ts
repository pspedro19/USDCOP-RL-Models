import { test, Page } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { readServedArtifact } from './support/artifact-freshness'

/**
 * Captura de evidencia para el canal de coordinacion.
 *
 * K-044: cada captura se nombra con el BUILD_ID del artefacto REALMENTE SERVIDO, para que
 * quien la mire pueda decidir si vale. Una captura sin sello no prueba de que version es.
 * No hace aserciones: su unico trabajo es dejar evidencia fechada y trazable.
 */

const EVIDENCE_DIR = path.resolve(__dirname, '../../../.claude/coordination/integration/evidence')

async function solveCaptcha(page: Page) {
  const qLoc = page.getByText(/Cuánto es/)
  await qLoc.waitFor({ timeout: 15_000 })
  const q = (await qLoc.textContent()) ?? ''
  const m = /(\d+)\s*([+×x*\-−])\s*(\d+)/.exec(q)
  if (!m) throw new Error(`captcha ilegible: "${q}"`)
  const [a, op, b] = [Number(m[1]), m[2], Number(m[3])]
  const answer = op === '+' ? a + b : op === '-' || op === '−' ? a - b : a * b
  await page.locator('input[placeholder="respuesta"]').fill(String(answer))
}

async function login(page: Page) {
  await page.goto('/login')
  const passwords = [process.env.PW_ADMIN_PASSWORD, 'Admin2026!', 'admin123'].filter(
    (p): p is string => Boolean(p),
  )
  for (const pw of passwords) {
    await page.locator('input[name="username"]').fill('admin')
    await page.locator('input[type="password"]').first().fill(pw)
    await solveCaptcha(page)
    await page.locator('button[type="submit"]').first().click()
    try {
      await page.waitForURL(/\/(hub|dashboard|production)/, { timeout: 10_000 })
      return
    } catch {
      /* siguiente candidata */
    }
  }
  throw new Error('login falló con todas las credenciales candidatas')
}

test('captura de evidencia sellada contra el BUILD_ID servido', async ({ page }) => {
  const baseURL = process.env.BASE_URL || 'http://localhost:5000'
  const { buildId } = await readServedArtifact(baseURL)
  fs.mkdirSync(EVIDENCE_DIR, { recursive: true })

  const shot = async (name: string) => {
    const file = path.join(EVIDENCE_DIR, `${buildId}__${name}.png`)
    await page.screenshot({ path: file, fullPage: true })
    console.log(`📸 ${path.basename(file)}`)
  }

  // 1. Anonimo: /replay no se sirve (el edge redirige a /login).
  await page.goto('/replay')
  await page.waitForLoadState('networkidle').catch(() => {})
  await shot('01_anon_replay_redirige_login')

  await login(page)

  // 2. Admin: /replay es read-only — sin Vote 2, con nota y etiqueta PREVIEW.
  await page.goto('/replay')
  await page.waitForLoadState('networkidle').catch(() => {})
  await page.waitForTimeout(2500)
  await shot('02_admin_replay_readonly')

  // 3. Admin, movil 375px: panel de candidatas A/B del paper ledger (BL-05).
  await page.setViewportSize({ width: 375, height: 812 })
  await page.goto('/production')
  await page.waitForLoadState('networkidle').catch(() => {})
  await page.waitForTimeout(4000)
  await shot('03_admin_production_375px')

  // 4. El mismo /production en escritorio, para contrastar.
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.reload()
  await page.waitForLoadState('networkidle').catch(() => {})
  await page.waitForTimeout(4000)
  await shot('04_admin_production_1440px')
})
