/**
 * K-044 — el perímetro que vigila el guard de frescura debe corresponderse con el repo REAL.
 *
 * Origen (CXD-085, 2026-07-28): la primera versión de la lista vigilaba `next.config.js` y el
 * fichero real es `next.config.ts`. Una alteración de la config de build **no elevaba el suelo**,
 * así que K-044 aceptaba como fresco un artefacto anterior a esa alteración. El guard existía,
 * pasaba verde, y tenía un agujero por el que cabía justo lo que decía impedir (K-033).
 *
 * Estos tests no comprueban que la lista sea "correcta" —eso es opinión—, comprueban que
 * **cada ruta declarada exista**, que es lo único objetivo y lo único que cierra esta clase
 * de defecto para siempre.
 */

import { describe, it, expect } from 'vitest'
import fs from 'node:fs'
import { assertServedPathsExist } from '../e2e/support/artifact-freshness'

describe('K-044 · perímetro de rutas servidas', () => {
  // Rojo si alguien declara una ruta que no existe (p.ej. vuelve a poner next.config.js).
  it('toda ruta declarada existe en el repo', () => {
    const rutas = assertServedPathsExist()
    expect(rutas.length).toBeGreaterThan(0)
    const inexistentes = rutas.filter((r) => !fs.existsSync(r))
    expect(inexistentes, `rutas vigiladas que no existen: ${inexistentes.join(', ')}`).toEqual([])
  })

  // Rojo si la config de build deja de estar vigilada: es el agujero exacto de CXD-085.
  it('la config de build está vigilada, sea cual sea su extensión', () => {
    const rutas = assertServedPathsExist()
    const config = rutas.filter((r) => /^next\.config\./.test(r))
    expect(
      config,
      'ninguna next.config.* en el perímetro: una alteración de la config de build no elevaría el suelo'
    ).not.toEqual([])
    for (const c of config) expect(fs.existsSync(c)).toBe(true)
  })

  // Rojo si el middleware —la capa que aplica RBAC— sale del perímetro sin que nadie se entere.
  it('el middleware está vigilado', () => {
    const rutas = assertServedPathsExist()
    expect(rutas.some((r) => /^middleware\./.test(r))).toBe(true)
  })

  // Rojo si se declara un directorio servido que no existe: el guard estaría mirando al vacío.
  it('los directorios servidos del build están todos presentes', () => {
    const rutas = assertServedPathsExist()
    for (const dir of ['app', 'components', 'lib', 'public']) {
      expect(rutas, `${dir} salió del perímetro vigilado`).toContain(dir)
    }
  })
})
