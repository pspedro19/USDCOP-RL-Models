/**
 * K-044 — La evidencia de runtime se descarta si el artefacto servido no contiene
 * el codigo bajo prueba.
 *
 * Origen: el 2026-07-28 se corrieron dos specs E2E contra `localhost:5000` y salieron
 * 4 passed / 2 failed. El contenedor servia un build de las 23:59 del dia anterior, con
 * 21 commits de dashboard por delante. Los rojos no eran defectos y los verdes no eran
 * garantias: el codigo bajo prueba no estaba en la imagen. Peor, la advertencia ya existia
 * escrita en el mensaje de un commit de la noche anterior ("contenedor :5000 = build viejo,
 * rebuild pendiente") y aun asi se tropezo con ella — un aviso en prosa no bloquea nada.
 *
 * Por eso esto ABORTA la corrida en vez de avisar. Fail-closed tambien en la evidencia.
 */

import { execFileSync } from 'node:child_process'

/**
 * `generateBuildId` en next.config.js emite `build-<epoch_ms>-<sufijo>`.
 * Se busca el literal en cualquier parte del HTML: con App Router los chunks cuelgan de
 * `/_next/static/chunks/`, NO de `/_next/static/<buildId>/` como en Pages Router, asi que
 * anclarlo a la ruta de los chunks no lo encuentra.
 */
const BUILD_ID_RE = /(build-(\d{13})-[a-z0-9-]+)/

export interface ArtifactStamp {
  buildId: string
  builtAt: Date
}

/**
 * Lee el BUILD_ID del artefacto REALMENTE SERVIDO. No se pregunta al repo ni al
 * Dockerfile: se lee de la respuesta HTTP, que es lo unico que el navegador vera.
 */
export async function readServedArtifact(baseURL: string): Promise<ArtifactStamp> {
  const res = await fetch(`${baseURL}/login`)
  if (!res.ok) {
    throw new Error(`K-044: no se pudo leer el artefacto servido (${baseURL}/login => HTTP ${res.status})`)
  }
  const html = await res.text()
  const m = html.match(BUILD_ID_RE)
  if (!m) {
    // Fail-closed: si no se puede identificar el artefacto, no se puede juzgar la evidencia.
    throw new Error(
      'K-044: el HTML servido no expone un BUILD_ID reconocible. ' +
        'Sin identificar el artefacto, una corrida E2E no es evidencia en ninguna direccion.'
    )
  }
  return { buildId: m[1], builtAt: new Date(Number(m[2])) }
}

/**
 * Solo se recorta por la DERECHA: en `git status --porcelain` la primera columna del
 * codigo de estado puede ser un espacio (` M fichero`), y un `.trim()` completo se lo come,
 * desplazando el `slice(3)` y devolviendo la ruta con la primera letra amputada.
 */
function git(args: string[]): string {
  return execFileSync('git', args, { encoding: 'utf8', cwd: process.cwd() }).replace(/\s+$/, '')
}

/** Ultimo commit que toco el dashboard, que es el suelo minimo que el build debe cubrir. */
export function lastDashboardCommit(): { sha: string; at: Date; subject: string } {
  // `%H %ct %s`: sha y epoch no contienen espacios, asi que un split acotado basta
  // y evita meter un caracter de control literal en el fuente.
  const out = git(['log', '-1', '--format=%H %ct %s', '--', '.'])
  const [sha, ct, ...rest] = out.split(' ')
  const subject = rest.join(' ')
  return { sha, at: new Date(Number(ct) * 1000), subject }
}

/** Fuentes del dashboard modificadas y sin commitear: tampoco pueden estar en el build. */
export function uncommittedSources(): string[] {
  const out = git(['status', '--porcelain', '--', 'app', 'components', 'lib', 'middleware.ts'])
  return out ? out.split('\n').map((l) => l.slice(3).trim()).filter(Boolean) : []
}

/**
 * Aborta si el artefacto servido es anterior al codigo bajo prueba.
 * Devuelve el sello para que la evidencia (capturas, informes) pueda nombrarlo.
 */
export async function assertArtifactCoversCode(baseURL: string): Promise<ArtifactStamp> {
  const artifact = await readServedArtifact(baseURL)
  const commit = lastDashboardCommit()
  const dirty = uncommittedSources()

  const stale = artifact.builtAt < commit.at
  const escape = process.env.E2E_ALLOW_STALE_BUILD === '1'

  const stamp =
    `   artefacto servido : ${artifact.buildId}  (${artifact.builtAt.toISOString()})\n` +
    `   ultimo commit     : ${commit.sha.slice(0, 8)}  (${commit.at.toISOString()})  ${commit.subject}\n` +
    `   fuentes sin sellar: ${dirty.length ? dirty.join(', ') : 'ninguna'}`

  if (stale && !escape) {
    throw new Error(
      'K-044 — CORRIDA ABORTADA: el artefacto servido es ANTERIOR al codigo bajo prueba.\n' +
        stamp +
        '\n\n   Un verde contra un build rancio es peor que no tener evidencia, porque parece\n' +
        '   que si la tienes. Reconstruye (`docker compose build dashboard && docker compose up -d dashboard`)\n' +
        '   o apunta BASE_URL a un servidor de desarrollo con el codigo actual.\n' +
        '   Escape de emergencia, logueado: E2E_ALLOW_STALE_BUILD=1'
    )
  }

  if (stale && escape) {
    console.warn('\n⚠️  K-044 IGNORADA POR ESCAPE EXPLICITO (E2E_ALLOW_STALE_BUILD=1).')
    console.warn('   Lo que salga de aqui NO es evidencia valida de este commit.')
    console.warn(stamp + '\n')
    return artifact
  }

  if (dirty.length) {
    // No aborta: hay flujos legitimos con WIP en el arbol. Pero se declara, porque
    // el build no puede contener lo que aun no se ha escrito en disco compilado.
    console.warn(`\n⚠️  K-044: ${dirty.length} fuente(s) del dashboard sin commitear; el artefacto no las contiene.`)
    console.warn(stamp + '\n')
    return artifact
  }

  console.log(`✅ K-044: el artefacto cubre el codigo bajo prueba.\n${stamp}\n`)
  return artifact
}
