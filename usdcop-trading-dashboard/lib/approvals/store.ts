/**
 * Approval-state store (CXD-057) — acceso SERVER-ONLY al estado de aprobación.
 * ============================================================================
 *
 * Los `approval_state*.json` viven FUERA de `public/`, en `<repo>/data/approvals/`,
 * por la misma razón que los artefactos SHAP (`data/interpretability/`, C-006) y la
 * proyección de gobernanza (`data/control-tower/`, H1): bajo `public/` el fichero se
 * sirve por el estático `/data/**`, que `middleware.ts` gatea con *una sesión* y nada
 * más, mientras el SSOT reserva gates / DSR / métricas de backtest a `research:read`
 * (`frontend-backend-contract.md` §6, `ux-navigation.md` P3, `VISUAL-SPEC-CHECKLIST` §B).
 * Un `free`/`subscriber` autenticado pedía el JSON directo y se saltaba el gate entero.
 *
 * Reglas de este módulo:
 *  - **Fail-closed**: artefacto ausente/corrupto ⇒ `null` con motivo declarado por el
 *    caller (404), JAMÁS un estado por defecto fabricado que parezca "sin gates".
 *  - **Allowlist, no blacklist**: `toPublicApproval` ENUMERA lo publicable. Un campo
 *    interno nuevo escrito por el pipeline queda fuera por construcción.
 *  - **Traversal**: el `strategy_id` pasa por whitelist ANTES de resolver, y el
 *    realpath del fichero debe quedar bajo el realpath de la raíz.
 */
import { promises as fs } from 'fs';
import path from 'path';

import type { ApprovalState, ProductionStatus } from '@/lib/contracts/production-approval.contract';

/** Estrategia por defecto (COP): el pipeline escribe el ACTIVO en el fichero sin sufijo. */
export const DEFAULT_APPROVAL_FILE = 'approval_state.json';

/** Whitelist de `strategy_id` — bloquea `..`, separadores y absolutos antes de resolver. */
export const SID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

/**
 * Raíz privada de artefactos de aprobación. Default `<repo>/data/approvals`
 * (el cwd del dashboard es `usdcop-trading-dashboard/`); override con
 * `APPROVALS_DATA_DIR` para contenedores/tests (igual que `CONTROL_TOWER_DATA_DIR`).
 */
export function approvalsRoot(): string {
  const env = process.env.APPROVALS_DATA_DIR;
  if (env && env.trim()) return path.resolve(env.trim());
  return path.resolve(process.cwd(), '..', 'data', 'approvals');
}

export function isValidStrategyId(sid: string | null | undefined): sid is string {
  return typeof sid === 'string' && SID_RE.test(sid) && !sid.includes('..');
}

/** Resuelve un nombre de fichero dentro de la raíz, verificando el prefijo real. */
async function resolveInRoot(fileName: string): Promise<string | null> {
  let realBase: string;
  try {
    realBase = await fs.realpath(approvalsRoot());
  } catch {
    return null; // sin directorio privado → fail-closed
  }
  const candidate = path.resolve(realBase, fileName);
  const rel = path.relative(realBase, candidate);
  if (rel.startsWith('..') || path.isAbsolute(rel) || rel.includes(path.sep)) return null;
  try {
    const real = await fs.realpath(candidate);
    if (!real.startsWith(realBase + path.sep)) return null; // symlink fuera de la raíz
    return real;
  } catch {
    return null; // ENOENT
  }
}

async function readJson(file: string): Promise<ApprovalState | null> {
  try {
    const st = await fs.stat(file);
    if (!st.isFile() || st.size > 2 * 1024 * 1024) return null;
    return JSON.parse(await fs.readFile(file, 'utf-8')) as ApprovalState;
  } catch {
    return null;
  }
}

export interface ApprovalRecord {
  /** Ruta absoluta del artefacto leído (para escrituras del Voto 2 y trazabilidad). */
  file: string;
  /** Ruta relativa al repo, apta para `source.path` del Passport. */
  repoPath: string;
  state: ApprovalState;
}

const repoRel = (fileName: string) => `data/approvals/${fileName}`;

/**
 * Lee el artefacto ÍNTEGRO de una estrategia.
 *
 * Resolución multi-estrategia — DEBE coincidir con la del DAG H5-L4b y la del approve:
 * `approval_state_<sid>.json` si existe; si no, el singleton `approval_state.json`
 * SOLO cuando pertenece a esa estrategia (una id obsoleta jamás aprueba/despliega el
 * bundle de otra). Sin `sid` ⇒ el singleton.
 */
export async function readApprovalState(sid?: string | null): Promise<ApprovalRecord | null> {
  if (sid && !isValidStrategyId(sid)) return null;

  if (sid) {
    const scopedName = `approval_state_${sid}.json`;
    const scoped = await resolveInRoot(scopedName);
    if (scoped) {
      const state = await readJson(scoped);
      if (state) return { file: scoped, repoPath: repoRel(scopedName), state };
      return null; // existe pero corrupto ⇒ fail-closed, no cae al singleton de otro
    }
  }

  const singleton = await resolveInRoot(DEFAULT_APPROVAL_FILE);
  if (!singleton) return null;
  const state = await readJson(singleton);
  if (!state) return null;
  if (sid && state.strategy !== sid) return null; // el singleton NO es de esta estrategia
  return { file: singleton, repoPath: repoRel(DEFAULT_APPROVAL_FILE), state };
}

// ═══════════════════════ transición del Voto 2: lock + CAS + publicación atómica ═══
//
// El estado de aprobación es un SSOT mutado por al menos TRES escritores (la ruta del
// Voto 2, el export de Python `--phase backtest|--reset-approval` y el DAG H5-L4b). El
// patrón anterior —leer, mutar en memoria, `fs.writeFile` sobre el destino— tenía dos
// defectos que en esta ruta concreta valen dinero real:
//
//   1. **Sin CAS**: dos peticiones simultáneas leían `PENDING_APPROVAL`, ambas
//      respondían 200 y la última ganaba el JSON. Pero el APPROVE perdedor ya había
//      disparado el deploy fire-and-forget ⇒ estado REJECTED con un deploy en marcha.
//   2. **Sin publicación atómica**: un crash a mitad de `writeFile` deja el SSOT
//      truncado, y el fail-closed del lector lo convierte en un 404 de aprobación.
//
// Mecanismo elegido (mismo primitivo que el publicador de interpretabilidad, C-006):
// **creación exclusiva** — allí `os.link`/`O_EXCL`, aquí `fs.open(lock,'wx')`, que es
// exactamente `O_CREAT|O_EXCL`. Es la única operación de sistema de ficheros que es
// atómica **y** falla si el destino existe, así que da *exactamente-un-ganador* sin
// depender de un servicio externo. Diferencia con C-006: allí el objeto publicado no
// existía (CAS de creación); aquí hay que MUTAR un fichero existente, y una creación
// exclusiva sobre el propio artefacto no aplica. Por eso el `O_EXCL` protege un
// *lockfile* y la comparación real se hace dentro: se RELEE del disco bajo el lock y se
// verifica la precondición (`status`) antes de publicar. El lock serializa; el compare
// decide. Uno gana con 200, el otro ve el estado ya resuelto y recibe 409.
//
// Por qué no otras opciones: `fs.rename` solo es atómico, no exclusivo (dos escritores
// divergentes lo llaman ambos y el último pisa — el mismo bug); un mutex en memoria del
// proceso Node no cubre al escritor Python ni a varias réplicas; una tabla en Postgres
// sería el mecanismo correcto a largo plazo pero ataría el Voto 2 a la disponibilidad
// de la DB y a una migración — queda declarado como deuda en el reporte.

/** Sufijo del lockfile interproceso. Espejo en `src/contracts/approval_store.py`. */
export const LOCK_SUFFIX = '.lock';
/** Un lock más viejo que esto solo puede venir de un escritor que murió. */
const LOCK_STALE_MS = 30_000;
/** Espera máxima por el lock: pasado esto se responde 409, nunca se escribe a ciegas. */
const LOCK_WAIT_MS = 4_000;
const LOCK_RETRY_MS = 15;

export class ApprovalLockTimeout extends Error {}

async function acquireLock(file: string): Promise<() => Promise<void>> {
  const lock = file + LOCK_SUFFIX;
  const deadline = Date.now() + LOCK_WAIT_MS;
  for (;;) {
    try {
      const fh = await fs.open(lock, 'wx'); // O_CREAT|O_EXCL — exactamente un ganador
      try {
        await fh.writeFile(JSON.stringify({ pid: process.pid, at: new Date().toISOString() }));
      } finally {
        await fh.close();
      }
      let released = false;
      return async () => {
        if (released) return;
        released = true;
        await fs.rm(lock, { force: true });
      };
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== 'EEXIST') throw e;
      // Se libera SIEMPRE en `finally`, así que un lock presente y viejo solo puede
      // venir de un proceso muerto: se retira una vez y se reintenta.
      try {
        const st = await fs.stat(lock);
        if (Date.now() - st.mtimeMs > LOCK_STALE_MS) {
          await fs.rm(lock, { force: true });
          continue;
        }
      } catch {
        continue; // el titular lo soltó entre el EEXIST y el stat
      }
      if (Date.now() >= deadline) {
        throw new ApprovalLockTimeout(`approval state busy: ${path.basename(file)}`);
      }
      await new Promise((r) => setTimeout(r, LOCK_RETRY_MS));
    }
  }
}

/**
 * Publica `state` en `file` sin ventana de truncado: temporal en el MISMO directorio
 * (mismo volumen ⇒ rename atómico), `fsync`, y `rename` sobre el destino. Un crash en
 * cualquier punto deja el destino con su contenido anterior ÍNTEGRO.
 */
async function publishAtomic(file: string, state: ApprovalState): Promise<void> {
  const payload = JSON.stringify(state, null, 2);
  if (payload === undefined) throw new Error('approval state is not serializable');
  const tmp = `${file}.tmp-${process.pid}-${Math.random().toString(36).slice(2, 10)}`;
  try {
    await fs.writeFile(tmp, payload, 'utf-8');
    const fh = await fs.open(tmp, 'r+');
    try {
      await fh.sync();
    } finally {
      await fh.close();
    }
    await fs.rename(tmp, file);
  } catch (e) {
    await fs.rm(tmp, { force: true });
    throw e;
  }
}

/**
 * Escritura directa (NO transaccional). Solo para el bootstrap del artefacto por el
 * pipeline; **el Voto 2 usa `commitApprovalTransition`**. Publica igualmente por
 * temporal + rename para no poder truncar el SSOT.
 */
export async function writeApprovalState(file: string, state: ApprovalState): Promise<void> {
  await publishAtomic(file, state);
}

export type TransitionOutcome =
  | { ok: true; state: ApprovalState }
  | { ok: false; code: 'CONFLICT' | 'BUSY' | 'MISSING'; status: ProductionStatus | null; message: string };

/**
 * Transición **compare-and-set** del estado de aprobación.
 *
 * - `precondition` recibe el estado RELEÍDO DEL DISCO bajo el lock (no el que leyó el
 *   handler antes) y devuelve `null` si puede proceder o el motivo del conflicto.
 * - `mutate` devuelve el estado nuevo; se publica atómicamente y solo entonces la
 *   función retorna `ok: true`. El caller puede disparar efectos (deploy) DESPUÉS,
 *   con la garantía de que el commit ya está en disco y de que fue el ganador.
 */
export async function commitApprovalTransition(
  file: string,
  precondition: (current: ApprovalState) => string | null,
  mutate: (current: ApprovalState) => ApprovalState,
): Promise<TransitionOutcome> {
  let release: (() => Promise<void>) | null = null;
  try {
    release = await acquireLock(file);
  } catch (e) {
    if (e instanceof ApprovalLockTimeout) {
      return { ok: false, code: 'BUSY', status: null, message: 'Another approval transition is in progress. Retry.' };
    }
    throw e;
  }
  try {
    // COMPARE: la verdad es lo que hay en disco AHORA, no lo que leyó el handler.
    const current = await readJson(file);
    if (!current) {
      return { ok: false, code: 'MISSING', status: null, message: 'Approval artifact vanished or is unreadable.' };
    }
    const why = precondition(current);
    if (why) {
      return { ok: false, code: 'CONFLICT', status: current.status ?? null, message: why };
    }
    // SET
    const next = mutate(current);
    await publishAtomic(file, next);
    return { ok: true, state: next };
  } finally {
    await release();
  }
}

/** Todos los artefactos publicados (admin: consola de modelos / cola de promoción). */
export async function listApprovalStates(): Promise<ApprovalRecord[]> {
  const out: ApprovalRecord[] = [];
  let names: string[];
  try {
    names = (await fs.readdir(approvalsRoot())).filter((f) => /^approval_state.*\.json$/.test(f));
  } catch {
    return out; // sin directorio → lista vacía (el caller decide si degrada o falla)
  }
  for (const n of names.sort()) {
    const file = await resolveInRoot(n);
    if (!file) continue;
    const state = await readJson(file);
    if (state) out.push({ file, repoPath: repoRel(n), state });
  }
  return out;
}

// ───────────────────────────────────── proyección pública (ALLOWLIST, no blacklist)

/**
 * Los ÚNICOS campos que pueden salir a una superficie de cliente (`signals:read`).
 * Todo lo demás —`gates`, `backtest_metrics`, `backtest_recommendation`,
 * `backtest_confidence`, `deploy_manifest`, notas de revisor, y cualquier campo
 * FUTURO— queda fuera por construcción. Añadir uno aquí es una decisión explícita.
 */
export const PUBLIC_APPROVAL_FIELDS = [
  'status',
  'strategy',
  'strategy_name',
  'approved_at',
  'created_at',
  'last_updated',
] as const;

export type PublicApprovalField = (typeof PUBLIC_APPROVAL_FIELDS)[number];

export interface PublicApprovalState {
  status: ProductionStatus;
  strategy: string;
  strategy_name: string | null;
  approved_at: string | null;
  created_at: string | null;
  last_updated: string | null;
}

/** Proyecta por allowlist. No recorre el objeto de entrada: lo RECONSTRUYE. */
export function toPublicApproval(state: ApprovalState): PublicApprovalState {
  return {
    status: state.status,
    strategy: state.strategy,
    strategy_name: state.strategy_name ?? null,
    approved_at: state.approved_at ?? null,
    created_at: state.created_at ?? null,
    last_updated: state.last_updated ?? null,
  };
}
