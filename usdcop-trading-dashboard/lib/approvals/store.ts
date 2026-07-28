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

/** Escritura del Voto 2 sobre el fichero YA resuelto por `readApprovalState`. */
export async function writeApprovalState(file: string, state: ApprovalState): Promise<void> {
  await fs.writeFile(file, JSON.stringify(state, null, 2), 'utf-8');
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
