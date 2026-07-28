/**
 * FUENTE ÚNICA de los fixtures de interpretabilidad.
 *
 * Por qué existe: había DOS fixtures escritos a mano del mismo artefacto (la suite de
 * seguridad de la API y el render test de la sección admin). Cuando BL-20 extendió el
 * schema compartido con `artifact_id` + `provenance` (y `fit.n_fits` / `fit.n_train_scheme`),
 * ninguno de los dos se actualizó:
 *  - el de la API dejó la ruta feliz en 500 y, peor, hizo que los casos *fail-closed*
 *    de esa suite pasaran por el fixture rancio en vez de por la defensa que dicen probar;
 *  - el del componente quedó rojo en `tsc` y con números que ya no son los publicados.
 * Dos fixtures a mano son dos verdades que divergen. Aquí hay una sola, y NO se escribe:
 * se LEE de los artefactos reales trackeados en `<repo>/data/interpretability/**` — los
 * mismos que sirve la API.
 *
 * Consecuencia deliberada: si el generador cambia la forma del artefacto, estos fixtures
 * cambian con él y el rojo aparece donde debe (en el contrato), no en un fixture olvidado.
 */
import realFs from 'node:fs';
import path from 'node:path';

import type {
  InterpLinearSummary,
  InterpRuleSummary,
  InterpYearFeatureRow,
} from '@/lib/contracts/admin-console.contract';

/**
 * `<repo>/data/interpretability` — la MISMA raíz por defecto que resuelve
 * `app/api/admin/interpretability/_lib/artifacts.ts` (cwd de vitest = `usdcop-trading-dashboard/`).
 * Se lee con `node:fs` síncrono a propósito: la suite de la API espía `fs/promises` y
 * cargar el fixture NO debe contar como una llamada al filesystem de la ruta bajo test.
 */
export const INTERP_DATA_ROOT = path.resolve(process.cwd(), '..', 'data', 'interpretability');

/** Última versión publicada del modelo (las versiones son fechas ISO ⇒ orden lexicográfico). */
function latestVersion(modelDir: string): string {
  const version = realFs
    .readdirSync(modelDir, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort()
    .at(-1);
  if (!version) throw new Error(`sin versiones publicadas bajo ${modelDir}`);
  return version;
}

function loadReal<T>(surface: string, asset: string, modelId: string): { summary: T; version: string } {
  const modelDir = path.join(INTERP_DATA_ROOT, surface, asset, modelId);
  const version = latestVersion(modelDir);
  const file = path.join(modelDir, version, 'summary.json');
  return { summary: JSON.parse(realFs.readFileSync(file, 'utf-8')) as T, version };
}

const linear = loadReal<InterpLinearSummary>('zoo', 'usdcop', 'ridge');
const rule = loadReal<InterpRuleSummary>('rule_based', 'spx500', 'spx500_regime_gated_v1');

/** Artefacto lineal REAL (zoo/usdcop/ridge) — la forma válida de HOY, sin retocar. */
export const REAL_LINEAR: InterpLinearSummary = linear.summary;
export const REAL_LINEAR_VERSION = linear.version;

/** Artefacto de atribución de reglas REAL (rule_based/spx500) — sin retocar. */
export const REAL_RULE: InterpRuleSummary = rule.summary;
export const REAL_RULE_VERSION = rule.version;

/**
 * Subconjunto FIEL del artefacto lineal: mismos valores, menos filas.
 * Sirve a los tests de render, que necesitan una tabla acotada para aserciones legibles.
 * Se conservan los `rank` originales (es un subconjunto, no una invención) y se ajusta
 * `n_features` a las features retenidas para que el contador X/N del badge siga siendo el
 * del artefacto recortado. Los kill-flags se filtran a las features retenidas: JAMÁS se
 * recalculan (los computa el generador — regla strategy-engines I-7).
 */
export function trimLinear(
  summary: InterpLinearSummary,
  features: string[],
  years: string[],
): InterpLinearSummary {
  const rowsFor = (year: string): InterpYearFeatureRow[] =>
    features.map((f) => {
      const hit = (summary.by_year[year] ?? []).find((r) => r.feature === f);
      if (!hit) throw new Error(`el artefacto real no trae ${f} en ${year}`);
      return hit;
    });

  return {
    ...summary,
    n_features: features.length,
    top_features: summary.top_features.filter((f) => features.includes(f.feature)),
    by_year: Object.fromEntries(years.map((y) => [y, rowsFor(y)])),
    kill_flags_sign_change_by_year: summary.kill_flags_sign_change_by_year.filter((f) =>
      features.includes(f),
    ),
  };
}

/** Subconjunto FIEL del artefacto de reglas: mismos valores, solo los últimos `n` años. */
export function trimRule(summary: InterpRuleSummary, lastNYears: number): InterpRuleSummary {
  const years = Object.keys(summary.by_year).sort().slice(-lastNYears);
  return {
    ...summary,
    by_year: Object.fromEntries(years.map((y) => [y, summary.by_year[y]])),
  };
}
