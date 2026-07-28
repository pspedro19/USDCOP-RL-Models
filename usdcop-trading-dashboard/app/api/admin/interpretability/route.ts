/**
 * GET /api/admin/interpretability — índice de artefactos de interpretabilidad (admin:all).
 *
 * BL-20 (fase UI): enumera `<repo>/data/interpretability/<surface>/<asset>/<model_id>/
 * <version>/summary.json` — artefactos que publica `scripts/analysis/
 * generate_interpretability.py` (SHAP lineal cerrado / atribución de reglas, solo
 * test-folds, 0 trials). La consola SOLO lee; regenerar = correr el script.
 *
 * C-006/CXD-040: los artefactos viven FUERA de public/ (los estáticos solo exigían
 * sesión ⇒ bypass del gate admin). La ÚNICA vía de acceso es esta API detrás de
 * `requirePermission('admin:all')` (prefijo `/api/admin` en rbac.contract.ts + gate
 * explícito aquí). Lectura fs vía `_lib/artifacts.ts` (whitelist + realpath).
 */
import path from 'path';

import { ok } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import type { InterpIndexEntry, InterpIndexResponse } from '@/lib/contracts/admin-console.contract';

import { artifactsRoot, resolveSummaryFile, safeSubdirs } from './_lib/artifacts';

export async function GET(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  const root = artifactsRoot();
  const entries: InterpIndexEntry[] = [];
  for (const surface of await safeSubdirs(root)) {
    for (const asset of await safeSubdirs(path.join(root, surface))) {
      for (const modelId of await safeSubdirs(path.join(root, surface, asset))) {
        const versions: string[] = [];
        for (const version of await safeSubdirs(path.join(root, surface, asset, modelId))) {
          // Misma resolución canónica que el summary route: solo cuentan las versiones
          // cuyo summary.json REAL vive bajo el base real (no symlinks fuera, no huecos).
          if (await resolveSummaryFile([surface, asset, modelId, version])) {
            versions.push(version);
          }
        }
        if (versions.length) {
          // Descendente: [0] = la más reciente (versiones son fechas ISO).
          entries.push({ surface, asset, model_id: modelId, versions: versions.reverse() });
        }
      }
    }
  }

  const body: InterpIndexResponse = { entries };
  return ok(body, { meta: { asOf: new Date().toISOString() } });
}
