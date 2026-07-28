/**
 * GET /api/admin/interpretability — índice de artefactos de interpretabilidad (admin:all).
 *
 * BL-20 (fase UI): enumera `public/data/interpretability/<surface>/<asset>/<model_id>/
 * <version>/summary.json` — artefactos que publica `scripts/analysis/
 * generate_interpretability.py` (SHAP lineal cerrado / atribución de reglas, solo
 * test-folds, 0 trials). La consola SOLO lee; regenerar = correr el script.
 * RBAC: cubierto por el prefijo `/api/admin` (rbac.contract.ts) + gate explícito aquí.
 */
import fs from 'fs/promises';
import path from 'path';

import { ok } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import type { InterpIndexEntry, InterpIndexResponse } from '@/lib/contracts/admin-console.contract';

const ROOT = path.join(process.cwd(), 'public', 'data', 'interpretability');
/** Segmentos de path válidos — bloquea traversal y nombres raros de raíz. */
const SEG = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

async function subdirs(p: string): Promise<string[]> {
  try {
    return (await fs.readdir(p, { withFileTypes: true }))
      .filter((d) => d.isDirectory() && SEG.test(d.name))
      .map((d) => d.name)
      .sort();
  } catch {
    return []; // sin directorio → índice vacío (degradación C5, no 500)
  }
}

export async function GET(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  const entries: InterpIndexEntry[] = [];
  for (const surface of await subdirs(ROOT)) {
    for (const asset of await subdirs(path.join(ROOT, surface))) {
      for (const modelId of await subdirs(path.join(ROOT, surface, asset))) {
        const versions: string[] = [];
        for (const version of await subdirs(path.join(ROOT, surface, asset, modelId))) {
          try {
            await fs.access(path.join(ROOT, surface, asset, modelId, version, 'summary.json'));
            versions.push(version);
          } catch { /* carpeta sin summary → se omite */ }
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
