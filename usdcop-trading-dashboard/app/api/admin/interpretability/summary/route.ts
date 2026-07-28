/**
 * GET /api/admin/interpretability/summary?surface=&asset=&model_id=&version= (admin:all).
 *
 * Devuelve el `summary.json` del artefacto seleccionado TAL CUAL lo publicó el
 * generador Python (safe_json_dump: sin NaN/Inf) — la UI renderiza, nunca recomputa.
 * Segmentos validados contra un whitelist estricto: cero traversal.
 */
import fs from 'fs/promises';
import path from 'path';

import { ok, fail } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import type { InterpSummary } from '@/lib/contracts/admin-console.contract';

const ROOT = path.join(process.cwd(), 'public', 'data', 'interpretability');
const SEG = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

export async function GET(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  const q = new URL(req.url).searchParams;
  const parts = ['surface', 'asset', 'model_id', 'version'].map((k) => q.get(k) ?? '');
  if (parts.some((p) => !SEG.test(p) || p.includes('..'))) {
    return fail('BAD_PARAMS', 'surface/asset/model_id/version requeridos (segmentos [A-Za-z0-9._-])', 400);
  }
  const [surface, asset, modelId, version] = parts;

  const file = path.join(ROOT, surface, asset, modelId, version, 'summary.json');
  try {
    const summary = JSON.parse(await fs.readFile(file, 'utf-8')) as InterpSummary;
    return ok(summary, { meta: { asOf: new Date().toISOString() } });
  } catch (e) {
    const code = (e as NodeJS.ErrnoException)?.code;
    if (code === 'ENOENT') {
      return fail('NOT_FOUND', `sin artefacto para ${surface}/${asset}/${modelId}/${version}`, 404);
    }
    return fail('ARTIFACT_UNREADABLE', String((e as Error)?.message ?? e), 500);
  }
}
