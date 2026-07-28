/**
 * GET /api/admin/interpretability/summary?surface=&asset=&model_id=&version= (admin:all).
 *
 * Devuelve el `summary.json` del artefacto seleccionado leído de
 * `<repo>/data/interpretability/**` (FUERA de public/ — C-006/CXD-040: la única vía
 * de acceso es esta API con gate admin), validado en runtime contra el JSON Schema
 * COMPARTIDO con el productor Python (`../_schema/interp-summary.schema.json`):
 * campos conocidos, unknown-field STRIP, números finitos, size-cap 2MB.
 *
 * Errores SIEMPRE genéricos: jamás se filtra el mensaje interno (path, errno, parse).
 *  - 400 BAD_PARAMS   → segmentos fuera del whitelist (traversal, absolutos, vacíos).
 *  - 404 NOT_FOUND    → no existe o su realpath escapa del base (symlink fuera).
 *  - 500 INVALID_ARTIFACT → existe pero no pasa size-cap/parse/schema.
 */
import { ok, fail } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import type { InterpSummary } from '@/lib/contracts/admin-console.contract';

import { resolveSummaryFile, readValidatedSummary, validSegments } from '../_lib/artifacts';

export async function GET(req: Request) {
  const gate = requirePermission(req, 'admin:all');
  if (gate instanceof Response) return gate;

  const q = new URL(req.url).searchParams;
  const parts = ['surface', 'asset', 'model_id', 'version'].map((k) => q.get(k) ?? '');
  if (!validSegments(parts)) {
    return fail('BAD_PARAMS', 'invalid params', 400);
  }

  const file = await resolveSummaryFile(parts);
  if (!file) {
    return fail('NOT_FOUND', 'artifact not found', 404);
  }

  const summary = await readValidatedSummary(file);
  if (summary === null) {
    return fail('INVALID_ARTIFACT', 'invalid artifact', 500);
  }

  return ok(summary as InterpSummary, { meta: { asOf: new Date().toISOString() } });
}
