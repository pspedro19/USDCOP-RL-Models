/**
 * GET /api/production/approval — proyección ÍNTEGRA del estado de aprobación.
 * ===========================================================================
 * CXD-057. Única vía de acceso a `gates`, al gate `deflated_sharpe` (DSR trial-aware)
 * y a `backtest_metrics`. Exige `research:read` (admin/developer) en el edge
 * (`rbac.contract.ts`) y OTRA vez aquí (defensa en profundidad: el middleware no es
 * el único gate). El artefacto vive en `<repo>/data/approvals/`, nunca bajo `public/`.
 *
 * Esta ruta es READ-ONLY. El Voto 2 (aprobar/rechazar) sigue viviendo SOLO en
 * `/api/production/approve` con `approval:vote` y superficie `/dashboard`
 * (`.claude/rules/approval-gates.md` invariantes 3 y 5) — aquí no se mueve
 * ninguna capacidad de acción.
 *
 * Fail-closed: sin artefacto ⇒ 404 `APPROVAL_ARTIFACT_MISSING` con motivo declarado;
 * jamás un estado por defecto que se lea como "sin gates".
 */
import { fail, ok } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import { isValidStrategyId, readApprovalState } from '@/lib/approvals/store';

export async function GET(req: Request) {
  const gate = requirePermission(req, 'research:read');
  if (gate instanceof Response) return gate;

  const sid = new URL(req.url).searchParams.get('strategy_id');
  if (sid !== null && !isValidStrategyId(sid)) {
    return fail('INVALID_STRATEGY_ID', 'identificador de estrategia inválido', 400);
  }

  const record = await readApprovalState(sid);
  if (!record) {
    return fail(
      'APPROVAL_ARTIFACT_MISSING',
      'no hay estado de aprobación publicado para esa estrategia '
        + '(el export de backtest --phase backtest aún no corrió, o el artefacto es ilegible)',
      404,
    );
  }

  // `source` viaja en cabecera (el envelope Meta es cerrado): trazabilidad sin
  // inventar campos en el contrato de respuesta.
  return ok(record.state, { headers: { 'x-artifact-source': record.repoPath } });
}
