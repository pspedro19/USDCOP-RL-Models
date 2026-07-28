/**
 * GET /api/production/status — proyección PÚBLICA (sanitizada) del estado de aprobación.
 * ======================================================================================
 * CXD-057. Esta es la superficie de CLIENTE (`signals:read`): devuelve SOLO los campos
 * enumerados en `PUBLIC_APPROVAL_FIELDS` (allowlist — nunca blacklist). `gates`,
 * el gate `deflated_sharpe`, `backtest_metrics`, `backtest_recommendation`,
 * `backtest_confidence`, `deploy_manifest` y las notas del revisor NO salen por aquí:
 * el SSOT los reserva a `research:read` (`frontend-backend-contract.md` §6,
 * `ux-navigation.md` P3, `docs/rbac/VISUAL-SPEC-CHECKLIST.md` §B).
 *
 * La proyección ÍNTEGRA vive en `GET /api/production/approval` (`research:read`).
 * El artefacto se lee de `<repo>/data/approvals/`, jamás de `public/`.
 *
 * Fail-closed: sin artefacto ⇒ 404 con motivo declarado. El `DEFAULT_STATE` fabricado
 * que había aquí antes mentía: presentaba `gates: []` + `PENDING_APPROVAL` como si
 * fueran el estado real del sistema cuando el fichero no existía.
 */
import { fail, ok } from '@/lib/api/envelope';
import { requirePermission } from '@/lib/api/relay';
import { isValidStrategyId, readApprovalState, toPublicApproval } from '@/lib/approvals/store';

export async function GET(req: Request) {
  const gate = requirePermission(req, 'signals:read');
  if (gate instanceof Response) return gate;

  const sid = new URL(req.url).searchParams.get('strategy_id');
  if (sid !== null && !isValidStrategyId(sid)) {
    return fail('INVALID_STRATEGY_ID', 'identificador de estrategia inválido', 400);
  }

  const record = await readApprovalState(sid);
  if (!record) {
    return fail(
      'APPROVAL_ARTIFACT_MISSING',
      'no hay estado de aprobación publicado para esa estrategia',
      404,
    );
  }

  return ok(toPublicApproval(record.state));
}
