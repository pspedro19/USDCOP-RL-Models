/**
 * GET /api/passport/tower — Control Tower snapshot (BL-32, FABRIC §24.5).
 *
 * READ-ONLY diagnostic surface. Composes LIBRO / SLEEVES / DATOS from PUBLISHED
 * artifacts only (`lib/passport/compose.ts`); anything without a producer comes
 * back as `unavailable` with the backlog item that will supply it. No approval,
 * promote or deploy affordance exists here by contract — Vote 2 lives on
 * /dashboard (approval-gates.md invariante 3).
 *
 * RBAC: `research:read` via `/api/passport` in rbac.contract.ts (edge-enforced).
 */
import { ok, fail } from '@/lib/api/envelope';
import { composeControlTower, listPassportStrategies } from '@/lib/passport/compose';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const [tower, strategies] = await Promise.all([
      composeControlTower(),
      listPassportStrategies(),
    ]);
    return ok({ tower, strategies }, { meta: { asOf: tower.generated_at } });
  } catch (e) {
    return fail('TOWER_COMPOSE_FAILED', (e as Error).message ?? 'no se pudo componer la torre', 500);
  }
}
