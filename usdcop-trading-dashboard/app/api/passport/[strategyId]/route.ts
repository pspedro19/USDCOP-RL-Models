/**
 * GET /api/passport/[strategyId] — Strategy Passport (BL-32, FABRIC §24.4).
 *
 * The composition of `v_strategy_passport_live` + `mv_strategy_performance_daily`
 * (see `.claude/specs/platform/passport-control-tower.md` for the DDL that replaces
 * this composer the day BL-18/BL-21/BL-22/BL-24 land). READ-ONLY: identity,
 * governance, lineage, performance × 5 environments, live state and risk — every
 * field carrying the published artifact it came from.
 *
 * The static `/api/passport/tower` segment wins over this dynamic one in the App
 * Router, so `tower` is never treated as a strategy id.
 *
 * RBAC: `research:read` via `/api/passport` in rbac.contract.ts (edge-enforced).
 */
import { ok, fail } from '@/lib/api/envelope';
import { composeStrategyPassport } from '@/lib/passport/compose';

export const dynamic = 'force-dynamic';

export async function GET(_req: Request, ctx: { params: Promise<{ strategyId: string }> }) {
  const { strategyId } = await ctx.params;
  // Path traversal guard: the id indexes a directory under public/data/strategies.
  if (!/^[a-z0-9_-]+$/i.test(strategyId)) {
    return fail('BAD_STRATEGY_ID', 'strategy_id inválido', 400);
  }
  try {
    const passport = await composeStrategyPassport(strategyId);
    if (!passport) {
      return fail('NOT_FOUND', `sin registry/manifiesto publicado para ${strategyId}`, 404);
    }
    return ok(passport, { meta: { asOf: passport.generated_at } });
  } catch (e) {
    return fail('PASSPORT_COMPOSE_FAILED', (e as Error).message ?? 'no se pudo componer el passport', 500);
  }
}
