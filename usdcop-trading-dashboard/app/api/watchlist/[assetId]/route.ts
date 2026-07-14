/**
 * DELETE /api/watchlist/{assetId} (CTR-FE-BE-001 §4.3) — remove from watchlist.
 * Idempotent: deleting an absent row still returns ok (removed=false).
 */
import { fail, ok } from '@/lib/api/envelope';
import { requireSession } from '@/lib/api/relay';
import { query } from '@/lib/db/postgres-client';
import { isValidAssetIdShape } from '@/lib/services/catalog-registry';

export const dynamic = 'force-dynamic';

export async function DELETE(req: Request, ctx: { params: Promise<{ assetId: string }> }) {
  const gate = requireSession(req);
  if (gate instanceof Response) return gate;

  const { assetId } = await ctx.params;
  if (!isValidAssetIdShape(assetId)) return fail('BAD_REQUEST', 'asset_id inválido.', 400);

  try {
    const res = await query(
      'DELETE FROM user_watchlist WHERE user_id = $1 AND asset_id = $2',
      [gate.userId, assetId],
    );
    return ok({ asset_id: assetId, removed: (res.rowCount ?? 0) > 0 });
  } catch {
    return fail('UPSTREAM_UNAVAILABLE', 'Base de datos no disponible.', 502);
  }
}
