/**
 * /api/watchlist (CTR-FE-BE-001 §4.3) — per-user watchlist over `user_watchlist`
 * (migration 057). Free feature: it grants NO access; entitlements are untouched.
 *
 *   GET  → { items: [{ asset_id, created_at }] }
 *   POST { asset_id } → idempotent add (INSERT ON CONFLICT DO NOTHING)
 *
 * asset_id must exist in the catalog (registry assets + coming-soon statics —
 * watching a coming-soon asset is allowed: it measures demand).
 */
import { fail, ok } from '@/lib/api/envelope';
import { requireSession } from '@/lib/api/relay';
import { query } from '@/lib/db/postgres-client';
import type { WatchlistResponse } from '@/lib/contracts/catalog.contract';
import { allCatalogAssets, isValidAssetIdShape } from '@/lib/services/catalog-registry';

export const dynamic = 'force-dynamic';

export async function GET(req: Request) {
  const gate = requireSession(req);
  if (gate instanceof Response) return gate;

  try {
    const res = await query<{ asset_id: string; created_at: string }>(
      'SELECT asset_id, created_at FROM user_watchlist WHERE user_id = $1 ORDER BY created_at DESC',
      [gate.userId],
    );
    const body: WatchlistResponse = {
      items: res.rows.map((r) => ({ asset_id: r.asset_id, created_at: new Date(r.created_at).toISOString() })),
    };
    return ok(body);
  } catch {
    return fail('UPSTREAM_UNAVAILABLE', 'Base de datos no disponible.', 502);
  }
}

export async function POST(req: Request) {
  const gate = requireSession(req);
  if (gate instanceof Response) return gate;

  let body: { asset_id?: unknown };
  try {
    body = await req.json();
  } catch {
    return fail('BAD_REQUEST', 'JSON inválido.', 400);
  }
  if (!isValidAssetIdShape(body.asset_id)) {
    return fail('BAD_REQUEST', 'asset_id inválido.', 400);
  }

  const known = await allCatalogAssets();
  if (!known.some((a) => a.asset_id === body.asset_id)) {
    return fail('NOT_FOUND', `Activo desconocido: ${body.asset_id}`, 404);
  }

  try {
    await query(
      'INSERT INTO user_watchlist (user_id, asset_id) VALUES ($1, $2) ON CONFLICT DO NOTHING',
      [gate.userId, body.asset_id],
    );
    return ok({ asset_id: body.asset_id, added: true });
  } catch {
    return fail('UPSTREAM_UNAVAILABLE', 'Base de datos no disponible.', 502);
  }
}
