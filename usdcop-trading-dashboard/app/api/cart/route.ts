/**
 * /api/cart (CTR-FE-BE-001 §4.3) — per-user add-on cart over `user_cart`
 * (migration 057). The cart is a staging area only: entitlements are ONLY
 * granted by the billing webhook after payment (CTR-RBAC-001 rule 7).
 *
 *   GET  → { items: CartItem[] } — rows ENRICHED with registry metadata
 *          (symbol/name/class/price) so the drawer needs no second fetch.
 *   POST { asset_id } → idempotent add. Only 'available' catalog assets the
 *          user is NOT already entitled to can enter the cart.
 */
import { fail, ok } from '@/lib/api/envelope';
import { requireSession } from '@/lib/api/relay';
import { getEntitlements } from '@/lib/auth/entitlements';
import { query } from '@/lib/db/postgres-client';
import type { CartResponse } from '@/lib/contracts/catalog.contract';
import { allCatalogAssets, isValidAssetIdShape } from '@/lib/services/catalog-registry';

export const dynamic = 'force-dynamic';

export async function GET(req: Request) {
  const gate = requireSession(req);
  if (gate instanceof Response) return gate;

  try {
    const [rows, known] = await Promise.all([
      query<{ asset_id: string; created_at: string }>(
        'SELECT asset_id, created_at FROM user_cart WHERE user_id = $1 ORDER BY created_at ASC',
        [gate.userId],
      ),
      allCatalogAssets(),
    ]);
    const byId = new Map(known.map((a) => [a.asset_id, a]));
    const body: CartResponse = {
      items: rows.rows.map((r) => {
        const meta = byId.get(r.asset_id);
        return {
          asset_id: r.asset_id,
          symbol: meta?.symbol ?? r.asset_id.toUpperCase(),
          name: meta?.name ?? r.asset_id,
          asset_class: meta?.asset_class ?? null,
          addon_price_month: meta?.addon_price_month ?? null,
          created_at: new Date(r.created_at).toISOString(),
        };
      }),
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
  const asset = known.find((a) => a.asset_id === body.asset_id);
  if (!asset) return fail('NOT_FOUND', `Activo desconocido: ${body.asset_id}`, 404);
  if (asset.status !== 'available') {
    return fail('ASSET_NOT_AVAILABLE', 'Este activo aún no está disponible para compra.', 409);
  }

  const entitlements = await getEntitlements(gate.userId);
  if (entitlements.assets.includes(asset.asset_id)) {
    return fail('ALREADY_ENTITLED', 'Ya tienes este activo desbloqueado.', 409);
  }

  try {
    await query(
      'INSERT INTO user_cart (user_id, asset_id) VALUES ($1, $2) ON CONFLICT DO NOTHING',
      [gate.userId, asset.asset_id],
    );
    return ok({ asset_id: asset.asset_id, added: true });
  } catch {
    return fail('UPSTREAM_UNAVAILABLE', 'Base de datos no disponible.', 502);
  }
}
