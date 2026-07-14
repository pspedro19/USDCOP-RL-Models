/**
 * GET /api/catalog — full catalog per category (CTR-FE-BE-001 §4.3).
 *
 * Composes, server-side:
 *   registry SSOT (public/data/registry.json → lib/services/catalog-registry)
 *   + user entitlements (lib/auth/entitlements — DB truth, JWT is a cache)
 *   + user watchlist (user_watchlist, migration 057)
 *   + live spot price for the supported FX symbols (joins the public
 *     /api/public/market-price endpoint — no logic duplicated)
 * into the spec's `CatalogResponse`. price/change_pct come from the live FX quote
 * for USD/COP·MXN·BRL; crypto/commodity stay null (no source ⇒ honest "—").
 * addon_price_month comes from the billing SSOT via catalog-registry.
 * Coming-soon teasers are marked statics.
 *
 * Auth: 'authenticated' in the RBAC matrix; identity via middleware headers.
 * Watchlist read AND the price join are best-effort — a DB/upstream hiccup
 * degrades to null price / in_watchlist=false, never failing the catalog.
 */
import { ok } from '@/lib/api/envelope';
import { requireSession } from '@/lib/api/relay';
import { getEntitlements } from '@/lib/auth/entitlements';
import { query } from '@/lib/db/postgres-client';
import {
  CATALOG_CATEGORY_LABELS,
  type CatalogCategoryId,
  type CatalogResponse,
} from '@/lib/contracts/catalog.contract';
import { allCatalogAssets } from '@/lib/services/catalog-registry';

export const dynamic = 'force-dynamic';

const CATEGORY_ORDER: readonly CatalogCategoryId[] = ['fx', 'crypto', 'equity_index', 'commodity'];

/** FX symbols the public market-price endpoint serves a live quote for. */
const LIVE_FX_SYMBOLS = new Set(['USD/COP', 'USD/MXN', 'USD/BRL']);

interface LivePrice { price: number | null; change_pct: number | null }
const NO_PRICE: LivePrice = { price: null, change_pct: null };

/** Fetch one live FX quote via the public endpoint. Fault-tolerant: any failure
 *  (unsupported/unavailable/upstream) ⇒ null price, never throws. */
async function fetchLivePrice(origin: string, symbol: string): Promise<LivePrice> {
  try {
    const res = await fetch(
      `${origin}/api/public/market-price?symbol=${encodeURIComponent(symbol)}`,
      { cache: 'no-store' },
    );
    if (!res.ok) return NO_PRICE;
    const j = (await res.json()) as {
      unavailable?: boolean; price?: unknown; changePercent?: unknown;
    };
    if (j?.unavailable) return NO_PRICE;
    return {
      price: typeof j.price === 'number' ? j.price : null,
      change_pct: typeof j.changePercent === 'number' ? j.changePercent : null,
    };
  } catch {
    return NO_PRICE;
  }
}

/** Live quotes for every supported FX symbol present in the catalog. Whole thing
 *  is best-effort — a total failure yields an empty map (catalog still renders). */
async function fetchCatalogPrices(
  origin: string, symbols: Iterable<string>,
): Promise<Map<string, LivePrice>> {
  try {
    const unique = [...new Set(symbols)].filter((s) => LIVE_FX_SYMBOLS.has(s));
    const entries = await Promise.all(
      unique.map(async (s) => [s, await fetchLivePrice(origin, s)] as const),
    );
    return new Map(entries);
  } catch {
    return new Map();
  }
}

export async function GET(req: Request) {
  const gate = requireSession(req);
  if (gate instanceof Response) return gate;

  const [assets, entitlements, watchlist] = await Promise.all([
    allCatalogAssets(),
    getEntitlements(gate.userId),
    query<{ asset_id: string }>(
      'SELECT asset_id FROM user_watchlist WHERE user_id = $1', [gate.userId],
    ).then((r) => new Set(r.rows.map((x) => x.asset_id))).catch(() => new Set<string>()),
  ]);

  const origin = new URL(req.url).origin;
  const priceMap = await fetchCatalogPrices(
    origin,
    assets.filter((a) => a.status === 'available').map((a) => a.symbol),
  );

  const catalogAssets = assets
    .map((a) => {
      const live = priceMap.get(a.symbol) ?? NO_PRICE;
      return {
        asset_id: a.asset_id,
        symbol: a.symbol,
        name: a.name,
        asset_class: a.asset_class,
        status: a.status,
        // Live FX quote where available; crypto/commodity have no source ⇒ null ("—").
        price: a.status === 'available' ? live.price : null,
        change_pct: a.status === 'available' ? live.change_pct : null,
        addon_price_month: a.status === 'coming_soon' ? null : a.addon_price_month,
        entitled: a.status === 'available' && entitlements.assets.includes(a.asset_id),
        in_watchlist: watchlist.has(a.asset_id),
      };
    })
    .sort((x, y) =>
      x.status === y.status ? x.symbol.localeCompare(y.symbol) : x.status === 'available' ? -1 : 1);

  const body: CatalogResponse = {
    categories: CATEGORY_ORDER.map((id) => ({
      id,
      label: CATALOG_CATEGORY_LABELS[id],
      count: catalogAssets.filter((a) => a.asset_class === id).length,
    })),
    assets: catalogAssets,
  };

  return ok(body, { meta: { asOf: new Date().toISOString() } });
}
