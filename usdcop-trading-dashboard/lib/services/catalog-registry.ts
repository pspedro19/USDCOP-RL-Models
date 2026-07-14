/**
 * Server-side catalog source (CTR-FE-BE-001 §4.3) — the ONE place that turns the
 * registry SSOT (`public/data/registry.json`, produced by the Python
 * RegistryBuilder) into catalog entries, plus the marked-static "coming soon"
 * teasers. Consumed by /api/catalog, /api/watchlist and /api/cart handlers so
 * asset lists are never duplicated (§4.3 / CLAUDE.md SSOT rule).
 *
 * Server-only (fs): do NOT import from client components.
 */
import { promises as fs } from 'fs';
import path from 'path';

import { addonPriceCop } from '@/lib/billing/prices';
import type { RegistryIndex } from '@/lib/contracts/strategy-manifest.contract';
import type { CatalogCategoryId } from '@/lib/contracts/catalog.contract';

const REGISTRY_FILE = path.join(process.cwd(), 'public', 'data', 'registry.json');

export interface CatalogSourceAsset {
  asset_id: string;
  symbol: string;
  name: string;
  asset_class: CatalogCategoryId;
  status: 'available' | 'coming_soon';
  /** COP/month add-on price from the billing SSOT (lib/billing/prices), or null. */
  addon_price_month: number | null;
}

/**
 * Static roadmap teasers (§4.3 "coming_soon"): NOT in the registry, NOT
 * purchasable, NOT entitle-able — cards render with a lock. Watchlist-able so
 * demand is measurable. Keep this list tiny and explicit.
 */
export const COMING_SOON_ASSETS: readonly CatalogSourceAsset[] = [
  // Forex
  {
    asset_id: 'eurusd', symbol: 'EUR/USD', name: 'Euro / Dólar',
    asset_class: 'fx', status: 'coming_soon', addon_price_month: null,
  },
  {
    asset_id: 'gbpusd', symbol: 'GBP/USD', name: 'Libra / Dólar',
    asset_class: 'fx', status: 'coming_soon', addon_price_month: null,
  },
  // Cripto
  {
    asset_id: 'ethusdt', symbol: 'ETH/USDT', name: 'Ethereum',
    asset_class: 'crypto', status: 'coming_soon', addon_price_month: null,
  },
  {
    asset_id: 'solusdt', symbol: 'SOL/USDT', name: 'Solana',
    asset_class: 'crypto', status: 'coming_soon', addon_price_month: null,
  },
  // Acciones / índices
  {
    asset_id: 'spx500', symbol: 'SPX500', name: 'S&P 500',
    asset_class: 'equity_index', status: 'coming_soon', addon_price_month: null,
  },
  {
    asset_id: 'ndx100', symbol: 'NDX100', name: 'Nasdaq 100',
    asset_class: 'equity_index', status: 'coming_soon', addon_price_month: null,
  },
  // Materias primas
  {
    asset_id: 'xagusd', symbol: 'XAG/USD', name: 'Plata',
    asset_class: 'commodity', status: 'coming_soon', addon_price_month: null,
  },
  {
    asset_id: 'wti_oil', symbol: 'WTI', name: 'Petróleo WTI',
    asset_class: 'commodity', status: 'coming_soon', addon_price_month: null,
  },
] as const;

const VALID_CLASSES: readonly CatalogCategoryId[] = ['fx', 'crypto', 'equity_index', 'commodity'];

function normalizeClass(c: string | undefined): CatalogCategoryId {
  return (VALID_CLASSES as readonly string[]).includes(c ?? '') ? (c as CatalogCategoryId) : 'fx';
}

/** Registry assets as catalog entries (status 'available'). Missing/corrupt
 *  registry ⇒ empty list (graceful degradation, same policy as /api/registry). */
export async function readRegistryCatalogAssets(): Promise<CatalogSourceAsset[]> {
  try {
    const idx = JSON.parse(await fs.readFile(REGISTRY_FILE, 'utf-8')) as RegistryIndex;
    return (idx.assets ?? []).map((a) => {
      // Registry may one day publish its own price (additive field); otherwise the
      // billing SSOT (lib/billing/prices) is authoritative. Never a hardcoded literal.
      const registryPrice = (a as { addon_price_month?: unknown }).addon_price_month;
      return {
        asset_id: a.asset_id,
        symbol: a.symbol,
        name: a.display_name,
        asset_class: normalizeClass(a.asset_class),
        status: 'available' as const,
        addon_price_month:
          typeof registryPrice === 'number' ? registryPrice : addonPriceCop(a.asset_id),
      };
    });
  } catch {
    return [];
  }
}

/** Registry + coming-soon statics, deduped by asset_id (registry wins). */
export async function allCatalogAssets(): Promise<CatalogSourceAsset[]> {
  const registry = await readRegistryCatalogAssets();
  const seen = new Set(registry.map((a) => a.asset_id));
  return [...registry, ...COMING_SOON_ASSETS.filter((a) => !seen.has(a.asset_id))];
}

const ASSET_ID_RE = /^[a-z0-9_]{1,32}$/;

/** Shape gate for user-supplied asset_id (before any DB round-trip). */
export function isValidAssetIdShape(id: unknown): id is string {
  return typeof id === 'string' && ASSET_ID_RE.test(id);
}
