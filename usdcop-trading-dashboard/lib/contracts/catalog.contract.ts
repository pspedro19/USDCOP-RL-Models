/**
 * Catálogo · Watchlist · Carrito contract (CTR-FE-BE-001 §4.3).
 *
 * `CatalogResponse` is the exact shape of `GET /api/catalog` from the spec —
 * composed server-side from the registry SSOT (`public/data/registry.json`) +
 * the user's entitlements (`lib/auth/entitlements.ts`) + their watchlist rows.
 * The frontend NEVER duplicates asset lists: Hub strip, catalog grid and cart
 * drawer all derive from this contract.
 *
 * Category mapping (spec §4.3): asset_class → UI label
 *   fx→Forex · crypto→Cripto · equity_index→Acciones · commodity→Materias primas
 */
import type { PlanId } from './rbac.contract';

export type CatalogCategoryId = 'fx' | 'crypto' | 'equity_index' | 'commodity';

export const CATALOG_CATEGORY_LABELS: Record<CatalogCategoryId, string> = {
  fx: 'Forex',
  crypto: 'Cripto',
  equity_index: 'Acciones',
  commodity: 'Materias primas',
};

export interface CatalogCategory {
  id: CatalogCategoryId;
  label: string;
  count: number;
}

export interface CatalogAsset {
  asset_id: string;            // 'usdcop' | 'btcusdt' | 'spx500' | 'xauusd' | …
  symbol: string;              // 'USD/COP'
  name: string;                // 'Peso colombiano'
  asset_class: CatalogCategoryId;
  status: 'available' | 'coming_soon';
  price: number | null;        // último precio (si available) — sin fuente aún ⇒ null
  change_pct: number | null;
  addon_price_month: number | null;  // COP; null si incluido/próximamente
  entitled: boolean;           // si el usuario ya lo tiene desbloqueado
  in_watchlist: boolean;
}

/** GET /api/catalog → 200 (envelope `data`) */
export interface CatalogResponse {
  categories: CatalogCategory[];
  assets: CatalogAsset[];
}

// ── watchlist ────────────────────────────────────────────────────────────────

export interface WatchlistItem {
  asset_id: string;
  created_at: string; // ISO
}

/** GET /api/watchlist → 200 (envelope `data`) */
export interface WatchlistResponse {
  items: WatchlistItem[];
}

// ── cart ─────────────────────────────────────────────────────────────────────

/** Cart rows come back ENRICHED with registry metadata so the drawer renders
 *  without a second fetch (single asset-list source: the catalog module). */
export interface CartItem {
  asset_id: string;
  symbol: string;
  name: string;
  asset_class: CatalogCategoryId | null;
  addon_price_month: number | null;
  created_at: string; // ISO
}

/** GET /api/cart → 200 (envelope `data`) */
export interface CartResponse {
  items: CartItem[];
}

/** POST /api/cart/checkout body — plan elegido en el drawer/página de carrito. */
export interface CartCheckoutRequest {
  plan: 'signals' | 'auto';
}

/** POST /api/cart/checkout → 200 (envelope `data`) — mirror of the billing
 *  provider's CheckoutSession, snake_cased for the wire. */
export interface CartCheckoutResponse {
  provider: string;
  checkout_url: string;
  reference: string;
  /** Add-on asset_ids that were sent to the provider (cart minus already-entitled). */
  addon_assets: string[];
}

/** DOM event fired after any cart mutation so decoupled surfaces (TerminalShell
 *  badge, CatalogView buttons, CartDrawer) refresh without prop-drilling. */
export const CART_CHANGED_EVENT = 'gm:cart-changed';

// ── billing prices (public) ────────────────────────────────────────────────────

/** One plan's public price (COP). Derived server-side from the lib/billing/prices
 *  SSOT — the SAME numbers that drive the Wompi charge and admin revenue. */
export interface PlanPriceInfo {
  plan: PlanId;
  label: string;              // PLAN_LABELS
  price_month_cop: number;    // whole COP; 0 for free
  price_month_cents: number;  // COP cents (charge granularity)
}

/** GET /api/billing/prices → 200 (envelope `data`). Public, readable pre-login. */
export interface BillingPricesResponse {
  currency: 'COP';
  plans: PlanPriceInfo[];
  /** asset_id → add-on monthly price (whole COP). Only published assets appear. */
  addons: Record<string, number>;
}
