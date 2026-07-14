-- ============================================================================
-- 057_catalog_watchlist_cart.sql — user watchlist + cart (CTR-FE-BE-001 §4.3)
-- ============================================================================
-- Backs the Catálogo · Watchlist · Carrito module:
--   GET/POST /api/watchlist, DELETE /api/watchlist/{asset_id}
--   GET/POST /api/cart,      DELETE /api/cart/{asset_id}, POST /api/cart/checkout
--
-- asset_id maps 1:1 to the multi-asset registry (public/data/registry.json:
-- 'usdcop' | 'xauusd' | 'btcusdt' | …) plus catalog coming-soon statics
-- ('spx500', 'eurusd'). Validation against the registry happens in the BFF
-- handlers — the DB stays schema-only (the registry is file-based, not a table).
--
-- Additive & idempotent (IF NOT EXISTS everywhere). Never touches strategy logic.
-- Date: 2026-07-10
-- ============================================================================

-- 1) Watchlist — assets a user follows (read-only, free; no entitlement change)
CREATE TABLE IF NOT EXISTS user_watchlist (
    user_id     UUID         NOT NULL REFERENCES sb_users(id) ON DELETE CASCADE,
    asset_id    VARCHAR(32)  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, asset_id)
);

COMMENT ON TABLE user_watchlist IS
    'CTR-FE-BE-001 §4.3: per-user asset watchlist (registry asset_id). '
    'Free feature — grants NO access; entitlements stay in sb_users.entitlements.';

-- PK covers (user_id, asset_id) lookups; extra index for per-asset analytics.
CREATE INDEX IF NOT EXISTS idx_user_watchlist_asset ON user_watchlist (asset_id);

-- 2) Cart — add-on assets pending checkout (plan is chosen at checkout time,
--    the provider webhook is the truth for entitlements — CTR-RBAC-001 rule 7).
CREATE TABLE IF NOT EXISTS user_cart (
    user_id     UUID         NOT NULL REFERENCES sb_users(id) ON DELETE CASCADE,
    asset_id    VARCHAR(32)  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, asset_id)
);

COMMENT ON TABLE user_cart IS
    'CTR-FE-BE-001 §4.3: per-user add-on cart. Checkout forwards to the billing '
    'provider; entitlements are ONLY updated by the payment webhook, never here.';

CREATE INDEX IF NOT EXISTS idx_user_cart_asset ON user_cart (asset_id);
