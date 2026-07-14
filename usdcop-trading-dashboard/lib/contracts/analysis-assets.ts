/**
 * analysis-assets.contract — SSOT for the assets that have weekly/daily analysis.
 *
 * This is the single source of truth that drives BOTH the frontend asset selector
 * (via GET /api/analysis/assets) and the asset-aware API routing. Adding a new
 * analysed asset = ONE entry here (+ its Python mirror in
 * config/analysis/analysis_assets.yaml) — no per-route or per-component changes.
 *
 * asset_id is the canonical, path-safe key. It must match:
 *   - the registry.json asset_id (usdcop / xauusd / btcusdt)
 *   - the on-disk namespace: public/data/analysis/<asset_id>/...
 *   - the Python AssetProfile key in config/analysis/analysis_assets.yaml
 */

/**
 * How an asset's /forecasting surface is rendered:
 *   - 'model_zoo'         : 9 ML models × 7 horizons (bi_dashboard_unified.csv + PNGs).
 *                           USD/COP (root) and BTC/USDT (btcusdt/ subdir).
 *   - 'weekly_inference'  : rule-based causal weekly positioning JSON (Gold).
 */
export type ForecastMode = 'model_zoo' | 'weekly_inference';

export interface AnalysisAsset {
  /** Canonical, path-safe key (lowercase, no slashes). */
  asset_id: string;
  /** Human pair symbol, e.g. "USD/COP". */
  symbol: string;
  /** Chart/exchange symbol with no slash, e.g. "USDCOP". */
  chart_symbol: string;
  /** Label shown in the selector. */
  display_name: string;
  /** fx | commodity | crypto — used only for grouping/icons. */
  asset_class: string;
  /** /forecasting rendering mode (model-zoo vs weekly-inference). */
  forecast_mode: ForecastMode;
}

/**
 * The analysed assets. USD/COP is the production track; Gold and BTC are the
 * onboarded science stacks. Kept in registry order-of-importance for the selector.
 */
export const ANALYSIS_ASSETS: AnalysisAsset[] = [
  { asset_id: 'usdcop', symbol: 'USD/COP', chart_symbol: 'USDCOP', display_name: 'USD/COP', asset_class: 'fx', forecast_mode: 'model_zoo' },
  { asset_id: 'xauusd', symbol: 'XAU/USD', chart_symbol: 'XAUUSD', display_name: 'Oro (Gold)', asset_class: 'commodity', forecast_mode: 'model_zoo' },
  { asset_id: 'btcusdt', symbol: 'BTC/USDT', chart_symbol: 'BTCUSDT', display_name: 'Bitcoin', asset_class: 'crypto', forecast_mode: 'model_zoo' },
];

/** Default asset when none is specified (backward-compatible with legacy COP-only URLs). */
export const DEFAULT_ANALYSIS_ASSET = 'usdcop';

export const ANALYSIS_ASSET_IDS: string[] = ANALYSIS_ASSETS.map((a) => a.asset_id);

/** Path-safe validator: only known, slug-shaped ids are accepted (prevents traversal). */
export function isValidAnalysisAsset(id: string | null | undefined): id is string {
  return !!id && ANALYSIS_ASSET_IDS.includes(id);
}

/** Normalise an arbitrary query param to a valid asset id, falling back to the default. */
export function resolveAnalysisAsset(id: string | null | undefined): string {
  return isValidAnalysisAsset(id) ? id : DEFAULT_ANALYSIS_ASSET;
}

export function getAnalysisAsset(id: string | null | undefined): AnalysisAsset {
  const resolved = resolveAnalysisAsset(id);
  return ANALYSIS_ASSETS.find((a) => a.asset_id === resolved) ?? ANALYSIS_ASSETS[0];
}
