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
/**
 * `none` NO es un tercer modo de render: declara que el activo **no tiene superficie de
 * forecasting publicada**. Se añadió el 2026-08-05 tras medir que `spx500` estaba declarado
 * `model_zoo` con CERO artefactos en `public/forecasting/spx500/`, así que el selector lo
 * ofrecía y su `csvPath` apuntaba a un fichero inexistente. La alternativa —sacarlo de
 * `ANALYSIS_ASSETS`— no vale: esta lista es SSOT compartida con /analysis (news-feed,
 * weekly analysis), y ahí spx500 sí participa. El contrato declara la ausencia; no la oculta.
 */
export type ForecastMode = 'model_zoo' | 'weekly_inference' | 'none';

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
  asset_class: 'fx' | 'commodity' | 'crypto' | 'equity_index';
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
  // spx500: `none` porque public/forecasting/spx500/ está VACÍO (medido 2026-08-05, 0 ficheros
  // frente a los 459 de xauusd y btcusdt). Sigue en la lista porque /analysis sí lo cubre.
  { asset_id: 'spx500', symbol: 'SPX500', chart_symbol: 'SPX500', display_name: 'S&P 500', asset_class: 'equity_index', forecast_mode: 'none' },
];
// NOTA deliberada: NO se exporta aquí una lista `FORECASTING_ASSETS` ya filtrada. Se intentó
// (2026-08-05) y lo rechazó `forecasting-weekly-branch.test.tsx`: ese test sustituye este módulo
// entero con `vi.mock`, así que CADA export nuevo obliga a actualizar todos los mocks o revienta
// con "No X export is defined on the mock". La verdad vive en el DATO (`forecast_mode: 'none'`)
// y cada vista filtra por él; la superficie del módulo se mantiene mínima a propósito.

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
