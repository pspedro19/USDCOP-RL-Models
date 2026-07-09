// Types for Forecasting Dashboard

export interface ForecastRecord {
  record_id: string;
  view_type: 'backtest' | 'forward_forecast';
  model_id: string;
  model_name: string;
  model_type: 'linear' | 'boosting' | 'hybrid';
  horizon_days: number;
  horizon_label: string;
  horizon_category: 'short' | 'medium' | 'long';
  inference_week: string;
  inference_year: number;
  inference_date: string;
  direction_accuracy: number;
  rmse: number;
  mae: number;
  r2: number;
  sharpe: number;
  profit_factor: number;
  max_drawdown: number;
  total_return: number;
  wf_direction_accuracy: number;
  model_avg_direction_accuracy: number;
  model_avg_rmse: number;
  is_best_overall_model: boolean;
  is_best_for_this_horizon: boolean;
  best_da_for_this_horizon: number;
  image_path: string;
  image_backtest: string;
  generated_at: string;
  image_forecast: string;
}

export interface ModelMetrics {
  model_id: string;
  da: number | null;
  sharpe: number | null;
  pf: number | null;
  mdd: number | null;
  totalReturn: number | null;
}

export interface EnsembleVariant {
  value: string;
  label: string;
  imageKey: string;
}

export type ViewType = 'forward_forecast' | 'backtest';
export type HorizonFilter = 'ALL' | string;
export type ModelFilter = 'ALL' | string;

// ============================================================================
// Per-asset Weekly Inference (Gold / BTC rule-based science stacks)
// ----------------------------------------------------------------------------
// USD/COP uses the 9-model ML model-zoo above. Gold & BTC are rule-based daily
// strategies with no ML forecast — their honest "weekly inference" is the
// strategy's causal weekly positioning (direction / exposure / regime) vs what
// actually happened. Produced by scripts/pipeline/generate_asset_weekly_forecast.py
// → public/forecasting/<asset>/weekly_inference_<year>.json.
// Methodology (all pairs): trained on history ≤ Dec-2024, 2025 = backtest (OOS,
// default view), 2026 = production. See .claude/specs/assets/_strategy-science.md.
// ============================================================================

export type WeeklyDirection = 'LONG' | 'SHORT' | 'FLAT';

export interface WeeklyInferenceRecord {
  iso_week: string;          // "2025-W21"
  week_start: string;        // ISO date
  week_end: string;
  direction: WeeklyDirection;
  exposure: number | null;   // 0..1 (normalized to the per-asset cap) — for the bar
  exposure_raw: number | null; // true position magnitude
  regime: string;
  confidence: number | null; // 0..1 conviction proxy
  expected_return_pct: number | null;  // rule-based EDGE PROXY, not an ML prediction
  realized_return_pct: number | null;  // strategy realized that week
  buyhold_return_pct: number | null;   // asset realized that week
  entry_price: number | null;
  close_price: number | null;
  hit: boolean;              // was the week's positioning directionally right
}

export interface WeeklyInferenceSummary {
  weeks_total: number;
  weeks_in_market: number;
  weeks_flat: number;
  hit_rate_pct: number | null;
  ytd_strategy_return_pct: number | null;
  ytd_buyhold_return_pct: number | null;
  avg_exposure: number | null;
}

export interface WeeklyInferenceStrategy {
  strategy_id: string;
  strategy_name: string;
  strategy_type: string;
  is_primary: boolean;
  weeks: WeeklyInferenceRecord[];
  summary: WeeklyInferenceSummary;
}

export interface AssetWeeklyInference {
  asset_id: string;
  display_name: string;
  symbol: string;
  chart_symbol: string;
  asset_class: string;
  year: number;
  generated_at: string;
  kind: string;
  strategies: WeeklyInferenceStrategy[];
}

export interface WeeklyInferenceIndex {
  asset_id: string;
  display_name: string;
  chart_symbol: string;
  years: number[];
  primary_strategy_id: string;
  strategies: { strategy_id: string; strategy_name: string; strategy_type: string }[];
  generated_at: string;
}
