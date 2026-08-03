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
  regime_shift_score?: number;
  regime_action?: 'NORMAL' | 'RETRAIN_REQUIRED' | string;
  eligible_for_signal?: boolean;
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
// USD/COP causal directional replay
// ----------------------------------------------------------------------------
// Produced by scripts/pipeline/generate_usdcop_directional_replay.py. Features
// and model policy are frozen before the replay period; every evidence metric
// is computed only from targets mature at the weekly origin.
// ============================================================================

export type DirectionalPrediction = 'UP' | 'DOWN';
export type DirectionalDecision = DirectionalPrediction | 'FLAT';

export interface DirectionalEvidence {
  n: number;
  directional_accuracy: number | null;
  balanced_accuracy: number | null;
  up_recall: number | null;
  down_recall: number | null;
  minimum_class_recall: number | null;
  brier: number | null;
  prediction_up_rate: number | null;
  actual_up_rate: number | null;
  up_count: number;
  down_count: number;
  raw_score: number | null;
  shrunk_score: number | null;
  eligible: boolean;
}

export interface DirectionalHorizonForecast {
  horizon_days: number;
  role: 'execution' | 'tactical' | 'swing' | string;
  target_date: string;
  target_date_estimated: boolean;
  probability_up: number;
  threshold: number;
  prediction: DirectionalPrediction;
  forecast_log_return: number;
  forecast_return_pct: number;
  forecast_price: number;
  forecast_price_change: number;
  forecast_interval_lower: number;
  forecast_interval_upper: number;
  forecast_interval_level: number;
  point_forecast_direction: DirectionalPrediction;
  direction_price_agree: boolean;
  point_forecast_clipped: boolean;
  point_forecast_validation_eligible: boolean;
  actual: 0 | 1 | null;
  actual_log_return: number | null;
  actual_price: number | null;
  point_abs_error_price: number | null;
  point_abs_error_pct: number | null;
  hit: boolean | null;
  train_count: number;
  train_label_end: string | null;
  point_train_label_end: string | null;
  training_mode: string;
  model_family: string;
  feature_hash: string;
  validation_eligible: boolean;
  evidence: DirectionalEvidence;
  eligible_for_direction: boolean;
  selected: boolean;
  selection_rank: number;
}

export interface DirectionalReplayDecision {
  direction: DirectionalDecision;
  action: 'LONG_USD_SHORT_COP' | 'SHORT_USD_LONG_COP' | 'FLAT';
  status: 'SHADOW' | 'ABSTAIN';
  signal_authorized: false;
  promotion_gate_passed: boolean;
  primary_horizon: number | null;
  confirmation_horizons: number[];
  selected_horizons: number[];
  confidence_proxy: number;
  target_date: string | null;
  forecast_price: number | null;
  forecast_return_pct: number | null;
  forecast_interval_lower: number | null;
  forecast_interval_upper: number | null;
  actual: 0 | 1 | null;
  hit: boolean | null;
  rationale: string;
}

export interface DirectionalReplayWeek {
  iso_week: string;
  year: number;
  origin_date: string;
  origin_is_partial_week: boolean;
  base_price: number;
  training_mode: string;
  regime: { state: string; direction_shift_z: number };
  decision: DirectionalReplayDecision;
  horizons: DirectionalHorizonForecast[];
  image_path: string;
}

export interface DirectionalPointMetrics {
  n: number;
  mae_log_return: number | null;
  rmse_log_return: number | null;
  naive_mae_log_return: number | null;
  naive_rmse_log_return: number | null;
  mae_skill_vs_spot: number | null;
  rmse_skill_vs_spot: number | null;
  mae_price: number | null;
  mape_price: number | null;
  directional_accuracy: number | null;
  balanced_accuracy: number | null;
  up_recall: number | null;
  down_recall: number | null;
  minimum_class_recall: number | null;
  bias_log_return: number | null;
}

export interface DirectionalHorizonMetrics {
  horizon_days: number;
  n: number;
  directional_accuracy: number | null;
  balanced_accuracy: number | null;
  up_recall: number | null;
  down_recall: number | null;
  minimum_class_recall: number | null;
  brier: number | null;
  prediction_up_rate: number | null;
  actual_up_rate: number | null;
  up_count: number;
  down_count: number;
  point_forecast: DirectionalPointMetrics;
}

export interface DirectionalYearSummary {
  year: number;
  weeks_total: number;
  shadow_decisions: number;
  abstentions: number;
  coverage: number;
  matured_decisions: number;
  pending_decisions: number;
  decision_metrics: Omit<DirectionalHorizonMetrics, 'horizon_days' | 'point_forecast'>;
  horizon_metrics: DirectionalHorizonMetrics[];
}

export interface DirectionalReplayIndex {
  schema_version: string;
  contract_hash: string;
  asset_id: 'usdcop';
  symbol: 'USD/COP';
  chart_symbol: 'USDCOP';
  generated_at: string;
  data_cutoff: string;
  latest_week: string;
  years: number[];
  methodology: {
    experiment_id: string;
    status: string;
    signal_authorized: false;
    feature_selection_cutoff: string;
    validation_years: number[];
    frozen_replay_year: number;
    expanding_retrain_year: number;
    label_maturity_rule: string;
    horizon_selection: string;
  };
  models: Record<string, {
    horizon_days: number;
    half_life: string;
    threshold: number;
    validation_score: number;
    validation_eligible: boolean;
    feature_hash: string;
    selected_features: string[];
  }>;
  summaries: DirectionalYearSummary[];
  weeks: DirectionalReplayWeek[];
}

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
