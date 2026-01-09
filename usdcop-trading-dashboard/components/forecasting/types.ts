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
