/**
 * Forecasting API Contracts
 * =========================
 *
 * Zod schemas for forecasting endpoints.
 * MUST match backend Pydantic schemas in services/inference_api/contracts/forecasting.py
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';

// ============================================================================
// ENUMS
// ============================================================================

export const ForecastDirectionSchema = z.enum(['UP', 'DOWN']);
export type ForecastDirection = z.infer<typeof ForecastDirectionSchema>;

export const HorizonCategorySchema = z.enum(['short', 'medium', 'long']);
export type HorizonCategory = z.infer<typeof HorizonCategorySchema>;

export const ModelTypeSchema = z.enum(['linear', 'boosting', 'hybrid']);
export type ModelType = z.infer<typeof ModelTypeSchema>;

// ============================================================================
// MODEL SCHEMAS
// ============================================================================

export const ModelInfoSchema = z.object({
  model_id: z.string(),
  model_name: z.string(),
  model_type: ModelTypeSchema,
  requires_scaling: z.boolean().default(false),
  supports_early_stopping: z.boolean().default(false),
  is_active: z.boolean().default(true),
  // Optional extended fields
  avg_direction_accuracy: z.number().optional(),
  avg_rmse: z.number().optional(),
  horizons: z.array(z.number()).optional(),
});
export type ModelInfo = z.infer<typeof ModelInfoSchema>;

export const ModelMetricsSchema = z.object({
  model_id: z.string(),
  horizon_id: z.number().int(),
  direction_accuracy: z.number().min(0).max(100),
  rmse: z.number().nonnegative(),
  mae: z.number().nonnegative().nullable().optional(),
  mape: z.number().nullable().optional(),
  r2: z.number().nullable().optional(),
  sample_count: z.number().int().nonnegative(),
});
export type ModelMetrics = z.infer<typeof ModelMetricsSchema>;

export const ModelComparisonSchema = z.object({
  model_id: z.string(),
  is_selected: z.boolean(),
  avg_direction_accuracy: z.number().optional(),
  avg_rmse: z.number().optional(),
  rank: z.number().int().positive(),
});
export type ModelComparison = z.infer<typeof ModelComparisonSchema>;

export const ModelListResponseSchema = z.object({
  models: z.array(ModelInfoSchema),
  count: z.number().int().nonnegative(),
});
export type ModelListResponse = z.infer<typeof ModelListResponseSchema>;

export const ModelDetailResponseSchema = z.object({
  model: ModelInfoSchema,
  metrics_by_horizon: z.array(ModelMetricsSchema),
  total_forecasts: z.number().int().nonnegative(),
});
export type ModelDetailResponse = z.infer<typeof ModelDetailResponseSchema>;

export const ModelRankingResponseSchema = z.object({
  metric: z.string(),
  horizon: z.number().int().nullable(),
  rankings: z.array(ModelComparisonSchema),
});
export type ModelRankingResponse = z.infer<typeof ModelRankingResponseSchema>;

// ============================================================================
// FORECAST SCHEMAS
// ============================================================================

export const ForecastSchema = z.object({
  id: z.string().nullable().optional(),
  inference_date: z.string(), // ISO date string
  inference_week: z.number().int().min(1).max(53).optional(),
  inference_year: z.number().int().min(2020).max(2100).optional(),
  target_date: z.string().optional(),
  model_id: z.string().optional(),
  model_name: z.string().optional(), // CSV uses model_name
  horizon_id: z.number().int().optional(),
  horizon: z.number().int().optional(), // CSV uses horizon
  base_price: z.number().positive().optional(),
  predicted_price: z.number().positive().optional(),
  predicted_return_pct: z.number().nullable().optional(),
  price_change: z.number().nullable().optional(),
  price_change_pct: z.number().nullable().optional(),
  direction: ForecastDirectionSchema.optional(),
  signal: z.number().int().min(-1).max(1).optional(),
  confidence: z.number().min(0).max(1).nullable().optional(),
  actual_price: z.number().nullable().optional(),
  direction_correct: z.boolean().nullable().optional(),
  // Metrics fields (from CSV)
  direction_accuracy: z.number().optional(),
  rmse: z.number().optional(),
  mae: z.number().optional(),
  // Image fields
  image_backtest: z.string().nullable().optional(),
  image_forecast: z.string().nullable().optional(),
});
export type Forecast = z.infer<typeof ForecastSchema>;

export const ForecastListResponseSchema = z.object({
  source: z.string(),
  count: z.number().int().nonnegative(),
  data: z.array(ForecastSchema),
});
export type ForecastListResponse = z.infer<typeof ForecastListResponseSchema>;

// ============================================================================
// CONSENSUS SCHEMAS
// ============================================================================

export const ConsensusSchema = z.object({
  inference_date: z.string().optional(),
  horizon_id: z.number().int(),
  horizon_label: z.string().nullable().optional(),
  avg_predicted_price: z.number().optional(),
  median_predicted_price: z.number().nullable().optional(),
  std_predicted_price: z.number().nullable().optional(),
  min_predicted_price: z.number().nullable().optional(),
  max_predicted_price: z.number().nullable().optional(),
  consensus_direction: ForecastDirectionSchema.optional(),
  bullish_count: z.number().int().nonnegative(),
  bearish_count: z.number().int().nonnegative(),
  total_models: z.number().int().nonnegative(),
  agreement_pct: z.number().nullable().optional(),
});
export type Consensus = z.infer<typeof ConsensusSchema>;

export const ConsensusResponseSchema = z.object({
  source: z.string(),
  count: z.number().int().nonnegative(),
  data: z.array(ConsensusSchema),
});
export type ConsensusResponse = z.infer<typeof ConsensusResponseSchema>;

// ============================================================================
// DASHBOARD SCHEMAS
// ============================================================================

export const DashboardResponseSchema = z.object({
  source: z.string(),
  forecasts: z.array(z.record(z.any())),
  consensus: z.array(z.record(z.any())),
  metrics: z.array(z.record(z.any())),
  last_update: z.string(),
  error: z.string().optional(),
});
export type DashboardResponse = z.infer<typeof DashboardResponseSchema>;

// ============================================================================
// IMAGE SCHEMAS
// ============================================================================

export const ImageMetadataSchema = z.object({
  image_type: z.string(),
  model_id: z.string(),
  horizon_id: z.number().int().nullable().optional(),
  filename: z.string(),
  url: z.string(),
  size: z.number().nullable().optional(),
  last_modified: z.string().nullable().optional(),
});
export type ImageMetadata = z.infer<typeof ImageMetadataSchema>;

export const ImageListResponseSchema = z.object({
  images: z.array(ImageMetadataSchema),
  count: z.number().int().nonnegative(),
});
export type ImageListResponse = z.infer<typeof ImageListResponseSchema>;

// ============================================================================
// API ENDPOINTS
// ============================================================================

export const FORECASTING_API_BASE = process.env.NEXT_PUBLIC_INFERENCE_API_URL || 'http://localhost:8000';

export const FORECASTING_ENDPOINTS = {
  // Forecasts
  FORECASTS: '/api/v1/forecasting/forecasts',
  FORECASTS_LATEST: '/api/v1/forecasting/forecasts/latest',
  FORECASTS_CONSENSUS: '/api/v1/forecasting/forecasts/consensus',
  FORECASTS_BY_WEEK: (year: number, week: number) =>
    `/api/v1/forecasting/forecasts/by-week/${year}/${week}`,
  FORECASTS_BY_HORIZON: (horizon: number) =>
    `/api/v1/forecasting/forecasts/by-horizon/${horizon}`,

  // Models
  MODELS: '/api/v1/forecasting/models',
  MODEL_DETAIL: (modelId: string) => `/api/v1/forecasting/models/${modelId}`,
  MODEL_COMPARISON: (modelId: string) => `/api/v1/forecasting/models/${modelId}/comparison`,
  MODEL_RANKING: '/api/v1/forecasting/models/ranking',

  // Images
  IMAGES: '/api/v1/forecasting/images',
  IMAGE_BACKTEST: (model: string, horizon: number) =>
    `/api/v1/forecasting/images/backtest/${model}/${horizon}`,
  IMAGE_FORECAST: (model: string) =>
    `/api/v1/forecasting/images/forecast/${model}`,
  IMAGE_HEATMAP: (model: string) =>
    `/api/v1/forecasting/images/heatmap/${model}`,

  // Dashboard
  DASHBOARD: '/api/v1/forecasting/dashboard',
  HEALTH: '/api/v1/forecasting/health',
} as const;

// ============================================================================
// HELPER TYPES
// ============================================================================

export interface ForecastQueryParams {
  model?: string;
  horizon?: number;
  week?: number;
  year?: number;
  limit?: number;
}

export const HORIZONS = [1, 5, 10, 15, 20, 25, 30] as const;
export type Horizon = typeof HORIZONS[number];

export const HORIZON_LABELS: Record<number, string> = {
  1: '1 day',
  5: '5 days',
  10: '10 days',
  15: '15 days',
  20: '20 days',
  25: '25 days',
  30: '30 days',
};

export const MODEL_TYPE_COLORS: Record<ModelType, string> = {
  linear: 'bg-blue-500/20 text-blue-400',
  boosting: 'bg-green-500/20 text-green-400',
  hybrid: 'bg-purple-500/20 text-purple-400',
};

export const DIRECTION_COLORS = {
  UP: 'text-green-400',
  DOWN: 'text-red-400',
} as const;
