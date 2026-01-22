/**
 * Contratos Zod para Frontend - USD/COP Forecasting
 *
 * IMPORTANTE: Estos schemas deben coincidir EXACTAMENTE con los
 * schemas Pydantic del backend (api/schemas/*.py)
 *
 * Single Source of Truth: Backend Pydantic → Frontend Zod
 */

import { z } from 'zod';

// =============================================================================
// AUTH CONTRACTS
// =============================================================================

export const LoginRequestSchema = z.object({
  username: z.string().min(1, 'Username es requerido'),
  password: z.string().min(1, 'Password es requerido'),
});

export const TokenResponseSchema = z.object({
  access_token: z.string(),
  token_type: z.literal('bearer').default('bearer'),
  expires_at: z.string().datetime(),
});

export const UserResponseSchema = z.object({
  id: z.number().int(),
  username: z.string(),
  email: z.string().email(),
  is_active: z.boolean(),
  created_at: z.string().datetime(),
});

// =============================================================================
// FORECAST CONTRACTS
// =============================================================================

export const ForecastResponseSchema = z.object({
  /** Nombre del modelo (ridge, hybrid_xgboost, etc) */
  model: z.string(),
  /** Horizonte en días (1, 5, 10, 15, 20, 25, 30) */
  horizon: z.number().int().min(1).max(30),
  /** Predicción de log-return */
  prediction: z.number(),
  /** Precio predicho (opcional) */
  prediction_price: z.number().optional().nullable(),
  /** Dirección predicha */
  direction: z.enum(['up', 'down']),
  /** Confianza [0-1] */
  confidence: z.number().min(0).max(1),
  /** Fecha objetivo de la predicción */
  prediction_date: z.string(),
  /** Timestamp de generación */
  generated_at: z.string().datetime(),
});

export const ForecastListResponseSchema = z.object({
  forecasts: z.array(ForecastResponseSchema),
  total: z.number().int(),
  page: z.number().int().default(1),
  page_size: z.number().int().default(50),
});

export const ConsensusResponseSchema = z.object({
  /** Predicción de consenso (promedio ponderado) */
  consensus_prediction: z.number(),
  consensus_direction: z.enum(['up', 'down']),
  confidence: z.number(),
  models_used: z.array(z.string()),
  weights: z.record(z.string(), z.number()),
  horizon: z.number().int(),
  generated_at: z.string().datetime(),
});

export const ForecastDashboardSchema = z.object({
  latest_forecasts: z.array(ForecastResponseSchema),
  consensus: ConsensusResponseSchema,
  best_model: z.string(),
  best_model_da: z.number(),
  avg_da_all_models: z.number(),
  last_actual_price: z.number(),
  last_actual_date: z.string(),
  generated_at: z.string().datetime(),
});

// =============================================================================
// MODEL CONTRACTS
// =============================================================================

export const ModelMetricsSchema = z.object({
  model_name: z.string(),
  model_type: z.enum(['linear', 'boosting', 'hybrid']),
  horizon: z.number().int(),
  /** Direction Accuracy train (%) */
  da_train: z.number(),
  /** Direction Accuracy test (%) */
  da_test: z.number(),
  /** pred_std / true_std */
  variance_ratio: z.number(),
  rmse: z.number().optional().nullable(),
  mae: z.number().optional().nullable(),
  r2: z.number().optional().nullable(),
  sharpe: z.number().optional().nullable(),
  max_drawdown: z.number().optional().nullable(),
  /** da_train - da_test */
  overfitting_gap: z.number(),
});

export const ModelSummarySchema = z.object({
  model_name: z.string(),
  model_type: z.string(),
  avg_da_test: z.number(),
  best_horizon: z.number().int(),
  best_da: z.number(),
  requires_scaling: z.boolean(),
  supports_early_stopping: z.boolean(),
});

export const ModelComparisonSchema = z.object({
  models: z.array(ModelSummarySchema),
  /** Métricas agrupadas por horizonte */
  metrics_by_horizon: z.record(z.string(), z.array(ModelMetricsSchema)),
  best_model_overall: z.string(),
  best_model_by_horizon: z.record(z.string(), z.string()),
  generated_at: z.string().datetime(),
});

export const FeatureImportanceSchema = z.object({
  model_name: z.string(),
  horizon: z.number().int(),
  /** {feature_name: importance_score} */
  features: z.record(z.string(), z.number()),
  top_n: z.number().int().default(15),
});

// =============================================================================
// IMAGE CONTRACTS
// =============================================================================

export const ImageMetadataSchema = z.object({
  image_id: z.string(),
  image_type: z.enum([
    'model_ranking',
    'metrics_heatmap',
    'backtest_equity',
    'backtest_drawdown',
    'forecast_fan_chart',
    'feature_importance',
  ]),
  model_name: z.string().optional().nullable(),
  horizon: z.number().int().optional().nullable(),
  generated_at: z.string().datetime(),
  file_size_bytes: z.number().int(),
  dimensions: z.tuple([z.number().int(), z.number().int()]),
});

export const ImageListResponseSchema = z.object({
  images: z.array(ImageMetadataSchema),
  total: z.number().int(),
});

// =============================================================================
// HEALTH CONTRACTS
// =============================================================================

export const ServiceHealthSchema = z.object({
  name: z.string(),
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  latency_ms: z.number().optional().nullable(),
  message: z.string().optional().nullable(),
});

export const HealthResponseSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  version: z.string(),
  environment: z.string(),
  services: z.record(z.string(), ServiceHealthSchema),
  timestamp: z.string().datetime(),
});

// =============================================================================
// ERROR CONTRACTS
// =============================================================================

export const ErrorDetailSchema = z.object({
  field: z.string().optional().nullable(),
  message: z.string(),
  code: z.string(),
});

export const ErrorResponseSchema = z.object({
  error: z.string(),
  message: z.string(),
  details: z.array(ErrorDetailSchema).optional().nullable(),
  status_code: z.number().int(),
  timestamp: z.string().datetime(),
});

// =============================================================================
// TYPE EXPORTS (Inferidos de los schemas)
// =============================================================================

// Auth
export type LoginRequest = z.infer<typeof LoginRequestSchema>;
export type TokenResponse = z.infer<typeof TokenResponseSchema>;
export type UserResponse = z.infer<typeof UserResponseSchema>;

// Forecasts
export type ForecastResponse = z.infer<typeof ForecastResponseSchema>;
export type ForecastListResponse = z.infer<typeof ForecastListResponseSchema>;
export type ConsensusResponse = z.infer<typeof ConsensusResponseSchema>;
export type ForecastDashboard = z.infer<typeof ForecastDashboardSchema>;

// Models
export type ModelMetrics = z.infer<typeof ModelMetricsSchema>;
export type ModelSummary = z.infer<typeof ModelSummarySchema>;
export type ModelComparison = z.infer<typeof ModelComparisonSchema>;
export type FeatureImportance = z.infer<typeof FeatureImportanceSchema>;

// Images
export type ImageMetadata = z.infer<typeof ImageMetadataSchema>;
export type ImageListResponse = z.infer<typeof ImageListResponseSchema>;

// Health
export type ServiceHealth = z.infer<typeof ServiceHealthSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;

// Errors
export type ErrorDetail = z.infer<typeof ErrorDetailSchema>;
export type ErrorResponse = z.infer<typeof ErrorResponseSchema>;

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Valida una respuesta de forecast y retorna el tipo correcto
 */
export function parseForecastResponse(data: unknown): ForecastResponse {
  return ForecastResponseSchema.parse(data);
}

/**
 * Valida una respuesta de dashboard y retorna el tipo correcto
 */
export function parseForecastDashboard(data: unknown): ForecastDashboard {
  return ForecastDashboardSchema.parse(data);
}

/**
 * Valida una respuesta de comparación de modelos
 */
export function parseModelComparison(data: unknown): ModelComparison {
  return ModelComparisonSchema.parse(data);
}

/**
 * Valida una respuesta de health
 */
export function parseHealthResponse(data: unknown): HealthResponse {
  return HealthResponseSchema.parse(data);
}

/**
 * Safe parse que retorna null en caso de error
 */
export function safeParseForecast(data: unknown): ForecastResponse | null {
  const result = ForecastResponseSchema.safeParse(data);
  return result.success ? result.data : null;
}

// =============================================================================
// CONSTANTS (Deben coincidir con backend/src/core/config.py)
// =============================================================================

export const HORIZONS = [1, 5, 10, 15, 20, 25, 30] as const;
export type Horizon = (typeof HORIZONS)[number];

export const MODELS = [
  'ridge',
  'bayesian_ridge',
  'ard',
  'xgboost',
  'lightgbm',
  'catboost',
  'hybrid_xgboost',
  'hybrid_lightgbm',
  'hybrid_catboost',
] as const;
export type ModelName = (typeof MODELS)[number];

export const MODEL_TYPES = ['linear', 'boosting', 'hybrid'] as const;
export type ModelType = (typeof MODEL_TYPES)[number];
