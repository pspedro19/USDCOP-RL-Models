/**
 * Pipeline Data Types
 * ====================
 *
 * Tipos para el pipeline de datos L0-L6 y métricas de calidad
 */

import { Nullable } from './common';

// === ENUMS ===

/**
 * Estado del pipeline
 */
export enum PipelineStatus {
  PASS = 'pass',
  FAIL = 'fail',
  WARNING = 'warning',
  LOADING = 'loading',
  IDLE = 'idle',
  ERROR = 'error',
}

/**
 * Capas del pipeline
 */
export enum PipelineLayer {
  L0 = 'L0',
  L1 = 'L1',
  L2 = 'L2',
  L3 = 'L3',
  L4 = 'L4',
  L5 = 'L5',
  L6 = 'L6',
}

/**
 * Tipo de validación
 */
export enum ValidationType {
  SCHEMA = 'schema',
  DATA_QUALITY = 'data_quality',
  COMPLETENESS = 'completeness',
  CONSISTENCY = 'consistency',
  ACCURACY = 'accuracy',
  TIMELINESS = 'timeliness',
}

// === BASE TYPES ===

/**
 * Punto de datos del pipeline
 */
export interface PipelineDataPoint {
  timestamp: number;
  value: number;
  layer: string;
  metadata?: Record<string, any>;
}

/**
 * Estado de salud de capa del pipeline
 */
export interface LayerHealth {
  layer: PipelineLayer;
  status: PipelineStatus;
  message: string;
  timestamp: string;
  metrics?: Record<string, number>;
  errors?: string[];
  warnings?: string[];
}

/**
 * Estado de salud del pipeline completo
 */
export interface PipelineHealth {
  overall_status: PipelineStatus;
  timestamp: string;
  layers: {
    [key in PipelineLayer]?: LayerHealth;
  };
  summary?: {
    total_layers: number;
    passing: number;
    failing: number;
    warnings: number;
  };
}

// === L0 - RAW DATA ===

/**
 * Datos raw L0
 */
export interface L0RawData {
  timestamp: string;
  time?: string;
  datetime?: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source?: string;
}

/**
 * Estadísticas L0
 */
export interface L0Statistics {
  total_records: number;
  date_range: {
    start: string;
    end: string;
  };
  data_quality: {
    completeness: number;
    null_percentage: number;
    duplicate_percentage: number;
  };
  ohlcv_stats: {
    avg_close: number;
    avg_volume: number;
    max_high: number;
    min_low: number;
  };
}

/**
 * Estado L0
 */
export interface L0Status extends LayerHealth {
  layer: PipelineLayer.L0;
  data_source: string;
  last_update: string;
  records_count: number;
  data_quality_score: number;
}

// === L1 - STANDARDIZED DATA ===

/**
 * Datos estandarizados L1
 */
export interface L1StandardizedData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  normalized_close?: number;
  standardized?: boolean;
}

/**
 * Reporte de calidad L1
 */
export interface L1QualityReport {
  timestamp: string;
  total_records: number;
  missing_data_points: number;
  completeness_score: number;
  validation_errors: Array<{
    type: string;
    count: number;
    severity: 'error' | 'warning' | 'info';
  }>;
  data_gaps: Array<{
    start: string;
    end: string;
    duration_minutes: number;
  }>;
}

/**
 * Completitud L1
 */
export interface L1Completeness {
  total_expected: number;
  total_actual: number;
  completeness_percentage: number;
  missing_intervals: number;
  gaps: Array<{
    start: string;
    end: string;
    count: number;
  }>;
}

/**
 * Episodio L1
 */
export interface L1Episode {
  episode_id: number;
  start_timestamp: string;
  end_timestamp: string;
  num_steps: number;
  avg_close: number;
  volatility: number;
  total_volume: number;
}

/**
 * Estado L1
 */
export interface L1Status extends LayerHealth {
  layer: PipelineLayer.L1;
  completeness: number;
  quality_score: number;
  episodes_count: number;
  last_standardization: string;
}

// === L2 - PREPARED DATA ===

/**
 * Datos preparados L2
 */
export interface L2PreparedData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  returns?: number;
  log_returns?: number;
  normalized_values?: Record<string, number>;
}

/**
 * Estado L2
 */
export interface L2Status extends LayerHealth {
  layer: PipelineLayer.L2;
  preparation_complete: boolean;
  normalization_applied: boolean;
  outliers_removed: number;
  feature_engineering_steps: number;
}

// === L3 - FEATURES ===

/**
 * Features L3
 */
export interface L3Features {
  timestamp: string;
  features: Record<string, number>;
  feature_groups?: {
    price?: Record<string, number>;
    volume?: Record<string, number>;
    technical?: Record<string, number>;
    statistical?: Record<string, number>;
    custom?: Record<string, number>;
  };
}

/**
 * Feature metadata
 */
export interface FeatureMetadata {
  name: string;
  type: 'numerical' | 'categorical' | 'binary' | 'ordinal';
  description: string;
  importance?: number;
  correlation?: number;
  nulls_percentage?: number;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
}

/**
 * Estado L3
 */
export interface L3Status extends LayerHealth {
  layer: PipelineLayer.L3;
  total_features: number;
  feature_groups: string[];
  feature_importance_available: boolean;
  last_feature_calculation: string;
}

// === L4 - RL READY ===

/**
 * Dataset RL-ready L4
 */
export interface L4Dataset {
  timestamp: string;
  state: number[];
  action?: number;
  reward?: number;
  next_state?: number[];
  done?: boolean;
  info?: Record<string, any>;
}

/**
 * Observation space
 */
export interface ObservationSpace {
  shape: number[];
  dtype: string;
  low: number[];
  high: number[];
  feature_names: string[];
}

/**
 * Action space
 */
export interface ActionSpace {
  type: 'discrete' | 'continuous';
  n?: number; // for discrete
  shape?: number[]; // for continuous
  low?: number[];
  high?: number[];
  action_names?: string[];
}

/**
 * Estado L4
 */
export interface L4Status extends LayerHealth {
  layer: PipelineLayer.L4;
  observation_space: ObservationSpace;
  action_space: ActionSpace;
  dataset_size: number;
  ready_for_training: boolean;
}

// === L5 - TRAINING ===

/**
 * Configuración de entrenamiento
 */
export interface TrainingConfig {
  model_type: 'RL' | 'ML' | 'LLM' | 'Ensemble';
  algorithm: string;
  hyperparameters: Record<string, any>;
  training_duration: number;
  validation_split: number;
}

/**
 * Métricas de entrenamiento
 */
export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy?: number;
  validation_loss?: number;
  validation_accuracy?: number;
  learning_rate: number;
  timestamp: string;
}

/**
 * Estado del modelo
 */
export interface ModelState {
  model_id: string;
  model_name: string;
  version: string;
  status: 'training' | 'trained' | 'deployed' | 'archived';
  created_at: string;
  updated_at: string;
  metrics: TrainingMetrics[];
  config: TrainingConfig;
}

/**
 * Estado L5
 */
export interface L5Status extends LayerHealth {
  layer: PipelineLayer.L5;
  active_models: number;
  training_in_progress: boolean;
  last_training_completed: string;
  best_model_performance: number;
}

// === L6 - BACKTEST ===

/**
 * Configuración de backtest
 */
export interface BacktestConfig {
  strategy_code: string;
  strategy_name: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  position_size: number;
  commission: number;
  slippage: number;
}

/**
 * Resultado de backtest
 */
export interface L6BacktestResult {
  strategy_code: string;
  strategy_name: string;
  start_date: string;
  end_date: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  avg_trade_duration: number;
  profit_factor: number;
  sortino_ratio: number;
  calmar_ratio: number;
  equity_curve: Array<{
    timestamp: string;
    equity: number;
    drawdown: number;
  }>;
}

/**
 * Estado L6
 */
export interface L6Status extends LayerHealth {
  layer: PipelineLayer.L6;
  backtests_completed: number;
  best_strategy: string;
  best_sharpe_ratio: number;
  last_backtest: string;
}

// === QUALITY METRICS ===

/**
 * Métricas de calidad de datos
 */
export interface DataQualityMetrics {
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
  validity: number;
  uniqueness: number;
  overall_score: number;
  timestamp: string;
}

/**
 * Validación de datos
 */
export interface DataValidation {
  validation_id: string;
  layer: PipelineLayer;
  type: ValidationType;
  passed: boolean;
  score: number;
  errors: Array<{
    field: string;
    message: string;
    severity: 'error' | 'warning' | 'info';
    count: number;
  }>;
  timestamp: string;
}

/**
 * Audit log entry
 */
export interface AuditLogEntry {
  id: string;
  timestamp: string;
  layer: PipelineLayer;
  action: string;
  user?: string;
  details: Record<string, any>;
  status: 'success' | 'failure';
  error_message?: string;
}

// === PIPELINE ORCHESTRATION ===

/**
 * Tarea del pipeline
 */
export interface PipelineTask {
  task_id: string;
  name: string;
  layer: PipelineLayer;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  started_at?: string;
  completed_at?: string;
  error?: string;
  dependencies?: string[];
}

/**
 * Pipeline run completo
 */
export interface PipelineRun {
  run_id: string;
  started_at: string;
  completed_at?: string;
  status: PipelineStatus;
  tasks: PipelineTask[];
  summary: {
    total_tasks: number;
    completed: number;
    failed: number;
    pending: number;
  };
  metadata?: Record<string, any>;
}

/**
 * Endpoints del pipeline
 */
export interface PipelineEndpoints {
  layer: PipelineLayer;
  endpoints: Array<{
    name: string;
    url: string;
    method: 'GET' | 'POST' | 'PUT' | 'DELETE';
    description: string;
    status: 'available' | 'unavailable';
  }>;
}
