/**
 * TypeScript Types Index
 * =======================
 *
 * Punto central de exportación para todos los tipos del proyecto.
 * Sistema de tipos profesional y centralizado para USD/COP Trading Dashboard.
 *
 * CANONICAL IMPORTS (Zod-validated):
 * @example
 * import { TradingSignal, SignalType, Trade, apiClient } from '@/types'
 *
 * LEGACY IMPORTS (for backwards compatibility):
 * @example
 * import { OrderType, Position } from '@/types'
 */

// ============================================================================
// CANONICAL SCHEMAS (Zod validated - Single Source of Truth)
// These are the preferred types for new code
// ============================================================================
export * from './schemas';

// ============================================================================
// Re-export all type modules (legacy compatibility)
// ============================================================================

// Common utility types
export * from './common';

// Trading types (some overlap with schemas - schemas take precedence)
export {
  OrderType,
  OrderSide as LegacyOrderSide,
  OrderStatus,
  MarketStatus as LegacyMarketStatus,
  Timeframe,
} from './trading';

export type {
  OHLCVData,
  CandlestickData,
  CandlestickExtended,
  CandlestickResponse,
  MarketDataPoint,
  MarketUpdate,
  PriceTick,
  TechnicalIndicators as LegacyTechnicalIndicators,
  TradingSignal as LegacyTradingSignal,
  SignalAlert,
  SignalPerformance,
  Position,
  Trade as LegacyTrade,
  TradeUpdate,
  Order,
  OrderBookLevel,
  OrderBook,
  OrderBookUpdate,
  SymbolStats,
  VolumeProfileLevel,
  VolumeProfile,
  BacktestResult,
  PerformanceMetrics,
} from './trading';

// Pipeline types
export * from './pipeline';

// API types (excluding ApiError which conflicts with schemas)
export type {
  HttpMethod,
  HttpHeaders,
  RequestConfig,
  TypedResponse,
  ApiError as LegacyApiError,
  HealthCheckResponse,
  ServiceHealth,
  DatabaseStats,
  QueryResult,
  AnalyticsRequest,
  AnalyticsResponse,
  PredictionRequest,
  PredictionResponse,
  ModelInfo,
  BacktestRequest,
  BacktestTriggerResponse,
  AlertLevel,
  Alert,
  SystemAlertsResponse,
  ApiUsageStats,
  UsageMonitoringResponse,
  BackupStatus,
  MarketStatusResponse,
  VolumeProfileRequest,
  ExportFormat,
  ExportRequest,
  ExportResponse,
  FeatureFlag,
  AppConfiguration,
  NotificationType,
  Notification,
  NotificationPreferences,
  RateLimitInfo,
  RateLimitResponse,
} from './api';

// WebSocket types
export * from './websocket';

// Chart types
export * from './charts';

// Contracts (some types re-exported from schemas)
export type {
  StrategySignal,
  LatestSignalsResponse,
  StrategyPerformance,
  PerformanceResponse,
  RiskLimits,
  RiskStatusResponse,
  HealthStatus,
  ModelHealth,
  ModelHealthResponse,
  APIError,
} from './contracts';

// ============================================================================
// LEGACY TYPES (Preserving backwards compatibility)
// ============================================================================
// These types are kept for compatibility with existing code
// Consider migrating to the new type system over time

export interface L0RawData {
  coverage: number;
  ohlc_invariants: number;
  cross_source_delta: number;
  duplicates: number;
  gaps: number;
  stale_rate: number;
  acquisition_latency: number;
  volume_data_points: number;
  data_source_health: 'healthy' | 'warning' | 'critical';
  total_records?: number;
}

export interface L1StandardizedData {
  grid_perfection: number;
  terminal_correctness: number;
  hod_baselines: number;
  processed_volume: number;
  transformation_latency: number;
  validation_passed: number;
  data_integrity: 'healthy' | 'warning' | 'critical';
  quality_score?: number;
  status?: string;
  total_records?: number;
}

export interface L1Episode {
  episode_id: string;
  start_time: string;
  end_time: string;
  total_steps: number;
  quality_score: number;
}

export interface L2PreparedData {
  winsorization?: { rate_pct: number };
  hod_deseasonalization?: { median_abs_mean: number };
  nan_rate_pct?: number;
  indicators_count?: number;
  pass?: boolean;
}

export interface L3Features {
  features?: Array<{ max_abs_ic: number }>;
  metadata?: { features_count: number };
  summary?: { pass: boolean };
  forward_ic?: number;
  max_correlation?: number;
  nan_post_warmup?: number;
  train_schema_valid?: number;
  feature_engineering?: number;
  anti_leakage_checks?: number;
  correlation_analysis?: string;
}

export interface L4QualityCheck {
  max_clip_rate?: number;
  reward_check?: {
    std: number;
    zero_pct: number;
    rmse: number;
  };
  overall_pass?: boolean;
  observation_features?: number;
  clip_rate?: number;
  zero_rate_t33?: number;
  reward_std?: number;
  reward_zero_rate?: number;
  reward_rmse?: number;
  episode_completeness?: number;
  rl_readiness?: string;
}

export interface L5Models {
  models?: Array<{
    model_id: string;
    model_name: string;
    version: string;
    type: string;
    status: string;
  }>;
  count?: number;
}

export interface L6BacktestResults {
  results?: Array<{
    strategy: string;
    sharpeRatio: number;
    totalReturn: number;
    maxDrawdown: number;
  }>;
  summary?: BacktestSummary;
}

export interface BacktestSummary {
  total_strategies: number;
  best_performer: string;
  worst_performer: string;
  avg_sharpe_ratio: number;
  avg_return: number;
  correlation_matrix?: Record<string, Record<string, number>>;
}

export interface RiskMetrics {
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  beta: number;
  alpha: number;
  volatility: number;
  skewness: number;
  kurtosis: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  avg_drawdown: number;
  drawdown_duration: number;
}

export interface PerformanceKPIs {
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  avg_trade_pnl: number;
  total_trades: number;
}

export interface MarketStats {
  symbol: string;
  last_price: number;
  change_24h: number;
  change_24h_pct: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
  volatility: number;
  beta: number;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'down';
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_latency: number;
  uptime: number;
  services: ServiceStatus[];
}

export interface ServiceStatus {
  name: string;
  status: 'running' | 'stopped' | 'error';
  uptime: number;
  last_check: string;
  error_message?: string;
}

export interface ExportOptions {
  format: 'pdf' | 'excel' | 'csv';
  data: unknown;
  filename: string;
  includeCharts?: boolean;
  includeTables?: boolean;
  includeMetrics?: boolean;
}

export interface ExportConfig {
  name: string;
  format: 'pdf' | 'excel' | 'csv';
  description: string;
}

export interface ReportConfig {
  title: string;
  subtitle?: string;
  author?: string;
  company?: string;
  watermark?: string;
}

export interface MetricData {
  label: string;
  value: string | number;
  format?: 'number' | 'percentage' | 'currency' | 'time';
}

export interface ExcelSheetData {
  name: string;
  data: Record<string, unknown>[];
}

export type Status = 'success' | 'error' | 'pending' | 'idle';

export type DataQuality = 'excellent' | 'good' | 'fair' | 'poor';

export type TrendDirection = 'up' | 'down' | 'neutral';

export type TimeFrame = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';

export type PartialDeep<T> = {
  [P in keyof T]?: T[P] extends object ? PartialDeep<T[P]> : T[P];
};

export type EventHandler<E = Event> = (event: E) => void;

export type ChangeHandler<T = unknown> = (value: T) => void;
