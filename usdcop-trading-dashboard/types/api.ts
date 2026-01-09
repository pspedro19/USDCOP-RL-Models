/**
 * API Types
 * ==========
 *
 * Tipos para respuestas de API, configuración de cliente HTTP, etc.
 */

import { ApiResponse, PaginatedResponse, Nullable } from './common';

// === HTTP CLIENT ===

/**
 * Método HTTP
 */
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'HEAD' | 'OPTIONS';

/**
 * Headers HTTP
 */
export type HttpHeaders = Record<string, string>;

/**
 * Request config
 */
export interface RequestConfig {
  method?: HttpMethod;
  headers?: HttpHeaders;
  params?: Record<string, string | number | boolean | undefined>;
  body?: unknown;
  timeout?: number;
  signal?: AbortSignal;
  credentials?: RequestCredentials;
  mode?: RequestMode;
  cache?: RequestCache;
}

/**
 * Response con tipo genérico
 */
export interface TypedResponse<T> extends Response {
  data: T;
}

/**
 * Error de API
 */
export interface ApiError {
  message: string;
  code?: string;
  status?: number;
  details?: Record<string, unknown>;
  timestamp?: string;
  path?: string;
}

// === HEALTH CHECK ===

/**
 * Health check response
 */
export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  uptime?: number;
  version?: string;
  services?: {
    database?: ServiceHealth;
    websocket?: ServiceHealth;
    external_api?: ServiceHealth;
    [key: string]: ServiceHealth | undefined;
  };
}

/**
 * Service health
 */
export interface ServiceHealth {
  status: 'up' | 'down' | 'degraded';
  message?: string;
  responseTime?: number;
  lastCheck?: string;
}

// === DATABASE API ===

/**
 * Database stats
 */
export interface DatabaseStats {
  total_records: number;
  tables: Array<{
    name: string;
    row_count: number;
    size_mb: number;
  }>;
  last_updated: string;
  connection_status: 'connected' | 'disconnected';
}

/**
 * Query result
 */
export interface QueryResult<T = unknown> {
  rows: T[];
  rowCount: number;
  fields: Array<{
    name: string;
    dataType: string;
  }>;
  executionTime?: number;
}

// === ANALYTICS API ===

/**
 * Analytics request
 */
export interface AnalyticsRequest {
  metric: string;
  timeRange: {
    start: string;
    end: string;
  };
  groupBy?: string;
  filters?: Record<string, string | number | boolean | null>;
  aggregation?: 'sum' | 'avg' | 'min' | 'max' | 'count';
}

/**
 * Analytics response
 */
export interface AnalyticsResponse {
  metric: string;
  data: Array<{
    timestamp: string;
    value: number;
    metadata?: Record<string, unknown>;
  }>;
  summary: {
    total: number;
    average: number;
    min: number;
    max: number;
  };
}

// === ML ANALYTICS API ===

/**
 * Prediction request
 */
export interface PredictionRequest {
  model_id: string;
  input_data: number[] | Record<string, number>;
  return_probabilities?: boolean;
  return_confidence?: boolean;
}

/**
 * Prediction response
 */
export interface PredictionResponse {
  prediction: number | string;
  probabilities?: number[];
  confidence?: number;
  model_version?: string;
  timestamp: string;
  inference_time_ms?: number;
}

/**
 * Model info
 */
export interface ModelInfo {
  model_id: string;
  model_name: string;
  version: string;
  type: 'RL' | 'ML' | 'LLM' | 'Ensemble';
  status: 'active' | 'inactive' | 'training' | 'archived';
  created_at: string;
  updated_at: string;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    sharpe_ratio?: number;
    [key: string]: number | undefined;
  };
}

// === BACKTEST API ===

/**
 * Backtest request
 */
export interface BacktestRequest {
  strategy_code: string;
  start_date: string;
  end_date: string;
  initial_capital?: number;
  commission?: number;
  slippage?: number;
  position_size?: number;
  parameters?: Record<string, string | number | boolean | null>;
}

/**
 * Backtest trigger response
 */
export interface BacktestTriggerResponse {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  message: string;
  estimated_duration?: number;
}

// === ALERTS API ===

/**
 * Alert level
 */
export type AlertLevel = 'info' | 'warning' | 'error' | 'critical';

/**
 * Alert
 */
export interface Alert {
  id: string;
  level: AlertLevel;
  title: string;
  message: string;
  source: string;
  timestamp: string;
  acknowledged: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * System alerts response
 */
export interface SystemAlertsResponse {
  alerts: Alert[];
  unacknowledged_count: number;
  severity_counts: {
    info: number;
    warning: number;
    error: number;
    critical: number;
  };
}

// === USAGE MONITORING API ===

/**
 * API usage stats
 */
export interface ApiUsageStats {
  endpoint: string;
  method: HttpMethod;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  avg_response_time: number;
  max_response_time: number;
  min_response_time: number;
  requests_per_minute: number;
  last_called: string;
}

/**
 * Usage monitoring response
 */
export interface UsageMonitoringResponse {
  period: {
    start: string;
    end: string;
  };
  total_requests: number;
  total_errors: number;
  error_rate: number;
  avg_response_time: number;
  endpoints: ApiUsageStats[];
  quota?: {
    limit: number;
    used: number;
    remaining: number;
    reset_at: string;
  };
}

// === BACKUP API ===

/**
 * Backup status
 */
export interface BackupStatus {
  last_backup: Nullable<string>;
  next_backup: Nullable<string>;
  backup_size: number;
  backup_location: string;
  status: 'success' | 'in_progress' | 'failed' | 'none';
  error_message?: string;
}

// === MARKET DATA API ===

/**
 * Market status response
 */
export interface MarketStatusResponse {
  is_open: boolean;
  current_time: string;
  timezone: string;
  market_hours: {
    open: string;
    close: string;
  };
  next_open?: string;
  next_close?: string;
}

/**
 * Volume profile request
 */
export interface VolumeProfileRequest {
  symbol: string;
  start_date: string;
  end_date: string;
  bin_size?: number;
  timeframe?: string;
}

// === EXPORT API ===

/**
 * Export format
 */
export type ExportFormat = 'csv' | 'json' | 'excel' | 'pdf' | 'png' | 'svg';

/**
 * Export request
 */
export interface ExportRequest {
  data_type: 'chart' | 'table' | 'report';
  format: ExportFormat;
  data: unknown;
  filename?: string;
  options?: {
    include_headers?: boolean;
    delimiter?: string;
    compression?: boolean;
    [key: string]: string | number | boolean | undefined;
  };
}

/**
 * Export response
 */
export interface ExportResponse {
  download_url: string;
  filename: string;
  size_bytes: number;
  format: ExportFormat;
  expires_at?: string;
}

// === CONFIGURATION API ===

/**
 * Feature flag
 */
export interface FeatureFlag {
  name: string;
  enabled: boolean;
  description?: string;
  rollout_percentage?: number;
}

/**
 * App configuration
 */
export interface AppConfiguration {
  features: FeatureFlag[];
  settings: Record<string, unknown>;
  environment: 'development' | 'staging' | 'production';
  version: string;
}

// === NOTIFICATION API ===

/**
 * Notification type
 */
export type NotificationType = 'signal' | 'trade' | 'alert' | 'system' | 'update';

/**
 * Notification
 */
export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  priority: 'low' | 'medium' | 'high';
  action_url?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Notification preferences
 */
export interface NotificationPreferences {
  email_enabled: boolean;
  push_enabled: boolean;
  sms_enabled: boolean;
  types: {
    [key in NotificationType]?: boolean;
  };
}

// === RATE LIMITING ===

/**
 * Rate limit info
 */
export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number; // Unix timestamp
  reset_in_seconds: number;
}

/**
 * Rate limit response
 */
export interface RateLimitResponse {
  rate_limit: RateLimitInfo;
  message?: string;
}
