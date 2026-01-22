/**
 * Models Configuration
 * ====================
 * Configuration for RL/ML trading models
 *
 * IMPORTANT: This file only contains TYPE DEFINITIONS and API endpoint configuration.
 * Actual model data (list of models, metrics, status) comes from the API.
 * No hardcoded model data here.
 *
 * SSOT: Algorithm types and colors are imported from ssot.contract.ts
 */

import { apiConfig } from './api.config';
import {
  RL_ALGORITHMS,
  ML_ALGORITHMS,
  ALL_ALGORITHMS,
  ALGORITHM_COLORS,
  Action,
  ACTION_NAMES,
  type Algorithm,
  type RLAlgorithm,
} from '../contracts/ssot.contract';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Supported RL/ML algorithm types - uses SSOT
 * Includes all algorithms from RL_ALGORITHMS and ML_ALGORITHMS
 */
export type ModelAlgorithm = Algorithm;

/**
 * Model deployment status
 */
export type ModelStatus = 'production' | 'testing' | 'staging' | 'deprecated' | 'training';

/**
 * Trading signal types
 */
export type SignalType = 'LONG' | 'SHORT' | 'HOLD';

/**
 * Trade status
 */
export type TradeStatus = 'OPEN' | 'CLOSED' | 'CANCELLED';

/**
 * Model type category
 */
export type ModelType = 'rl' | 'ml' | 'llm' | 'ensemble';

/**
 * Database status for model registry
 */
export type ModelDbStatus = 'registered' | 'deployed' | 'archived';

/**
 * Model configuration from API
 */
export interface ModelConfig {
  id: string;
  name: string;
  algorithm: ModelAlgorithm | string;
  version: string;
  status: ModelStatus;
  color: string;
  type?: ModelType;
  description?: string;
  createdAt?: string;
  updatedAt?: string;
  /** Whether this model uses real production data (true) or demo data (false) */
  isRealData?: boolean;
  /** Raw database status from model_registry table */
  dbStatus?: ModelDbStatus;
  config?: {
    episodeLength?: number;
    actionThreshold?: number;
    features?: string[];
    hyperparameters?: Record<string, number | string>;
  };
  backtest?: {
    sharpe: number;
    maxDrawdown: number;
    winRate: number;
    holdPercent: number;
    totalTrades?: number;
    dataRange?: {
      start: string;
      end: string;
    };
  };
}

/**
 * Model signal from API
 */
export interface ModelSignal {
  id?: string;
  modelId: string;
  timestamp: string;
  signal: SignalType;
  actionRaw: number;
  confidence: number;
  price: number;
  features?: Record<string, number>;
}

/**
 * Model trade from API
 */
export interface ModelTrade {
  tradeId: number;
  modelId: string;
  openTime: string;
  closeTime: string | null;
  signal: SignalType;
  entryPrice: number;
  exitPrice: number | null;
  pnl: number | null;
  pnlPct: number | null;
  durationMinutes: number | null;
  status: TradeStatus;
  confidence: number;
}

/**
 * Trades summary from API
 */
export interface TradesSummary {
  total: number;
  wins: number;
  losses: number;
  holds: number;
  winRate: number;
  streak: number;
  pnlTotal: number;
  pnlPct: number;
  avgDuration: number | null;
  bestTrade: number | null;
  worstTrade: number | null;
}

/**
 * Model metrics from API
 */
export interface ModelMetrics {
  modelId: string;
  period: string;
  live: {
    sharpe: number | null;
    maxDrawdown: number | null;
    winRate: number | null;
    holdPercent: number | null;
    totalTrades: number;
    pnlToday: number | null;
    pnlTodayPct: number | null;
    pnlMonth: number | null;
    pnlMonthPct: number | null;
  };
  backtest: {
    sharpe: number;
    maxDrawdown: number;
    winRate: number;
    holdPercent: number;
  };
}

/**
 * Equity curve data point
 */
export interface EquityCurvePoint {
  date: string;
  value: number;
  pnl?: number;
  pnlPct?: number;
}

/**
 * Model comparison result
 */
export interface ModelComparison {
  modelId: string;
  name: string;
  color: string;
  status: ModelStatus;
  metrics: {
    sharpe: number | null;
    maxDrawdown: number | null;
    winRate: number | null;
    holdPercent: number | null;
    totalTrades: number;
    pnlMonth: number | null;
  };
}

// ============================================================================
// API Endpoints Configuration
// ============================================================================

/**
 * Models API endpoints
 * These endpoints should be implemented in the backend
 */
export const modelsApiEndpoints = {
  /**
   * List all available models
   * GET /api/models
   */
  list: '/api/models',

  /**
   * Get specific model configuration
   * GET /api/models/{modelId}
   */
  get: (modelId: string) => `/api/models/${modelId}`,

  /**
   * Get model signals
   * GET /api/models/{modelId}/signals?period=today&limit=100
   */
  signals: (modelId: string) => `/api/models/${modelId}/signals`,

  /**
   * Stream model signals (SSE)
   * GET /api/models/{modelId}/signals/stream
   */
  signalsStream: (modelId: string) => `/api/models/${modelId}/signals/stream`,

  /**
   * Get model trades
   * GET /api/models/{modelId}/trades?period=today&status=all
   */
  trades: (modelId: string) => `/api/models/${modelId}/trades`,

  /**
   * Get model metrics
   * GET /api/models/{modelId}/metrics?period=30d
   */
  metrics: (modelId: string) => `/api/models/${modelId}/metrics`,

  /**
   * Get model equity curve
   * GET /api/models/{modelId}/equity-curve?period=6m
   */
  equityCurve: (modelId: string) => `/api/models/${modelId}/equity-curve`,

  /**
   * Compare multiple models
   * GET /api/models/compare?ids=ppo_v19,sac_v1&period=30d
   */
  compare: '/api/models/compare',

  /**
   * Promote a model to deployed status
   * POST /api/models/{modelId}/promote
   */
  promote: (modelId: string) => `/api/models/${modelId}/promote`,
} as const;

// ============================================================================
// Configuration Constants
// ============================================================================

/**
 * Default colors for models (used as fallback if API doesn't provide color)
 * Uses SSOT ALGORITHM_COLORS
 */
export const defaultModelColors = ALGORITHM_COLORS;

/**
 * Status badge configuration
 */
export const statusConfig: Record<ModelStatus, { label: string; color: string; bgColor: string }> = {
  production: { label: 'Production', color: '#10B981', bgColor: 'rgba(16, 185, 129, 0.1)' },
  testing: { label: 'Testing', color: '#F59E0B', bgColor: 'rgba(245, 158, 11, 0.1)' },
  staging: { label: 'Staging', color: '#3B82F6', bgColor: 'rgba(59, 130, 246, 0.1)' },
  deprecated: { label: 'Deprecated', color: '#6B7280', bgColor: 'rgba(107, 114, 128, 0.1)' },
  training: { label: 'Training', color: '#8B5CF6', bgColor: 'rgba(139, 92, 246, 0.1)' },
};

/**
 * Signal badge configuration
 */
export const signalConfig: Record<SignalType, { label: string; color: string; bgColor: string; icon: string }> = {
  LONG: { label: 'LONG', color: '#10B981', bgColor: 'rgba(16, 185, 129, 0.1)', icon: '↑' },
  SHORT: { label: 'SHORT', color: '#EF4444', bgColor: 'rgba(239, 68, 68, 0.1)', icon: '↓' },
  HOLD: { label: 'HOLD', color: '#6B7280', bgColor: 'rgba(107, 114, 128, 0.1)', icon: '−' },
};

/**
 * Period options for filtering
 */
export const periodOptions = [
  { value: 'today', label: 'Hoy' },
  { value: '7d', label: '7 D\u00edas' },
  { value: '30d', label: '30 D\u00edas' },
  { value: '90d', label: '90 D\u00edas' },
  { value: 'all', label: 'Todo' },
] as const;

export type PeriodOption = typeof periodOptions[number]['value'];

/**
 * Refresh intervals for model data (in milliseconds)
 */
export const modelRefreshIntervals = {
  signals: 5000,        // 5 seconds for live signals
  trades: 10000,        // 10 seconds for trades
  metrics: 30000,       // 30 seconds for metrics
  equityCurve: 60000,   // 1 minute for equity curve
  modelList: 300000,    // 5 minutes for model list
} as const;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get color for a model based on algorithm
 * Uses SSOT ALGORITHM_COLORS
 */
export function getModelColor(algorithm: ModelAlgorithm | string, customColor?: string): string {
  return customColor || defaultModelColors[algorithm as Algorithm] || '#6B7280';
}

/**
 * Get status badge props
 */
export function getStatusBadgeProps(status: ModelStatus) {
  return statusConfig[status] || statusConfig.deprecated;
}

/**
 * Get signal badge props
 */
export function getSignalBadgeProps(signal: SignalType) {
  return signalConfig[signal] || signalConfig.HOLD;
}

/**
 * Format duration from minutes to readable string
 */
export function formatDuration(minutes: number | null): string {
  if (minutes === null || minutes === undefined) return '—';

  if (minutes < 60) {
    return `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  if (hours < 24) {
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  }

  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;

  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
}

/**
 * Format P&L value with sign
 */
export function formatPnL(value: number | null, currency: boolean = true): string {
  if (value === null || value === undefined) return '—';

  const prefix = value > 0 ? '+' : '';
  const formatted = currency
    ? `$${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`
    : value.toFixed(2);

  return `${prefix}${value < 0 ? '-' : ''}${formatted.replace('-', '')}`;
}

/**
 * Format percentage with sign
 */
export function formatPct(value: number | null, decimals: number = 2): string {
  if (value === null || value === undefined) return '—';

  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(decimals)}%`;
}

/**
 * Get P&L color class
 */
export function getPnLColorClass(value: number | null): string {
  if (value === null || value === undefined) return 'text-slate-400';
  if (value > 0) return 'text-green-500';
  if (value < 0) return 'text-red-500';
  return 'text-slate-400';
}

/**
 * Check if a model is the production model
 */
export function isProductionModel(model: ModelConfig): boolean {
  return model.status === 'production';
}

/**
 * Check if a model is active (not deprecated)
 */
export function isActiveModel(model: ModelConfig): boolean {
  return model.status !== 'deprecated';
}

export default {
  endpoints: modelsApiEndpoints,
  colors: defaultModelColors,
  status: statusConfig,
  signals: signalConfig,
  periods: periodOptions,
  refresh: modelRefreshIntervals,
};
