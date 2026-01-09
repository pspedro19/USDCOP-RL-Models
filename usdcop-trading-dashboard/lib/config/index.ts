/**
 * Configuration System
 * ====================
 * Centralized configuration management for the USDCOP Trading Dashboard
 *
 * This module provides a single source of truth for all configuration values,
 * replacing hardcoded values throughout the application.
 *
 * ## Features
 * - Environment variable support (NEXT_PUBLIC_* for client-side)
 * - Type-safe configuration objects
 * - Default values with ability to override
 * - Organized by domain (API, Market, Risk, UI)
 * - Helper functions for common operations
 *
 * ## Usage
 * ```typescript
 * import { apiConfig, marketConfig, riskConfig, uiConfig } from '@/lib/config'
 *
 * // Use specific configuration
 * const tradingUrl = apiConfig.trading.baseUrl
 * const defaultSymbol = marketConfig.defaultSymbol
 * const maxLeverage = riskConfig.leverage.max
 * const chartHeight = uiConfig.chart.dimensions.defaultHeight
 *
 * // Or import everything
 * import config from '@/lib/config'
 * const url = config.api.trading.baseUrl
 * ```
 *
 * ## Environment Variables
 * Set these in .env.local for client-side access:
 *
 * ### API Endpoints
 * - NEXT_PUBLIC_TRADING_API_URL - Trading API base URL (default: http://localhost:8000)
 * - NEXT_PUBLIC_ANALYTICS_API_URL - Analytics API base URL (default: http://localhost:8001)
 * - NEXT_PUBLIC_PIPELINE_API_URL - Pipeline API base URL (default: http://localhost:8002)
 * - NEXT_PUBLIC_COMPLIANCE_API_URL - Compliance API base URL (default: http://localhost:8003)
 * - NEXT_PUBLIC_ML_ANALYTICS_API_URL - ML Analytics API base URL (default: http://localhost:8004)
 * - BACKTEST_API_URL - Backtest API base URL (default: http://localhost:8006)
 * - MULTI_MODEL_API_URL - Multi-model API base URL (default: http://usdcop-multi-model-api:8006)
 *
 * ### Market Configuration
 * - NEXT_PUBLIC_DEFAULT_SYMBOL - Default trading symbol (default: USDCOP)
 * - NEXT_PUBLIC_PORTFOLIO_VALUE - Default portfolio value (default: 10000000)
 *
 * ### Risk Configuration
 * - NEXT_PUBLIC_MAX_LEVERAGE - Maximum leverage (default: 5.0)
 * - NEXT_PUBLIC_WARNING_LEVERAGE - Warning leverage threshold (default: 4.0)
 * - NEXT_PUBLIC_MAX_VAR_PCT - Maximum VaR percentage (default: 0.10)
 * - NEXT_PUBLIC_WARNING_VAR_PCT - Warning VaR percentage (default: 0.08)
 * - NEXT_PUBLIC_MAX_DRAWDOWN_PCT - Maximum drawdown (default: 0.15)
 * - NEXT_PUBLIC_WARNING_DRAWDOWN_PCT - Warning drawdown (default: 0.12)
 * - NEXT_PUBLIC_CRITICAL_DRAWDOWN_PCT - Critical drawdown (default: 0.20)
 * - NEXT_PUBLIC_MAX_CONCENTRATION_PCT - Max position concentration (default: 0.40)
 * - NEXT_PUBLIC_MIN_LIQUIDITY_SCORE - Minimum liquidity score (default: 0.7)
 *
 * ### External APIs
 * - NEXT_PUBLIC_TWELVEDATA_API_KEY - TwelveData API key
 */

// ============================================================================
// Main Configuration Exports
// ============================================================================

export { apiConfig, buildApiUrl, apiTimeouts, retryConfig, isDevelopment, isProduction } from './api.config';
export type { } from './api.config';

export {
  marketConfig,
  timeframeConfig,
  refreshIntervals,
  marketHours,
  dataRangeConfig,
  tradingSession,
  marketDataConfig,
  volumeProfileConfig,
  getTimeframeInfo,
  getRefreshInterval,
  isMarketOpen,
} from './market.config';

export {
  riskConfig,
  riskThresholds,
  calculateRiskSeverity,
  breachesThreshold,
  getRecommendedAction,
} from './risk.config';

export {
  chartConfig,
  tableConfig,
  layoutConfig,
  animationConfig,
  notificationConfig,
  loadingConfig,
  formConfig,
  a11yConfig,
  performanceConfig,
  dataDisplayConfig,
  getChartHeight,
  getThemeColors,
} from './ui.config';

// Models configuration
export {
  modelsApiEndpoints,
  defaultModelColors,
  statusConfig,
  signalConfig,
  periodOptions,
  modelRefreshIntervals,
  getModelColor,
  getStatusBadgeProps,
  getSignalBadgeProps,
  formatDuration,
  formatPnL,
  formatPct,
  getPnLColorClass,
  isProductionModel,
  isActiveModel,
} from './models.config';

// Real-time configuration (WebSocket, SSE)
export {
  realtimeConfig,
  getRealtimeConfig,
  isWithinMarketHours,
  getOptimalPollingInterval,
} from './realtime.config';

export type { RealtimeConfig } from './realtime.config';

export type {
  ModelAlgorithm,
  ModelStatus,
  SignalType,
  TradeStatus,
  ModelConfig,
  ModelSignal,
  ModelTrade,
  TradesSummary,
  ModelMetrics,
  EquityCurvePoint,
  ModelComparison,
  PeriodOption,
} from './models.config';

// ============================================================================
// Unified Configuration Object
// ============================================================================

import { apiConfig } from './api.config';
import { marketConfig, timeframeConfig, refreshIntervals, marketHours } from './market.config';
import { riskConfig, riskThresholds } from './risk.config';
import uiConfigDefault from './ui.config';
import modelsConfigDefault, { modelsApiEndpoints, modelRefreshIntervals } from './models.config';
import { realtimeConfig } from './realtime.config';

/**
 * Unified configuration object
 * Provides access to all configuration domains from a single import
 */
const config = {
  /**
   * API configuration
   */
  api: apiConfig,

  /**
   * Market configuration
   */
  market: {
    ...marketConfig,
    timeframes: timeframeConfig,
    refreshIntervals,
    hours: marketHours,
  },

  /**
   * Risk configuration
   */
  risk: {
    ...riskConfig,
    thresholds: riskThresholds,
  },

  /**
   * UI configuration
   */
  ui: uiConfigDefault,

  /**
   * Models configuration
   */
  models: {
    ...modelsConfigDefault,
    endpoints: modelsApiEndpoints,
    refreshIntervals: modelRefreshIntervals,
  },

  /**
   * Real-time configuration (WebSocket, SSE)
   */
  realtime: realtimeConfig,

  /**
   * Application metadata
   */
  app: {
    name: 'USDCOP Trading Dashboard',
    version: '3.0.0',
    environment: process.env.NODE_ENV || 'development',
    buildTime: new Date().toISOString(),
  },
} as const;

export default config;

// ============================================================================
// Configuration Validation
// ============================================================================

/**
 * Validate configuration on import (development only)
 */
if (process.env.NODE_ENV === 'development' && typeof window !== 'undefined') {
  console.group('📋 Configuration System Loaded');
  console.log('API Endpoints:', {
    trading: apiConfig.trading.baseUrl,
    analytics: apiConfig.analytics.baseUrl,
    pipeline: apiConfig.pipeline.baseUrl,
  });
  console.log('Market Settings:', {
    symbol: marketConfig.defaultSymbol,
    defaultTimeframe: timeframeConfig.defaultTimeframe,
    portfolioValue: marketConfig.defaultPortfolioValue,
  });
  console.log('Risk Thresholds:', {
    maxLeverage: riskConfig.leverage.max,
    maxVaR: `${riskConfig.var.maxPercentage * 100}%`,
    maxDrawdown: `${riskConfig.drawdown.max * 100}%`,
  });
  console.groupEnd();
}

// ============================================================================
// Type Exports
// ============================================================================

/**
 * Configuration type for type-safe access
 */
export type Config = typeof config;

/**
 * API configuration type
 */
export type ApiConfig = typeof apiConfig;

/**
 * Market configuration type
 */
export type MarketConfig = typeof marketConfig;

/**
 * Risk configuration type
 */
export type RiskConfig = typeof riskConfig;

/**
 * UI configuration type
 */
export type UiConfig = typeof uiConfigDefault;

/**
 * Timeframe type
 */
export type Timeframe = typeof timeframeConfig.supported[number]['value'];

/**
 * Chart type
 */
export type ChartType = typeof uiConfigDefault.chart.types.available[number];

/**
 * Theme type
 */
export type Theme = 'light' | 'dark';

/**
 * Risk severity type
 */
export type RiskSeverity = 'low' | 'medium' | 'high' | 'critical';

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get configuration value by path
 * Example: getConfig('api.trading.baseUrl')
 */
export function getConfig(path: string): any {
  return path.split('.').reduce((obj, key) => obj?.[key], config as any);
}

/**
 * Check if feature is enabled
 */
export function isFeatureEnabled(feature: string): boolean {
  const featureFlags = {
    performanceMonitoring: uiConfigDefault.performance.enableMonitoring,
    animations: uiConfigDefault.animation.enabled,
    keyboardNav: uiConfigDefault.a11y.enableKeyboardNav,
  };

  return featureFlags[feature as keyof typeof featureFlags] ?? false;
}

/**
 * Get environment name
 */
export function getEnvironment(): 'development' | 'production' | 'test' {
  return (process.env.NODE_ENV as any) || 'development';
}

/**
 * Check if running in browser
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined';
}

/**
 * Check if running on server
 */
export function isServer(): boolean {
  return typeof window === 'undefined';
}
