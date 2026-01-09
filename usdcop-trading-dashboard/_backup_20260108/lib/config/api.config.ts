/**
 * API Configuration
 * =================
 * Centralized API endpoint configuration
 * All backend service URLs are managed here
 */

/**
 * API Base URLs Configuration
 * Override using environment variables with NEXT_PUBLIC_ prefix
 */
export const apiConfig = {
  /**
   * Trading API - Real-time trading operations, orders, positions
   * Default: http://localhost:8000
   */
  trading: {
    baseUrl: process.env.NEXT_PUBLIC_TRADING_API_URL || 'http://localhost:8000',
    endpoints: {
      health: '/health',
      positions: '/api/positions',
      orderbook: '/api/orderbook',
      latest: '/api/latest',
      market: {
        update: '/api/market/update',
        realtime: '/api/market/realtime',
        volumeProfile: '/api/market/volume-profile',
        health: '/api/market/health',
      },
      signals: '/api/signals',
      execution: '/api/execution',
    },
  },

  /**
   * Analytics API - Trading analytics, metrics, RL performance
   * Default: http://localhost:8001
   */
  analytics: {
    baseUrl: process.env.NEXT_PUBLIC_ANALYTICS_API_URL || 'http://localhost:8001',
    endpoints: {
      rlMetrics: '/api/analytics/rl-metrics',
      performanceKpis: '/api/analytics/performance-kpis',
      marketStats: '/api/analytics/market-stats',
      marketConditions: '/api/analytics/market-conditions',
      riskMetrics: '/api/analytics/risk-metrics',
      marketOverview: '/api/analytics/market-overview',
    },
  },

  /**
   * Pipeline API - L0-L6 data pipeline endpoints
   * Default: http://localhost:8002
   */
  pipeline: {
    baseUrl: process.env.NEXT_PUBLIC_PIPELINE_API_URL || 'http://localhost:8002',
    endpoints: {
      health: '/api/pipeline/health',
      status: '/api/pipeline/status',
      consolidated: '/api/pipeline/consolidated',
      l0: {
        rawData: '/api/pipeline/l0/raw-data',
        statistics: '/api/pipeline/l0/statistics',
        status: '/api/pipeline/l0/status',
        health: '/api/l0/health',
      },
      l1: {
        episodes: '/api/pipeline/l1/episodes',
        qualityReport: '/api/pipeline/l1/quality-report',
        completeness: '/api/pipeline/l1/completeness',
        status: '/api/pipeline/l1/status',
      },
      l2: {
        preparedData: '/api/pipeline/l2/prepared-data',
        status: '/api/pipeline/l2/status',
      },
      l3: {
        features: '/api/pipeline/l3/features',
        status: '/api/pipeline/l3/status',
      },
      l4: {
        dataset: '/api/pipeline/l4/dataset',
        status: '/api/pipeline/l4/status',
      },
      l5: {
        status: '/api/pipeline/l5/status',
      },
      l6: {
        backtestResults: '/api/pipeline/l6/backtest-results',
        status: '/api/pipeline/l6/status',
      },
    },
  },

  /**
   * Compliance API - Regulatory compliance and auditing
   * Default: http://localhost:8003
   */
  compliance: {
    baseUrl: process.env.NEXT_PUBLIC_COMPLIANCE_API_URL || 'http://localhost:8003',
    endpoints: {
      signals: '/api/compliance/signals',
      audit: '/api/compliance/audit',
      reports: '/api/compliance/reports',
    },
  },

  /**
   * Backtest/Multi-Model API - Backtesting and multi-strategy analysis
   * Default: http://localhost:8006 or usdcop-multi-model-api:8006
   */
  backtest: {
    baseUrl: process.env.BACKTEST_API_URL || 'http://localhost:8006',
    multiModelUrl: process.env.MULTI_MODEL_API_URL || 'http://usdcop-multi-model-api:8006',
    endpoints: {
      trigger: '/api/backtest/trigger',
      results: '/api/backtest/results',
      multiStrategy: {
        signals: '/api/multi-strategy/signals',
        performance: '/api/multi-strategy/performance',
        equityCurves: '/api/multi-strategy/equity-curves',
      },
    },
  },

  /**
   * ML Analytics API - Machine learning model analytics
   * Default: http://localhost:8004
   */
  mlAnalytics: {
    baseUrl: process.env.NEXT_PUBLIC_ML_ANALYTICS_API_URL || 'http://localhost:8004',
    endpoints: {
      health: '/api/ml-analytics/health',
      predictions: '/api/ml-analytics/predictions',
      models: '/api/ml-analytics/models',
    },
  },

  /**
   * WebSocket Configuration
   */
  websocket: {
    tradingUrl: process.env.NEXT_PUBLIC_WS_TRADING_URL || 'ws://localhost:8000/ws',
    reconnectDelay: 3000,
    maxReconnectAttempts: 5,
    pingInterval: 30000,
  },

  /**
   * Next.js API Routes (Internal)
   * These are proxied through Next.js to avoid CORS issues
   * Dashboard runs on port 3001 to avoid conflict with MLflow (port 5000)
   */
  internal: {
    baseUrl: typeof window !== 'undefined' ? '' : 'http://localhost:3001',
    endpoints: {
      trading: '/api/trading',
      analytics: '/api/analytics',
      pipeline: '/api/pipeline',
      agent: {
        actions: '/api/agent/actions',
        state: '/api/agent/state',
      },
      alerts: {
        system: '/api/alerts/system',
      },
      backup: {
        status: '/api/backup/status',
      },
      health: '/api/health',
    },
  },

  /**
   * External APIs
   */
  external: {
    twelvedata: {
      baseUrl: 'https://api.twelvedata.com',
      apiKey: process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY || '',
    },
  },
} as const;

/**
 * Helper function to build full URL
 */
export function buildApiUrl(
  service: keyof typeof apiConfig,
  endpoint: string,
  params?: Record<string, string | number>
): string {
  const config = apiConfig[service] as { baseUrl: string };
  const baseUrl = config.baseUrl;

  if (!params) {
    return `${baseUrl}${endpoint}`;
  }

  const queryString = new URLSearchParams(
    Object.entries(params).reduce((acc, [key, value]) => {
      acc[key] = String(value);
      return acc;
    }, {} as Record<string, string>)
  ).toString();

  return `${baseUrl}${endpoint}?${queryString}`;
}

/**
 * Check if running in development mode
 */
export const isDevelopment = process.env.NODE_ENV === 'development';

/**
 * Check if running in production mode
 */
export const isProduction = process.env.NODE_ENV === 'production';

/**
 * API timeout configuration (in milliseconds)
 */
export const apiTimeouts = {
  default: 30000, // 30 seconds
  long: 60000, // 1 minute
  short: 10000, // 10 seconds
  realtime: 5000, // 5 seconds
} as const;

/**
 * Retry configuration for failed requests
 */
export const retryConfig = {
  maxAttempts: 3,
  backoffMultiplier: 2,
  initialDelay: 1000,
} as const;

export default apiConfig;
