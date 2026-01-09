/**
 * Real-time Configuration
 * =======================
 * Configuration for WebSocket, SSE, and real-time features.
 * By default, real-time features are ENABLED in production.
 */

export interface RealtimeConfig {
  // WebSocket configuration
  websocket: {
    enabled: boolean;
    url: string;
    autoConnect: boolean;
    maxReconnectAttempts: number;
    heartbeatInterval: number;
    connectionTimeout: number;
  };

  // SSE (Server-Sent Events) configuration
  sse: {
    enabled: boolean;
    equityCurveUrl: string;
    signalsUrl: string;
    metricsUrl: string;
    fallbackToPolling: boolean;
    pollingInterval: number;
  };

  // Polling fallback configuration
  polling: {
    equityCurveInterval: number;
    signalsInterval: number;
    performanceInterval: number;
    marketDataInterval: number;
  };

  // Market hours (USD/COP)
  marketHours: {
    enabled: boolean;
    timezone: string;
    openHour: number;
    openMinute: number;
    closeHour: number;
    closeMinute: number;
    tradingDays: number[]; // 1 = Monday, 5 = Friday
  };
}

// Environment detection
const isProduction = process.env.NODE_ENV === 'production';
const isDevelopment = process.env.NODE_ENV === 'development';

// API URLs from environment
const MULTI_MODEL_API_URL =
  process.env.NEXT_PUBLIC_MULTI_MODEL_API_URL || 'http://localhost:8006';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

/**
 * Default real-time configuration
 * WebSocket and SSE are ENABLED by default
 */
export const realtimeConfig: RealtimeConfig = {
  websocket: {
    // WebSocket is ENABLED by default in all environments
    enabled: true,
    url: WS_URL,
    // Auto-connect on page load
    autoConnect: true,
    maxReconnectAttempts: isProduction ? 15 : 5,
    heartbeatInterval: 30000, // 30 seconds
    connectionTimeout: isProduction ? 15000 : 10000,
  },

  sse: {
    // SSE is ENABLED by default
    enabled: true,
    equityCurveUrl: `${MULTI_MODEL_API_URL}/api/stream/equity-curves`,
    signalsUrl: `${MULTI_MODEL_API_URL}/api/stream/signals`,
    metricsUrl: `${MULTI_MODEL_API_URL}/api/stream/metrics`,
    // Fall back to polling if SSE fails
    fallbackToPolling: true,
    pollingInterval: 10000, // 10 seconds
  },

  polling: {
    // Polling intervals (used as fallback)
    equityCurveInterval: 60000, // 1 minute
    signalsInterval: 5000, // 5 seconds
    performanceInterval: 30000, // 30 seconds
    marketDataInterval: isProduction ? 3000 : 5000, // 3-5 seconds
  },

  marketHours: {
    // Market hours awareness is ENABLED
    enabled: true,
    timezone: 'America/Bogota',
    openHour: 8,
    openMinute: 0,
    closeHour: 12,
    closeMinute: 55,
    tradingDays: [1, 2, 3, 4, 5], // Monday to Friday
  },
};

/**
 * Get real-time configuration
 * Allows runtime overrides via environment variables
 */
export function getRealtimeConfig(): RealtimeConfig {
  return {
    ...realtimeConfig,
    websocket: {
      ...realtimeConfig.websocket,
      // Allow disable via env var for debugging
      enabled:
        process.env.NEXT_PUBLIC_DISABLE_WEBSOCKET !== 'true' &&
        realtimeConfig.websocket.enabled,
    },
    sse: {
      ...realtimeConfig.sse,
      // Allow disable via env var for debugging
      enabled:
        process.env.NEXT_PUBLIC_DISABLE_SSE !== 'true' &&
        realtimeConfig.sse.enabled,
    },
  };
}

/**
 * Check if we're within market hours
 */
export function isWithinMarketHours(): boolean {
  if (!realtimeConfig.marketHours.enabled) {
    return true; // If disabled, always return true
  }

  const now = new Date();
  const bogotaTime = new Date(
    now.toLocaleString('en-US', { timeZone: realtimeConfig.marketHours.timezone })
  );

  const dayOfWeek = bogotaTime.getDay();
  const currentHour = bogotaTime.getHours();
  const currentMinute = bogotaTime.getMinutes();

  // Check if it's a trading day
  if (!realtimeConfig.marketHours.tradingDays.includes(dayOfWeek)) {
    return false;
  }

  // Check if within trading hours
  const currentTime = currentHour * 60 + currentMinute;
  const openTime =
    realtimeConfig.marketHours.openHour * 60 + realtimeConfig.marketHours.openMinute;
  const closeTime =
    realtimeConfig.marketHours.closeHour * 60 + realtimeConfig.marketHours.closeMinute;

  return currentTime >= openTime && currentTime <= closeTime;
}

/**
 * Get optimal polling interval based on market status
 */
export function getOptimalPollingInterval(type: 'equity' | 'signals' | 'performance' | 'market'): number {
  const config = realtimeConfig.polling;

  // During market hours, use shorter intervals
  const marketOpen = isWithinMarketHours();

  switch (type) {
    case 'equity':
      return marketOpen ? config.equityCurveInterval / 2 : config.equityCurveInterval;
    case 'signals':
      return marketOpen ? config.signalsInterval : config.signalsInterval * 2;
    case 'performance':
      return marketOpen ? config.performanceInterval : config.performanceInterval * 2;
    case 'market':
      return marketOpen ? config.marketDataInterval : config.marketDataInterval * 3;
    default:
      return 10000;
  }
}

export default realtimeConfig;
