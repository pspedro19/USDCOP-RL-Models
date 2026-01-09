/**
 * Market Configuration
 * ====================
 * Market-related configuration including symbols, timeframes, and refresh intervals
 */

/**
 * Trading Symbol Configuration
 */
export const marketConfig = {
  /**
   * Default trading symbol
   * Override: NEXT_PUBLIC_DEFAULT_SYMBOL
   */
  defaultSymbol: (process.env.NEXT_PUBLIC_DEFAULT_SYMBOL || 'USDCOP') as string,

  /**
   * Supported trading symbols
   */
  supportedSymbols: ['USDCOP', 'USDBRL', 'USDMXN', 'USDCLP'] as const,

  /**
   * Default portfolio value (for risk calculations)
   * Override: NEXT_PUBLIC_PORTFOLIO_VALUE
   */
  defaultPortfolioValue: parseFloat(process.env.NEXT_PUBLIC_PORTFOLIO_VALUE || '10000000'),

  /**
   * Currency display settings
   */
  currency: {
    symbol: 'COP',
    decimals: 2,
    locale: 'es-CO',
  },

  /**
   * Price display settings
   */
  price: {
    decimals: 4,
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  },
} as const;

/**
 * Timeframe Configuration
 */
export const timeframeConfig = {
  /**
   * Default timeframe for charts
   */
  defaultTimeframe: '5m' as const,

  /**
   * Supported timeframes with display names and intervals
   */
  supported: [
    { value: '1m', label: '1 Minute', interval: 60, seconds: 60 },
    { value: '5m', label: '5 Minutes', interval: 300, seconds: 300 },
    { value: '15m', label: '15 Minutes', interval: 900, seconds: 900 },
    { value: '30m', label: '30 Minutes', interval: 1800, seconds: 1800 },
    { value: '1h', label: '1 Hour', interval: 3600, seconds: 3600 },
    { value: '4h', label: '4 Hours', interval: 14400, seconds: 14400 },
    { value: '1d', label: '1 Day', interval: 86400, seconds: 86400 },
  ] as const,

  /**
   * Timeframe groups for UI organization
   */
  groups: {
    intraday: ['1m', '5m', '15m', '30m'],
    hourly: ['1h', '4h'],
    daily: ['1d'],
  } as const,
} as const;

/**
 * Refresh Interval Configuration (in milliseconds)
 */
export const refreshIntervals = {
  /**
   * Real-time market data updates
   */
  realtime: 1000, // 1 second

  /**
   * Market statistics updates
   */
  stats: 30000, // 30 seconds

  /**
   * Position updates
   */
  positions: 5000, // 5 seconds

  /**
   * Analytics updates
   */
  analytics: 60000, // 1 minute

  /**
   * Risk metrics updates
   */
  risk: 30000, // 30 seconds

  /**
   * Health check updates
   */
  health: 60000, // 1 minute

  /**
   * Database statistics updates
   */
  database: 60000, // 1 minute

  /**
   * Chart data refresh
   */
  chart: 10000, // 10 seconds

  /**
   * Pipeline status updates
   */
  pipeline: 120000, // 2 minutes

  /**
   * Market status check
   */
  marketStatus: 60000, // 1 minute

  /**
   * Trading session updates
   */
  session: 60000, // 1 minute

  /**
   * Execution metrics
   */
  execution: 30000, // 30 seconds

  /**
   * Agent actions refresh
   */
  agentActions: 30000, // 30 seconds
} as const;

/**
 * Market Hours Configuration
 */
export const marketHours = {
  /**
   * Colombian market hours (COT - UTC-5)
   */
  colombia: {
    open: '09:00',
    close: '16:00',
    timezone: 'America/Bogota',
  },

  /**
   * US market hours (EST/EDT)
   */
  us: {
    open: '09:30',
    close: '16:00',
    timezone: 'America/New_York',
  },
} as const;

/**
 * Data Range Configuration
 */
export const dataRangeConfig = {
  /**
   * Default date range for historical data
   */
  default: {
    start: '2020-01-01',
    end: '2025-12-31',
  },

  /**
   * Maximum lookback period in days
   */
  maxLookbackDays: 365 * 5, // 5 years

  /**
   * Default number of candles to load
   */
  defaultCandleCount: 100000,

  /**
   * Maximum number of candles to display
   */
  maxDisplayCandles: 10000,
} as const;

/**
 * Trading Session Configuration
 */
export const tradingSession = {
  /**
   * Session types
   */
  types: ['live', 'paper', 'backtest'] as const,

  /**
   * Default session type
   */
  defaultType: 'paper' as const,
} as const;

/**
 * Market Data Configuration
 */
export const marketDataConfig = {
  /**
   * Minimum required data points for analysis
   */
  minDataPoints: 100,

  /**
   * Data quality thresholds
   */
  quality: {
    minCompleteness: 0.95, // 95%
    maxGapDuration: 300, // 5 minutes in seconds
    maxStaleness: 60, // 1 minute in seconds
  },

  /**
   * OHLC validation rules
   */
  ohlcValidation: {
    enabled: true,
    strictMode: false,
  },
} as const;

/**
 * Volume Profile Configuration
 */
export const volumeProfileConfig = {
  /**
   * Number of price levels for volume profile
   */
  priceLevels: 50,

  /**
   * Volume profile display mode
   */
  displayMode: 'overlay' as 'overlay' | 'sidebar',

  /**
   * Value area percentage (typically 70%)
   */
  valueAreaPercentage: 0.7,
} as const;

/**
 * Helper function to get timeframe info
 */
export function getTimeframeInfo(timeframe: string) {
  return timeframeConfig.supported.find((tf) => tf.value === timeframe);
}

/**
 * Helper function to get refresh interval
 */
export function getRefreshInterval(type: keyof typeof refreshIntervals): number {
  return refreshIntervals[type];
}

/**
 * Helper function to check if market is open
 * Note: This is a simplified check - actual implementation should consider holidays
 */
export function isMarketOpen(timezone: 'colombia' | 'us' = 'colombia'): boolean {
  const now = new Date();
  const hours = marketHours[timezone];

  // Get current time in the market's timezone
  const timeString = now.toLocaleTimeString('en-US', {
    timeZone: hours.timezone,
    hour12: false,
  });

  const [hour, minute] = timeString.split(':').map(Number);
  const currentMinutes = hour * 60 + minute;

  const [openHour, openMinute] = hours.open.split(':').map(Number);
  const openMinutes = openHour * 60 + openMinute;

  const [closeHour, closeMinute] = hours.close.split(':').map(Number);
  const closeMinutes = closeHour * 60 + closeMinute;

  return currentMinutes >= openMinutes && currentMinutes < closeMinutes;
}

export default marketConfig;
