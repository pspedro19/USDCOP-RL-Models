/**
 * Indicators & Analytics Engine
 * ============================
 *
 * Professional-grade technical indicators library for institutional trading
 * with advanced analytics, visualization, and backtesting capabilities.
 *
 * Key Features:
 * - 30+ optimized technical indicators
 * - Web Worker pool for parallel processing
 * - Advanced volume profile analytics
 * - Correlation matrix analysis
 * - Custom indicator builder
 * - Comprehensive backtesting framework
 * - Professional visualizations
 * - Institutional performance metrics
 */

// Core Engine
export { IndicatorEngine } from './engine/IndicatorEngine';
export type { IndicatorEngineConfig } from './engine/IndicatorEngine';

// Types
export * from './types';

// Indicator Calculators
export { TrendIndicators } from './calculators/TrendIndicators';
export { MomentumIndicators } from './calculators/MomentumIndicators';
export { VolumeIndicators } from './calculators/VolumeIndicators';

// Analytics Engines
export { VolumeAnalytics } from './analytics/VolumeAnalytics';
export type {
  VolumeAnalyticsConfig,
  MarketProfile,
  VolumeWeightedMetrics,
  TimeAndSalesAnalysis,
  SessionAnalytics
} from './analytics/VolumeAnalytics';

export { CorrelationAnalytics } from './analytics/CorrelationAnalytics';
export type {
  CorrelationAnalyticsConfig,
  DynamicCorrelation,
  RollingCorrelation,
  ClusterAnalysis,
  RiskFactorAnalysis,
  RegimeDetection,
  PortfolioRiskMetrics
} from './analytics/CorrelationAnalytics';

// Backtesting Framework
export { BacktestEngine } from './backtesting/BacktestEngine';
export type {
  BacktestConfig,
  TradingStrategy,
  TradingRule,
  RiskManagementConfig,
  PositionSizingConfig,
  MarketFilter,
  MonteCarloConfig,
  WalkForwardConfig,
  OptimizationConfig
} from './backtesting/BacktestEngine';

// Visualization Components
export { EChartsIndicator, VolumeProfileChart, CorrelationHeatmap } from './visualization/EChartsIndicators';
export { OrderFlowChart, MicrostructureChart, VolumeDeltaLadder } from './visualization/D3Indicators';

// Custom Indicator Builder
export { CustomIndicatorBuilder } from './builder/CustomIndicatorBuilder';

// Performance Analytics
export { PerformanceAnalytics } from './analytics/PerformanceAnalytics';

// Integration with Data Bus
export { IndicatorDataBusIntegration } from './integration/DataBusIntegration';

/**
 * Factory function to create a fully configured indicator engine
 */
export function createIndicatorEngine(config?: Partial<any>) {
  return new IndicatorEngine({
    maxWorkers: Math.min(navigator?.hardwareConcurrency || 4, 8),
    cacheSize: 1000,
    enableProfiling: true,
    workerTimeout: 30000,
    batchSize: 100,
    ...config
  });
}

/**
 * Pre-configured indicator sets for common use cases
 */
export const IndicatorPresets = {
  // Scalping indicators for short-term trading
  scalping: [
    { name: 'EMA_9', type: 'ema', period: 9, source: 'close' },
    { name: 'EMA_21', type: 'ema', period: 21, source: 'close' },
    { name: 'RSI_7', type: 'rsi', period: 7 },
    { name: 'MACD_5_13_9', type: 'macd', fastPeriod: 5, slowPeriod: 13, signalPeriod: 9 },
    { name: 'BB_20_2', type: 'bollinger', period: 20, stdDev: 2 },
    { name: 'Volume_SMA_20', type: 'sma', period: 20, source: 'volume' }
  ],

  // Swing trading indicators
  swing: [
    { name: 'SMA_50', type: 'sma', period: 50, source: 'close' },
    { name: 'EMA_20', type: 'ema', period: 20, source: 'close' },
    { name: 'RSI_14', type: 'rsi', period: 14 },
    { name: 'MACD_12_26_9', type: 'macd', fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 },
    { name: 'Stochastic_14_3', type: 'stochastic', kPeriod: 14, dPeriod: 3 },
    { name: 'ADX_14', type: 'adx', period: 14 },
    { name: 'Volume_Profile', type: 'volume_profile', levels: 50 }
  ],

  // Position trading indicators
  position: [
    { name: 'SMA_200', type: 'sma', period: 200, source: 'close' },
    { name: 'EMA_50', type: 'ema', period: 50, source: 'close' },
    { name: 'EMA_100', type: 'ema', period: 100, source: 'close' },
    { name: 'RSI_21', type: 'rsi', period: 21 },
    { name: 'MACD_12_26_9', type: 'macd', fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 },
    { name: 'Ichimoku', type: 'ichimoku', tenkanPeriod: 9, kijunPeriod: 26, senkouPeriod: 52 },
    { name: 'Volume_SMA_50', type: 'sma', period: 50, source: 'volume' }
  ],

  // Institutional analytics suite
  institutional: [
    { name: 'VWAP', type: 'vwap' },
    { name: 'Volume_Profile', type: 'volume_profile', levels: 100 },
    { name: 'Order_Flow', type: 'order_flow' },
    { name: 'Market_Structure', type: 'market_structure' },
    { name: 'TEMA_21', type: 'tema', period: 21 },
    { name: 'KAMA_14', type: 'kama', period: 14 },
    { name: 'Supertrend_10_3', type: 'supertrend', period: 10, multiplier: 3 },
    { name: 'Parabolic_SAR', type: 'psar', step: 0.02, max: 0.2 }
  ]
};

/**
 * Common trading strategies using indicators
 */
export const TradingStrategies = {
  // Mean reversion strategy
  meanReversion: {
    name: 'Mean Reversion',
    description: 'Buy oversold, sell overbought conditions',
    indicators: [
      { name: 'RSI_14', type: 'rsi', period: 14 },
      { name: 'BB_20_2', type: 'bollinger', period: 20, stdDev: 2 },
      { name: 'SMA_20', type: 'sma', period: 20 }
    ],
    entryRules: [
      { type: 'threshold', indicator: 'RSI_14', condition: 'less_than', value: 30 },
      { type: 'threshold', indicator: 'price', condition: 'less_than', indicator2: 'BB_lower' }
    ],
    exitRules: [
      { type: 'threshold', indicator: 'RSI_14', condition: 'greater_than', value: 70 },
      { type: 'indicator_crossover', indicator: 'price', condition: 'crosses_above', indicator2: 'SMA_20' }
    ]
  },

  // Trend following strategy
  trendFollowing: {
    name: 'Trend Following',
    description: 'Follow strong trends with momentum confirmation',
    indicators: [
      { name: 'EMA_20', type: 'ema', period: 20 },
      { name: 'EMA_50', type: 'ema', period: 50 },
      { name: 'MACD', type: 'macd', fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 },
      { name: 'ADX_14', type: 'adx', period: 14 }
    ],
    entryRules: [
      { type: 'indicator_crossover', indicator: 'EMA_20', condition: 'crosses_above', indicator2: 'EMA_50' },
      { type: 'threshold', indicator: 'MACD_histogram', condition: 'greater_than', value: 0 },
      { type: 'threshold', indicator: 'ADX_14', condition: 'greater_than', value: 25 }
    ],
    exitRules: [
      { type: 'indicator_crossover', indicator: 'EMA_20', condition: 'crosses_below', indicator2: 'EMA_50' },
      { type: 'threshold', indicator: 'MACD_histogram', condition: 'less_than', value: 0 }
    ]
  },

  // Breakout strategy
  breakout: {
    name: 'Breakout',
    description: 'Trade breakouts from consolidation patterns',
    indicators: [
      { name: 'BB_20_2', type: 'bollinger', period: 20, stdDev: 2 },
      { name: 'ATR_14', type: 'atr', period: 14 },
      { name: 'Volume_SMA_20', type: 'sma', period: 20, source: 'volume' }
    ],
    entryRules: [
      { type: 'threshold', indicator: 'price', condition: 'greater_than', indicator2: 'BB_upper' },
      { type: 'threshold', indicator: 'volume', condition: 'greater_than', indicator2: 'Volume_SMA_20' },
      { type: 'threshold', indicator: 'ATR_14', condition: 'greater_than', value: 0.01 }
    ],
    exitRules: [
      { type: 'threshold', indicator: 'price', condition: 'less_than', indicator2: 'BB_middle' }
    ]
  }
};

/**
 * Risk management templates
 */
export const RiskManagementTemplates = {
  conservative: {
    stopLoss: { type: 'percentage', value: 1 }, // 1% stop loss
    takeProfit: { type: 'risk_reward_ratio', value: 2 }, // 2:1 risk/reward
    maxDailyLoss: 0.02, // 2% of capital
    maxDrawdown: 0.05, // 5% maximum drawdown
    positionTimeout: 7 // Close after 7 days
  },

  moderate: {
    stopLoss: { type: 'percentage', value: 2 }, // 2% stop loss
    takeProfit: { type: 'risk_reward_ratio', value: 2.5 }, // 2.5:1 risk/reward
    maxDailyLoss: 0.03, // 3% of capital
    maxDrawdown: 0.08, // 8% maximum drawdown
    positionTimeout: 10 // Close after 10 days
  },

  aggressive: {
    stopLoss: { type: 'percentage', value: 3 }, // 3% stop loss
    takeProfit: { type: 'risk_reward_ratio', value: 3 }, // 3:1 risk/reward
    maxDailyLoss: 0.05, // 5% of capital
    maxDrawdown: 0.12, // 12% maximum drawdown
    positionTimeout: 14 // Close after 14 days
  }
};

/**
 * Performance benchmarks for different market conditions
 */
export const PerformanceBenchmarks = {
  bull_market: {
    sharpe: 1.5,
    maxDrawdown: 0.08,
    winRate: 0.55,
    profitFactor: 1.8
  },

  bear_market: {
    sharpe: 0.8,
    maxDrawdown: 0.15,
    winRate: 0.45,
    profitFactor: 1.3
  },

  sideways_market: {
    sharpe: 1.0,
    maxDrawdown: 0.10,
    winRate: 0.50,
    profitFactor: 1.5
  },

  volatile_market: {
    sharpe: 0.6,
    maxDrawdown: 0.20,
    winRate: 0.48,
    profitFactor: 1.2
  }
};