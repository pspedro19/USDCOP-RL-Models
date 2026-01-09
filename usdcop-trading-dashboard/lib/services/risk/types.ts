/**
 * Risk-Related Type Definitions
 * ==============================
 *
 * Centralized type definitions for the risk management system.
 * These types are shared across all risk management modules.
 */

/**
 * Position interface for risk calculations
 */
export interface Position {
  symbol: string;
  quantity: number;
  marketValue: number;
  avgPrice: number;
  currentPrice: number;
  pnl: number;
  weight: number;
  sector: string;
  country: string;
  currency: string;
}

/**
 * Risk alert interface
 */
export interface RiskAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  position?: string;
  currentValue?: number;
  limitValue?: number;
  recommendation?: string;
  details?: Record<string, any>;
}

/**
 * Comprehensive risk metrics interface
 */
export interface RealTimeRiskMetrics {
  // Portfolio metrics
  portfolioValue: number;
  grossExposure: number;
  netExposure: number;
  leverage: number;

  // Risk measures
  portfolioVaR95: number;
  portfolioVaR99: number;
  expectedShortfall95: number;
  portfolioVolatility: number;

  // Drawdown metrics
  currentDrawdown: number;
  maximumDrawdown: number;

  // Liquidity metrics
  liquidityScore: number;
  timeToLiquidate: number;

  // Scenario analysis
  bestCaseScenario: number;
  worstCaseScenario: number;
  stressTestResults?: Record<string, number>;

  // Timestamps
  lastUpdated: Date;
  calculationTime: number;
}

/**
 * Risk threshold configuration
 */
export interface RiskThresholds {
  maxLeverage: number;
  varLimit: number;
  maxDrawdown: number;
  minLiquidityScore: number;
  maxConcentration: number;
}

/**
 * Subscribable interface for pub-sub pattern
 */
export interface ISubscribable<T> {
  subscribe(callback: (data: T) => void): void;
  unsubscribe(callback: (data: T) => void): void;
}

/**
 * API response interface for risk metrics
 */
export interface RiskMetricsApiResponse {
  risk_metrics: {
    portfolioValue: number;
    grossExposure: number;
    netExposure: number;
    leverage: number;
    portfolioVaR95: number;
    portfolioVaR99: number;
    expectedShortfall95: number;
    portfolioVolatility: number;
    currentDrawdown: number;
    maximumDrawdown: number;
    liquidityScore: number;
    timeToLiquidate: number;
    bestCaseScenario: number;
    worstCaseScenario: number;
  };
}

/**
 * Backwards compatibility interface
 */
export interface RiskMetrics {
  var: number;
  exposure: number;
  leverage: number;
  positions: number;
}
