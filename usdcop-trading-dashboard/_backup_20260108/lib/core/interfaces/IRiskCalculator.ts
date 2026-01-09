/**
 * IRiskCalculator Interface
 * ==========================
 *
 * Interface for risk calculation and management operations.
 */

import { Position, Order, OrderSide } from '@/types/trading';

/**
 * Risk metrics for a position or portfolio
 */
export interface RiskMetrics {
  exposure: number;
  valueAtRisk: number;
  expectedShortfall: number;
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  beta?: number;
  leverage?: number;
}

/**
 * Position sizing result
 */
export interface PositionSize {
  quantity: number;
  notionalValue: number;
  riskAmount: number;
  riskPercent: number;
  leverage: number;
}

/**
 * Stop loss and take profit levels
 */
export interface RiskLevels {
  stopLoss: number;
  takeProfit: number;
  breakeven: number;
  riskRewardRatio: number;
}

/**
 * Risk limits configuration
 */
export interface RiskLimits {
  maxPositionSize: number;
  maxLeverage: number;
  maxDrawdown: number;
  maxDailyLoss: number;
  maxExposure: number;
  maxCorrelation?: number;
}

/**
 * Risk calculator interface
 */
export interface IRiskCalculator {
  /**
   * Calculate risk metrics for a position
   */
  calculatePositionRisk(position: Position): RiskMetrics;

  /**
   * Calculate portfolio risk metrics
   */
  calculatePortfolioRisk(positions: Position[]): RiskMetrics;

  /**
   * Calculate position size based on risk parameters
   */
  calculatePositionSize(
    accountBalance: number,
    entryPrice: number,
    stopLoss: number,
    riskPercent: number
  ): PositionSize;

  /**
   * Calculate stop loss and take profit levels
   */
  calculateRiskLevels(
    entryPrice: number,
    side: OrderSide,
    atr: number,
    riskRewardRatio?: number
  ): RiskLevels;

  /**
   * Validate if a trade meets risk requirements
   */
  validateTrade(
    order: Partial<Order>,
    currentPositions: Position[],
    accountBalance: number,
    limits: RiskLimits
  ): { isValid: boolean; violations: string[] };

  /**
   * Calculate Value at Risk (VaR)
   */
  calculateVaR(
    positions: Position[],
    confidenceLevel: number,
    timeHorizon: number
  ): number;

  /**
   * Calculate Expected Shortfall (CVaR)
   */
  calculateExpectedShortfall(
    positions: Position[],
    confidenceLevel: number
  ): number;
}

/**
 * Extended risk calculator with advanced features
 */
export interface IAdvancedRiskCalculator extends IRiskCalculator {
  /**
   * Calculate correlation between positions
   */
  calculateCorrelation(position1: Position, position2: Position): number;

  /**
   * Calculate portfolio beta
   */
  calculateBeta(positions: Position[], benchmark: string): number;

  /**
   * Optimize position sizing across portfolio
   */
  optimizePortfolio(
    positions: Position[],
    constraints: RiskLimits
  ): Record<string, PositionSize>;

  /**
   * Simulate risk scenarios
   */
  runStressTest(
    positions: Position[],
    scenarios: Array<{ name: string; priceChanges: Record<string, number> }>
  ): Array<{ scenario: string; pnl: number; metrics: RiskMetrics }>;

  /**
   * Calculate margin requirements
   */
  calculateMarginRequirement(
    position: Position,
    leverage: number
  ): number;

  /**
   * Calculate liquidation price
   */
  calculateLiquidationPrice(
    position: Position,
    leverage: number,
    maintenanceMargin: number
  ): number;
}

/**
 * Risk calculator configuration
 */
export interface RiskCalculatorConfig {
  defaultRiskPercent: number;
  defaultRiskRewardRatio: number;
  varConfidenceLevel: number;
  varTimeHorizon: number;
  atrMultiplier: number;
  limits: RiskLimits;
}
