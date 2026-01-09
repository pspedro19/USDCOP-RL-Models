/**
 * Risk Metrics Calculator
 * ========================
 *
 * Single Responsibility: Calculate risk metrics.
 * This class is responsible ONLY for:
 * - Calculating portfolio risk metrics (VaR, leverage, volatility, etc.)
 * - Fetching risk data from the Analytics API
 * - Computing derived metrics from position data
 */

import { createLogger } from '@/lib/utils/logger';
import type {
  Position,
  RealTimeRiskMetrics,
  RiskMetricsApiResponse,
} from './types';

const logger = createLogger('RiskMetricsCalculator');

export class RiskMetricsCalculator {
  private readonly ANALYTICS_API_URL = '/api/analytics';
  private readonly DEFAULT_PORTFOLIO_VALUE = 10000000;

  /**
   * Fetch risk metrics from the Analytics API
   */
  async fetchRiskMetricsFromAPI(
    symbol: string = 'USDCOP',
    portfolioValue?: number,
    days: number = 30
  ): Promise<RealTimeRiskMetrics | null> {
    // Skip if running on server-side
    if (typeof window === 'undefined') {
      logger.debug('Skipping API fetch during SSR/SSG build');
      return null;
    }

    const effectivePortfolioValue =
      portfolioValue ||
      this.getPortfolioValueFromEnvironment() ||
      this.DEFAULT_PORTFOLIO_VALUE;

    try {
      logger.info('Fetching risk metrics from API', {
        symbol,
        portfolioValue: effectivePortfolioValue,
        days,
      });

      const url = `${this.ANALYTICS_API_URL}/risk-metrics?symbol=${symbol}&portfolio_value=${effectivePortfolioValue}&days=${days}`;
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        logger.error(`Analytics API returned status ${response.status}`);
        logger.error('NO DATA AVAILABLE - API must be running to show risk metrics');
        return null;
      }

      const data: RiskMetricsApiResponse = await response.json();
      const metrics = data.risk_metrics;

      logger.info('Successfully fetched REAL risk metrics from API');

      return this.mapApiResponseToMetrics(metrics);
    } catch (error) {
      logger.error('Failed to fetch risk metrics from API', error);
      logger.error('Check that Analytics API is running on http://localhost:8001');
      return null;
    }
  }

  /**
   * Calculate risk metrics from position data
   */
  calculateMetricsFromPositions(
    positions: Position[],
    baseMetrics?: RealTimeRiskMetrics
  ): RealTimeRiskMetrics {
    const startTime = Date.now();

    // Calculate portfolio value
    const portfolioValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);
    const grossExposure = positions.reduce(
      (sum, pos) => sum + Math.abs(pos.marketValue),
      0
    );
    const netExposure = Math.abs(portfolioValue);
    const leverage = grossExposure / Math.max(portfolioValue, 1);

    // Calculate volatility-based metrics
    const volatility = this.calculatePortfolioVolatility(positions);
    const var95 = portfolioValue * volatility * 1.645; // 95% confidence z-score
    const var99 = portfolioValue * volatility * 2.326; // 99% confidence z-score
    const expectedShortfall95 = var95 * 1.6; // Rough approximation

    const calculationTime = Date.now() - startTime;

    logger.debug('Calculated metrics from positions', {
      positionCount: positions.length,
      portfolioValue,
      leverage,
      calculationTime,
    });

    return {
      portfolioValue,
      grossExposure,
      netExposure,
      leverage,
      portfolioVaR95: Math.abs(var95),
      portfolioVaR99: Math.abs(var99),
      expectedShortfall95: Math.abs(expectedShortfall95),
      portfolioVolatility: volatility,
      currentDrawdown: baseMetrics?.currentDrawdown || 0,
      maximumDrawdown: baseMetrics?.maximumDrawdown || 0,
      liquidityScore: baseMetrics?.liquidityScore || 1.0,
      timeToLiquidate: baseMetrics?.timeToLiquidate || 0,
      bestCaseScenario: baseMetrics?.bestCaseScenario || 0,
      worstCaseScenario: baseMetrics?.worstCaseScenario || 0,
      stressTestResults: baseMetrics?.stressTestResults,
      lastUpdated: new Date(),
      calculationTime,
    };
  }

  /**
   * Calculate portfolio volatility (simplified placeholder)
   * In production, this should use historical returns data
   */
  private calculatePortfolioVolatility(positions: Position[]): number {
    if (positions.length === 0) return 0;

    // This is a placeholder - actual volatility should come from Analytics API
    logger.warn('Portfolio volatility calculation not implemented - using 0');
    logger.warn('This should be provided by Analytics API risk-metrics endpoint');
    return 0;
  }

  /**
   * Get portfolio value from environment or window
   */
  private getPortfolioValueFromEnvironment(): number | null {
    if (typeof window !== 'undefined' && (window as any).PORTFOLIO_VALUE) {
      return (window as any).PORTFOLIO_VALUE;
    }

    if (process.env.NEXT_PUBLIC_PORTFOLIO_VALUE) {
      return parseFloat(process.env.NEXT_PUBLIC_PORTFOLIO_VALUE);
    }

    return null;
  }

  /**
   * Map API response to internal metrics format
   */
  private mapApiResponseToMetrics(
    apiMetrics: RiskMetricsApiResponse['risk_metrics']
  ): RealTimeRiskMetrics {
    return {
      portfolioValue: apiMetrics.portfolioValue,
      grossExposure: apiMetrics.grossExposure,
      netExposure: apiMetrics.netExposure,
      leverage: apiMetrics.leverage,
      portfolioVaR95: apiMetrics.portfolioVaR95,
      portfolioVaR99: apiMetrics.portfolioVaR99,
      expectedShortfall95: apiMetrics.expectedShortfall95,
      portfolioVolatility: apiMetrics.portfolioVolatility,
      currentDrawdown: apiMetrics.currentDrawdown,
      maximumDrawdown: apiMetrics.maximumDrawdown,
      liquidityScore: apiMetrics.liquidityScore,
      timeToLiquidate: apiMetrics.timeToLiquidate,
      bestCaseScenario: apiMetrics.bestCaseScenario,
      worstCaseScenario: apiMetrics.worstCaseScenario,
      lastUpdated: new Date(),
      calculationTime: 125, // ms - could be extracted from API response
    };
  }

  /**
   * Merge position-based metrics with API metrics
   */
  mergeMetrics(
    apiMetrics: RealTimeRiskMetrics,
    positionMetrics: Partial<RealTimeRiskMetrics>
  ): RealTimeRiskMetrics {
    return {
      ...apiMetrics,
      ...positionMetrics,
      lastUpdated: new Date(),
    };
  }
}
