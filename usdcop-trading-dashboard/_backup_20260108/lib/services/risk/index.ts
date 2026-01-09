/**
 * Risk Management Module
 * =======================
 *
 * Modular risk management system following Single Responsibility Principle.
 * Each component has a single, well-defined responsibility.
 *
 * Architecture:
 * - PortfolioTracker: Manages position tracking
 * - RiskMetricsCalculator: Calculates risk metrics
 * - RiskAlertSystem: Manages alerts
 * - RealTimeRiskEngine: Orchestrates all components
 *
 * Usage:
 *   import { realTimeRiskEngine } from '@/lib/services/risk';
 *
 *   // Subscribe to updates
 *   realTimeRiskEngine.subscribe((metrics) => {
 *     console.log('New metrics:', metrics);
 *   });
 *
 *   // Update positions
 *   realTimeRiskEngine.updatePosition(position);
 *
 *   // Get alerts
 *   const alerts = realTimeRiskEngine.getAlerts(true); // unacknowledged only
 */

// Export type definitions
export type {
  Position,
  RiskAlert,
  RealTimeRiskMetrics,
  RiskThresholds,
  ISubscribable,
  RiskMetricsApiResponse,
  RiskMetrics,
} from './types';

// Export classes
export { PortfolioTracker } from './PortfolioTracker';
export { RiskMetricsCalculator } from './RiskMetricsCalculator';
export { RiskAlertSystem } from './RiskAlertSystem';
export { RealTimeRiskEngine } from './RealTimeRiskEngine';

// Create and export singleton instance
import { RealTimeRiskEngine } from './RealTimeRiskEngine';
export const realTimeRiskEngine = new RealTimeRiskEngine();

/**
 * Backwards compatibility function
 * Maps new RealTimeRiskMetrics to legacy RiskMetrics format
 */
export async function getRiskMetrics(): Promise<import('./types').RiskMetrics | null> {
  const metrics = realTimeRiskEngine.getRiskMetrics();
  if (!metrics) {
    console.warn('[getRiskMetrics] No risk metrics available - Analytics API may not be running');
    console.warn('[getRiskMetrics] Returning null to force UI to show "No data" state');
    return null;
  }

  const portfolioSummary = realTimeRiskEngine.getPortfolioSummary();

  return {
    var: metrics.portfolioVaR95 / metrics.portfolioValue,
    exposure: metrics.netExposure / metrics.portfolioValue,
    leverage: metrics.leverage,
    positions: portfolioSummary.positionCount,
  };
}
