/**
 * Real-Time Risk Engine
 * ======================
 *
 * Orchestrator for the risk management system.
 * Follows Single Responsibility Principle by delegating to specialized components:
 * - PortfolioTracker: Manages positions
 * - RiskMetricsCalculator: Calculates risk metrics
 * - RiskAlertSystem: Manages alerts
 *
 * This class is responsible ONLY for:
 * - Coordinating the specialized components
 * - Managing real-time update lifecycle
 * - Implementing the subscription pattern for metrics updates
 * - Providing a unified API for clients
 */

import { createLogger } from '@/lib/utils/logger';
import { PortfolioTracker } from './PortfolioTracker';
import { RiskMetricsCalculator } from './RiskMetricsCalculator';
import { RiskAlertSystem } from './RiskAlertSystem';
import type {
  Position,
  RiskAlert,
  RealTimeRiskMetrics,
  RiskThresholds,
  ISubscribable,
} from './types';

const logger = createLogger('RealTimeRiskEngine');

/**
 * Default risk thresholds
 * TODO: These should be loaded from a configuration service
 */
const DEFAULT_RISK_THRESHOLDS: RiskThresholds = {
  maxLeverage: 5.0,
  varLimit: 0.10, // 10% of portfolio value
  maxDrawdown: 0.15, // 15%
  minLiquidityScore: 0.7,
  maxConcentration: 0.4, // 40% in single position
};

export class RealTimeRiskEngine implements ISubscribable<RealTimeRiskMetrics> {
  private portfolioTracker: PortfolioTracker;
  private metricsCalculator: RiskMetricsCalculator;
  private alertSystem: RiskAlertSystem;

  private subscribers: ((metrics: RealTimeRiskMetrics) => void)[] = [];
  private currentMetrics: RealTimeRiskMetrics | null = null;
  private updateInterval: NodeJS.Timeout | null = null;
  private riskThresholds: RiskThresholds;

  private readonly UPDATE_INTERVAL_MS = 30000; // 30 seconds

  constructor(thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS) {
    this.portfolioTracker = new PortfolioTracker();
    this.metricsCalculator = new RiskMetricsCalculator();
    this.alertSystem = new RiskAlertSystem();
    this.riskThresholds = thresholds;

    logger.info('Real-Time Risk Engine initialized', {
      thresholds: this.riskThresholds,
    });

    this.initialize();
  }

  /**
   * Initialize the risk engine
   */
  private async initialize(): Promise<void> {
    await this.initializeMetrics();
    this.startRealTimeUpdates();
  }

  /**
   * Fetch initial metrics from API
   */
  private async initializeMetrics(): Promise<void> {
    logger.info('Initializing risk metrics from API');
    const metrics = await this.metricsCalculator.fetchRiskMetricsFromAPI();

    if (metrics) {
      this.currentMetrics = metrics;
      this.alertSystem.checkRiskThresholds(metrics, this.riskThresholds);
      logger.info('Risk metrics initialized successfully');
    } else {
      logger.warn('Failed to initialize risk metrics - no data available');
      this.currentMetrics = null;
    }
  }

  /**
   * Update or add a position to the portfolio
   */
  updatePosition(position: Position): void {
    this.portfolioTracker.updatePosition(position);
    this.recalculateMetrics();
    this.checkAlerts();
  }

  /**
   * Remove a position from the portfolio
   */
  removePosition(symbol: string): void {
    this.portfolioTracker.removePosition(symbol);
    this.recalculateMetrics();
    this.checkAlerts();
  }

  /**
   * Get current risk metrics
   */
  getRiskMetrics(): RealTimeRiskMetrics | null {
    return this.currentMetrics;
  }

  /**
   * Subscribe to real-time risk metric updates
   */
  subscribe(callback: (metrics: RealTimeRiskMetrics) => void): void {
    this.subscribers.push(callback);
    logger.debug('New subscriber added', {
      totalSubscribers: this.subscribers.length,
    });

    // Immediately send current metrics to new subscriber
    if (this.currentMetrics) {
      try {
        callback(this.currentMetrics);
      } catch (error) {
        logger.error('Error notifying new subscriber', error);
      }
    }
  }

  /**
   * Unsubscribe from updates
   */
  unsubscribe(callback: (metrics: RealTimeRiskMetrics) => void): void {
    const index = this.subscribers.indexOf(callback);
    if (index > -1) {
      this.subscribers.splice(index, 1);
      logger.debug('Subscriber removed', {
        totalSubscribers: this.subscribers.length,
      });
    }
  }

  /**
   * Get risk alerts
   */
  getAlerts(unacknowledgedOnly: boolean = false): RiskAlert[] {
    return this.alertSystem.getAlerts(unacknowledgedOnly);
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string): boolean {
    return this.alertSystem.acknowledgeAlert(alertId);
  }

  /**
   * Get alert statistics
   */
  getAlertStatistics() {
    return this.alertSystem.getAlertStatistics();
  }

  /**
   * Get portfolio summary
   */
  getPortfolioSummary() {
    return this.portfolioTracker.getPortfolioSummary();
  }

  /**
   * Update risk thresholds
   */
  updateRiskThresholds(thresholds: Partial<RiskThresholds>): void {
    this.riskThresholds = {
      ...this.riskThresholds,
      ...thresholds,
    };

    logger.info('Risk thresholds updated', { thresholds: this.riskThresholds });

    // Re-check alerts with new thresholds
    if (this.currentMetrics) {
      this.checkAlerts();
    }
  }

  /**
   * Recalculate metrics based on current positions
   */
  private recalculateMetrics(): void {
    if (!this.currentMetrics) {
      logger.warn('Cannot recalculate metrics - no base metrics available');
      return;
    }

    const positions = this.portfolioTracker.getAllPositions();
    const updatedMetrics = this.metricsCalculator.calculateMetricsFromPositions(
      positions,
      this.currentMetrics
    );

    this.currentMetrics = updatedMetrics;
    this.notifySubscribers();
  }

  /**
   * Check for risk alerts
   */
  private checkAlerts(): void {
    if (!this.currentMetrics) return;

    this.alertSystem.checkRiskThresholds(
      this.currentMetrics,
      this.riskThresholds
    );
  }

  /**
   * Notify all subscribers of metric updates
   */
  private notifySubscribers(): void {
    if (!this.currentMetrics || this.subscribers.length === 0) return;

    logger.debug('Notifying subscribers', {
      subscriberCount: this.subscribers.length,
    });

    this.subscribers.forEach((callback) => {
      try {
        callback(this.currentMetrics!);
      } catch (error) {
        logger.error('Error notifying subscriber', error);
      }
    });
  }

  /**
   * Start real-time updates from API
   */
  private startRealTimeUpdates(): void {
    logger.info(`Starting real-time updates (interval: ${this.UPDATE_INTERVAL_MS}ms)`);

    this.updateInterval = setInterval(() => {
      this.refreshMetricsFromAPI();
    }, this.UPDATE_INTERVAL_MS);
  }

  /**
   * Refresh metrics from Analytics API
   */
  private async refreshMetricsFromAPI(): Promise<void> {
    try {
      const portfolioValue = this.currentMetrics?.portfolioValue;
      const metrics = await this.metricsCalculator.fetchRiskMetricsFromAPI(
        'USDCOP',
        portfolioValue
      );

      if (metrics) {
        this.currentMetrics = metrics;
        this.checkAlerts();
        this.notifySubscribers();
        logger.debug('Metrics refreshed from API');
      } else {
        logger.warn('Failed to refresh metrics from API');
      }
    } catch (error) {
      logger.error('Error refreshing metrics from API', error);
    }
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    logger.info('Destroying Real-Time Risk Engine');

    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }

    this.subscribers = [];
    this.portfolioTracker.clearAllPositions();
    this.alertSystem.clearAllAlerts();
    this.currentMetrics = null;

    logger.info('Real-Time Risk Engine destroyed');
  }
}
