/**
 * Risk Alert System
 * ==================
 *
 * Single Responsibility: Manage risk alerts.
 * This class is responsible ONLY for:
 * - Generating risk alerts based on metrics and thresholds
 * - Managing alert lifecycle (create, acknowledge, retrieve)
 * - Maintaining alert history
 */

import { createLogger } from '@/lib/utils/logger';
import type {
  RiskAlert,
  RealTimeRiskMetrics,
  RiskThresholds,
} from './types';

const logger = createLogger('RiskAlertSystem');

export class RiskAlertSystem {
  private alerts: RiskAlert[] = [];
  private readonly MAX_ALERTS = 100;

  /**
   * Check metrics against thresholds and generate alerts
   */
  checkRiskThresholds(
    metrics: RealTimeRiskMetrics,
    thresholds: RiskThresholds
  ): RiskAlert[] {
    const newAlerts: RiskAlert[] = [];

    // Check leverage limit
    if (metrics.leverage > thresholds.maxLeverage) {
      newAlerts.push(
        this.createLeverageAlert(metrics.leverage, thresholds.maxLeverage)
      );
    }

    // Check VaR limit
    const varRatio = metrics.portfolioVaR95 / metrics.portfolioValue;
    if (varRatio > thresholds.varLimit) {
      newAlerts.push(
        this.createVarAlert(
          varRatio,
          thresholds.varLimit,
          metrics.portfolioValue,
          metrics.portfolioVaR95
        )
      );
    }

    // Check drawdown
    if (Math.abs(metrics.currentDrawdown) > thresholds.maxDrawdown) {
      newAlerts.push(
        this.createDrawdownAlert(
          metrics.currentDrawdown,
          thresholds.maxDrawdown,
          metrics.maximumDrawdown
        )
      );
    }

    // Check liquidity score
    if (metrics.liquidityScore < thresholds.minLiquidityScore) {
      newAlerts.push(
        this.createLiquidityAlert(
          metrics.liquidityScore,
          thresholds.minLiquidityScore
        )
      );
    }

    // Add new alerts to the system
    if (newAlerts.length > 0) {
      this.addAlerts(newAlerts);
      logger.warn(`Generated ${newAlerts.length} new risk alerts`, {
        alertTypes: newAlerts.map((a) => a.type),
      });
    }

    return newAlerts;
  }

  /**
   * Create a leverage alert
   */
  private createLeverageAlert(
    currentLeverage: number,
    maxLeverage: number
  ): RiskAlert {
    return {
      id: `leverage-${Date.now()}`,
      severity: 'critical',
      type: 'leverage_limit',
      message: `Portfolio leverage (${currentLeverage.toFixed(2)}x) exceeds maximum allowed (${maxLeverage}x)`,
      currentValue: currentLeverage,
      limitValue: maxLeverage,
      timestamp: new Date(),
      acknowledged: false,
      recommendation: 'Reduce position sizes or close some positions to lower leverage',
      details: {
        currentLeverage,
        maxLeverage,
        breach: currentLeverage - maxLeverage,
      },
    };
  }

  /**
   * Create a VaR alert
   */
  private createVarAlert(
    varRatio: number,
    varLimit: number,
    portfolioValue: number,
    varAmount: number
  ): RiskAlert {
    return {
      id: `var-${Date.now()}`,
      severity: 'high',
      type: 'var_breach',
      message: `Portfolio VaR (${(varRatio * 100).toFixed(1)}%) exceeds limit (${(varLimit * 100).toFixed(1)}%)`,
      currentValue: varRatio,
      limitValue: varLimit,
      timestamp: new Date(),
      acknowledged: false,
      recommendation: 'Consider hedging strategies or reducing risk exposure',
      details: {
        currentVaR: varRatio,
        limit: varLimit,
        portfolioValue,
        varAmount,
      },
    };
  }

  /**
   * Create a drawdown alert
   */
  private createDrawdownAlert(
    currentDrawdown: number,
    maxDrawdown: number,
    maximumDrawdown: number
  ): RiskAlert {
    return {
      id: `drawdown-${Date.now()}`,
      severity: 'critical',
      type: 'drawdown',
      message: `Current drawdown (${(currentDrawdown * 100).toFixed(1)}%) exceeds maximum (${(maxDrawdown * 100).toFixed(1)}%)`,
      currentValue: Math.abs(currentDrawdown),
      limitValue: maxDrawdown,
      timestamp: new Date(),
      acknowledged: false,
      recommendation: 'Review trading strategy and consider position reduction',
      details: {
        currentDrawdown,
        maxDrawdown,
        maximumDrawdown,
      },
    };
  }

  /**
   * Create a liquidity alert
   */
  private createLiquidityAlert(
    currentScore: number,
    minScore: number
  ): RiskAlert {
    return {
      id: `liquidity-${Date.now()}`,
      severity: 'medium',
      type: 'liquidity',
      message: `Liquidity score (${currentScore.toFixed(2)}) below minimum threshold (${minScore})`,
      currentValue: currentScore,
      limitValue: minScore,
      timestamp: new Date(),
      acknowledged: false,
      recommendation: 'Increase cash reserves or reduce illiquid positions',
      details: {
        currentScore,
        minScore,
        deficit: minScore - currentScore,
      },
    };
  }

  /**
   * Add alerts to the system
   */
  private addAlerts(newAlerts: RiskAlert[]): void {
    this.alerts.push(...newAlerts);

    // Keep only the most recent alerts
    if (this.alerts.length > this.MAX_ALERTS) {
      this.alerts = this.alerts.slice(-this.MAX_ALERTS);
      logger.debug(`Trimmed alert history to ${this.MAX_ALERTS} most recent alerts`);
    }
  }

  /**
   * Get all alerts, optionally filtering for unacknowledged only
   */
  getAlerts(unacknowledgedOnly: boolean = false): RiskAlert[] {
    if (unacknowledgedOnly) {
      return this.alerts.filter((alert) => !alert.acknowledged);
    }
    return [...this.alerts];
  }

  /**
   * Acknowledge a specific alert
   */
  acknowledgeAlert(alertId: string): boolean {
    const alert = this.alerts.find((a) => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      logger.info(`Alert acknowledged: ${alertId}`, { type: alert.type });
      return true;
    }

    logger.warn(`Attempted to acknowledge non-existent alert: ${alertId}`);
    return false;
  }

  /**
   * Acknowledge all alerts
   */
  acknowledgeAllAlerts(): number {
    let count = 0;
    this.alerts.forEach((alert) => {
      if (!alert.acknowledged) {
        alert.acknowledged = true;
        count++;
      }
    });

    logger.info(`Acknowledged ${count} alerts`);
    return count;
  }

  /**
   * Clear all alerts
   */
  clearAllAlerts(): void {
    const count = this.alerts.length;
    this.alerts = [];
    logger.info(`Cleared ${count} alerts from system`);
  }

  /**
   * Get alert statistics
   */
  getAlertStatistics(): {
    total: number;
    unacknowledged: number;
    bySeverity: Record<string, number>;
    byType: Record<string, number>;
  } {
    const unacknowledged = this.alerts.filter((a) => !a.acknowledged).length;

    const bySeverity = this.alerts.reduce(
      (acc, alert) => {
        acc[alert.severity] = (acc[alert.severity] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    const byType = this.alerts.reduce(
      (acc, alert) => {
        acc[alert.type] = (acc[alert.type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    return {
      total: this.alerts.length,
      unacknowledged,
      bySeverity,
      byType,
    };
  }
}
