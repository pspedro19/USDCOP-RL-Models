/**
 * Performance Analyzer
 * Advanced performance analysis and reporting
 */

import {
  Trade,
  Position,
  FinancialMetrics,
  PerformanceSummary,
  GroupedMetrics,
  TimePeriod,
  MetricsOptions,
} from './types';
import { MetricsCalculator } from './MetricsCalculator';
import { EquityCurveBuilder } from './EquityCurveBuilder';

export class PerformanceAnalyzer {
  /**
   * Generate comprehensive performance summary
   */
  static generateSummary(
    trades: Trade[],
    positions: Position[],
    options: MetricsOptions = {}
  ): PerformanceSummary {
    const metrics = MetricsCalculator.calculateMetrics(trades, positions, options);

    // Get top and worst trades
    const sortedByPnL = [...trades]
      .filter(t => t.status === 'closed')
      .sort((a, b) => b.pnl - a.pnl);

    const topTrades = sortedByPnL.slice(0, 10);
    const worstTrades = sortedByPnL.slice(-10).reverse();

    // Calculate recent activity
    const now = Date.now();
    const recentActivity = {
      last24h: this.countTradesInPeriod(trades, now - 24 * 60 * 60 * 1000, now),
      last7d: this.countTradesInPeriod(trades, now - 7 * 24 * 60 * 60 * 1000, now),
      last30d: this.countTradesInPeriod(trades, now - 30 * 24 * 60 * 60 * 1000, now),
    };

    // Analyze risk
    const riskIndicators = this.analyzeRisk(metrics);

    return {
      metrics,
      topTrades,
      worstTrades,
      recentActivity,
      riskIndicators,
    };
  }

  /**
   * Analyze risk levels
   */
  static analyzeRisk(metrics: FinancialMetrics): {
    isHighRisk: boolean;
    riskLevel: 'low' | 'medium' | 'high';
    warnings: string[];
  } {
    const warnings: string[] = [];
    let riskScore = 0;

    // Check drawdown
    if (Math.abs(metrics.maxDrawdownPercent) > 20) {
      warnings.push(`High maximum drawdown: ${Math.abs(metrics.maxDrawdownPercent).toFixed(1)}%`);
      riskScore += 3;
    } else if (Math.abs(metrics.maxDrawdownPercent) > 10) {
      riskScore += 1;
    }

    // Check current drawdown
    if (Math.abs(metrics.currentDrawdownPercent) > 15) {
      warnings.push(`Currently in significant drawdown: ${Math.abs(metrics.currentDrawdownPercent).toFixed(1)}%`);
      riskScore += 2;
    }

    // Check volatility
    if (metrics.volatility > 0.3) {
      warnings.push(`High volatility: ${(metrics.volatility * 100).toFixed(1)}%`);
      riskScore += 2;
    }

    // Check Sharpe ratio
    if (metrics.sharpeRatio < 0.5) {
      warnings.push(`Low Sharpe ratio: ${metrics.sharpeRatio.toFixed(2)}`);
      riskScore += 1;
    }

    // Check win rate
    if (metrics.winRate < 0.4) {
      warnings.push(`Low win rate: ${(metrics.winRate * 100).toFixed(1)}%`);
      riskScore += 1;
    }

    // Check profit factor
    if (metrics.profitFactor < 1.2) {
      warnings.push(`Low profit factor: ${metrics.profitFactor.toFixed(2)}`);
      riskScore += 1;
    }

    // Check exposure
    const exposureRatio = metrics.exposure / (metrics.totalPnL + 100000); // Assuming 100k base
    if (exposureRatio > 2) {
      warnings.push(`High market exposure: ${(exposureRatio * 100).toFixed(0)}%`);
      riskScore += 2;
    }

    // Determine risk level
    let riskLevel: 'low' | 'medium' | 'high';
    if (riskScore >= 7) {
      riskLevel = 'high';
    } else if (riskScore >= 4) {
      riskLevel = 'medium';
    } else {
      riskLevel = 'low';
    }

    return {
      isHighRisk: riskLevel === 'high',
      riskLevel,
      warnings,
    };
  }

  /**
   * Calculate metrics for specific time period
   */
  static analyzeTimePeriod(
    trades: Trade[],
    period: TimePeriod,
    options: MetricsOptions = {}
  ): FinancialMetrics {
    const filteredTrades = trades.filter(
      t => t.exitTime && t.exitTime >= period.start && t.exitTime <= period.end
    );

    return MetricsCalculator.calculateMetrics(filteredTrades, [], options);
  }

  /**
   * Group metrics by strategy
   */
  static groupByStrategy(
    trades: Trade[],
    positions: Position[],
    options: MetricsOptions = {}
  ): GroupedMetrics {
    const strategies = new Set<string>();

    trades.forEach(t => {
      if (t.strategy) strategies.add(t.strategy);
    });

    positions.forEach(p => {
      if (p.strategy) strategies.add(p.strategy);
    });

    const grouped: GroupedMetrics = {};

    for (const strategy of strategies) {
      const strategyTrades = trades.filter(t => t.strategy === strategy);
      const strategyPositions = positions.filter(p => p.strategy === strategy);

      grouped[strategy] = MetricsCalculator.calculateMetrics(
        strategyTrades,
        strategyPositions,
        options
      );
    }

    return grouped;
  }

  /**
   * Compare two time periods
   */
  static comparePeriods(
    trades: Trade[],
    period1: TimePeriod,
    period2: TimePeriod,
    options: MetricsOptions = {}
  ): {
    period1: FinancialMetrics;
    period2: FinancialMetrics;
    improvements: string[];
    deteriorations: string[];
  } {
    const metrics1 = this.analyzeTimePeriod(trades, period1, options);
    const metrics2 = this.analyzeTimePeriod(trades, period2, options);

    const improvements: string[] = [];
    const deteriorations: string[] = [];

    // Compare key metrics
    if (metrics2.sharpeRatio > metrics1.sharpeRatio) {
      improvements.push(`Sharpe Ratio improved: ${metrics1.sharpeRatio.toFixed(2)} → ${metrics2.sharpeRatio.toFixed(2)}`);
    } else if (metrics2.sharpeRatio < metrics1.sharpeRatio) {
      deteriorations.push(`Sharpe Ratio declined: ${metrics1.sharpeRatio.toFixed(2)} → ${metrics2.sharpeRatio.toFixed(2)}`);
    }

    if (metrics2.winRate > metrics1.winRate) {
      improvements.push(`Win Rate improved: ${(metrics1.winRate * 100).toFixed(1)}% → ${(metrics2.winRate * 100).toFixed(1)}%`);
    } else if (metrics2.winRate < metrics1.winRate) {
      deteriorations.push(`Win Rate declined: ${(metrics1.winRate * 100).toFixed(1)}% → ${(metrics2.winRate * 100).toFixed(1)}%`);
    }

    if (Math.abs(metrics2.maxDrawdownPercent) < Math.abs(metrics1.maxDrawdownPercent)) {
      improvements.push(`Max Drawdown improved: ${metrics1.maxDrawdownPercent.toFixed(1)}% → ${metrics2.maxDrawdownPercent.toFixed(1)}%`);
    } else if (Math.abs(metrics2.maxDrawdownPercent) > Math.abs(metrics1.maxDrawdownPercent)) {
      deteriorations.push(`Max Drawdown worsened: ${metrics1.maxDrawdownPercent.toFixed(1)}% → ${metrics2.maxDrawdownPercent.toFixed(1)}%`);
    }

    if (metrics2.profitFactor > metrics1.profitFactor) {
      improvements.push(`Profit Factor improved: ${metrics1.profitFactor.toFixed(2)} → ${metrics2.profitFactor.toFixed(2)}`);
    } else if (metrics2.profitFactor < metrics1.profitFactor) {
      deteriorations.push(`Profit Factor declined: ${metrics1.profitFactor.toFixed(2)} → ${metrics2.profitFactor.toFixed(2)}`);
    }

    return {
      period1: metrics1,
      period2: metrics2,
      improvements,
      deteriorations,
    };
  }

  /**
   * Calculate rolling metrics
   */
  static calculateRollingMetrics(
    trades: Trade[],
    windowDays: number,
    options: MetricsOptions = {}
  ): { timestamp: number; metrics: FinancialMetrics }[] {
    if (trades.length === 0) return [];

    const sortedTrades = [...trades].sort((a, b) => a.exitTime! - b.exitTime!);
    const windowMs = windowDays * 24 * 60 * 60 * 1000;
    const rolling: { timestamp: number; metrics: FinancialMetrics }[] = [];

    // Get unique timestamps
    const timestamps = [...new Set(sortedTrades.map(t => t.exitTime!))].sort();

    for (const timestamp of timestamps) {
      const windowStart = timestamp - windowMs;
      const windowTrades = sortedTrades.filter(
        t => t.exitTime! >= windowStart && t.exitTime! <= timestamp
      );

      if (windowTrades.length > 0) {
        const metrics = MetricsCalculator.calculateMetrics(windowTrades, [], options);
        rolling.push({ timestamp, metrics });
      }
    }

    return rolling;
  }

  /**
   * Calculate monthly returns
   */
  static calculateMonthlyReturns(
    trades: Trade[],
    initialCapital: number
  ): { month: string; return: number; pnl: number; trades: number }[] {
    const monthlyData = new Map<string, { pnl: number; trades: number }>();

    for (const trade of trades) {
      if (trade.status !== 'closed' || !trade.exitTime) continue;

      const date = new Date(trade.exitTime);
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;

      const existing = monthlyData.get(monthKey) || { pnl: 0, trades: 0 };
      monthlyData.set(monthKey, {
        pnl: existing.pnl + trade.pnl,
        trades: existing.trades + 1,
      });
    }

    return Array.from(monthlyData.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([month, data]) => ({
        month,
        return: (data.pnl / initialCapital) * 100,
        pnl: data.pnl,
        trades: data.trades,
      }));
  }

  /**
   * Calculate win/loss streaks
   */
  static calculateStreaks(trades: Trade[]): {
    currentStreak: { type: 'win' | 'loss'; count: number };
    longestWinStreak: number;
    longestLossStreak: number;
  } {
    const sortedTrades = [...trades]
      .filter(t => t.status === 'closed')
      .sort((a, b) => a.exitTime! - b.exitTime!);

    let currentStreakType: 'win' | 'loss' | null = null;
    let currentStreakCount = 0;
    let longestWinStreak = 0;
    let longestLossStreak = 0;
    let tempWinStreak = 0;
    let tempLossStreak = 0;

    for (const trade of sortedTrades) {
      const isWin = trade.pnl > 0;

      if (isWin) {
        tempWinStreak++;
        tempLossStreak = 0;
        longestWinStreak = Math.max(longestWinStreak, tempWinStreak);
      } else {
        tempLossStreak++;
        tempWinStreak = 0;
        longestLossStreak = Math.max(longestLossStreak, tempLossStreak);
      }

      // Track current streak
      if (currentStreakType === null || (isWin && currentStreakType === 'win') || (!isWin && currentStreakType === 'loss')) {
        currentStreakType = isWin ? 'win' : 'loss';
        currentStreakCount++;
      } else {
        currentStreakType = isWin ? 'win' : 'loss';
        currentStreakCount = 1;
      }
    }

    return {
      currentStreak: {
        type: currentStreakType || 'win',
        count: currentStreakCount,
      },
      longestWinStreak,
      longestLossStreak,
    };
  }

  /**
   * Calculate trade distribution
   */
  static calculateTradeDistribution(trades: Trade[]): {
    hourly: { [hour: number]: number };
    daily: { [day: number]: number };
    byPnLRange: { [range: string]: number };
  } {
    const hourly: { [hour: number]: number } = {};
    const daily: { [day: number]: number } = {};
    const byPnLRange: { [range: string]: number } = {
      'loss > -5%': 0,
      'loss -5% to -2%': 0,
      'loss -2% to 0%': 0,
      'gain 0% to 2%': 0,
      'gain 2% to 5%': 0,
      'gain > 5%': 0,
    };

    for (const trade of trades) {
      if (trade.status !== 'closed' || !trade.exitTime) continue;

      // Hourly distribution
      const hour = new Date(trade.exitTime).getHours();
      hourly[hour] = (hourly[hour] || 0) + 1;

      // Daily distribution (0 = Sunday, 6 = Saturday)
      const day = new Date(trade.exitTime).getDay();
      daily[day] = (daily[day] || 0) + 1;

      // P&L range distribution
      const pnlPercent = trade.pnlPercent;
      if (pnlPercent < -5) {
        byPnLRange['loss > -5%']++;
      } else if (pnlPercent < -2) {
        byPnLRange['loss -5% to -2%']++;
      } else if (pnlPercent < 0) {
        byPnLRange['loss -2% to 0%']++;
      } else if (pnlPercent < 2) {
        byPnLRange['gain 0% to 2%']++;
      } else if (pnlPercent < 5) {
        byPnLRange['gain 2% to 5%']++;
      } else {
        byPnLRange['gain > 5%']++;
      }
    }

    return { hourly, daily, byPnLRange };
  }

  /**
   * Helper: Count trades in period
   */
  private static countTradesInPeriod(trades: Trade[], start: number, end: number): number {
    return trades.filter(
      t => t.exitTime && t.exitTime >= start && t.exitTime <= end
    ).length;
  }
}
