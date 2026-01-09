/**
 * Financial Metrics Calculator
 * Core calculations for trading performance metrics
 */

import {
  Trade,
  Position,
  FinancialMetrics,
  MetricsOptions,
  TimePeriod,
  MetricsCalculationError,
  EquityPoint,
} from './types';

export class MetricsCalculator {
  private static readonly DEFAULT_OPTIONS: Required<MetricsOptions> = {
    riskFreeRate: 0.03,
    confidenceLevel: 0.95,
    tradingDaysPerYear: 252,
    initialCapital: 100000,
    includeOpenPositions: true,
  };

  /**
   * Calculate comprehensive financial metrics from trades and positions
   */
  static calculateMetrics(
    trades: Trade[],
    positions: Position[] = [],
    options: MetricsOptions = {}
  ): FinancialMetrics {
    const opts = { ...this.DEFAULT_OPTIONS, ...options };

    try {
      // Filter to only closed trades for most calculations
      const closedTrades = trades.filter(t => t.status === 'closed');

      // Calculate P&L metrics
      const pnlMetrics = this.calculatePnLMetrics(closedTrades, positions, opts);

      // Calculate trade statistics
      const tradeStats = this.calculateTradeStatistics(closedTrades);

      // Calculate returns array for ratio calculations
      const returns = this.calculateReturns(closedTrades, opts.initialCapital);

      // Calculate performance ratios
      const ratios = this.calculatePerformanceRatios(returns, opts);

      // Calculate risk metrics
      const riskMetrics = this.calculateRiskMetrics(returns, opts);

      // Calculate position metrics
      const positionMetrics = this.calculatePositionMetrics(positions);

      // Calculate time-based metrics
      const timeMetrics = this.calculateTimeMetrics(closedTrades);

      // Build equity curve
      const equityCurve = this.buildEquityCurve(closedTrades, opts.initialCapital);

      // Find drawdowns
      const drawdownCurve = this.calculateDrawdowns(equityCurve);

      return {
        ...pnlMetrics,
        ...tradeStats,
        ...ratios,
        ...riskMetrics,
        ...positionMetrics,
        ...timeMetrics,
        equityCurve,
        drawdownCurve,
      };
    } catch (error) {
      throw new MetricsCalculationError(
        'Failed to calculate metrics',
        { error, tradesCount: trades.length, positionsCount: positions.length }
      );
    }
  }

  /**
   * Calculate P&L metrics
   */
  private static calculatePnLMetrics(
    trades: Trade[],
    positions: Position[],
    options: Required<MetricsOptions>
  ) {
    const realizedPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
    const unrealizedPnL = options.includeOpenPositions
      ? positions.reduce((sum, p) => sum + p.unrealizedPnL, 0)
      : 0;
    const totalPnL = realizedPnL + unrealizedPnL;

    const now = Date.now();
    const oneDayAgo = now - 24 * 60 * 60 * 1000;
    const oneWeekAgo = now - 7 * 24 * 60 * 60 * 1000;
    const oneMonthAgo = now - 30 * 24 * 60 * 60 * 1000;

    const dailyPnL = trades
      .filter(t => t.exitTime && t.exitTime >= oneDayAgo)
      .reduce((sum, t) => sum + t.pnl, 0);

    const weeklyPnL = trades
      .filter(t => t.exitTime && t.exitTime >= oneWeekAgo)
      .reduce((sum, t) => sum + t.pnl, 0);

    const monthlyPnL = trades
      .filter(t => t.exitTime && t.exitTime >= oneMonthAgo)
      .reduce((sum, t) => sum + t.pnl, 0);

    // Calculate returns
    const totalReturn = totalPnL / options.initialCapital;
    const dailyReturn = dailyPnL / options.initialCapital;
    const weeklyReturn = weeklyPnL / options.initialCapital;
    const monthlyReturn = monthlyPnL / options.initialCapital;

    // Annualized return
    const tradingDays = this.calculateTradingDays(trades);
    const annualizedReturn = tradingDays > 0
      ? Math.pow(1 + totalReturn, options.tradingDaysPerYear / tradingDays) - 1
      : 0;

    return {
      totalPnL,
      realizedPnL,
      unrealizedPnL,
      dailyPnL,
      weeklyPnL,
      monthlyPnL,
      totalReturn,
      dailyReturn,
      weeklyReturn,
      monthlyReturn,
      annualizedReturn,
    };
  }

  /**
   * Calculate trade statistics
   */
  private static calculateTradeStatistics(trades: Trade[]) {
    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl <= 0);

    const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));

    const avgWin = winningTrades.length > 0
      ? grossProfit / winningTrades.length
      : 0;

    const avgLoss = losingTrades.length > 0
      ? grossLoss / losingTrades.length
      : 0;

    const largestWin = winningTrades.length > 0
      ? Math.max(...winningTrades.map(t => t.pnl))
      : 0;

    const largestLoss = losingTrades.length > 0
      ? Math.min(...losingTrades.map(t => t.pnl))
      : 0;

    const avgTradeDuration = trades.length > 0
      ? trades
          .filter(t => t.duration !== undefined)
          .reduce((sum, t) => sum + (t.duration || 0), 0) / trades.length
      : 0;

    const expectancy = trades.length > 0
      ? trades.reduce((sum, t) => sum + t.pnl, 0) / trades.length
      : 0;

    const payoffRatio = avgLoss > 0 ? avgWin / avgLoss : 0;

    return {
      totalTrades: trades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate: trades.length > 0 ? winningTrades.length / trades.length : 0,
      avgWin,
      avgLoss,
      largestWin,
      largestLoss,
      avgTradeDuration,
      expectancy,
      payoffRatio,
      profitFactor: grossLoss > 0 ? grossProfit / grossLoss : 0,
    };
  }

  /**
   * Calculate returns array from trades
   */
  private static calculateReturns(trades: Trade[], initialCapital: number): number[] {
    let capital = initialCapital;
    const returns: number[] = [];

    for (const trade of trades.sort((a, b) => a.exitTime! - b.exitTime!)) {
      const returnPct = trade.pnl / capital;
      returns.push(returnPct);
      capital += trade.pnl;
    }

    return returns;
  }

  /**
   * Calculate performance ratios
   */
  private static calculatePerformanceRatios(
    returns: number[],
    options: Required<MetricsOptions>
  ) {
    if (returns.length === 0) {
      return {
        sharpeRatio: 0,
        sortinoRatio: 0,
        calmarRatio: 0,
      };
    }

    const mean = this.mean(returns);
    const stdDev = this.standardDeviation(returns);
    const downsideReturns = returns.filter(r => r < 0);
    const downsideStdDev = this.standardDeviation(downsideReturns);

    const dailyRiskFreeRate = options.riskFreeRate / options.tradingDaysPerYear;
    const annualizationFactor = Math.sqrt(options.tradingDaysPerYear);

    const sharpeRatio = stdDev > 0
      ? ((mean - dailyRiskFreeRate) / stdDev) * annualizationFactor
      : 0;

    const sortinoRatio = downsideStdDev > 0
      ? ((mean - dailyRiskFreeRate) / downsideStdDev) * annualizationFactor
      : 0;

    return {
      sharpeRatio,
      sortinoRatio,
      calmarRatio: 0, // Will be calculated with max drawdown
    };
  }

  /**
   * Calculate risk metrics
   */
  private static calculateRiskMetrics(
    returns: number[],
    options: Required<MetricsOptions>
  ) {
    if (returns.length === 0) {
      return {
        maxDrawdown: 0,
        maxDrawdownPercent: 0,
        currentDrawdown: 0,
        currentDrawdownPercent: 0,
        valueAtRisk95: 0,
        expectedShortfall: 0,
        volatility: 0,
        downsideVolatility: 0,
      };
    }

    const stdDev = this.standardDeviation(returns);
    const downsideReturns = returns.filter(r => r < 0);
    const downsideStdDev = this.standardDeviation(downsideReturns);

    const volatility = stdDev * Math.sqrt(options.tradingDaysPerYear);
    const downsideVolatility = downsideStdDev * Math.sqrt(options.tradingDaysPerYear);

    // VaR and CVaR
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const varIndex = Math.floor((1 - options.confidenceLevel) * sortedReturns.length);
    const valueAtRisk95 = sortedReturns[varIndex] || 0;

    const tailReturns = sortedReturns.slice(0, varIndex + 1);
    const expectedShortfall = tailReturns.length > 0
      ? tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length
      : 0;

    return {
      maxDrawdown: 0, // Calculated separately from equity curve
      maxDrawdownPercent: 0,
      currentDrawdown: 0,
      currentDrawdownPercent: 0,
      valueAtRisk95,
      expectedShortfall,
      volatility,
      downsideVolatility,
    };
  }

  /**
   * Calculate position metrics
   */
  private static calculatePositionMetrics(positions: Position[]) {
    const openPositions = positions.length;

    const avgPositionSize = positions.length > 0
      ? positions.reduce((sum, p) => sum + p.quantity, 0) / positions.length
      : 0;

    const largestPosition = positions.length > 0
      ? Math.max(...positions.map(p => p.quantity))
      : 0;

    const exposure = positions.reduce((sum, p) => sum + (p.quantity * p.currentPrice), 0);

    return {
      openPositions,
      avgPositionSize,
      largestPosition,
      exposure,
    };
  }

  /**
   * Calculate time-based metrics
   */
  private static calculateTimeMetrics(trades: Trade[]) {
    if (trades.length === 0) {
      return {
        firstTradeTime: null,
        lastTradeTime: null,
        tradingDays: 0,
        profitablePercentOfTime: 0,
        avgDailyReturn: 0,
        returnStdDev: 0,
        informationRatio: 0,
        kellyFraction: 0,
      };
    }

    const sortedTrades = [...trades].sort((a, b) => a.exitTime! - b.exitTime!);
    const firstTradeTime = sortedTrades[0].exitTime || sortedTrades[0].entryTime;
    const lastTradeTime = sortedTrades[sortedTrades.length - 1].exitTime || Date.now();

    const tradingDays = this.calculateTradingDays(trades);

    // Calculate profitable time percentage
    const winningTrades = trades.filter(t => t.pnl > 0);
    const profitablePercentOfTime = trades.length > 0
      ? winningTrades.length / trades.length
      : 0;

    // Daily statistics
    const returns = trades.map(t => t.pnlPercent / 100);
    const avgDailyReturn = this.mean(returns);
    const returnStdDev = this.standardDeviation(returns);

    // Information ratio (simplified)
    const informationRatio = returnStdDev > 0 ? avgDailyReturn / returnStdDev : 0;

    // Kelly fraction (simplified)
    const winRate = trades.length > 0 ? winningTrades.length / trades.length : 0;
    const avgWinReturn = winningTrades.length > 0
      ? winningTrades.reduce((sum, t) => sum + t.pnlPercent, 0) / winningTrades.length / 100
      : 0;
    const losingTrades = trades.filter(t => t.pnl <= 0);
    const avgLossReturn = losingTrades.length > 0
      ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnlPercent, 0) / losingTrades.length / 100)
      : 0;

    const kellyFraction = avgLossReturn > 0
      ? (winRate * avgWinReturn - (1 - winRate) * avgLossReturn) / avgWinReturn
      : 0;

    return {
      firstTradeTime,
      lastTradeTime,
      tradingDays,
      profitablePercentOfTime,
      avgDailyReturn,
      returnStdDev,
      informationRatio,
      kellyFraction: Math.max(0, Math.min(1, kellyFraction * 0.5)), // Half Kelly for safety
    };
  }

  /**
   * Build equity curve from trades
   */
  private static buildEquityCurve(trades: Trade[], initialCapital: number): EquityPoint[] {
    const curve: EquityPoint[] = [{
      timestamp: Date.now() - 30 * 24 * 60 * 60 * 1000, // 30 days ago as starting point
      value: initialCapital,
      cumReturn: 0,
      drawdown: 0,
      drawdownPercent: 0,
    }];

    let capital = initialCapital;
    let peak = initialCapital;

    const sortedTrades = [...trades]
      .filter(t => t.exitTime)
      .sort((a, b) => a.exitTime! - b.exitTime!);

    for (const trade of sortedTrades) {
      capital += trade.pnl;
      peak = Math.max(peak, capital);

      const drawdown = peak - capital;
      const drawdownPercent = peak > 0 ? (drawdown / peak) * 100 : 0;

      curve.push({
        timestamp: trade.exitTime!,
        value: capital,
        cumReturn: ((capital - initialCapital) / initialCapital) * 100,
        drawdown,
        drawdownPercent,
      });
    }

    return curve;
  }

  /**
   * Calculate drawdown periods
   */
  private static calculateDrawdowns(equityCurve: EquityPoint[]) {
    const drawdowns: any[] = [];
    let currentDrawdown: any = null;
    let peak = equityCurve[0]?.value || 0;
    let peakTime = equityCurve[0]?.timestamp || Date.now();

    for (const point of equityCurve) {
      if (point.value > peak) {
        // New peak - end current drawdown if exists
        if (currentDrawdown) {
          currentDrawdown.end = point.timestamp;
          currentDrawdown.duration = point.timestamp - currentDrawdown.start;
          currentDrawdown.recovered = true;
          drawdowns.push(currentDrawdown);
          currentDrawdown = null;
        }
        peak = point.value;
        peakTime = point.timestamp;
      } else if (point.value < peak) {
        // In drawdown
        if (!currentDrawdown) {
          currentDrawdown = {
            start: peakTime,
            peak,
            trough: point.value,
            value: peak - point.value,
            percent: ((peak - point.value) / peak) * 100,
            duration: 0,
            recovered: false,
          };
        } else {
          // Update if deeper
          if (point.value < currentDrawdown.trough) {
            currentDrawdown.trough = point.value;
            currentDrawdown.value = peak - point.value;
            currentDrawdown.percent = ((peak - point.value) / peak) * 100;
          }
        }
      }
    }

    // Add current drawdown if not recovered
    if (currentDrawdown) {
      currentDrawdown.duration = Date.now() - currentDrawdown.start;
      drawdowns.push(currentDrawdown);
    }

    return drawdowns.sort((a, b) => b.percent - a.percent);
  }

  /**
   * Calculate trading days
   */
  private static calculateTradingDays(trades: Trade[]): number {
    if (trades.length === 0) return 0;

    const sortedTrades = [...trades].sort((a, b) => a.exitTime! - b.exitTime!);
    const firstTrade = sortedTrades[0];
    const lastTrade = sortedTrades[sortedTrades.length - 1];

    const startTime = firstTrade.exitTime || firstTrade.entryTime;
    const endTime = lastTrade.exitTime || Date.now();

    return Math.max(1, Math.ceil((endTime - startTime) / (24 * 60 * 60 * 1000)));
  }

  /**
   * Statistical helpers
   */
  private static mean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  }

  private static standardDeviation(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = this.mean(values);
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  /**
   * Update Calmar ratio once max drawdown is calculated
   */
  static updateCalmarRatio(metrics: FinancialMetrics): FinancialMetrics {
    const maxDrawdownAbs = Math.abs(metrics.maxDrawdownPercent);
    const calmarRatio = maxDrawdownAbs > 0
      ? (metrics.annualizedReturn * 100) / maxDrawdownAbs
      : 0;

    return {
      ...metrics,
      calmarRatio,
    };
  }
}
