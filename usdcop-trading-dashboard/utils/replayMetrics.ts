/**
 * Replay Metrics Calculation
 *
 * Provides precise calculation of trading metrics during replay,
 * including an incremental calculator for real-time updates.
 */

import { ReplayTrade, EquityPoint, ReplayMetrics, EMPTY_METRICS } from '@/types/replay';

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

const TRADING_DAYS_PER_YEAR = 252;
const MINUTES_PER_DAY = 1440;
const RISK_FREE_RATE_ANNUAL = 0.05; // 5% annual

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICAL HELPERS
// ═══════════════════════════════════════════════════════════════════════════

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function standardDeviation(values: number[]): number {
  if (values.length < 2) return 0;
  const avg = mean(values);
  const squaredDiffs = values.map(v => Math.pow(v - avg, 2));
  return Math.sqrt(mean(squaredDiffs));
}

function maxConsecutive(values: boolean[]): number {
  let max = 0;
  let current = 0;
  for (const v of values) {
    if (v) {
      current++;
      max = Math.max(max, current);
    } else {
      current = 0;
    }
  }
  return max;
}

// ═══════════════════════════════════════════════════════════════════════════
// TRADE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get PnL from trade (handles multiple field names)
 */
function getTradePnL(trade: ReplayTrade): number {
  return trade.pnl_usd ?? trade.pnl ?? 0;
}

/**
 * Get hold time from trade (handles multiple field names)
 */
function getTradeHoldTime(trade: ReplayTrade): number {
  if (trade.hold_time_minutes) return trade.hold_time_minutes;
  if (trade.duration_minutes) return trade.duration_minutes;
  if (trade.duration_bars) return trade.duration_bars * 5; // Assuming 5-min bars
  return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN CALCULATION FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Calculate complete replay metrics from trades and equity curve
 */
export function calculateReplayMetrics(
  trades: ReplayTrade[],
  equityCurve: EquityPoint[]
): ReplayMetrics {
  // Empty case
  if (trades.length === 0) {
    return { ...EMPTY_METRICS };
  }

  // Extract PnLs and hold times
  const pnls = trades.map(getTradePnL);
  const holdTimes = trades.map(getTradeHoldTime);

  // Classify trades
  const wins = trades.filter(t => getTradePnL(t) > 0);
  const losses = trades.filter(t => getTradePnL(t) < 0);

  // PnL calculations
  const totalPnL = pnls.reduce((sum, p) => sum + p, 0);
  const grossProfit = wins.reduce((sum, t) => sum + getTradePnL(t), 0);
  const grossLoss = Math.abs(losses.reduce((sum, t) => sum + getTradePnL(t), 0));

  // Profit Factor
  const profitFactor = grossLoss > 0
    ? grossProfit / grossLoss
    : grossProfit > 0 ? 999.99 : 0;

  // Win/Loss averages
  const avgWin = wins.length > 0 ? grossProfit / wins.length : 0;
  const avgLoss = losses.length > 0 ? grossLoss / losses.length : 0;

  // Extremes
  const winPnLs = wins.map(getTradePnL);
  const lossPnLs = losses.map(getTradePnL);
  const largestWin = winPnLs.length > 0 ? Math.max(...winPnLs) : 0;
  const largestLoss = lossPnLs.length > 0 ? Math.min(...lossPnLs) : 0;

  // Hold time
  const avgHoldTime = mean(holdTimes);

  // Consecutive streaks
  const winSequence = trades.map(t => getTradePnL(t) > 0);
  const lossSequence = trades.map(t => getTradePnL(t) < 0);
  const consecutiveWins = maxConsecutive(winSequence);
  const consecutiveLosses = maxConsecutive(lossSequence);

  // Sharpe Ratio (annualized)
  const avgReturn = mean(pnls);
  const stdDev = standardDeviation(pnls);

  // Estimate trades per day based on hold time
  const avgHoldDays = avgHoldTime / MINUTES_PER_DAY;
  const tradesPerYear = avgHoldDays > 0
    ? TRADING_DAYS_PER_YEAR / avgHoldDays
    : TRADING_DAYS_PER_YEAR;

  // Annualized Sharpe
  const sharpeRatio = stdDev > 0
    ? ((avgReturn - (RISK_FREE_RATE_ANNUAL / tradesPerYear)) / stdDev) * Math.sqrt(tradesPerYear)
    : 0;

  // Max Drawdown from equity curve
  let maxDrawdown = 0;
  if (equityCurve.length > 0) {
    let peak = equityCurve[0].equity;
    for (const point of equityCurve) {
      if (point.equity > peak) {
        peak = point.equity;
      }
      const drawdown = peak > 0 ? (peak - point.equity) / peak : 0;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
  }

  return {
    sharpe_ratio: Number(sharpeRatio.toFixed(3)),
    max_drawdown: Number((-maxDrawdown * 100).toFixed(2)), // Negative percentage
    win_rate: Number(((wins.length / trades.length) * 100).toFixed(2)),
    avg_hold_time_minutes: Number(avgHoldTime.toFixed(1)),
    total_trades: trades.length,
    winning_trades: wins.length,
    losing_trades: losses.length,
    total_pnl: Number(totalPnL.toFixed(2)),
    profit_factor: Number(profitFactor.toFixed(2)),
    avg_win: Number(avgWin.toFixed(2)),
    avg_loss: Number(avgLoss.toFixed(2)),
    largest_win: Number(largestWin.toFixed(2)),
    largest_loss: Number(largestLoss.toFixed(2)),
    consecutive_wins: consecutiveWins,
    consecutive_losses: consecutiveLosses,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// INCREMENTAL METRICS CALCULATOR
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Incremental metrics calculator for real-time replay updates.
 * Caches calculations and only recalculates when data changes.
 */
export class IncrementalMetricsCalculator {
  private trades: ReplayTrade[] = [];
  private equityPoints: EquityPoint[] = [];
  private cachedMetrics: ReplayMetrics | null = null;
  private lastTradeIndex: number = 0;

  /**
   * Reset the calculator
   */
  reset(): void {
    this.trades = [];
    this.equityPoints = [];
    this.cachedMetrics = null;
    this.lastTradeIndex = 0;
  }

  /**
   * Set all trades at once (for initial load)
   */
  setTrades(trades: ReplayTrade[]): void {
    this.trades = trades;
    this.cachedMetrics = null;
  }

  /**
   * Set all equity points at once (for initial load)
   */
  setEquityPoints(points: EquityPoint[]): void {
    this.equityPoints = points;
    this.cachedMetrics = null;
  }

  /**
   * Add a single trade (for incremental updates)
   */
  addTrade(trade: ReplayTrade): void {
    this.trades.push(trade);
    this.cachedMetrics = null;
  }

  /**
   * Add a single equity point (for incremental updates)
   */
  addEquityPoint(point: EquityPoint): void {
    this.equityPoints.push(point);
    this.cachedMetrics = null;
  }

  /**
   * Get metrics for trades up to a specific index
   */
  getMetricsUpTo(tradeIndex: number): ReplayMetrics {
    if (tradeIndex === this.lastTradeIndex && this.cachedMetrics) {
      return this.cachedMetrics;
    }

    const tradesUpTo = this.trades.slice(0, tradeIndex + 1);

    // Find equity points up to the last trade's timestamp
    let equityUpTo = this.equityPoints;
    if (tradesUpTo.length > 0) {
      const lastTradeTime = tradesUpTo[tradesUpTo.length - 1].timestamp;
      equityUpTo = this.equityPoints.filter(p => p.timestamp <= lastTradeTime);
    }

    this.cachedMetrics = calculateReplayMetrics(tradesUpTo, equityUpTo);
    this.lastTradeIndex = tradeIndex;

    return this.cachedMetrics;
  }

  /**
   * Get current metrics (all trades)
   */
  getMetrics(): ReplayMetrics {
    if (this.cachedMetrics && this.lastTradeIndex === this.trades.length - 1) {
      return this.cachedMetrics;
    }

    this.cachedMetrics = calculateReplayMetrics(this.trades, this.equityPoints);
    this.lastTradeIndex = this.trades.length - 1;

    return this.cachedMetrics;
  }

  /**
   * Get the number of trades
   */
  getTradeCount(): number {
    return this.trades.length;
  }

  /**
   * Get trades filtered by date
   */
  getTradesUpToDate(date: Date): ReplayTrade[] {
    const dateStr = date.toISOString();
    return this.trades.filter(t => {
      const tradeTime = t.timestamp || t.entry_time;
      return tradeTime && tradeTime <= dateStr;
    });
  }

  /**
   * Get equity points filtered by date
   */
  getEquityUpToDate(date: Date): EquityPoint[] {
    const dateStr = date.toISOString();
    return this.equityPoints.filter(p => p.timestamp <= dateStr);
  }

  /**
   * Get metrics for a specific date (filters both trades and equity)
   */
  getMetricsForDate(date: Date): ReplayMetrics {
    const trades = this.getTradesUpToDate(date);
    const equity = this.getEquityUpToDate(date);
    return calculateReplayMetrics(trades, equity);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// QUICK STATS (for performance-critical updates)
// ═══════════════════════════════════════════════════════════════════════════

export interface QuickStats {
  totalTrades: number;
  winRate: number;
  totalPnL: number;
  lastTradePnL: number;
  currentDrawdown: number;
}

/**
 * Calculate quick stats without full metrics recalculation
 */
export function calculateQuickStats(
  trades: ReplayTrade[],
  currentEquity: number,
  peakEquity: number
): QuickStats {
  if (trades.length === 0) {
    return {
      totalTrades: 0,
      winRate: 0,
      totalPnL: 0,
      lastTradePnL: 0,
      currentDrawdown: 0,
    };
  }

  const wins = trades.filter(t => getTradePnL(t) > 0);
  const totalPnL = trades.reduce((sum, t) => sum + getTradePnL(t), 0);
  const lastTrade = trades[trades.length - 1];
  const currentDrawdown = peakEquity > 0
    ? ((peakEquity - currentEquity) / peakEquity) * 100
    : 0;

  return {
    totalTrades: trades.length,
    winRate: (wins.length / trades.length) * 100,
    totalPnL,
    lastTradePnL: getTradePnL(lastTrade),
    currentDrawdown,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// METRICS FORMATTING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format metrics for display
 */
export function formatMetric(value: number, type: keyof ReplayMetrics): string {
  switch (type) {
    case 'sharpe_ratio':
      return value.toFixed(2);
    case 'max_drawdown':
    case 'win_rate':
      return `${value.toFixed(1)}%`;
    case 'total_pnl':
    case 'avg_win':
    case 'avg_loss':
    case 'largest_win':
    case 'largest_loss':
      return `$${value.toFixed(2)}`;
    case 'profit_factor':
      return value >= 999 ? '∞' : value.toFixed(2);
    case 'avg_hold_time_minutes':
      return value >= 60
        ? `${(value / 60).toFixed(1)}h`
        : `${value.toFixed(0)}m`;
    default:
      return String(value);
  }
}

/**
 * Get metric color based on value
 */
export function getMetricColor(value: number, type: keyof ReplayMetrics): string {
  switch (type) {
    case 'sharpe_ratio':
      if (value >= 2) return 'text-emerald-400';
      if (value >= 1) return 'text-green-400';
      if (value >= 0) return 'text-yellow-400';
      return 'text-red-400';

    case 'max_drawdown':
      if (value >= -5) return 'text-emerald-400';
      if (value >= -10) return 'text-green-400';
      if (value >= -20) return 'text-yellow-400';
      return 'text-red-400';

    case 'win_rate':
      if (value >= 60) return 'text-emerald-400';
      if (value >= 50) return 'text-green-400';
      if (value >= 40) return 'text-yellow-400';
      return 'text-red-400';

    case 'profit_factor':
      if (value >= 2) return 'text-emerald-400';
      if (value >= 1.5) return 'text-green-400';
      if (value >= 1) return 'text-yellow-400';
      return 'text-red-400';

    case 'total_pnl':
    case 'avg_win':
    case 'largest_win':
      return value > 0 ? 'text-emerald-400' : value < 0 ? 'text-red-400' : 'text-slate-400';

    case 'avg_loss':
    case 'largest_loss':
      return 'text-red-400';

    default:
      return 'text-slate-300';
  }
}
