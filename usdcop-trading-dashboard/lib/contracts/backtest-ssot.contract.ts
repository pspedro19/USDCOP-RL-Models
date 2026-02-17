/**
 * Backtest Configuration - Single Source of Truth (SSOT)
 * ======================================================
 *
 * This file defines THE canonical backtest configuration used across:
 * - L4 Backtest Validation (Python backend)
 * - Dashboard Backtest Replay (TypeScript frontend)
 * - API Backtest Endpoints
 *
 * CRITICAL: These values MUST match src/config/backtest_ssot.py
 *           Any changes require updating BOTH files.
 *
 * @version 1.0.0
 */

// === TRANSACTION COSTS ===
// Based on MEXC forex pricing for USDCOP
export const SPREAD_BPS = 2.5;           // Bid-ask spread in basis points
export const SLIPPAGE_BPS = 1.0;         // Execution slippage
export const TOTAL_COST_BPS = (SPREAD_BPS + SLIPPAGE_BPS) * 2; // Round-trip

// === ENTRY/EXIT THRESHOLDS ===
// Must match training environment for consistency
export const THRESHOLD_LONG = 0.50;      // Model confidence for LONG entry
export const THRESHOLD_SHORT = -0.50;    // Model confidence for SHORT entry
export const EXIT_THRESHOLD = 0.0;       // Neutral zone for exit

// === CAPITAL & POSITION ===
export const INITIAL_CAPITAL = 10_000;   // Starting capital in USD
export const POSITION_SIZE_PCT = 1.0;    // 100% of capital per trade
export const MAX_POSITION_BARS = 576;    // Max bars to hold (2 days @ 5min)

// === RISK MANAGEMENT ===
export const STOP_LOSS_PCT = 0.025;      // 2.5% stop loss
export const TAKE_PROFIT_PCT = 0.030;    // 3.0% take profit
export const TRAILING_STOP_ENABLED = true;
export const TRAILING_STOP_ACTIVATION_PCT = 0.015;
export const TRAILING_STOP_TRAIL_FACTOR = 0.5;

// === MARKET ASSUMPTIONS ===
export const BARS_PER_TRADING_DAY = 144;  // 12 hours * 12 bars/hour (5min)
export const TRADING_DAYS_PER_YEAR = 252; // Standard forex trading days

// === REPLAY SPEEDS ===
export const REPLAY_SPEEDS = [0.5, 1, 2, 4, 8, 16] as const;
export type ReplaySpeed = typeof REPLAY_SPEEDS[number];

// === UNIFIED CONFIG OBJECT ===
export interface BacktestConfigSSOT {
  // Transaction costs
  spread_bps: number;
  slippage_bps: number;
  total_cost_bps: number;

  // Thresholds
  threshold_long: number;
  threshold_short: number;
  exit_threshold: number;

  // Capital
  initial_capital: number;
  position_size_pct: number;
  max_position_bars: number;

  // Risk management
  stop_loss_pct: number;
  take_profit_pct: number;
  trailing_stop_enabled: boolean;
  trailing_stop_activation_pct: number;
  trailing_stop_trail_factor: number;

  // Market assumptions
  bars_per_trading_day: number;
  trading_days_per_year: number;
}

/**
 * The canonical backtest configuration.
 * Use this throughout the dashboard codebase.
 */
export const BACKTEST_SSOT: BacktestConfigSSOT = {
  spread_bps: SPREAD_BPS,
  slippage_bps: SLIPPAGE_BPS,
  total_cost_bps: TOTAL_COST_BPS,
  threshold_long: THRESHOLD_LONG,
  threshold_short: THRESHOLD_SHORT,
  exit_threshold: EXIT_THRESHOLD,
  initial_capital: INITIAL_CAPITAL,
  position_size_pct: POSITION_SIZE_PCT,
  max_position_bars: MAX_POSITION_BARS,
  stop_loss_pct: STOP_LOSS_PCT,
  take_profit_pct: TAKE_PROFIT_PCT,
  trailing_stop_enabled: TRAILING_STOP_ENABLED,
  trailing_stop_activation_pct: TRAILING_STOP_ACTIVATION_PCT,
  trailing_stop_trail_factor: TRAILING_STOP_TRAIL_FACTOR,
  bars_per_trading_day: BARS_PER_TRADING_DAY,
  trading_days_per_year: TRADING_DAYS_PER_YEAR,
};

// === L4 VALIDATION THRESHOLDS ===
export interface L4ValidationThresholds {
  min_sharpe_ratio: number;
  max_drawdown_pct: number;
  min_win_rate: number;
  min_trades: number;
  min_profit_factor: number;
}

export const L4_VALIDATION_THRESHOLDS: L4ValidationThresholds = {
  min_sharpe_ratio: 0.5,
  max_drawdown_pct: 0.15,      // 15% max drawdown
  min_win_rate: 0.45,          // 45% minimum win rate
  min_trades: 30,              // At least 30 trades
  min_profit_factor: 1.2,
};

// === METRIC CALCULATION HELPERS ===

/**
 * Calculate Sharpe ratio using L4-consistent methodology.
 * Uses daily return aggregation with sqrt(252) annualization.
 *
 * @param returns Array of per-bar returns
 * @param barsPerDay Number of bars per trading day (default 144)
 */
export function calculateSharpeRatio(
  returns: number[],
  barsPerDay: number = BARS_PER_TRADING_DAY
): number {
  if (returns.length === 0) return 0;

  // Aggregate into daily returns
  const dailyReturns: number[] = [];
  for (let i = 0; i < returns.length; i += barsPerDay) {
    const dayReturns = returns.slice(i, i + barsPerDay);
    if (dayReturns.length > 0) {
      const daySum = dayReturns.reduce((a, b) => a + b, 0);
      dailyReturns.push(daySum);
    }
  }

  if (dailyReturns.length < 2) return 0;

  const mean = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
  const variance = dailyReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / dailyReturns.length;
  const std = Math.sqrt(variance);

  if (std < 1e-10) return 0;

  // Annualize: mean/std * sqrt(252)
  return (mean / std) * Math.sqrt(TRADING_DAYS_PER_YEAR);
}

/**
 * Calculate max drawdown from equity curve.
 *
 * @param equityCurve Array of equity values
 * @returns Max drawdown as decimal (e.g., 0.10 = 10%)
 */
export function calculateMaxDrawdown(equityCurve: number[]): number {
  if (equityCurve.length === 0) return 0;

  let peak = equityCurve[0];
  let maxDrawdown = 0;

  for (const equity of equityCurve) {
    if (equity > peak) {
      peak = equity;
    }
    const drawdown = (peak - equity) / peak;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  return maxDrawdown;
}

/**
 * Calculate win rate from trades.
 *
 * @param trades Array of trades with pnl property
 * @returns Win rate as decimal (e.g., 0.55 = 55%)
 */
export function calculateWinRate(trades: Array<{ pnl: number }>): number {
  if (trades.length === 0) return 0;
  const winners = trades.filter(t => t.pnl > 0).length;
  return winners / trades.length;
}

/**
 * Calculate profit factor from trades.
 *
 * @param trades Array of trades with pnl property
 * @returns Profit factor (gross profit / gross loss)
 */
export function calculateProfitFactor(trades: Array<{ pnl: number }>): number {
  const grossProfit = trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0);
  const grossLoss = Math.abs(trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));

  if (grossLoss === 0) return grossProfit > 0 ? 999 : 0;
  return grossProfit / grossLoss;
}

/**
 * Validate that metrics pass L4 thresholds.
 *
 * @param metrics Object with sharpe, maxDrawdown, winRate, trades, profitFactor
 * @returns Array of validation results with pass/fail status
 */
export function validateL4Criteria(metrics: {
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitFactor: number;
}): Array<{ criterion: string; passed: boolean; value: number; threshold: number }> {
  return [
    {
      criterion: `Sharpe > ${L4_VALIDATION_THRESHOLDS.min_sharpe_ratio}`,
      passed: metrics.sharpeRatio >= L4_VALIDATION_THRESHOLDS.min_sharpe_ratio,
      value: metrics.sharpeRatio,
      threshold: L4_VALIDATION_THRESHOLDS.min_sharpe_ratio,
    },
    {
      criterion: `Max DD < ${L4_VALIDATION_THRESHOLDS.max_drawdown_pct * 100}%`,
      passed: metrics.maxDrawdown <= L4_VALIDATION_THRESHOLDS.max_drawdown_pct,
      value: metrics.maxDrawdown,
      threshold: L4_VALIDATION_THRESHOLDS.max_drawdown_pct,
    },
    {
      criterion: `Win Rate > ${L4_VALIDATION_THRESHOLDS.min_win_rate * 100}%`,
      passed: metrics.winRate >= L4_VALIDATION_THRESHOLDS.min_win_rate,
      value: metrics.winRate,
      threshold: L4_VALIDATION_THRESHOLDS.min_win_rate,
    },
    {
      criterion: `Min ${L4_VALIDATION_THRESHOLDS.min_trades} Trades`,
      passed: metrics.totalTrades >= L4_VALIDATION_THRESHOLDS.min_trades,
      value: metrics.totalTrades,
      threshold: L4_VALIDATION_THRESHOLDS.min_trades,
    },
    {
      criterion: `Profit Factor > ${L4_VALIDATION_THRESHOLDS.min_profit_factor}`,
      passed: metrics.profitFactor >= L4_VALIDATION_THRESHOLDS.min_profit_factor,
      value: metrics.profitFactor,
      threshold: L4_VALIDATION_THRESHOLDS.min_profit_factor,
    },
  ];
}
