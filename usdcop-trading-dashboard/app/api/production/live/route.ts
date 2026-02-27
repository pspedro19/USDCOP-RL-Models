/**
 * GET /api/production/live
 * ========================
 * Single endpoint returning complete live state from H5 DB tables.
 * Falls back gracefully when DB is unavailable or queries fail partially.
 *
 * Tables queried (from migrations 043 + 044):
 *   - forecast_h5_signals
 *   - forecast_h5_executions
 *   - forecast_h5_subtrades
 *   - forecast_h5_paper_trading
 *   - usdcop_m5_ohlcv
 */

import { NextResponse } from 'next/server';
import { query } from '@/lib/db/postgres-client';
import type {
  LiveProductionResponse,
  CurrentSignal,
  ActivePosition,
  LiveTrade,
  LiveEquityCurve,
  LiveStats,
  Guardrails,
  MarketState,
  EquityCurvePoint,
} from '@/lib/contracts/production-monitor.contract';

const STRATEGY_ID = 'smart_simple_v11';
const STRATEGY_NAME = 'Smart Simple v1.1';
const INITIAL_CAPITAL = 10000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isMarketOpenCOT(): boolean {
  const now = new Date();
  const utcHour = now.getUTCHours();
  const utcDay = now.getUTCDay(); // 0=Sun, 6=Sat
  // COT = UTC-5. Market 8:00-13:00 COT = 13:00-18:00 UTC, Mon-Fri
  return utcDay >= 1 && utcDay <= 5 && utcHour >= 13 && utcHour < 18;
}

function safeNumber(val: unknown): number | null {
  if (val == null) return null;
  const n = Number(val);
  if (!isFinite(n)) return null;
  return n;
}

// ---------------------------------------------------------------------------
// Q1: Current signal
// ---------------------------------------------------------------------------
async function fetchCurrentSignal(): Promise<CurrentSignal | null> {
  const res = await query(`
    SELECT signal_date, direction, ensemble_return,
           confidence_tier, adjusted_leverage,
           hard_stop_pct, take_profit_pct, skip_trade
    FROM forecast_h5_signals
    WHERE signal_date = (SELECT MAX(signal_date) FROM forecast_h5_signals)
    LIMIT 1
  `);
  if (res.rows.length === 0) return null;
  const r = res.rows[0];
  // DB stores HS/TP as fractions (0.03 = 3%), convert to percentages for frontend
  const hsPct = safeNumber(r.hard_stop_pct);
  const tpPct = safeNumber(r.take_profit_pct);
  return {
    signal_date: r.signal_date,
    direction: Number(r.direction),
    confidence_tier: r.confidence_tier ?? null,
    adjusted_leverage: safeNumber(r.adjusted_leverage),
    hard_stop_pct: hsPct != null ? hsPct * 100 : null,
    take_profit_pct: tpPct != null ? tpPct * 100 : null,
    ensemble_return: Number(r.ensemble_return),
    skip_trade: Boolean(r.skip_trade),
  };
}

// ---------------------------------------------------------------------------
// Q2: Active position (execution with status='positioned')
// ---------------------------------------------------------------------------
async function fetchActivePosition(currentPrice: number | null, currentBarTime: string | null): Promise<ActivePosition | null> {
  const res = await query(`
    SELECT e.id, e.entry_price, e.entry_timestamp, e.direction, e.leverage,
           e.hard_stop_pct, e.take_profit_pct, e.status,
           s.peak_price, s.bar_count
    FROM forecast_h5_executions e
    LEFT JOIN forecast_h5_subtrades s
      ON s.execution_id = e.id AND s.exit_timestamp IS NULL
    WHERE e.status = 'positioned'
    ORDER BY e.signal_date DESC
    LIMIT 1
  `);
  if (res.rows.length === 0) return null;
  const r = res.rows[0];

  const entryPrice = Number(r.entry_price);
  const direction = Number(r.direction);
  const leverage = Number(r.leverage);
  const price = currentPrice ?? entryPrice;

  // For SHORT: profit when price drops, loss when price rises
  // unrealized_pnl = direction * (entry - current) / entry * leverage
  const unrealizedPnl = direction === -1
    ? (entryPrice - price) / entryPrice * leverage * 100
    : (price - entryPrice) / entryPrice * leverage * 100;

  // DB stores HS/TP as fractions (0.03 = 3%), convert to percentages
  const tpPct = safeNumber(r.take_profit_pct);
  const hsPct = safeNumber(r.hard_stop_pct);
  const tpPctScaled = tpPct != null ? tpPct * 100 : null;
  const hsPctScaled = hsPct != null ? hsPct * 100 : null;

  // Distance to TP/HS (how much more move needed to hit)
  let distanceToTp: number | null = null;
  let distanceToHs: number | null = null;

  if (tpPctScaled != null) {
    // TP progress: how far towards TP (0% = at entry, 100% = at TP)
    // For SHORT: tp hit when price falls tp_pct below entry
    const priceMovePct = direction === -1
      ? (entryPrice - price) / entryPrice * 100
      : (price - entryPrice) / entryPrice * 100;
    distanceToTp = Math.max(0, tpPctScaled - priceMovePct);
  }
  if (hsPctScaled != null) {
    // HS distance: how far towards HS
    const adverseMovePct = direction === -1
      ? (price - entryPrice) / entryPrice * 100
      : (entryPrice - price) / entryPrice * 100;
    distanceToHs = Math.max(0, hsPctScaled - adverseMovePct);
  }

  return {
    execution_id: Number(r.id),
    entry_price: entryPrice,
    entry_timestamp: r.entry_timestamp,
    direction,
    leverage,
    current_price: price,
    current_bar_time: currentBarTime ?? new Date().toISOString(),
    unrealized_pnl_pct: Math.round(unrealizedPnl * 100) / 100,
    distance_to_tp_pct: distanceToTp != null ? Math.round(distanceToTp * 100) / 100 : null,
    distance_to_hs_pct: distanceToHs != null ? Math.round(distanceToHs * 100) / 100 : null,
    bar_count: Number(r.bar_count ?? 0),
    peak_price: safeNumber(r.peak_price),
    status: r.status,
  };
}

// ---------------------------------------------------------------------------
// Q3: Latest price from OHLCV
// ---------------------------------------------------------------------------
async function fetchLatestPrice(): Promise<{ price: number; time: string } | null> {
  const res = await query(`
    SELECT time, close
    FROM usdcop_m5_ohlcv
    WHERE symbol = 'USD/COP'
    ORDER BY time DESC
    LIMIT 1
  `);
  if (res.rows.length === 0) return null;
  return {
    price: Number(res.rows[0].close),
    time: res.rows[0].time,
  };
}

// ---------------------------------------------------------------------------
// Q4: 2026 closed/positioned executions -> trades + equity curve + stats
// ---------------------------------------------------------------------------
async function fetchExecutions(): Promise<{
  trades: LiveTrade[];
  equityCurve: LiveEquityCurve;
  stats: LiveStats;
}> {
  const res = await query(`
    SELECT id, signal_date, direction, leverage,
           entry_price, entry_timestamp, exit_price, exit_timestamp,
           exit_reason, week_pnl_pct, status,
           confidence_tier, hard_stop_pct, take_profit_pct
    FROM forecast_h5_executions
    WHERE inference_year = 2026
      AND status IN ('closed', 'positioned')
    ORDER BY signal_date ASC
  `);

  const trades: LiveTrade[] = [];
  const points: EquityCurvePoint[] = [];
  let equity = INITIAL_CAPITAL;
  let peakEquity = INITIAL_CAPITAL;
  let maxDd = 0;
  let wins = 0;
  let losses = 0;
  let nLong = 0;
  let nShort = 0;
  let grossProfit = 0;
  let grossLoss = 0;
  const exitReasons: Record<string, number> = {};
  const weeklyReturns: number[] = [];

  // Starting point for equity curve
  points.push({ date: '2026-01-01', equity: INITIAL_CAPITAL, pnl_pct: 0 });

  for (let i = 0; i < res.rows.length; i++) {
    const r = res.rows[i];
    // DB stores week_pnl_pct as fraction (0.029772 = 2.9772%), convert to %
    const rawPnl = safeNumber(r.week_pnl_pct) ?? 0;
    const pnlPct = rawPnl * 100;
    const direction = Number(r.direction);
    const lev = Number(r.leverage);
    const entryPrice = Number(r.entry_price ?? 0);
    const exitPrice = Number(r.exit_price ?? entryPrice);

    const pnlUsd = equity * (pnlPct / 100);
    const equityAtEntry = equity;
    equity += pnlUsd;
    const equityAtExit = equity;

    if (equity > peakEquity) peakEquity = equity;
    const dd = peakEquity > 0 ? ((peakEquity - equity) / peakEquity) * 100 : 0;
    if (dd > maxDd) maxDd = dd;

    if (pnlPct > 0) { wins++; grossProfit += pnlPct; }
    else if (pnlPct < 0) { losses++; grossLoss += Math.abs(pnlPct); }
    if (direction === 1) nLong++;
    if (direction === -1) nShort++;

    const reason = r.exit_reason ?? (r.status === 'positioned' ? 'open' : 'unknown');
    exitReasons[reason] = (exitReasons[reason] ?? 0) + 1;
    weeklyReturns.push(pnlPct);

    // Only add closed trades to the trade list
    if (r.status === 'closed') {
      // DB stores HS/TP as fractions, convert to %
      const tradeHs = safeNumber(r.hard_stop_pct);
      const tradeTp = safeNumber(r.take_profit_pct);
      trades.push({
        trade_id: i + 1,
        timestamp: r.entry_timestamp ?? r.signal_date,
        exit_timestamp: r.exit_timestamp ?? null,
        side: direction === -1 ? 'SHORT' : 'LONG',
        entry_price: entryPrice,
        exit_price: exitPrice,
        pnl_usd: Math.round(pnlUsd * 100) / 100,
        pnl_pct: Math.round(pnlPct * 100) / 100,
        exit_reason: r.exit_reason ?? 'unknown',
        equity_at_entry: Math.round(equityAtEntry * 100) / 100,
        equity_at_exit: Math.round(equityAtExit * 100) / 100,
        leverage: lev,
        confidence_tier: r.confidence_tier ?? null,
        hard_stop_pct: tradeHs != null ? tradeHs * 100 : null,
        take_profit_pct: tradeTp != null ? tradeTp * 100 : null,
      });
    }

    points.push({
      date: r.signal_date,
      equity: Math.round(equityAtExit * 100) / 100,
      pnl_pct: Math.round(pnlPct * 100) / 100,
    });
  }

  // Compute Sharpe (annualized from weekly returns)
  let sharpe: number | null = null;
  if (weeklyReturns.length >= 2) {
    const mean = weeklyReturns.reduce((s, v) => s + v, 0) / weeklyReturns.length;
    const variance = weeklyReturns.reduce((s, v) => s + (v - mean) ** 2, 0) / (weeklyReturns.length - 1);
    const std = Math.sqrt(variance);
    if (std > 0) {
      sharpe = Math.round((mean / std) * Math.sqrt(52) * 100) / 100;
    }
  }

  const nTrades = wins + losses;
  const totalReturnPct = INITIAL_CAPITAL > 0 ? ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100 : 0;
  const pf = grossLoss > 0 ? Math.round((grossProfit / grossLoss) * 100) / 100 : null;

  return {
    trades,
    equityCurve: {
      initial_capital: INITIAL_CAPITAL,
      points,
      current_equity: Math.round(equity * 100) / 100,
    },
    stats: {
      total_return_pct: Math.round(totalReturnPct * 100) / 100,
      sharpe,
      max_dd_pct: Math.round(maxDd * 100) / 100,
      win_rate_pct: nTrades > 0 ? Math.round((wins / nTrades) * 10000) / 100 : null,
      profit_factor: pf,
      n_trades: nTrades,
      n_long: nLong,
      n_short: nShort,
      exit_reasons: exitReasons,
    },
  };
}

// ---------------------------------------------------------------------------
// Q5: Guardrails from paper_trading
// ---------------------------------------------------------------------------
async function fetchGuardrails(): Promise<Guardrails | null> {
  const res = await query(`
    SELECT cumulative_pnl_pct, consecutive_losses,
           running_sharpe, running_da_short_pct,
           circuit_breaker, notes
    FROM forecast_h5_paper_trading
    ORDER BY signal_date DESC
    LIMIT 1
  `);
  if (res.rows.length === 0) return null;
  const r = res.rows[0];

  const alerts: string[] = [];
  if (r.circuit_breaker) alerts.push('Circuit breaker activo');
  if (Number(r.consecutive_losses ?? 0) >= 3) alerts.push(`${r.consecutive_losses} perdidas consecutivas`);
  if (r.notes) alerts.push(r.notes);

  return {
    cumulative_pnl_pct: safeNumber(r.cumulative_pnl_pct),
    consecutive_losses: Number(r.consecutive_losses ?? 0),
    rolling_sharpe_16w: safeNumber(r.running_sharpe),
    rolling_da_short_16w: safeNumber(r.running_da_short_pct),
    circuit_breaker_active: Boolean(r.circuit_breaker),
    alerts,
  };
}

// ---------------------------------------------------------------------------
// GET handler
// ---------------------------------------------------------------------------
export async function GET() {
  const partialErrors: string[] = [];

  // Try all DB queries. Each one fails gracefully.
  let currentSignal: CurrentSignal | null = null;
  let activePosition: ActivePosition | null = null;
  let latestPrice: { price: number; time: string } | null = null;
  let execData: { trades: LiveTrade[]; equityCurve: LiveEquityCurve; stats: LiveStats } | null = null;
  let guardrails: Guardrails | null = null;
  let dbAvailable = true;

  // Q3 first — we need the price for position calculations
  try {
    latestPrice = await fetchLatestPrice();
  } catch (e) {
    dbAvailable = false;
    partialErrors.push(`price: ${e instanceof Error ? e.message : String(e)}`);
  }

  // If first query failed, DB is likely down. Return unavailable.
  if (!dbAvailable) {
    const resp: LiveProductionResponse = {
      strategy_id: STRATEGY_ID,
      strategy_name: STRATEGY_NAME,
      current_signal: null,
      active_position: null,
      trades: [],
      equity_curve: { initial_capital: INITIAL_CAPITAL, points: [], current_equity: INITIAL_CAPITAL },
      stats: {
        total_return_pct: 0, sharpe: null, max_dd_pct: null,
        win_rate_pct: null, profit_factor: null,
        n_trades: 0, n_long: 0, n_short: 0, exit_reasons: {},
      },
      guardrails: null,
      market: { is_open: isMarketOpenCOT(), current_price: null, last_bar_time: null },
      generated_at: new Date().toISOString(),
      data_source: 'unavailable',
      partial_errors: partialErrors,
    };
    return NextResponse.json(resp);
  }

  // Run remaining queries in parallel
  const [signalResult, execResult, guardrailResult] = await Promise.allSettled([
    fetchCurrentSignal(),
    fetchExecutions(),
    fetchGuardrails(),
  ]);

  if (signalResult.status === 'fulfilled') {
    currentSignal = signalResult.value;
  } else {
    partialErrors.push(`signal: ${signalResult.reason}`);
  }

  if (execResult.status === 'fulfilled') {
    execData = execResult.value;
  } else {
    partialErrors.push(`executions: ${execResult.reason}`);
  }

  if (guardrailResult.status === 'fulfilled') {
    guardrails = guardrailResult.value;
  } else {
    partialErrors.push(`guardrails: ${guardrailResult.reason}`);
  }

  // Q2: Active position (needs current price from Q3)
  try {
    activePosition = await fetchActivePosition(
      latestPrice?.price ?? null,
      latestPrice?.time ?? null,
    );
  } catch (e) {
    partialErrors.push(`position: ${e instanceof Error ? e.message : String(e)}`);
  }

  const market: MarketState = {
    is_open: isMarketOpenCOT(),
    current_price: latestPrice?.price ?? null,
    last_bar_time: latestPrice?.time ?? null,
  };

  const resp: LiveProductionResponse = {
    strategy_id: STRATEGY_ID,
    strategy_name: STRATEGY_NAME,
    current_signal: currentSignal,
    active_position: activePosition,
    trades: execData?.trades ?? [],
    equity_curve: execData?.equityCurve ?? {
      initial_capital: INITIAL_CAPITAL,
      points: [],
      current_equity: INITIAL_CAPITAL,
    },
    stats: execData?.stats ?? {
      total_return_pct: 0, sharpe: null, max_dd_pct: null,
      win_rate_pct: null, profit_factor: null,
      n_trades: 0, n_long: 0, n_short: 0, exit_reasons: {},
    },
    guardrails,
    market,
    generated_at: new Date().toISOString(),
    data_source: 'db',
    partial_errors: partialErrors.length > 0 ? partialErrors : undefined,
  };

  return NextResponse.json(resp);
}
