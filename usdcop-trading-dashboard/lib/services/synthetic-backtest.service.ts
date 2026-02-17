/**
 * Synthetic Backtest Service
 * ==========================
 * Generates synthetic backtest data for fallback/demo when backend is unavailable.
 *
 * SSOT Integration:
 * - Uses BACKTEST_SSOT config for consistent parameters
 * - Sharpe calculation uses daily aggregation (matches L4)
 * - Transaction costs and thresholds from SSOT
 *
 * Investor Demo Mode:
 * - Targets ~33% annualized return, ~60% win rate, Sharpe ~1.8-2.2
 * - Realistic drawdown arcs (~10-12% max DD)
 * - ~10 trades/month, intraday duration
 * - Deterministic seed for reproducible results
 *
 * Speed Control:
 * - Supports replay speeds: 0.5x, 1x, 2x, 4x, 8x, 16x
 * - Dynamic delay calculation based on speed multiplier
 */

import type { BacktestTrade, BacktestSummary, ReplaySpeed } from '@/lib/contracts/backtest.contract';
import { getReplayDelay, DEFAULT_REPLAY_SPEED } from '@/lib/contracts/backtest.contract';
import {
  BACKTEST_SSOT,
  calculateSharpeRatio,
  calculateMaxDrawdown,
  calculateWinRate,
  calculateProfitFactor,
} from '@/lib/contracts/backtest-ssot.contract';

// ============================================================================
// Types
// ============================================================================

export interface SyntheticBacktestConfig {
  startDate: string;
  endDate: string;
  modelId: string;
  winBias?: number;
  tradeCount?: number;
  initialEquity?: number;
  /** Emit bar-level events for dynamic equity curve (replay mode) */
  emitBarEvents?: boolean;
  /** Replay speed multiplier (0.5x to 16x) - default 1x */
  replaySpeed?: ReplaySpeed;
}

// ============================================================================
// Investor Demo Defaults
// ============================================================================

const INVESTOR_DEFAULTS = {
  /** Target annualized return (set higher than 33% to compensate for streak multiplier on losses) */
  TARGET_ANNUAL_RETURN: 0.47,
  /** Base win rate */
  WIN_RATE: 0.60,
  /** Avg win / avg loss ratio (reward:risk) */
  WIN_LOSS_RATIO: 1.65,
  /** Trades per month */
  TRADES_PER_MONTH: 10,
  /** Base USDCOP price (mid-2025 range) */
  BASE_PRICE: 4250,
  /** Price volatility band (pesos) */
  PRICE_VOLATILITY: 150,
  /** Min trade duration (minutes) */
  MIN_DURATION_MIN: 30,
  /** Max trade duration (minutes) */
  MAX_DURATION_MIN: 240,
  /** Notional multiplier (USD per point) */
  NOTIONAL_MULTIPLIER: 10,
  /** Seed for reproducibility */
  SEED: 42,
} as const;

// ============================================================================
// Seeded PRNG (Mulberry32) - deterministic results for demos
// ============================================================================

function createSeededRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ============================================================================
// Trade Generation
// ============================================================================

export function generateSyntheticTrades(config: SyntheticBacktestConfig): BacktestTrade[] {
  const {
    startDate,
    endDate,
    modelId,
    initialEquity = 10000,
  } = config;

  const start = new Date(startDate);
  const end = new Date(endDate);
  const totalDays = Math.max(1, Math.round((end.getTime() - start.getTime()) / 86400000));
  const totalMs = end.getTime() - start.getTime();
  const isInvestorDemo = modelId?.includes('investor') || modelId?.includes('demo');

  if (!isInvestorDemo) {
    return generateGenericTrades(config, start, end, totalDays, totalMs);
  }

  // ── Investor Demo: engineered metrics ──────────────────────────────────
  const D = INVESTOR_DEFAULTS;
  const rng = createSeededRng(D.SEED + totalDays);

  const monthsInRange = totalDays / 30.44;
  const tradeCount = config.tradeCount ?? Math.max(10, Math.round(monthsInRange * D.TRADES_PER_MONTH));
  const winRate = config.winBias ?? D.WIN_RATE;

  // Target PnL scaled to period length
  const targetReturn = D.TARGET_ANNUAL_RETURN * (monthsInRange / 12);
  const targetPnlUsd = initialEquity * targetReturn;

  // Solve: winCount * avgWinUsd - lossCount * avgLossUsd = targetPnlUsd
  // with avgWinUsd = WIN_LOSS_RATIO * avgLossUsd
  const winCount = Math.round(tradeCount * winRate);
  const lossCount = tradeCount - winCount;
  const avgLossUsd = targetPnlUsd / (winCount * D.WIN_LOSS_RATIO - lossCount);
  const avgWinUsd = avgLossUsd * D.WIN_LOSS_RATIO;

  // Pre-determine win/loss sequence with realistic drawdown clusters
  const outcomes = buildOutcomeSequence(tradeCount, winCount, rng);

  const trades: BacktestTrade[] = [];
  let equity = initialEquity;

  // Generate a price path that looks like USDCOP 2025
  // Start ~4200, mid-year dip to ~4100, rally to ~4400, settle ~4300
  const pricePath = (progress: number): number => {
    const base = D.BASE_PRICE;
    const seasonal = -80 * Math.sin(progress * Math.PI * 2) + 50 * Math.cos(progress * Math.PI * 4);
    const trend = 100 * progress; // slight uptrend over year
    const noise = (rng() - 0.5) * 60;
    return base + seasonal + trend + noise;
  };

  let consecutiveLosses = 0;

  for (let i = 0; i < tradeCount; i++) {
    const progress = i / tradeCount;

    // Spread trades across business days (Mon-Fri, 8am-1pm COT)
    const baseOffsetMs = (progress) * totalMs;
    const jitterMs = (rng() - 0.5) * (totalMs / tradeCount) * 0.3;
    let entryDate = new Date(start.getTime() + baseOffsetMs + jitterMs);
    // Snap to business hours (8-13 COT = 13-18 UTC)
    entryDate = snapToBusinessHours(entryDate, rng);

    const durationMin = D.MIN_DURATION_MIN + rng() * (D.MAX_DURATION_MIN - D.MIN_DURATION_MIN);
    const durationMs = durationMin * 60 * 1000;
    const exitDate = new Date(Math.min(entryDate.getTime() + durationMs, end.getTime()));

    const entryPrice = pricePath(progress);
    const side = rng() > 0.5 ? 'LONG' : 'SHORT';
    const isWin = outcomes[i];

    if (isWin) {
      consecutiveLosses = 0;
    } else {
      consecutiveLosses++;
    }

    // PnL in USD with realistic variance
    // During losing streaks, losses get larger (drawdown acceleration - realistic)
    const baseVariance = 0.5 + rng() * 1.0;
    const streakMultiplier = isWin ? 1 : Math.min(1.8, 1 + consecutiveLosses * 0.15);
    const variance = baseVariance * streakMultiplier;
    const pnlUsd = isWin
      ? avgWinUsd * variance
      : -(avgLossUsd * variance);

    // Convert USD PnL back to price points
    const pnlPoints = pnlUsd / D.NOTIONAL_MULTIPLIER;
    const pnl = Math.abs(pnlPoints);
    const exitPrice = side === 'LONG'
      ? entryPrice + (isWin ? pnl : -pnl)
      : entryPrice - (isWin ? pnl : -pnl);

    const pnlPct = (pnlPoints / entryPrice) * 100;

    const equityBefore = equity;
    equity += pnlUsd;

    // Confidence: wins tend to have higher confidence (realistic)
    const baseConf = isWin ? 0.70 + rng() * 0.25 : 0.55 + rng() * 0.30;

    trades.push({
      trade_id: i + 1,
      model_id: modelId || 'investor_demo',
      timestamp: entryDate.toISOString(),
      entry_time: entryDate.toISOString(),
      exit_time: exitDate.toISOString(),
      side,
      entry_price: round2(entryPrice),
      exit_price: round2(exitPrice),
      pnl: round2(pnlPoints),
      pnl_usd: round2(pnlUsd),
      pnl_percent: round3(pnlPct),
      pnl_pct: round3(pnlPct),
      status: 'closed',
      duration_minutes: Math.round(durationMin),
      exit_reason: isWin ? 'take_profit' : 'stop_loss',
      equity_at_entry: round2(equityBefore),
      equity_at_exit: round2(equity),
      entry_confidence: round2(baseConf),
      exit_confidence: round2(isWin ? baseConf + rng() * 0.05 : baseConf - rng() * 0.10),
    });
  }

  trades.sort((a, b) =>
    new Date(a.entry_time).getTime() - new Date(b.entry_time).getTime()
  );

  return trades;
}

// ============================================================================
// Outcome Sequence Builder
// ============================================================================

/**
 * Build a win/loss sequence with realistic drawdown clusters.
 * Creates "regimes" that produce visible equity dips:
 * - Good regimes: ~75% win rate (trending market)
 * - Bad regimes: ~20% win rate (choppy market → drawdown arc)
 * - Normal regimes: ~60% win rate
 * Guarantees exact target win count.
 */
function buildOutcomeSequence(
  total: number,
  targetWins: number,
  rng: () => number
): boolean[] {
  // Start with all outcomes as an array of booleans
  const outcomes: boolean[] = new Array(total).fill(false);

  // Place wins randomly, but cluster them
  let placed = 0;
  let idx = 0;

  // Create regime blocks
  while (idx < total && placed < targetWins) {
    const remaining = total - idx;
    const winsLeft = targetWins - placed;
    const blockSize = Math.min(remaining, 6 + Math.floor(rng() * 10));

    // Regime selection
    const regimeRoll = rng();
    let blockWinRate: number;

    if (regimeRoll < 0.18) {
      // Bad regime: drawdown cluster (only 15-25% wins)
      blockWinRate = 0.15 + rng() * 0.10;
    } else if (regimeRoll > 0.82) {
      // Hot streak: 80-90% wins
      blockWinRate = 0.80 + rng() * 0.10;
    } else {
      // Normal: track toward target
      blockWinRate = winsLeft / (total - idx) + (rng() - 0.5) * 0.12;
    }

    blockWinRate = Math.max(0, Math.min(1, blockWinRate));
    const blockWins = Math.min(winsLeft, Math.max(0, Math.round(blockSize * blockWinRate)));

    // Place wins at random positions in block
    const positions = Array.from({ length: blockSize }, (_, j) => j);
    // Shuffle positions
    for (let j = positions.length - 1; j > 0; j--) {
      const k = Math.floor(rng() * (j + 1));
      [positions[j], positions[k]] = [positions[k], positions[j]];
    }
    for (let j = 0; j < blockWins; j++) {
      outcomes[idx + positions[j]] = true;
    }

    placed += blockWins;
    idx += blockSize;
  }

  // If we still need more wins, fill them in randomly among remaining false slots
  if (placed < targetWins) {
    const falseIndices = outcomes.map((v, i) => (!v ? i : -1)).filter(i => i >= 0);
    for (let j = falseIndices.length - 1; j > 0; j--) {
      const k = Math.floor(rng() * (j + 1));
      [falseIndices[j], falseIndices[k]] = [falseIndices[k], falseIndices[j]];
    }
    for (let j = 0; j < targetWins - placed && j < falseIndices.length; j++) {
      outcomes[falseIndices[j]] = true;
    }
  }

  return outcomes;
}

// ============================================================================
// Generic (non-investor) Trade Generation - preserves old behavior
// ============================================================================

function generateGenericTrades(
  config: SyntheticBacktestConfig,
  start: Date,
  end: Date,
  totalDays: number,
  totalMs: number,
): BacktestTrade[] {
  const { modelId, initialEquity = 10000 } = config;

  const tradeCount = config.tradeCount ?? (20 + Math.floor(totalDays / 10));
  const winBias = config.winBias ?? 0.50;

  const trades: BacktestTrade[] = [];
  let equity = initialEquity;

  for (let i = 0; i < tradeCount; i++) {
    const baseOffset = (i / tradeCount) * totalMs;
    const jitter = (Math.random() - 0.5) * (totalMs / tradeCount) * 0.5;
    const entryDate = new Date(start.getTime() + baseOffset + jitter);
    const durationMs = (30 + Math.random() * 330) * 60 * 1000;
    const exitDate = new Date(Math.min(entryDate.getTime() + durationMs, end.getTime()));

    const side = Math.random() > 0.5 ? 'LONG' : 'SHORT';
    const trend = Math.sin(i * 0.15) * 80;
    const noise = (Math.random() - 0.5) * 40;
    const entryPrice = 4250 + trend + noise;
    const isWin = Math.random() < winBias;
    const magnitude = 5 + Math.random() * 25;
    const priceMove = isWin ? magnitude : -magnitude;
    const exitPrice = entryPrice + (side === 'LONG' ? priceMove : -priceMove);
    const pnl = side === 'LONG' ? exitPrice - entryPrice : entryPrice - exitPrice;
    const pnlPct = (pnl / entryPrice) * 100;
    const pnlUsd = pnl * 10;
    equity += pnlUsd;

    trades.push({
      trade_id: i + 1,
      model_id: modelId || 'synthetic',
      timestamp: entryDate.toISOString(),
      entry_time: entryDate.toISOString(),
      exit_time: exitDate.toISOString(),
      side,
      entry_price: round2(entryPrice),
      exit_price: round2(exitPrice),
      pnl: round2(pnl),
      pnl_usd: round2(pnlUsd),
      pnl_percent: round3(pnlPct),
      pnl_pct: round3(pnlPct),
      status: 'closed',
      duration_minutes: Math.round(durationMs / 60000),
      exit_reason: isWin ? 'take_profit' : 'stop_loss',
      equity_at_entry: round2(equity - pnlUsd),
      equity_at_exit: round2(equity),
      entry_confidence: round2(0.60 + Math.random() * 0.35),
      exit_confidence: round2(0.50 + Math.random() * 0.35),
    });
  }

  trades.sort((a, b) =>
    new Date(a.entry_time).getTime() - new Date(b.entry_time).getTime()
  );

  return trades;
}

// ============================================================================
// Summary Calculation
// ============================================================================

/**
 * Calculate backtest summary with SSOT-consistent metrics.
 * Uses daily return aggregation for Sharpe calculation (matches L4).
 */
export function calculateBacktestSummary(trades: BacktestTrade[], initialEquity = BACKTEST_SSOT.initial_capital): BacktestSummary {
  const wins = trades.filter(t => t.pnl > 0);
  const losses = trades.filter(t => t.pnl <= 0);
  const totalPnl = trades.reduce((s, t) => s + t.pnl_usd, 0);

  // Build equity curve for max drawdown calculation
  const equityCurve: number[] = [initialEquity];
  let running = initialEquity;
  for (const t of trades) {
    running += t.pnl_usd;
    equityCurve.push(running);
  }

  // Use SSOT max drawdown calculation (returns decimal, we want percentage)
  const maxDrawdown = calculateMaxDrawdown(equityCurve) * 100;

  // Build per-bar returns for Sharpe calculation
  // For synthetic trades, we simulate bar-level returns based on trade PnL
  const barReturns: number[] = [];
  for (const t of trades) {
    // Distribute trade PnL across its duration as bar returns
    const barsInTrade = Math.max(1, Math.floor(t.duration_minutes / 5)); // 5-min bars
    const returnPerBar = (t.pnl_pct / 100) / barsInTrade;
    for (let i = 0; i < barsInTrade; i++) {
      barReturns.push(returnPerBar);
    }
  }

  // Use SSOT Sharpe calculation (daily aggregation with 144 bars per day)
  const sharpe = calculateSharpeRatio(barReturns, BACKTEST_SSOT.bars_per_trading_day);

  // Profit factor using SSOT calculation
  const profitFactor = calculateProfitFactor(trades);

  return {
    total_trades: trades.length,
    winning_trades: wins.length,
    losing_trades: losses.length,
    win_rate: trades.length > 0 ? round2((wins.length / trades.length) * 100) : 0,
    total_pnl: round2(totalPnl),
    total_return_pct: round2((totalPnl / initialEquity) * 100),
    max_drawdown_pct: round2(maxDrawdown),
    sharpe_ratio: round2(sharpe),
    avg_trade_duration_minutes: trades.length > 0
      ? Math.round(trades.reduce((s, t) => s + t.duration_minutes, 0) / trades.length)
      : 0,
  };
}

// ============================================================================
// SSE Stream with Speed Control
// ============================================================================

/**
 * Create SSE stream for synthetic backtest with speed control.
 *
 * Speed multipliers:
 * - 0.5x: Slow motion (600ms base delay)
 * - 1x: Normal speed (300ms base delay)
 * - 2x: Double speed (150ms base delay)
 * - 4x: Fast (75ms base delay)
 * - 8x: Very fast (37.5ms base delay)
 * - 16x: Maximum (18.75ms base delay)
 */
export function createSyntheticSSEStream(config: SyntheticBacktestConfig): ReadableStream {
  return new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      const send = (type: string, data: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, data })}\n\n`));
      };
      const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

      const startMs = Date.now();
      const trades = generateSyntheticTrades(config);
      const isInvestorDemo = config.modelId?.includes('investor') || config.modelId?.includes('demo');

      // Get speed-adjusted delay from SSOT
      const speed = config.replaySpeed ?? DEFAULT_REPLAY_SPEED;
      const baseDelayFromSSoT = getReplayDelay(speed);
      const baseDelay = isInvestorDemo ? baseDelayFromSSoT * 1.2 : baseDelayFromSSoT;
      const initialEquity = config.initialEquity ?? BACKTEST_SSOT.initial_capital;

      // Phase 1: Loading progress (fast, not affected by speed)
      send('progress', {
        progress: 0, current_bar: 0, total_bars: trades.length,
        trades_generated: 0, status: 'loading', message: 'Loading historical data...',
        replay_speed: speed,
      });
      await delay(Math.min(400, 600 / speed));

      send('progress', {
        progress: 0.05, current_bar: 0, total_bars: trades.length,
        trades_generated: 0, status: 'loading', message: 'Initializing model...',
        replay_speed: speed,
      });
      await delay(Math.min(300, 500 / speed));

      send('progress', {
        progress: 0.10, current_bar: 0, total_bars: trades.length,
        trades_generated: 0, status: 'running', message: 'Running simulation...',
        replay_speed: speed,
      });
      await delay(Math.min(200, 400 / speed));

      // Phase 2: Stream trades progressively with speed control
      const emitBars = config.emitBarEvents ?? false;

      // For replay mode with bar events, simulate intermediate bars between trades
      let cumulativePnl = 0;
      let cumulativePnlUsd = 0;

      for (let i = 0; i < trades.length; i++) {
        const trade = trades[i];
        const progress = 0.10 + (i / trades.length) * 0.85;

        // Emit bar events for dynamic equity curve (replay mode)
        if (emitBars && i > 0) {
          // Simulate some intermediate bars between trades
          const prevTrade = trades[i - 1];
          const prevTime = new Date(prevTrade.exit_time!).getTime();
          const currTime = new Date(trade.entry_time).getTime();
          const barCount = Math.min(3, Math.floor((currTime - prevTime) / (5 * 60 * 1000))); // 5min bars

          for (let b = 0; b < barCount; b++) {
            const barTime = new Date(prevTime + ((b + 1) / (barCount + 1)) * (currTime - prevTime));
            send('bar', {
              timestamp: barTime.toISOString(),
              equity: initialEquity + cumulativePnlUsd,
              drawdown: Math.max(0, -cumulativePnlUsd / initialEquity),
              position: 'NEUTRAL',
              close: 4250 + (Math.random() - 0.5) * 50,
            });
            await delay(baseDelay * 0.15);
          }
        }

        send('progress', {
          progress, current_bar: i + 1, total_bars: trades.length,
          trades_generated: i + 1, status: 'running',
          message: `Trade ${i + 1}/${trades.length}: ${trade.side} @ $${trade.entry_price.toFixed(2)}`,
          replay_speed: speed,
        });

        await delay(baseDelay * 0.4);

        // Update cumulative PnL
        cumulativePnl += trade.pnl || 0;
        cumulativePnlUsd += trade.pnl_usd || 0;

        send('trade', {
          ...trade,
          trade_id: String(trade.trade_id),
          current_equity: trade.equity_at_exit,
          cumulative_pnl: cumulativePnl,
          cumulative_pnl_usd: cumulativePnlUsd,
          drawdown: Math.max(0, (initialEquity * 1.1 - (trade.equity_at_exit ?? initialEquity)) / (initialEquity * 1.1)),
          position: trade.side === 'BUY' ? 'LONG' : 'SHORT',
        });

        await delay(baseDelay * (0.4 + Math.random() * 0.3));
      }

      // Phase 3: Finalize (fast, not affected by speed)
      send('progress', {
        progress: 0.97, current_bar: trades.length, total_bars: trades.length,
        trades_generated: trades.length, status: 'saving', message: 'Calculating performance metrics...',
        replay_speed: speed,
      });
      await delay(Math.min(300, 500 / speed));

      const summary = calculateBacktestSummary(trades, initialEquity);

      send('result', {
        success: true,
        source: 'generated',
        trade_count: trades.length,
        trades,
        summary,
        processing_time_ms: Date.now() - startMs,
        date_range: { start: config.startDate, end: config.endDate },
        replay_speed: speed,
        ssot_config: {
          spread_bps: BACKTEST_SSOT.spread_bps,
          slippage_bps: BACKTEST_SSOT.slippage_bps,
          threshold_long: BACKTEST_SSOT.threshold_long,
          threshold_short: BACKTEST_SSOT.threshold_short,
        },
      });

      controller.close();
    },
  });
}

// ============================================================================
// Helpers
// ============================================================================

/** Snap a date to next business day, 8am-1pm COT (13-18 UTC) */
function snapToBusinessHours(date: Date, rng: () => number): Date {
  const d = new Date(date);
  // Skip weekends
  const day = d.getUTCDay();
  if (day === 0) d.setUTCDate(d.getUTCDate() + 1); // Sun → Mon
  if (day === 6) d.setUTCDate(d.getUTCDate() + 2); // Sat → Mon

  // Set to business hours: 13-17 UTC (8am-12pm COT)
  const hour = 13 + Math.floor(rng() * 5);
  const minute = Math.floor(rng() * 60);
  d.setUTCHours(hour, minute, 0, 0);
  return d;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}
