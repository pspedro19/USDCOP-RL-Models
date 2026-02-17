/**
 * Forecast Backtest Service
 * ==========================
 * Serves pre-computed forecast strategy trades as SSE streams.
 *
 * Reads static JSON files from public/data/forecast-trades/
 * and streams them progressively using the same BacktestTrade contract
 * that the RL backtest replay uses.
 *
 * Strategies:
 * - fc_buy_hold: Buy & Hold USDCOP 2025
 * - fc_forecast_1x: Forecast direction only (1x leverage)
 * - fc_forecast_vt: Forecast + Vol-Targeting (dynamic leverage)
 * - fc_forecast_vt_trail: Forecast + VT + Trailing Stop (best strategy)
 */

import type { BacktestTrade, BacktestSummary, ReplaySpeed } from '@/lib/contracts/backtest.contract';
import { getReplayDelay, DEFAULT_REPLAY_SPEED } from '@/lib/contracts/backtest.contract';

// ============================================================================
// Strategy Registry
// ============================================================================

export const FORECAST_STRATEGIES: Record<string, {
  file: string;
  name: string;
  color: string;
  description: string;
}> = {
  fc_buy_hold: {
    file: 'data/forecast-trades/buy_and_hold.json',
    name: 'Buy & Hold USDCOP',
    color: '#6B7280',
    description: 'Passive long position in USD/COP for all of 2025',
  },
  fc_forecast_1x: {
    file: 'data/forecast-trades/forecast_1x.json',
    name: 'Forecast 1x (Direction)',
    color: '#3B82F6',
    description: '9-model ensemble, daily direction, 1x fixed leverage',
  },
  fc_forecast_vt: {
    file: 'data/forecast-trades/forecast_vol_target.json',
    name: 'Forecast + Vol-Target',
    color: '#8B5CF6',
    description: 'Forecast direction + dynamic leverage via vol-targeting',
  },
  fc_forecast_vt_trail: {
    file: 'data/forecast-trades/forecast_vt_trailing.json',
    name: 'Forecast + VT + Trail Stop',
    color: '#10B981',
    description: 'Full system: forecast + vol-target + intraday trailing stop',
  },
};

// ============================================================================
// Universal Strategy Registry (from replay_backtest_universal.py exports)
// ============================================================================

export const UNIVERSAL_STRATEGIES: Record<string, {
  file: string;
  name: string;
  color: string;
  description: string;
}> = {
  smart_simple_v11: {
    file: 'data/production/trades/smart_simple_v11_2025.json',
    name: 'Smart Simple v1.1',
    color: '#10B981',
    description: 'H5 weekly: Ridge+BR ensemble, adaptive stops, confidence sizing',
  },
  forecast_vt_trailing: {
    file: 'data/production/trades/forecast_vt_trailing_2025.json',
    name: 'H1 Forecast+VT+Trail',
    color: '#8B5CF6',
    description: 'H1 daily: 9-model ensemble, vol-target, intraday trailing stop',
  },
  rl_v215b: {
    file: 'data/production/trades/rl_v215b_2025.json',
    name: 'RL V21.5b',
    color: '#F59E0B',
    description: 'RL PPO agent on 5-min bars (deprioritized)',
  },
};

/**
 * Check if a model ID is a universal strategy
 */
export function isUniversalStrategy(modelId: string): boolean {
  return modelId in UNIVERSAL_STRATEGIES;
}

/**
 * Check if a model ID is a forecast strategy
 */
export function isForecastStrategy(modelId: string): boolean {
  return modelId in FORECAST_STRATEGIES;
}

// ============================================================================
// SSE Stream
// ============================================================================

export interface ForecastStreamConfig {
  startDate?: string;
  endDate?: string;
  replaySpeed?: ReplaySpeed;
}

/**
 * Create an SSE stream from pre-computed forecast trades.
 *
 * Phases:
 * 1. Loading events (~300ms)
 * 2. Stream each trade with speed-adjusted delay
 * 3. Final result event with summary
 */
export function createForecastSSEStream(
  trades: BacktestTrade[],
  summary: BacktestSummary,
  config: ForecastStreamConfig = {},
): ReadableStream {
  return new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      const send = (type: string, data: unknown) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, data })}\n\n`));
      };
      const delay = (ms: number) => new Promise(r => setTimeout(r, ms));

      const startMs = Date.now();
      const speed = config.replaySpeed ?? DEFAULT_REPLAY_SPEED;
      const baseDelay = getReplayDelay(speed);

      // Filter trades by date range if specified
      let filteredTrades = trades;
      if (config.startDate || config.endDate) {
        filteredTrades = trades.filter(t => {
          const ts = new Date(t.entry_time).getTime();
          if (config.startDate && ts < new Date(config.startDate).getTime()) return false;
          if (config.endDate && ts > new Date(config.endDate + 'T23:59:59Z').getTime()) return false;
          return true;
        });
      }

      const totalTrades = filteredTrades.length;

      // Phase 1: Loading
      send('progress', {
        progress: 0, current_bar: 0, total_bars: totalTrades,
        trades_generated: 0, status: 'loading',
        message: 'Loading forecast trades...',
        replay_speed: speed,
      });
      await delay(Math.min(200, 300 / speed));

      send('progress', {
        progress: 0.05, current_bar: 0, total_bars: totalTrades,
        trades_generated: 0, status: 'loading',
        message: 'Preparing equity curve...',
        replay_speed: speed,
      });
      await delay(Math.min(150, 250 / speed));

      send('progress', {
        progress: 0.10, current_bar: 0, total_bars: totalTrades,
        trades_generated: 0, status: 'running',
        message: 'Streaming trades...',
        replay_speed: speed,
      });
      await delay(Math.min(100, 200 / speed));

      // Phase 2: Stream trades
      let cumulativePnl = 0;
      let cumulativePnlUsd = 0;

      for (let i = 0; i < totalTrades; i++) {
        const trade = filteredTrades[i];
        const progress = 0.10 + (i / totalTrades) * 0.85;

        send('progress', {
          progress, current_bar: i + 1, total_bars: totalTrades,
          trades_generated: i + 1, status: 'running',
          message: `Trade ${i + 1}/${totalTrades}: ${trade.side} @ $${trade.entry_price.toFixed(2)}`,
          replay_speed: speed,
        });

        await delay(baseDelay * 0.3);

        cumulativePnl += trade.pnl || 0;
        cumulativePnlUsd += trade.pnl_usd || 0;

        send('trade', {
          ...trade,
          trade_id: String(trade.trade_id),
          current_equity: trade.equity_at_exit,
          cumulative_pnl: cumulativePnl,
          cumulative_pnl_usd: cumulativePnlUsd,
        });

        await delay(baseDelay * (0.3 + Math.random() * 0.2));
      }

      // Phase 3: Result
      send('progress', {
        progress: 0.97, current_bar: totalTrades, total_bars: totalTrades,
        trades_generated: totalTrades, status: 'saving',
        message: 'Finalizing results...',
        replay_speed: speed,
      });
      await delay(Math.min(200, 300 / speed));

      send('result', {
        success: true,
        source: 'generated',
        trade_count: totalTrades,
        trades: filteredTrades,
        summary,
        processing_time_ms: Date.now() - startMs,
        date_range: {
          start: config.startDate || (filteredTrades[0]?.entry_time ?? ''),
          end: config.endDate || (filteredTrades[totalTrades - 1]?.entry_time ?? ''),
        },
        replay_speed: speed,
      });

      controller.close();
    },
  });
}
