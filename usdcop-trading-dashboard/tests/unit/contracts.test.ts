/**
 * Contract Tests - Bidirectional validation
 * ==========================================
 * Ensures Zod schemas match expected backend shapes and SSOT invariants hold.
 */

import { describe, it, expect } from 'vitest';
import {
  BacktestResultSchema,
  BacktestProgressSchema,
  BacktestTradeSchema,
  BacktestSummarySchema,
} from '@/lib/contracts/backtest.contract';
import {
  OBSERVATION_DIM,
  FEATURE_ORDER,
  ACTION_COUNT,
  Action,
} from '@/lib/contracts/ssot.contract';

// ============================================================================
// Fixtures
// ============================================================================

const VALID_TRADE = {
  trade_id: 1,
  model_id: 'ppo_v20',
  timestamp: '2025-07-01T13:00:00Z',
  entry_time: '2025-07-01T13:00:00Z',
  exit_time: '2025-07-01T14:30:00Z',
  side: 'LONG',
  entry_price: 4200.50,
  exit_price: 4215.30,
  pnl: 14.80,
  pnl_usd: 148.00,
  pnl_percent: 0.352,
  pnl_pct: 0.352,
  status: 'closed',
  duration_minutes: 90,
  exit_reason: 'take_profit',
  equity_at_entry: 10000,
  equity_at_exit: 10148,
  entry_confidence: 0.85,
  exit_confidence: 0.72,
};

const VALID_SUMMARY = {
  total_trades: 30,
  winning_trades: 17,
  losing_trades: 13,
  win_rate: 56.67,
  total_pnl: 523.40,
  total_return_pct: 5.23,
  max_drawdown_pct: 3.12,
  sharpe_ratio: 1.45,
  avg_trade_duration_minutes: 95,
};

const VALID_RESULT = {
  success: true,
  source: 'generated' as const,
  trade_count: 1,
  trades: [VALID_TRADE],
  summary: VALID_SUMMARY,
  processing_time_ms: 1234,
  date_range: { start: '2025-01-01', end: '2025-06-30' },
};

// ============================================================================
// BacktestResultSchema
// ============================================================================

describe('BacktestResultSchema', () => {
  it('accepts a valid result', () => {
    const result = BacktestResultSchema.safeParse(VALID_RESULT);
    expect(result.success).toBe(true);
  });

  it('rejects missing required field (success)', () => {
    const { success: _s, ...rest } = VALID_RESULT;
    const result = BacktestResultSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });

  it('rejects invalid source enum', () => {
    const result = BacktestResultSchema.safeParse({ ...VALID_RESULT, source: 'invalid' });
    expect(result.success).toBe(false);
  });
});

// ============================================================================
// BacktestProgressSchema
// ============================================================================

describe('BacktestProgressSchema', () => {
  it('accepts valid progress at boundaries', () => {
    const r0 = BacktestProgressSchema.safeParse({
      progress: 0, current_bar: 0, total_bars: 100, trades_generated: 0, status: 'loading', message: 'Starting',
    });
    const r1 = BacktestProgressSchema.safeParse({
      progress: 1, current_bar: 100, total_bars: 100, trades_generated: 30, status: 'completed', message: 'Done',
    });
    expect(r0.success).toBe(true);
    expect(r1.success).toBe(true);
  });

  it('rejects progress outside [0,1]', () => {
    const rNeg = BacktestProgressSchema.safeParse({
      progress: -0.1, current_bar: 0, total_bars: 100, trades_generated: 0, status: 'x', message: '',
    });
    const rOver = BacktestProgressSchema.safeParse({
      progress: 1.1, current_bar: 0, total_bars: 100, trades_generated: 0, status: 'x', message: '',
    });
    expect(rNeg.success).toBe(false);
    expect(rOver.success).toBe(false);
  });
});

// ============================================================================
// BacktestTradeSchema
// ============================================================================

describe('BacktestTradeSchema', () => {
  it('accepts a complete trade', () => {
    const result = BacktestTradeSchema.safeParse(VALID_TRADE);
    expect(result.success).toBe(true);
  });

  it('accepts trade with nullable optional fields as null', () => {
    const trade = { ...VALID_TRADE, exit_time: null, exit_price: null, exit_reason: null };
    const result = BacktestTradeSchema.safeParse(trade);
    expect(result.success).toBe(true);
  });
});

// ============================================================================
// BacktestSummarySchema
// ============================================================================

describe('BacktestSummarySchema', () => {
  it('accepts valid summary with all metrics', () => {
    const result = BacktestSummarySchema.safeParse(VALID_SUMMARY);
    expect(result.success).toBe(true);
  });

  it('rejects win_rate > 100', () => {
    const result = BacktestSummarySchema.safeParse({ ...VALID_SUMMARY, win_rate: 101 });
    expect(result.success).toBe(false);
  });

  it('accepts null sharpe_ratio', () => {
    const result = BacktestSummarySchema.safeParse({ ...VALID_SUMMARY, sharpe_ratio: null });
    expect(result.success).toBe(true);
  });
});

// ============================================================================
// SSOT Invariants
// ============================================================================

describe('SSOT Invariants', () => {
  it('OBSERVATION_DIM === FEATURE_ORDER.length', () => {
    expect(OBSERVATION_DIM).toBe(FEATURE_ORDER.length);
  });

  it('ACTION_COUNT === Object.keys(Action).length', () => {
    expect(ACTION_COUNT).toBe(Object.keys(Action).length);
  });
});
