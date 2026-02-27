/**
 * Universal Strategy Contract (SDD)
 * ==================================
 * Strategy-agnostic types for the dashboard.
 * ANY strategy (ML, RL, hybrid) conforms to these interfaces.
 *
 * Spec: .claude/rules/sdd-strategy-spec.md
 * Python mirror: src/contracts/strategy_schema.py
 */

// -----------------------------------------------------------------------------
// Universal Trade Record
// -----------------------------------------------------------------------------

export interface StrategyTrade {
  trade_id: number;
  timestamp: string;           // ISO8601 with timezone
  exit_timestamp?: string | null;     // ISO8601 (null if open)
  side: 'LONG' | 'SHORT';
  entry_price: number;
  exit_price: number | null;         // null if trade still open
  pnl_usd: number | null;           // null if trade still open
  pnl_pct: number | null;           // null if trade still open
  exit_reason: string | null;        // null if trade still open
  equity_at_entry: number;
  equity_at_exit: number | null;     // null if trade still open
  leverage: number;
  [key: string]: unknown;      // Strategy-specific metadata (confidence_tier, etc.)
}

// -----------------------------------------------------------------------------
// Universal Strategy Stats
// -----------------------------------------------------------------------------

export interface StrategyStats {
  final_equity: number;
  total_return_pct: number;
  annualized_return_pct?: number;
  volatility_pct?: number;
  sharpe?: number;
  max_dd_pct?: number;
  win_rate_pct?: number;
  profit_factor?: number | null;  // null if no losses (NEVER Infinity)
  trading_days?: number;
  avg_leverage?: number;
  exit_reasons?: Record<string, number>;
  n_long?: number;
  n_short?: number;
  [key: string]: unknown;         // Strategy-specific extras
}

// -----------------------------------------------------------------------------
// Universal Summary (strategy-agnostic)
// -----------------------------------------------------------------------------

export interface StrategySummary {
  generated_at: string;
  strategy_name: string;
  strategy_id: string;                          // KEY: lookup in strategies map
  year: number;
  initial_capital: number;
  n_trading_days: number;
  direction_accuracy_pct?: number | null;
  strategies: Record<string, StrategyStats>;    // Dynamic keys (buy_and_hold + strategy_id)
  statistical_tests: {
    p_value: number;
    significant: boolean;
    t_stat?: number;
    bootstrap_95ci_ann?: [number, number];
  };
  monthly?: {
    months: string[];
    trades?: number[];
    pnl_pct?: number[];
  };
}

// -----------------------------------------------------------------------------
// Universal Trade File
// -----------------------------------------------------------------------------

export interface StrategyTradeFile {
  strategy_name: string;
  strategy_id: string;
  initial_capital: number;
  date_range: { start: string; end: string };
  trades: StrategyTrade[];
  summary: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_pnl: number;
    total_return_pct: number;
    max_drawdown_pct: number;
    sharpe_ratio: number;
    profit_factor?: number | null;
    p_value?: number;
    direction_accuracy_pct?: number;
    n_long?: number;
    n_short?: number;
  };
}

// -----------------------------------------------------------------------------
// Exit Reason Display Metadata
// -----------------------------------------------------------------------------

export const EXIT_REASON_COLORS: Record<string, { bg: string; text: string; bar: string }> = {
  take_profit:     { bg: 'bg-emerald-500/20', text: 'text-emerald-400', bar: 'bg-emerald-500' },
  trailing_stop:   { bg: 'bg-emerald-500/20', text: 'text-emerald-400', bar: 'bg-emerald-500' },
  hard_stop:       { bg: 'bg-red-500/20',     text: 'text-red-400',     bar: 'bg-red-500'     },
  week_end:        { bg: 'bg-blue-500/20',    text: 'text-blue-400',    bar: 'bg-blue-500'    },
  session_close:   { bg: 'bg-blue-500/20',    text: 'text-blue-400',    bar: 'bg-blue-500'    },
  circuit_breaker: { bg: 'bg-amber-500/20',   text: 'text-amber-400',   bar: 'bg-amber-500'   },
  no_bars:         { bg: 'bg-slate-500/20',   text: 'text-slate-400',   bar: 'bg-slate-500'   },
};

/** Default style for unknown exit reasons */
export const DEFAULT_EXIT_REASON_COLOR = {
  bg: 'bg-slate-500/20', text: 'text-slate-400', bar: 'bg-slate-500',
};

/**
 * Get exit reason color with fallback for unknown reasons.
 */
export function getExitReasonColor(reason: string) {
  return EXIT_REASON_COLORS[reason] || DEFAULT_EXIT_REASON_COLOR;
}
