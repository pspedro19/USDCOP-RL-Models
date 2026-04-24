/**
 * PITCH METRICS — Single Source of Truth
 *
 * All numbers shown on-screen in the pitch come from here.
 * Every metric has a `source` path pointing to the file it was audited from.
 * NEVER hardcode these values elsewhere; always import from `PITCH_METRICS`.
 */

export const PITCH_METRICS = {
  /** 2025 Out-of-Sample backtest — data never seen during training */
  oos_2025: {
    label: "Validación Out-of-Sample 2025",
    return_pct: 25.63,
    sharpe: 3.347,
    p_value: 0.0063,
    max_dd_pct: 6.12,
    trades_total: 34,
    trades_long: 5,
    trades_short: 29,
    win_rate: 82.4,
    tp_exits: 21,
    hs_exits: 2,
    initial_capital: 10_000,
    final_equity: 12_563,
    source: "../usdcop-trading-dashboard/public/data/production/summary_2025.json",
  },

  /** 2026 Year-to-Date — live production */
  ytd_2026: {
    label: "Producción 2026 YTD",
    return_pct: 0.61,
    trades: 1,
    losses: 0,
    gate_blocked_weeks: 13,
    total_weeks: 14,
    buy_and_hold_pct: -2.82,
    alpha_pp: 3.43,
    initial_capital: 10_000,
    final_equity: 10_061,
    source: "../usdcop-trading-dashboard/public/data/production/summary.json",
  },

  /** Regime gate — Hurst R/S coefficient */
  regime_gate: {
    hurst_2025: 0.532,
    hurst_2026_q1: 0.28,
    threshold_mean_rev: 0.42,
    threshold_trending: 0.52,
    label_mean_rev: "Mean-Reverting",
    label_trending: "Trending",
    label_indet: "Indeterminate",
    source: "src/forecasting/regime_gate.py",
  },

  /** Infrastructure / system scale */
  infra: {
    airflow_dags: 37,
    docker_services: 25,
    ml_models: 9,
    backtests_run: 63,
    db_migrations: 43,
    dashboard_pages: 8,
    db_ohlcv_rows: 286_000,
    db_macro_rows: 10_800,
    source: "Audited 2026-04-16 via 10-agent project audit",
  },

  /** Approval gates (5/5 passed in 2025 backtest) */
  approval: {
    gates_total: 5,
    gates_passed: 5,
    min_return_pct: -15,
    actual_return_pct: 25.63,
    min_sharpe: 0,
    actual_sharpe: 3.347,
    max_dd_pct: 20,
    actual_dd_pct: 6.12,
    min_trades: 10,
    actual_trades: 34,
    p_threshold: 0.05,
    p_actual: 0.0063,
    source: "../usdcop-trading-dashboard/public/data/production/approval_state.json",
  },
} as const;

export type PitchMetrics = typeof PITCH_METRICS;

/** Helper formatters used across scenes */
export const fmt = {
  pct: (v: number, digits = 2) => `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`,
  ratio: (v: number, digits = 2) => v.toFixed(digits),
  money: (v: number) =>
    `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`,
  int: (v: number) => v.toLocaleString("en-US"),
  pValue: (v: number) => (v < 0.01 ? `p=${v.toFixed(4)}` : `p=${v.toFixed(3)}`),
};
