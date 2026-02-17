# SDD Spec: Dashboard Integration

> **Responsibility**: Authoritative source for file layouts, JSON export examples,
> CSV data contract, PNG conventions, and page data flows.
> Strategy schemas (TradeRecord, MetricsSummary) are defined in `sdd-strategy-spec.md`.
> Approval gate schemas are defined in `sdd-approval-spec.md`.
>
> Contracts:
> - TypeScript: `lib/contracts/strategy.contract.ts`
> - Python: `src/contracts/strategy_schema.py`

---

## Page Inventory

| Page | Route | Data Source | Purpose |
|------|-------|-------------|---------|
| Hub | `/hub` | Various | Landing page with navigation to all sections |
| Forecasting | `/forecasting` | CSV + PNGs in `public/forecasting/` | Model zoo: 9 models, backtests, weekly forwards |
| Dashboard | `/dashboard` | `summary_2025.json` + trades + `approval_state.json` | 2025 backtest review + human approval (Vote 2/2) |
| Production | `/production` | `summary.json` + trades + `approval_state.json` | 2026 YTD read-only monitoring |

### Page Activation (by lifecycle stage)

| Page | Activates After | Key Files Required |
|------|----------------|--------------------|
| `/hub` | Always available | None |
| `/forecasting` | Stage 2 (Model Training) | `bi_dashboard_unified.csv`, backtest PNGs |
| `/dashboard` | Stage 2 (`--phase backtest`) | `summary_2025.json`, `approval_state.json`, `trades/*_2025.json` |
| `/production` | Stage 6 (`--phase production`) | `summary.json`, `trades/*.json` |

---

## File Layout

### Production Data (`public/data/production/`)

```
public/data/production/
├── summary.json                          <- Production metrics (2026 YTD)
├── summary_2025.json                     <- Backtest metrics (2025 OOS)
├── approval_state.json                   <- Gate results + approval status
├── trades/
│   ├── smart_simple_v11.json             <- Production trades (2026)
│   └── smart_simple_v11_2025.json        <- Backtest trades (2025)
├── equity_curve_2025.png                 <- Optional: backtest equity curve
├── equity_curve_2026.png                 <- Optional: production equity curve
├── monthly_pnl_{year}.png                <- Optional: monthly PnL bar chart
└── trade_distribution_{year}.png         <- Optional: PnL histogram
```

### Forecasting Data (`public/forecasting/`)

```
public/forecasting/
├── bi_dashboard_unified.csv              <- All model metrics (505+ rows)
├── backtest_{model}_h{horizon}.png       <- 63 files (9 models x 7 horizons)
├── forward_{model}_{week}.png            <- Weekly forward PNGs per model
├── forward_consensus_{week}.png          <- Consensus direction PNG
├── forward_ensemble_top_3_{week}.png     <- Ensemble variant PNGs
├── forward_ensemble_best_of_breed_{week}.png
└── forward_ensemble_top_6_mean_{week}.png
```

---

## Forecasting Page Data Contract

### CSV Schema (`bi_dashboard_unified.csv`)

The forecasting CSV is the single data source for the `/forecasting` page. It contains all model metrics across horizons, views, and weeks.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `record_id` | string | Unique row identifier | `backtest_ridge_h1` |
| `view_type` | enum | `"backtest"` or `"forward"` | `"backtest"` |
| `model_name` | string | Model identifier | `"ridge"`, `"hybrid_xgboost"` |
| `horizon` | int | Forecast horizon in days | 1, 5, 10, 15, 20, 25, 30 |
| `week` | string | ISO week (forward view only) | `"2026_W07"` |
| `da_pct` | float | Direction accuracy percentage | 58.3 |
| `rmse` | float | Root mean square error | 0.0123 |
| `sharpe` | float | Annualized Sharpe ratio | 2.15 |
| `return_pct` | float | Total return percentage | 21.6 |
| `pred_direction` | string | Predicted direction (forward) | `"SHORT"` |
| `pred_return` | float | Predicted return (forward) | -0.8 |
| `confidence` | float | Prediction confidence | 0.72 |

### How ForecastingDashboard Loads Data

```
1. Fetch /forecasting/bi_dashboard_unified.csv
2. Parse CSV → array of ForecastRecord objects
3. Filter by view_type → "backtest" or "forward"
4. Filter by model_name → selected model
5. Filter by horizon → selected horizon (backtest) or week (forward)
6. Render:
   - Backtest: MetricsRankingPanel ranks models by DA, return, Sharpe
   - Forward: consensus direction + ensemble variants
```

### PNG Naming Convention

| Type | Pattern | Example | Count |
|------|---------|---------|-------|
| Backtest | `backtest_{model}_h{horizon}.png` | `backtest_ridge_h1.png` | 63 (9 x 7) |
| Forward (model) | `forward_{model}_{week}.png` | `forward_ridge_2026_W07.png` | ~9 x N weeks |
| Forward (consensus) | `forward_consensus_{week}.png` | `forward_consensus_2026_W07.png` | N weeks |
| Forward (ensemble) | `forward_ensemble_{variant}_{week}.png` | `forward_ensemble_top_3_2026_W07.png` | ~3 x N weeks |

**Models**: `ridge`, `bayesian_ridge`, `ard`, `xgboost_pure`, `lightgbm_pure`, `catboost_pure`, `hybrid_xgboost`, `hybrid_lightgbm`, `hybrid_catboost`

**Horizons**: 1, 5, 10, 15, 20, 25, 30

**Ensemble variants**: `top_3`, `best_of_breed`, `top_6_mean`

---

## JSON Schema: StrategySummary (`summary.json` / `summary_{year}.json`)

```json
{
  "generated_at": "2026-02-16T14:30:00",
  "strategy_name": "Smart Simple v1.1.0",
  "strategy_id": "smart_simple_v11",
  "year": 2026,
  "initial_capital": 10000.0,
  "n_trading_days": 35,
  "direction_accuracy_pct": 62.5,

  "strategies": {
    "buy_and_hold": {
      "final_equity": 8771.0,
      "total_return_pct": -12.29
    },
    "smart_simple_v11": {
      "final_equity": 12003.0,
      "total_return_pct": 20.03,
      "sharpe": 3.516,
      "max_dd_pct": 3.83,
      "win_rate_pct": 70.8,
      "profit_factor": 5.23,
      "trading_days": 120,
      "exit_reasons": {
        "take_profit": 9,
        "week_end": 15
      },
      "n_long": 0,
      "n_short": 24
    }
  },

  "statistical_tests": {
    "p_value": 0.0097,
    "significant": true,
    "t_stat": 3.12,
    "bootstrap_95ci_ann": [0.05, 0.35]
  },

  "monthly": {
    "months": ["2025-01", "2025-02", "2025-03"],
    "trades": [4, 3, 5],
    "pnl_pct": [2.1, -0.5, 3.2]
  }
}
```

### Key Rules

1. **`strategy_id`** is the dynamic key in `strategies` object
2. **`buy_and_hold`** is always present as a baseline comparison
3. **`profit_factor`** is `null` if no losses (NEVER `Infinity`)
4. **`statistical_tests.t_stat`** and **`bootstrap_95ci_ann`** are optional
5. **`monthly`** is optional; `trades` and `pnl_pct` arrays must match `months` length
6. **Dashboard lookup**: `summary.strategies[summary.strategy_id]` gets the active strategy stats

### JSON Safety (CRITICAL)

JSON files MUST NOT contain:
- `Infinity` — use `null` instead
- `NaN` — use `null` instead
- `undefined` — omit the field or use `null`

Python export MUST use a safe serializer:
```python
import math

def _json_default(obj):
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)
```

---

## JSON Schema: StrategyTradeFile (`trades/{strategy_id}.json`)

```json
{
  "strategy_name": "Smart Simple v1.1.0",
  "strategy_id": "smart_simple_v11",
  "initial_capital": 10000.0,
  "date_range": {
    "start": "2026-01-06",
    "end": "2026-02-14"
  },
  "trades": [
    {
      "trade_id": 1,
      "timestamp": "2026-01-06T09:00:00-05:00",
      "exit_timestamp": "2026-01-10T12:50:00-05:00",
      "side": "SHORT",
      "entry_price": 4380.5,
      "exit_price": 4350.2,
      "pnl_usd": 103.5,
      "pnl_pct": 1.035,
      "exit_reason": "take_profit",
      "equity_at_entry": 10000.0,
      "equity_at_exit": 10103.5,
      "leverage": 1.5,
      "confidence_tier": "HIGH",
      "hard_stop_pct": 2.81,
      "take_profit_pct": 1.41
    }
  ],
  "summary": {
    "total_trades": 24,
    "winning_trades": 17,
    "losing_trades": 7,
    "win_rate": 70.8,
    "total_pnl": 2003.0,
    "total_return_pct": 20.03,
    "max_drawdown_pct": 3.83,
    "sharpe_ratio": 3.516,
    "profit_factor": 5.23,
    "p_value": 0.0097,
    "direction_accuracy_pct": 62.5,
    "n_long": 0,
    "n_short": 24
  }
}
```

---

## Strategy Selector (Multi-Strategy Support)

For the StrategyRegistry (list of all strategies), StrategySelector TypeScript interface,
and `GET /api/production/strategies` API contract, see `sdd-strategy-spec.md` § StrategySelector Contract.

### How `/production` Uses the Selector

```
1. GET /api/production/strategies → strategy list
2. If > 1 strategy → render dropdown
3. On selection → re-fetch summary + trades for selected strategy_id
4. Default = active_strategy_id
```

**Current limitation**: Only ONE active `summary.json` at a time. Multi-strategy comparison via per-strategy summary files is a future enhancement.

---

## PNG Convention (Optional)

Strategies MAY generate PNG charts for the dashboard. The dashboard renders these
with graceful fallback (`onError` hides the container if PNG is missing).

### Production PNGs

| File | Content | Size |
|------|---------|------|
| `equity_curve_{year}.png` | Equity curve vs buy-and-hold | ~1200x600 |
| `monthly_pnl_{year}.png` | Monthly PnL bar chart | ~800x400 |
| `trade_distribution_{year}.png` | PnL histogram per trade | ~800x400 |

### Style Guide

- Dark background (`#0f172a` or similar slate-950)
- White text, grid with low opacity
- Strategy line in emerald/green, buy-and-hold in gray
- Use matplotlib with `plt.style.use('dark_background')` or equivalent

### Graceful Fallback

Dashboard images use:
```tsx
onError={(e) => {
  (e.target as HTMLImageElement).closest('div')!.style.display = 'none';
}}
```

This means missing PNGs silently hide their container — no broken image icons.

---

## Dashboard Page (`/dashboard`) Lookup Flow — Backtest 2025 + Approval

The `ForecastingBacktestSection` component renders at the top of `/dashboard`, before RL sections.

```
1. Fetch /data/production/summary_2025.json       (backtest metrics)
2. Fetch /api/production/status                    (approval state)
3. Read strategy_id from summary_2025.strategy_id
4. Fetch /data/production/trades/{strategy_id}_2025.json  (backtest trades)
5. Render KPIs, chart (2025 candles), gates, trades
6. Show interactive Approve/Reject panel (Vote 2/2) if PENDING_APPROVAL
7. POST /api/production/approve on user action
```

**Key**: Vote 2 (human review) happens on `/dashboard`, NOT on `/production`.

---

## Production Page (`/production`) Lookup Flow — 2026 Read-Only

```
1. GET /api/production/strategies                  (strategy list for selector)
2. Fetch /data/production/summary.json             (production 2026 metrics)
3. Fetch /api/production/status                    (approval state, read-only)
4. Read strategy_id from summary.strategy_id
5. Fetch /data/production/trades/{strategy_id}.json (production trades)
6. Render KPIs, chart (full 2026 YTD candles), gates (read-only)
7. Show read-only status badge (no approve/reject buttons)
8. Attempt to load PNGs (graceful fallback on error)
```

**Key**: `/production` is read-only. No interactive approval actions.

---

## Related Specs

- `sdd-mlops-lifecycle.md` — **Master lifecycle document** (Stages 0-7)
- `sdd-strategy-spec.md` — Universal strategy schemas + StrategySelector contract
- `sdd-approval-spec.md` — Approval gates and lifecycle
- `sdd-pipeline-lifecycle.md` — 8-stage pipeline quick reference
