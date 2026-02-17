# SDD Spec: Universal Strategy Interface

> **Responsibility**: Authoritative source for strategy schemas (TradeRecord, MetricsSummary,
> ExitReasonRegistry), StrategyRegistry, and StrategySelector contract.
> Other specs reference this file for schema definitions — do not duplicate schemas elsewhere.
>
> Contracts:
> - TypeScript: `lib/contracts/strategy.contract.ts`
> - Python: `src/contracts/strategy_schema.py`

---

## StrategyManifest

Every strategy declares these identity fields in its exported JSON files:

| Field | Type | Example | Required |
|-------|------|---------|----------|
| `strategy_id` | string | `"smart_simple_v11"`, `"forecast_vt_trailing"`, `"rl_v215b"` | Yes |
| `strategy_name` | string | `"Smart Simple v1.1.0"`, `"Forecast + VT + Trail"` | Yes |
| `strategy_type` | enum | `"ml_supervised"` / `"rl_ppo"` / `"hybrid"` / `"rule_based"` | Optional |
| `timeframe` | enum | `"intraday_5m"` / `"daily"` / `"weekly"` | Optional |
| `pair` | string | `"USD/COP"` | Optional |
| `version` | semver | `"1.1.0"` | Optional |

**Key rule**: `strategy_id` is the universal lookup key. The dashboard uses it to:
1. Find the correct strategy stats in `summary.json -> strategies[strategy_id]`
2. Find the correct trade file: `trades/{strategy_id}.json`
3. Display the strategy name badge

---

## StrategyRegistry

The registry tracks all known strategies, their status, and metadata. This is the
authoritative list for multi-strategy support across the dashboard.

### Current Strategies

| Strategy ID | Name | Type | Timeframe | Status | Pipeline Script |
|-------------|------|------|-----------|--------|-----------------|
| `smart_simple_v11` | Smart Simple v1.1.0 | ml_supervised | weekly | PAPER TRADING | `scripts/train_and_export_smart_simple.py` |
| `forecast_vt_trailing` | Forecast + VT + Trail | ml_supervised | daily | PRODUCTION | `scripts/generate_weekly_forecasts.py` |
| `rl_v215b` | RL V21.5b | rl_ppo | intraday_5m | DEPRIORITIZED | `scripts/run_ssot_pipeline.py` |

### Strategy Status Enum

| Status | Meaning | Dashboard Visibility |
|--------|---------|---------------------|
| `PAPER_TRADING` | Active paper trading, not yet live | `/dashboard` + `/production` |
| `PRODUCTION` | Live trading with real capital | `/production` |
| `DEPRIORITIZED` | Not actively maintained | Hidden from `/production` |
| `ARCHIVED` | Historical, no longer running | Hidden from all pages |

### Adding a New Strategy

1. Create a pipeline script conforming to `--phase backtest|production|both` convention
2. Add to the StrategyRegistry table above
3. Export JSON files conforming to `StrategySummary` + `StrategyTradeFile` schemas
4. Add any new exit reasons to the `ExitReasonRegistry` (both TS and Python)
5. Run `--phase backtest` to generate `approval_state.json` and trigger approval flow

---

## StrategySelector Contract

When multiple strategies have exported data, the production page needs a way to list and select between them.

### TypeScript Interface

```typescript
interface StrategyOption {
  strategy_id: string;
  strategy_name: string;
  status: 'PENDING_APPROVAL' | 'APPROVED' | 'REJECTED' | 'LIVE';
  year: number;
  return_pct: number;
  has_backtest: boolean;
  has_production: boolean;
}

interface StrategyListResponse {
  strategies: StrategyOption[];
  active_strategy_id: string;
}
```

### API Endpoint

```
GET /api/production/strategies -> StrategyListResponse
```

The endpoint scans `public/data/production/` for all `summary*.json` files and
constructs the strategy list dynamically. No hardcoded strategy IDs.

### Dashboard Usage

```
1. /production loads → GET /api/production/strategies
2. If strategies.length > 1 → render dropdown selector
3. If strategies.length === 1 → hide dropdown, use single strategy
4. On selection change → re-fetch summary.json + trades for selected strategy
5. Default selection → active_strategy_id
```

### File Organization for Multi-Strategy

```
public/data/production/
├── summary.json                          <- Active strategy (current year)
├── summary_2025.json                     <- Active strategy backtest
├── approval_state.json                   <- Active strategy approval
├── trades/
│   ├── smart_simple_v11.json             <- Strategy 1 production
│   ├── smart_simple_v11_2025.json        <- Strategy 1 backtest
│   ├── forecast_vt_trailing.json         <- Strategy 2 production
│   └── forecast_vt_trailing_2025.json    <- Strategy 2 backtest
```

**Current limitation**: Only ONE `summary.json` (active strategy). Multi-strategy
comparison requires per-strategy summary files (`summary_{strategy_id}.json`), which
is a future enhancement.

---

## TradeRecord

Universal trade format. ALL strategies produce trades conforming to this schema.

| Field | Type | Description |
|-------|------|-------------|
| `trade_id` | int | Sequential within the file |
| `timestamp` | ISO8601 | Entry time (with timezone, COT -05:00) |
| `exit_timestamp` | ISO8601 or null | Exit time |
| `side` | `"LONG"` / `"SHORT"` | Trade direction |
| `entry_price` | float | Entry price |
| `exit_price` | float | Exit price |
| `pnl_usd` | float | Profit/loss in USD |
| `pnl_pct` | float | Profit/loss as percentage |
| `exit_reason` | string | Strategy-defined (see ExitReasonRegistry) |
| `equity_at_entry` | float | Portfolio equity before this trade |
| `equity_at_exit` | float | Portfolio equity after this trade |
| `leverage` | float | Leverage used |

Additional strategy-specific fields are allowed as extra keys (e.g., `confidence_tier`,
`hard_stop_pct`, `take_profit_pct`). The dashboard ignores unknown keys.

---

## MetricsSummary

Universal metrics computed by every strategy. Returned in `summary.json -> strategies[strategy_id]`.

| Field | Type | Description |
|-------|------|-------------|
| `final_equity` | float | Final portfolio value |
| `total_return_pct` | float | Total return percentage |
| `sharpe` | float or null | Annualized Sharpe ratio |
| `max_dd_pct` | float or null | Maximum drawdown percentage |
| `win_rate_pct` | float or null | Win rate percentage |
| `profit_factor` | float or **null** | Gross profit / gross loss. **null if no losses** (NEVER Infinity) |
| `trading_days` | int or null | Number of trading days |
| `exit_reasons` | `Record<string, number>` | Count per exit reason |
| `n_long` | int or null | Number of long trades |
| `n_short` | int or null | Number of short trades |

### JSON Safety Rules

- `profit_factor` MUST be `null` (not `Infinity`) when there are no losses
- Cap `profit_factor` at `999.99` for display if needed
- JSON files MUST NOT contain `Infinity`, `NaN`, or `undefined`
- Use `null` for any undefined numeric value

---

## ExitReasonRegistry

Known exit reasons with display metadata. Strategies can define new reasons;
the dashboard falls back to a neutral gray style for unknown reasons.

| Reason | Color | Label | Used By |
|--------|-------|-------|---------|
| `take_profit` | emerald | Take Profit | Smart Simple, VT+Trail |
| `trailing_stop` | emerald | Trailing Stop | VT+Trail |
| `hard_stop` | red | Hard Stop | Smart Simple, VT+Trail |
| `week_end` | blue | Fin de Semana | Smart Simple |
| `session_close` | blue | Cierre Sesion | VT+Trail |
| `circuit_breaker` | amber | Circuit Breaker | Any |
| `no_bars` | slate | No Bars | Any |

### Adding New Exit Reasons

1. Add to `EXIT_REASON_COLORS` in `strategy.contract.ts`
2. Add to `EXIT_REASONS` in `strategy_schema.py`
3. Unknown reasons render with slate/gray styling (no crash)

---

## StatisticalTests

Reported in `summary.json -> statistical_tests`:

| Field | Type | Required |
|-------|------|----------|
| `p_value` | float | Yes |
| `significant` | bool | Yes (true if p < 0.05) |
| `t_stat` | float | Optional |
| `bootstrap_95ci_ann` | `[lo, hi]` | Optional |

---

## UniversalSignalRecord (Signal Contract)

Standard signal format that decouples signal generation from execution.
Any strategy produces `UniversalSignalRecord[]`, and a single `ReplayBacktestEngine`
executes them to produce `StrategyTrade[]`.

### Contract: `src/contracts/signal_contract.py`

| Field | Type | Description |
|-------|------|-------------|
| `signal_id` | string | Unique ID: `"h5_2025-W03"`, `"h1_2025-01-15"` |
| `strategy_id` | string | Strategy key from StrategyRegistry |
| `signal_date` | string | ISO date of signal generation |
| `direction` | int | +1 (LONG), -1 (SHORT), 0 (HOLD/SKIP) |
| `magnitude` | float | Signal strength (|predicted_return|) |
| `confidence` | float | 0.0 to 1.0 |
| `skip_trade` | bool | True if strategy says "don't trade" |
| `leverage` | float | Final leverage after vol-target + multipliers |
| `hard_stop_pct` | float? | Hard stop level (null = not used) |
| `take_profit_pct` | float? | Take profit level |
| `trailing_activation_pct` | float? | Trailing stop activation |
| `trailing_distance_pct` | float? | Trailing stop distance |
| `entry_price` | float | Price at signal time |
| `entry_type` | string | "limit" or "market" |
| `horizon_bars` | int | 1 (daily), 5 (weekly), 60 (RL session) |
| `bar_frequency` | string | "daily", "weekly", "5min" |
| `metadata` | dict? | Strategy-specific opaque data |

### Adapters: `src/contracts/signal_adapters.py`

| Adapter | Strategy | Input |
|---------|----------|-------|
| `H5SmartSimpleAdapter` | smart_simple_v11 | Walk-forward Ridge+BR predictions |
| `H1ForecastVTAdapter` | forecast_vt_trailing | 9-model ensemble daily predictions |
| `RLPPOAdapter` | rl_v215b | PPO model.predict() on 5-min bars |

### Execution Strategies: `src/contracts/execution_strategies.py`

| Strategy | Used By | Logic |
|----------|---------|-------|
| `WeeklyTPHSExecution` | H5 | TP/HS/Friday close on daily bars |
| `DailyTrailingStopExecution` | H1 | Trailing stop on 5-min intraday bars |
| `IntradaySLTPExecution` | RL | SL/TP per 5-min bar |

### CLI Tools

```bash
# Generate signals
python scripts/generate_universal_signals.py --strategy smart_simple_v11 --year 2025

# Replay backtest
python scripts/replay_backtest_universal.py --strategy smart_simple_v11 --year 2025 --export-dashboard

# Compare strategies
python scripts/replay_backtest_universal.py --compare smart_simple_v11,forecast_vt_trailing --year 2025
```

### Storage: `data/signals/{strategy_id}_{year}.parquet`

---

## Related Specs

- `sdd-mlops-lifecycle.md` — **Master lifecycle document** (Stages 0-7)
- `sdd-approval-spec.md` — Approval gates and lifecycle
- `sdd-dashboard-integration.md` — JSON/PNG file schemas, page data flows
- `sdd-pipeline-lifecycle.md` — 8-stage pipeline quick reference
