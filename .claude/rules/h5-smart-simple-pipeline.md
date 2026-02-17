# Rule: H5 Weekly Smart Simple Pipeline

> Governs the H5 (5-day horizon) weekly forecasting pipeline with Smart Simple v1.1 execution.
> This is the second production track alongside the H1 daily pipeline.
> Created: 2026-02-16

---

## Architecture

```
Sunday 01:30 COT
    |
    v
H5-L3: Weekly Training (forecast_h5_l3_weekly_training.py)
    |  Train Ridge + BayesianRidge on expanding window (2020 -> last Friday)
    |  21 features, target = ln(close[t+5]/close[t])
    |  Write to forecast_h5_predictions
    |
Monday 08:15 COT
    |
    v
H5-L5: Weekly Signal (forecast_h5_l5_weekly_signal.py)
    |  Generate ensemble prediction (mean of Ridge + BR)
    |  Score confidence (3-tier: HIGH/MEDIUM/LOW)
    |  Skip LOW-confidence LONGs
    |  Write to forecast_h5_signals
    |
Monday 08:45 COT
    |
    v
H5-L5: Vol-Targeting (forecast_h5_l5_vol_targeting.py)
    |  Compute realized vol (21d lookback)
    |  Base leverage from vol-targeting (tv=0.15)
    |  Apply asymmetric sizing + confidence multiplier
    |  Compute adaptive stops (HS = clamp(vol*sqrt(5)*2.0, 1%, 3%))
    |  Write adjusted_leverage, hard_stop_pct, take_profit_pct to forecast_h5_signals
    |
Monday 09:00 COT
    |
    v
H5-L7: Entry (forecast_h5_l7_multiday_executor.py)
    |  Read signal + stops from DB
    |  Place limit entry order (0% maker fee)
    |  Write to forecast_h5_executions + forecast_h5_subtrades
    |
Mon-Fri */30 9:00-13:00 COT
    |
    v
H5-L7: Monitor TP/HS (same DAG, monitor_position task)
    |  Check hard_stop: direction=SHORT -> bar_high >= entry * (1 + HS%)
    |  Check take_profit: direction=SHORT -> bar_low <= entry * (1 - TP%)
    |  If hit: close subtrade, update execution status
    |  No trailing stop, no re-entry
    |
Friday 12:50 COT
    |
    v
H5-L7: Friday Close (same DAG, close_week task)
    |  Market order to close remaining position
    |  Update execution status + subtrade
    |
Friday 14:30 COT
    |
    v
H5-L6: Weekly Monitor (forecast_h5_l6_weekly_monitor.py)
    |  Calculate DA, Sharpe, MaxDD
    |  Check guardrails (circuit breaker, long insistence, rolling DA)
    |  Write to forecast_h5_paper_trading
    |  Evaluate promotion gates (week 15)
```

---

## Smart Simple v1.1 Config

**File**: `config/execution/smart_simple_v1.yaml`

### Key Parameters (DO NOT change without backtest evidence)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `vol_multiplier` | **2.0** | Wider stops eliminate hard stops (v1.0 had 2, v1.1 has 0) |
| `tp_ratio` | 0.5 | TP = HS * 0.5 (sweet spot from sensitivity analysis) |
| `hard_stop_min_pct` | 1% | Floor for calm markets |
| `hard_stop_max_pct` | 3% | Cap for volatile markets |
| SHORT sizing | **flat 1.5x** | Scorer doesn't discriminate (LOW WR 79% > MEDIUM 60%, N=24) |
| LONG LOW sizing | **0.0 (SKIP)** | Net if taken = -0.75%, correct to skip |
| LONG HIGH/MED sizing | 1.0x / 0.5x | Exploratory, small size |
| `target_vol` | 0.15 | 15% annualized target |
| `max_leverage` | 2.0 | Absolute ceiling |
| `min_leverage` | 0.5 | Absolute floor |

### Confidence Scoring

Inputs: Ridge prediction, BayesianRidge prediction, ensemble direction.

| Tier | Condition | SHORT mult | LONG mult |
|------|-----------|------------|-----------|
| HIGH | Tight agreement + high magnitude | 1.5x | 1.0x |
| MEDIUM | Loose agreement OR medium magnitude | 1.5x | 0.5x |
| LOW | Neither | 1.5x | **SKIP** |

Agreement = `|ridge_pred - br_pred|`. Tight < 0.1%, Loose < 0.5%.
Magnitude = `|ensemble_mean|`. High > 1.0%, Medium > 0.5%.

### Adaptive Stops Formula

```python
vol_weekly = realized_vol_annualized * sqrt(5/252)
hard_stop_pct = clamp(vol_weekly * vol_multiplier, min_pct, max_pct)
take_profit_pct = hard_stop_pct * tp_ratio
```

Examples (vol_multiplier=2.0):
- vol_ann=10% -> HS=2.81%, TP=1.41%
- vol_ann=15% -> HS=3.00%, TP=1.50% (capped)
- vol_ann= 5% -> HS=1.41%, TP=0.70%

---

## Database Tables

**Migration**: `database/migrations/043_forecast_h5_tables.sql` + `044_smart_simple_columns.sql`

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `forecast_h5_predictions` | Model predictions per week | week, model_name, pred_return, pred_direction |
| `forecast_h5_signals` | Ensemble signal + confidence + stops | week, ensemble_return, direction, confidence_tier, sizing_multiplier, skip_trade, hard_stop_pct, take_profit_pct, adjusted_leverage |
| `forecast_h5_executions` | Weekly execution tracking | week, direction, entry_price, exit_price, exit_reason, pnl_pct, confidence_tier |
| `forecast_h5_subtrades` | Individual subtrade records | execution_id, entry_bar, exit_bar, entry_price, exit_price |
| `forecast_h5_paper_trading` | Weekly paper trading evaluation | week, da_pct, sharpe, max_dd, gate_status |

**Views**:
- `v_h5_performance_summary` — Aggregated performance metrics
- `v_h5_collapse_monitor` — Direction collapse detection

---

## DAG Files

| DAG | File | Schedule (COT) |
|-----|------|-----------------|
| H5-L3 Training | `airflow/dags/forecast_h5_l3_weekly_training.py` | Sun 01:30 |
| H5-L4 Backtest Promotion | `airflow/dags/forecast_h5_l4_backtest_promotion.py` | Manual |
| H5-L5 Signal | `airflow/dags/forecast_h5_l5_weekly_signal.py` | Mon 08:15 |
| H5-L5 Vol-Target | `airflow/dags/forecast_h5_l5_vol_targeting.py` | Mon 08:45 |
| H5-L7 Executor | `airflow/dags/forecast_h5_l7_multiday_executor.py` | Mon-Fri */30 9-13 |
| H5-L6 Monitor | `airflow/dags/forecast_h5_l6_weekly_monitor.py` | Fri 14:30 |

---

## Guardrails

| Guardrail | Config Key | Threshold | Action |
|-----------|-----------|-----------|--------|
| Long insistence | `guardrails.long_insistence_alarm` | >60% LONGs in 8-week window | Alert only |
| Rolling DA (SHORT) | `guardrails.rolling_da_monitor` | SHORT DA < 55% in 16 weeks | Pause SHORTs |
| Rolling DA (LONG) | `guardrails.rolling_da_monitor` | LONG DA < 45% in 16 weeks | Pause LONGs |
| Circuit breaker | `guardrails.circuit_breaker` | 5 consecutive losses OR 12% DD | Pause + alert |

---

## Promotion Gates (Week 15)

| Gate | Threshold | Action if PASS | Action if FAIL |
|------|-----------|----------------|----------------|
| DA overall > 55% AND DA SHORT > 60% | Promote | Check keep conditions |
| DA overall < 50% | -- | Discard |
| SHORT DA > 60% but LONG DA < 45% | -- | Switch to SHORT-only |

---

## Backtest Results (v1.1, 2025 OOS)

| Metric | Value |
|--------|-------|
| Return | +20.03% |
| Sharpe | 3.516 |
| p-value | 0.0097 |
| MaxDD | -3.83% |
| Trades | 24 (all SHORT) |
| WR | 70.8% |
| DA | 62.5% |
| Hard stops | 0 |
| Exits | 9 TP, 15 week_end |
| $10K -> | $12,003 |

### 2026 YTD

| Metric | Value |
|--------|-------|
| Return | +7.12% |
| Trades | 3 (all SHORT, all winners) |
| $10K -> | $10,712 |

---

## Dashboard Integration (SDD)

The H5 pipeline exports to the dashboard via the **SDD (Spec-Driven Development)** universal contract.
See these specs for the full data flow:

- `.claude/rules/sdd-strategy-spec.md` — Universal strategy schemas (trades, metrics, exit reasons)
- `.claude/rules/sdd-dashboard-integration.md` — JSON file schemas, PNG conventions
- `.claude/rules/sdd-approval-spec.md` — Approval lifecycle and gates
- `.claude/rules/sdd-pipeline-lifecycle.md` — 8-stage pipeline lifecycle
- `.claude/rules/sdd-mlops-lifecycle.md` — Master lifecycle (bootstrap to production)

### Export Script

```bash
python scripts/train_and_export_smart_simple.py --phase both                  # Full pipeline
python scripts/train_and_export_smart_simple.py --phase backtest              # 2025 OOS only
python scripts/train_and_export_smart_simple.py --phase production            # 2026 only
python scripts/train_and_export_smart_simple.py --phase production --seed-db  # 2026 + seed DB tables
python scripts/train_and_export_smart_simple.py --reset-approval              # Reset to PENDING
```

> `--phase backtest` embeds a `deploy_manifest` in `approval_state.json`. On Approve,
> the deploy API reads the manifest and spawns the correct script with `--seed-db`.

### Key Contracts
- TypeScript: `lib/contracts/strategy.contract.ts` (universal types)
- TypeScript: `lib/contracts/production-approval.contract.ts` (DeployManifest, ApprovalState)
- Python: `src/contracts/strategy_schema.py` (mirror types + safe JSON serializer)
- YAML: `config/strategy_registry.yaml` (strategy-to-pipeline mapping reference)

### DB Tables (NOT yet connected to dashboard)
The H5 pipeline also writes to 5 DB tables for production monitoring.
These are read by the Airflow DAGs but NOT by the dashboard frontend:
1. API routes: `/api/h5/signals`, `/api/h5/performance`, `/api/h5/executions`
2. Dashboard page: `app/h5/page.tsx` or section in `/production`
3. Update `/api/production/monitor` to include H5 data

---

## DO NOT

- Do NOT add confidence tiers for SHORT sizing (flat 1.5x is correct, N too small)
- Do NOT reduce HS multiplier below 2.0 (eliminates hard stops)
- Do NOT enable trailing stop in Smart Simple (simplicity is the edge)
- Do NOT take LOW-confidence LONGs (net effect = -0.75%)
- Do NOT hardcode params in scripts (read from smart_simple_v1.yaml)
- Do NOT modify smart_simple_v1.yaml stops without running full comparative backtest
- Do NOT compare H5 vs H1 returns directly (different horizons, different trade frequency)
