# SDD Spec: MLOps Lifecycle — Bootstrap to Production

> Master operator's guide for the complete USDCOP trading system lifecycle.
> Covers all 8 stages from first-time setup to production monitoring.
> Follow stages sequentially for initial setup; Stage 7 runs automatically via Airflow.
>
> **Responsibility**: End-to-end operator workflow (WHAT to do, WHEN, and WHY).
> For deep-dives into specific concerns, see the cross-referenced specs below.
>
> | Concern | Authoritative Spec |
> |---------|-------------------|
> | Strategy schemas (TradeRecord, MetricsSummary) | `sdd-strategy-spec.md` |
> | Approval gates, 2-vote system, approval_state.json | `sdd-approval-spec.md` |
> | JSON/CSV/PNG file schemas, page data flows | `sdd-dashboard-integration.md` |
> | Quick-reference lifecycle + CLI conventions | `sdd-pipeline-lifecycle.md` |
>
> Created: 2026-02-16

---

## Overview

```
Stage 0: BOOTSTRAP        docker-compose up → DB seeded from parquets
    |
Stage 1: DATA ALIGNMENT   L0 backfill DAGs → OHLCV + macro current through yesterday
    |
Stage 2: MODEL TRAINING   Train H1 (9 models) + H5 (Ridge+BR) on expanding window
    |
Stage 3: FORECASTING       /forecasting page activates → model zoo with walk-forward metrics
    |
Stage 4: BACKTEST REVIEW   /dashboard shows 2025 OOS results → KPIs, trades, gates
    |
Stage 5: HUMAN APPROVAL    Vote 2/2 on /dashboard → APPROVED or REJECTED
    |
Stage 6: PRODUCTION DEPLOY Retrain with 2025 data → /production shows 2026 YTD
    |
Stage 7: MONITORING        Airflow weekly cycle → retrain, signal, execute, evaluate
```

**Stages 0-6** are manual (run once during initial setup or on-demand).
**Stage 7** is automated (Airflow DAGs on weekly/daily schedules).

---

## Stage 0: Bootstrap (First-Time Setup)

### What Happens

Docker Compose starts PostgreSQL + TimescaleDB. Init scripts run in order:

```
docker-compose up -d
    |
    v
00-extensions.sql         → Enable TimescaleDB, pg_cron
01-essential-tables.sql   → Core tables: usdcop_m5_ohlcv, macro_indicators_*
02-macro-tables.sql       → 4-table macro architecture (daily, monthly, quarterly)
03-views.sql              → Monitoring views: fx_latest_bar, fx_daily_bar_counts
04-data-seeding.py        → Detect empty DB → restore from seed parquets
```

### Seed Restore (`04-data-seeding.py`)

The seeding script detects empty tables and restores from local parquets:

| Seed File | Target Table | Rows | Coverage |
|-----------|-------------|------|----------|
| `seeds/latest/fx_multi_m5_ohlcv.parquet` | `usdcop_m5_ohlcv` | ~266K | 3 FX pairs, 5-min bars (2019-2026) |
| `seeds/latest/usdcop_daily_ohlcv.parquet` | (file-based, no table) | ~3K | Daily OHLCV for H1/H5 training |
| `data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet` | `macro_indicators_daily` | ~10K | 17 macro cols (2015-2026) |

### State After Stage 0

- DB has historical data but is NOT up-to-date (gap from last seed export to today)
- Dashboard is EMPTY — no models trained, no forecasts generated, no images
- All pages (`/forecasting`, `/dashboard`, `/production`) show empty states

### Operator Checklist

```
[ ] docker-compose up -d
[ ] Verify DB is running: docker exec -it usdcop-postgres psql -U postgres -c "SELECT count(*) FROM usdcop_m5_ohlcv"
[ ] Check seed restore log: docker logs usdcop-postgres 2>&1 | grep "seeding"
```

---

## Stage 1: L0 Data Alignment

### What Happens

Two backfill DAGs bring the DB current:

1. **OHLCV Backfill** (`l0_ohlcv_backfill.py`): Detects gap from last seed date to today, fetches missing bars from TwelveData for all 3 pairs (COP, MXN, BRL). Exports updated seed parquets.

2. **Macro Backfill** (`l0_macro_backfill.py`): Full extraction from 7 sources (FRED, Investing.com, BanRep, BCRP, Fedesarrollo, DANE, BanRep BOP). Exports 9 MASTER files.

### Commands

```bash
# Backfill all 3 FX pairs
airflow dags trigger core_l0_01_ohlcv_backfill

# Backfill all macro variables
airflow dags trigger core_l0_03_macro_backfill

# Or run standalone (no Airflow):
python scripts/build_unified_fx_seed.py          # Rebuild FX seeds from raw sources
python data/pipeline/04_cleaning/run_clean.py     # Rebuild MACRO_DAILY_CLEAN.parquet
```

### Going Forward

After initial alignment, realtime DAGs keep data fresh automatically:

| DAG | Schedule (COT) | Action |
|-----|----------------|--------|
| `core_l0_02_ohlcv_realtime` | `*/5 8:00-12:55 Mon-Fri` | 3 FX pairs, circuit breaker |
| `core_l0_04_macro_update` | `hourly 8:00-12:00 Mon-Fri` | 40 vars, 7 sources |

### Key Contract

- All OHLCV timestamps = `America/Bogota` timezone, session 8:00-12:55 COT
- BRL fetched with `timezone=UTC` then converted (TwelveData quirk)
- Macro features use T-1 lag: `merge_asof(direction='backward')` + `.shift(1)`

### State After Stage 1

- DB has complete OHLCV (5-min + daily) through yesterday
- DB has complete macro indicators through yesterday
- Seed files updated in `seeds/latest/` and `data/pipeline/04_cleaning/output/`

### Operator Checklist

```
[ ] OHLCV backfill completed without errors
[ ] Macro backfill completed without errors
[ ] Verify: SELECT MAX(time) FROM usdcop_m5_ohlcv WHERE symbol='USD/COP'  → yesterday
[ ] Verify: SELECT MAX(fecha) FROM macro_indicators_daily                  → yesterday
[ ] Seed parquets updated (check git diff seeds/latest/)
```

---

## Stage 2: Model Training

Two independent training scripts, run once after L0 alignment.

### A. H1 Multi-Model Training (`generate_weekly_forecasts.py`)

Trains 9 models across 7 horizons on an expanding window (2020 to last Friday).

```bash
python scripts/generate_weekly_forecasts.py
```

**Input**: `seeds/latest/usdcop_daily_ohlcv.parquet` + `MACRO_DAILY_CLEAN.parquet`

**Output** (written to `usdcop-trading-dashboard/public/forecasting/`):
- `bi_dashboard_unified.csv` — 505 rows, all model metrics across horizons and weeks
- 63 backtest PNGs: `backtest_{model}_h{horizon}.png` (9 models x 7 horizons)
- Weekly forward PNGs: `forward_{model}_{week}.png` (9 models x N weeks)

**Models**: ridge, bayesian_ridge, ard, xgboost, lightgbm, catboost, hybrid_xgboost, hybrid_lightgbm, hybrid_catboost

### B. H5 Smart Simple Backtest (`train_and_export_smart_simple.py`)

Walk-forward weekly retraining: train on 2020-2024, evaluate OOS on 2025.

```bash
python scripts/train_and_export_smart_simple.py --phase backtest
```

**Input**: Same daily OHLCV + macro parquets. Config from `config/execution/smart_simple_v1.yaml`.

**Output** (written to `usdcop-trading-dashboard/public/data/production/`):
- `summary_2025.json` — OOS backtest metrics
- `approval_state.json` — 5 gates evaluated automatically (Vote 1/2)
- `trades/smart_simple_v11_2025.json` — 24 trades with entry/exit/PnL/confidence
- Optional PNGs: `equity_curve_2025.png`, `monthly_pnl_2025.png`, `trade_distribution_2025.png`

**Vote 1/2 (Automatic)**: The export script evaluates 5 gates and writes the recommendation:

| Gate | Threshold | Smart Simple v1.1 Result |
|------|-----------|--------------------------|
| `min_return_pct` | > -15% | +20.03% PASS |
| `min_sharpe_ratio` | > 0.0 | 3.516 PASS |
| `max_drawdown_pct` | < 20% | 3.83% PASS |
| `min_trades` | >= 10 | 24 PASS |
| `statistical_significance` | p < 0.05 | 0.0097 PASS |

### State After Stage 2

- `/forecasting` page has data: CSV + 63+ PNGs (model zoo)
- `/dashboard` has 2025 backtest: summary, trades, gates with `PENDING_APPROVAL` status
- `/production` still empty (no production deploy yet)

### Operator Checklist

```
[ ] generate_weekly_forecasts.py completed (check bi_dashboard_unified.csv exists)
[ ] train_and_export_smart_simple.py --phase backtest completed
[ ] summary_2025.json exists in public/data/production/
[ ] approval_state.json shows status: PENDING_APPROVAL
[ ] All 5 gates show passed: true
```

---

## Stage 3: Forecasting Dashboard Activation (Automatic)

> No operator action required — the dashboard reads the generated files directly.

After Stage 2, the `/forecasting` page becomes fully functional:

- **Backtest View**: Select model + horizon → walk-forward validated metrics (DA, RMSE, Sharpe, return)
- **Forward View**: Select model + week → consensus direction + ensemble variants
- **Metrics Ranking Panel**: Ranks all 9 models by DA, return, Sharpe across horizons

For the full CSV schema (`bi_dashboard_unified.csv`) and PNG naming conventions,
see `sdd-dashboard-integration.md` § Forecasting Page Data Contract.

### State After Stage 3

- `/forecasting` shows full model zoo with backtests and forward forecasts
- Operator can compare all 9 models across 7 horizons
- This stage is informational — no approval action needed

---

## Stage 4: Backtest Review on `/dashboard`

### What Happens

The `ForecastingBacktestSection` component (top of `/dashboard`) loads 2025 OOS results and presents them for human review.

### Data Flow

```
/dashboard loads:
    |
    ├── GET /data/production/summary_2025.json          (backtest metrics)
    ├── GET /api/production/status                      (approval state + gates)
    └── GET /data/production/trades/{strategy_id}_2025.json  (trade details)
         |
         v
    Renders:
    ├── 5 KPI cards (return, Sharpe, WR, MaxDD, p-value)
    ├── p-value significance badge (green if < 0.05)
    ├── 2025 candlestick chart with trade entry/exit markers
    ├── Full trade table (24 rows: confidence, HS%, TP%, exit reason, PnL)
    ├── Interactive replay (startDate="2025-01-01", endDate="2025-12-31")
    └── Gates panel (5/5 passed)
```

### What the Operator Reviews

1. **Return vs Buy-and-Hold**: +20.03% vs -12.29% (32.32 pp alpha)
2. **Statistical Significance**: p=0.0097 (highly significant)
3. **Risk Metrics**: Sharpe 3.516, MaxDD 3.83%
4. **Trade Quality**: 70.8% WR, 24 trades (all SHORT), 0 hard stops
5. **Exit Composition**: 9 take_profit + 15 week_end (no forced exits)
6. **Gate Results**: All 5/5 passed → recommendation = PROMOTE

### State After Stage 4

- Operator has reviewed all 2025 backtest evidence
- Decision pending: Approve or Reject

---

## Stage 5: Human Approval (Vote 2/2)

> **IMPORTANT**: Vote 2 happens on `/dashboard`, NOT on `/production`.
> The `/production` page is read-only — status badge only, no interactive buttons.

On `/dashboard`, the operator clicks **"Aprobar"** (Approve) or **"Rechazar"** (Reject).
The API updates `approval_state.json` accordingly.

For the full two-vote system, gate schema, approval_state.json format, and dashboard
components (ApprovalPanel vs ApprovalStatusCard), see `sdd-approval-spec.md`.

| Outcome | Next Step |
|---------|-----------|
| **APPROVED** | Proceed to Stage 6 (`--phase production`) |
| **REJECTED** | Fix issues, then `--reset-approval` to re-evaluate |

### State After Stage 5

- `approval_state.json` has `status: "APPROVED"` (or `"REJECTED"`)
- `/production` shows read-only status badge

---

## Stage 6: Production Deploy (Retrain with 2025)

### What Happens

After approval, the operator runs the production phase. This retrains models on an expanded window that NOW INCLUDES 2025 data.

```bash
python scripts/train_and_export_smart_simple.py --phase production
```

### Why Retrain?

| Phase | Training Window | Data | Purpose |
|-------|----------------|------|---------|
| Backtest (Stage 2) | 2020 → 2024 | Hold out 2025 for OOS | Validate strategy |
| Production (Stage 6) | 2020 → 2025 | Include 2025 | Best model for 2026 trading |

The production model has +1 year of training data compared to the backtest model. This is the standard ML practice: validate on hold-out, then retrain on all available data for deployment.

### Output

Written to `usdcop-trading-dashboard/public/data/production/`:

| File | Content |
|------|---------|
| `summary.json` | 2026 YTD metrics (updates weekly) |
| `trades/smart_simple_v11.json` | 2026 production trades |
| `equity_curve_2026.png` | Optional equity curve chart |
| `monthly_pnl_2026.png` | Optional monthly PnL chart |
| `trade_distribution_2026.png` | Optional PnL histogram |

### `/production` Page Activation

After Stage 6, `/production` shows:

```
/production loads:
    |
    ├── GET /data/production/summary.json               (2026 YTD metrics)
    ├── GET /api/production/status                      (read-only approval state)
    └── GET /data/production/trades/{strategy_id}.json  (2026 trades)
         |
         v
    Renders:
    ├── 5 KPI cards (2026 YTD: return, Sharpe, WR, MaxDD, trades)
    ├── Full-year candlestick chart with trade markers
    ├── Trade table (2026 trades only)
    ├── Read-only status badge: APPROVED
    └── Optional PNGs (graceful fallback if missing)
```

### State After Stage 6

- `/production` shows 2026 YTD performance
- Models saved to disk, ready for Airflow DAGs
- System is ready for automated weekly cycle (Stage 7)

### Operator Checklist

```
[ ] train_and_export_smart_simple.py --phase production completed
[ ] summary.json exists in public/data/production/
[ ] /production page shows 2026 YTD trades
[ ] Airflow DAGs are configured and unpaused (see Stage 7)
```

---

## Stage 7: Production Monitoring (Automated)

### What Happens

Airflow DAGs run the weekly cycle automatically. No operator action required unless guardrails trigger.

### Weekly Cycle Summary

Two pipelines run in parallel, each with 5 DAGs:

**H5 Weekly** (Smart Simple v1.1): Sun retrain → Mon signal+entry → Mon-Fri TP/HS monitor → Fri close+evaluate
**H1 Daily** (Forecast VT+Trail): Sun retrain → Mon-Fri signal+execute+trail → Mon-Fri paper monitor

For complete DAG schedules, see `sdd-pipeline-lifecycle.md` § Monitoring.
For H5 pipeline architecture and DAG details, see `h5-smart-simple-pipeline.md`.

### Guardrails

| Guardrail | Trigger | Action |
|-----------|---------|--------|
| Circuit breaker | 5 consecutive losses OR 12% drawdown | Pause trading + alert |
| Long insistence | > 60% LONGs in 8-week window | Alert only |
| Rolling DA | SHORT DA < 55% or LONG DA < 45% (16-week window) | Pause direction |

### Promotion Gates (Week 15)

After 15 weeks of paper trading, evaluate for live promotion.
See `h5-smart-simple-pipeline.md` § Promotion Gates for full threshold table.

### State During Stage 7

- Weekly trades appear in DB tables (`forecast_h5_executions`, `forecast_h5_subtrades`)
- `/production` page updated with new trades after each week
- Operator monitors guardrail alerts

---

## Strategy Execution — How Smart Simple v1.1 Makes Money

### The Edge

1. **Directional Forecast**: Ridge + BayesianRidge predict `ln(close[t+5]/close[t])` — the 5-day log return. The ensemble mean provides the signal direction.

2. **SHORT Bias**: The 2026 regime change (p=0.0014) broke LONG profitability (WR 58% → 28%). SHORTs still work (WR 56%). Both pipelines are SHORT-biased.

3. **Vol-Adaptive Stops**: Hard stop = `clamp(vol * sqrt(5) * 2.0, 1%, 3%)`. Wider in volatile markets, tighter in calm. Take profit = 50% of hard stop. This eliminates premature stop-outs (v1.0 had 2 hard stops, v1.1 has 0).

4. **Confidence Filtering**: 3-tier scoring from model agreement + magnitude. LOW-confidence LONGs are skipped entirely (net effect if taken = -0.75%).

### Trade Lifecycle

```
Monday 08:15: Signal generated (e.g., SHORT, confidence=HIGH, HS=2.81%, TP=1.41%)
Monday 09:00: Limit entry order placed at current price (0% maker fee)
    |
    ├── TP hit (bar_low <= entry * (1 - 1.41%)) → close with profit
    ├── HS hit (bar_high >= entry * (1 + 2.81%)) → close with loss
    └── Friday 12:50: Market close remaining position (exit_reason=week_end)
```

### Sizing

- SHORT: flat 1.5x leverage (all confidence tiers)
- LONG HIGH: 1.0x leverage
- LONG MEDIUM: 0.5x leverage
- LONG LOW: **SKIP** (do not trade)

### 2025 Backtest Results

| Metric | Value |
|--------|-------|
| Return | +20.03% ($10K → $12,003) |
| Sharpe | 3.516 |
| p-value | 0.0097 |
| MaxDD | -3.83% |
| Trades | 24 (all SHORT) |
| WR | 70.8% |
| Hard stops | 0 |
| Exits | 9 TP + 15 week_end |

---

## Quick Reference: All Stage Commands

| Stage | Command | Phase |
|-------|---------|-------|
| 0 | `docker-compose up -d` | Bootstrap |
| 1a | `airflow dags trigger core_l0_01_ohlcv_backfill` | Data |
| 1b | `airflow dags trigger core_l0_03_macro_backfill` | Data |
| 2a | `python scripts/generate_weekly_forecasts.py` | Train H1 |
| 2b | `python scripts/train_and_export_smart_simple.py --phase backtest` | Train H5 |
| 3 | (automatic — dashboard reads generated files) | Forecasting |
| 4 | (navigate to `/dashboard` and review) | Review |
| 5 | (click Approve/Reject on `/dashboard`) | Approve |
| 6 | `python scripts/train_and_export_smart_simple.py --phase production` | Deploy |
| 7 | (Airflow DAGs — automated weekly cycle) | Monitor |

### Combined Command (Stages 2b + 6)

```bash
# Run backtest + production in one command:
python scripts/train_and_export_smart_simple.py --phase both
```

### Reset and Re-evaluate

```bash
# Reset approval to PENDING after retraining:
python scripts/train_and_export_smart_simple.py --reset-approval

# Skip PNG generation (faster):
python scripts/train_and_export_smart_simple.py --phase both --no-png
```

---

## Cross-References

| Spec | Focus Area |
|------|------------|
| `sdd-strategy-spec.md` | Universal strategy schemas (TradeRecord, MetricsSummary, ExitReasons) |
| `sdd-approval-spec.md` | Approval lifecycle, gate system, 2-vote flow |
| `sdd-dashboard-integration.md` | JSON/PNG file schemas, page data flows |
| `sdd-pipeline-lifecycle.md` | 8-stage lifecycle quick reference + DAG schedules |
| `h5-smart-simple-pipeline.md` | H5 weekly pipeline architecture + DAGs |
| `l1-l5-inference-pipeline.md` | RL inference pipeline (deprioritized) |
| `l0-data-governance.md` | L0 data layer: OHLCV + macro DAGs |
| `l2-l3-l4-training-pipeline.md` | RL training pipeline (deprioritized) |

---

## DO NOT

- Do NOT skip Stage 1 (data alignment) — models trained on stale data will underperform
- Do NOT run Stage 6 before Stage 5 — production deploy requires human approval
- Do NOT modify `approval_state.json` manually — use `--reset-approval` CLI or dashboard API
- Do NOT retrain manually once Stage 7 is active — Airflow handles weekly retraining
- Do NOT take LONG trades with LOW confidence — net effect is negative (-0.75%)
- Do NOT compare H5 vs H1 returns directly — different horizons, different trade frequency
