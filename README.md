# USDCOP Trading System

A production-grade algorithmic trading system for USD/COP (US Dollar / Colombian Peso) using supervised ML forecasting. The system runs two statistically significant pipelines — **H1 Daily** and **H5 Weekly** — both beating buy-and-hold by 30+ percentage points in 2025 out-of-sample testing.

---

## Key Results (2025 Out-of-Sample)

| Pipeline | Strategy | Return | Sharpe | p-value | $10K &rarr; |
|----------|----------|--------|--------|---------|-------------|
| **H1 Daily** | 9-model ensemble + trailing stop | **+36.84%** | 3.135 | 0.0178 | **$13,684** |
| **H5 Weekly** | Ridge+BR + Smart Simple v1.1 | **+20.03%** | 3.516 | 0.0097 | **$12,003** |
| RL (deprioritized) | PPO V21.5b on 5-min bars | +2.51% | 0.321 | 0.272 | $10,251 |
| Baseline | Buy & Hold USD/COP | -12.29% | -- | -- | $8,771 |

Both pipelines are **SHORT-biased** due to a 2026 regime change (p=0.0014) that broke LONG profitability.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start: Replication on a New Server](#quick-start-replication-on-a-new-server)
- [ML Forecasting vs Reinforcement Learning](#ml-forecasting-vs-reinforcement-learning)
- [SSOT Configuration Files](#ssot-configuration-files)
- [21 Shared Features](#21-shared-features)
- [Pipeline Responsibilities](#pipeline-responsibilities)
- [Backtest-to-Production Promotion](#backtest-to-production-promotion)
- [Experiment Protocol & A/B Testing](#experiment-protocol--ab-testing)
- [Dashboard Pages](#dashboard-pages)
- [Data Sources](#data-sources)
- [Infrastructure](#infrastructure)
- [Database Schema](#database-schema)
- [Guardrails & Promotion Gates](#guardrails--promotion-gates)
- [Project Structure](#project-structure)
- [SDD Specifications](#sdd-specifications)
- [Known Gotchas](#known-gotchas)

---

## Architecture Overview

```
                            USDCOP Trading System
    ============================================================

    DATA LAYER (L0)                    TRAINING (L3)
    5 Airflow DAGs                     Weekly expanding window
    3 FX pairs (COP, MXN, BRL)        2020 -> last Friday
    40 macro variables, 7 sources      21 shared features
         |                                  |
         v                                  v
    +-----------+     +----------+    +------------+    +----------+
    | PostgreSQL| --> | Features | -> |   Models   | -> | Backtest |
    | TimescaleDB     | 21 cols  |    | H1: 9      |    | OOS 2025 |
    | OHLCV+Macro     | anti-leak|    | H5: 2      |    | p < 0.05 |
    +-----------+     +----------+    +------------+    +----------+
                                                              |
                      +-- APPROVAL (2-vote) --+               |
                      | Vote 1: Auto gates    | <-------------+
                      | Vote 2: Human review  |
                      +-----------+-----------+
                                  |
                                  v
                      +--- PRODUCTION ---+       +--- MONITORING ---+
                      | Retrain + deploy | ----> | Weekly Airflow   |
                      | 2020->2025 data  |       | Guardrails       |
                      +-----------------+        | Circuit breaker  |
                                                 +------------------+
```

### Three Tracks (by priority)

1. **H1 Daily Pipeline (PRODUCTION)**: 9 supervised models, H=1 day horizon, ensemble of top 3 by prediction magnitude, trailing stop execution. SHORT-only mode.

2. **H5 Weekly Pipeline (PAPER TRADING)**: 2 linear models (Ridge + BayesianRidge), H=5 day horizon, mean ensemble, Smart Simple v1.1 execution (TP/HS/Friday close). 15-week evaluation period starting 2026-02-16.

3. **RL Pipeline (DEPRIORITIZED)**: PPO agent on 5-min bars. NOT statistically significant (p=0.272, +2.51%). Kept for research only.

---

## Quick Start: Replication on a New Server

### Prerequisites

- Docker + Docker Compose
- Python 3.10+ with dependencies (`pip install -e .`)
- Node.js 18+ (for dashboard)
- Git LFS (seed parquet files are tracked with LFS)

### Stage 0: Bootstrap Infrastructure

```bash
git clone <repo-url>
cd USDCOP-RL-Models

# Pull Git LFS files (seed parquets)
git lfs pull

# Configure secrets (copy template and fill in real values)
cp secrets/secrets.template.txt secrets/
# Create individual secret files:
#   db_password.txt, redis_password.txt, minio_secret_key.txt,
#   airflow_password.txt, airflow_fernet_key.txt, grafana_password.txt

# Copy .env template
cp .env.example .env
# Edit .env with your credentials

# Start all services
docker-compose up -d
```

**What happens on startup**: PostgreSQL + TimescaleDB starts, init scripts run in order:
1. `00-extensions.sql` — Enable TimescaleDB, pg_cron
2. `01-essential-tables.sql` — Core tables (`usdcop_m5_ohlcv`, `macro_indicators_*`)
3. `02-macro-tables.sql` — 4-table macro architecture (daily, monthly, quarterly)
4. `03-views.sql` — Monitoring views (`fx_latest_bar`, `fx_daily_bar_counts`)
5. `04-data-seeding.py` — Detect empty DB, restore from seed parquets

**Restore priority chain** (seeder tries each source in order):
1. Daily backup parquets (`data/backups/seeds/*.parquet`) — freshest, written by automated DAG
2. Git LFS seed files (`seeds/latest/*.parquet`) — available after `git lfs pull`
3. MinIO bucket (`s3://seeds/latest/`) — if Git LFS was not pulled
4. Legacy CSV backups — last resort

After Stage 0, the DB has historical data but is NOT up-to-date (gap from last seed export to today).

### Stage 1: Data Alignment (L0 Backfill)

```bash
# Backfill all 3 FX pairs (OHLCV 5-min bars)
airflow dags trigger core_l0_01_ohlcv_backfill

# Backfill all macro variables (40 vars from 7 sources)
airflow dags trigger core_l0_03_macro_backfill

# Or standalone (no Airflow required):
python scripts/build_unified_fx_seed.py          # Rebuild FX seeds from raw sources
python data/pipeline/04_cleaning/run_clean.py     # Rebuild MACRO_DAILY_CLEAN.parquet
```

**Verify**:
```sql
SELECT MAX(time) FROM usdcop_m5_ohlcv WHERE symbol='USD/COP';   -- should be yesterday
SELECT MAX(fecha) FROM macro_indicators_daily;                    -- should be yesterday
```

Going forward, realtime DAGs keep data fresh automatically:
- `core_l0_02_ohlcv_realtime`: every 5 min during market hours (8:00-12:55 COT)
- `core_l0_04_macro_update`: hourly during market hours
- `core_l0_05_seed_backup`: daily at 13:00 COT (dumps DB to backup parquets)

### Stage 2: Model Training

**A. H1 Multi-Model Training** (9 models x 7 horizons — populates `/forecasting` page):
```bash
python scripts/generate_weekly_forecasts.py
```
Output: `usdcop-trading-dashboard/public/forecasting/bi_dashboard_unified.csv` + 63 backtest PNGs + weekly forward PNGs.

**B. H5 Smart Simple Backtest** (Ridge + BayesianRidge, 2025 OOS — populates `/dashboard`):
```bash
python scripts/train_and_export_smart_simple.py --phase backtest
```
Output: `summary_2025.json`, `approval_state.json` (5 gates evaluated automatically = Vote 1/2), `trades/smart_simple_v11_2025.json`.

### Stage 3: Forecasting Dashboard Activation

No action required. After Stage 2, navigate to `/forecasting` in the dashboard. It reads the generated CSV and PNGs directly. Shows all 9 models, walk-forward metrics, weekly forward predictions.

### Stage 4: Backtest Review

Navigate to `/dashboard`. The `ForecastingBacktestSection` loads:
- `summary_2025.json` — 5 KPI cards (return, Sharpe, WR, MaxDD, p-value)
- `approval_state.json` — 5 gates with pass/fail results
- `trades/smart_simple_v11_2025.json` — 24 trades with candlestick chart + interactive replay

### Stage 5: Human Approval (Vote 2/2)

On `/dashboard`, review the backtest evidence and click **"Aprobar"** (Approve) or **"Rechazar"** (Reject).

> **Important**: Vote 2 happens on `/dashboard`, NOT on `/production`. The `/production` page is read-only.

### Stage 6: Production Deploy

After approval, retrain on the expanded window that now includes 2025 data:

```bash
python scripts/train_and_export_smart_simple.py --phase production
```

This retrains models on 2020-2025 (vs 2020-2024 in backtest), generates 2026 production trades, and exports to the `/production` page. Optional `--seed-db` flag writes to PostgreSQL tables.

### Stage 7: Automated Monitoring

Airflow DAGs handle the weekly cycle automatically:
- **Sunday**: Retrain models on expanding window (adds ~5 rows/week)
- **Monday**: Generate signals, place entries
- **Mon-Fri**: Monitor TP/HS, trailing stops
- **Friday**: Close remaining positions, evaluate weekly performance
- **Guardrails**: Circuit breaker (5 consecutive losses OR 12% DD), rolling DA monitor

### Combined Commands

```bash
# Full pipeline (backtest + production in one command):
python scripts/train_and_export_smart_simple.py --phase both

# Skip PNGs (faster iteration):
python scripts/train_and_export_smart_simple.py --phase both --no-png

# Reset approval (to re-evaluate after retraining):
python scripts/train_and_export_smart_simple.py --reset-approval

# Seed DB tables (for DAG monitoring):
python scripts/train_and_export_smart_simple.py --phase production --seed-db
```

---

## ML Forecasting vs Reinforcement Learning

This project contains two fundamentally different ML approaches. Forecasting is the proven alpha source; RL is deprioritized after extensive experimentation.

### Supervised ML Forecasting (H1 + H5) — ACTIVE

| Aspect | Details |
|--------|---------|
| **Paradigm** | Supervised regression: predict `target = ln(close[t+H] / close[t])` |
| **Data** | Daily OHLCV (~1,500 rows, 2020-2026) + 4 macro variables (DXY, oil, VIX, EMBI) |
| **Models** | H1: 9 models (ridge, bayesian_ridge, ard, xgboost, lightgbm, catboost, 3 hybrids). H5: 2 models (Ridge + BayesianRidge) |
| **Training** | Weekly expanding window: ALL data from 2020-01-01 to last Friday. Retrained every Sunday |
| **Validation** | Walk-forward with 5 expanding windows for experiments; weekly OOS for production |
| **Features** | 21 shared features (4 price + 4 returns + 3 vol + 3 technical + 3 calendar + 4 macro T-1) |
| **Anti-leakage** | Macro T-1 shift: `merge_asof(direction='backward')` + `.shift(1)` |
| **Statistical proof** | H1: p=0.0178, H5: p=0.0097. Both significant and both beat buy-and-hold by 30+ pp |
| **Exchange** | MEXC (0% maker fees, 1 bps slippage estimate) |

### Reinforcement Learning (RL) — DEPRIORITIZED

| Aspect | Details |
|--------|---------|
| **Paradigm** | PPO agent (Stable-Baselines3) learning to trade |
| **Data** | 5-min OHLCV bars, ~70K training bars (2019-2024) |
| **Models** | PPO with MlpPolicy (256x256), mandatory 5 seeds [42, 123, 456, 789, 1337] |
| **Training** | Fixed date splits: train 2019-2024, val 2025-H1, test 2025-H2 |
| **Features** | 18 market features + 2 state features (position, time normalization) |
| **Result** | +2.51% mean across 4/5 seeds, **p=0.272 — NOT significant** |
| **Key insight** | Eval reward does not predict OOS performance (seed 456: eval=131, lost -20.6%) |

**Why forecasting wins**: Supervised models on daily bars capture macro-driven FX movements more effectively. RL on 5-min bars struggles with the low signal-to-noise ratio of intraday FX.

---

## SSOT Configuration Files

Each concern has ONE authoritative configuration file — the **Single Source of Truth** (SSOT). Never duplicate config across files.

| File | Purpose | Used By |
|------|---------|---------|
| `config/execution/smart_simple_v1.yaml` | H5 Smart Simple v1.1: stops, sizing, confidence | H5 pipeline scripts + DAGs |
| `config/execution/smart_executor_v1.yaml` | H1 Smart Executor: trailing stop, SHORT-only | H1 pipeline scripts + DAGs |
| `config/pipeline_ssot.yaml` | RL pipeline: features, environment, PPO hyperparams | RL training + inference |
| `config/macro_variables_ssot.yaml` | L0 macro variable definitions (40 vars, 7 sources) | Macro DAGs |
| `config/forecasting_ssot.yaml` | Forecasting model definitions and horizons | Forecasting engine |

### H5 Smart Simple v1.1 Key Parameters

```yaml
# config/execution/smart_simple_v1.yaml
stops:
  vol_multiplier: 2.0           # HS = clamp(vol * sqrt(5) * 2.0, 1%, 3%)
  tp_ratio: 0.5                 # TP = HS * 0.5
  hard_stop_min_pct: 0.01       # 1% floor
  hard_stop_max_pct: 0.03       # 3% cap

sizing:
  short_multipliers:             # Flat 1.5x for ALL SHORT tiers
    HIGH: 1.5                    # (scorer doesn't discriminate with N=24)
    MEDIUM: 1.5
    LOW: 1.5
  long_multipliers:
    HIGH: 1.0
    MEDIUM: 0.5
    LOW: 0.0                    # SKIP — net effect if taken = -0.75%
```

### H1 Smart Executor Key Parameters

```yaml
# config/execution/smart_executor_v1.yaml
trailing_stop:
  activation_pct: 0.002         # Activate trailing at +0.2% profit
  trail_distance_pct: 0.003     # Trail by 0.3%
  hard_stop_pct: 0.015          # Hard stop at 1.5%
direction_filter: SHORT_ONLY    # 2026 regime: LONG broken (WR 58% -> 28%)
```

### Experiment Configs (Frozen)

Each experiment is a complete, frozen SSOT file in `config/experiments/`. The baseline (`v215b_baseline.yaml`) is NEVER modified. All experiments derive from it by changing ONE variable:

```
config/experiments/
  v215b_baseline.yaml      <- REFERENCE (NEVER modify)
  exp_asym_001.yaml        <- Asymmetric SL/TP (FAILED)
  exp_reward_asym_001.yaml <- Reward asymmetry (FAILED)
  ...
```

---

## 21 Shared Features

Both H1 and H5 pipelines use identical features, computed from daily OHLCV + macro data:

| # | Category | Feature | Source |
|---|----------|---------|--------|
| 1-4 | Price | `close`, `open`, `high`, `low` | Daily OHLCV |
| 5-8 | Returns | `return_1d`, `return_5d`, `return_10d`, `return_20d` | Log returns |
| 9-11 | Volatility | `volatility_5d`, `volatility_10d`, `volatility_20d` | Std of returns |
| 12-14 | Technical | `rsi_14d` (Wilder's EMA), `ma_ratio_20d`, `ma_ratio_50d` | Price-derived |
| 15-17 | Calendar | `day_of_week`, `month`, `is_month_end` | Date |
| 18-21 | Macro (T-1) | `dxy_close_lag1`, `oil_close_lag1`, `vix_close_lag1`, `embi_close_lag1` | Macro DB |

### Anti-Leakage Mechanisms

| Mechanism | Implementation | Purpose |
|-----------|---------------|---------|
| Macro T-1 shift | `df_macro[col].shift(1)` | Use yesterday's macro, not today's |
| Backward merge | `merge_asof(direction='backward')` | Never forward-fill future data |
| Expanding window | Train on `date >= 2020-01-01` to last Friday | No future data in training set |
| Target shift | `df["close"].shift(-H)` | Predict H days ahead (labels aligned) |
| Train-only normalization | `StandardScaler.fit(X_train)` | Scaler fitted on training data only |

---

## Pipeline Responsibilities

### L0: Data Layer (5 DAGs)

Maintains OHLCV and macro data in PostgreSQL. All OHLCV timestamps are in `America/Bogota` timezone.

| DAG | Schedule (COT) | Purpose |
|-----|----------------|---------|
| `core_l0_01_ohlcv_backfill` | Manual | Gap-fill OHLCV for 3 FX pairs (COP, MXN, BRL) + export seeds |
| `core_l0_02_ohlcv_realtime` | `*/5 8:00-12:55 Mon-Fri` | Realtime 5-min bars with circuit breaker per pair |
| `core_l0_03_macro_backfill` | `Sun 06:00 UTC` / Manual | Full macro extraction from 7 sources, export 9 MASTER files |
| `core_l0_04_macro_update` | `hourly 8:00-12:00 Mon-Fri` | Incremental update of 40 macro variables |
| `core_l0_05_seed_backup` | `daily 13:00 COT` | Atomic DB-to-parquet dump for startup restore |

**Key rules**:
- ALL OHLCV timestamps = `America/Bogota` timezone (session 8:00-12:55 COT, Mon-Fri)
- BRL fetched with `timezone=UTC` from TwelveData (API quirk), then converted to COT
- Daily backup parquets are the primary restore source on `docker-compose up -d`
- OHLCV table `usdcop_m5_ohlcv` has composite PK `(time, symbol)` supporting 3 FX pairs

### H1 Daily Pipeline (5 DAGs)

9-model ensemble with trailing stop execution, SHORT-only mode.

| Day | Time (COT) | DAG | Action |
|-----|-----------|-----|--------|
| Sun | 01:00 | H1-L3 Training | Retrain 9 models on expanding window (2020 -> last Friday) |
| Mon-Fri | 13:00 | H1-L5 Inference | Ensemble top_3 by prediction magnitude, generate daily signal |
| Mon-Fri | 13:30 | H1-L5 Vol-Targeting | Position sizing based on realized vol + trailing stop params |
| Mon-Fri | 13:35 | H1-L7 Smart Executor | Place SHORT order + activate trailing stop |
| Mon-Fri | 19:00 | H1-L6 Paper Monitor | Log paper trading PnL |

**Execution**: Trailing stop activates at +0.2% profit, trails by 0.3%, hard stop at 1.5%. All trades are SHORT due to 2026 regime change.

### H5 Weekly Pipeline (6 DAGs)

Ridge + BayesianRidge with Smart Simple v1.1 execution (TP/HS/Friday close).

| Day | Time (COT) | DAG | Action |
|-----|-----------|-----|--------|
| Sun | 01:30 | H5-L3 Training | Retrain Ridge + BayesianRidge on expanding window |
| Mon | 08:15 | H5-L5 Signal | Ensemble mean prediction + 3-tier confidence scoring |
| Mon | 08:45 | H5-L5 Vol-Target | Compute adaptive stops (vol-based) + asymmetric sizing |
| Mon-Fri | */30 9-13 | H5-L7 Executor | Monitor for TP/HS hits + Friday market close |
| Fri | 14:30 | H5-L6 Monitor | Weekly evaluation: DA, Sharpe, MaxDD + guardrail checks |
| Manual | -- | H5-L4 Backtest | OOS walk-forward backtest + SDD dashboard export |

**Trade lifecycle**:
```
Monday 08:15: Signal generated (e.g., SHORT, HIGH confidence, HS=2.81%, TP=1.41%)
Monday 09:00: Limit entry order placed (0% maker fee on MEXC)
Mon-Fri */30:  Monitor for TP hit (close with profit) or HS hit (close with loss)
Friday 12:50:  Market close any remaining position (exit_reason = week_end)
```

**Confidence scoring**: Ridge-BR agreement + magnitude -> HIGH/MEDIUM/LOW. LOW LONGs are skipped entirely (net effect if taken = -0.75%). All SHORTs use flat 1.5x leverage.

### RL Pipeline (6 DAGs, deprioritized)

| DAG | Schedule | Purpose |
|-----|----------|---------|
| L1 Feature Refresh | `*/5 8:00-12:55 Mon-Fri` | Compute 18 features -> `inference_ready_nrt` table |
| L1 Model Promotion | Manual | Populate historical features on model approval |
| L5 Multi-Model | Triggered by L1 | Read pre-normalized features + predict |
| L3 Training | Manual | PPO multi-seed training (5 seeds required) |
| L4 Backtest | Manual | OOS validation + promotion gates |
| L4 Experiment | Manual | A/B experiment runner |

---

## Backtest-to-Production Promotion

The system uses **Spec-Driven Development (SDD)** with a 3-layer architecture:

```
Layer 1: SPEC (defines what)      -> .claude/rules/sdd-*.md           (5 specs)
Layer 2: CONTRACT (enforces how)  -> lib/contracts/ + src/contracts/   (TS + Python types)
Layer 3: IMPLEMENTATION           -> scripts/, pages, DAGs            (conform to contracts)
```

### The 2-Vote Approval System

Every strategy requires two independent approvals before production deployment:

```
  BACKTEST COMPLETE (--phase backtest)
        |
        v
  Vote 1/2 (Automatic)
  Python evaluates 5 gates:
  +-------------------------------+--------+----------------+
  | Gate                          | Thresh | Smart Simple   |
  +-------------------------------+--------+----------------+
  | min_return_pct > -15%         | -15%   | +20.03% PASS   |
  | min_sharpe_ratio > 0.0        | 0.0    | 3.516   PASS   |
  | max_drawdown_pct < 20%        | 20%    | 3.83%   PASS   |
  | min_trades >= 10              | 10     | 24      PASS   |
  | statistical_significance p<5% | 0.05   | 0.0097  PASS   |
  +-------------------------------+--------+----------------+
        |
        v
  approval_state.json: status = PENDING_APPROVAL
        |
        v
  Vote 2/2 (Human on /dashboard)
  Operator reviews:
  - 5 KPI cards (return, Sharpe, WR, MaxDD, p-value)
  - 2025 candlestick chart with trade entry/exit markers
  - Full trade table (24 rows with confidence, HS%, TP%, exit reason, PnL)
  - Interactive backtest replay
  - Gate results panel with recommendation badge
        |
    +---+---+
    |       |
  APPROVE  REJECT
    |       |
    v       v
  Stage 6  Fix issues, then
  Deploy   --reset-approval
```

### Post-Approval: Automatic Deploy

When the operator clicks Approve, the deploy API reads a `deploy_manifest` embedded in `approval_state.json` and automatically spawns the production pipeline. The manifest specifies which script to run and which DB tables to seed — making the deploy strategy-agnostic.

```json
{
  "deploy_manifest": {
    "pipeline_type": "ml_forecasting",
    "script": "scripts/train_and_export_smart_simple.py",
    "args": ["--phase", "production", "--no-png", "--seed-db"],
    "db_tables": ["forecast_h5_predictions", "forecast_h5_signals",
                  "forecast_h5_executions", "forecast_h5_subtrades",
                  "forecast_h5_paper_trading"]
  }
}
```

### Why Retrain for Production?

| Phase | Training Window | Purpose |
|-------|----------------|---------|
| Backtest (Stage 2) | 2020 -> 2024 | Hold out 2025 for OOS validation |
| Production (Stage 6) | 2020 -> 2025 | Use ALL available data for best 2026 model |

Standard ML practice: validate on hold-out, then retrain on everything for deployment.

---

## Experiment Protocol & A/B Testing

### Core Rules

1. **ONE variable per experiment** — never change model type AND feature set in the same run
2. **5 seeds for RL** — [42, 123, 456, 789, 1337], no exceptions
3. **Statistical validation before declaring success** — p < 0.05, walk-forward DA > 55%, Sharpe > 1.0
4. **Always compare baselines** — buy-and-hold (-12.29% in 2025), previous best, random agent (RL)
5. **Record everything** — `.claude/experiments/EXPERIMENT_LOG.md` (append-only)
6. **Eval reward != OOS performance** (RL only) — never select "best model" by eval reward alone

### Experiment Lifecycle

```
1. Define hypothesis in EXPERIMENT_QUEUE.md
2. Create frozen SSOT config: config/experiments/{id}.yaml (complete copy, not a diff)
3. Run: python scripts/run_ssot_pipeline.py --config config/experiments/{id}.yaml --multi-seed
4. Report per-seed table + aggregate metrics + bootstrap 95% CI
5. Compare vs baseline, log results in EXPERIMENT_LOG.md
6. If significant and improvement: promote config as new pipeline_ssot.yaml
```

### Experiment History

| ID | Variable Changed | Result | Status |
|----|-----------------|--------|--------|
| V21.5b | RL Baseline | +2.51%, p=0.272 | Best RL (not significant) |
| EXP-ASYM-001 | SL/TP ratio | -8.67%, 0/5 seeds | FAILED |
| EXP-HOURLY-001 | 1H bars | -17.49% | FAILED |
| EXP-DAILY-001 | Daily bars | -8.83% | FAILED |
| Smart Simple v1.0 | Weekly H5 | +13.75%, p=0.032 | Superseded by v1.1 |
| **Smart Simple v1.1** | HS 2.0x + flat SHORT | **+20.03%, p=0.0097** | **Active (paper trading)** |

---

## Dashboard Pages

The dashboard is a Next.js 15 application (Tailwind CSS + shadcn/ui + Lightweight Charts).

### `/forecasting` — Model Zoo

Displays all 9 models across 7 horizons (1, 5, 10, 15, 20, 25, 30 days). Walk-forward validated metrics with backtest PNGs and weekly forward prediction charts.

| Data Source | File | Content |
|-------------|------|---------|
| Metrics CSV | `public/forecasting/bi_dashboard_unified.csv` | 505+ rows: all model metrics |
| Backtest PNGs | `backtest_{model}_h{horizon}.png` | 63 files (9 models x 7 horizons) |
| Forward PNGs | `forward_{model}_{week}.png` | Weekly predictions per model |
| Consensus | `forward_consensus_{week}.png` | All-model consensus direction |

**Activates after**: `python scripts/generate_weekly_forecasts.py` (Stage 2A)

### `/dashboard` — Backtest Review + Approval

Shows 2025 OOS results with interactive approval panel (Vote 2/2).

| Data Source | File |
|-------------|------|
| Backtest metrics | `public/data/production/summary_2025.json` |
| Gate results | `public/data/production/approval_state.json` |
| Trade details | `public/data/production/trades/{strategy_id}_2025.json` |

Components: 5 KPI cards, p-value significance badge, candlestick chart with trade markers, full trade table, interactive replay, Approve/Reject panel.

**Activates after**: `python scripts/train_and_export_smart_simple.py --phase backtest` (Stage 2B)

### `/production` — Live Monitoring (Read-Only)

Shows 2026 YTD performance. No interactive approval buttons — status badge only.

| Data Source | File |
|-------------|------|
| Production metrics | `public/data/production/summary.json` |
| Approval status | `public/data/production/approval_state.json` (read-only badge) |
| Trade details | `public/data/production/trades/{strategy_id}.json` |
| Optional PNGs | `equity_curve_2026.png`, `monthly_pnl_2026.png` (graceful fallback) |

**Activates after**: `python scripts/train_and_export_smart_simple.py --phase production` (Stage 6)

### JSON Safety

All Python-to-Dashboard JSON exports MUST use `safe_json_dump()` from `src/contracts/strategy_schema.py`. This converts `Infinity` and `NaN` to `null` — JavaScript's `JSON.parse()` crashes on Python's `float("inf")`.

---

## Data Sources

### OHLCV (3 FX Pairs)

| Pair | Source | Seed File | Approx Rows |
|------|--------|-----------|-------------|
| USD/COP | TwelveData | `seeds/latest/usdcop_m5_ohlcv.parquet` | ~81K (5-min) |
| USD/MXN | Dukascopy | `seeds/latest/usdmxn_m5_ohlcv.parquet` | ~95K (5-min) |
| USD/BRL | TwelveData | `seeds/latest/usdbrl_m5_ohlcv.parquet` | ~90K (5-min) |
| ALL | Unified | `seeds/latest/fx_multi_m5_ohlcv.parquet` | ~266K (5-min) |
| Daily COP | TwelveData | `seeds/latest/usdcop_daily_ohlcv.parquet` | ~3K (daily, for H1/H5) |

All timestamps in `America/Bogota` timezone. Session: 8:00-12:55 COT, Monday-Friday.

### Macro (40 Variables, 7 Sources)

| Source | Example Variables | API/Method |
|--------|-------------------|------------|
| FRED | FEDFUNDS, CPI, unemployment, UST10Y, UST2Y | FRED API |
| Investing.com | DXY, VIX, commodities (oil, Brent) | Scraping |
| BanRep | IBR, TPM, EMBI_COL | BanRep API |
| BCRP | Peru macro data | BCRP API |
| Fedesarrollo | Colombia consumer confidence | Scraping |
| DANE | Colombia trade balance | DANE API |
| BanRep BOP | Balance of payments | BanRep API |

Clean macro output: `data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet` (17 columns, ~10K rows, 2015-2026).

---

## Infrastructure

### Docker Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| PostgreSQL | timescale/timescaledb:pg15 | 5432 | Primary data store (OHLCV + macro + H5 tables) |
| Redis | redis:7-alpine | 6379 | Caching + circuit breaker state |
| MinIO | minio/minio | 9000/9001 | Object storage (models, artifacts, seeds) |
| Airflow | apache/airflow | 8080 | DAG orchestration (20 DAGs total) |
| Dashboard | Next.js 15 | 3000 | Trading dashboard (4 pages) |

### Volume Mounts

| Host Path | Container Path | Service | Mode |
|-----------|---------------|---------|------|
| `./data` | `/opt/airflow/data` | Airflow scheduler | RW |
| `./data` | `/app/data` | Data seeder | RO |
| `./seeds` | `/app/seeds` | Data seeder | RO |
| `./airflow/dags` | `/opt/airflow/dags` | Airflow scheduler/worker | RW |
| `./init-scripts` | `/docker-entrypoint-initdb.d` | PostgreSQL | RO |

### Backup System

Two-tier automated backup ensures no data is lost on container restart:

1. **Daily backup DAG** (`core_l0_05_seed_backup`): Runs at 13:00 COT after market close. Dumps complete OHLCV + macro tables to `data/backups/seeds/*.parquet` with atomic writes (`.tmp` -> rename) and SHA256 manifest (`backup_manifest.json`).

2. **Seed export on backfill**: OHLCV backfill DAG exports updated seeds to `seeds/latest/`. Macro backfill exports 9 MASTER files (3 frequencies x 3 formats). These are committed to Git as the baseline restore point.

**Restore priority on `docker-compose up -d`**: Daily backup parquet (freshest) -> Git LFS seed -> MinIO -> legacy CSV.

---

## Database Schema

### Core Tables

| Table | Purpose | Primary Key |
|-------|---------|-------------|
| `usdcop_m5_ohlcv` | 5-min OHLCV for 3 FX pairs | `(time, symbol)` composite |
| `macro_indicators_daily` | Daily macro variables (18 cols) | `fecha` |
| `macro_indicators_monthly` | Monthly macro variables (8 cols) | `fecha` |
| `macro_indicators_quarterly` | Quarterly macro variables (4 cols) | `fecha` |
| `inference_ready_nrt` | Pre-normalized features for RL inference | `timestamp` |

### H5 Pipeline Tables (migration 043 + 044)

| Table | Purpose |
|-------|---------|
| `forecast_h5_predictions` | Ridge + BR predictions per week |
| `forecast_h5_signals` | Ensemble signal + confidence tier + stops |
| `forecast_h5_executions` | Weekly execution tracking (entry, exit, PnL) |
| `forecast_h5_subtrades` | Individual subtrade records within a week |
| `forecast_h5_paper_trading` | Weekly evaluation: DA, Sharpe, MaxDD, gate status |

---

## Guardrails & Promotion Gates

### Active Guardrails (Automated)

| Guardrail | Trigger Condition | Action |
|-----------|-------------------|--------|
| Circuit breaker | 5 consecutive losses OR 12% drawdown | Pause trading + alert |
| Long insistence | > 60% LONGs in 8-week window | Alert only (informational) |
| Rolling DA (SHORT) | SHORT DA < 55% in 16-week window | Pause SHORTs |
| Rolling DA (LONG) | LONG DA < 45% in 16-week window | Pause LONGs |

### H5 Promotion Gates (Evaluated at Week 15)

| Gate | Threshold | Action if PASS | Action if FAIL |
|------|-----------|----------------|----------------|
| DA overall > 55% AND DA SHORT > 60% | Promote to live trading | Check keep conditions |
| DA overall < 50% | -- | Discard strategy |
| SHORT DA > 60% but LONG DA < 45% | -- | Switch to SHORT-only |

---

## Project Structure

```
USDCOP-RL-Models/
├── .claude/                              # Claude Code project config
│   ├── rules/                            # SDD specs + pipeline rules (11 files)
│   └── experiments/                      # Queue, log, playbook
├── airflow/dags/                         # 20 Airflow DAGs
│   ├── contracts/                        # DAG registry, XCom contracts
│   ├── extractors/                       # Macro data extractors (7 sources)
│   ├── sensors/                          # PostgreSQL notify sensor
│   ├── services/                         # Upsert, DLQ, backtest factory
│   ├── utils/                            # Circuit breaker, retry policy
│   ├── validators/                       # Data quality validators
│   ├── l0_*.py                           # L0 data DAGs (5)
│   ├── l1_*.py                           # L1 RL feature DAGs (2)
│   ├── l3_*.py, l4_*.py, l5_*.py         # RL pipeline DAGs (4)
│   ├── forecast_h1_*.py                  # H1 daily DAGs (6)
│   └── forecast_h5_*.py                  # H5 weekly DAGs (6)
├── config/
│   ├── execution/                        # H1 + H5 execution configs (SSOT)
│   ├── experiments/                      # Frozen RL experiment configs
│   ├── forecast_experiments/             # Frozen forecasting configs
│   ├── pipeline_ssot.yaml                # Active RL SSOT
│   └── macro_variables_ssot.yaml         # L0 macro definitions (40 vars)
├── data/
│   ├── backups/seeds/                    # Daily parquet backups (automated)
│   ├── pipeline/04_cleaning/output/      # MACRO_DAILY_CLEAN.parquet
│   └── forecasting/                      # Historical forecast signals
├── database/migrations/                  # SQL migrations (044+)
├── init-scripts/                         # Docker startup scripts (00-04)
├── scripts/                              # Pipeline scripts, backtests, diagnostics
├── seeds/latest/                         # Git LFS seed parquets (5 files)
├── src/
│   ├── config/                           # Config loaders
│   ├── contracts/                        # SDD contracts (strategy_schema.py)
│   ├── core/contracts/                   # Feature contract, production contract
│   ├── data/                             # Data loaders, SSOT dataset builder
│   ├── execution/                        # Trailing stop, smart executor
│   ├── features/                         # Calculator registry, technical indicators
│   ├── forecasting/                      # Engine, confidence scorer, adaptive stops, vol targeting
│   ├── inference/                        # Ensemble predictor
│   ├── training/                         # RL training (PPO, environments, rewards)
│   └── services/                         # Deprecated NRT services
├── tests/
│   ├── unit/                             # Unit tests (confidence scorer, adaptive stops, etc.)
│   ├── integration/                      # Integration tests
│   └── chaos/                            # Chaos/resilience tests
├── usdcop-trading-dashboard/             # Next.js 15 dashboard
│   ├── app/                              # Pages: hub, dashboard, production, forecasting
│   ├── components/                       # React components (charts, trading, forecasting, production)
│   ├── lib/contracts/                    # TypeScript contracts (strategy, approval, backtest)
│   ├── lib/services/                     # Data services
│   └── public/data/production/           # JSON/PNG export directory
├── docker-compose.yml                    # Infrastructure (PostgreSQL, Redis, MinIO, Airflow)
├── CLAUDE.md                             # Claude Code project instructions (auto-loaded)
└── README.md                             # This file
```

---

## SDD Specifications

The project follows **Spec-Driven Development**. All architectural decisions are documented in `.claude/rules/`:

| Spec | Purpose |
|------|---------|
| `sdd-mlops-lifecycle.md` | **Master lifecycle**: 8 stages from bootstrap to production |
| `sdd-strategy-spec.md` | Universal strategy interface (TradeRecord, MetricsSummary, ExitReasons) |
| `sdd-approval-spec.md` | 2-vote approval system, gate definitions, approval_state.json schema |
| `sdd-dashboard-integration.md` | JSON/CSV/PNG data contracts, page data flows |
| `sdd-pipeline-lifecycle.md` | Quick reference: stage table, CLI commands, DAG schedules |
| `h5-smart-simple-pipeline.md` | H5 weekly pipeline: architecture, DAGs, Smart Simple v1.1 |
| `l0-data-governance.md` | L0 data layer: 5 DAGs, OHLCV timezone rules, macro pipeline |
| `l1-l5-inference-pipeline.md` | RL inference: feature computation, model promotion |
| `l2-l3-l4-training-pipeline.md` | RL training: dataset build, PPO training, backtest validation |
| `experiment-protocol.md` | Experiment rules: one variable, 5 seeds, statistical validation |
| `ssot-versioning.md` | Config versioning: frozen SSOT per experiment |

### SDD Contracts (Code)

| Contract | Language | Location |
|----------|----------|----------|
| `strategy.contract.ts` | TypeScript | `usdcop-trading-dashboard/lib/contracts/` |
| `production-approval.contract.ts` | TypeScript | `usdcop-trading-dashboard/lib/contracts/` |
| `strategy_schema.py` | Python | `src/contracts/` |

---

## Known Gotchas

- **sklearn 1.6.1**: Use `max_iter` not `n_iter` for BayesianRidge/ARD
- **CatBoost params**: `iterations`/`depth` not `n_estimators`/`max_depth`
- **Hybrid models**: Filter `alpha`/`linear_alpha` keys before passing to boosting fit
- **BRL API**: TwelveData needs `timezone=UTC` for USD/BRL (Bogota returns incomplete data)
- **OHLCV timezone**: Always `America/Bogota`, session 8:00-12:55 COT, Mon-Fri
- **Macro anti-leakage**: Always `merge_asof(direction='backward')` + `.shift(1)` for T-1
- **JSON safety**: NEVER `float("inf")` in JSON — use `None` + `safe_json_dump()`
- **Feature count**: 21 features (not 19) — 4 price + 4 returns + 3 vol + 3 tech + 3 cal + 4 macro
- **RL CPU > GPU**: PPO MlpPolicy is faster on CPU (~300 FPS vs 65-187 GPU on RTX 3050)
- **Smart Simple**: Always read params from YAML config, never hardcode in scripts
- **RSI**: Use Wilder's EMA (`alpha=1/period`), NOT pandas `ewm()` default
- **Composite PK**: `usdcop_m5_ohlcv` uses `(time, symbol)` — multi-pair UPSERT needs both columns

---

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Specific test modules
pytest tests/unit/test_confidence_scorer.py -v
pytest tests/unit/test_adaptive_stops.py -v
pytest tests/unit/test_all_layer_contracts.py -v

# Integration tests
pytest tests/integration/ -v

# Dashboard E2E (Playwright)
cd usdcop-trading-dashboard
npx playwright test
```

---

*Last updated: February 2026*
