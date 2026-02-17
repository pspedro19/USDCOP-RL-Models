# CLAUDE.md — USDCOP Trading System

> Loaded automatically by Claude Code at the start of every session.
> Contains project context, architecture, contracts, and rules that MUST be followed.

---

## PROJECT IDENTITY

**Goal**: Build a profitable USDCOP trading system using supervised forecasting.
**Exchange**: MEXC (0% maker fees, 1 bps slippage estimate)
**Architecture**: Spec-Driven Development (SDD) — specs define contracts, contracts enforce code.

**Three Tracks** (in priority order):

1. **H1 Daily Pipeline (PRODUCTION)**: 9 models, H=1 horizon, trailing stop execution
   - +36.84%, Sharpe 3.135, **p=0.0178** (2025 backtest)
   - DAGs: H1-L3/L4/L5/L6/L7 (see `.claude/rules/h5-smart-simple-pipeline.md`)

2. **H5 Weekly Pipeline (PAPER TRADING)**: Ridge+BR, H=5 horizon, Smart Simple v1.1
   - +20.03%, Sharpe 3.516, **p=0.0097** (2025 backtest)
   - +7.12% (2026 YTD, 3/3 wins)
   - DAGs: H5-L3/L4/L5/L6/L7 (see `.claude/rules/h5-smart-simple-pipeline.md`)

3. **RL (DEPRIORITIZED)**: PPO agent, 5-min bars, NOT significant (p=0.272)

**Strategic Pivot (2026-02-15)**: Forecasting pipeline is the proven alpha source.
**2026 Regime Change** (p=0.0014): LONG broken (WR 58%->28%), SHORT holds (56%). Both pipelines are SHORT-biased.

**Current Best**:
- H1 Daily: Forecast + Vol-Target + Trailing Stop -> $10K -> $13,684 (2025)
- H5 Weekly: Smart Simple v1.1 -> $10K -> $12,003 (2025), $10K -> $10,712 (2026 YTD)
- RL: V21.5b — +2.51% mean (4/5 seeds), NOT significant

---

## QUICK START (First-Time Setup)

> See `.claude/rules/sdd-mlops-lifecycle.md` for the full operator guide with checklists.

```
1. docker-compose up -d                           → DB seeded from parquets (Stage 0)
2. airflow dags trigger core_l0_01_ohlcv_backfill  → OHLCV current (Stage 1)
3. airflow dags trigger core_l0_03_macro_backfill  → Macro current (Stage 1)
4. python scripts/generate_weekly_forecasts.py     → /forecasting page activates (Stage 2)
5. python scripts/train_and_export_smart_simple.py --phase backtest  → /dashboard shows 2025 (Stage 2)
6. Review & approve on /dashboard                  → Vote 2/2 (Stage 5)
7. python scripts/train_and_export_smart_simple.py --phase production → /production shows 2026 (Stage 6)
8. Airflow DAGs take over weekly cycle (Stage 7)   → L3→L5→L7→L6
```

---

## DESIGN PATTERNS & BEST PRACTICES

| Pattern | Where Applied | Rule |
|---------|---------------|------|
| **SSOT** (Single Source of Truth) | Config YAMLs, feature contracts | ONE authoritative source per concern; never duplicate config |
| **SDD** (Spec-Driven Development) | Dashboard integration, approval flow | Specs (`.claude/rules/sdd-*.md`) -> Contracts (TS/Python types) -> Implementation |
| **Contract-First** | All layer boundaries (L0-L7) | Define types BEFORE writing implementation; contracts are the API |
| **DRY** | Feature engineering (21 features shared H1/H5) | Same feature code for training and inference; never duplicate |
| **Expanding Window** | Weekly retraining (H1/H5) | Train on ALL history from 2020-01-01 to last Friday; no fixed splits |
| **Anti-Leakage** | Macro T-1 shift, train-only norm_stats | `merge_asof(direction='backward')` + `.shift(1)` for all macro features |
| **Graceful Degradation** | Dashboard PNGs, Feast fallback | `onError` hides missing images; `CanonicalFeatureBuilder` replaces Feast if down |
| **JSON Safety** | All Python->Dashboard exports | `safe_json_dump()` converts Infinity/NaN to null; NEVER produce invalid JSON |
| **Immutable Config** | Experiment SSOT files | Config is FROZEN once training starts; no mid-experiment changes |

---

## SDD ARCHITECTURE (Spec-Driven Development)

### 3-Layer Stack

```
Layer 1: SPEC (defines what)      -> .claude/rules/sdd-*.md           (5 specs)
Layer 2: CONTRACT (enforces how)  -> lib/contracts/ + src/contracts/   (2 contracts)
Layer 3: IMPLEMENTATION           -> scripts/, pages, DAGs             (conform to contracts)
```

### SDD Specs (`.claude/rules/`)

| Spec | Purpose | Key Rules |
|------|---------|-----------|
| `sdd-mlops-lifecycle.md` | **Master lifecycle (bootstrap→production)** | 8 stages, operator guide, strategy execution details |
| `sdd-strategy-spec.md` | Universal strategy interface | `strategy_id` lookup; TradeRecord, MetricsSummary, StrategyRegistry |
| `sdd-approval-spec.md` | Approval lifecycle + gates | 2-vote system; Vote 1=auto, Vote 2=human on `/dashboard`; 5 gates |
| `sdd-dashboard-integration.md` | Python->Dashboard data contract | JSON schemas, CSV contract, PNG conventions, strategy selector |
| `sdd-pipeline-lifecycle.md` | 8-stage pipeline lifecycle | BOOTSTRAP -> DATA -> TRAIN -> FORECAST -> BACKTEST -> APPROVE -> DEPLOY -> MONITOR |

### SDD Contracts

| Contract | Language | Location | Exports |
|----------|----------|----------|---------|
| `strategy.contract.ts` | TypeScript | `usdcop-trading-dashboard/lib/contracts/` | `StrategyTrade`, `StrategyStats`, `StrategySummary`, `EXIT_REASON_COLORS`, `getExitReasonColor()` |
| `strategy_schema.py` | Python | `src/contracts/` | `StrategyTrade`, `StrategyStats`, `safe_json_dump()`, `EXIT_REASONS` |
| `production-approval.contract.ts` | TypeScript | `usdcop-trading-dashboard/lib/contracts/` | `ApprovalState`, `GateResult`, `ApproveRequest` (re-exports from strategy.contract) |

### Approval Workflow (2-Vote System)

```
Vote 1 (Automatic):  Python export script evaluates 5 gates -> writes approval_state.json
                      Gates: min_return, min_sharpe, max_drawdown, min_trades, p<0.05

Vote 2 (Human):      User reviews gates on /dashboard page -> clicks Approve/Reject
                      API: POST /api/production/approve -> updates approval_state.json
                      /production is READ-ONLY (status badge only, no interactive buttons)
```

### Dashboard File Layout

```
usdcop-trading-dashboard/public/data/production/
├── summary.json                          <- Production metrics (current year, strategy-agnostic)
├── summary_2025.json                     <- Backtest metrics (OOS year)
├── approval_state.json                   <- Gate results + approval status
├── trades/
│   ├── {strategy_id}.json                <- Production trades
│   └── {strategy_id}_{year}.json         <- Backtest trades
├── equity_curve_{year}.png               <- Optional (onError hides)
├── monthly_pnl_{year}.png                <- Optional
└── trade_distribution_{year}.png         <- Optional
```

**Dashboard lookup**: `summary.strategies[summary.strategy_id]` — dynamic key, never hardcoded.

### Pipeline CLI Convention

```bash
python scripts/train_and_export_smart_simple.py --phase backtest    # Stages 1-3 (train+backtest+export)
python scripts/train_and_export_smart_simple.py --phase production   # Stage 5 (deploy)
python scripts/train_and_export_smart_simple.py --phase both         # Stages 1-3 + 5
python scripts/train_and_export_smart_simple.py --reset-approval     # Reset to PENDING
python scripts/train_and_export_smart_simple.py --no-png             # Skip PNG generation
```

---

## ARCHITECTURE (Do NOT deviate)

### Configuration (SSOT = Single Source of Truth)
```
config/pipeline_ssot.yaml                <- ACTIVE RL config (copy of current experiment)
config/macro_variables_ssot.yaml         <- L0: All 40 macro variable definitions + scrapers
config/execution/smart_simple_v1.yaml    <- H5 Smart Simple SSOT (stops, sizing, confidence)
config/execution/smart_executor_v1.yaml  <- H1 Smart Executor SSOT (trailing stop, SHORT-only)
config/experiments/                      <- Frozen RL SSOT configs, one per experiment
  ├── v215b_baseline.yaml                <- REFERENCE baseline (NEVER modify)
  ├── exp_asym_001.yaml                  <- EXP-ASYM-001: Asymmetric SL/TP (FAILED)
  └── ...                                <- Future experiments
config/forecast_experiments/             <- Frozen forecasting configs
  ├── baseline_v1.yaml                   <- Forecasting baseline (DA=53%, PF=1.19)
  └── ...
```

### Code — SDD Contracts (universal, strategy-agnostic)
```
src/contracts/__init__.py                <- Package exports
src/contracts/strategy_schema.py         <- Python: StrategyTrade, StrategyStats, safe_json_dump()
usdcop-trading-dashboard/lib/contracts/strategy.contract.ts    <- TS: universal types + EXIT_REASON_COLORS
usdcop-trading-dashboard/lib/contracts/production-approval.contract.ts <- TS: ApprovalState, GateResult
```

### Code — Forecasting Pipeline (supervised, daily)
```
src/forecasting/engine.py                <- ForecastingEngine: train() + predict()
src/forecasting/config.py                <- ForecastingConfig dataclass (SSOT)
src/forecasting/contracts.py             <- HORIZONS, MODEL_DEFINITIONS, HORIZON_CONFIGS
src/forecasting/data_contracts.py        <- FEATURE_COLUMNS (21), TARGET_HORIZONS (7)
src/forecasting/models/                  <- 9 models: ridge, bayesian_ridge, ard, xgboost, lightgbm, catboost, hybrids
src/forecasting/evaluation/walk_forward.py <- WalkForwardValidator (5-fold expanding window)
src/forecasting/evaluation/backtest.py   <- BacktestEngine (train/test split)
src/forecasting/evaluation/metrics.py    <- DA, Sharpe, RMSE, PF, MaxDD
src/forecasting/experiment_manager.py    <- A/B experiment tracking
src/forecasting/ab_statistics.py         <- Statistical significance tests
scripts/run_forecast_experiment.py       <- Experiment runner
scripts/run_forecasting_pipeline_e2e.py  <- End-to-end pipeline
scripts/build_forecasting_dataset_aligned.py <- Dataset builder
```

### Code — H5 Weekly Pipeline (Smart Simple v1.1)
```
config/execution/smart_simple_v1.yaml    <- SSOT config: HS 2.0x, flat SHORT 1.5x, TP ratio 0.5
src/forecasting/confidence_scorer.py     <- 3-tier confidence (HIGH/MEDIUM/LOW), skip LOW LONGs
src/forecasting/adaptive_stops.py        <- Vol-adaptive TP/HS: HS=clamp(vol*sqrt(5)*2.0, 1%, 3%)
src/forecasting/vol_targeting.py         <- Vol-targeting + asymmetric sizing
scripts/backtest_smart_simple_v1.py      <- Comparative backtest (reads config from YAML)
scripts/diagnose_smart_simple_v1.py      <- 10-section root cause diagnostic
scripts/train_and_export_smart_simple.py <- SDD export: JSON + PNGs + approval gates
database/migrations/043_forecast_h5_tables.sql   <- 5 tables + 2 views
database/migrations/044_smart_simple_columns.sql <- Confidence + stops columns
```
> See `.claude/rules/h5-smart-simple-pipeline.md` for full architecture.

### Code — H1 Daily Pipeline (Production)
```
airflow/dags/forecast_h1_l3_weekly_training.py   <- Sun: Train 9 models
airflow/dags/forecast_h1_l4_backtest_promotion.py <- Manual: OOS backtest + dashboard export
airflow/dags/forecast_h1_l5_daily_inference.py   <- Mon-Fri: Daily signal generation
airflow/dags/forecast_h1_l5_vol_targeting.py     <- Mon-Fri: Vol-targeting
airflow/dags/forecast_h1_l7_smart_executor.py    <- Mon-Fri: Trailing stop execution
airflow/dags/forecast_h1_l6_paper_monitor.py     <- Mon-Fri: Paper trading monitor
config/execution/smart_executor_v1.yaml          <- H1 config: SHORT-only + trailing stop
```

### Code — RL Training (L2->L3->L4, deprioritized)
```
src/data/ssot_dataset_builder.py         <- L2: Dataset build (anti-leakage)
src/training/environments/trading_env.py <- Gymnasium environment
src/training/trainers/ppo_trainer.py     <- SB3 PPO/RecurrentPPO wrapper
src/training/multi_seed_trainer.py       <- Multi-seed variance reduction
src/training/reward_calculator.py        <- Modular reward (registry pattern)
src/training/config.py                   <- Frozen dataclasses
scripts/run_ssot_pipeline.py             <- L2->L3->L4 orchestrator
```

### Code — RL Inference (L1->L5, deprioritized)
```
airflow/dags/l1_feature_refresh.py       <- L1: Compute + normalize features -> inference_ready_nrt
airflow/dags/l1_model_promotion.py       <- L1: Populate historical features on model approval
airflow/dags/l5_multi_model_inference.py <- L5: Read features + predict -> trading signal
airflow/dags/sensors/postgres_notify_sensor.py <- FeatureReadySensor (LISTEN + polling)
src/core/contracts/feature_contract.py   <- FEATURE_ORDER + FEATURE_ORDER_HASH
src/core/contracts/production_contract.py <- 2-vote model approval
```
> See `.claude/rules/l1-l5-inference-pipeline.md` for full architecture and contracts.

### Code — Dashboard (Next.js)
```
usdcop-trading-dashboard/app/production/page.tsx             <- Strategy-agnostic production page
usdcop-trading-dashboard/app/api/production/status/route.ts  <- GET approval state (file-based)
usdcop-trading-dashboard/app/api/production/approve/route.ts <- POST approve/reject (file-based)
usdcop-trading-dashboard/app/dashboard/page.tsx              <- Main RL dashboard
usdcop-trading-dashboard/app/hub/page.tsx                    <- Hub page
usdcop-trading-dashboard/components/forecasting/             <- Forecasting dashboard components
```

### Data Sources (for local training without DB)
```
seeds/latest/
├── usdcop_daily_ohlcv.parquet      <- Daily COP OHLCV (~3K rows, 2015 -> 2026, COT tz) [H1/H5 training]
├── usdcop_m5_ohlcv.parquet        <- 5-min COP (81K rows, 2019-12 -> 2026-01, COT tz) [RL training]
├── usdmxn_m5_ohlcv.parquet        <- 5-min MXN (95K rows, 2020-01 -> 2026-01, COT tz)
├── usdbrl_m5_ohlcv.parquet        <- 5-min BRL (90K rows, 2020-02 -> 2026-02, COT tz)
├── fx_multi_m5_ohlcv.parquet      <- Unified 3-pair seed (266K rows, for DB restore)
└── macro_indicators_daily.parquet <- Macro ALL 41 cols (10K rows, 1954 -> 2026)

data/pipeline/04_cleaning/output/
└── MACRO_DAILY_CLEAN.parquet      <- Macro CLEAN 17 cols (H1/H5 reads THIS for 4 macro features)
```
> All OHLCV seeds are in **America/Bogota timezone**, session 8:00-12:55 COT, Mon-Fri.
> Regenerate with: `python scripts/build_unified_fx_seed.py`
> See `.claude/rules/l0-data-governance.md` for timezone rules and BRL API quirk.

---

## DATA FLOW: ML FORECASTING (H1 & H5)

### Training Data Pipeline (Weekly, Sunday)

```
seeds/latest/usdcop_daily_ohlcv.parquet  ──┐
                                            ├──> Load + Merge ──> Build 21 Features ──> Target
data/pipeline/04_cleaning/output/          │    (merge_asof      (returns, vol,       (H=1 or H=5
  MACRO_DAILY_CLEAN.parquet ───────────────┘     backward,        RSI, calendar,       log return)
                                                  T-1 shift)      macro T-1)
                                                      │
                                                      v
                                              Expanding Window
                                              (2020-01-01 to last Friday)
                                                      │
                                                      v
                                              StandardScaler.fit(X_train)
                                                      │
                                                      v
                                          ┌───────────┴───────────┐
                                    H1: Train 9 models      H5: Train 2 models
                                    (ridge, BR, ARD,        (Ridge + BayesianRidge)
                                     XGB, LGB, CAT,
                                     3 hybrids)
                                          │                       │
                                          v                       v
                                    .pkl models +           .pkl models +
                                    scaler.pkl              scaler_h5.pkl
```

### 21 SSOT Features (Shared by H1 and H5)

| # | Category | Features | Count |
|---|----------|----------|-------|
| 1-4 | Price | close, open, high, low | 4 |
| 5-8 | Returns | return_1d, return_5d, return_10d, return_20d | 4 |
| 9-11 | Volatility | volatility_5d, volatility_10d, volatility_20d | 3 |
| 12-14 | Technical | rsi_14d (Wilder's), ma_ratio_20d, ma_ratio_50d | 3 |
| 15-17 | Calendar | day_of_week, month, is_month_end | 3 |
| 18-21 | Macro (T-1) | dxy_close_lag1, oil_close_lag1, vix_close_lag1, embi_close_lag1 | 4 |

### Anti-Leakage Mechanisms

| Mechanism | Code | Purpose |
|-----------|------|---------|
| Macro T-1 shift | `df_macro[col].shift(1)` | Use yesterday's macro, not today's |
| Backward merge | `merge_asof(direction='backward')` | Never forward-fill future data |
| Expanding window | Train on `date >= 2020-01-01` to last Friday | No future data in training |
| Target shift | `df["close"].shift(-H)` | Predict H days ahead |

---

## TRAINING & INFERENCE SCHEDULE (All 20 DAGs)

### H1 Daily Pipeline (5 DAGs)

| Day | Time (COT) | DAG | File | Action |
|-----|-----------|-----|------|--------|
| Sun | 01:00 | H1-L3 Weekly Training | `forecast_h1_l3_weekly_training.py` | Retrain 9 models on expanding window |
| Mon-Fri | 13:00 | H1-L5 Daily Inference | `forecast_h1_l5_daily_inference.py` | Ensemble top_3, generate daily signal |
| Mon-Fri | 13:30 | H1-L5 Vol-Targeting | `forecast_h1_l5_vol_targeting.py` | Position sizing + trailing stop params |
| Mon-Fri | 13:35 | H1-L7 Smart Executor | `forecast_h1_l7_smart_executor.py` | Place SHORT order + trail |
| Mon-Fri | 19:00 | H1-L6 Paper Monitor | `forecast_h1_l6_paper_monitor.py` | Log paper trading results |

### H5 Weekly Pipeline (5 DAGs)

| Day | Time (COT) | DAG | File | Action |
|-----|-----------|-----|------|--------|
| Sun | 01:30 | H5-L3 Training | `forecast_h5_l3_weekly_training.py` | Retrain Ridge+BR on expanding window |
| Mon | 08:15 | H5-L5 Signal | `forecast_h5_l5_weekly_signal.py` | Ensemble mean, confidence scoring |
| Mon | 08:45 | H5-L5 Vol-Target | `forecast_h5_l5_vol_targeting.py` | Adaptive stops + sizing |
| Mon-Fri | */30 9-13 | H5-L7 Executor | `forecast_h5_l7_multiday_executor.py` | TP/HS monitor, Friday close |
| Fri | 14:30 | H5-L6 Monitor | `forecast_h5_l6_weekly_monitor.py` | Weekly evaluation + guardrails |

### L0 Data Layer (4 DAGs, shared by all tracks)

| Schedule | DAG | File | Action |
|----------|-----|------|--------|
| `*/5 13-17 * * 1-5` | OHLCV Realtime | `l0_ohlcv_realtime.py` | 3 FX pairs, circuit breaker |
| Manual/Sun | OHLCV Backfill | `l0_ohlcv_backfill.py` | Gap fill + seed export |
| `0 13-17 * * 1-5` | Macro Update | `l0_macro_update.py` | 40 vars, 7 sources, hourly |
| Sun 06:00 UTC | Macro Backfill | `l0_macro_backfill.py` | Full history + seed export |

### RL Pipeline (6 DAGs, deprioritized)

| Schedule | DAG | File | Action |
|----------|-----|------|--------|
| `*/5 13-17 * * 1-5` | L1 Feature Refresh | `l1_feature_refresh.py` | Features -> inference_ready_nrt |
| Manual | L1 Model Promotion | `l1_model_promotion.py` | Historical features on approval |
| After L1 | L5 Multi-Model | `l5_multi_model_inference.py` | Read features + predict |
| Manual | L3 Training | `l3_model_training.py` | PPO multi-seed training |
| Manual | L4 Backtest | `l4_backtest_validation.py` | OOS validation + gates |
| Manual | L4 Experiment | `l4_experiment_runner.py` | A/B experiment runner |

**Key**: H1 and H5 retrain **WEEKLY** (every Sunday), not monthly. Expanding window grows ~5 rows/week.

---

## PIPELINE STAGES

### ML Forecasting (H1 Daily + H5 Weekly)
- **Paradigm**: Supervised regression, `target = ln(close[t+H] / close[t])`
- **H1**: 9 models, H=1 horizon, ensemble top_3 by magnitude, trailing stop
- **H5**: 2 models (Ridge+BR), H=5 horizon, mean ensemble, Smart Simple v1.1
- **Features**: 21 shared features (see table above)
- **Training**: WEEKLY expanding window (Sunday), ALL data from 2020 to last Friday
- **Inference**: H1 = daily (Mon-Fri 13:00), H5 = weekly (Monday 08:15)
- **Execution**: H1 = trailing stop SHORT-only, H5 = TP/HS/Friday-close SHORT-biased

### H5 Smart Simple v1.1 Specifics
- **Confidence**: 3-tier (HIGH/MEDIUM/LOW) from Ridge-BR agreement + magnitude
  - SHORT: flat 1.5x sizing (all tiers)
  - LONG: HIGH=1.0x, MEDIUM=0.5x, LOW=**SKIP**
- **Stops**: Vol-adaptive (HS = clamp(vol * sqrt(5) * 2.0, 1%, 3%), TP = HS * 0.5)
- **Orders**: 100% limit (0% maker fee on MEXC), market only on Friday close
- **Config**: `config/execution/smart_simple_v1.yaml` (SSOT, always read from YAML)

### RL Pipeline (5-min bars, deprioritized)
- **L0**: Data acquisition -> PostgreSQL (4 DAGs, see `.claude/rules/l0-data-governance.md`)
- **L1**: Feature computation + normalization -> `inference_ready_nrt` (2 DAGs)
- **L2**: Dataset build (features, normalization, train/val/test splits)
- **L3**: Training (PPO, multi-seed, eval callbacks)
- **L4**: Backtest (out-of-sample, equity curve, gate validation + promotion proposal)
- **L5**: Inference (read pre-normalized features, predict, signal)

### Experiment = Frozen SSOT
Each experiment is a complete `pipeline_ssot.yaml` stored in `config/experiments/`.
See `.claude/rules/ssot-versioning.md` for full spec. Key points:
- Baseline (`v215b_baseline.yaml`) is NEVER modified
- Each experiment derives from baseline, changing ONE variable
- Config is FROZEN once training starts
- Pipeline runner: `python scripts/run_ssot_pipeline.py --config config/experiments/{id}.yaml`

**RL Data splits (5-min)**:
- Train: 2019-12-24 -> 2024-12-30 (70,072 bars)
- Val: 2025-01-02 -> 2025-06-27 (6,937 bars)
- Test: 2025-07-01 -> 2025-12-30 (7,316 bars)

**Forecasting**: Weekly expanding window (no fixed val/test split in production). Walk-forward with 5 expanding windows used for validation experiments.

---

## DATA LINEAGE & FEATURE STORE

| Concern | RL Pipeline | ML Forecasting | Status |
|---------|-------------|----------------|--------|
| **Lineage** | Formal `lineage.json` with SHA256 hashes | Implicit via Airflow DAG dependencies | RL comprehensive, ML implicit |
| **Feature Store** | Feast + `CanonicalFeatureBuilder` + `inference_ready_nrt` | None — features computed inline in DAGs | Different patterns by design |
| **DVC** | Active: `dvc.yaml` tracks L2/L3/L4 artifacts | Partial: `forecast_prepare_data` -> `forecast_train` | Both tracked |
| **Feature Registry** | `src/feature_store/registry.py` (versioned specs) | `src/forecasting/data_contracts.py` (SSOT constants) | Both exist |
| **Hash Validation** | `FEATURE_ORDER_HASH` checked at L5 inference | No formal hash check | Gap for ML |
| **Norm Stats** | Train-only Z-score in `norm_stats.json` | `StandardScaler.fit()` on full training data | Different approaches |

**Note**: ML forecasting uses inline feature computation (simpler, fewer moving parts). This is intentional — the ML pipeline prioritizes simplicity over infrastructure.

---

## EXPERIMENT RULES (MANDATORY)

### Rule 1: ONE variable per experiment
Never change more than ONE major variable at a time. If an experiment changes action space
AND model type AND features, it's invalid. The only exception is bug fixes (which don't
count as variable changes).

**Major variables** (each requires its own experiment):
- Action space (continuous vs discrete)
- Model architecture (MLP vs LSTM)
- Feature set (adding/removing features)
- Reward function (new components or weights)
- Hyperparameters (ent_coef, lr, gamma, etc.)
- Stop levels (SL/TP percentages)
- Position sizing (fixed vs Kelly)

### Rule 2: Always run 5 seeds (RL only)
Seeds: [42, 123, 456, 789, 1337]. No exceptions. Single-seed results are meaningless.

### Rule 3: Statistical validation BEFORE declaring success
A model is "profitable" only if:
- RL: >=3/5 seeds positive, bootstrap 95% CI excludes zero, PF > 1.05
- ML: p-value < 0.05, walk-forward DA > 55%, Sharpe > 1.0

### Rule 4: Compare against baselines
Every experiment must report comparison vs:
- Buy-and-hold USDCOP in same period (-12.29% in 2025)
- Previous best model in same track
- Random agent with same stops/costs (RL only, if available)

### Rule 5: Record EVERYTHING in experiment log
After EVERY result, append to `.claude/experiments/EXPERIMENT_LOG.md`

### Rule 6: Eval reward != OOS performance (RL)
NEVER select "best model" by eval reward alone. Eval reward is a poor predictor of L4
performance (proven: seed 456 eval=131 lost -20.6%, seed 1337 eval=111 gained +9.6%).

### Rule 7: Use CPU for PPO MlpPolicy, GPU for RecurrentPPO only
RTX 3050 laptop throttles heavily. PPO MlpPolicy is faster on CPU (~300 FPS vs 65-187 GPU).

---

## REQUIRED METRICS (for every report)

### RL Per-seed table (MANDATORY format):
```
| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars | WR_Long% | WR_Short% |
```

### ML Forecasting (MANDATORY format):
```
| Strategy | Return% | Sharpe | p-value | DA% | WR% | PF | MaxDD% | Trades | $10K -> |
```

### Aggregate metrics (MANDATORY):
- Mean return +/- std across seeds (RL) or walk-forward windows (ML)
- Bootstrap 95% CI of mean return
- Comparison vs buy-and-hold
- If p > 0.05: "NOT statistically significant"

---

## KNOWN BUGS & FIXES (already applied)

1. **Infinity in JSON** — `profit_factor: float("inf")` crashes `JSON.parse()`. Fixed: use `None` + `safe_json_dump()`
2. **Hardcoded strategy IDs** — Dashboard read `forecast_vt_trailing` but export wrote `smart_simple_v11`. Fixed: dynamic `strategy_id` lookup
3. **Exit reason mismatch** — Dashboard only knew `trailing_stop`/`session_close`, not `take_profit`/`week_end`. Fixed: universal `EXIT_REASON_COLORS` registry
4. **min_hold_bars bypass** — CLOSE action and reversals must check min_hold_bars BEFORE executing
5. **LSTM states in backtest** — RecurrentPPO requires `model.predict(obs, state=lstm_states, episode_start=...)`
6. **close_reason not passed** — Must propagate from env -> info dict -> reward calculator
7. **flat_reward_weight** — Must be 0.0 (non-zero creates HOLD bias)
8. **backtest max_drawdown** — Must be 99% (not 15% which kills equity curve)
9. **volume_zscore dead** — OHLCV volume is 100% zeros, feature removed
10. **dow encoding** — Use /5.0 (trading days) not /7.0 (calendar days)
11. **USDCOP seed timezone** — Raw TwelveData timestamps were UTC mislabeled as COT. Fixed by `build_unified_fx_seed.py`

---

## VERSION HISTORY (condensed)

### RL Pipeline (5-min bars)

| Version | Return | Sharpe | WR% | PF | Trades | Seeds+/5 | Status |
|---------|--------|--------|-----|------|--------|----------|--------|
| V20 | all negative | -- | -- | -- | -- | 0/? | Failed |
| V21 | -3% to -17% | -- | -- | -- | -- | 0/? | Failed |
| V21.5 | +1.26% | +0.20 | 56.3% | 1.006 | 213 | 1/1 | Superseded |
| V22 post-fix | -31% to +9.6% | -4.1 to +1.1 | 45-54% | 0.89-1.03 | 373-392 | 1/5 | Mixed |
| **V21.5b** | **+2.51%** | **+0.321** | **51.3%** | **1.009** | **349** | **4/5** | **Best RL** |
| EXP-ASYM-001 | -8.67% | -0.93 | 49.5% | 0.976 | 367 | 0/5 | FAILED |
| EXP-HOURLY-001 | -17.49% | -2.62 | 43.8% | 0.74 | 73 | 0/1 | FAILED |
| EXP-DAILY-001 | -8.83% | -2.80 | 12.5% | 0.44 | 8 | 0/1 | FAILED |

**RL Statistical baselines**: Buy-and-hold -14.66%, Random -4.12%, Bootstrap CI [-0.69%, +6.15%], p=0.272 NOT significant.

### H1 Daily Forecasting Pipeline

| Strategy | Return 2025 | Sharpe | p-value | $10K -> | Status |
|----------|-------------|--------|---------|---------|--------|
| Buy & Hold | -12.29% | -- | -- | $8,771 | Baseline |
| Forecast 1x | +21.60% | 2.109 | 0.068 | $12,160 | -- |
| + Vol-target | +32.62% | 2.062 | 0.053 | $13,262 | -- |
| **+ Trailing stop** | **+36.84%** | **3.135** | **0.0178** | **$13,684** | **PRODUCTION** |

### H5 Weekly Smart Simple Pipeline

| Strategy | Return 2025 | Sharpe | p-value | $10K -> | Status |
|----------|-------------|--------|---------|---------|--------|
| Smart Simple v1.0 | +13.75% | 2.867 | 0.032 | $11,375 | Superseded |
| **Smart Simple v1.1** | **+20.03%** | **3.516** | **0.0097** | **$12,003** | **PAPER TRADING** |
| Smart Simple v1.1 (2026) | +7.12% | 19.12 | 0.000 | $10,712 | 3/3 wins |

**v1.0 -> v1.1 changes**: HS multiplier 1.5->2.0 (eliminated 2 hard stops), flat SHORT sizing 1.5x.
**Diagnostic**: Theoretical ceiling +40.84%, capture 26.8%. Next lever = DA (features), not stops.

---

## EXPERIMENT SYSTEM

| File | Purpose |
|------|---------|
| `.claude/experiments/EXPERIMENT_QUEUE.md` | Prioritized queue + forecasting-first plan |
| `.claude/experiments/EXPERIMENT_LOG.md` | Append-only results log (12+ experiments) |
| `.claude/rules/experiment-protocol.md` | Execution protocol (phases, rules, reporting) |
| `.claude/rules/ssot-versioning.md` | Config versioning spec (one SSOT per experiment) |
| `.claude/rules/l0-data-governance.md` | L0 layer: 4 DAGs (OHLCV + Macro), contracts |
| `.claude/rules/l2-l3-l4-training-pipeline.md` | L2-L4 RL training pipeline |
| `.claude/rules/l1-l5-inference-pipeline.md` | L1-L5 RL inference pipeline, feature store |
| `.claude/rules/h5-smart-simple-pipeline.md` | H5 weekly Smart Simple pipeline (v1.1) |
| `.claude/rules/sdd-mlops-lifecycle.md` | SDD: **Master lifecycle** (bootstrap to production, 8 stages) |
| `.claude/rules/sdd-strategy-spec.md` | SDD: Universal strategy interface + StrategyRegistry |
| `.claude/rules/sdd-approval-spec.md` | SDD: Two-vote approval + gates |
| `.claude/rules/sdd-dashboard-integration.md` | SDD: JSON/CSV/PNG data contracts + strategy selector |
| `.claude/rules/sdd-pipeline-lifecycle.md` | SDD: 8-stage pipeline lifecycle (condensed) |

---

## DO NOT

### JSON & Dashboard
- Do NOT produce `Infinity`, `NaN`, or `undefined` in JSON — use `null` via `safe_json_dump()`
- Do NOT hardcode `strategy_id` in dashboard — use `summary.strategy_id` dynamic lookup
- Do NOT add exit reasons without updating BOTH `strategy.contract.ts` AND `strategy_schema.py`
- Do NOT bypass the 2-vote approval — backtest gates (auto) + human review (manual) required
- Do NOT modify `approval_state.json` manually — use `--reset-approval` CLI or dashboard API

### RL Pipeline
- Do NOT hardcode values — everything from pipeline_ssot.yaml
- Do NOT train RL without multi-seed (5 seeds minimum)
- Do NOT declare a model "profitable" without statistical tests
- Do NOT change multiple variables in one experiment
- Do NOT use eval reward to select best model for production
- Do NOT use GPU for PPO MlpPolicy (use CPU)
- Do NOT skip L2 rebuild if features changed
- Do NOT use RL on hourly or daily bars (proven insufficient data)

### Forecasting Pipeline
- Do NOT use macro_score composite as a standalone filter (ANOVA p=0.297, not significant)
- Do NOT skip walk-forward validation — single train/test split is insufficient
- Do NOT add features without checking walk-forward DA improvement (overfitting risk)
- Do NOT use future data in features — all macro features must be lagged T-1
- Do NOT use pandas `ewm()` for RSI — use Wilder's EMA (alpha=1/period)

### H5 Smart Simple Pipeline
- Do NOT add confidence tiers for SHORT sizing — flat 1.5x is correct (N too small)
- Do NOT reduce HS multiplier below 2.0x — eliminates hard stops in backtest
- Do NOT enable trailing stop or re-entry in Smart Simple — simplicity is the edge
- Do NOT take LONG trades with LOW confidence — net effect is negative (-0.75%)
- Do NOT hardcode stops/sizing in backtest scripts — always read from smart_simple_v1.yaml

### Data & Infrastructure
- Do NOT store OHLCV timestamps in UTC — always America/Bogota
- Do NOT fetch BRL from TwelveData with `timezone=America/Bogota` (returns incomplete data, use UTC + convert)
- Do NOT compute features in L5 — L1 is the ONLY feature computation layer (RL)
- Do NOT write to `inference_ready_nrt` from outside L1 DAGs (RL)
