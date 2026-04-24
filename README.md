# USDCOP Trading System

A production-grade algorithmic trading system for USD/COP (US Dollar / Colombian Peso) using supervised ML forecasting, AI-powered market analysis, and automated execution. The system runs two statistically significant pipelines — **H1 Daily** and **H5 Weekly** — both beating buy-and-hold by 30+ percentage points in 2025 out-of-sample testing.

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

## 🎓 MLOps Course Project — Final Deliverable

> Presentation: **2026-04-23** • Repository: **2026-04-30**
> Full deliverable doc: [docs/COURSE_PROJECT.md](docs/COURSE_PROJECT.md)

### Course requirements compliance

| Requirement | Status | Where |
|---|---|---|
| ≥2 non-REST course technologies | ✅ **gRPC + Kafka** | `services/grpc_predictor/`, `services/kafka_bridge/` |
| Docker containerization | ✅ 15+ services | `docker-compose.compact.yml` |
| Orchestration (best grade) | ✅ Airflow + MLflow | `airflow/dags/`, MLflow @ :5001 |
| Functional live demo | ✅ | `make course-demo` |
| Toy-model-acceptable | ✅ H5 Smart Simple (Ridge + BR + XGB) | Realistic but simple ML |
| Cloud (optional) | ⬜ All local Docker | — |
| Federated learning (optional) | ⬜ | — |

### Quickstart (~3 minutes to demo-ready)

```bash
# 1. Clone + start
git clone <repo-url> && cd USDCOP-RL-Models
docker compose -f docker-compose.compact.yml up -d
sleep 60   # Wait for services to initialize

# 2. Run full demo (8-10 min walkthrough)
make course-demo

# 3. Individual demos
make course-grpc     # gRPC Predict() call
make course-kafka    # Kafka producer→consumer roundtrip
make course-mlflow   # MLflow runs in experiment tracker

# 4. Open the UIs
# Airflow:         http://localhost:8080
# MLflow:          http://localhost:5001
# Grafana:         http://localhost:3002
# Redpanda Console http://localhost:8088
# Dashboard:       http://localhost:5000
# gRPC:            localhost:50051 (use client_example.py)
```

### Architecture snapshot (MLOps course view)

```
┌─────────────────┐   ┌──────────────┐   ┌────────────────┐
│ Airflow DAGs    │──▶│ MLflow runs  │   │ gRPC Predictor │
│ (27, L0→L7)     │   │ (:5001)      │◀──│ (:50051)       │
└────────┬────────┘   └──────────────┘   └────────────────┘
         │                                        ▲
         ▼                                        │
┌─────────────────┐     ┌──────────────────┐     │
│ Signal DAG L5   │────▶│ Kafka / Redpanda │     │
│ (Mon 08:15 COT) │     │ topic signals.h5 │     │
└─────────────────┘     └────────┬─────────┘     │
                                 ▼               │
                       ┌─────────────────────────┴──┐
                       │ SignalBridge OMS (:8085)   │
                       │ (FastAPI — REST baseline)  │
                       └────────────────────────────┘
```

### Team

| Name | Role | Email |
|---|---|---|
| _(TBD)_ | Lead / MLOps | — |
| _(TBD)_ | — | — |
| _(TBD)_ | — | — |
| _(TBD)_ | — | — |

---

## Table of Contents

- [MLOps Course Project — Final Deliverable](#-mlops-course-project--final-deliverable)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Four Tracks](#four-tracks)
- [21 Shared Features](#21-shared-features)
- [DAG Schedule (26 DAGs)](#dag-schedule-26-dags)
- [Backtest-to-Production Promotion](#backtest-to-production-promotion)
- [Dashboard (8 Pages)](#dashboard-8-pages)
- [Infrastructure (25+ Services)](#infrastructure-25-services)
- [Data Sources & Backup](#data-sources--backup)
- [Database Schema](#database-schema)
- [Experiment Protocol](#experiment-protocol)
- [Project Structure](#project-structure)
- [SDD Specifications](#sdd-specifications)
- [Testing & CI/CD](#testing--cicd)
- [Known Gotchas](#known-gotchas)

---

## Architecture Overview

```
                              USDCOP Trading System
    ================================================================

    DATA LAYER (L0)                    TRAINING (weekly, Sunday)
    5 Airflow DAGs                     Expanding window: 2020 -> last Friday
    3 FX pairs (COP, MXN, BRL)        21 shared features, anti-leakage
    40 macro variables, 7 sources      H1: 9 models  |  H5: Ridge + BR
         |                                  |
         v                                  v
    +-----------+     +----------+    +------------+    +----------+
    | PostgreSQL| --> | Features | -> |   Models   | -> | Backtest |
    | TimescaleDB     | 21 cols  |    | H1: 9      |    | OOS 2025 |
    | OHLCV+Macro     | anti-leak|    | H5: 2      |    | p < 0.05 |
    +-----------+     +----------+    +------------+    +----------+
         |                                                    |
         |            +-- APPROVAL (2-vote) --+               |
         |            | Vote 1: Auto gates    | <-------------+
         |            | Vote 2: Human review  |
         |            +-----------+-----------+
         |                        |
         v                        v
    +--- NEWS ENGINE ---+  +--- PRODUCTION ---+    +--- MONITORING ---+
    | 5 source adapters |  | Retrain + deploy | -> | Weekly Airflow   |
    | LLM analysis      |  | 2020->2025 data  |    | Guardrails       |
    | /analysis page    |  +-----------------+     | Circuit breaker  |
    +------------------+                           +------------------+
                                    |
                           +--- EXECUTION ---+
                           | SignalBridge OMS |
                           | MEXC / Binance   |
                           | Paper / Testnet  |
                           +-----------------+
```

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11+ (`pip install -e ".[all]"`)
- Node.js 18+ (for dashboard)
- Git LFS (`git lfs install`)

### Deployment Modes

| Mode | Command | Services | RAM | Use Case |
|------|---------|----------|-----|----------|
| **Compact** | `make compact` | 12 | ~6-8GB | Daily: training, APIs, MLflow, SignalBridge, dashboard |
| **Compact + Monitoring** | `make compact-monitoring` | 15 | ~8-10GB | + Prometheus, Grafana, AlertManager |
| **Full Enterprise** | `make docker-up` | 25+ | ~12GB | + Vault, Jaeger, Loki, Promtail, pgAdmin |

### Full Setup (Stages 0-7)

```bash
git clone <repo-url>
cd USDCOP-RL-Models
git lfs pull

# Stage 0: Bootstrap
cp .env.example .env                               # Edit with your credentials
make compact                                        # 12 services (or docker-compose up -d for 25+)

# Stage 1: Data alignment
airflow dags trigger core_l0_01_ohlcv_backfill      # OHLCV current
airflow dags trigger core_l0_03_macro_backfill      # Macro current

# Stage 2: Model training
python scripts/generate_weekly_forecasts.py          # /forecasting activates (9 models x 7 horizons)
python scripts/train_and_export_smart_simple.py --phase backtest  # /dashboard shows 2025

# Stage 3-4: Review on /dashboard (automatic)
# Stage 5: Click "Aprobar" on /dashboard (Vote 2/2)

# Stage 6: Production deploy
python scripts/train_and_export_smart_simple.py --phase production  # /production shows 2026

# Stage 7: Airflow DAGs take over weekly cycle (L3->L5->L7->L6)
```

**What happens on startup**: PostgreSQL starts, init scripts run in order (00-extensions -> 01-tables -> 02-macro -> 03-views -> 04-seeding). The seeder auto-detects empty tables and restores from: daily backup parquets -> Git LFS seeds -> MinIO -> legacy CSV.

> Full operator guide with checklists: `.claude/rules/sdd-mlops-lifecycle.md`

---

## Four Tracks

### 1. H1 Daily Pipeline (PRODUCTION)

9 supervised models, H=1 day horizon, ensemble of top 3 by prediction magnitude, trailing stop execution. **SHORT-only** mode.

- **Models**: ridge, bayesian_ridge, ard, xgboost, lightgbm, catboost, 3 hybrids
- **Execution**: Trailing stop (activation +0.2%, trail 0.3%, hard stop 1.5%)
- **Result**: +36.84%, Sharpe 3.135, p=0.0178 (2025 OOS)
- **Config**: `config/execution/smart_executor_v1.yaml`

### 2. H5 Weekly Pipeline (PAPER TRADING)

2 linear models (Ridge + BayesianRidge), H=5 day horizon, mean ensemble, Smart Simple v1.1 execution (TP/HS/Friday close). 15-week evaluation started 2026-02-16.

- **Confidence**: 3-tier (HIGH/MEDIUM/LOW) from Ridge-BR agreement + magnitude
- **Sizing**: SHORT flat 1.5x (all tiers); LONG: HIGH=1.0x, MEDIUM=0.5x, LOW=**SKIP**
- **Stops**: Vol-adaptive: `HS = clamp(vol * sqrt(5) * 2.0, 1%, 3%)`, `TP = HS * 0.5`
- **Result**: +20.03%, Sharpe 3.516, p=0.0097 (2025 OOS). +7.12% (2026 YTD, 3/3 wins)
- **Config**: `config/execution/smart_simple_v1.yaml`

### 3. News Engine & Analysis Module (OPERATIONAL)

AI-powered market analysis combining 5 news source adapters with LLM-generated Spanish narratives.

- **Adapters**: GDELT Doc+Context, NewsAPI, Investing.com, La Republica, Portafolio
- **Enrichment**: 5 stages (categorize, relevance, sentiment, NER, breaking detection)
- **Analysis**: MacroAnalyzer (13 vars: SMA/Bollinger/RSI/MACD/z-score) + LLM narratives
- **LLM**: Azure OpenAI primary + Anthropic Claude fallback, $1/day budget, file-based caching
- **Output**: ~60 daily news features for ML, weekly analysis JSONs for `/analysis` page
- **Dashboard**: 14 React components, 4 API routes, macro chart grid, AI chat widget
- **Spec**: `.claude/rules/news-and-analysis-sdd.md`

### 4. RL Pipeline (DEPRIORITIZED)

PPO agent on 5-min bars. NOT statistically significant (p=0.272, +2.51%). Kept for research.

---

## 21 Shared Features

Both H1 and H5 pipelines use identical features from daily OHLCV + macro data:

| # | Category | Feature | Source |
|---|----------|---------|--------|
| 1-4 | Price | `close`, `open`, `high`, `low` | Daily OHLCV |
| 5-8 | Returns | `return_1d`, `return_5d`, `return_10d`, `return_20d` | Log returns |
| 9-11 | Volatility | `volatility_5d`, `volatility_10d`, `volatility_20d` | Std of returns |
| 12-14 | Technical | `rsi_14d` (Wilder's EMA), `ma_ratio_20d`, `ma_ratio_50d` | Price-derived |
| 15-17 | Calendar | `day_of_week`, `month`, `is_month_end` | Date |
| 18-21 | Macro (T-1) | `dxy_close_lag1`, `oil_close_lag1`, `vix_close_lag1`, `embi_close_lag1` | Macro DB |

**Anti-leakage**: Macro T-1 shift (`shift(1)`), backward merge (`merge_asof(direction='backward')`), expanding window (2020 to last Friday), train-only `StandardScaler`.

---

## DAG Schedule (26 DAGs)

| Pipeline | DAGs | Key Timing (COT) | Spec |
|----------|------|-------------------|------|
| **H1 Daily** | 5 | Sun 01:00 train; Mon-Fri 13:00 signal, 13:30 vol-target, 13:35 executor, 19:00 monitor | `h5-smart-simple-pipeline.md` |
| **H5 Weekly** | 5 | Sun 01:30 train; Mon 08:15 signal, 08:45 vol-target; Mon-Fri */30 9-13 executor; Fri 14:30 monitor | `h5-smart-simple-pipeline.md` |
| **L0 Data** | 5 | OHLCV: */5 8-12 Mon-Fri; Macro: hourly; Backfill: Sun/Manual; Seed backup: daily 13:00 | `l0-data-governance.md` |
| **RL** | 6 | All manual/event-triggered except L1 Feature Refresh (*/5 8-12 Mon-Fri) | `l1-l5-inference-pipeline.md` |
| **News+Analysis** | 5 | News: 3x/day (02,07,13 COT); Alert: */30; Digest: Mon; Analysis: 14:00 Mon-Fri | `news-and-analysis-sdd.md` |

H1 and H5 retrain **WEEKLY** (every Sunday). Expanding window grows ~5 rows/week.

> Full collision-free timeline: `.claude/rules/elite-operations.md`

---

## Backtest-to-Production Promotion

### 2-Vote Approval System

```
  BACKTEST COMPLETE (--phase backtest)
        |
  Vote 1/2 (Automatic) — Python evaluates 5 gates:
    min_return > -15%, Sharpe > 0.0, MaxDD < 20%, trades >= 10, p < 0.05
        |
  approval_state.json: PENDING_APPROVAL
        |
  Vote 2/2 (Human on /dashboard)
    Reviews: KPIs, candlestick chart, trade table, gate results
        |
    APPROVE  →  Stage 6 deploy (auto via deploy_manifest)
    REJECT   →  Fix, then --reset-approval
```

**Post-approval**: Deploy API reads `deploy_manifest` from `approval_state.json` and automatically spawns the production pipeline. Strategy-agnostic — any strategy (H1, H5, RL) auto-deploys.

> Full approval spec: `.claude/rules/sdd-approval-spec.md`

---

## Dashboard (8 Pages)

Next.js 15 application (Tailwind CSS + shadcn/ui + Lightweight Charts). 47 API routes.

| Page | Route | Purpose | Activates After |
|------|-------|---------|-----------------|
| Landing | `/` | Root redirect | Always |
| Hub | `/hub` | Navigation to all sections | Always |
| Forecasting | `/forecasting` | Model zoo: 9 models x 7 horizons, walk-forward metrics | `generate_weekly_forecasts.py` |
| Dashboard | `/dashboard` | 2025 backtest review + human approval (Vote 2/2) | `--phase backtest` |
| Production | `/production` | 2026 YTD monitoring (read-only) | `--phase production` |
| Analysis | `/analysis` | Weekly AI market analysis, macro charts, chat widget | `generate_weekly_analysis.py` |
| Execution | `/execution` | SignalBridge OMS: exchanges, kill switch, risk limits | Exchange API keys |
| Login | `/login` | Authentication | Always |

**JSON safety**: All Python-to-Dashboard exports use `safe_json_dump()` — converts `Infinity`/`NaN` to `null`.

---

## Infrastructure (25+ Services)

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL 15 + TimescaleDB | 5432 | Primary OLTP (OHLCV, macro, H5 tables, news, analysis) |
| Redis 7 | 6379 | Caching, pub/sub, signal streaming, circuit breaker state |
| MinIO | 9001 | S3-compatible object store (models, artifacts, seeds) |
| Airflow (scheduler + webserver) | 8080 | DAG orchestration (26 DAGs) |
| Dashboard (Next.js 15) | 5000 | Trading dashboard (8 pages, 47 API routes) |
| SignalBridge API | 8085 | OMS: MEXC/Binance via CCXT, paper/testnet/live toggle |
| MLflow | 5001 | Experiment tracking |
| HashiCorp Vault | 8200 | Secrets management (AES-256-GCM for exchange API keys) |
| Prometheus | 9090 | Metrics collection + 37+ alert rules |
| Grafana | 3002 | 4 dashboards: trading, MLOps, system health, macro ingestion |
| AlertManager | 9093 | Alert routing: 3 Slack channels + PagerDuty |
| Loki | 3100 | Log aggregation (31-day retention) |
| Promtail | -- | Log shipping (Docker socket bind) |
| Jaeger | 16686 | Distributed tracing (OTLP) |
| pgAdmin | 5050 | Database admin UI |

> Full monitoring spec: `.claude/rules/sdd-observability.md`
> Full execution spec: `.claude/rules/sdd-execution-bridge.md`

---

## Data Sources & Backup

### OHLCV (3 FX Pairs, Git LFS)

| Pair | Source | Seed File | Rows |
|------|--------|-----------|------|
| USD/COP | TwelveData | `seeds/latest/usdcop_m5_ohlcv.parquet` | ~81K (5-min) |
| USD/MXN | Dukascopy | `seeds/latest/usdmxn_m5_ohlcv.parquet` | ~95K (5-min) |
| USD/BRL | TwelveData | `seeds/latest/usdbrl_m5_ohlcv.parquet` | ~90K (5-min) |
| Daily COP | TwelveData | `seeds/latest/usdcop_daily_ohlcv.parquet` | ~3K (H1/H5 training) |

All timestamps in `America/Bogota` timezone. Session: 8:00-12:55 COT, Monday-Friday.

### Macro (40 Variables, 7 Sources)

FRED, Investing.com, BanRep, BCRP, Fedesarrollo, DANE, BanRep BOP.
Clean output: `data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet` (17 cols, ~10K rows).

### News (5 Sources, Git LFS)

| Source | File | Articles |
|--------|------|----------|
| GDELT | `data/news/gdelt_articles_historical.csv` | ~128K (2017-2026) |
| Colombia news | `data/news/colombia_news_historical.csv` | ~5.7K |
| Investing.com | `data/news/investing_articles_historical.csv` | ~1.7K |

### Backup & Disaster Recovery

| Tier | What | Where | Frequency |
|------|------|-------|-----------|
| T1: Live DB | OHLCV, macro, forecasts, signals, trades | PostgreSQL | Realtime |
| T2: Daily backup | OHLCV + macro tables | `data/backups/seeds/*.parquet` | Daily 13:00 COT |
| T3: Git LFS | Seeds, news CSVs, analysis JSONs, trade exports | Git repository | On commit |

**Restore on `docker-compose up -d`**: Daily backup (freshest) -> Git LFS seeds -> MinIO -> legacy CSV.

> Full backup spec: `.claude/rules/backup-recovery-protocol.md`

---

## Database Schema

### Core Tables (48 migrations)

| Table | Purpose | PK |
|-------|---------|------|
| `usdcop_m5_ohlcv` | 5-min OHLCV for 3 FX pairs | `(time, symbol)` |
| `macro_indicators_daily` | Daily macro (18 cols) | `fecha` |
| `macro_indicators_monthly` | Monthly macro (8 cols) | `fecha` |
| `forecast_h5_predictions` | Ridge + BR predictions per week | `(week, model_name)` |
| `forecast_h5_signals` | Ensemble signal + confidence + stops | `week` |
| `forecast_h5_executions` | Weekly execution tracking | `week` |
| `forecast_h5_subtrades` | Subtrade records within a week | `execution_id` |
| `news_articles` | Enriched news articles (5 sources) | `(source_id, url_hash)` |
| `news_feature_snapshots` | ~60 daily news features | `snapshot_date` |
| `weekly_analysis` | LLM weekly reports | `(iso_year, iso_week)` |
| `daily_analysis` | LLM daily entries | `analysis_date` |

---

## Experiment Protocol

### Core Rules

1. **ONE variable per experiment** — never change model + features simultaneously
2. **5 seeds for RL** — [42, 123, 456, 789, 1337], no exceptions
3. **Statistical validation** — p < 0.05, walk-forward DA > 55%, Sharpe > 1.0
4. **Compare baselines** — buy-and-hold (-12.29%), previous best, random (RL)
5. **Log everything** — `.claude/experiments/EXPERIMENT_LOG.md`

### Experiment History

| ID | Variable | Result | Status |
|----|----------|--------|--------|
| V21.5b | RL Baseline | +2.51%, p=0.272 | Best RL (not significant) |
| EXP-ASYM-001 | SL/TP ratio | -8.67%, 0/5 seeds | FAILED |
| Smart Simple v1.0 | Weekly H5 | +13.75%, p=0.032 | Superseded |
| **Smart Simple v1.1** | HS 2.0x + flat SHORT | **+20.03%, p=0.0097** | **Active** |

> Full protocol: `.claude/rules/experiment-protocol.md`

---

## Project Structure

```
USDCOP-RL-Models/
├── CLAUDE.md                          # Claude Code instructions (auto-loaded)
├── README.md                          # This file
├── Makefile                           # 268 lines: test, lint, docker, db, validate
├── pyproject.toml                     # Python config (deps, ruff, mypy, pytest, coverage)
├── docker-compose.yml                 # Full enterprise (25+ services)
├── docker-compose.compact.yml         # Compact mode (12 services)
├── dvc.yaml                           # DVC pipeline definition
├── params.yaml                        # DVC parameters
├── LICENSE
│
├── .claude/rules/                     # SDD specs + pipeline rules (19 files)
├── .claude/experiments/               # Experiment queue + log
├── .github/workflows/                 # 9 CI/CD workflows
│
├── airflow/dags/                      # 26 Airflow DAGs
│   ├── l0_*.py                        # L0 data layer (5 DAGs)
│   ├── forecast_h1_*.py               # H1 daily pipeline (5 DAGs)
│   ├── forecast_h5_*.py               # H5 weekly pipeline (5 DAGs)
│   ├── l1_*.py, l5_*.py               # RL inference (3 DAGs)
│   ├── l3_*.py, l4_*.py               # RL training (3 DAGs)
│   ├── news_*.py                      # News engine (4 DAGs)
│   └── analysis_*.py                  # Analysis module (1 DAG)
│
├── config/                            # SSOT configuration files
│   ├── execution/                     # H1 + H5 execution configs
│   ├── experiments/                   # Frozen RL experiment configs
│   ├── analysis/                      # Analysis SSOT (LLM budget, macro vars)
│   ├── prometheus/                    # Alert rules (37+ rules)
│   ├── alertmanager/                  # Slack + PagerDuty routing
│   └── grafana/                       # 4 dashboards + datasources
│
├── src/                               # Core Python packages
│   ├── forecasting/                   # ForecastingEngine, 9 models, confidence, stops
│   ├── execution/                     # SmartExecutor, MultiDayExecutor, TrailingStop
│   ├── risk/                          # RiskCheckChain (9 checks) + Commands (6)
│   ├── trading/                       # RiskEnforcer (7 rules)
│   ├── news_engine/                   # 5 adapters, enrichment, cross-reference, features
│   ├── analysis/                      # MacroAnalyzer, LLMClient, WeeklyGenerator
│   ├── contracts/                     # Python SDD contracts (6 files)
│   ├── training/                      # RL: PPO, environments, rewards
│   └── data/                          # Dataset builders, loaders
│
├── services/                          # Standalone API services
│   └── signalbridge_api/              # SignalBridge OMS (FastAPI + CCXT)
│
├── scripts/                           # Pipeline scripts, backtests, diagnostics
├── database/migrations/               # 48 SQL migrations
├── init-scripts/                      # Docker startup scripts (00-04)
├── seeds/latest/                      # Git LFS seed parquets
├── data/                              # Pipeline data, news CSVs, backups
├── tests/                             # Unit, integration, contracts, load tests
│
├── usdcop-trading-dashboard/          # Next.js 15 dashboard
│   ├── app/                           # 8 pages + 47 API routes
│   ├── components/                    # React components (forecasting, analysis, production)
│   ├── lib/contracts/                 # TypeScript SDD contracts (10 files)
│   ├── hooks/                         # React Query hooks
│   └── public/data/                   # JSON/PNG exports (production, analysis, forecasting)
│
├── docker/                            # Docker configs (Dockerfiles, nginx, pgadmin, archive)
├── docs/                              # Documentation (organized by category)
│   ├── architecture/                  # API, data flow, DB schema (15 files)
│   ├── operations/                    # Runbooks, checklists, SLA (17 files)
│   ├── guides/                        # Deployment, development, onboarding (14 files)
│   └── archive/                       # Historical audits, plans, legacy docs
└── prometheus/                        # Prometheus main config
```

---

## SDD Specifications

The project follows **Spec-Driven Development** (SDD): specs define contracts, contracts enforce code.

| Spec | Purpose |
|------|---------|
| `sdd-mlops-lifecycle.md` | **Master lifecycle**: 8 stages (bootstrap to production) |
| `sdd-strategy-spec.md` | Universal strategy interface (TradeRecord, MetricsSummary, ExitReasons) |
| `sdd-approval-spec.md` | 2-vote approval, 5 gates, approval_state.json schema |
| `sdd-dashboard-integration.md` | JSON/CSV/PNG data contracts, page data flows |
| `sdd-pipeline-lifecycle.md` | Quick reference: stages, CLI commands, DAG schedules |
| `sdd-execution-bridge.md` | SignalBridge OMS, MEXC/Binance adapters, kill switch |
| `sdd-risk-management.md` | 9-check chain, command pattern, 7-rule enforcer |
| `sdd-observability.md` | Prometheus 37+ rules, Grafana 4 dashboards, AlertManager |
| `sdd-cicd-testing.md` | 9 GitHub Actions, Makefile, 70% coverage gate |
| `news-and-analysis-sdd.md` | News Engine + Analysis Module architecture |

Additional pipeline specs: `h5-smart-simple-pipeline.md`, `l0-data-governance.md`, `l1-l5-inference-pipeline.md`, `l2-l3-l4-training-pipeline.md`, `experiment-protocol.md`, `ssot-versioning.md`, `elite-operations.md`, `data-freshness-enforcement.md`, `backup-recovery-protocol.md`.

### SDD Contracts

| Language | Location | Key Exports |
|----------|----------|-------------|
| Python (6) | `src/contracts/` | `StrategyTrade`, `StrategyStats`, `safe_json_dump()`, `UniversalSignalRecord` |
| TypeScript (10) | `usdcop-trading-dashboard/lib/contracts/` | `StrategyTrade`, `ApprovalState`, `WeeklyViewData`, `EXIT_REASON_COLORS` |

---

## Testing & CI/CD

### Local Testing

```bash
make test-unit         # Unit tests
make test-contracts    # SSOT contract validation
make test-integration  # Integration tests (requires docker-up)
make coverage          # All tests with 70% coverage gate
make ci                # lint + typecheck + test (pre-push)
```

### CI/CD (9 GitHub Actions Workflows)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR | Lint + typecheck + test (70% coverage gate) |
| `security.yml` | Push/PR + weekly | Bandit + Safety + pip-audit |
| `contracts-check.yml` | Python changes | SSOT compliance + hash validation |
| `drift-check.yml` | Daily 6AM UTC | Feature drift detection (KS test) |
| `deploy.yml` | Manual | Staging -> production with approval gate |
| `canary-promote.yml` | Manual | Gradual canary: 10% -> 25% -> 50% -> 100% |

> Full CI/CD spec: `.claude/rules/sdd-cicd-testing.md`

---

## Known Gotchas

- **sklearn 1.6.1**: Use `max_iter` not `n_iter` for BayesianRidge/ARD
- **CatBoost params**: `iterations`/`depth` not `n_estimators`/`max_depth`
- **BRL API**: TwelveData needs `timezone=UTC` for USD/BRL (Bogota returns incomplete)
- **OHLCV timezone**: Always `America/Bogota`, session 8:00-12:55 COT, Mon-Fri
- **JSON safety**: NEVER `float("inf")` — use `None` + `safe_json_dump()`
- **RSI**: Use Wilder's EMA (`alpha=1/period`), NOT pandas `ewm()` default
- **Smart Simple**: Always read params from `smart_simple_v1.yaml`, never hardcode
- **News DB tables**: `news_daily_digests` (not news_digests), `news_ingestion_log` (singular)
- **News DB connection**: Use `POSTGRES_*` env vars, NOT `USDCOP_DB_*`
- **EXECUTION_MODE**: Default `paper`, validate on `testnet` before `live`
- **LLM budget**: $1/day, $15/month hard limits in `weekly_analysis_ssot.yaml`
- **Container deps**: `feedparser` + `vaderSentiment` needed in Airflow container for news pipeline

---

*Last updated: March 2026*
