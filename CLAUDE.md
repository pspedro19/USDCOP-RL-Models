# CLAUDE.md — USDCOP Trading System

> Loaded automatically by Claude Code at the start of every session.
> Contains project context, architecture, contracts, and rules that MUST be followed.

---

## PROJECT IDENTITY

**Goal**: Build a profitable USDCOP trading system using supervised forecasting.
**Exchange**: MEXC (0% maker fees, 1 bps slippage estimate)
**Architecture**: Spec-Driven Development (SDD) — specs define contracts, contracts enforce code.

**Three Tracks** (in priority order):

1. **H5 Weekly Pipeline (PRODUCTION)**: Smart Simple v2.0, Ridge+BR+XGBoost + Regime Gate
   - **+25.63%, Sharpe 3.35, p=0.006** (2025 backtest, 34 trades)
   - **+0.61%** (2026 YTD, 1/1 wins — regime gate blocked 11 of 12 mean-reverting weeks)
   - Architecture: Regime Gate (Hurst) → Ridge/BR/XGB ensemble → Effective HS → DL → CB
   - DAGs: H5-L3/L4/L5/L6/L7 (see `.claude/rules/h5-smart-simple-pipeline.md`)

2. **News Engine & Analysis Module (OPERATIONAL)**: AI-generated market analysis + news intelligence
   - Active sources: Investing.com (78 articles), Portafolio (276 articles) in DB
   - LLM weekly analysis: W01-W15 generated (Azure OpenAI GPT-4o-mini)
   - Dashboard: `/analysis` page (14 components, 4 API routes)
   - See `.claude/rules/news-and-analysis-sdd.md`

3. **H1 Daily Pipeline (PAUSED)**: 9 models, H=1 horizon — DAGs paused pending v2.0 validation

4. **RL (DEPRIORITIZED)**: PPO agent, 5-min bars, NOT significant (p=0.272)

**Strategic Pivot (2026-03-18)**: 10-agent audit revealed Ridge/BR model has R² < 0 in both years.
Alpha comes from regime gate (knows when NOT to trade) + TP/HS mechanics, not from model predictions.
**2026 Regime**: Hurst = 0.28-0.49 (mean-reverting → transitioning). Gate correctly blocks most weeks.

**Current Best**:
- H5 Weekly v2.0: Ridge+Gate+EffectiveHS → $10K → $12,563 (2025), $10K → $10,061 (2026 YTD)
- Momentum v3.0 (paper): $10K → $10,353 (2026 YTD, better in mean-reverting but worse in trending)
- RL: V21.5b — +2.51% mean (4/5 seeds), NOT significant

---

## QUICK START (First-Time Setup)

> See `.claude/rules/sdd-mlops-lifecycle.md` for the full operator guide with checklists.

### Two Deployment Modes

| Mode | Command | Services | RAM | Use Case |
|------|---------|----------|-----|----------|
| **Compact** | `make compact` | 12 | ~6-8GB | Daily: training, APIs, MLflow, SignalBridge, dashboard |
| **Compact + Monitoring** | `make compact-monitoring` | 15 | ~8-10GB | + Prometheus, Grafana, AlertManager |
| **Full Enterprise** | `make docker-up` | 25+ | ~12GB | + Vault, Jaeger, Loki, Promtail, pgAdmin |

```
# COMPACT (recommended for daily use):
1. make compact                                    → 12 services (Stage 0)

# FULL ENTERPRISE (all services):
1. docker-compose up -d                            → 25+ containers (Stage 0)

# Then (same for both modes):
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
Layer 1: SPEC (defines what)      -> .claude/rules/sdd-*.md           (10 specs)
Layer 2: CONTRACT (enforces how)  -> lib/contracts/ + src/contracts/   (10 TS + 6 Python contracts)
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
| `news-and-analysis-sdd.md` | **News Engine + Analysis Module** | 5 adapters, enrichment, LLM analysis, `/analysis` page, 6 Airflow DAGs |
| `sdd-execution-bridge.md` | **SignalBridge OMS + Execution** | MEXC/Binance via CCXT, paper/testnet/live toggle, kill switch, reconciliation |
| `sdd-risk-management.md` | **Risk Checks & Circuit Breakers** | 9-check chain, command pattern, RiskEnforcer 7 rules, kill switch audit |
| `sdd-observability.md` | **Monitoring & Alerting** | Prometheus 37+ rules, Grafana 4 dashboards, AlertManager Slack/PagerDuty |
| `sdd-cicd-testing.md` | **CI/CD & Quality Gates** | 9 GitHub Actions, Makefile 268 lines, 70% coverage gate, contract validation |

### SDD Contracts

**Python contracts** (`src/contracts/`):

| Contract | Exports |
|----------|---------|
| `strategy_schema.py` | `StrategyTrade`, `StrategyStats`, `safe_json_dump()`, `EXIT_REASONS` |
| `analysis_schema.py` | `MacroSnapshot`, `DailyAnalysisRecord`, `WeeklyAnalysisRecord`, `WeeklyViewExport` |
| `news_engine_schema.py` | `ArticleRecord`, `DigestRecord`, `FeatureSnapshotRecord`, `CrossReferenceRecord` |
| `signal_contract.py` | `UniversalSignalRecord` — strategy-agnostic signal format |
| `signal_adapters.py` | `H5SmartSimpleAdapter`, `H1ForecastVTAdapter`, `RLPPOAdapter` |
| `execution_strategies.py` | `WeeklyTPHSExecution`, `DailyTrailingStopExecution`, `IntradaySLTPExecution` |

**TypeScript contracts** (`usdcop-trading-dashboard/lib/contracts/`):

| Contract | Exports |
|----------|---------|
| `strategy.contract.ts` | `StrategyTrade`, `StrategyStats`, `StrategySummary`, `EXIT_REASON_COLORS` |
| `production-approval.contract.ts` | `ApprovalState`, `GateResult`, `ApproveRequest` |
| `weekly-analysis.contract.ts` | `WeeklyViewData`, `DailyAnalysisEntry`, `MacroVariableSnapshot`, `SignalSummaries` |
| `production-monitor.contract.ts` | Production monitoring types |
| `forecasting.contract.ts` | Forecasting dashboard types |
| `experiments.contract.ts` | Experiment tracking types |
| `backtest.contract.ts` | Backtest result types |
| `backtest-ssot.contract.ts` | Backtest SSOT configuration types |
| `model.contract.ts` | Model registry types |
| `ssot.contract.ts` | SSOT configuration types |

> **Approval workflow**: 2-vote system (Vote 1 auto via Python gates, Vote 2 human on `/dashboard`). See `sdd-approval-spec.md`.
> **Dashboard file layout**: `public/data/production/{summary,approval_state,trades/}.json`. See `sdd-dashboard-integration.md`.
> **Pipeline CLI**: `--phase backtest|production|both`, `--reset-approval`, `--no-png`. See `sdd-pipeline-lifecycle.md`.

---

## ARCHITECTURE (Do NOT deviate)

> Detailed file listings, schemas, and migration details are in the referenced `.claude/rules/*.md` specs.
> This section provides a navigation map — read the linked spec for full details.

### Configuration (SSOT)
- `config/pipeline_ssot.yaml` — Active RL config
- `config/macro_variables_ssot.yaml` — L0: 40 macro variable definitions
- `config/execution/smart_simple_v1.yaml` — H5 Smart Simple SSOT
- `config/execution/smart_executor_v1.yaml` — H1 Smart Executor SSOT
- `config/experiments/` — Frozen RL SSOT configs (baseline: `v215b_baseline.yaml`)

### Forecasting Pipeline
`src/forecasting/` — ForecastingEngine, 9 models, 21 features, walk-forward validation, experiment tracking.
Scripts: `generate_weekly_forecasts.py`, `run_forecast_experiment.py`, `build_forecasting_dataset_aligned.py`.

### H5 Weekly Pipeline (Smart Simple v1.1)
`src/forecasting/{confidence_scorer,adaptive_stops,vol_targeting}.py` — 3-tier confidence, vol-adaptive TP/HS.
Script: `train_and_export_smart_simple.py`. Migrations: 043-044. See `h5-smart-simple-pipeline.md`.

### H1 Daily Pipeline
`airflow/dags/forecast_h1_l3..l7*.py` — 5 DAGs (Sun train, Mon-Fri signal+execute+monitor).
Config: `config/execution/smart_executor_v1.yaml`.

### RL Pipeline (deprioritized)
Training: `src/{data,training}/` + `scripts/run_ssot_pipeline.py` (L2→L3→L4). See `l2-l3-l4-training-pipeline.md`.
Inference: `airflow/dags/{l1,l5}_*.py` + `src/core/contracts/feature_contract.py`. See `l1-l5-inference-pipeline.md`.

### News Engine
`src/news_engine/` — 5 adapters, enrichment pipeline, cross-reference, feature export (~60 features/day).
Migration: 045. See `news-and-analysis-sdd.md`.

### Analysis Module
`src/analysis/` — MacroAnalyzer (13 vars), LLMClient (Azure+Anthropic), WeeklyGenerator.
Script: `generate_weekly_analysis.py`. Migration: 046. See `news-and-analysis-sdd.md`.

### Dashboard (Next.js, 8 pages, 47 API routes)
Pages: `/`, `/hub`, `/dashboard`, `/production`, `/forecasting`, `/analysis`, `/execution`, `/login`.
API groups: production (6), analysis (4), execution (10+), experiments (6), backtest (4), models (3), trading (3), market (2), health, replay.
Components: `components/{forecasting,analysis,production}/`. Hooks: `useWeeklyAnalysis.ts`, `useLiveProduction.ts`.

### Execution Layer (OMS + Risk)
SignalBridge: `services/signalbridge_api/` — FastAPI + CCXT, MEXC/Binance adapters, WebSocket + Redis bridges.
Executors: `src/execution/{smart_executor,multiday_executor,trailing_stop,broker_adapter}.py`.
Risk: `src/risk/` (9-check chain + commands) + `src/trading/risk_enforcer.py` (7 rules).
See `sdd-execution-bridge.md` + `sdd-risk-management.md`.

### Infrastructure (25+ Docker services)
PostgreSQL+TimescaleDB (5432), Redis (6379), MinIO (9001), Airflow (8080), SignalBridge (8085),
Vault (8200), Prometheus (9090), Grafana (3002), AlertManager (9093), Loki (3100), Promtail, pgAdmin (5050), MLflow (5001).
See `sdd-observability.md`.

**Infra lista, activación en próximos pasos (status realista 2026-04-16):**
- **MinIO** — 11 buckets operativos pero actualmente solo usado por init-scripts (seed fallback).
  Bucket storage disponible para artefactos; los modelos se persisten en filesystem. **Roadmap**: migración modelos → MinIO.
- **MLflow** — Tracking server + SQLite + artifact store desplegados. Scripts ad-hoc loguean runs,
  DAGs L3 (H1/H5) no lo invocan. **Roadmap**: integración automatizada en `forecast_h1_l3_weekly_training` y `forecast_h5_l3_weekly_training`.
- **AlertManager** — Rules cargadas (53 alertas), Slack webhook vacío por defecto. Usar la UI estática
  para reglas; alertas activas requieren `SLACK_WEBHOOK_URL` en `.env`. **No crítico** si se opera en local.
- **Jaeger** — Deployado pero 0 servicios instrumentados (OpenTelemetry no implementado). Omitir del
  spec operacional hasta que se instrumenten servicios. **Roadmap**: bajo demanda.

### CI/CD & Testing
9 GitHub Actions: ci, deploy, security (x2), contracts-check, drift-check, dvc-validate, experiment, canary-promote.
Makefile: 268 lines (test, lint, docker, db, validate). 70% coverage gate. See `sdd-cicd-testing.md`.

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

## DAG SCHEDULE (27 DAGs + watchdog)

| Pipeline | DAGs | Key Timing (COT) | Spec |
|----------|------|-------------------|------|
| **H1 Daily** | 5 | Sun 01:00 train; Mon-Fri 13:00 signal, 13:30 vol-target, 13:35 executor, 19:00 monitor | `h5-smart-simple-pipeline.md` |
| **H5 Weekly** | 5 | Sun 01:30 train; Mon 08:15 signal, 08:45 vol-target; Mon-Fri */30 9-13 executor; Fri 14:30 monitor | `h5-smart-simple-pipeline.md` |
| **Forecasting Weekly** | 1 | **Mon 09:00 COT** (14:00 UTC) — `forecast_weekly_generation` regenerates dashboard CSV + 76 PNGs | `sdd-dashboard-integration.md` |
| **L0 Data** | 5 | OHLCV: */5 8-12 Mon-Fri; Macro: hourly 8-12 Mon-Fri; Backfill: Sun/Manual; Seed backup: daily 15:00 | `l0-data-governance.md` |
| **RL** | 6 | All manual/event-triggered except L1 (*/5 8-12 Mon-Fri) | `l1-l5-inference-pipeline.md` |
| **News+Analysis** | 5 | News: 3x/day (02,07,13 COT); Alert: */30; Weekly digest: Mon; Analysis L8: 14:00 Mon-Fri | `news-and-analysis-sdd.md` |
| **Watchdog** | 1 | `core_watchdog`: hourly 8-13 COT Mon-Fri; auto-heals stale data, forecasting, analysis | `elite-operations.md` |

> H1 and H5 retrain **WEEKLY** (every Sunday). Expanding window grows ~5 rows/week.
> Analysis DAG runs 2h after last news ingestion to ensure fresh articles.
> `forecast_weekly_generation` runs Mon 09:00 COT post-training; also triggered by watchdog auto-heal if stale.
> See `elite-operations.md` for collision-free timeline and `data-freshness-enforcement.md` for thresholds.

---

## EXPERIMENT RULES (MANDATORY)

1. **ONE variable per experiment** — never change action space + model + features simultaneously. Major variables: action space, architecture, features, reward, hyperparams, stops, sizing.
2. **5 seeds for RL** — [42, 123, 456, 789, 1337], no exceptions.
3. **Statistical validation** — RL: >=3/5 seeds+, CI excludes zero, PF>1.05. ML: p<0.05, DA>55%, Sharpe>1.0.
4. **Compare vs baselines** — buy-and-hold (-12.29% 2025), previous best, random agent (RL).
5. **Log everything** — append to `.claude/experiments/EXPERIMENT_LOG.md`.
6. **Eval reward ≠ OOS** (RL) — seed 456 eval=131 lost -20.6%, seed 1337 eval=111 gained +9.6%.
7. **CPU for PPO MlpPolicy** — RTX 3050 throttles; GPU only for RecurrentPPO.

> Experiment files: `EXPERIMENT_QUEUE.md`, `EXPERIMENT_LOG.md` in `.claude/experiments/`.
> Protocol: `.claude/rules/experiment-protocol.md`. Config versioning: `.claude/rules/ssot-versioning.md`.
> RL data splits: Train 2019-12→2024-12 (70K bars), Val 2025-01→06 (7K), Test 2025-07→12 (7K).

### Required Report Formats

```
RL:  | Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars | WR_Long% | WR_Short% |
ML:  | Strategy | Return% | Sharpe | p-value | DA% | WR% | PF | MaxDD% | Trades | $10K -> |
```
Plus: mean±std, bootstrap 95% CI, comparison vs buy-and-hold. If p>0.05: "NOT statistically significant".

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

| Track | Best Strategy | Return 2025 | Sharpe | p-value | $10K -> | Status |
|-------|---------------|-------------|--------|---------|---------|--------|
| **H5 Weekly** | **Smart Simple v2.0** | **+25.63%** | **3.35** | **0.006** | **$12,563** | **PRODUCTION** |
| H5 Weekly | v2.0 (2026 YTD) | +0.61% | -- | -- | $10,061 | Gate active (1 trade) |
| H1 Daily | Forecast+VT+Trailing | +36.84% | 3.135 | 0.0178 | $13,684 | PAUSED |
| RL | V21.5b | +2.51% | 0.321 | 0.272 | $10,251 | NOT significant |
| Baseline | Buy & Hold | -12.29% | -- | -- | $8,771 | -- |

**v1.1→v2.0** (2026-03-18): Added regime gate (Hurst), effective HS (3.5% portfolio cap),
XGBoost in ensemble, vol_regime_ratio + trend_slope_60d features, dynamic leverage, circuit breaker.
Weekly retraining restored (was monthly in v1.1.0 — methodology bug).
**10-agent audit** revealed R² < 0 in both years, model alpha negative vs Always SHORT.
Gate is the MVP: blocked 11/12 mean-reverting weeks in Q1 2026, converting -5.17% into +0.61%.
**RL baselines**: Buy-and-hold -14.66%, Random -4.12%, Bootstrap CI [-0.69%, +6.15%].
**RL history**: V20 failed, V21 failed, V21.5 superseded, V22 mixed (1/5), V21.5b best (4/5), EXP-ASYM-001 failed, EXP-HOURLY/DAILY failed.

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

### News Engine & Analysis
- Do NOT call LLM without checking budget limits ($1/day, $15/month)
- Do NOT skip LLM caching — always check file cache before calling API
- Do NOT use VADER as primary sentiment for GDELT articles — GDELT tone is more reliable
- Do NOT modify enrichment categories without updating both Python + TypeScript contracts
- Do NOT hardcode LLM provider — use strategy pattern (primary + fallback)
- Do NOT run `analysis_l8_daily_generation` before news ingestion DAGs complete
- Do NOT skip `_sanitize_for_json()` on analysis exports — same JSON safety as strategy exports

### Data & Infrastructure
- Do NOT store OHLCV timestamps in UTC — always America/Bogota
- Do NOT fetch BRL from TwelveData with `timezone=America/Bogota` (returns incomplete data, use UTC + convert)
- Do NOT compute features in L5 — L1 is the ONLY feature computation layer (RL)
- Do NOT write to `inference_ready_nrt` from outside L1 DAGs (RL)
- Do NOT train models on stale data — OHLCV must be <3 days, macro <7 days (enforced by `data_quality.py`)
- Do NOT skip DB migrations on fresh install — 043-046 are required for H5/News/Analysis
- Do NOT delete seed parquets or MACRO_DAILY_CLEAN.parquet — they are restore fallbacks
- Do NOT ignore model freshness warnings — they indicate L3 training failure

### Execution & Risk
- Do NOT place live orders without setting `EXECUTION_MODE=testnet` first — validate on testnet before going live
- Do NOT bypass risk checks (`RiskCheckChain`) — they are fail-safe (block on unavailability)
- Do NOT disable the kill switch in production — use `ResetKillSwitchCommand(confirmed=True)` with audit trail
- Do NOT hardcode exchange API keys — use Vault encryption (`AES-256-GCM`) via `VaultService`
- Do NOT skip the testnet validation week before switching to `EXECUTION_MODE=live`

### CI/CD & Monitoring
- Do NOT skip CI checks — 70% coverage gate is enforced, contract validation is mandatory
- Do NOT merge without green CI checks — security scan + contracts-check must pass
- Do NOT ignore critical alerts (PagerDuty) — they indicate service/model/data failures
- Do NOT modify Prometheus alert thresholds without backtest evidence
- Do NOT lower the coverage threshold below 70%
