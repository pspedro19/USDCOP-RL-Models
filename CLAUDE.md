# CLAUDE.md — USDCOP Trading System

> Loaded automatically by Claude Code at the start of every session.
> Contains project context, architecture, contracts, and rules that MUST be followed.

---

## PROJECT IDENTITY

**Goal**: Build a profitable USDCOP trading system using supervised forecasting.
**Exchange**: MEXC (0% maker fees, 1 bps slippage estimate)
**Architecture**: Spec-Driven Development (SDD) — specs define contracts, contracts enforce code.

**Three Tracks** (in priority order):

1. **H5 Weekly Pipeline (PRODUCTION)**: Smart Simple v2.0, Ridge+BR + Regime Gate (XGBoost = experimento offline, no promovido — `smart_simple_v1.yaml:196`)
   - **+25.63%, Sharpe 3.35, p=0.006** (2025 backtest, 34 trades)
   - **+0.61%** (2026 YTD, 1/1 wins — regime gate blocked 11 of 12 mean-reverting weeks)
   - Architecture: Regime Gate (Hurst) → Ridge/BR/XGB ensemble → Effective HS → DL → CB
   - DAGs: H5-L3/L4/L5/L6/L7 (see `.claude/specs/tracks/h5-smart-simple.md`)

2. **News Engine & Analysis Module (OPERATIONAL)**: AI-generated market analysis + news intelligence
   - Active sources: Investing.com (78 articles), Portafolio (276 articles) in DB
   - LLM weekly analysis: W01-W15 generated (Azure OpenAI GPT-4o-mini)
   - Dashboard: `/analysis` page (componentes + rutas API: ver `.claude/generated/inventory.json`)
   - See `.claude/specs/tracks/news-analysis/_summary.md`

3. **H1 Daily Pipeline (PAUSED)**: 9 models, H=1 horizon — DAGs paused pending v2.0 validation

4. **RL (DEPRIORITIZED)**: PPO agent, 5-min bars, NOT significant (p=0.272)

**Strategic Pivot (2026-03-18)**: 10-agent audit revealed Ridge/BR model has R² < 0 in both years.
Alpha comes from regime gate (knows when NOT to trade) + TP/HS mechanics, not from model predictions.
**2026 Regime**: Hurst = 0.28-0.49 (mean-reverting → transitioning). Gate correctly blocks most weeks.

**Methodology audit (2026-07-06)**: the p=0.006/p=0.0097 backtest p-values came from iterating on the
same OOS-2025 (42-cell grid, "#8 of 42"). **Trial-aware DSR of v11 = 0.50-0.92 < 0.95 in all scenarios**
— the 2025 backtest cannot prove edge after selection; **v11 is FROZEN and the 2026 forward is the only
clean judge** (`.claude/specs/assets/usdcop/{HYPOTHESIS-REGISTRY,WITHDRAWAL-PROTOCOL}.md`).
Anti-selection discipline is now transversal (`.claude/rules/quant-constitution.md`). Master plan:
`.claude/specs/audit/PLAN-completar-sistema-2026-07.md`.

**Current Best**:
- H5 Weekly v2.0: Ridge+Gate+EffectiveHS → $10K → $12,563 (2025), $10K → $10,061 (2026 YTD)
- Momentum v3.0 (paper): $10K → $10,353 (2026 YTD, better in mean-reverting but worse in trending)
- RL: V21.5b — +2.51% mean (4/5 seeds), NOT significant

---

## QUICK START (First-Time Setup)

> See `.claude/specs/platform/mlops-lifecycle.md` for the full operator guide with checklists.

Arranque: `make compact` (uso diario) · `make compact-monitoring` (+observabilidad) ·
`make docker-up` (enterprise completo). Luego backfill L0 → forecasts → backtest → Vote 2 en
`/dashboard` → producción → los DAGs toman el ciclo semanal.

**Los conteos de servicios y la secuencia completa con checklists están en
`.claude/specs/platform/mlops-lifecycle.md`** (Stages 0-7). No se duplican aquí: la versión
manual de esta tabla ya divergía del compose real.

---

## DESIGN PATTERNS & BEST PRACTICES

| Pattern | Where Applied | Rule |
|---------|---------------|------|
| **SSOT** (Single Source of Truth) | Config YAMLs, feature contracts | ONE authoritative source per concern; never duplicate config |
| **SDD** (Spec-Driven Development) | Dashboard integration, approval flow | Specs (`.claude/rules/` + `.claude/specs/`) -> Contracts (TS/Python types) -> Implementation |
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
Layer 1: SPEC (defines what)      -> .claude/rules/ (auto-loaded) + .claude/specs/ (on-demand)
Layer 2: CONTRACT (enforces how)  -> lib/contracts/ + src/contracts/
Layer 3: IMPLEMENTATION           -> scripts/, pages, DAGs             (conform to contracts)
```

> **Spec layout** (see `.claude/README.md`): `rules/*.md` is auto-loaded every session (thin,
> always-true rules); `specs/**` is on-demand reference. Rule index: `.claude/rules/00-INDEX.md`.

### Always-loaded rules (`.claude/rules/`)

| Rule | Governs |
|------|---------|
| `data-governance.md` | L0 OHLCV + macro governance, timezone golden rule (America/Bogota) |
| `data-freshness.md` | Freshness thresholds (OHLCV 3d/macro 7d/models 10d) + recovery **(SSOT)** |
| `strategy-contract.md` | Universal strategy/trade/gate schemas, StrategyRegistry, exit reasons, signal contract |
| `approval-gates.md` | 2-vote approval (Vote 1 auto, Vote 2 human on `/dashboard`) + 5 gates |
| `experiment-protocol.md` | Experiment discipline: 1 variable, 5 seeds, statistical validation |
| `ssot-versioning.md` | Frozen experiment SSOT configs + versioning lifecycle |

### Reference specs (`.claude/specs/`, on-demand)

| Spec | Purpose |
|------|---------|
| `architecture-overview.md` | As-built architecture map (infra, DAGs, contracts + drift, COP/Gold coupling) |
| `platform/mlops-lifecycle.md` | **Master lifecycle** (bootstrap→production): 8 stages, CLI `--phase`, operator guide |
| `platform/frontend-architecture.md` | **Dashboard as-built** (Next.js app: routing, BFF API layer, data-flow/dynamism, contracts boundary, scalability + roadmap) |
| `platform/dashboard-integration.md` | Python→Dashboard **data contract** (JSON/CSV/PNG conventions, strategy selector) |
| `platform/registry-lifecycle.md` | Dynamic multi-strategy registry + replay (manifest, immutable bundles, TDD R1–R9) |
| `platform/execution-bridge.md` | SignalBridge OMS + Execution (MEXC/Binance CCXT, kill switch) |
| `platform/risk-management.md` | Risk checks & circuit breakers (9-check chain, RiskEnforcer, kill-switch audit) |
| `platform/observability.md` | Monitoring & alerting (Prometheus/Grafana/AlertManager/Loki) |
| `platform/cicd-testing.md` | CI/CD & quality gates (GitHub Actions, Makefile, 70% coverage gate) |
| `platform/codex-review-integration.md` | **Codex como revisor independiente** (Claude piloto): CLI 0.144.6 as-built, config validada con `--strict-config`, perfil `audit` con deny-read de secretos, loop de auditoría |
| `platform/authentication.md` | Authentication & user creation as-built (SignalBridge JWT/bcrypt/lockout, dashboard NextAuth, `sb_users`) |
| `pipelines/{training-l2-l3-l4,inference-l1-l5}.md` | RL training + inference pipeline internals |
| `operations/elite-operations.md` | **DAG schedule / collision-free timeline (SSOT)** + recovery playbooks |
| `tracks/h5-smart-simple.md` | H5 weekly production track (COP) |
| `tracks/news-analysis/` | News Engine + Analysis package (`_summary.md` + `NN_*.md` detail) |
| `assets/_onboarding-playbook.md` | Add a tradeable asset (AssetProfile, file-touch map, TDD A1–F1) |
| `assets/_asbuilt-implementation.md` | Multi-asset AS-BUILT (per-asset **session/timezone/annualization**; COP prod + Gold & BTC onboarded to web) |
| `assets/_strategy-science.md` | **How the rule-based strategies work** (intent×vol-target×regime, variables, anti-leakage, evaluation, **results per pair**, what's created per new strategy) |
| `assets/_ds-cycle-asbuilt.md` | **DS-cycle AS-BUILT per asset × strategy**: data lineage + descriptive stats, transformations, what entered training, **2025 backtest-replay + forecasting/inference results**, news per asset, **conclusions**, results→UI mapping |
| `assets/xauusd/` | Gold spec package (SPEC-00..12, ADR, roadmap, status) |
| `assets/btcusdt/` | **BTC/USDT crypto** spec package (24/7, exposure engine; `SPEC-13` integration + `design/`, roadmap, status) |
| `audit/AUDIT-2026-07-remediation.md` | 10-agent code↔spec audit → tasks-to-fix (~114 findings, P0/P1/P2 backlog) |
| `audit/STRATEGIC-ASSESSMENT-2026-07.md` | **Honest state-of-system + forward plan**: infra>signal meta-problem, edge=risk-control (not prediction), data/model/orchestration/FE-BE gaps, **BTC deep-dive** (price-only ceiling → crypto-native data is the unblock), P1-P6 priorities |

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

> **Approval workflow**: 2-vote system (Vote 1 auto via Python gates, Vote 2 human on `/dashboard`). See `approval-gates.md`.
> **Dashboard file layout**: `public/data/production/{summary,approval_state,trades/}.json`. See `dashboard-integration.md`.
> **Pipeline CLI**: `--phase backtest|production|both`, `--reset-approval`, `--no-png`. See `mlops-lifecycle.md`.

---

## ARCHITECTURE (Do NOT deviate)

> Detailed file listings, schemas, and migration details are in the referenced `.claude/specs/**` reference specs.
> This section provides a navigation map — read the linked spec for full details.

### Configuration (SSOT)
- `config/pipeline_ssot.yaml` — Active RL config
- `config/macro_variables_ssot.yaml` — L0: 40 macro variable definitions
- `config/execution/smart_simple_v1.yaml` — H5 Smart Simple SSOT
- `config/execution/smart_executor_v1.yaml` — H1 Smart Executor SSOT
- `config/experiments/` — Frozen RL SSOT configs (baseline: `v215b_baseline.yaml`)

### Forecasting Pipeline
`src/forecasting/` — ForecastingEngine, 9 models, 21 features, walk-forward validation, experiment tracking.
Scripts: `scripts/pipeline/generate_weekly_forecasts.py`, `scripts/pipeline/run_forecast_experiment.py`, `scripts/data/build_forecasting_dataset_aligned.py`.
**`/forecasting` is multi-asset** (pair selector), branched by `analysis-assets.ts::forecast_mode`: **USD/COP + BTC = 9-model ML zoo** (CSV+PNG, whole-year via `generate_weekly_forecasts.py --asset <id> --num-weeks 30` → root for COP, `public/forecasting/btcusdt/` for BTC; `AssetModelZoo`); **Gold = rule-based weekly inference** (`generate_asset_weekly_forecast.py` → `public/forecasting/xauusd/weekly_inference_<year>.json`, `AssetWeeklyBody`). **BTC uses a BTC-appropriate 19-feature set** (17 price/technical/calendar + DXY + VIX; drops the Colombia-only WTI/EMBI; √365; config `config/assets/btcusdt_forecasting.yaml`) — same STRUCTURE as COP, honest features. BTC price-only DA ≈ 0.46 (a transparency surface, **not** an edge claim — quant-constitution). **Methodology (all pairs): trained ≤ Dec-2024, 2025 = backtest (OOS, default), 2026 = production.** See `dashboard-integration.md`.

### H5 Weekly Pipeline (Smart Simple v1.1)
`src/forecasting/{confidence_scorer,adaptive_stops,vol_targeting}.py` — 3-tier confidence, vol-adaptive TP/HS.
Script: `scripts/pipeline/train_and_export_smart_simple.py`. Migrations: 043-044. See `h5-smart-simple.md`.

### H1 Daily Pipeline
`airflow/dags/forecast_h1_l3..l7*.py` (Sun train, Mon-Fri signal+execute+monitor).
Config: `config/execution/smart_executor_v1.yaml`.

### RL Pipeline (deprioritized)
Training: `src/{data,training}/` + `scripts/pipeline/run_ssot_pipeline.py` (L2→L3→L4). See `training-l2-l3-l4.md`.
Inference: `airflow/dags/{l1,l5}_*.py` + `src/core/contracts/feature_contract.py`. See `inference-l1-l5.md`.

### News Engine
`src/news_engine/` — 5 adapters, enrichment pipeline, cross-reference, feature export (~60 features/day).
Migration: 045. See `news-analysis/_summary.md`.

### Analysis Module
`src/analysis/` — MacroAnalyzer (13 vars), LLMClient (Azure+Anthropic), WeeklyGenerator.
Script: `scripts/pipeline/generate_weekly_analysis.py`. Migration: 046. See `news-analysis/_summary.md`.

### Scripts (`scripts/`, purpose-based layout — reorganized 2026-07)
`scripts/` is an importable package (`from scripts.<subdir>.<mod> import …`); every script lives in a purpose subdir, root holds only `__init__.py`.

| Subdir | Holds |
|--------|-------|
| `pipeline/` | production/RL/forecasting/gold entrypoints, promote, registry (`train_and_export_smart_simple`, `generate_weekly_forecasts`, `run_ssot_pipeline`, `export_to_onnx`…) |
| `data/` | seeds, features, datasets, resampling, scrapers, DB seeding |
| `ops/` | `db_migrate`, backfill, `backup/`, rotate, one-off migrations, DVC/maintenance shells |
| `analysis/` | research backtests, sharpe decomp, sensitivity, ablation |
| `diagnostics/` | `diagnose_`/`verify_`/`trace_`/`preflight_` |
| `validation/` | reusable health/smoke/gate validators |
| `presentation/` | pptx/pitch/screenshots/diagrams |
| `tools/` | misc CLI utilities + test/experiment runners |
| `lib/` | shared importable modules (`vol_target_backtest`) |
| `sql/` · `migrations/` · `deployment/` · `demo/` · `vault/` | SQL queries · schema migrations · deploy · demo · vault |
| `archive/` | legacy/superseded/dups + completed-milestone gates (see `scripts/archive/README.md`) |

**Add a script**: place it in the subdir matching its purpose — NEVER at `scripts/` root (a regression test enforces this: `tests/regression/test_scripts_layout.py`). If it's wired into a DAG / Makefile / `dvc.yaml` / deploy-manifest, that path is **load-bearing** — update the reference when moving.

### Dashboard (Next.js 15 App Router)

<!-- inv:frontend -->
**22 páginas activas** (8 en `/legacy`) · **93 rutas API**
<!-- /inv -->

Pages (8 sections + 5 `/execution` sub-pages): `/`, `/hub`, `/dashboard`, `/production`, `/forecasting`, `/analysis`, `/execution/*`, `/login`.
API groups: execution (13), experiments (7), production (6), backtest (5), analysis (4), trading (3), registry (2), models (2), market (2), strategies, replay, pipeline, health, auth.
Data flow: file-based BFF (`public/data/**`) + DB-live (`production/live`) + proxy (`INFERENCE_API_URL`) + SSE + WS; adaptive polling + graceful degradation. Contracts: `lib/contracts/*.ts` mirror `src/contracts/`.
**UI = GlobalMarkets Terminal (2026-07-10, CTR-GM-UI-001)**: chrome `components/gm/TerminalShell` + design system `components/gm/*` (tokens `lib/ui/gm-tokens.ts`, estados `AsyncBoundary`, hook `useGmQuery`); BFF contract CTR-FE-BE-001 (`lib/api/{envelope,relay,gm-client}.ts`, spec `frontend-backend-contract.md` + `docs/api/openapi.yaml`); páginas pre-GM archivadas en `/legacy/*` (admin-only). Migración/estado/gaps: `.claude/specs/platform/gm-terminal-migration.md`.
**Full as-built: `.claude/specs/platform/frontend-architecture.md`.** Data contract: `dashboard-integration.md`.

### Execution Layer (OMS + Risk)
SignalBridge: `services/signalbridge_api/` — FastAPI + CCXT, MEXC/Binance adapters, WebSocket + Redis bridges.
Executors: `src/execution/{smart_executor,multiday_executor,trailing_stop,broker_adapter}.py`.
Risk: `src/risk/` (9-check chain + commands) + `src/trading/risk_enforcer.py` (7 rules).
See `execution-bridge.md` + `risk-management.md`.

### Infrastructure (25+ Docker services)
PostgreSQL+TimescaleDB (5432), Redis (6379), MinIO (9001), Airflow (8080), SignalBridge (8085),
Vault (8200), Prometheus (9090), Grafana (3002), AlertManager (9093), Loki (3100), Promtail, pgAdmin (5050), MLflow (5001).
See `observability.md`.

**Infra desplegada pero parcialmente activada** (MinIO solo como fallback de seeds · MLflow
invocado por H5-L3, pendiente en H1-L3 · AlertManager necesita `SLACK_WEBHOOK_URL` · Jaeger/OTel
ya instrumentado): detalle y roadmap en `observability.md`.

### CI/CD & Testing
<!-- inv:workflows -->
**12 GitHub Actions**
<!-- /inv -->

Includes: ci, deploy, security (x2), contracts-check, drift-check, dvc-validate, experiment,
canary-promote, rbac-gate, a11y.
Makefile: 268 lines (test, lint, docker, db, validate). 70% coverage gate. See `cicd-testing.md`.

### Data Sources (for local training without DB)
```
seeds/latest/
├── usdcop_daily_ohlcv.parquet      <- Daily COP OHLCV (~3K rows, 2015 -> 2026, COT tz) [H1/H5 training]
├── usdcop_m5_ohlcv.parquet        <- 5-min COP (81K rows, 2019-12 -> 2026-01, COT tz) [RL training]
├── usdmxn_m5_ohlcv.parquet        <- 5-min MXN (2.3K rows, 2026-03 -> 2026-07 ONLY)
├── usdbrl_m5_ohlcv.parquet        <- 5-min BRL (2.3K rows, 2026-03 -> 2026-07 ONLY)
├── fx_multi_m5_ohlcv.parquet      <- Unified 3-pair seed (266K rows, for DB restore)
└── macro_indicators_daily.parquet <- Macro ALL 41 cols (10K rows, 1954 -> 2026)

data/pipeline/04_cleaning/output/
└── MACRO_DAILY_CLEAN.parquet      <- Macro CLEAN 17 cols (H1/H5 reads THIS for 4 macro features)
```
> All OHLCV seeds are in **America/Bogota timezone**, session 8:00-12:55 COT, Mon-Fri.
> Regenerate with: `python scripts/data/build_unified_fx_seed.py`
> See `.claude/rules/data-governance.md` for timezone rules and BRL API quirk.

**Git-tracking policy (updated 2026-07-09, operator directive)**: restore-critical data AND everything
the dashboard serves are tracked — `seeds/latest/*`, `data/backups/seeds/*` + `data/backups/*.csv.gz`
(startup DB restore), `data/backups/features/*` (news/analysis/H5/asset table dumps — news history is
NOT regenerable), `data/pipeline/04_cleaning/output/*` (MACRO_DAILY_CLEAN + 9 MASTER files), and the
dashboard's `public/data/**` (strategy bundles, production approval state, market daily JSONs, analysis)
+ `public/forecasting/**` (weekly inference JSONs + forward PNGs + CSV). **A fresh clone must render
every dashboard page; DAGs/watchdog are the refresh path, not the bootstrap path.** Still gitignored
(truly regenerable/runtime): `data/{cache,news,forecasting}/`, `data/pipeline/{00,01,02,03,05,06,07}/`,
`data/backups/{full_backup_*,pre_v20_*}/`, `models/**` binaries, `results/`, `outputs/`,
`video-pitch/{out,public}/`, and `public/data/production/deploy_status.json` (container-written runtime
state — must also stay OUT of the docker build context: its NTFS mode breaks `docker build` tar).

---

## DAG SCHEDULE

<!-- inv:dags -->
**46 DAGs** (43 declarados en 43 módulos + 3 generados por factory)
<!-- /inv -->


| Pipeline | DAGs | Key Timing (COT) | Spec |
|----------|------|-------------------|------|
| **H1 Daily** | 5 | Sun 01:00 train; Mon-Fri 13:00 signal, 13:30 vol-target, 13:35 executor, 19:00 monitor | `h5-smart-simple.md` |
| **H5 Weekly** | 7 | Sun 01:30 train; Mon 08:15 signal, 08:45 vol-target (+tenant fan-out); Mon-Fri */30 08:00-12:55 executor (`*/30 13-17` UTC); Fri 14:30 monitor; event-driven: L4 backtest-promotion (Vote 1) + **L4b production-deploy** (post-Vote-2, dashboard→Airflow REST, 2026-07-07) | `h5-smart-simple.md` |
| **Asset DS-cycle** | 2 | Sun 01:45 Gold (`asset_xauusd_pipeline_weekly`), 02:00 BTC (`asset_btcusdt_pipeline_weekly`): l0_ingest→l0b_export_chart_ohlcv→l4_backtest_publish→l5_weekly_forecast→l6_verify_registry. Factory from `config/assets/pipelines.yaml` (CTR-ASSET-PIPELINE-001); Gold/BTC DAG-driven (incl. `/forecasting` weekly inference), COP keeps bespoke H5 chain | `architecture-overview.md` |
| **Forecasting Weekly** | 1 | **Mon 09:00 COT** (14:00 UTC) — `forecast_weekly_generation` regenerates USD/COP dashboard CSV + PNGs, whole-year `--num-weeks 30` (~30-45 min) | `dashboard-integration.md` |
| **L0 Data** | 5 | OHLCV: */5 8-12 Mon-Fri; Macro: hourly 8-12 Mon-Fri; Backfill: Sun/Manual; Seed backup: Mon-Fri 15:00 (`0 20 * * 1-5`, no weekend) | `data-governance.md` |
| **RL** | 6 | All manual/event-triggered except L1 (*/5 8-12 Mon-Fri) | `inference-l1-l5.md` |
| **News+Analysis** | 5 | News: 3x/day (02,07,13 COT); Alert: */30; Weekly digest: Mon; Analysis L8: 14:00 Mon-Fri | `news-analysis/_summary.md` |
| **Watchdog** | 1 | `core_watchdog`: hourly 8-13 COT Mon-Fri; auto-heals stale data, forecasting, analysis | `elite-operations.md` |

> H1 and H5 retrain **WEEKLY** (every Sunday). Expanding window grows ~5 rows/week.
> Analysis DAG runs 2h after last news ingestion to ensure fresh articles.
> `forecast_weekly_generation` runs Mon 09:00 COT post-training; also triggered by watchdog auto-heal if stale.
> See `elite-operations.md` for collision-free timeline and `data-freshness.md` for thresholds.

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

## KNOWN BUGS · VERSION HISTORY

Ambos son **referencia**, no reglas de cada sesión → `.claude/specs/platform/known-issues-and-history.md`
(12 bugs ya corregidos + historial de resultados por track + advertencia de DSR sobre 2025).

## DO NOT

> **JSON/dashboard, aprobación, datos L0 y frescura**: sus DO-NOTs viven en las rules
> auto-cargadas (`strategy-contract.md`, `approval-gates.md`, `data-governance.md`,
> `data-freshness.md`). No se duplican aquí.

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

### Data & Infrastructure (solo lo NO cubierto por `data-governance.md` / `data-freshness.md`)
- Do NOT compute features in L5 — L1 is the ONLY feature computation layer (RL)
- Do NOT write to `inference_ready_nrt` from outside L1 DAGs (RL)
- Do NOT skip DB migrations on fresh install — 043-046 are required for H5/News/Analysis

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
