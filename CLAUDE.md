# CLAUDE.md — USDCOP Trading System

> Loaded automatically by Claude Code at the start of every session.
> Contains project context, architecture, contracts, and rules that MUST be followed.

---

## PROJECT IDENTITY

**Goal**: Build a profitable USDCOP trading system using supervised forecasting.
**Exchange**: MEXC (0% maker fees, 1 bps slippage estimate)
**Architecture**: Spec-Driven Development (SDD) — specs define contracts, contracts enforce code.

**Four Tracks** (in priority order):

1. **H5 Weekly Pipeline (PRODUCTION, v11 FROZEN)**: Smart Simple v2.0, Ridge+BR + Regime Gate (XGBoost = experimento offline, `use_xgboost: false` — `smart_simple_v1.yaml:206`)
   - **2025 OOS oficial: +7.35%, Sharpe 0.942, p=0.2277 → NOT statistically significant** (32 trades, 2L/30S, MaxDD 7.84%, $10K → $10,735). Cascada de honestidad +26.05 → +13.05 → +7.66 → +7.35 (datos, purga, fills open-aware); los +25.63%/p=0.006 anteriores están **superseded** (`HYPOTHESIS-REGISTRY.md` § RE-MEDICIÓN #3). Fuente: bundle `public/data/production/summary_2025.json`.
   - **2026 forward (único juez limpio)**: paper ledger tal como lo corrieron los DAGs = **+0.66% YTD, 12 trades hasta 2026-W33** (`production/paper/candidates_ledger_2026.json`, 2026-08-28); replay 2026 con el método corregido = +3.36%, 11 trades (`summary.json`). **N<20 ⇒ solo conteo y PnL** (constitución §6). Corte A del protocolo de retiro (26 sem) ≈ **2026-09-16**.
   - Architecture: Regime Gate (Hurst) → Ridge/BR ensemble → Effective HS → DL → CB
   - DAGs: H5-L3/L4/L4b/L5/L6/L7 (see `.claude/specs/tracks/h5-smart-simple.md`)

2. **News Engine & Analysis Module (OPERATIONAL)**: AI-generated market analysis + news intelligence
   - LLM weekly analysis on disk: USD/COP W01-W27, Gold/BTC W01-W35 de 2026 (`public/data/analysis/**`; Azure OpenAI GPT-4o-mini)
   - Dashboard: `/analysis` page (componentes + rutas API: ver `.claude/generated/inventory.json`)
   - See `.claude/specs/tracks/news-analysis/_summary.md`

3. **H1 Daily Pipeline (PAUSED)**: 9 models, H=1 horizon — DAGs paused pending v2.0 validation; tres shadow (`forecast_h1_*shadow*`) con ledgers vacíos a la espera de decisión del operador

4. **RL (REJECTED as thesis, 2026-08-25)**: EXP-TESIS-RL-01 — PPO intradía bruto +27.95% pero neto **−54.87% en hold-out, DSR 0.000, `always_flat` gana; el alfa (0.67 pips/op) no cubre la comisión (break-even negativo)**. Solo sobrevive el piloto forward exploratorio RL-vs-LLM (`06-PRE-REGISTRATION.md`, `status: exploratory`). Ver `.claude/specs/planes/06-RESULTADOS.md`.

**Strategic Pivot (2026-03-18)**: 10-agent audit revealed Ridge/BR model has R² < 0 in both years.
Alpha comes from regime gate (knows when NOT to trade) + TP/HS mechanics, not from model predictions.
**2026 Regime**: Hurst = 0.28-0.49 (mean-reverting → transitioning). Gate correctly blocks most weeks.

**Methodology audit (2026-07-06)**: the p=0.006/p=0.0097 backtest p-values came from iterating on the
same OOS-2025 (42-cell grid, "#8 of 42"). **Trial-aware DSR of v11 = 0.50-0.92 < 0.95 in all scenarios**
— the 2025 backtest cannot prove edge after selection; **v11 is FROZEN and the 2026 forward is the only
clean judge** (`.claude/specs/assets/usdcop/{HYPOTHESIS-REGISTRY,WITHDRAWAL-PROTOCOL}.md`).
Anti-selection discipline is now transversal (`.claude/rules/quant-constitution.md`). Master plan:
`.claude/specs/audit/PLAN-completar-sistema-2026-07.md`.

**Current Best** (números de decisión = bundle publicado + registro de hipótesis, nunca de memoria):
- H5 Weekly v2.0 (`smart_simple_v11`): $10K → $10,735 (2025 OOS oficial), $10K → $10,066 (2026 forward paper, W33)
- Candidatas v12/v14: replay descriptivo + forward post-freeze (ventana juez 4 sem), +0.43% / +0.40% YTD — ninguna bate a v11
- RL: rechazado (ver track 4); V21.5b (+2.51%, Jan-2026) queda como historia

---

## QUICK START (First-Time Setup)

> See `.claude/specs/platform/mlops-lifecycle.md` for the full operator guide with checklists.

Arranque: `make compact` (uso diario) · `make compact-monitoring` (+observabilidad) ·
`make docker-up` (enterprise completo). Luego backfill L0 → forecasts → backtest → Vote 2 en
`/dashboard` → producción → los DAGs toman el ciclo semanal.

**La secuencia completa con checklists está en `.claude/specs/platform/mlops-lifecycle.md`**
(Stages 0-7). Los conteos de servicios NO se duplican aquí: la fuente es `docker-compose*.yml`
(la versión manual de esta tabla ya divergía del compose real).

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
| `approval-gates.md` | 2-vote approval (Vote 1 auto, Vote 2 human on `/dashboard`) + **6 gates** (incl. DSR trial-aware > 0.95) |
| `experiment-protocol.md` | Experiment discipline: 1 variable, 5 seeds, statistical validation |
| `quant-constitution.md` | **Transversal anti-selection discipline**: trials registry + DSR, mandatory baselines, look-ahead layers, withdrawal protocol; wins over specs/code/opinions |
| `rbac.md` | RBAC + monetización: deny-by-default, rol ≠ plan, Vote 2/kill = admin, paper-first |
| `strategy-engines.md` | Policy engines (`rule_based | ml | rl | composite`): one decision contract, DSL whitelist, `rule_trace` |
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
> **Dashboard file layout**: `public/data/production/{summary,trades/}.json` + estado de aprobación **privado** en `data/approvals/` (gates/DSR = `research:read`, CXD-057). See `dashboard-integration.md`, `approval-lifecycle.md` §4.
> **Pipeline CLI**: `--phase backtest|production|both`, `--reset-approval`, `--no-png`. See `mlops-lifecycle.md`.

---

## ARCHITECTURE (Do NOT deviate)

> Detailed file listings, schemas, and migration details are in the referenced `.claude/specs/**` reference specs.
> This section provides a navigation map — read the linked spec for full details.

### Configuration (SSOT)
- `config/pipeline_ssot.yaml` — Active RL config
- `config/macro_variables_ssot.yaml` — L0: 51 macro variable definitions (H1/H5 consume 4 vía `MACRO_DAILY_CLEAN`)
- `config/execution/smart_simple_v1.yaml` — H5 Smart Simple SSOT
- `config/execution/smart_executor_v1.yaml` — H1 Smart Executor SSOT
- `config/experiments/` — Frozen RL SSOT configs (baseline: `v215b_baseline.yaml`)

### Forecasting Pipeline
`src/forecasting/` — ForecastingEngine, 9 models, 21 features, walk-forward validation, experiment tracking.
Scripts: `scripts/pipeline/generate_weekly_forecasts.py`, `scripts/pipeline/run_forecast_experiment.py`, `scripts/data/build_forecasting_dataset_aligned.py`.
**`/forecasting` is multi-asset** (pair selector), branched by `analysis-assets.ts::forecast_mode`: **USD/COP + BTC = 9-model ML zoo** (CSV+PNG, whole-year via `generate_weekly_forecasts.py --asset <id> --num-weeks 30` → root for COP, `public/forecasting/btcusdt/` for BTC; `AssetModelZoo`); **Gold = rule-based weekly inference** (`generate_asset_weekly_forecast.py` → `public/forecasting/xauusd/weekly_inference_<year>.json`, `AssetWeeklyBody`). **BTC uses a BTC-appropriate 19-feature set** (17 price/technical/calendar + DXY + VIX; drops the Colombia-only WTI/EMBI; √365; config `config/assets/btcusdt_forecasting.yaml`) — same STRUCTURE as COP, honest features. BTC price-only DA ≈ 0.46 (a transparency surface, **not** an edge claim — quant-constitution). **Methodology (all pairs): trained ≤ Dec-2024, 2025 = backtest (OOS, default), 2026 = production.** See `dashboard-integration.md`.

### H5 Weekly Pipeline (Smart Simple v2.0, config `smart_simple_v1.yaml` version 2.0.0)
`src/forecasting/{confidence_scorer,adaptive_stops,vol_targeting}.py` — 3-tier confidence, vol-adaptive TP/HS.
Script: `scripts/pipeline/train_and_export_smart_simple.py`. Migrations: 043/044/049/054. See `h5-smart-simple.md`.

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
**24 páginas activas** (8 en `/legacy`) · **98 rutas API**
<!-- /inv -->

Lista de páginas y rutas API por grupo: **solo** en `.claude/generated/inventory.json` (`frontend.pages`, `frontend.api_routes`) — no se enumeran a mano aquí (la lista manual omitía `/admin` y sus 22 rutas).
Data flow: file-based BFF (`public/data/**`) + DB-live (`production/live`) + proxy (`INFERENCE_API_URL`) + SSE + WS; adaptive polling + graceful degradation. Contracts: `lib/contracts/*.ts` mirror `src/contracts/`.
**UI = GlobalMarkets Terminal (2026-07-10, CTR-GM-UI-001)**: chrome `components/gm/TerminalShell` + design system `components/gm/*` (tokens `lib/ui/gm-tokens.ts`, estados `AsyncBoundary`, hook `useGmQuery`); BFF contract CTR-FE-BE-001 (`lib/api/{envelope,relay,gm-client}.ts`, spec `frontend-backend-contract.md` + `docs/api/openapi.yaml`); páginas pre-GM archivadas en `/legacy/*` (admin-only). Migración/estado/gaps: `.claude/specs/platform/gm-terminal-migration.md`.
**Full as-built: `.claude/specs/platform/frontend-architecture.md`.** Data contract: `dashboard-integration.md`.

### Execution Layer (OMS + Risk)
SignalBridge: `services/signalbridge_api/` — FastAPI + CCXT, MEXC/Binance adapters, WebSocket + Redis bridges.
Executors: `src/execution/{smart_executor,multiday_executor,trailing_stop,broker_adapter}.py`.
Risk: `src/risk/` (9-check chain + commands) + `src/trading/risk_enforcer.py` (7 rules).
See `execution-bridge.md` + `risk-management.md`.

### Infrastructure (24 servicios en `docker-compose.yml`, 21 en el perfil compact)
PostgreSQL+TimescaleDB (5432), Redis (6379), MinIO (9001), Airflow (8080), SignalBridge (8085),
Vault (8200), Prometheus (9090), Grafana (3002), AlertManager (9093), Loki (3100), Promtail, pgAdmin (5050), MLflow (5001).
See `observability.md`.

**Infra desplegada pero parcialmente activada** (MinIO solo como fallback de seeds · MLflow
invocado por H5-L3 y H1-L3 · AlertManager necesita `SLACK_WEBHOOK_URL` · Jaeger/OTel
ya instrumentado en `services/common/tracing.py` + inference/signalbridge): detalle y roadmap en `observability.md`.

### CI/CD & Testing
<!-- inv:workflows -->
**13 GitHub Actions**
<!-- /inv -->

Includes: ci, deploy, security (x2), contracts-check, fabric-contracts, drift-check, dvc-validate, experiment,
canary-promote, rbac-gate, a11y, **specs-gate** (knowledge system: inventario, índices, front matter, enlaces, grafo Obsidian, skills).
Makefile (test, lint, docker, db, validate). 70% coverage gate. See `cicd-testing.md`.

### Data Sources (for local training without DB)
```
seeds/latest/                          (medido 2026-09-10; `manifest.json` es la fuente exacta)
├── usdcop_daily_ohlcv.parquet (+_full) <- Daily COP 2019-12 -> 2026-08-28 (1.7K rows) [H1/H5]; `_full` = tramo 2015-2019 (1.2K)
├── usdcop_m5_ohlcv.parquet            <- 5-min COP (100K rows, 2019-12 -> 2026-08-24, COT tz) [RL]; también `usdcop_1h`
├── usdmxn_m5 / usdbrl_m5              <- 5-min MXN/BRL (4.7K rows each, 2026-03 -> 2026-07 ONLY)
├── xauusd_* / btcusdt_* / spx500_*    <- Gold daily 2004-> (5.9K) + m5; BTC daily 2017-> (3.3K) + m5 + derivatives_daily (2.5K, funding); SPX daily 1995-> (8.0K)
├── fx_multi_m5_ohlcv.parquet          <- Unified 5-symbol m5 seed (144K rows: COP/XAU/BTC/MXN/BRL, DB restore)
└── macro_indicators_daily.parquet     <- Macro ALL 41 cols (10.9K rows)

data/pipeline/04_cleaning/output/
└── MACRO_DAILY_CLEAN.parquet          <- Macro CLEAN 28 cols, 26K rows 1954-07 -> 2026-08-24 (H1/H5 reads THIS for 4 macro features)
```
> All OHLCV seeds are in **America/Bogota timezone**, session 8:00-12:55 COT, Mon-Fri.
> Regenerate with: `python scripts/data/build_unified_fx_seed.py`
> See `.claude/rules/data-governance.md` for timezone rules and BRL API quirk.

**Git-tracking policy (operator directive 2026-07-09, peso revisado 2026-08-24)**: se versiona lo
restore-critical **y todo lo que sirve el dashboard** — `seeds/latest/*`, `data/backups/{seeds,features}/*`
+ `*.csv.gz`, `data/pipeline/04_cleaning/output/*`, `public/data/**` y `public/forecasting/**`.
**Un clon limpio debe renderizar todas las páginas; los DAGs son la vía de refresco, no la de bootstrap.**
Gitignorado lo regenerable/runtime (`data/{cache,news,forecasting}/`, `data/pipeline/{00..03,05..07}/`,
`models/**`, `results/`, `outputs/`, `deploy_status.json` — este último también fuera del build context
de Docker: su modo NTFS rompe el tar). Para que ese peso no crezca sin motivo: los generadores de PNG
llaman a `enable_deterministic_png()` (`src/utils/plot_determinism.py`) y LFS cubre `data/backups/**`.
Inventario exacto, cifras del pack y DO-NOTs: `dashboard-integration.md` · guard `test_png_determinism.py`.

---

## DAG SCHEDULE

<!-- inv:dags -->
**55 DAGs** (52 declarados en 53 módulos + 3 generados por factory)
<!-- /inv -->

| Pipeline | DAGs | Key Timing (COT) | Spec |
|----------|------|-------------------|------|
| **H1 Daily** | 6 + 3 shadow | Sun 01:00 train; Mon-Fri 13:00 signal, 13:30 vol-target, 13:35 executor, 19:00 monitor (PAUSED); shadow `forecast_h1_daily_shadow_v1` / `regime_shadow_v2` (`regime_shadow` v1 DEPRECATED) | `h5-smart-simple.md` |
| **H5 Weekly** | 7 | Sun 01:30 train; Mon 08:15 signal, 08:45 vol-target (+tenant fan-out); Mon-Fri */30 08:00-12:55 executor (`*/30 13-17` UTC); Fri 14:30 monitor; event-driven: L4 backtest-promotion (Vote 1) + **L4b production-deploy** (post-Vote-2, dashboard→Airflow REST, 2026-07-07) | `h5-smart-simple.md` |
| **Asset DS-cycle** | 3 | Sun 01:45 Gold (`asset_xauusd_pipeline_weekly`), 02:00 BTC (`asset_btcusdt_pipeline_weekly`), **SPX500 Mon-Fri 07:30** (`asset_spx500_pipeline_weekly`, cron `30 7 * * 1-5`): l0_ingest→l0b_export_chart_ohlcv→l4_backtest_publish→l5_weekly_forecast→l6_verify_registry. Factory from `config/assets/pipelines.yaml` (CTR-ASSET-PIPELINE-001); COP keeps bespoke H5 chain | `architecture-overview.md` |
| **Forecasting** | 3 | **Mon 09:00 COT** `forecast_weekly_generation` (USD/COP CSV + PNGs, `--num-weeks 30`, ~30-45 min); `forecast_asset_analysis_weekly` Mon 14:20 UTC; `forecast_l3_01_model_training` monthly | `dashboard-integration.md` |
| **L0 Data** | 7 | OHLCV: */5 8-12 Mon-Fri; Macro: hourly 8-12 Mon-Fri (+ `core_l0_04_usdcop_forward_macro_pit`); Backfill: Sun/Manual; Seed backup: Mon-Fri 15:00 (`0 20 * * 1-5`, no weekend); `l0_multiframe_catchup` hh:20 | `data-governance.md` |
| **RL** | 12 (9 activos + 3 `rl_l4_01/02/03` DEPRECATED) | All manual/event-triggered except L1 (*/5 8-12 Mon-Fri) | `inference-l1-l5.md` |
| **News+Analysis** | 5 | News: 3x/day (02,07,13 COT); Alert: */30; Weekly digest: Mon; maintenance; Analysis L8: 14:00 Mon-Fri | `news-analysis/_summary.md` |
| **Ops/Control** | 7 | `core_watchdog` hourly 8-13 COT Mon-Fri (auto-heal); `core_l6_01_alert_monitor`, `core_l6_02_weekly_report`, `control_system_health`, `reconciliation_daily`, `rbac_entitlements_daily`, `forward_ledger_weekly` (Fri 19:00 UTC, juez forward) | `elite-operations.md` |
| **Research (tesis)** | 2 | `research_thesis_ppo_training` (manual), `research_forward_arms` (12:15 UTC Mon-Fri, brazos RL-vs-LLM exploratorios) | `planes/06-PRE-REGISTRATION.md` |

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
- Do NOT skip DB migrations on fresh install — 043-046 (H5/News/Analysis), 049/054 (H5), 055/056 (RBAC) are required; the series now reaches 083

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
