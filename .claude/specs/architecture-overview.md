# SDD Spec: Architecture Overview & Implementation Map

> **Responsibility**: Single entry-point that describes HOW the USDCOP trading system is
> actually infrastructured and implemented on disk (verified 2026-07-03), so any engineer
> — or a new-asset onboarding effort — can orient before touching code.
> This spec is descriptive (the *as-built* map). For prescriptive rules, follow the
> cross-referenced specs. For onboarding a NEW asset (BTC, Gold), see
> `_onboarding-playbook.md`.
>
> Contract: CTR-ARCH-001
> Version: 1.1.0
> Date: 2026-07-03 (updated 2026-07-05: repo reorg — see §2.1)
> Verified against disk (not docs) via 4-agent audit.

---

## 0. The One-Paragraph Model

The system is a **Spec-Driven, multi-layer MLOps monolith** for trading **USD/COP**.
Data (L0) → features (L1/L2) → models (L3) → backtest+approval (L4/Stage 4-5) →
signal+execution (L5/L7) → monitoring (L6). Three *strategy tracks* share this spine:
**H5 Weekly (PRODUCTION)**, **H1 Daily (PAUSED)**, **RL PPO (DEPRIORITIZED)**. Orchestration
is Airflow (40 DAGs); state is PostgreSQL/TimescaleDB; artifacts are filesystem + MinIO;
the operator surface is a Next.js dashboard (13 pages) that reads file-based JSON/CSV
artifacts and a few live APIs. Everything the pipeline promises is enforced by **contracts**
(6 Python + 10 TypeScript) whose job is to keep training, inference, and the UI in agreement.

**Load-bearing truth (updated 2026-07-05)**: the system is now genuinely **multi-asset for
offline backtest + web display**. The `AssetProfile` keystone (`src/contracts/asset_profile.py` +
`config/assets/{usdcop,xauusd,btcusdt}.yaml`) parameterizes symbol / chart_symbol / **session /
timezone / bars_per_year** / price_range / macro drivers / regime thresholds. A **second asset, Gold
(XAU/USD)**, is fully onboarded: ingested (5-min into `usdcop_m5_ohlcv`, daily into new
`asset_daily_ohlcv`), scored (`src/gold_rl/`), backtested with gates, published to the dynamic
`registry.json` (2 assets, 5 strategies), and **visible + replayable on the dashboard** as `XAUUSD`
— COP untouched. A **third asset, Bitcoin (BTC/USDT)**, is now **onboarded end-to-end and visible on the web**:
AssetProfile (crypto, **24/7 UTC, √365**), crypto-native tables (migration **052**), **canonical
data ingested from Binance's public API (no key needed)** — daily 3,245 bars 2017→2026, UTC 00:00
close — a spot-exposure science stack (`src/btc_strategy/`, `exposure∈[0,1]`), and **3 backtested
bundles published to the registry** (B2 trend-follower: +351%, Sharpe 1.40, Calmar 1.83, MaxDD −11%,
PROMOTE). The dashboard builds a "Bitcoin" selector as `BTCUSDT` with per-version replay. The
regime-gated engine (S3) does not yet beat the baseline — it needs the on-chain HMM fed by the
still-pending crypto-native extractors (BGeometrics/funding/Farside → tables 052). Spec package +
integration bridge: `assets/btcusdt/` (SPEC-13). Still COP-hardcoded: **live intraday execution**, forecast
`forecast_h5_*` DB tables, RL/H1/H5 `bars_per_year:19656`, the live market API, and landing branding.
For the complete as-built multi-asset map (incl. the per-asset session/timezone/annualization rules), see
**`_asbuilt-implementation.md`** (CTR-ASSET-IMPL-001). See also §7 and the onboarding spec.

---

## 1. Infrastructure (as deployed)

Two compose files at repo root. `docker-compose.yml` (full enterprise) and
`docker-compose.compact.yml` (daily driver, RAM-capped).

### 1.1 Services — full stack (`docker-compose.yml`)

| Service | Image / Build | Port | Role |
|---------|---------------|------|------|
| postgres | timescale/timescaledb:latest-pg15 | 5432 | Primary DB (TimescaleDB) |
| redis | redis:7-alpine | 6379 | Cache, pub/sub, Airflow broker |
| minio + minio-init | minio/minio | 9000/9001 | Object storage (~15 pipeline buckets) |
| data-seeder | Dockerfile.data-seeder | — | One-shot seed load |
| airflow-init/scheduler/webserver | Dockerfile.airflow-ml | 8080 | LocalExecutor; runs all DAGs |
| pgadmin | dpage/pgadmin4 | 5050 | DB admin |
| dashboard | usdcop-trading-dashboard/Dockerfile.prod | 5000→3000 | Next.js frontend |
| prometheus / grafana / alertmanager | prom + grafana | 9090 / 3002 / 9094 | Metrics, dashboards, alerts |
| loki / promtail | grafana | 3100 | Log aggregation + shipping |
| jaeger | jaegertracing/all-in-one | 16686 | Tracing (0 services instrumented) |
| vault + vault-init | hashicorp/vault:1.15 | 8201 | Secrets — **profile `full` only** |
| trading-api | services/Dockerfile.api | 8000 | Realtime market data + WS |
| analytics-api | services/Dockerfile.api | 8001 | Trading analytics |
| mlops-inference-api | services/Dockerfile.api | 8090 | RL/ML inference + risk |
| backtest-api | services/inference_api/Dockerfile | 8003 | Backtest + SSE replay |
| signalbridge-api | services/signalbridge_api/Dockerfile | 8085 | Exchange OMS / execution |
| mlflow | ghcr.io/mlflow/mlflow:v2.10.2 | 5001 | Experiment tracking (sqlite + MinIO) |

Network `usdcop-trading-network` (172.29.0.0/16). Several services are present-but-commented
(realtime-orchestrator, websocket-service, compliance-api, multi-model-api, nginx, …) —
treat as removed.

### 1.2 Compact vs full (real diffs)

- **Omits**: vault, jaeger, loki/promtail, pgadmin, data-seeder, mlops-inference-api, the
  uncommented `secrets` block. Monitoring (prometheus/grafana/alertmanager) is gated behind
  `--profile monitoring`.
- **Adds a "TIER 5: Course Project"** block absent from full: `redpanda` (Kafka, 19092),
  `redpanda-console` (8088), `grpc-predictor` (50051), `kafka-bridge-producer/consumer`.
- Compact scheduler adds `POLARS_SKIP_CPU_CHECK`, Azure/Anthropic LLM keys, extra mounts
  (`./seeds` — now mounted **`rw`** on airflow scheduler+webserver so the asset ingest DAGs can refresh
  seed parquets; dashboard `public`), and Airflow pool `api_requests=8`.
- `minio-init` creates 3 buckets in compact vs ~15 in full.
- signalbridge hardcodes `TRADING_MODE=PAPER` in compact.
- Header says "12 core services" but the always-on core count is 13.

### 1.3 Known infra discrepancies (disk vs config)

1. Both compose files set `FEATURE_CONFIG_PATH=…/config/features/feature_registry_v19.json`
   but **`config/features/` does not exist**. Real files are `config/feature_config.json`
   and `config/feature_registry.yaml`. → dead path, must fix before relying on it.
2. Prometheus config is split: scrape config in top-level `prometheus/`, rules in
   `config/prometheus/`.
3. `rl_l4_02_backtest_validation` and `rl_l4_01_experiment_runner` are DEPRECATED in
   `dag_registry.py` yet the files exist and one still has an active weekly schedule.
4. **Known issue** (pre-existing, non-blocking): on clean-slate cold boot the `usdcop-mlops-inference`
   container (:8003, RL inference — deprioritized) fails with `ModuleNotFoundError: No module named
   'src'` (missing PYTHONPATH/`src` mount). Does NOT affect H5 production, which reads
   `usdcop_m5_ohlcv` directly. See `inference-l1-l5.md`.

---

## 2. Data & Storage Layer (L0)

| Store | What | Where |
|-------|------|-------|
| PostgreSQL/TimescaleDB | live OHLCV, macro, forecasts, signals, trades, news, analysis | container `usdcop-postgres` |
| Seed parquets | restore + local-training fallback | `seeds/latest/*.parquet` |
| MinIO | 11+ buckets (mlflow artifacts active; model backup = roadmap) | container `usdcop-minio` |
| MACRO_DAILY_CLEAN | 17-col cleaned macro for H1/H5 | `data/pipeline/04_cleaning/output/` |

**Seeds on disk** (`seeds/latest/`, manifest 2026-01-23):
`usdcop_m5_ohlcv.parquet` (91,157 rows, 2020→2026), `usdmxn_m5_ohlcv.parquet`,
`usdbrl_m5_ohlcv.parquet`, `fx_multi_m5_ohlcv.parquet` (consolidated), `usdcop_daily_ohlcv.parquet`,
`usdcop_1h_ohlcv.parquet`, `macro_indicators_daily.parquet` (10,759 rows, 41 cols, 1954→2026).

**Golden rule**: every OHLCV timestamp is `America/Bogota`, session 08:00–12:55 COT, Mon-Fri.
The `usdcop_m5_ohlcv` table has `PRIMARY KEY (time, symbol)` — it is multi-pair by design
(migration 040). See `data-governance.md`.

**Storage model (as-built + roadmap).** OHLCV uses **one table per granularity, multi-asset via a
`symbol` discriminator** — 5-min → `usdcop_m5_ohlcv` (legacy name kept; symbol-parameterized), daily →
`asset_daily_ohlcv` (migration 051). The two are schema-identical; the granularity→table map is a single
SSOT constant `GRANULARITY_TABLE` in `scripts/ingest_asset_ohlcv.py`, and asset→seed routing reads the
`AssetProfile` (no hardcoded per-asset dicts; the chart exporter is `AssetProfile`-driven too). **Reality:
`asset_daily_ohlcv` is currently a durability mirror** — the Gold/BTC serving path is 100% seed-parquet →
bundle-JSON; the DB table is written best-effort and read only by the backup job. **Roadmap** (not done —
low ROI while serving is file-driven): a canonical `market_bars(asset_id, granularity, ts, …)` table (or
partitioned) with granularity+asset routing fully config-driven, retiring the misnamed `usdcop_m5_ohlcv`.
A new granularity (e.g. 1h) today = a `GRANULARITY_TABLE` entry + a migration + a backup-table entry.

**Macro catalog** (`config/macro_variables_ssot.yaml`, 40 variables): the SSOT for all macro.
Notably it already extracts **Gold** (`comm_metal_gold_glb_d_gold`), Brent, coffee, DXY, VIX,
UST 2Y/10Y, fed funds, US CPI/PCE, and spot USDMXN/USDCLP/USDCOP — plus Colombia-specific
EMBI, IBR, TPM, COLCAP, COL10Y. This catalog is the driver-menu for any new asset (§ onboarding).

---

## 2.1 Repository Layout & Git-Tracking Policy (2026-07 reorg)

**`scripts/`** — reorganized into a **purpose-based, importable package** (`from scripts.<subdir>.<mod>
import …`); root holds only `__init__.py`, guarded by `tests/regression/test_scripts_layout.py`:

| Subdir | Holds | Subdir | Holds |
|--------|-------|--------|-------|
| `pipeline/` | prod/RL/forecasting entrypoints, promote, registry | `diagnostics/` | `diagnose_`/`verify_`/`trace_` |
| `data/` | seeds, features, datasets, scrapers, DB seeding | `validation/` | reusable health/smoke gates |
| `ops/` | `db_migrate`, backfill, `backup/`, migrations, DVC/maint | `presentation/` | pptx/pitch/screenshots |
| `analysis/` | research backtests, sharpe/sensitivity | `tools/` · `lib/` | CLI utils · shared libs |
| `sql/` · `migrations/` · `deployment/` · `demo/` · `vault/` | SQL/schema/deploy/demo/vault | `archive/` | legacy/superseded (README-documented) |

Load-bearing script paths (DAG `subprocess`, deploy manifest, `Makefile`, `dvc.yaml`, CI, service imports)
were updated in the same reorg. See `CLAUDE.md` § "Scripts" for the "add a script" rule.

**Git-tracking policy**: only restore-critical data is tracked — `seeds/latest/*`,
`data/backups/seeds/*` + `*.csv.gz`, `data/pipeline/04_cleaning/output/*` (MACRO masters). Regenerable
artifacts are **gitignored** (on disk, out of git): `data/{cache,news,forecasting,pipeline/{00-03,05-07}}/`,
`data/backups/{full_backup_*,pre_v20_*}/`, `models/**` binaries, `results/`, `outputs/`,
`video-pitch/{out,public}/`, dashboard `public/forecasting/` + `public/data/analysis/`. Curated analyses
live tracked in `docs/analysis/`. Loose presentations were consolidated under `presentation/`.

---

## 3. Pipeline Layers & Strategy Tracks

```
L0 data ─► L1 features (RL) / forecasting feature build ─► L2 dataset ─► L3 train
       ─► L4 backtest+gates ─► Stage5 human approval ─► L5 signal ─► L7 execute ─► L6 monitor
```

| Track | Horizon | Bars | Models | Status | SSOT config |
|-------|---------|------|--------|--------|-------------|
| **H5 Weekly** | 5d | daily | Ridge+BR+XGB + Regime Gate | **PRODUCTION** | `config/execution/smart_simple_v1.yaml` |
| H1 Daily | 1d | daily | 9-model ensemble | PAUSED | `config/execution/smart_executor_v1.yaml` |
| RL PPO | intraday | 5-min | PPO MlpPolicy | DEPRIORITIZED | `config/experiments/v215b_baseline.yaml` |

Shared forecasting SSOT: `config/forecasting_ssot.yaml` (21 features, 9 models, 7 horizons,
`ln(close[t+H]/close[t])` target). The H5 alpha comes from the **regime gate** (Hurst) + TP/HS
mechanics, NOT model R² (audited R² < 0 both years). See `h5-smart-simple.md`.

---

## 4. Airflow DAG Inventory (39 modules → 40 DAGs)

| Group | Count | Representative dag_id (schedule UTC) |
|-------|-------|--------------------------------------|
| **L0 data** | 5 | `core_l0_02_ohlcv_realtime` (`*/5 13-17 * * 1-5`), `core_l0_03_macro_backfill` (`0 4 * * 0`), `core_l0_05_seed_backup` (`0 20 * * 1-5`) |
| **RL L1-L6** | 14 | `rl_l1_01_feature_refresh`, `rl_l5_01_production_inference`, `rl_l6_01_production_monitor`; 2 DEPRECATED (`rl_l4_01/02`) |
| **Forecast H1** | 6 | `forecast_h1_l3_weekly_training` (`0 6 * * 0`), `forecast_h1_l7_smart_executor` (`*/5 18-22 * * 1-5`) — **PAUSED (audit A3-01)**: `is_paused_upon_creation=True`, no auto-run; schedules are intended-only |
| **Forecast H5** | 7 | `forecast_h5_l3_weekly_training` (`30 6 * * 0`), `forecast_h5_l5_weekly_signal` (`15 13 * * 1`), `forecast_weekly_generation` (`0 14 * * 1`) |
| **Asset DS-cycle** | 2 | `asset_xauusd_pipeline_weekly` (`45 6 * * 0`), `asset_btcusdt_pipeline_weekly` (`0 7 * * 0`) — emitted by **one** factory module `asset_pipeline_factory.py` from SSOT `config/assets/pipelines.yaml` (CTR-ASSET-PIPELINE-001); tasks `l0_ingest`→`l4_backtest_publish`→`l6_verify_registry`. Makes Gold/BTC fully DAG-driven (were manual script-runners); USD/COP keeps its bespoke H5 chain, NOT modeled here |
| **News** | 4 | `news_daily_pipeline` (`0 7,12,18 * * 1-5`) |
| **Analysis/Recon/Watchdog** | 4 | `analysis_l8_daily_generation` (`0 19 * * 1-5`), `core_watchdog` (`0 13-18 * * 1-5`) |

DAG IDs resolve from `airflow/dags/contracts/dag_registry.py` (the asset DS-cycle DAGs are dynamically generated, not in the registry). Schedules assume the COP
session — a 24/7 asset breaks the `*/5 13-17`, weekday, and Friday-close cadence (see onboarding §5).

---

## 5. Contracts (the enforcement layer)

### 5.1 Python (`src/contracts/` + `src/core/contracts/`)

| File | Key exports | Asset coupling |
|------|-------------|----------------|
| `strategy_schema.py` | `StrategyTrade/Stats/Summary/TradeFile`, `GateResult`, `ApprovalState`, `safe_json_dump`, `EXIT_REASONS` | **None** (asset-agnostic) |
| `signal_contract.py` | `UniversalSignalRecord`, `SignalStore`, enums | **None** (no `symbol` field) |
| `signal_adapters.py` | `H5SmartSimpleAdapter`, `H1ForecastVTAdapter`, `RLPPOAdapter` | **COP-hardcoded** paths + macro cols |
| `execution_strategies.py` | `WeeklyTPHSExecution`, `DailyTrailingStopExecution`, `IntradaySLTPExecution` | MEXC/COT comments, `short_only` default |
| `analysis_schema.py` | `MacroSnapshot`, `Daily/WeeklyAnalysisRecord`, `WeeklyViewExport` | **Strongly COP** (`usdcop_*`, `embi_col`, `banrep`) |
| `news_engine_schema.py` | `ArticleRecord`, `DigestRecord`, `FeatureSnapshotRecord` | CO-domain (`country_focus="CO"`) |
| `core/contracts/feature_contract.py` | `FEATURE_ORDER`, `OBSERVATION_DIM`, `FEATURE_ORDER_HASH` (v3.1.0, 20-dim) | Implicit COP (`embi_z`, `usdmxn_change_1d`) |
| `core/contracts/production_contract.py` | `ProductionContract`, promote/archive | **None** (generic model registry) |

### 5.2 TypeScript (`usdcop-trading-dashboard/lib/contracts/`)

`strategy.contract.ts`, `weekly-analysis.contract.ts`, `ssot.contract.ts`,
`backtest-ssot.contract.ts`, `backtest.contract.ts`, `experiments.contract.ts`,
`forecasting.contract.ts`, `model.contract.ts`, `production-approval.contract.ts`,
`production-monitor.contract.ts` (+ an `execution/` subfolder where `symbol` IS modeled,
defaulting to `'USD/COP'`).

### 5.3 ⚠ Contract drift found (action items)

1. **Feature SSOT drift (highest severity)**: `ssot.contract.ts` = **15 dims, v2.0.0**, legacy
   features (`atr_pct`, `adx_14`, `time_normalized`, `rate_spread`) with a `join('|')` hash;
   Python `feature_contract.py` = **20 dims, v3.1.0** (EXP-B-001) with `volatility_pct`,
   `trend_z`, `rate_spread_change`, `log_ret_1d`, `rsi_21`, `yield_curve_z`, `gold_change_1d`,
   `position`, `unrealized_pnl`, and a `sha256(...)[:16]` hash. TS and Python disagree on the
   observation space AND the hash algorithm. `contracts-check.yml` should be catching this.
2. **Threshold inconsistency**: `THRESHOLD_LONG/SHORT = ±0.33` in `ssot.contract.ts` vs `±0.50`
   in `backtest-ssot.contract.ts`, the RL adapter, and `pipeline_ssot.yaml`.
3. **No TS mirror** for `news_engine_schema.py` or for `UniversalSignalRecord`.

These are pre-existing and out of scope to fix here, but MUST be reconciled before adding an
asset (a new asset re-derives the feature contract; you don't want to fork a drifted baseline).

---

## 6. Dashboard — Visual Inventory (13 pages)

Chrome: `navigation/GlobalNavbar` (public) or `app/execution/layout.tsx` sidebar. Dark theme.

| Route | What the user sees | Data source | Hardcoded asset assumption |
|-------|--------------------|-------------|-----------------------------|
| `/` | Marketing landing (hero, metrics band, features, pricing, FAQ) | static | "USD/COP" branding |
| `/hub` | 6 module cards + 4 static "Quick Stats" | static | Title "Terminal USD/COP"; fake stats (PPO v2.4, 67.3%) |
| `/forecasting` | Filter panel (view/week/model/horizon) + KPI grid + forecast PNG | `public/forecasting/bi_dashboard_unified.csv` + PNGs | "USD/COP" header; ensemble variants |
| `/dashboard` | Header strategy dropdown + live price + `ForecastingBacktestSection`: KPIs, candlestick, trades table, equity curve, **Approve/Reject (Vote 2/2)** + Deploy | `summary_2025.json`, `strategies.json`, `trades/*_2025.json`, `/api/production/*` | chart `symbol="USDCOP"`; fallback `smart_simple_v11`; year 2025 baked in paths |
| `/production` | Realtime price card, KPIs, live position, candlestick, approval status (read-only), gates, equity, guardrails, monthly PnL, trade table | `summary.json`, `trades/*.json`, `/api/production/live`, `/api/market/realtime-price` | `symbol="USDCOP"`; default year 2026 |
| `/analysis` | **Multi-asset selector** (USD/COP · Gold · Bitcoin) + week selector, weekly summary, technical card, scenarios, regime, signal cards, macro charts, daily timeline, calendar, chat widget | `/api/analysis/weeks`, `/week/{y}/{w}`, `/calendar`, `/chat` (asset-namespaced) | selector drives asset; all 3 assets have macro snapshot/charts (Gold/BTC via `src/analysis/asset_macro_charts.py` global drivers) + technical card (`AssetTechnicalCard` for the lean Gold/BTC schema) |
| `/execution` (+5 sub) | Kill-switch, bridge status, exchanges (Binance/MEXC), executions table, risk settings, login | signalbridge `/api/execution/*` | trades **crypto** exchanges yet executions table defaults display symbol to `'USD/COP'` (mismatch) |
| `/login` | 2-panel login + live USDCOP spot widget + system health | `/api/proxy/trading/stats/USDCOP` | hardcoded creds; `symbol="USDCOP"`; "RL v2.4" branding |

**Component folders**: `components/forecasting/` (dashboard, image viewer, ranking panel),
`components/analysis/` (14 widgets), `components/production/` (backtest section, live position,
guardrails), shared `charts/TradingChartWithSignals` (always `symbol="USDCOP"`, `5m`).

**Canonical strategy artifact**: `public/data/production/strategies.json` →
`smart_simple_v11` ("Smart Simple v1.1.0", 2025 return 23.07%, sharpe 3.822), used as the
fallback id across `/dashboard` and production pages.

**Multi-asset analysis (as-built 2026-07-05)**: `/analysis` is now **multi-asset** — a dynamic
selector renders per-week weekly+daily analysis for **USD/COP · Gold (`xauusd`) · Bitcoin (`btcusdt`)**,
consistent with the multi-asset registry (3 assets / 8 strategies). USD/COP keeps its macro **LangGraph**
pipeline (`scripts/pipeline/generate_weekly_analysis.py`); **Gold/BTC** analysis is generated by
`src/analysis/asset_analysis_generator.py` (CLI `scripts/pipeline/generate_asset_analysis.py`) from **real
data** — daily OHLCV seeds, computed technicals, real strategy positioning (published `gold_trend_b2`/
`btc_trend_b2` trade bundles), and real news. News is a **pluggable ports-&-adapters** module
`src/analysis/news_sources.py` (registry + factory + facade): Google News RSS (primary, no key,
date-scoped) → GDELT (paced fallback); a new source = one adapter + one registry line. SSOT is
`config/analysis/analysis_assets.yaml` (Python) mirrored by `lib/contracts/analysis-assets.ts` (TS), with
data namespaced under `public/data/analysis/<asset>/` — adding an analysed asset is one additive entry per
side. Gold/BTC remain **backtest+web science stacks** (no dedicated analysis DAGs).

---

## 7. The Multi-Asset Reality (why this matters)

> **As-built update (2026-07-04)**: the tiers below were the pre-onboarding map. The **Easy** and most
> **Moderate** tiers are now RESOLVED via `AssetProfile` — Gold ingests, scores, backtests, and renders
> without touching COP. What remains COP-coupled: the **Hard** structural tier for *live execution*
> (forecast `forecast_h5_*` tables, week-lifecycle, DAG schedules) and the live market API. See
> `_asbuilt-implementation.md` for the resolved-vs-remaining breakdown.

The literal `usdcop`/`USD/COP` appears **~1,294 times across ~250 files**. Most are the
(intentionally frozen) `usdcop_m5_ohlcv` table name and branding. The **behaviorally
load-bearing** couplings are ~15 files:

| Tier | Files / config |
|------|----------------|
| **Easy** (already `symbol`-parameterized) | `airflow/dags/l0_ohlcv_realtime.py` + `l0_ohlcv_backfill.py` (`SYMBOL_CONFIG`), `src/data/ohlcv_loader.py:54-64` (`SYMBOL_PRICE_RANGES`, `SYMBOL_SEED_FILES`), `config/seed_config.yaml`, `usdcop_m5_ohlcv` table (multi-pair PK) |
| **Moderate** (config + retrain) | macro drivers in `config/macro_variables_ssot.yaml`, `config/pipeline_ssot.yaml` (`macro_column_map`, cross-pair calcs), `config/forecasting_ssot.yaml`, `config/execution/smart_simple_v1.yaml` features, `feature_contract.py` + `norm_stats*.json` |
| **Hard** (structural COP assumptions) | session/timezone in `src/core/constants.py:40-43,399-413`, `trading_schedule`/`bars_per_day:78`, `aux_pairs.session`, `smart_simple_v1.yaml` week_lifecycle (Mon entry/Fri close), forecast DB tables `forecast_h5_*` (no `symbol` column), dashboard defaults (`lib/services/twelvedata.ts`, `lib/config/index.ts` `NEXT_PUBLIC_DEFAULT_SYMBOL=USDCOP`), regime-gate Hurst thresholds (COP-fit) |

The existing `aux_pairs` (USD/MXN, USD/BRL) machinery enriches the *COP* model with LATAM
cross-pair leaders; it does NOT trade those pairs. Promoting a symbol from "aux" to "primary
tradeable" is exactly the onboarding problem. → `_onboarding-playbook.md`.

---

## 8. Cross-References

| Concern | Spec |
|---------|------|
| **As-built multi-asset today** (Gold onboarded, session/tz/annualization) | **`_asbuilt-implementation.md`** |
| Add a new tradeable asset (BTC, Gold) — prescriptive playbook | **`_onboarding-playbook.md`** |
| Dynamic multi-strategy registry + replay (test any strategy/version/year in the UI) | **`registry-lifecycle.md`** |
| L0 data governance (OHLCV + macro) | `data-governance.md` |
| H5 production pipeline | `h5-smart-simple.md` |
| RL training pipeline | `training-l2-l3-l4.md`, `inference-l1-l5.md` |
| Strategy schemas | `strategy-contract.md` |
| Approval gates | `approval-gates.md` |
| Dashboard file schemas | `dashboard-integration.md` |
| Master lifecycle | `mlops-lifecycle.md` |
| Risk / execution / observability / cicd | `risk-management.md`, `execution-bridge.md`, `observability.md`, `cicd-testing.md` |

---

## DO NOT

- Do NOT treat this file as prescriptive — it is the *as-built* map. Rules live in the
  cross-referenced specs.
- Do NOT assume the dashboard is multi-asset — every chart hardcodes `symbol="USDCOP"`.
- Do NOT fork the feature contract without first reconciling the TS↔Python drift (§5.3).
- Do NOT rely on `config/features/feature_registry_v19.json` — the path does not exist on disk.
- Do NOT add a new asset by copy-pasting COP macro drivers — EMBI/IBR/TPM/oil/COL-yields are
  COP-specific and meaningless for BTC/Gold (see onboarding spec §3).
