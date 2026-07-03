# SDD Spec: Strategy Lifecycle & Dynamic Registry

> **Responsibility**: Authoritative contract that makes the system **multi-strategy,
> multi-asset, multi-version and multi-year at the frontend level** without hardcoding.
> Defines the self-describing **StrategyBundleManifest**, the **Registry index** the frontend
> reads to build itself dynamically, the **DAG I/O contract** (entry/exit rules) every pipeline
> must honor to publish a strategy, the **immutability/versioning** rule for backtest artifacts,
> and the **Replay contract** so any (strategy, asset, version, year) backtest can be replayed
> from the UI.
>
> This spec EVOLVES `sdd-strategy-spec.md` (StrategyManifest/Registry/Selector were single-asset,
> single-summary) into a scalable, registry-driven design. Read `sdd-architecture-overview.md`
> for the as-built state and `sdd-multi-asset-onboarding.md` for the AssetProfile it consumes.
>
> Contract: CTR-STRAT-REGISTRY-001
> Version: 1.1.0
> Date: 2026-07-03
> Status: IMPLEMENTED (file-based backend + frontend) · PROPOSED (DAG factory §6.3)
>
> **Implementation status (2026-07-03)** — the file-based registry + versioned immutable
> bundle + dynamic frontend are BUILT and tested; the DAG factory (§6.3) remains the
> proposed scale-out. Delivered, all ADDITIVE (no legacy consumption modified):
>
> | Piece | Artifact | State |
> |-------|----------|-------|
> | Manifest + registry contracts (Contracts A/B) | `src/contracts/strategy_manifest.py` (`StrategyBundleManifest`, `RegistryIndex`, `LegacyBundleAdapter`, `RegistryBuilder`) | ✅ |
> | Immutable versioned publisher (§5) | `strategy_manifest.py::BundlePublisher` — content-addressed `(strategy_id, version, year)`, never overwrites | ✅ |
> | Pipeline wiring | `scripts/train_and_export_smart_simple.py::publish_versioned_bundle()` — called AFTER legacy writes, guarded | ✅ |
> | DAG EXIT contract (§6.2) | `airflow/dags/utils/register_bundle.py::register_strategy_bundle()` | ✅ |
> | Registry index API (Contract B) | `GET /api/registry` (`app/api/registry/route.ts`) — legacy fallback shim (§11 step 1) | ✅ |
> | Manifest API (Contract A) | `GET /api/strategies/[strategyId]/manifest` | ✅ |
> | Promote API | `POST /api/registry/promote` `{strategy_id, version, status?}` — flips active version in manifest + syncs registry status | ✅ |
> | Version dropdown + per-version replay + Promote button | `components/production/ForecastingBacktestSection.tsx` (version state, `loadVersionData`, `handleVersionChange`, `handlePromote`) | ✅ |
> | Replay per version | Client-side (Option A): version dropdown overrides `summary`+`trades`; existing replay machinery + `TradingChartWithSignals` react automatically. No backend/SSE needed | ✅ |
> | Tests | `tests/contracts/test_strategy_registry.py` (8: immutability, coexistence, legacy-untouched, JSON-safe, chart-symbol derived, discovery) | ✅ |
> | DAG factory (config-driven pipeline gen) | §6.3 | ⏳ PROPOSED |
> | `asset_id`/`symbol` column on `forecast_h5_*` tables | migration | ⏳ PROPOSED |

---

## 1. Vision & Problem

**Goal**: adding a new strategy, a new asset, or a new model version must require **zero
frontend code changes** and **zero hardcoded file paths**. The operator opens the dashboard,
picks *any* asset → strategy → version → year, sees its backtest KPIs/trades/gates, and can
**replay** that backtest bar-by-bar — even for an experimental XAU/USD strategy or a v2 of an
existing model.

**Current blockers** (from `sdd-architecture-overview.md`):
- Single `summary.json` / `approval_state.json`; only ONE active strategy.
- Year `2025` and `symbol="USDCOP"` baked into paths, charts, and `production-monitor.contract.ts`.
- `strategies.json` exists but falls back to `smart_simple_v11`; not a true dynamic registry.
- Backtest artifacts are **overwritten** (`summary_2025.json`), so a new model version destroys
  the previous version's replayable backtest.

**Design principle**: **the frontend is a pure function of the Registry.** Pipelines produce
validated, immutable bundles and register them. The UI discovers everything from the registry.

---

## 2. The Mental Model (confirmed, with the scalability nuance)

| Change | Requires new DAG code? | What you actually do |
|--------|------------------------|----------------------|
| New **asset**, same strategy (e.g. Smart Simple on XAU) | **No** | Add `AssetProfile` + registry entry → DAG **factory** instantiates the pipeline |
| New **model version**, same strategy (e.g. Smart Simple v2) | **No** | New frozen config version → factory reruns; new immutable bundle version |
| New **strategy type** with new execution logic (e.g. Momentum) | **Yes** (strategy module + maybe `ExecutionStrategy`) | Implement against the signal/execution contracts; then register like any other |
| New **year** of backtest for an existing strategy | **No** | Pipeline appends a new immutable `backtests[]` entry |

So "new pair + new strategy = new pipelines" is right — but the *scalable* form is a **DAG
factory driven by the registry**, not copy-pasted DAG files. The contract below is what makes
that possible and what keeps the frontend dynamic.

---

## 3. Contract A — StrategyBundleManifest (self-describing bundle)

One manifest per strategy bundle. It is the ONLY thing the frontend needs to render everything
about that strategy. Location: `public/data/strategies/<strategy_id>/manifest.json`.
Python producer: `src/contracts/strategy_manifest.py` (mirrors `lib/contracts/strategy-manifest.contract.ts`).

```jsonc
{
  "schema_version": "1.0.0",
  "strategy_id": "smart_simple_xauusd_v2",     // globally unique; includes asset + version intent
  "asset_id": "xauusd",                          // FK → AssetProfile (sdd-multi-asset-onboarding)
  "symbol": "XAU/USD",                           // resolved from AssetProfile; UI never hardcodes
  "chart_symbol": "XAUUSD",                       // what TradingChartWithSignals must use
  "display_name": "Smart Simple — Gold v2",
  "pipeline_type": "ml_forecasting",              // ml_forecasting | rl | rule_based | hybrid
  "timeframe": "weekly",                          // weekly | daily | intraday_5m
  "status": "experimental",                       // experimental | paper | production | archived
  "capabilities": { "replay": true, "live": false, "approval": true },
  "produced_by": {                                 // provenance = DAG exit contract stamp
    "dag_id": "strat_xauusd_smart_simple_l4_backtest",
    "run_id": "manual__2026-07-03T…",
    "git_sha": "abc1234",
    "generated_at": "2026-07-03T14:00:00Z"
  },

  // --- Immutable, per (version, year). NEVER overwritten (see §5) ---
  "backtests": [
    {
      "model_version": "2.0.0",
      "year": 2025,
      "immutable_id": "smart_simple_xauusd_v2__2.0.0__2025",   // content key
      "summary":  "backtests/2.0.0/summary_2025.json",
      "trades":   "backtests/2.0.0/trades_2025.json",
      "signals":  "backtests/2.0.0/signals_2025.parquet",       // enables replay
      "replayable": true,
      "gates": { "passed": 5, "of": 5, "recommendation": "PROMOTE" },
      "headline": { "return_pct": 18.4, "sharpe": 2.9, "p_value": 0.012 }
    },
    { "model_version": "2.0.0", "year": 2024, "immutable_id": "…__2024", "…": "…" }
  ],

  // --- Mutable pointers (current production view) ---
  "production": { "model_version": "2.0.0", "year": 2026,
                  "summary": "production/summary.json", "trades": "production/trades.json" },
  "approval":   { "file": "approval_state.json", "status": "PENDING_APPROVAL" },

  // --- Model lineage (pick a version to replay its backtest) ---
  "model_versions": [
    { "version": "2.0.0", "trained_at": "…", "train_window": "2020-2024",
      "feature_hash": "…", "norm_stats_hash": "…", "artifact_uri": "…", "active": true },
    { "version": "1.0.0", "trained_at": "…", "active": false }
  ]
}
```

Rules:
- `symbol`/`chart_symbol` are the **only** source of truth for the UI — no `"USDCOP"` literal survives.
- Every `backtests[]` entry is keyed by `(model_version, year)` and is **immutable** (§5).
- `capabilities.replay` gates the replay button; requires the `signals` parquet to exist.
- JSON safety per `sdd-strategy-spec.md` (`safe_json_dump`, no Infinity/NaN).

---

## 4. Contract B — Registry index (dynamic discovery)

The single index the frontend fetches to build all selectors. Location: `public/data/registry.json`.
Written/updated by the DAG exit contract (§6) — never hand-edited. Superseding `strategies.json`.

```jsonc
{
  "schema_version": "1.0.0",
  "generated_at": "2026-07-03T14:00:00Z",
  "assets": [
    { "asset_id": "usdcop", "symbol": "USD/COP", "chart_symbol": "USDCOP",
      "display_name": "USD/COP", "asset_class": "fx" },
    { "asset_id": "xauusd", "symbol": "XAU/USD", "chart_symbol": "XAUUSD",
      "display_name": "Gold",   "asset_class": "commodity" }
  ],
  "strategies": [
    { "strategy_id": "smart_simple_v11",        "asset_id": "usdcop", "status": "production",
      "display_name": "Smart Simple v1.1", "pipeline_type": "ml_forecasting", "timeframe": "weekly",
      "manifest": "data/strategies/smart_simple_v11/manifest.json",
      "backtest_years": [2024, 2025], "has_production": true, "has_replay": true },
    { "strategy_id": "smart_simple_xauusd_v2",  "asset_id": "xauusd", "status": "experimental",
      "manifest": "data/strategies/smart_simple_xauusd_v2/manifest.json",
      "backtest_years": [2024, 2025], "has_production": false, "has_replay": true }
  ],
  "default": { "asset_id": "usdcop", "strategy_id": "smart_simple_v11" }
}
```

The registry is a **flat, append-friendly index** — publishing a new strategy adds one entry.
The frontend filters `strategies` by selected `asset_id`, then by `status` (e.g. hide `archived`).

---

## 5. Immutability & Versioning Rule (the scalability keystone)

> A backtest artifact, once published, is **immutable and content-addressed** by
> `(strategy_id, model_version, year)`. Producers MUST NOT overwrite it.

- Path convention embeds the version: `backtests/<model_version>/summary_<year>.json`.
  A new model version writes a NEW folder; the old backtest stays replayable forever.
- This is what makes "replay the 2025 backtest of model v1 vs v2" possible — both coexist.
- `production/summary.json` and `approval_state.json` are the ONLY mutable files (the "current"
  pointer), and they just reference an immutable backtest by `immutable_id`.
- Contrast with today: `summary_2025.json` is overwritten each run → previous versions are lost.
  Migrating to versioned paths is the single highest-leverage change for experiment scalability.

Aligns with `ssot-versioning.md` (frozen experiment configs) — the config version IS the
`model_version` in the manifest.

---

## 6. Contract C — DAG I/O Contract ("reglas de entrada y salida")

Every strategy pipeline DAG (train → signal → vol-target → execute → monitor, or the backtest
promotion DAG) MUST implement a standard input/output interface. This is what keeps the registry
trustworthy and the frontend dynamic.

### 6.1 ENTRY rules (inputs a DAG declares)

| Input | Contract |
|-------|----------|
| Asset params | Reads `AssetProfile` (`config/assets/<asset_id>.yaml`) — session, drivers, `chart_symbol` |
| Strategy config | Reads frozen `StrategyConfig` version (`config/execution/…` or experiment SSOT) |
| Upstream deps | `ExternalTaskSensor` on required upstream DAG (e.g. signal waits on training) |
| Data freshness | `validate_training_data_freshness()` gate (BLOCKING) per `data-freshness-enforcement.md` |
| Idempotency | Keyed by `(strategy_id, model_version, run_date)`; re-run must be safe (UPSERT) |

### 6.2 EXIT rules (outputs a DAG guarantees) — the enforcement point

The pipeline's **final task is `register_bundle`**. It is a hard gate: nothing reaches the
registry until it passes.

```
… produce artifacts …
        │
        ▼
register_bundle (EXIT CONTRACT — atomic, validated):
  1. VALIDATE artifacts against schema:
       - summary/trades JSON-safe (no Inf/NaN), required fields present
       - signals parquet present iff capabilities.replay
       - gates array complete (5 gates), recommendation computed (Vote 1/2)
       - feature_hash == norm_stats_hash lineage consistent
  2. WRITE immutable backtest under backtests/<version>/… (never overwrite)
  3. UPDATE manifest.json atomically (write .tmp → rename)
  4. UPSERT registry.json entry for this strategy_id
  5. (optional) SEED DB tables via --seed-db (UPSERT, idempotent)
  IF any step fails → do NOT touch registry.json (frontend never sees a broken bundle)
```

This makes the "regla de salida" literal: **a strategy becomes visible in the UI only after its
own DAG validated and registered a conformant, immutable bundle.** A failed/partial run is invisible.

### 6.3 DAG factory (recommended, for scale)

Rather than copy-pasting 5 DAG files per (asset, strategy), define a factory:

```python
# airflow/dags/factories/forecast_pipeline_factory.py
def build_forecast_pipeline(asset_profile, strategy_config) -> list[DAG]:
    # emits: <sid>_l3_train, _l5_signal, _l5_voltarget, _l7_execute, _l6_monitor
    # all reading AssetProfile (session/drivers) + StrategyConfig, all honoring §6.1/§6.2
```

`airflow/dags/generated_pipelines.py` loops the registry's enabled `(asset, strategy)` pairs and
calls the factory. Adding a strategy = a registry entry + a frozen config, **no new DAG file**.
Per-asset schedules come from `AssetProfile.session` (24/7 for BTC, metals for XAU, COT for COP).
Hand-written H1/H5 DAGs remain valid; migrate them under the factory incrementally.

---

## 7. Contract D — Replay (test any backtest from the UI)

Replay = stream a stored backtest bar-by-bar so the operator watches entries/exits play out.
Builds on the existing `ReplayBacktestEngine` (`src/contracts/replay_engine.py`),
`backtest.contract.ts` (SSE events, `REPLAY_SPEEDS`, state machine), and the backtest-api SSE.

### 7.1 Inputs (resolved dynamically, never hardcoded)

`replay(strategy_id, model_version, year)` → resolve manifest → load:
- `signals` parquet (`UniversalSignalRecord[]`, `sdd-strategy-spec.md`),
- OHLCV bars for `asset_id` + `year` from `usdcop_m5_ohlcv WHERE symbol = AssetProfile.symbol`
  (multi-pair table; the symbol comes from the manifest, not a literal).

### 7.2 API contract (dynamic)

```
GET  /api/backtest/replay?strategy_id=…&model_version=…&year=2025   → SSE stream (BacktestSSEEvent)
POST /api/backtest/run  { strategy_id, model_version, year, speed } → creates BacktestState (state machine)
GET  /api/registry                                                  → registry.json (dynamic index)
GET  /api/strategies/{strategy_id}/manifest                         → manifest.json
```

The chart plays with `chart_symbol` from the manifest. Because backtests are immutable/versioned
(§5), you can replay v1 and v2 of a model over the same 2025 and compare directly.

---

## 8. Dynamic Frontend Resolution Flow

```
1. GET /api/registry                              → assets[], strategies[], default
2. Render selectors: Asset → Strategy(filtered by asset) → Model Version → Year → View
3. GET manifest for selected strategy_id
4. Resolve artifact paths from manifest.backtests[(version, year)]  (NO hardcoded paths)
5. Render: KPIs (summary), TradesTable (trades), GatesPanel (gates),
           Chart with symbol = manifest.chart_symbol   (NO "USDCOP" literal)
6. If capabilities.replay → enable Replay → GET /api/backtest/replay?…  (SSE)
7. If status == PENDING_APPROVAL and view == dashboard → ApprovalPanel (Vote 2/2)
```

Every hardcoded value called out in `sdd-architecture-overview.md` §6 (`symbol="USDCOP"`,
fallback `smart_simple_v11`, year 2025, `NEXT_PUBLIC_DEFAULT_SYMBOL`) is replaced by a manifest/
registry lookup. `TradingChartWithSignals` takes `symbol` as a required prop sourced from the manifest.

---

## 9. File Layout (multi-asset / strategy / version / year)

```
public/data/
├── registry.json                                   ← Contract B (dynamic index)
├── strategies/
│   ├── smart_simple_v11/
│   │   ├── manifest.json                            ← Contract A
│   │   ├── backtests/
│   │   │   ├── 1.1.0/summary_2024.json  trades_2024.json  signals_2024.parquet   (immutable)
│   │   │   └── 1.1.0/summary_2025.json  trades_2025.json  signals_2025.parquet   (immutable)
│   │   ├── production/summary.json  trades.json     ← mutable "current" pointer
│   │   └── approval_state.json                       ← mutable
│   └── smart_simple_xauusd_v2/
│       ├── manifest.json
│       └── backtests/2.0.0/…
data/signals/<strategy_id>/<model_version>/<year>.parquet   ← backend source for replay
```

Legacy flat files (`public/data/production/summary_2025.json`) are supported during migration via
a compatibility shim in `/api/registry` that synthesizes a manifest for the legacy `smart_simple_v11`.

---

## 10. TDD Test-Contract (registry/dynamic frontend)

| ID | Test | Asserts |
|----|------|---------|
| R1 | `test_registry_schema` | `registry.json` validates; every `strategies[].manifest` path exists |
| R2 | `test_manifest_schema` | each manifest validates; `symbol`/`chart_symbol` present; JSON-safe |
| R3 | `test_backtest_immutability` | publishing model v2 does NOT modify v1's `backtests/1.x/**` files |
| R4 | `test_no_hardcoded_symbol_frontend` | no `"USDCOP"`/`"USD/COP"` literal on any strategy render path; chart symbol comes from manifest |
| R5 | `test_register_bundle_gate` | a bundle failing schema validation is NOT added to `registry.json` |
| R6 | `test_replay_resolves_dynamically` | `/api/backtest/replay` for (strategy, version, year) loads the right signals+bars via manifest, no literal path |
| R7 | `test_multi_asset_selector` | registry with 2 assets yields 2 asset options; strategies filter by `asset_id` |
| R8 | `test_version_coexistence` | v1 and v2 backtests of the same strategy/year are both replayable |
| R9 | `test_dag_exit_contract` | pipeline's `register_bundle` runs last, is idempotent, and stamps `produced_by` provenance |

R4 also closes the drift risk from `sdd-architecture-overview.md` §5.3 by making the UI symbol
manifest-driven.

---

## 11. Migration Path (from today's hardcoded state)

1. ✅ **DONE — Shim first (no pipeline change)**: `GET /api/registry` reads `registry.json`,
   synthesizes from legacy `strategies.json` when absent. Frontend resolves selectors from it.
2. ✅ **DONE — Versioned immutable paths**: `BundlePublisher` writes `backtests/<version>/…` +
   `manifest.json` (Contract A) via `publish_versioned_bundle()` in the export script; never
   overwrites (immutability proven by test `test_republish_same_version_is_immutable`).
3. ✅ **DONE — DAG exit contract**: `register_strategy_bundle()` util ready
   (`airflow/dags/utils/register_bundle.py`) for the `register_bundle` final task (Contract C §6.2).
4. ✅ **DONE — chart symbol from manifest**: `ForecastingBacktestSection` passes
   `symbol={manifest?.chart_symbol || 'USDCOP'}` into `TradingChartWithSignals` (fallback-safe).
   Version dropdown + per-version replay + `POST /api/registry/promote` button all wired.
5. ⏳ **PROPOSED — DAG factory**: when onboarding the 2nd asset (XAU), introduce the factory
   (§6.3) instead of copying H5 DAGs.
6. ⏳ **PROPOSED — Reconcile feature-contract drift** (`sdd-architecture-overview.md` §5.3)
   before multiplying strategies — do not multiply a drifted baseline.

### 11.1 Delivered API surface (all additive, off `PROTECTED_API_ROUTES`)

| Route | Method | Reads/Writes | Response |
|-------|--------|--------------|----------|
| `/api/registry` | GET | `public/data/registry.json` (fallback: `production/strategies.json`) | RegistryIndex |
| `/api/strategies/[strategyId]/manifest` | GET | `public/data/strategies/<sid>/manifest.json` (slug-guarded) | StrategyBundleManifest |
| `/api/registry/promote` | POST `{strategy_id, version, status?}` | writes manifest `model_versions[].active` + `production.model_version` + optional `status`; best-effort registry `status` sync | `{success, active_version, status}` |

### 11.2 Frontend replay decision (Option A — client-side)

Per-version replay is driven **purely client-side** from static JSON — no SSE/backtest-api.
On version select, `loadVersionData()` fetches the version's `summary`/`trades` (served from
`public/` at `/data/<relative-path>`) and overrides the `summary`+`trades` React state; the
existing self-contained replay machinery in `ForecastingBacktestSection` (replayIndex, play
speed, `visibleTrades`, progressive `chartEndDate`) and `TradingChartWithSignals`'
`isReplayMode`+`replayTrades` props re-render automatically. Approval stays strategy-level
(untouched). The backtest-api SSE path is keyed by `model_id`+date-range only and does NOT
know the per-version bundle layout — intentionally not used for per-version replay.

---

## 12. Cross-References

| Concern | Spec |
|---------|------|
| As-built state + hardcoded coupling | `sdd-architecture-overview.md` |
| AssetProfile (asset params consumed by ENTRY rules) | `sdd-multi-asset-onboarding.md` |
| Trade/summary/gate schemas + UniversalSignalRecord | `sdd-strategy-spec.md` |
| Approval gates (Vote 1 computed in register_bundle) | `sdd-approval-spec.md` |
| Dashboard artifact schemas / page flows | `sdd-dashboard-integration.md` |
| Frozen config = model_version | `ssot-versioning.md`, `experiment-protocol.md` |
| Data-freshness ENTRY gate | `data-freshness-enforcement.md` |
| Replay engine + SSE | `sdd-execution-bridge.md`, `backtest.contract.ts` |

---

## DO NOT

- Do NOT hardcode `strategy_id`, `symbol`, or backtest year in the frontend — resolve from the
  registry + manifest. `TradingChartWithSignals` takes `symbol` as a manifest-sourced prop.
- Do NOT overwrite a published backtest — write a new `(model_version, year)` immutable bundle.
- Do NOT publish to `registry.json` from anywhere except a DAG's `register_bundle` exit task.
- Do NOT let a partial/invalid bundle reach the registry — validation is a hard gate (§6.2).
- Do NOT copy-paste 5 DAG files per new (asset, strategy) — use the factory once a 2nd asset lands.
- Do NOT expose a Replay button for a bundle whose `signals` parquet is absent (`capabilities.replay=false`).
- Do NOT multiply strategies on top of the TS↔Python feature-contract drift — reconcile first.
- Do NOT store per-version backtests in a mutable shared file (`summary_2025.json`) — version the path.
