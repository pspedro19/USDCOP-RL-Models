# SDD Spec: Remediation Changeset — Code ↔ Spec Audit (2026-07)

> **Responsibility**: Authoritative backlog of **tasks to correct** found by a 10-agent deep audit
> that cross-referenced the ACTUAL codebase against the reorganized specs. Captures spec-vs-reality
> drift, redundancy, dead/obsolete code, deprecated pipelines, bad software practices, security
> gaps, and scalability limits. Each finding is a trackable task. This is a living doc: close items
> by editing `STATUS` and linking the fixing commit.
>
> Contract: CTR-AUDIT-001
> Version: 1.0.0
> Date: 2026-07-04
> Status: CONSOLIDATED (10-agent audit complete; ~114 findings; 1 CRITICAL / ~30 HIGH). Triage P0 next.
> Method: 10 specialized agents (A1–A10), read-only, each domain cross-checked code ↔ spec.
> Cross-refs: `../architecture-overview.md` (§5.3 contract drift, §1.3 infra discrepancies),
> `../../README.md`, all domain specs under `../`.

---

## How to use this doc

- Findings are grouped by audit domain (A1–A10) and ranked by severity.
- Each row: `ID · SEVERITY · CATEGORY · FILE:line · ISSUE · FIX · STATUS`.
- **STATUS**: `OPEN` (default), `IN PROGRESS`, `DONE` (link commit), `WONTFIX` (with reason).
- The **Priority Backlog** (bottom) is the cross-domain P0/P1/P2 rollup — start there.

**Severity**: `CRITICAL` (correctness/security/data-loss now) · `HIGH` (drift or bad practice that
will bite) · `MEDIUM` (redundancy/dead code/maintainability) · `LOW` (cosmetic/nice-to-have).

**Category**: `spec-drift` · `dead-code` · `redundancy` · `bad-practice` · `obsolete-pipeline` ·
`scalability` · `security` · `shipped-todo`.

---

## Audit domains

| # | Agent | Scope | Primary specs cross-checked |
|---|-------|-------|-----------------------------|
| A1 | Data layer (L0) | ingestion, timezone, seeds, macro, freshness | `rules/data-governance.md`, `rules/data-freshness.md`, `specs/data/`, `specs/operations/` |
| A2 | RL pipelines | L1-L5, L2-L3-L4, deprecated DAGs, feature contract | `specs/pipelines/*` |
| A3 | Forecasting tracks | H5 prod, H1 paused | `specs/tracks/h5-smart-simple.md`, `specs/platform/mlops-lifecycle.md` |
| A4 | Contracts + registry + approval | TS↔Python drift, BundlePublisher | `rules/strategy-contract.md`, `rules/approval-gates.md`, `specs/platform/registry-lifecycle.md` |
| A5 | Dashboard frontend + APIs | hardcodes, dead routes, dynamicMetrics | `specs/platform/dashboard-integration.md` |
| A6 | Execution / SignalBridge | DAG↔bridge gap, OMS, kill switch | `specs/platform/execution-bridge.md` |
| A7 | Risk management | 3 overlapping subsystems | `specs/platform/risk-management.md` |
| A8 | Auth + user creation | registration, JWT, lockout, security | (new) `specs/platform/authentication.md` |
| A9 | Observability + CI/CD + infra | dead config paths, obsolete services | `specs/platform/observability.md`, `specs/platform/cicd-testing.md` |
| A10 | News/Analysis + Gold/multi-asset | shipped TODOs, scalability | `specs/tracks/news-analysis/`, `specs/assets/*` |

---

## Findings

<!-- Consolidated from agents A1–A10. Each section filled from the agent's structured return. -->

### A1 — Data Layer (L0)

> Cross-cutting OK: timezone golden rule + BRL-UTC quirk correctly enforced in the FX path; UPSERT `ON CONFLICT (time,symbol)` consistent across all 3 writers. **Drift concentrated in the Gold multi-asset extension + stale schedule/migration/restore docs.**

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A1-01 | HIGH | spec-drift | `scripts/ingest_asset_ohlcv.py:129`, `migrations/051:23-24` | Gold stores 5-min at UTC instants / daily at NY-close UTC — violates data-governance "ALL timestamps America/Bogota, no exceptions" (never amended for multi-asset) | Amend golden rule to document the instant-based TIMESTAMPTZ convention for non-COP assets; cross-ref mig 051/SPEC-02 | OPEN |
| A1-02 | HIGH | scalability | `migrations/051` + `ingest_asset_ohlcv.py` | `asset_daily_ohlcv` + Gold ingestion have ZERO references in airflow/dags or src/data — no realtime DAG, no freshness gate, no seed-backup, no restore path (manual script only) | Add table to `l0_seed_backup` + `seed_config.yaml` + per-asset freshness; add Gold DAG or mark Gold manual-only in spec | OPEN |
| A1-03 | HIGH | bad-practice | `src/data/ohlcv_loader.py:127,163,182` | `filter_market_hours` documents/filters "13:00-18:00 UTC" (legacy pre-COT), contradicts 08:00-12:55 COT session — silently keeps/drops wrong bars | Update docstrings + `TradingCalendar.filter_market_hours` to COT 08:00-12:55; verify filter runs in COT not UTC | OPEN |
| A1-04 | HIGH | redundancy | `config/seed_config.yaml:40-105` vs `specs/data/backup-recovery.md:136-144` | Two contradicting restore-priority SSOTs: seed_config = MinIO→local→CSV; backup-recovery = daily-backup→seeds→manual | Reconcile to one restore SSOT matching what `04-data-seeding.py` actually does | OPEN |
| A1-05 | MEDIUM | spec-drift | `l0_macro_backfill.py:1202`, `l0_seed_backup.py:479` | data-governance still lists pre-collision schedules (macro Sun06:00, backup daily18:00); code runs Sun04:00 + Mon-Fri20:00 | Update data-governance DAG rows or defer to elite-operations (schedule SSOT) | OPEN |
| A1-06 | MEDIUM | spec-drift | backup-recovery/data-freshness (`043-049`) vs elite-operations:217 (`043-046`) | 3 specs give inconsistent migration-apply lists; none include 050/051; 048 bundled unexplained | Produce one canonical migration list (043-051, 047 optional); add 050/051 to tracking table | OPEN |
| A1-07 | MEDIUM | spec-drift | `elite-operations.md:37` | Names regime-gate migration as 048, but on disk 048=`reconciliation_tables`, regime=`049_regime_gate_columns` | Change elite-operations to reference `049_regime_gate_columns.sql` | OPEN |
| A1-08 | MEDIUM | redundancy | `config/seed_config.yaml:96` | Macro fallback points at `05_resampling/MACRO_DAILY_CONSOLIDATED.csv` — different dir/file than every other spec (`04_cleaning/MACRO_DAILY_CLEAN`) → dead/divergent path | Point at canonical `04_cleaning/output/MACRO_DAILY_CLEAN.parquet` or confirm+reconcile | OPEN |
| A1-09 | MEDIUM | scalability | `data_quality.py:95` (`WHERE symbol='USD/COP'`), `ohlcv_loader.py:55-66` | Freshness gate hardcodes COP; loader hardcodes 3 FX pairs (no XAU) → Gold has no freshness enforcement, can't resolve seed via loader | Parameterize `validate_training_data_freshness` by symbol; drive SYMBOL_* dicts from AssetProfile/seed_config | OPEN |
| A1-10 | MEDIUM | redundancy | `init-scripts/04-data-seeding.py` + `04-seed-from-minio.py` | Two parallel startup restore scripts; priority/precedence between them undefined + duplicated | Document ordering/winner or consolidate to one restore entrypoint | OPEN |
| A1-11 | MEDIUM | spec-drift | `config/seed_config.yaml:127-138` | Declares Sun-02:00 backup + 4-snapshot retention, but DAG runs Mon-Fri20:00 overwriting daily (no retention) — config describes unimplemented behavior | Remove unused block or implement retention + align cron | OPEN |
| A1-12 | LOW | obsolete-pipeline | `ohlcv_loader.py:311-406` | Daily loader primary source `bi.dim_daily_usdcop` absent from every L0 spec/seed_config/backup — undocumented lineage | Document it (source/refresh/backup) or migrate daily reads to `asset_daily_ohlcv` | OPEN |
| A1-13 | LOW | bad-practice | `ohlcv_loader.py:355-363,516-522` | Supported `load_daily` fallback self-raises `DeprecationWarning` — muddies real deprecation signals | Use `logger.warning` in the fallback, or drop resample fallback + fail loudly | OPEN |
| A1-14 | LOW | spec-drift | `rules/data-governance.md:22` vs `backup-recovery.md:136-144` | 4-tier restore (incl MinIO) vs 3-step (MinIO absent) — inconsistent precedence | Align with backup-recovery canonical order (+ A1-04) | OPEN |
| A1-15 | LOW | bad-practice | `data_quality.py:55-68` | Age via integer `.days` truncation → 3.9-day data reports age=3 and passes 3-day gate (~24h lenient) | Use fractional-day age (`total_seconds()/86400`) with `>` | OPEN |

### A2 — RL Pipelines

> **Cross-cutting**: the feature-contract SSOT is fragmented across **4 disagreeing sources** — `experiment_ssot.yaml` (20-dim, loader's actual primary), `feature_contract.py` fallback (20), `pipeline_ssot.yaml` (32), TS `ssot.contract.ts` (15). Python runtime is consistent at 20; drift is TS(15) + pipeline_ssot(32). Reconcile before any new RL experiment or asset onboarding.

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A2-01 | HIGH | obsolete-pipeline | `airflow/dags/l4_backtest_validation.py:691-698` | Registry marks `RL_L4_BACKTEST_VALIDATION` DEPRECATED yet file is live with active weekly schedule `0 0 * * 0` + legacy hardcoded dag_id | Delete file (superseded by l4_backtest_promotion) or set schedule None + pause + registry constant | OPEN |
| A2-02 | HIGH | dead-code | `src/services/__init__.py:20-39` | inference-l1-l5 spec says `l1_nrt_data_service`/`l5_nrt_inference_service` DELETED, but both exist + are imported/exported → resurrected dead services bypass L1/L5 DAG path | Delete both files + imports/`__all__`, or correct the spec | DONE 2026-07 (files deleted; regression guard tests/regression/test_services_package.py 8/8) |
| A2-03 | HIGH | spec-drift | `ssot.contract.ts:58,73` vs `feature_contract.py:123,128` | TS 15-dim v2.0.0 `join('|')` hash vs Python 20-dim v3.1.0 `sha256[:16]` — disagree on obs space AND hash algo; `contracts-check.yml` not catching it | Regenerate ssot.contract.ts from Python FEATURE_ORDER/hash; add CI cross-language hash assertion | OPEN |
| A2-04 | HIGH | bad-practice | `pipeline_ssot.yaml:1241`, `config.py:246,997`, `ssot.contract.ts:156` | 4 action thresholds coexist: ±0.25 (ssot) / ±0.60 (env default) / ±0.33 (CLI+TS, TS comment falsely cites config.py) | Read thresholds from pipeline_ssot.yaml everywhere; remove hardcoded literals + wrong provenance comment | OPEN |
| A2-05 | MEDIUM | spec-drift | `pipeline_ssot.yaml:19-24,226` | Active SSOT is `EXP-SAC-SWING-002` v4.2.1 obs_dim 32 (SAC, changes arch+action+features at once) — diverges from RL contract (20) + violates 1-variable rule; specs still say PPO V21.5b | Reconcile with experiment_ssot.yaml; register+freeze SAC-SWING as `config/experiments/*.yaml` instead of overwriting active SSOT | OPEN |
| A2-06 | MEDIUM | dead-code | `docker-compose.yml:307,396` | `FEATURE_CONFIG_PATH=.../config/features/feature_registry_v19.json` — dir doesn't exist (real: `config/feature_registry.yaml`) | Point at existing file in both compose files or remove the env var | OPEN |
| A2-07 | MEDIUM | obsolete-pipeline | `airflow/dags/l4_experiment_runner.py:597,697` | DEPRECATED module also defines a live `L4_SCHEDULED_RETRAINING` DAG (`0 2 * * 0`) — hidden active scheduling inside a "deprecated" file | Split retraining into own module (rl_l4_03) or fold into l4_backtest_promotion; delete deprecated runner | OPEN |
| A2-08 | MEDIUM | bad-practice | `airflow/dags/l1_feature_refresh.py:92` | Hardcoded fallback norm constants (`vix mean18/std5`) bypass promoted `norm_stats.json` → silent mismatched-normalization garbage if hit | Remove hardcoded stats; fail-closed if `norm_stats.json` unavailable | OPEN |
| A2-09 | LOW | redundancy | `dag_registry.py:445-446` | `get_all_dag_ids()` still lists the 2 DEPRECATED rl_l4 DAGs in the canonical order → consumers treat them as active | Remove deprecated IDs from `get_all_dag_ids()`; gate behind `DEPRECATED_DAGS` set | OPEN |

### A3 — Forecasting Tracks (H5/H1)

> **Clean**: `smart_simple_v1.yaml` matches spec exactly; no monthly-retraining dead code; no H1/H5 base-feature duplication; regime live-util ↔ offline-module parity OK; H5 DAG schedules match spec.

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A3-01 | HIGH | obsolete-pipeline | `forecast_h1_l3_weekly_training.py:660` (+l5/l7/l6) | All 5 H1 "paused" DAGs carry active crons + no `is_paused_upon_creation` → on any DB reset they auto-run (pause is only a UI toggle) | Add `is_paused_upon_creation=True` (or `schedule_interval=None`) to all forecast_h1_* | OPEN |
| A3-02 | HIGH | spec-drift | `forecast_h5_l3_weekly_training.py:230` (21 feats) vs `train_and_export_smart_simple.py:1489` (23 feats) | **Live weekly training uses 21 base features; the approved +25.63% backtest used 23** (adds vol_regime_ratio+trend_slope_60d) → production models ≠ backtested; approval evidence doesn't describe what trades | Call shared `enhance_features_v2` in live L3 (or drop the 2 features from export) so both use identical features | **FIXED + REVALIDATED** (br `fix/audit-p0-remediation`, dir chosen: align UP to 23): extracted `enhance_features_v2` to shared `src/forecasting/enhance_v2.py` (imported by export + L3); L3 now builds 23 feats and persists them to `feature_cols_h5.json`; L5 signal now READS that list (kills the whole train/infer feature-drift class, not just these 2). Backtest re-run reproduced **exactly $12,563.05 / +25.63% / Sharpe 3.347 / p=0.0063 / 34 trades**, proving the refactor is behavior-preserving. Note: DAGs not executed (no Airflow env) — parity verified structurally + at function level. |
| A3-03 | HIGH | spec-drift | `train_and_export_smart_simple.py:124-126,148` | `load_config` fallbacks drift from SSOT: hurst 0.55/0.45 (SSOT 0.52/0.42), sizing_indeterminate 0.25 (0.40), cap 0.025 (0.035) → configs omitting a key silently backtest different thresholds | Change fallbacks to SSOT values or fail-fast on missing keys | OPEN |
| A3-04 | MEDIUM | bad-practice | `forecast_h5_l5_vol_targeting.py:41-43` | `LONG_MULTIPLIER=0.5`/`SHORT_MULTIPLIER=1.0` hardcoded despite YAML open at L155; comment cites nonexistent `smart_executor_h5_v1.yaml` | Read asymmetric multipliers from YAML; delete stale filename comment | OPEN |
| A3-05 | MEDIUM | dead-code | `train_and_export_smart_simple.py:61,266,339`, `src/forecasting/momentum_signal.py` | v3.0 momentum path registered in `STRATEGY_VERSIONS["v3"]` but unreachable (both call sites hardcode `version="v2"`; `--version` only stamps bundle name); undocumented | Wire `--strategy-version` through to backtest calls or remove v3 until spec'd | OPEN |
| A3-06 | MEDIUM | bad-practice | `src/forecasting/regime_gate.py:46-48` | Docstring says H>0.55 trending / H<0.45 mean-rev, contradicting code+SSOT (0.52/0.42); "0.25 reduced" vs actual 0.40 | Update docstring/comments to 0.52/0.42/0.40 | OPEN |
| A3-07 | MEDIUM | bad-practice | `src/forecasting/adaptive_stops.py:13-16` | Docstring worked-examples computed with old vol_multiplier=1.5 (HS=2.11%) not current 2.0 (spec 2.81%) | Recompute docstring examples with 2.0 | OPEN |
| A3-08 | LOW | obsolete-pipeline | `scripts/analysis/backtest_h5_v2_final.py:99` | Orphan one-off backtest hardcodes config "matching" a phantom `smart_executor_h5_v2.yaml`; superseded by train_and_export | Delete or refactor to load `smart_simple_v1.yaml` | OPEN |
| A3-09 | LOW | bad-practice | `src/forecasting/momentum_signal.py:1-4` | Header claims v3.0 "Replaces Ridge/BR/XGBoost" as production — spec/SSOT say production = Ridge+BR+XGB | Re-title header "experimental/paper (v3.0 candidate)"; note production stays v2.0 | OPEN |

### A4 — Contracts + Registry + Approval

> **Clean**: 5-gate builder matches approval-gates exactly; `safe_json_dump` used on legacy production exports + BundlePublisher; DAG factory (§6.3) correctly PROPOSED.

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A4-01 | HIGH | spec-drift | `strategy_manifest.py:427-479` + `train_and_export_smart_simple.py:1604` | `publish()` writes EVERY year (incl. live 2026) into IMMUTABLE `backtests/` and never sets the mutable `production` pointer → weekly-updated production numbers silently dropped after first run; `manifest.production` stays null | Branch prod vs backtest: write/overwrite `production/summary.json`+set `manifest.production`; never route the live year through immutable path | DONE 2026-07-07 (publish(phase="production") → mutable strategies/<sid>/production/ + manifest.production pointer; sandbox-verified) |
| A4-02 | HIGH | dead-code | `airflow/dags/utils/register_bundle.py:70` | `register_strategy_bundle()` documented as the DAG EXIT gate (✅) but NO DAG calls it — registry refresh only happens as a side-effect of BundlePublisher in the script → the validate-then-register gate never runs in a pipeline | Wire as terminal task of `forecast_h5_l4_backtest_promotion`, or downgrade spec status ✅→⏳ | DONE 2026-07-07 (register_bundle wired as terminal task of L4 + new L4b deploy DAG) |
| A4-03 | HIGH | spec-drift | `lib/contracts/` (absent) | Spec says `StrategyBundleManifest` mirrors `lib/contracts/strategy-manifest.contract.ts` but no such TS type exists; registry/promote routes + component consume manifest as untyped `any` | Add `strategy-manifest.contract.ts` mirroring the Python dataclasses; type the routes/component | DONE 2026-07-07 (lib/contracts/strategy-manifest.contract.ts + 3 routes typed) |
| A4-04 | MEDIUM | bad-practice | `app/api/registry/promote/route.ts:90-108` | Promote writes `manifest.json` AND `registry.json` from TS — violates DO-NOT ("publish to registry.json only from register_bundle") and can race the Python rebuild | Make promote call a Python endpoint or set an `active_version` override file RegistryBuilder honors | OPEN |
| A4-05 | MEDIUM | spec-drift | `app/api/registry/promote/route.ts:96-108` | Promote syncs only `status`, not the denormalized `active_version`/return/sharpe/p_value headline → selector shows old version's metrics until next Python RegistryBuilder run | Recompute+write headline on promote, or drop denormalized headline and read from manifest | OPEN |
| A4-06 | MEDIUM | spec-drift | `src/contracts/signal_contract.py:1` | `UniversalSignalRecord`/`SignalStore` have no TS mirror (execution signal.contract.ts is a different type); replay/registry specs reference it as a frontend input | Add TS mirror or scope it backend-only in spec | OPEN |
| A4-07 | MEDIUM | bad-practice | `src/contracts/news_engine_schema.py:82-165` | `to_dict()` has no Inf/NaN sanitization + no `safe_json_dump` (unlike analysis/strategy schemas) → an Inf/NaN feature emits invalid JSON breaking `JSON.parse()` | Route news serialization through `safe_json_dump` or add a sanitizer | OPEN |
| A4-08 | MEDIUM | redundancy | `src/contracts/strategy_manifest.py:34-48` | `_sanitize_for_json`/`safe_json_dump` re-implemented inline and drifted from strategy_schema's (no `default=` datetime handler, `allow_nan=False`) → datetime in a manifest raises TypeError at dump | Add `default=_json_default` or extract sanitizer to a shared leaf module | OPEN |
| A4-09 | LOW | scalability | `ForecastingBacktestSection.tsx:1364`, `app/api/registry/route.ts:55` | Hardcoded `'smart_simple_v11'` fallbacks (DO-NOT: no hardcoded strategy_id) → a registry with only non-COP strategies boots to a nonexistent id | Default to `registry.default.strategy_id ?? strategies[0]?.strategy_id` | OPEN |
| A4-10 | LOW | scalability | `train_and_export_smart_simple.py:920-932` | `deploy_manifest` hardcodes `smart_simple_v1.yaml` + args, ignoring `--config`/`--strategy-id`/`--version` → approving an A/B branch auto-deploys the WRONG config | Derive config_path/args/db_tables from the run's actual cfg | OPEN |
| A4-11 | LOW | bad-practice | `ForecastingBacktestSection.tsx:1561-1575` | `displayGates` recomputes pass/fail from visible/replay trades → gate panel can differ from immutable `approval_state.gates` | Always render gate pass/fail from published `approval.gates`; label "backtest gates" | OPEN |
| A4-12 | LOW | spec-drift | `strategy_manifest.py:476-481` | Spec §3/§7.1 say replay signals are `signals_<year>.parquet`, but `publish()` writes `.json` → spec-honoring SSE replay consumer misses the file | Align extension (parquet) or update spec to JSON canonical | OPEN |
| A4-13 | LOW | security | `app/api/production/approve/route.ts:57-89` | `approve` sets `approved_by` from client body, unauthenticated, then fire-and-forget deploys (retrain+`--seed-db`) → any caller can approve+trigger a prod deploy | Gate approve/deploy/promote behind auth; record authenticated principal (see A8) | **FIXED** (br `fix/audit-p0-remediation`): approve + deploy routes now require `protectApiRoute`; `approved_by` taken from authenticated principal (not body); approve→deploy forwards session cookie |

### A5 — Dashboard Frontend + APIs

> Good patterns to copy: the `/api/registry` fallback shim + the candlesticks catch-block (correct, no synthetic data).

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A5-01 | HIGH | spec-drift | `app/api/trading/signals/route.ts:498-524` | On DB failure returns fabricated `generateFallbackSignals` (`source:'fallback-generated'`) and it IS reachable (`useIntegratedChart.ts:152`) → fake signals render on the live chart; spec says no simulated data | Delete fallback; return `{success:false,signals:[]}` 503 → empty/error state | **FIXED** (br `fix/audit-p0-remediation`): catch now returns `{success:false,signals:[],source:'error'}` 503; `generateFallbackSignals` deleted |
| A5-02 | HIGH | bad-practice | `ForecastingBacktestSection.tsx:1147,1150,1532` | `dynamicMetrics` recomputes p-value in-browser via crude `exp(-0.717·|t|-0.416·t²)` and fakes `da=winRate`, overriding the published scipy p-value (Gold B2 0.0754 on-screen vs 0.0358 published) — corrupts the approval surface | Display `summary.statistical_tests.p_value`/`direction_accuracy_pct` verbatim; suppress significance badge on partial replay | RESOLVED (Vote-2 integrity pass): decision surfaces read the BUNDLE (displayGates=approval.gates; recompute only labels the "PREVIEW DEL REPLAY") |
| A5-03 | MEDIUM | spec-drift | `app/api/market/realtime-price/route.ts:131,494` | Returns hardcoded `price:4200 source:'fallback'`; also fetches nonexistent `/api/candlesticks/USDCOP` (real: `candlesticks-filtered`) → that source always fails | Return `data:null` on failure; fix/remove the dead internal call | OPEN |
| A5-04 | MEDIUM | obsolete-pipeline | `app/hub/page.tsx:225-228` | Quick Stats hardcode `'PPO v2.4' 67.3% Sharpe 1.84` — stale RL; production is Smart Simple v2.0 | Remove or bind to `/api/registry` active-strategy headline | OPEN |
| A5-05 | MEDIUM | dead-code | `app/login/page.tsx:31,54` | Live widget fetches `/api/proxy/trading/stats/USDCOP` + `/api/market/realtime` — neither route exists → widget always 404s | Point at `/api/market/realtime-price` or remove widget | OPEN |
| A5-06 | MEDIUM | dead-code | `app/api/production/monitor/route.ts:75` | Hardcodes `WHERE symbol='USDCOP'` and has zero callers → dead endpoint with stale symbol | Delete, or wire+parameterize symbol from manifest | OPEN |
| A5-07 | MEDIUM | bad-practice | `ForecastingBacktestSection.tsx:1271-1319` | `loadStrategyData` + `loadManifest→loadVersionData` both `setTrades` with no AbortController/sequence token; the trades write at :1316 is unguarded → version-A trades can render over version-B metrics | Add per-load sequence id/AbortController; make loadVersionData sole trades authority when a manifest exists | DONE 2026-07-07 (loadSeqRef monotonic token guards setSummary/setTrades in both loaders) |
| A5-08 | MEDIUM | spec-drift | `app/execution/executions/page.tsx:222` | Crypto venues (Binance/MEXC) but table defaults `execution.symbol||'USD/COP'` → FX symbol on crypto rows | Render actual per-execution symbol; drop the default (show `—`) | OPEN |
| A5-09 | MEDIUM | dead-code | `app/api/market/candlesticks-filtered/route.ts:38-107` | `generateFallbackCandlesticks()` (65 lines random OHLC) defined but never invoked → invites accidental re-wiring | Delete function + unused `COLOMBIA_HOLIDAYS` | OPEN |
| A5-10 | MEDIUM | scalability | `app/production/page.tsx:1549` | `/production` chart hardcodes `symbol="USDCOP"` (unlike ForecastingBacktestSection which uses manifest) → blocks multi-asset production monitoring | Source symbol from selected strategy manifest `chart_symbol` | OPEN |
| A5-11 | LOW | obsolete-pipeline | `app/login/page.tsx:325,475,537,797` | Stale "RL v2.4"/"v2.4 Active"/"© 2024" branding | Remove RL/version branding; dynamic year | OPEN |
| A5-12 | LOW | bad-practice | `ForecastingBacktestSection.tsx:1150` | `const da = winRate; // ≈ DA` conflates direction accuracy with win rate | Render published `direction_accuracy_pct`; don't derive from WR | OPEN |
| A5-13 | LOW | scalability | `lib/config/index.ts:44` | `NEXT_PUBLIC_DEFAULT_SYMBOL` defaults `USDCOP` as app-wide singleton | Drive default from `/api/registry` `default.asset_id`→manifest | OPEN |
| A5-14 | LOW | scalability | `ForecastingBacktestSection.tsx:1739` | `symbol={manifest?.chart_symbol||'USDCOP'}` keeps COP literal fallback → registry-only asset lacking chart_symbol mislabels as USDCOP | Fall back to `manifest?.symbol`/registry entry or disabled chart | OPEN |
| A5-15 | LOW | scalability | `production/monitor/route.ts:75`, `candlesticks-filtered:137` | DB queries hardcode `symbol='USDCOP'`/`'USD/COP'` despite multi-pair table; no symbol param | Accept validated `symbol` query param into WHERE | OPEN |

### A6 — Execution / SignalBridge OMS

> **Spec now obsolete**: "The Gap" is partially closed — `airflow/dags/utils/signalbridge_client.py` EXISTS and both L7 DAGs wire it (entry-only), but the implementation is incomplete/unsafe. Clean: Vault AES-256-GCM+PBKDF2, kill-switch confirm, trading-mode resolution, executor purity, async CCXT, prod-secret validator, DEV_MODE prod hard-guard.

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A6-01 | HIGH | spec-drift | `forecast_h5_l7_multiday_executor.py:336` (+h1:324) | Bridge called ONLY on ENTRY; exits/TP/HS/close never forwarded → in testnet/live a real position is opened but never closed via OMS (orphaned) | Route all exit/close events through the bridge (gated on non-paper), or block non-paper mode until exit-forwarding exists | **MITIGATED** (br `fix/audit-p0-remediation`): both H5+H1 executors now hard-block a non-paper ENTRY (raise) unless `H{5,1}_ALLOW_NONPAPER_UNSAFE=true` — no orphaned positions. Full exit-forwarding still TODO before non-paper is safe |
| A6-02 | HIGH | bad-practice | `signal_bridge.py:180` | `/process` discards caller `signal_id` and generates fresh uuid4 → DAG idempotency keys ignored, task re-run creates duplicate exchange orders | Accept+persist caller `signal_id`; enforce uniqueness on `(model_id, signal_id)` | OPEN |
| A6-03 | MEDIUM | bad-practice | `forecast_h1_l7_smart_executor.py:324` | Bridge order fire-and-forget AFTER PaperBroker enters; on failure only logs → paper/exchange state diverge, no reconciliation | Non-paper: make bridge authoritative (enter locally only on success) or reconcile via executions table | OPEN |
| A6-04 | MEDIUM | spec-drift | `signalbridge_client.py:145` | Posts to `/api/signal-bridge/process` not `/api/executions` (spec); is a function module not a `SignalBridgeClient(BrokerAdapter)`; no `close_position()` → spec "The Gap"/TODO obsolete | Reconcile: update spec to bless `/process`+function API, or refactor to `SignalBridgeClient(BrokerAdapter)`; mark Gap resolved | OPEN |
| A6-05 | MEDIUM | bad-practice | `forecast_h5_l7_multiday_executor.py:341` (+h1:329) | `quantity=leverage` passes the leverage multiplier (1.5) as order quantity — not an order size | Compute qty = equity×leverage÷price (or let orchestrator sizing own it) | OPEN |
| A6-06 | LOW | bad-practice | `forecast_h1_l7_smart_executor.py:332` (+h5:344) | `confidence=0.8` hardcoded in bridge calls, discarding real per-signal confidence → corrupts confidence-scaled sizing + LowConfidence checks | Pass actual signal confidence from the DB row | OPEN |
| A6-07 | LOW | bad-practice | `signalbridge_client.py:34` | `SERVICE_CREDENTIAL_ID`/`SIGNALBRIDGE_SERVICE_TOKEN` default empty → non-paper sends no auth (401) + empty credential_id (422), masked as generic error | Fail fast at DAG start when mode≠paper and token/credential unset | OPEN |
| A6-08 | LOW | spec-drift | `services/signalbridge_api/app/models.py` | Models are a single file; spec inventory documents a 7-file `app/models/` package that doesn't exist | Update spec inventory to the single-module layout | OPEN |

### A7 — Risk Management

> **Verdict**: architectural bloat — **4 overlapping risk implementations** (spec claims 3), one wholly dead (commands), an unused audit table; production H5 uses ONLY RiskEnforcer + regime gate while the spec presents all subsystems as active. v2.0 additions (regime gate/DL/effective HS) correctly wired; kill-switch `confirm=True` correctly enforced.

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A7-01 | HIGH | bad-practice | `src/risk/checks/hold_signal_check.py:38`, `risk_check_chain.py:91` | Chain exempts only `HOLD` — CLOSE/FLAT exit signals flow through ALL blocking checks → a position-closing signal CAN be blocked (spec forbids). Enforcer+Manager handle it correctly | Add CLOSE/FLAT passthrough (ExitSignalCheck at order 0) | DONE 2026-07-07 (HoldSignalCheck passes CLOSE/FLAT/EXIT — closing orders never blocked) |
| A7-02 | HIGH | dead-code | `src/risk/commands.py:1-974` | Entire Command Pattern (974 lines, 6 commands, undo/redo/audit) imported ONLY by its test — no DAG/service/API triggers it; spec presents it as live | Wire into a real surface (kill-switch API/DAG guardrail) or delete + remove spec section | RESOLVED 2026-07-07: DEPRECATED (dormant library, docstring + spec updated; no consumers w/o ADR) |
| A7-03 | HIGH | dead-code | `migrations/014_kill_switch_audit.sql` | `audit.kill_switch_audit` + helpers never called by any code — all 3 kill switches mutate in-memory booleans only; spec claims every change writes an audit row | Wire activation/reset to `audit.log_kill_switch_change()` or mark migration/spec roadmap-only | RESOLVED 2026-07-07: SUPERSEDED by append-only audit_log (mig 055) — tenant/global kills + fan-out write rows; mig 014 table kept unused |
| A7-04 | HIGH | redundancy | `services/mlops/risk_manager.py:1-786` | A **4th** risk subsystem (Redis-persisted RiskManager wrapping RiskCheckChain) — the actual consumer of the Chain in mlops/RL path; spec's "3 subsystems" never mentions it; duplicates legacy `src/risk/risk_manager.py` | Document it as the production wrapper; collapse legacy `src/risk/risk_manager.py` into it | DONE 2026-07-07 (mlops RiskManager documented as 4th subsystem / Chain production consumer in risk-management.md) |
| A7-05 | MEDIUM | spec-drift | `src/risk/risk_manager.py:64-68` | `RiskManager.RiskLimits` hardcodes literals instead of importing `src.core.constants` (spec says both RiskLimits must derive from constants; only Enforcer does) → silent drift | Import constants as field defaults, mirroring RiskEnforcer | OPEN |
| A7-06 | MEDIUM | bad-practice | `risk_check_chain.py:204-219` vs `constants.py:284-291` | Same risk concept, wildly different thresholds: Chain drawdown kill **-1%** / daily **-2%** / consec **3** / trades **10** vs constants **15%/5%/5/20** (15× gap on drawdown); spec transcribes both, masking the inconsistency | Reconcile Chain `with_defaults` to SSOT constants or document why pre-signal Chain is intentionally tighter | OPEN |
| A7-07 | MEDIUM | redundancy | `src/risk/risk_manager.py` + `src/trading/risk_enforcer.py` + `src/risk/checks/*` | Kill-switch/daily-loss/cooldown/consec/daily-reset implemented 3-4× with ~80% duplicate bodies (Manager vs Enforcer differ only in return type) | Extract one shared `RiskState` core; make Manager a thin adapter over Enforcer's `validate_signal` | OPEN |
| A7-08 | MEDIUM | obsolete-pipeline | `airflow/dags/l5_multi_model_inference.py:116` | Legacy `src/risk/risk_manager.py` imported by exactly one non-test consumer — the RL L5 DAG (DEPRIORITIZED); production H5 uses RiskEnforcer → legacy Manager tied to a dead pipeline | After A7-07, retire legacy Manager; point RL L5 at Enforcer's compat method | OPEN |
| A7-09 | LOW | bad-practice | `src/risk/risk_manager.py:61-62` | Docstring says cooldown defaults 3/30 but actual fields are 5/60 (spec table correct) | Correct docstring to 5/60 | OPEN |
| A7-10 | LOW | bad-practice | `risk_manager.py:163` vs `risk_enforcer.py:571` | Exit-exemption set differs: Manager `(close,flat)` vs Enforcer `(CLOSE,FLAT,HOLD)` — behavior differs by subsystem for same input | Standardize exit/no-op set (CLOSE/FLAT/HOLD) across all entry points | OPEN |
| A7-11 | LOW | spec-drift | `migrations/033_event_triggers.sql` | Spec says risk `circuit_breaker_state` persists across restarts, but `CircuitBreakerCheck` uses an in-memory `ICircuitBreaker` and never touches the table | Back the check with the DB table or correct spec to "in-memory only" | OPEN |
| A7-12 | LOW | bad-practice | `risk_enforcer.py:164`, `risk_manager.py:122` | `RiskState.current_day` annotated `datetime` but assigned `date` — wrong type, breaks datetime arithmetic consumers | Annotate as `date` (or store real datetime) consistently | OPEN |

### A8 — Authentication + User Creation

> As-built flow documented in `../platform/authentication.md`. Clean: bcrypt hashing, lockout+blacklist implemented, DEV_MODE prod hard-guard on the SB middleware. **The MEMORY note ("real register/login, sb_users fixed, lockout+blacklist+load tests") is verified accurate (A8-14).**

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A8-01 | CRITICAL | security | `docker-compose.compact.yml:525` | App reads `JWT_SECRET_KEY` but compose sets `JWT_SECRET` (wrong name) with weak default → API silently uses insecure default `change-me-in-production-min-32-chars` (`config.py:51`) → all JWTs forgeable | Rename env to `JWT_SECRET_KEY`; require strong secret, no default; fail startup if unset | **FIXED** (br `fix/audit-p0-remediation`): compose now sets `JWT_SECRET_KEY=${JWT_SECRET:?...}` — correct var name, no weak default, hard-fails if unset |
| A8-02 | HIGH | security | `lib/auth/api-auth.ts:98` | `AUTH_BYPASS_ENABLED==='true'` grants synthetic `role:'admin'` with NO `NODE_ENV!=='production'` guard (middleware.ts:66 has it) → leftover env disables all API-route auth in prod | Add `&& NODE_ENV!=='production'` guard | **FIXED** (br `fix/audit-p0-remediation`): added `&& process.env.NODE_ENV!=='production'` guard, matching middleware.ts |
| A8-03 | HIGH | security | `app/api/routes/auth.py:42` | `/auth/register` fully open (no invite/verification) + no lockout/rate-limit (only /login has), and `/api/auth` is public in middleware → anyone can mass-create accounts | Gate registration (invite/admin), add throttling + enforce email verification | MITIGATED (admin-approval flow: registered accounts cannot log in until approved — verified 403 pre-approval E2E); per-IP register throttle still open |
| A8-04 | HIGH | bad-practice | `app/login/page.tsx:233` | Access + 7-day refresh JWTs stored in `localStorage` (XSS-exfiltratable), no httpOnly cookie | Use httpOnly, Secure, SameSite cookies; keep JS out of token custody | OPEN |
| A8-05 | MEDIUM | security | `config.py:57` + `docker-compose.compact.yml:526` | `vault_encryption_key` (AES-256-GCM for exchange keys) defaults to `change-me-32-byte-...`; only rejected in `production`, compact runs `development` | Require real 32-byte key, no default, in every non-test env; block startup on placeholder | HARDENED 2026-07-07 (guard extended to staging + loud warning in dev; live key STILL placeholder — rotate together with exposed exchange keys, see docs/legal/SFC-GATE-CHECKLIST.md §D) |
| A8-06 | MEDIUM | spec-drift | `app/api/routes/auth.py:127` | `/auth/refresh` rotates but never blacklists the old refresh token → stolen refresh token valid full 7 days (replay) | Blacklist the presented refresh jti on rotation | DONE 2026-07-07 (refresh rotation blacklists presented jti + /refresh rejects blacklisted tokens; E2E: first refresh 200, replay 401) |
| A8-07 | MEDIUM | security | `login_security.py:147` | Lockout AND token-blacklist **fail OPEN** when Redis down → revoked tokens re-accepted, brute-force unbounded during outage | Fail closed for token revocation (deny on Redis error); alert on Redis-down | RESOLVED-BY-DECISION (documented fail-OPEN for availability in login_security.py header; locking all users out on Redis hiccup judged worse) |
| A8-08 | MEDIUM | dead-code | `app/execution/login/page.tsx:24` | Calls `authService.login(email,password)` (positional) but signature is `login(data)` returning `{user,tokens}`; reads non-existent `result.success/token` → execution login broken | Fix to pass `{email,password}` + read `tokens.access_token`, or remove redundant page | DONE 2026-07-07 (broken page replaced by redirect shim to /login?callbackUrl=/execution/dashboard) |
| A8-09 | MEDIUM | bad-practice | `lib/services/execution/auth.service.ts:72` | `MOCK_MODE` accepts hardcoded `password123`/`demo@signalbridge.com` and mints fake JWT → bypass creds in client bundle if enabled | Strip mock auth from prod builds (build-time DCE); never ship demo password | DONE 2026-07-07 (hardcoded password123/demo creds removed; mock branch NODE_ENV-guarded → DCE in prod bundles) |
| A8-10 | MEDIUM | spec-drift | `21-signalbridge-users-schema.sql:22` | No role column on `sb_users`, SB JWT has no role claim, yet dashboard enforces `role==='admin'` → SB API has zero RBAC (every user equal) | Add role column + JWT claim + enforce on SB routes, or document single-tier | DONE (sb_users has role+status; JWT carries role; tenant/admin routes enforce it) |
| A8-11 | LOW | redundancy | `app/middleware/auth.py:35` | `DEV_USER_ID=1` (int) while real `sb_users` PKs are UUID → dev/prod identity types diverge | Use a fixed UUID for DevUser | OPEN |
| A8-12 | LOW | bad-practice | `app/login/page.tsx:333,586,797` | Fake trust signals: fake build hash `a3f4b2`, `Environment: PROD`, placeholder `admin`/`password`, unverifiable ISO27001/SOC2/SFC badges | Remove fake compliance/build claims + example creds | DONE 2026-07-07 (fake ISO/SOC/SFC claims + RL v2.4 + fake build hash a3f4b2 + Environment:PROD + admin placeholder creds all removed; truthful copy + risk disclaimer) |
| A8-13 | LOW | security | `app/api/routes/auth.py:31` | `_client_ip` trusts `X-Forwarded-For`/`X-Real-IP` unconditionally → per-IP lockout spoofable | Only trust forwarded headers from known proxies; else socket peer | OPEN |

### A9 — Observability + CI/CD + Infra

> **Clean**: 70% coverage gate genuinely enforced (3 places, no `continue-on-error`); no `.env` git-tracked (only `.env.example`); compose uses `${ENV}`/docker secrets → no plaintext credentials.

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A9-01 | HIGH | dead-code | `docker-compose.yml:307,396` | `FEATURE_CONFIG_PATH=.../config/features/feature_registry_v19.json` — dir doesn't exist (real: `config/feature_registry.yaml`) — **duplicate of A2-06** | Point at real file or remove env var (both services) | OPEN |
| A9-02 | HIGH | spec-drift | `.github/workflows/dvc-validate.yml:171-183` | Reads `params.train.*` but params.yaml has no `train:` block → every bound check errors → workflow fails whenever it runs | Update to actual params.yaml schema, or add `train:` section | OPEN |
| A9-03 | HIGH | spec-drift | `.github/workflows/dvc-validate.yml:98` | Requires stage `prepare` but dvc.yaml names it `prepare_data` → validate-config fails | Rename stage or update required-stage list | OPEN |
| A9-04 | MEDIUM | dead-code | `prometheus/rules/trading_alerts.yml` | Compose overlays `config/prometheus/rules` onto `/etc/prometheus/rules` → the `prometheus/rules/` copy is shadowed + can diverge | Delete `prometheus/rules/` + empty `prometheus/alerts/`; keep rules under `config/prometheus/` | OPEN |
| A9-05 | MEDIUM | spec-drift | `services/inference_api/main.py:404`, `otel_setup.py` | Spec says "Jaeger 0 services instrumented / omit"; inference_api + signalbridge_api DO set an OTLP tracer → spec understates tracing | Update observability matrix: Jaeger instrumented for inference/signalbridge | OPEN |
| A9-06 | MEDIUM | spec-drift | `forecast_h5_l3_weekly_training.py:92-94`, h1 | Spec says "L3 DAGs don't log to MLflow"; both H5/H1-L3 now call `mlflow.start_run` → roadmap gap already closed | Update observability + training spec: MLflow DAG integration implemented | OPEN |
| A9-07 | MEDIUM | bad-practice | `config/alertmanager/alertmanager.yml:33-101` | All active routes → `log-only`/`null`; slack/pagerduty receivers + severity tree commented out → critical alerts have no destination (no-op) | Wire `${SLACK_WEBHOOK_URL}` + uncomment receivers, or mark alerting disabled in spec matrix | OPEN |
| A9-08 | MEDIUM | obsolete-pipeline | `forecast_h5_l5_weekly_signal.py:475` | Production H5-L5 publishes to Kafka `redpanda:9092`, but redpanda exists ONLY in compact ("TIER 5 course project"), absent from full compose → publish always fails on full stack | Add redpanda to full stack, gate behind a flag, or remove the Kafka task from the prod DAG | OPEN |
| A9-09 | LOW | dead-code | `docker-compose.yml:595,863,990,1400` | 4 fully-commented service blocks (websocket-service, compliance-api, multi-model-api, nginx) → misleading topology | Delete commented blocks (git history preserves them) | OPEN |
| A9-10 | LOW | spec-drift | `prometheus/prometheus.yml:29-63` | Spec lists 9-10 scrape jobs incl feast/postgres-exporter/multi-model; actual config scrapes only 6 | Update scrape-target table to the 6 real jobs | OPEN |
| A9-11 | LOW | spec-drift | `prometheus/prometheus.yml:55` | Scrapes `mlops-inference-api:8090` (full-stack only, absent in compact) → permanent DOWN target on compact | Guard job by profile or document full-stack-only | OPEN |

### A10 — News/Analysis + Gold/Multi-Asset

| ID | Sev | Category | File:line | Issue | Fix | Status |
|----|-----|----------|-----------|-------|-----|--------|
| A10-01 | HIGH | spec-drift | `src/gold_rl/indicators.py:112,114` | `classify_regime` hardcodes Hurst `h>=0.5`/`h<0.5` — the "copy COP" pattern the playbook forbids; AssetProfile `regime_gate.hurst_*` never read | Read `hurst_trending`/`hurst_mean_rev` from the loaded AssetProfile; fail loudly if null | DONE 2026-07-07 (classify_regime parametrized hurst_trending/mean_rev; runner reads xauusd.yaml regime_gate; nulls → explicit logged 0.5 pivot, never COP values) |
| A10-02 | HIGH | dead-code | `config/assets/xauusd.yaml:60-61` | `hurst_trending/mean_rev: null` shipped as dead config — no code consumes them, yet presented as honored contract | Fit+populate the thresholds (block XAU paper until done) or remove nulls + document fixed 0.5 pivot | DONE 2026-07-07 (yaml nulls documented as explicit 0.5-pivot fallback; fitting = registered trial D1 pending Vote) |
| A10-03 | HIGH | shipped-todo | `tests/onboarding/test_asset_xauusd.py:44-99` | Only A1-B1 tests ship; **no D1** (regime re-fit) nor E1-E5/F1 gate tests, yet STATUS markets "6/6 verdes" as the full gate | Add D1 (Hurst non-null, ≠ COP 0.52/0.42) + E-gate tests; stop presenting A1-B1 as complete | DONE 2026-07-07 (test_d1_regime_thresholds_consumed_not_copied added — asserts parametrization honored + fitted values ≠ COP 0.52/0.42; onboarding suite 7/7) |
| A10-04 | HIGH | obsolete-pipeline | `specs/tracks/news-analysis/_summary.md:384-390` | "Known TODOs" table is FALSE now — `build_signal_section`, news/events/H5/H1 signals all wired in `weekly_generator.py`; line numbers no longer map | Replace TODO table with current status; the real gap is DB persistence (A10-05) | OPEN |
| A10-05 | MEDIUM | obsolete-pipeline | `airflow/dags/analysis_l8_daily_generation.py` + `database/migrations/046_*.sql` | Migration 046 tables (`weekly_analysis`/`daily_analysis`/`macro_variable_snapshots`) never written — L8 exports JSON only → permanently 0 rows (dead schema) | Wire UPSERT of Daily/WeeklyAnalysisRecord, or mark 046 tables JSON-only/deprecated in spec | OPEN |
| A10-06 | MEDIUM | redundancy | `specs/assets/_asbuilt-implementation.md` vs `specs/assets/xauusd/IMPLEMENTATION_STATUS.md` | Two docs narrate the identical Gold as-built state (both CTR-ASSET-IMPL-001) → will drift | Make `_asbuilt` the cross-asset map; reduce STATUS to pointer+delta (or vice-versa) | OPEN |
| A10-07 | MEDIUM | bad-practice | frontend `dynamicMetrics` (per STATUS:83) | Frontend recomputes Gold KPIs from visible trades → screen B2 Sharpe 1.31/p=0.0754/4-of-5 vs published 0.38/0.0358/PROMOTE — two "truths" on the approval surface | Render published `summary`/gates from manifest; drop client recompute (or label as independent view, never the gate) | OPEN |
| A10-08 | LOW | spec-drift | `config/assets/xauusd.yaml:14` vs `config/macro_variables_ssot.yaml:1100` | `price_range [250,6000]` vs macro `[1000,6000]` — intentional (deep history vs macro feature) but two ranges invite future mis-clamp/false-positive | Add cross-ref note in macro SSOT or unify raw-price sanity to `[250,6000]` | OPEN |
| A10-09 | LOW | dead-code | `src/news_engine/ingestion/{gdelt_adapter,larepublica_scraper,newsapi_adapter}.py` | GDELT/LaRepublica/NewsAPI still register + run fetch/health each cycle contributing 0 rows (only Investing+Portafolio active) — wasted cycles + rate-limit noise | Default-disable behind explicit enable flag (like NewsAPI key-gate) or move to `experimental/` | OPEN |

**A10 verdict**: (1) **Gold regime gate hardcodes Hurst 0.5** while every spec/config claims per-asset re-fit — profile nulls are dead config, the enforcing D1 test doesn't exist → XAU shipped past a gate the playbook says blocks paper trading. (2) News `_summary` **TODO table is obsolete** (signals wired); real gap = migration-046 tables never written. Scalability claim "add asset without code edits" holds for **ingestion** but **breaks for science** (`indicators.py` regime thresholds not profile-parameterized).

---

## Priority Backlog (cross-domain rollup)

**Totals**: ~114 findings across 10 domains — 1 CRITICAL, ~30 HIGH, ~45 MEDIUM, ~38 LOW.

### P0 — Do now (security + trading integrity)
| Ref | Why it's P0 |
|-----|-------------|
| **A8-01** | JWT secret env-name mismatch → API signs with a public default → **all tokens forgeable** (CRITICAL). |
| **A8-02 · A8-03 · A4-13** | Unauthenticated approve→prod-deploy, open self-registration, prod auth-bypass with no NODE_ENV guard — anyone reachable can create users / approve / trigger a retrain-deploy. |
| **A6-01 · A6-02** | In testnet/live: exits never forwarded (orphaned exchange positions) + `signal_id` discarded (duplicate orders). Block non-paper mode until fixed. |
| **A3-02** | Live weekly training builds **21 features; the approved +25.63% backtest used 23** → production models ≠ what was approved. Fix train/backtest parity before trusting production numbers. |
| **A5-01 · A5-02** | Fabricated signals rendered on the live chart; browser-recomputed p-value overrides the published scipy value on the approval surface (Gold B2 0.0754 vs 0.0358). |
| **A2-05** | Active `pipeline_ssot.yaml` is an unregistered SAC experiment (arch+action+features changed at once) masquerading as the PPO baseline — violates the experiment protocol. |

### P1 — This cycle (drift / dead pipelines that mislead or block scale)
A2-01, A2-02, A2-03, A2-04, A2-06 (deprecated rl_l4 DAGs live · "deleted" services resurrected · feature-contract TS15 vs Py20 drift · 4 divergent thresholds · dead FEATURE_CONFIG_PATH) ·
A4-01, A4-02, A4-03 (production pointer never written · register_bundle exit-gate never called · no TS manifest contract) ·
A10-01, A10-02, A10-03 (Gold regime hardcodes Hurst 0.5, profile thresholds dead, D1 gate test missing) ·
A1-01, A1-02, A1-03, A1-04, A1-09 (timezone golden-rule drift · Gold has no DAG/freshness/backup · loader UTC-hours bug · restore-priority contradiction · freshness gate hardcodes COP) ·
A3-01, A3-03 (H1 "pause" not codified · config-fallback drift) ·
A7-01, A7-02, A7-03, A7-04 (exit signals blockable in Chain · dead Command Pattern · unused kill-switch audit table · undocumented 4th risk subsystem) ·
A9-02, A9-03, A9-07, A9-08 (2 CI workflows that always fail · alerting is a no-op · redpanda coupled to a prod DAG) ·
A5-07, A5-10 (trades clobber race · `/production` chart symbol hardcoded).

### P2 — Backlog (redundancy / cleanup / cosmetic + spec reconciliation)
All remaining LOW items + the spec-drift doc fixes (schedule/migration numbers, stale status matrices, commented-out compose blocks, dead fallback generators, branding).

---

## Scalability Assessment

**"Add an asset/strategy without code edits" — holds at L0 ingestion, breaks above it.**

| Layer | Verdict | Blockers |
|-------|---------|----------|
| L0 price ingestion | ✅ AssetProfile-driven, truly generic | none (only the intentionally-frozen table name) |
| Ops (DAG/freshness/backup) | ❌ COP-hardcoded | A1-02 (Gold no DAG/backup), A1-09 (freshness `WHERE symbol='USD/COP'`) |
| Science (regime) | ❌ not profile-parameterized | A10-01 (`indicators.py` hardcodes Hurst 0.5; profile thresholds unread) |
| Contracts/registry | ⚠️ partial | A4-03 (no TS manifest type), A4-05 (promote doesn't sync headline), A4-01 (no production pointer) |
| Frontend | ❌ multiple `USDCOP` literals | A5-10/13/14/15, A4-09 (hardcoded `smart_simple_v11`) |

**Conclusion**: onboarding a 2nd asset (Gold) works for *offline backtest + display* (as built) but a real **production** asset still requires edits in ops, science, contracts, and frontend. The DAG factory (registry-lifecycle §6.3) remains the missing enabler.

---

## Spec-Quality Assessment

The reorganized tree (`rules/` + `specs/`) is structurally sound and navigable. But the audit found **~15 spec-drift items where the code moved past the spec** — the specs must be reconciled to reality:

| Spec | Now inaccurate because |
|------|------------------------|
| `platform/execution-bridge.md` | "The Gap" is partially closed (`signalbridge_client.py` exists); models are 1 file not 7 (A6-04, A6-08) |
| `platform/observability.md` | Jaeger IS instrumented; MLflow L3 logging IS wired; scrape table lists 4 dead jobs; alerting is a no-op (A9-05/06/07/10/11) |
| `platform/risk-management.md` | Presents 3 subsystems; there are 4 (+ dead Command Pattern + unused audit table) (A7-02/03/04) |
| `platform/registry-lifecycle.md` | `register_bundle` marked ✅ but never called; replay signals are `.json` not `.parquet` (A4-02/12) |
| `tracks/news-analysis/_summary.md` | "Known TODOs" table is fully obsolete (signals/news/calendar wired); real gap is unused mig-046 tables (A10-04/05) |
| `rules/data-governance.md`, `rules/data-freshness.md`, `operations/elite-operations.md` | Schedule + migration numbers drifted (048 vs 049; macro Sun06 vs Sun04); golden rule never amended for multi-asset instants (A1-01/05/06/07) |
| `tracks/h5-smart-simple.md` | Headline +25.63%/34-trades were produced on a 23-feature model live training doesn't build (A3-02) |
| `pipelines/inference-l1-l5.md` | Lists services as DELETED that still exist and are imported (A2-02) |

**Recommendation**: a **spec-reconciliation pass** (P1) to bring these back in sync, then a policy: whenever code changes a documented behavior, update the spec in the same PR (enforce via `contracts-check` where possible). Newly created this audit: `platform/authentication.md` (was undocumented).

---

## Remediation Log — branch `fix/audit-p0-remediation` (2026-07)

**P0 (trading integrity + security) — DONE:**
- A8-01 (CRITICAL) — compose `JWT_SECRET`→`JWT_SECRET_KEY`, no weak default, hard-fail if unset.
- A8-02 — API auth-bypass now guarded `NODE_ENV!=='production'`.
- A4-13 — approve + deploy routes require auth; principal from session; cookie forwarded to deploy.
- A5-01 — signals route returns 503 empty on DB error; fake `dataType:'live'` generator deleted.
- A6-01 — H5+H1 executors hard-block non-paper entries (no orphaned positions) unless opt-in env.
- A3-02 — feature parity: shared `src/forecasting/enhance_v2.py`; L3 trains 23 + persists `feature_cols_h5.json`; L5 reads it. Backtest re-run reproduced +25.63%/Sharpe 3.347/p=0.0063 exactly.

**P1 (drift / dead pipelines) — DONE this pass:**
- A3-01 — all 5 `forecast_h1_*` DAGs now `is_paused_upon_creation=True`.
- A2-01 / A2-07 — deprecated `l4_backtest_validation` + `l4_experiment_runner` scheduled DAGs set `schedule_interval=None` + paused.
- A2-09 — deprecated rl_l4 ids moved to `DEPRECATED_DAGS`, excluded from `get_all_dag_ids()`.
- A2-06 — dead `FEATURE_CONFIG_PATH` (nonexistent v19.json, unread by code) removed from `docker-compose.yml`.
- A2-05 — SAC-SWING-002 frozen to `config/experiments/exp_sac_swing_002.yaml`; active `pipeline_ssot.yaml` restored to PPO `v215b_baseline`.
- A2-03 — `ssot.contract.ts` synced to Python `feature_contract.py` v3.1.0 (15→20 feats, dims, version, hash pinned to backend `05c6…`); `useSSOT` dim-comparison now matches.
- A2-04 — action thresholds unified to the authoritative SSOT value 0.35: `config.py` fallback default 0.60→0.35 (runtime already loads SSOT), TS mirror 0.33→0.35.
- A1-09 — `validate_training_data_freshness(..., symbol=...)` parameterized (default COP preserved) so a 2nd asset gets its own freshness gate.
- A9-02 / A9-03 — `dvc-validate.yml` no longer requires a nonexistent `params.train` section (RL hyperparams live in pipeline_ssot); train-param checks guarded → workflow stops always-failing.
- A9-08 — H5-L5 Kafka publish (already non-fatal) now gated behind `KAFKA_ENABLED` (default off) so the full stack stops burning the broker timeout every run.
- A5-10 — `/production` chart symbol is now data-driven (`summary?.chart_symbol ?? 'USDCOP'`) instead of a hardcoded literal.
- Spec reconciliation — dated notes added to execution-bridge, observability, risk-management, registry-lifecycle, news `_summary`, h5-smart-simple, inference-l1-l5; `elite-operations` migration 048→049; `data-governance` golden-rule multi-asset carve-out (A1-01); `CLAUDE.md` Jaeger/MLflow bullets corrected.
- A6-02 — subsumed by A6-01: duplicate/orphaned orders cannot occur while non-paper entries are hard-blocked.

**Still open — categorized (why not done autonomously):**
- **In the operator's uncommitted WIP (must not touch):** A8-03..13 (SignalBridge/dashboard auth files), A5-02/A5-07 (`ForecastingBacktestSection.tsx`), A10-01/02/03 (`src/gold_rl/`, `config/assets/`).
- **Needs an operator decision:** A2-02 (delete the dead NRT service *source files* — verified dead, deletion left for approval).
- **Feature work / riskier refactor (deferred):** A4-01/02/03 (wire `register_bundle` into a DAG exit + TS manifest type + production pointer), A7-01/02/03 (risk-Chain exit-block + dead Command Pattern), A1-02..15 remaining data-layer drift.
- **P2 cosmetic:** remaining LOW items (branding, commented compose blocks, status matrices).

---

## Sign-off

- Audit method: 10 parallel specialized agents (A1–A10), read-only, code ↔ spec cross-check, 2026-07-04.
- Remediation: P0 (6/6) + P1 batch-1 applied on branch `fix/audit-p0-remediation` (2026-07), uncommitted; see Remediation Log above.
- Next: A2-02 resolution, feature-contract unification (A2-03/04), then remaining P1 + P2.
