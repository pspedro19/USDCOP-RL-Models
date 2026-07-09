# SDD Spec: Multi-Asset Implementation (AS-BUILT, today)

> **Responsibility**: Descriptive *as-built* map of how the system is **actually implemented today**
> as a **multi-asset** platform (USD/COP in production + XAU/USD Gold onboarded through backtest and
> visible on the web + **BTC/USDT crypto onboarded end-to-end and visible on the web** — AssetProfile,
> migration 052, canonical Binance data (no key), spot-exposure science stack, 3 published bundles;
> see `assets/btcusdt/`). Documents every layer — data engineering, data science, backend, frontend —
> and the cross-cutting **session / timezone / annualization** rules that make a session-bounded asset
> (COP, 5h COT), a near-24h asset (Gold, metals), and a **fully 24/7 asset (BTC, crypto)** coexist
> correctly.
>
> This is the companion to two PROPOSED specs that are now PARTIALLY REALIZED:
> - `_onboarding-playbook.md` — the prescriptive playbook (AssetProfile contract). Now the
>   `AssetProfile` EXISTS on disk and Gold has been onboarded through Stage F (backtest+gates+web).
> - `registry-lifecycle.md` — the dynamic registry. Now has 2 assets, 5 strategies.
>
> Contract: CTR-ASSET-IMPL-001
> Version: 1.0.0
> Date: 2026-07-04
> Status: IMPLEMENTED (COP production + Gold rule-based, backtest→web) · Verified against disk.
> Constraint honored throughout: **everything is ADDITIVE** — no COP behavior or existing
> frontend↔backend contract was modified.

---

## 0. One-Paragraph As-Built Model

The system is no longer single-asset "by construction". A new keystone contract — **`AssetProfile`**
(`src/contracts/asset_profile.py` + `config/assets/<id>.yaml`) — parameterizes everything that used
to be hardcoded to USD/COP: `symbol`, `chart_symbol`, **session/timezone**, `bars_per_day`,
`bars_per_year`, `price_range`, macro drivers, and regime thresholds. USD/COP itself is now
`config/assets/usdcop.yaml` (not a special case). One **asset-generic ingestion script**
(`scripts/ingest_asset_ohlcv.py`) reads a profile and downloads/aligns/audits/UPSERTs OHLCV for any
asset. Gold's 5-min bars live in the existing multi-pair table `usdcop_m5_ohlcv` (symbol `XAU/USD`);
Gold's daily deep history lives in a new multi-asset table `asset_daily_ohlcv` (migration 051). A
rule-based science stack (`src/gold_rl/`) computes features → 4 regimes with hysteresis → vol-target
risk layer → baselines B1/B2 + regime-gated strategy → honest backtest (block-bootstrap p-value,
regime attribution, 5 gates). Results are published as **immutable versioned bundles** to the
**dynamic registry** (`registry.json` + per-strategy `manifest.json`), which the Next.js dashboard
reads to render **any** asset/strategy/version — Gold shows with `chart_symbol="XAUUSD"`, a populated
trades table, and per-version replay. **COP is untouched** and still the default view.

**Load-bearing truth**: the platform is now genuinely 2-asset (3 with BTC) for *offline* backtest +
web display, and Gold/BTC are now **DAG-driven** — a DAG factory (`airflow/dags/asset_pipeline_factory.py`,
SSOT `config/assets/pipelines.yaml`, CTR-ASSET-PIPELINE-001) emits `asset_xauusd_pipeline_weekly` (Sun
01:45 COT) + `asset_btcusdt_pipeline_weekly` (Sun 02:00 COT), each running `l0_ingest → l4_backtest_publish
→ l6_verify_registry`. Live intraday execution and RL (SPEC-05/08) remain COP-only / proposed. The
session/timezone handling (§4) is the piece most easily gotten wrong and is fully per-asset today.

---

## 1. The AssetProfile Keystone (`CTR-ASSET-PROFILE-001`)

**File**: `src/contracts/asset_profile.py` — a standalone leaf module (imports `yaml` only, no ML
stack) so it is cheap to load in DAGs, ingestion, and fast tests.

**Dataclasses** (all `frozen=True`):

| Class | Purpose | Key fields |
|-------|---------|-----------|
| `AssetProfile` | Root SSOT per asset | `asset_id`, `symbol`, `chart_symbol`, `asset_class`, `price_range`, `session`, `data_source`, `macro_drivers`, `regime_gate`, `strategy_id`, `pipeline_type`, `timeframe` |
| `SessionSpec` | Trading session (replaces COT 08:00-12:55 / bars=78 hardcodes) | `mode`, `timezone`, `days`, `open`, `close`, `bars_per_day`, `bars_per_year`, `trading_days_per_year`, `weekend_flat`, `forced_close`, `is_24x7` |
| `DataSourceSpec` | Provider wiring | `provider`, `provider_symbol`, `interval`, `needs_tz_convert`, `seed_file`, `daily_provider`, `daily_seed_file` |
| `MacroDriver` | One T-1 lagged macro driver | `series_id`, `feature`, `sign`, `role` |
| `RegimeGateSpec` | Hurst regime gate (logic reused, thresholds **per-asset**) | `hurst_lookback`, `hurst_trending`, `hurst_mean_rev`, `hysteresis_dwell_days` |

**Validation** (`validate()` → list of problems, backs onboarding test A1):
- `asset_id` is a slug (no `/`); `symbol` looks like `XAU/USD`; `chart_symbol == symbol.replace("/","")`.
- `asset_class ∈ {fx, crypto, commodity, equity_index}`; `price_range` is `(lo<hi)`.
- `session.mode ∈ {exchange_hours, metals, 24x7}`; `session.days` non-empty.

**Loaders**: `load_asset_profile(asset_id)` (validates), `list_assets()`, `profile_from_dict()`,
`profile_to_dict()`. `safe_name` → `"xauusd"`; `in_price_range(price)` for audits.

**On disk today**: `config/assets/usdcop.yaml`, `config/assets/xauusd.yaml`.

---

## 2. Data Engineering (as implemented)

### 2.1 Asset-generic ingestion — `scripts/ingest_asset_ohlcv.py` (`CTR-L0-ASSET-INGEST-001`)

Driven ENTIRELY by an `AssetProfile`. `--asset xauusd` runs the whole flow; no symbol is hardcoded.

```
_load_asset_profile → _paginate_back (5-min TwelveData, key rotation, back to intraday floor)
   → _clean → _filter_session (§4)  → _audit → measure bars_per_day → _write_seed → UPSERT
_paginate_back (daily TwelveData deep 2004+) → _daily_to_nyclose (§4.3, DST-correct)
   → _investing_daily (cross-check, best-effort) → cross-source agreement + gap-fill → seed → UPSERT
```

| Concern | Implementation |
|---------|----------------|
| API keys | 8-key rotation (`TWELVEDATA_API_KEY_1..8` + `_KEY`), `_KeyRotator` |
| Pagination | `_paginate_back` walks `end_date` backward until the API stops returning older bars or the floor is reached (TwelveData 5-min only serves a recent window; daily reaches 2004) |
| 5-min tz | Fetched in UTC (`needs_tz_convert=false` for both COP-native-COT and Gold); stored as UTC instants |
| Daily normalization | `_daily_to_nyclose` → 17:00 America/New_York instant in UTC (§4.3) |
| Investing.com | `_investing_daily` — cloudscraper browser session + `domain-id`/`Referer` headers + annual chunks; **graceful degradation** on 403/429 (TwelveData daily is authoritative) |
| Cross-source audit | overlap merge → `median_abs_diff_pct` / `max_abs_diff_pct` + `flag` (REVIEW if >2% median) |
| Gap-fill | union with TwelveData authoritative (`keep="first"`), Investing fills only TD-missing dates |
| Quality audit | `_audit`: dupes, NaN OHLC, `high<low`, OHLC-integrity, out-of-range closes (>2% ⇒ problem) |
| Seeds | `_to_seed_schema` → `[time,symbol,open,high,low,close,volume]` (same as FX pairs) |
| Idempotency | `_upsert` → `INSERT … ON CONFLICT (time,symbol) DO UPDATE …`, `page_size=1000` |

### 2.2 Storage — scalable tables, no silos

| Data | Table | Notes |
|------|-------|-------|
| 5-min OHLCV (all pairs + Gold) | `usdcop_m5_ohlcv` | Multi-symbol by `PRIMARY KEY (time, symbol)` (migration 040). Gold plugs in as `symbol='XAU/USD'` → 4 symbols coexist. Table name kept (renaming = 50+ file edits). |
| Daily OHLCV (multi-asset) | `asset_daily_ohlcv` | **New**, migration **051**. `(time, symbol)` PK, indexes, best-effort TimescaleDB hypertable, view `asset_daily_coverage`. |

`source` column records provenance (`twelvedata_xauusd`, `twelvedata_daily`). No `asset_id` column is
needed — `symbol` is the asset dimension.

### 2.3 Seeds (Git-tracked restore + local training)

`seeds/latest/xauusd_m5_ohlcv.parquet` (~976 KB), `seeds/latest/xauusd_daily_ohlcv.parquet` (~201 KB),
identical schema to the FX seeds. Regenerate: `python scripts/ingest_asset_ohlcv.py --asset xauusd`.

### 2.4 What was ingested (evidence, 2026-07-03)

- 21,558 5-min bars (recent intraday window; measured median **288 bars/day** ⇒ metals ~24h).
- 6,057 daily bars **2004→2026** (TwelveData 5,992 + Investing filled 65). Cross-source median diff 0.61%.
- All audits clean (0 dup / 0 NaN / 0 integrity). DST-correct daily close.

---

## 3. Data Science (as implemented — rule-based per the honest-gate thesis)

**Package**: `src/gold_rl/` (asset-generic math; no COP assumptions).

| File | Exports | Notes |
|------|---------|-------|
| `indicators.py` | `wilder_atr`, `wilder_adx`, `hurst_rs`, `build_daily_features`, `classify_regime`, `REGIME_RISK_MULT`, `regime_transitions_per_year` | Wilder EMA (not pandas ewm); Hurst rescaled-range; **4 regimes** COMPRESSION / TREND / STRETCHED / EVENT with **hysteresis dwell≥4** (regimes last weeks → low churn) |
| `strategies.py` | `vol_target_size`, `direction_long_only`, `direction_trend_follower`, `direction_regime_gated`, `STRATEGIES`, `build_positions` | Direction {-1,0,+1} decoupled from SIZE (vol-targeting). Position is **causal** (`shift(1)`) |
| `backtest.py` | `compute_returns`, `metrics`, `block_bootstrap_pvalue`, `extract_trades`, `regime_attribution`, `evaluate_gates`, `recommendation`, `run_backtest` | Realistic cost model; honest stats; dashboard-ready summary+trades |

### 3.1 Risk layer (the alpha, STRATEGY §4)
`vol_target_size`: `size = target_vol / realized_vol_20`, scaled by `regime_risk_mult`, **vol floor 6%**,
**leverage cap 1.5x**, `target_vol=0.10`. When gold gets violent, size shrinks automatically. (Loose
tuning — cap 3x, floor 3% — produced a 52% drawdown; retuned to the values above.)

### 3.2 Baselines are obligatory (STRATEGY §6)
- **B1** `gold_long_only_b1` — long-only vol-targeted (`rule_based`).
- **B2** `gold_trend_b2` — trend-follower: long when `close>SMA100 & ADX>25` (`rule_based`).
- Candidate `gold_regime_gated_v1` — secular trend rider (`close>SMA200`), flat in EVENT/STRETCHED
  (`hybrid`). A strategy is only promotable if it **beats BOTH baselines** — RL (SPEC-05/08) is
  deferred behind this published floor.

### 3.3 Honest statistics
- Cost model: **2 bps** turnover + **2.5%/yr** overnight swap on held exposure (`COST_BPS`, `SWAP_ANNUAL`).
- `block_bootstrap_pvalue`: moving-block bootstrap, **fixed `np.random.default_rng(seed=42)`**,
  `block=20` (preserves ~monthly autocorrelation), `n_boot=5000`; one-sided H0 no positive edge.
- `regime_attribution`: PnL / Sharpe / share per regime — the single most informative diagnostic.
- `evaluate_gates`: the same 5 approval gates as COP (return, Sharpe, max-DD, trades, p<0.05) = Vote 1/2.

### 3.4 Results (OOS, cap $10k)
B1 p=0.0094 · **B2 +61.1% Sharpe 0.38 p=0.0358 → PROMOTE** · regime-gated p=0.0152.

---

## 4. Session · Timezone · Annualization — the "multiplier" (the correctness core)

> **This is the concern you raised.** COP 5-min bars are session-bounded (08:00-12:55 COT, 60
> bars/day). Gold does NOT follow that schedule — it is metals ~23h weekday. If Gold were forced
> through COP's window or annualized with COP's `bars_per_year`, every metric would be wrong. Today
> the session and the annualization multiplier are **per-asset, sourced from `AssetProfile.session`**.

### 4.1 Per-asset session — enforced in ingestion (`_filter_session`)

```python
mask = local.dt.dayofweek.isin(session.days)                 # drop weekends (both assets)
if session.mode == "exchange_hours" and session.open/close:  # COP ONLY
    mask &= (mins >= open) & (mins <= close)                 # cut the 08:00-12:55 window
# metals / 24x7 => NO intraday window cut, only the weekend mask
```

| Aspect | USD/COP (`exchange_hours`) | XAU/USD (`metals`) | BTC/USDT (`24x7`) |
|--------|---------------------------|--------------------|-------------------|
| Storage tz | `America/Bogota` (golden rule) | `UTC` (honest for a global asset) | `UTC` (crypto-native) |
| Session window | 08:00–12:55 COT, Mon-Fri | ~22:00→21:00 UTC (18:00→17:00 ET), Mon-Fri, no intraday cut | **none** — 24h × 7d, no window, no weekend mask |
| Intraday cut applied? | **Yes** (5h window) | **No** (only weekends dropped) | **No** (nothing dropped; `is_24x7`) |
| Measured 5-min bars/day | 60 | **288** | **288** (all 7 days) |
| `bars_per_day` (profile) | 60 | 288 | 288 |
| `bars_per_year` (profile) | 15660 | **72000** | **105120** (288×365) |
| `trading_days_per_year` | 261 | 250 | **365** (√365 annualization) |
| Daily close convention | (COP daily = COT) | **17:00 America/New_York** | **00:00 UTC** |
| `forced_close` | friday | friday (`weekend_flat` avoids Sunday gap) | **none** (`weekend_flat:false`, continuous exposure) |
| Regime Hurst thresholds | 0.52 / 0.42 (COP-fit, in production) | **re-fit** (null until fitted; never copy COP's) | **re-fit** (null; + native HMM in `design/`) |

### 4.2 The annualization multiplier rule (mandatory)

Sharpe/vol annualization MUST use the **asset's own** multiplier, never a global constant:

```
sharpe_ann = mean(r) / std(r) * sqrt(profile.session.bars_per_year_or_trading_days)
```

- The Gold daily backtest (`src/gold_rl/backtest.py`) annualizes with `TRADING_DAYS = 252` because it
  operates on **daily** bars — correct for a daily strategy.
- An **intraday** Gold strategy MUST annualize with `bars_per_year = 72000` (288×250), **NOT** COP's
  19656/15660. Reusing COP's constant would overstate Sharpe by ~√(72000/19656) ≈ 1.9×.
- COP's RL/H1/H5 paths keep their existing `bars_per_year: 19656` from `pipeline_ssot.yaml` — that is
  COP's full-day 5-min count and stays valid for COP only.
- **BTC/USDT (24/7)** annualizes with **√365** (`trading_days_per_year: 365`) for daily strategies and
  `bars_per_year: 105120` (288×365) for intraday. A crypto asset trades every weekend, so using COP's
  261 or Gold's 250 trading days would **understate** the annualization. This is enforced by the
  `AssetProfile.session` of `config/assets/btcusdt.yaml`; see `assets/btcusdt/specs/SPEC-13 §5 (24/7)`.

### 4.3 DST-correct daily close (`_daily_to_nyclose`)
Anchor the **trading date in UTC** (`tz_convert("UTC").normalize()`), build the **naive** 17:00 wall-clock
for that date, THEN localize to ET — never add a `Timedelta` to a tz-aware midnight (that adds absolute
UTC duration and lands at 16:00/18:00 ET on the two DST-transition days per year). Localized with
`nonexistent="shift_forward", ambiguous=True`, then converted to UTC. **Fixed 2026-07:** the date used to
be anchored with `tz_convert("America/New_York").normalize()`, which on a 00:00-UTC TwelveData bar snapped
to the *previous* ET day → every daily bar shifted one calendar day onto **Sunday** (the "Gold weekday
anomaly"). UTC-anchoring is correct for both the TwelveData (00:00 UTC) and Investing (05:00 UTC) sources.
The daily path also now runs `_filter_session` + the shared **OHLCV validator**
(`src/data_quality/ohlcv_validators.py`, weekday/gap/tz/OHLC gate). Onboarding test A4 + the validator
regression tests (`tests/data_quality/test_ohlcv_validators.py`) guard this.

### 4.4 Cross-asset comparison caveat
Do NOT compare COP (60 bars/day, session-bounded) and Gold (288 bars/day, ~24h) returns or Sharpes as
if on the same clock. Compare within an asset, or normalize by the asset's own `bars_per_year`. The
registry keeps each strategy's metrics under its own bundle — the frontend never mixes them.

---

## 5. Backend — Dynamic Registry (as implemented)

**Contracts**: `src/contracts/strategy_manifest.py` (`StrategyBundleManifest`, `RegistryIndex`,
`BundlePublisher`, `RegistryBuilder`, `LegacyBundleAdapter`).

| Piece | Behavior |
|-------|----------|
| `BundlePublisher` | Writes **immutable content-addressed** bundles keyed by `(strategy_id, model_version, year)` under `public/data/strategies/<sid>/backtests/<version>/`. Never overwrites; a new version coexists. Refreshes `registry.json` + `manifest.json`. |
| `registry.json` | Dynamic index: today **3 assets** (`usdcop`, `xauusd`, `btcusdt`), **8 strategies** (2 COP + 3 Gold + 3 BTC). Each entry carries active-version headline (return/sharpe/p_value) so the selector shows real numbers without an N+1 fetch. |
| `manifest.json` | Per-strategy self-describing bundle: `symbol`, `chart_symbol`, `backtests[]`, `production`, `approval`, `model_versions[]`, `capabilities.replay`. |

**APIs** (Next.js, additive, unauthenticated like other production routes):

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/registry` | GET | Registry index; falls back to synthesizing from legacy `strategies.json` |
| `/api/strategies/[strategyId]/manifest` | GET | One strategy's manifest |
| `/api/registry/promote` | POST `{strategy_id, version, status?}` | Flip the active version |

**Publishers (one runner per asset, all additive — never touch each other's bundles):** these run
either manually or as the `l4_backtest_publish` task of the per-asset DS-cycle DAG (§0 load-bearing truth;
`asset_pipeline_factory.py`), which first refreshes the daily seed (`l0_ingest`, graceful) and afterwards
asserts the registry (`l6_verify_registry`).
- **Gold**: `scripts/run_gold_pipeline.py` → Gold daily seed → features → regime → backtest
  B1/B2/regime-gated → `BundlePublisher.publish(... asset_id="xauusd", symbol="XAU/USD" ...)`.
- **BTC**: `scripts/pipeline/run_btc_pipeline.py` → BTC daily seed → features → regime → backtest
  B1/B2/S3 → `BundlePublisher.publish(... asset_id="btcusdt", symbol="BTC/USDT" ...)`. BTC's
  `btcusdt` display metadata is registered in `strategy_manifest._DEFAULT_ASSETS` (Bitcoin / BTCUSDT / crypto).

See `_strategy-science.md` for how each strategy is built/sized/judged, and the per-asset results.

---

## 6. Frontend — Dynamic rendering (as implemented)

**Component**: `components/production/ForecastingBacktestSection.tsx` (Next.js App Router).

| Behavior | Implementation |
|----------|----------------|
| Discovery | Reads `/api/registry` → builds a custom button-dropdown StrategySelector (COP + Gold + **Bitcoin**), grouped by asset |
| Chart symbol | `symbol={manifest?.chart_symbol \|\| 'USDCOP'}` into `TradingChartWithSignals` — Gold renders as **XAUUSD**, BTC as **BTCUSDT**, never a hardcoded literal |
| Version dropdown | Lists `manifest.model_versions[]`; "Promover a activa" calls `/api/registry/promote` |
| Per-version replay | **Client-side (Option A)**: selecting a version fetches that bundle's `summary`+`trades` and overrides React state; existing replay machinery + chart react automatically (no SSE needed) |
| Replay window | `useEffect` sets `replayStart/EndDate` from `trades` min/max → adapts to each backtest's span (Gold shows 2004→2026) |
| No-clobber fix | `loadStrategyData` (legacy) uses **exact strategy_id match only** and never overwrites `summary`/`trades` with null/`[]` when the legacy file is absent (Gold has no legacy file) — this fixed the "0 operaciones" race without regressing COP |

**Known display caveat (documented, not a blocker)**: the frontend `dynamicMetrics` **recomputes**
KPIs/gates from the *visible trades* (trade-level basis), which differs from the published `summary`
(daily-return basis). E.g. B2 shows Sharpe 1.31 / p=0.0754 (4/5) on-screen vs summary Sharpe 0.38 /
p=0.0358 (PROMOTE). Both are honest; reconciling (display the published summary) is a polish item.

**Candle backdrop**: Gold's candle chart shows "Database unavailable" because the market API is
COP-hardcoded and host→DB is closed; trade markers + replay work off the trades JSON regardless.

---

## 7. What is asset-generic vs still COP-coupled (today)

| Layer | Asset-generic NOW | Still COP-coupled |
|-------|-------------------|-------------------|
| Price data | `usdcop_m5_ohlcv` (multi-symbol), `asset_daily_ohlcv`, ingestion, seeds, `AssetProfile` | table *name* `usdcop_m5_ohlcv` (cosmetic) |
| Session/tz/annualization | `AssetProfile.session` + `_filter_session` + per-asset `bars_per_year` (COP 15660, Gold 72000, **BTC 105120 / √365**) | RL/H1/H5 `pipeline_ssot.yaml bars_per_year:19656` (correct for COP only) |
| Science | `src/gold_rl/` + `src/btc_strategy/` (pure math, parallel modules), regime gate logic | COP `forecast_h5_*` tables have no `symbol` column; H5 executor week-lifecycle Mon/Fri COT |
| Registry/backend | `registry.json`, manifests, `BundlePublisher`, 3 APIs, chart `symbol` prop | live market API (`/api/market/realtime-price`) hardcodes USDCOP |
| Frontend | StrategySelector, version replay, chart symbol from manifest | landing/hub branding "USD/COP" |
| Execution | — | SignalBridge + DAG executors are COP-only; Gold + BTC are backtest+web only |

---

## 8. File Inventory (added/changed for multi-asset, all additive)

| Path | Role |
|------|------|
| `src/contracts/asset_profile.py` | AssetProfile contract + loaders |
| `config/assets/usdcop.yaml`, `config/assets/xauusd.yaml` | Asset SSOTs |
| `scripts/ingest_asset_ohlcv.py` | Asset-generic OHLCV ingestion |
| `database/migrations/051_asset_daily_ohlcv.sql` | Multi-asset daily table + coverage view |
| `src/gold_rl/{__init__,indicators,strategies,backtest}.py` | Gold science stack |
| `scripts/run_gold_pipeline.py` | Gold E2E runner → publishes bundles |
| `tests/onboarding/test_asset_xauusd.py` | Onboarding tests A1–B1 (6, green) |
| `seeds/latest/xauusd_{m5,daily}_ohlcv.parquet` | Gold seeds |
| `usdcop-trading-dashboard/public/data/{registry.json,strategies/xauusd*}` | Registry + Gold bundles |
| `components/production/ForecastingBacktestSection.tsx` | Selector + version replay + no-clobber fix (edited additively) |
| `.claude/specs/assets/xauusd/**` | Gold spec package + `IMPLEMENTATION_STATUS.md` |
| `config/assets/btcusdt.yaml` | **BTC AssetProfile** (crypto, 24/7 UTC, √365) — keystone |
| `database/migrations/052_crypto_native_data.sql` | **Crypto-native tables** (on-chain/funding/flows/events/exposure) — additive |
| `scripts/data/ingest_btc_ohlcv.py` | **BTC ingestion** — canonical Binance public klines (no key) + TwelveData fallback; UTC 00:00 daily |
| `src/btc_strategy/{__init__,indicators,strategies,backtest}.py` | **BTC science stack** — spot `exposure∈[0,1]`, √365, crypto cost model; new parallel module (does not touch `src/gold_rl`) |
| `scripts/pipeline/run_btc_pipeline.py` | **BTC E2E runner** → backtest B1/B2/S3 → publishes bundles |
| `seeds/latest/btcusdt_{daily,m5}_ohlcv.parquet` | BTC seeds (daily 3,245 bars 2017→2026; 5m 288/day) |
| `usdcop-trading-dashboard/public/data/strategies/btc_*` | 3 published BTC bundles + manifests (registry now 3 assets / 8 strategies) |
| `tests/onboarding/test_asset_btcusdt.py` | BTC onboarding tests A1–C1 — **7/7 green** with real seeds |
| `.claude/specs/assets/btcusdt/**` | BTC spec package: `README` + `SPEC-13` (integration) + roadmap/status + `design/` (26 files) |

---

## 9. Reproduce

```bash
# 1. Ingest Gold (5-min + daily deep history, idempotent)
python scripts/ingest_asset_ohlcv.py --asset xauusd --daily-start 2004-01-01

# 2. Onboarding tests
python -m pytest tests/onboarding/test_asset_xauusd.py -q

# 3. Science + publish bundles (additive; COP untouched)
python scripts/run_gold_pipeline.py

# 4. Open /dashboard → strategy selector → "Gold · Trend-follower Daily (B2)"
```

---

## 10. Cross-References

| Concern | Spec |
|---------|------|
| Prescriptive onboarding playbook + AssetProfile §2 | `_onboarding-playbook.md` |
| Dynamic registry contracts (A/B/C/D) | `registry-lifecycle.md` |
| As-built architecture (whole system) | `architecture-overview.md` |
| L0 data governance (OHLCV/macro, timezone golden rule) | `data-governance.md` |
| Strategy/trade/gate schemas | `strategy-contract.md`, `approval-gates.md` |
| Dashboard artifact schemas | `dashboard-integration.md` |
| Gold spec package + live status | `.claude/specs/assets/xauusd/` + `.claude/specs/assets/xauusd/IMPLEMENTATION_STATUS.md` |

---

## DO NOT

- Do NOT string-replace "USDCOP" to add an asset — add a `config/assets/<id>.yaml` and read the profile.
- Do NOT apply COP's 08:00-12:55 session window to a non-`exchange_hours` asset — `_filter_session`
  only cuts the intraday window in `exchange_hours` mode; metals/24x7 keep the full day.
- Do NOT annualize an intraday Gold metric with COP's `bars_per_year` (19656) — use the asset's own
  (`AssetProfile.session.bars_per_year`, 72000 for Gold) or Sharpe is overstated ~1.9×.
- Do NOT compare COP and Gold Sharpes/returns as if on the same clock (60 vs 288 bars/day).
- Do NOT copy COP's Hurst regime thresholds (0.52/0.42) to Gold — re-fit them (they are null until fit).
- Do NOT add a `Timedelta` to a tz-aware midnight for the NY daily close — build naive 17:00 then
  localize (DST hazard, guarded by test A4).
- Do NOT overwrite a published bundle — `BundlePublisher` writes a new immutable `(version, year)`.
- Do NOT let `loadStrategyData` clobber `summary`/`trades` with null/`[]` for an asset that has no
  legacy file — exact strategy_id match + no-clobber is what keeps Gold's trades table populated.
- Do NOT treat Gold as production-executable — it is backtest + web display only today.
