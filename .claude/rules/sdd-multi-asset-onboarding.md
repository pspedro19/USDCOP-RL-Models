# SDD Spec: Multi-Asset Onboarding (Add a New Tradeable Asset)

> **Responsibility**: Authoritative, spec-driven + test-driven playbook for onboarding a NEW
> tradeable asset (e.g. **Bitcoin `BTC/USD`**, **Gold `XAU/USD`**) into the trading system that
> today trades only **USD/COP**. Defines the `AssetProfile` contract that must parameterize
> everything currently hardcoded to COP, the exact files to touch, the onboarding stages, and
> the TDD acceptance test-contract a new asset MUST pass before it can reach paper trading.
>
> Read `sdd-architecture-overview.md` first (the as-built map). This spec turns that map into
> a repeatable procedure.
>
> Contract: CTR-ASSET-ONBOARD-001
> Version: 1.0.0
> Date: 2026-07-03
> Status: PROPOSED (net-new capability; `AssetProfile` does not yet exist on disk)

---

## 1. Problem Statement

The system is **single-asset by construction**. The L0 price table (`usdcop_m5_ohlcv`) is
already multi-symbol (PK `(time, symbol)`, migration 040), and USD/MXN + USD/BRL coexist in it —
**but only as auxiliary cross-pair leaders that enrich the COP model, not as tradeable assets**.
Everything above L0 (macro drivers, feature contract, session/timezone, forecast DB tables,
strategy config, dashboard) hardcodes USD/COP.

Onboarding a new asset = **promoting a `symbol` from "aux/absent" to "primary tradeable"**.
That crosses three difficulty tiers (Easy / Moderate / Hard, per `sdd-architecture-overview.md` §7).
This spec makes those crossings explicit and testable.

**Two reference assets are used throughout:**
- **Gold `XAU/USD`** — "moderate": a commodity with a near-24h session; Gold is *already* a
  macro driver in `config/macro_variables_ssot.yaml` (`comm_metal_gold_glb_d_gold`), so only its
  intraday OHLCV feed is missing. FX-like enough to reuse most COP machinery with a re-tuned session.
- **Bitcoin `BTC/USD`** — "hard": **24/7**, no macro driver exists, and its 24/7 nature breaks
  the session-length, `bars_per_day`, weekend-exclusion, Friday-close, and annualization
  assumptions system-wide. BTC is the stress-test that forces a truly asset-generic design.

---

## 2. The `AssetProfile` Contract (net-new SSOT)

The core deliverable of any onboarding is a frozen `AssetProfile` — ONE YAML that parameterizes
every COP-hardcoded value. Proposed location: `config/assets/<asset_id>.yaml`.
Proposed Python contract: `src/contracts/asset_profile.py` (`AssetProfile` dataclass +
`load_asset_profile(asset_id)`), mirrored by `lib/contracts/asset-profile.contract.ts`.

```yaml
# config/assets/xauusd.yaml  (Contract: CTR-ASSET-PROFILE-001)
asset_id: xauusd                 # slug; used in file paths, table partitions, strategy_id suffix
symbol: "XAU/USD"                # exchange/provider symbol
display_name: "Gold"
asset_class: commodity           # fx | crypto | commodity | equity_index
quote_ccy: USD
base_ccy: XAU

# --- price sanity (replaces SYMBOL_PRICE_RANGES) ---
price_range: [1000.0, 6000.0]
decimals: 2
tick_size: 0.01

# --- SESSION (replaces the hardcoded COT 08:00-12:55 assumptions) ---
session:
  mode: metals                   # exchange_hours | metals | 24x7
  timezone: "America/New_York"   # profile-owned; NOT global America/Bogota
  days: [0,1,2,3,4]              # 0=Mon; crypto = [0..6]
  open: "18:00"                  # Sun 18:00 ET metals open (see mode-specific notes)
  close: "17:00"                 # Fri 17:00 ET
  bars_per_day: 276              # 23h * 12 (5-min); MUST equal observed median (test-enforced)
  bars_per_year: 69000           # for Sharpe annualization; = bars_per_day * trading_days
  trading_days_per_year: 250
  weekend_flat: true             # BTC=false (holds over weekend)
  forced_close: friday           # BTC=none (continuous); metals=friday

# --- DATA SOURCE (replaces per-symbol SYMBOL_CONFIG dicts) ---
data_source:
  provider: twelvedata
  provider_symbol: "XAU/USD"
  interval: "5min"
  needs_tz_convert: true         # provider tz -> profile tz
  seed_file: "seeds/latest/xauusd_m5_ohlcv.parquet"

# --- MACRO DRIVERS (replaces the COP driver set; see §3) ---
macro_drivers:                   # ordered; each maps DB column -> feature name (T-1 lagged)
  - {db_col: "fxrt_index_dxy_usa_d_dxy",       feature: "dxy_close_lag1",       sign: "-"}
  - {db_col: "finc_bond_yield10y_usa_d_ust10y", feature: "ust10y_close_lag1",    sign: "-"}  # real-yield proxy
  - {db_col: "volt_vix_usa_d_vix",             feature: "vix_close_lag1",       sign: "+"}
  - {db_col: "comm_oil_brent_glb_d_brent",     feature: "brent_close_lag1",     sign: "+"}  # inflation proxy
# cross_asset_leaders: peers whose lead/lag informs the model (COP used MXN/BRL). Optional.
cross_asset_leaders: []          # gold has no clean intraday LATAM-style leader; leave empty

# --- REGIME GATE (logic reusable; thresholds are per-asset, must be re-tuned) ---
regime_gate:
  hurst_lookback: 60
  hurst_trending: null           # MUST be re-fit on this asset's history (do NOT copy COP 0.52)
  hurst_mean_rev: null

# --- STRATEGY BINDING ---
strategy_id: "smart_simple_xauusd"   # per-asset strategy_id (registry entry)
base_strategy: "smart_simple_v11"    # inherits execution logic; overrides session + drivers
```

Every consumer below reads `AssetProfile` instead of a hardcoded literal. USD/COP itself
becomes `config/assets/usdcop.yaml` (the default) so the refactor is symmetric, not a special case.

---

## 3. Macro Drivers — the analytical core

The single most important onboarding decision is **which macro variables drive the new asset**.
COP's drivers (EMBI Colombia, IBR, TPM, WTI oil, COL yields, LATAM cross-pairs) are meaningless
for BTC/Gold. Use the existing `config/macro_variables_ssot.yaml` catalog as the driver menu;
add only what is missing.

### 3.1 Driver comparison (what to identify)

| Driver | USD/COP (current) | Gold XAU/USD | Bitcoin BTC/USD | In catalog today? |
|--------|-------------------|--------------|-----------------|-------------------|
| USD strength | DXY (+) | **DXY (−, primary)** | DXY (−) | ✅ `fxrt_index_dxy_usa_d_dxy` |
| US real yields | via UST10Y | **10Y real yield (−, primary)** | 10Y real yield (−) | ⚠ have UST10Y nominal; real = UST10Y − breakeven (add breakeven) |
| Risk / vol | VIX (+) | VIX (+, safe-haven) | VIX (+, risk-off hurts) | ✅ `volt_vix_usa_d_vix` |
| Oil / inflation | **WTI (−, COP is exporter)** | Brent (+, inflation hedge) | weak | ✅ WTI/Brent |
| Country risk | **EMBI_COL (+)** | n/a | n/a | ✅ (COP-only, drop for others) |
| Local rates | **IBR, TPM (COP)** | n/a | n/a | ✅ (COP-only, drop) |
| Equities correlation | COLCAP | Gold vs SPX (low) | **Nasdaq/QQQ (+, high beta)** | ⚠ add SPX/Nasdaq index |
| Crypto-native | n/a | n/a | **funding rate, ETF net flows, stablecoin supply, on-chain MVRV, Fear&Greed** | ❌ none exist |
| Metal-native | n/a | **central-bank demand, ETF (GLD) holdings** | n/a | ❌ none exist |

### 3.2 Rule

- **Reuse** the pair-agnostic catalog variables (DXY, VIX, UST yields, gold, Brent) — they are
  already extracted daily.
- **Add** only asset-native drivers as NEW entries in `config/macro_variables_ssot.yaml`
  (the SSOT — never define a variable elsewhere first). For BTC that means new extractors
  (funding, ETF flows, on-chain) — a real research/engineering cost, not a config edit.
- **Drop** every COP-only driver (EMBI, IBR, TPM, WTI-as-export, COL yields, LATAM cross-pairs).
- Keep T-1 lag (`merge_asof(direction='backward')` + `.shift(1)`) — anti-leakage is asset-agnostic.

Do NOT ship a new asset whose driver set is "COP drivers minus EMBI". Identify the *real* drivers
first (this table is the starting hypothesis, not the final feature set — validate with
walk-forward DA per `experiment-protocol.md`).

---

## 4. File-Touch Map (keyed to `AssetProfile`)

For each currently-hardcoded location, the target is: **read the value from `AssetProfile`**.

| # | Tier | File(s) | Today | After |
|---|------|---------|-------|-------|
| 1 | Easy | `airflow/dags/l0_ohlcv_realtime.py`, `l0_ohlcv_backfill.py` | `SYMBOL_CONFIG`/`ALL_SYMBOLS` dicts | iterate over enabled `AssetProfile`s |
| 2 | Easy | `src/data/ohlcv_loader.py:54-64` | `SYMBOL_PRICE_RANGES`, `SYMBOL_SEED_FILES` | read `price_range`, `seed_file` from profile |
| 3 | Easy | `config/seed_config.yaml` + `seeds/latest/<asset>_m5_ohlcv.parquet` | single-pair | per-asset seed entry |
| 4 | Easy | `usdcop_m5_ohlcv` table | multi-pair PK | reuse as-is (rows tagged by `symbol`) |
| 5 | Moderate | `config/macro_variables_ssot.yaml` | 40 COP-centric vars | add asset-native drivers (§3) |
| 6 | Moderate | `config/forecasting_ssot.yaml` (`data_sources.macro.column_mapping`, `features`) | COP drivers | profile `macro_drivers` |
| 7 | Moderate | `config/pipeline_ssot.yaml` (`macro_column_map`, cross-pair calculators) | MXN/BRL leaders | profile `cross_asset_leaders` (may be empty) |
| 8 | Moderate | `config/execution/smart_simple_v1.yaml` (features list) | 21 COP features | per-asset feature list |
| 9 | Moderate | `src/core/contracts/feature_contract.py` + `config/norm_stats*.json` | one COP contract | per-asset contract hash + norm stats → **retrain** |
| 10 | Moderate | `src/features/calculator_registry.py` | `calculate_cross_pair_lead`, `latam_basket_z`, `cop_vs_peers_z` | skip/replace when no leaders |
| 11 | Hard | `src/core/constants.py:40-43,399-413` | `TRADING_TIMEZONE`, USDCOP tick/pip block | profile `session` + instrument block |
| 12 | Hard | `config/pipeline_ssot.yaml` `trading_schedule`, `bars_per_day:78`, `bars_per_year:19656` | COP session | profile `session.bars_per_*` |
| 13 | Hard | `config/execution/smart_simple_v1.yaml` `week_lifecycle` (Mon 09:00 / Fri 12:50 COT) | COP week | profile session + `forced_close` |
| 14 | Hard | `smart_simple_v1.yaml` `regime_gate.hurst_*` | COP-fit 0.52/0.42 | re-fit per asset (§6) |
| 15 | Hard | Forecast DB tables `forecast_h5_*` (no `symbol` col) | implicit COP | **add `asset_id`/`symbol` column OR parallel `forecast_h5_<asset>_*`** (new migration) |
| 16 | Hard | Airflow DAG schedules (`*/5 13-17`, weekday, Fri close) | COP hours | per-asset schedule (24/7 for BTC) |
| 17 | Hard | Dashboard: `charts/TradingChartWithSignals` `symbol="USDCOP"`, `lib/services/twelvedata.ts`, `lib/config/index.ts` `NEXT_PUBLIC_DEFAULT_SYMBOL`, artifact paths `public/data/production/` | COP literals | asset selector + per-asset artifact folder |

**Reusable as-is (no change)**: `src/forecasting/regime_gate.py` (pure Hurst math),
`dynamic_leverage.py`, `vol_targeting.py`, `adaptive_stops.py`, `confidence_scorer.py`,
`strategy_schema.py`, `signal_contract.py`, `production_contract.py` — logic is math, not COP.

---

## 5. The 24/7 Problem (Bitcoin) — structural, not cosmetic

Gold's near-24h metals session is close enough to FX that re-tuning `session` + drivers suffices.
**Bitcoin is different** and forces genuine design changes. A 24/7 asset invalidates:

| COP assumption | Why it breaks for BTC | Required change |
|----------------|-----------------------|-----------------|
| `bars_per_day: 78` (5h session) | BTC has 288 5-min bars/day | `session.bars_per_day: 288`; re-derive `bars_per_year` |
| Weekend exclusion (`dayofweek < 5`) | BTC trades Sat/Sun | `session.days: [0..6]`, `weekend_flat: false` |
| Friday 12:50 forced close (`week_lifecycle`) | no weekly close exists | `forced_close: none`; use rolling/continuous exit or a fixed UTC weekly boundary |
| Sharpe annualization `bars_per_year: 19656` | wrong scaling → garbage Sharpe | recompute from profile |
| Regime gate Hurst thresholds (COP-fit) | different regime distribution | re-fit on BTC history |
| Macro T-1 daily lag | crypto moves intraday on 24/7 news | consider intraday-native features (funding, order-flow) |
| DAG cadence `*/5 13-17 * * 1-5` | misses 80% of BTC bars | `*/5 * * * *` (24/7) |

**Decision required before onboarding BTC** (raise to the operator): does the H5 weekly
"Monday-entry / Friday-close" execution model even apply to a 24/7 asset? Options:
(a) keep a synthetic weekly boundary (e.g. Mon 00:00 UTC → Fri 23:55 UTC), or
(b) switch BTC to a continuous-horizon executor (`IntradaySLTPExecution` from
`execution_strategies.py`). This is a strategy-design choice, not a config toggle.

---

## 6. Onboarding Stages (SDD procedure)

```
Stage A: PROFILE      Author config/assets/<asset>.yaml (AssetProfile). Freeze it.
Stage B: DATA         Add provider symbol; backfill OHLCV → seeds/latest/<asset>_m5_ohlcv.parquet;
                      verify session/bars_per_day match observed data (TEST A1-A3).
Stage C: DRIVERS      Identify + add asset-native macro drivers (§3); drop COP-only drivers.
                      Confirm all drivers exist in macro_indicators_daily (TEST B1).
Stage D: FEATURES     Derive per-asset feature set + contract hash + train-only norm_stats.
                      Reconcile TS↔Python contract (do NOT fork the drifted baseline). (TEST C1-C2)
Stage E: REGIME FIT   Re-fit Hurst thresholds on the asset's own history; write into profile. (TEST D1)
Stage F: BACKTEST     Walk-forward OOS on last year. Evaluate the 5 gates (Vote 1/2). (TEST E1-E5)
Stage G: DASHBOARD    Register strategy_id; add per-asset artifact folder; asset selector. (TEST F1)
Stage H: APPROVE      Human Vote 2/2 on /dashboard, then --phase production (Stage 6 of MLOps).
Stage I: MONITOR      Per-asset DAG schedule + monitor + guardrails.
```

Follow one-variable-per-experiment discipline (`experiment-protocol.md`) when tuning the
new asset's model. The AssetProfile is FROZEN once Stage F backtest starts (`ssot-versioning.md`).

---

## 7. TDD Acceptance Test-Contract

A new asset MUST NOT reach paper trading until these tests pass. Proposed home:
`tests/onboarding/test_asset_<asset_id>.py`, parameterized by `AssetProfile`. Each asserts a
specific onboarding invariant — write them RED first, make them GREEN by completing the stage.

| ID | Test | Asserts | Blocks |
|----|------|---------|--------|
| A1 | `test_profile_schema_valid` | `AssetProfile` loads, all required fields present, `price_range[0] < price_range[1]` | Stage A |
| A2 | `test_seed_loads_and_in_range` | seed parquet loads, all `close` within `price_range`, no NaN in OHLC, `high>=low` | Stage B |
| A3 | `test_session_matches_data` | observed median bars/day == `session.bars_per_day`; observed trading days match `session.days` (catches 24/7 vs weekday) | Stage B |
| A4 | `test_timestamps_in_profile_tz` | all timestamps in `session.timezone`, none outside declared session window | Stage B |
| B1 | `test_macro_drivers_available` | every `macro_drivers[].db_col` exists in `macro_indicators_daily` with < `max_null_ratio` nulls; no COP-only driver leaked into a non-COP profile | Stage C |
| C1 | `test_feature_contract_hash` | recomputed `FEATURE_ORDER_HASH` == stored; `len(FEATURE_ORDER)==OBSERVATION_DIM`; TS mirror hash matches Python (no drift) | Stage D |
| C2 | `test_norm_stats_train_only` | norm_stats computed on train split only; feature keys == feature contract; no leakage of val/test rows | Stage D |
| C3 | `test_no_lookahead` | macro features are T-1 lagged (`merge_asof` backward + shift 1); target uses future close only | Stage D |
| D1 | `test_regime_thresholds_fit` | `hurst_trending`/`hurst_mean_rev` are non-null AND fit on this asset (not equal to COP 0.52/0.42 unless justified) | Stage E |
| E1 | `test_backtest_gate_return` | OOS return > −15% | Stage F |
| E2 | `test_backtest_gate_sharpe` | Sharpe > 0 (annualized with profile `bars_per_year`) | Stage F |
| E3 | `test_backtest_gate_drawdown` | max DD < 20% | Stage F |
| E4 | `test_backtest_gate_trades` | trades >= 10 | Stage F |
| E5 | `test_backtest_gate_significance` | p < 0.05 (bootstrap CI excludes 0) | Stage F |
| F1 | `test_dashboard_asset_parameterized` | no hardcoded `"USDCOP"`/`"USD/COP"` on the asset's render path; artifact paths resolve for `asset_id`; `strategies.json` lists the new `strategy_id` | Stage G |

Gates E1-E5 mirror the existing 5 approval gates (`sdd-approval-spec.md`) — the onboarding
tests just bind them to the new asset's OOS backtest so Vote 1/2 is automatic and identical
across assets.

---

## 8. Worked Example — Gold `XAU/USD` (moderate)

- **Profile**: `config/assets/xauusd.yaml` (§2). `asset_class: commodity`, `session.mode: metals`,
  `America/New_York`, `bars_per_day: 276`, `weekend_flat: true`, `forced_close: friday`.
- **Data**: Gold already exists as a *daily macro close* but NOT as intraday OHLCV. Add
  TwelveData `provider_symbol: "XAU/USD"` to the L0 DAGs; backfill → `seeds/latest/xauusd_m5_ohlcv.parquet`.
- **Drivers**: DXY (−), UST10Y real-yield proxy (−, primary), VIX (+), Brent (+). All already
  in the catalog → **no new extractor needed** (add breakeven for a true real yield later).
- **What breaks**: only the session (metals ≠ COT 5h) and the regime thresholds. Week lifecycle
  can stay (Mon entry / Fri close works for a 5-day metals week).
- **Effort**: mostly config + retrain + a metals session. Reuses execution, regime logic, gates.

## 9. Worked Example — Bitcoin `BTC/USD` (hard)

- **Profile**: `config/assets/btcusd.yaml`. `asset_class: crypto`, `session.mode: 24x7`,
  `timezone: UTC`, `days: [0..6]`, `bars_per_day: 288`, `weekend_flat: false`,
  `forced_close: none`, `price_range: [10000, 200000]`.
- **Data**: TwelveData/exchange `BTC/USD` 5-min, 24/7 backfill → `seeds/latest/btcusd_m5_ohlcv.parquet`.
  DAG cadence becomes `*/5 * * * *`.
- **Drivers**: **none exist** — must build extractors for funding rate, spot-ETF net flows,
  stablecoin supply, on-chain MVRV, Fear&Greed; reuse DXY/UST-real-yield/VIX; add Nasdaq/QQQ
  correlation. This is a genuine data-engineering project, not a config edit.
- **What breaks**: session length, `bars_per_day`, weekend logic, Friday-close, Sharpe
  annualization, regime thresholds, DAG cadence, macro T-1 daily lag (crypto is intraday-native).
- **Strategy decision**: the H5 "Monday-entry/Friday-close" model likely does NOT fit 24/7.
  Decide (§5): synthetic UTC weekly boundary vs continuous `IntradaySLTPExecution`. Raise to operator.
- **Effort**: high. BTC is the asset that justifies building `AssetProfile` properly rather than
  string-replacing "USDCOP".

---

## 10. Cross-References

| Concern | Spec |
|---------|------|
| As-built architecture map | `sdd-architecture-overview.md` |
| Dynamic registry so the new asset appears in the UI without frontend edits | `sdd-strategy-lifecycle-registry.md` |
| L0 OHLCV + macro governance (timezone, TwelveData quirks, multi-pair) | `l0-data-governance.md` |
| Feature contract + norm stats + retrain | `l2-l3-l4-training-pipeline.md`, `l1-l5-inference-pipeline.md` |
| Strategy schemas + registry + signal contract | `sdd-strategy-spec.md` |
| Approval gates (mirror onboarding tests E1-E5) | `sdd-approval-spec.md` |
| Dashboard artifact schemas (per-asset folders) | `sdd-dashboard-integration.md` |
| Experiment discipline for tuning the new asset | `experiment-protocol.md`, `ssot-versioning.md` |
| H5 execution logic (inherited by per-asset strategies) | `h5-smart-simple-pipeline.md` |

---

## DO NOT

- Do NOT string-replace "USDCOP" to add an asset — introduce `AssetProfile` and read from it.
- Do NOT reuse COP macro drivers for a non-COP asset (EMBI/IBR/TPM/WTI-as-export/COL-yields are
  COP-specific). Identify the asset's real drivers first (§3), validate with walk-forward DA.
- Do NOT copy COP's Hurst regime thresholds (0.52/0.42) — re-fit them per asset (TEST D1).
- Do NOT apply the COP `bars_per_day:78` / Friday-close / weekday session to a 24/7 asset — it
  corrupts annualization, exits, and the regime gate (§5).
- Do NOT write to `forecast_h5_*` tables for a new asset without adding an `asset_id`/`symbol`
  dimension (or parallel tables) via a new migration — they are implicitly COP today.
- Do NOT fork the feature contract while TS↔Python drift exists (`sdd-architecture-overview.md`
  §5.3) — reconcile first, then derive the asset's contract.
- Do NOT promote an asset past Stage F without all TDD tests (A1–F1) GREEN.
- Do NOT skip the operator decision on 24/7 execution semantics for BTC (§5) — it is a strategy
  choice, not a config default.
