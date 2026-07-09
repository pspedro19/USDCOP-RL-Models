# SDD Spec: Strategy Science — how the rule-based strategies are built, sized, and judged (AS-BUILT)

> **Responsibility**: The cross-asset, *as-built* explanation of the strategy science that powers the
> multi-asset backtests visible on the web today — **Gold (XAU/USD)** and **Bitcoin (BTC/USDT)**. It
> documents, for a working operator: the strategies themselves, the exact **variables/features**, the
> **vol-targeting risk layer**, the **regime classifier**, the **anti-leakage discipline**, and the
> **evaluation** (gates + block-bootstrap + the honest gate), plus the **per-asset results** and
> exactly **what you create when you add a new strategy or a new asset**.
>
> Companion to: `_asbuilt-implementation.md` (the platform/backend/frontend as-built),
> `_onboarding-playbook.md` (the AssetProfile contract + stages), and
> `../platform/registry-lifecycle.md` (the dynamic registry that publishes these backtests).
>
> Contract: CTR-STRAT-SCIENCE-001 · Date: 2026-07-05

---

## 0. The thesis (why these strategies look the way they do)

For Gold and BTC the **direction** of the next bar is close to noise. The alpha is not in *predicting*
— it is in **sizing and gating**: hold *how much* exposure, and *when to sit out*. So every strategy is
built from three separable pieces (the classic separation of concerns):

```
INTENT (direction ∈ {0, ±1})  ×  SIZE (vol-targeting × regime mult)  =  POSITION  ── shift(1) ──►  applied at t
   the "strategy"                    the deterministic risk layer          effective exposure     (causal)
```

- The **intent** is the only thing that changes between strategies (HODL / trend / regime-gated).
- The **size** and **evaluation** are shared, deterministic, and identical across strategies + assets.
- This is why adding a strategy is *one function*, and adding an asset is *one config + one seed*.

Two independent science stacks exist today, one per non-COP asset, deliberately kept parallel (not a
shared import) so an asset's quirks never leak into another's:

| Module | Asset | Notes |
|--------|-------|-------|
| `src/gold_rl/{indicators,strategies,backtest}.py` | XAU/USD | long/short capable; √252 annualization; NY-close daily |
| `src/btc_strategy/{indicators,strategies,backtest}.py` | BTC/USDT | **spot-only** exposure ∈ [0,1]; **√365**; UTC-00:00 daily |

> COP's production strategy (`smart_simple_v11`) is a *different* lineage (Ridge/BR/XGBoost + regime
> gate on weekly bars; see `tracks/h5-smart-simple.md`). This spec covers the **rule-based daily
> science stacks** (Gold, BTC) that share the intent×size×evaluate shape above.

---

## 1. The variables (features) — identical shape, per-asset annualization

Both daily science stacks compute the **same causal feature set** on `[time, open, high, low, close]`.
Every value at row *t* uses only data ≤ *t* (see §4 anti-leakage).

| Feature | Definition | Role |
|---------|-----------|------|
| `log_ret` | `ln(close_t / close_{t-1})` | return base for vol + bootstrap |
| `sma_{20,50,100,200}` | simple MA (min_periods=window) | trend reference (100 = mid, 200 = secular) |
| `atr_14` | **Wilder** ATR (EMA α=1/14) | true-range volatility |
| `atr_pct` | `atr_14 / close` | scale-free volatility |
| `adx_14` | **Wilder** ADX | trend *strength* (>25 = trending) |
| `realized_vol_20` | `std(log_ret, 20) × √N` | vol-targeting denominator |
| `z_sma50` | `(close − sma_50) / rolling_std_50` | stretch / mean-reversion |
| `hurst` | rescaled-range Hurst on log-price (window 100) | persistence vs mean-reversion (slow, smoothed — never a binary switch) |
| `hurst_smooth` | 10-bar mean of `hurst` | regime input |

**The one per-asset difference is the annualization constant `N`** (the single most common way to get
a fake Sharpe wrong):

| Asset | `N` (annualization) | Why |
|-------|--------------------|-----|
| Gold (daily) | **252** | weekday commodity |
| BTC (daily) | **365** | crypto trades every weekend |

Sourced from `AssetProfile.session` (`trading_days_per_year`), never a global constant. Reusing Gold's
252 for BTC would understate BTC's annualized vol/Sharpe; reusing COP's intraday `19656` for either
would be nonsense. See `_asbuilt-implementation.md §4.2`.

> **Not yet wired (honest):** BTC's *design* (`assets/btcusdt/design/`) calls for on-chain valuation
> (MVRV-Z/NUPL), funding z-score, ETF flows, and an event gate as additional variables. Those need the
> crypto-native extractors (→ migration 052 tables) that are **still pending**. The price-only feature
> set above is what is *actually computed today*; it is the honest floor, not the final feature set.

---

## 2. The risk layer — vol-targeting (deterministic, shared)

`vol_target_size()` turns a raw intent into a *sized* position. It is deterministic and identical
across strategies: when the asset gets violent, exposure shrinks automatically — that is where the
edge lives.

```
size = clip( target_vol / max(realized_vol_20, vol_floor) × regime_risk_mult ,  min , max )
```

| Param | Gold | BTC | Meaning |
|-------|------|-----|---------|
| `target_vol` | 0.10 | **0.30** | annualized vol budget (BTC runs hotter, PRE-REGISTRATION §2) |
| `vol_floor` | 0.06 | **0.30** | caps leverage in ultra-calm periods (stops a 3× blow-up) |
| `max` (leverage/exposure) | 1.5× | **1.0** (spot-only) | BTC never levers or shorts — no liquidation by design |
| `min` | 0.0 | 0.0 | flat is allowed |
| `regime_risk_mult` | per-regime | per-regime | scales size down in hostile regimes (§3) |

**Spot-only invariant (BTC):** `exposure ∈ [0, 1]` on *every* bar — never negative, never > 1. The
Gold layer permits a 1.5× leverage cap (it can express a stronger long); BTC deliberately cannot.

---

## 3. The regime classifier — 4 states with hysteresis (shared shape, per-asset labels)

A rule-based, **causal** 4-regime classifier assigns each day a state; a per-regime multiplier feeds
the risk layer, and some strategies gate *entirely* on regime. **Hysteresis** (a minimum "dwell" of
consecutive days before switching) keeps labels stable so the strategy doesn't churn on noise.

| Gold regimes (`indicators.py`) | BTC regimes (`indicators.py`) | Trigger (causal) | risk mult |
|--------------------------------|-------------------------------|------------------|-----------|
| `trend` | `markup` | ADX≥25 & Hurst≥0.5 & (BTC: price>SMA200) | 1.0 |
| `compression` | `accumulation` | low-vol / low-ADX basing (default) | 1.0 / 0.8 |
| `stretched` | `distribution` | \|z_sma50\|≥2 & Hurst<0.5 | 0.6 / 0.5 |
| `event` | `markdown` | Gold: macro blackout flag · BTC: price<SMA200 & ADX≥20 | 0.35 |

- **Gold dwell = 4**, **BTC dwell = 5** days. Both report `regime_transitions_per_year` as a stability
  check (Gold's regimes last weeks → low count; BTC's price-only proxy is churnier at ~12.7/yr, an
  honest limitation the on-chain HMM in `design/` is meant to fix).
- Regime attribution (§5) shows *which regime made or lost the money* — the single most informative
  diagnostic.

> **Design vs as-built:** `design/SPEC-01` specifies a **frozen-fit 4-state Gaussian HMM** on the
> on-chain vector. Until that data lands, the rule-based price-only classifier above is the stand-in.
> The `AssetProfile.regime_gate.hurst_*` thresholds are `null` (re-fit per asset — never copy COP's
> 0.52/0.42).

---

## 4. Anti-leakage discipline (why the backtest is honest)

Every mechanism that could leak the future is closed deliberately:

| Guard | Where | What it prevents |
|-------|-------|------------------|
| **Causal features** | `build_daily_features` | row *t* uses only data ≤ *t*; `min_periods=window` so no partial-window peeking |
| **`position = raw.shift(1)`** | `build_positions` | the exposure held on day *t* was decided from info at *t−1* — no same-bar look-ahead |
| **Wilder EMA (α=1/period)** | ATR/ADX | causal smoothing; never a centered/future-window filter |
| **Causal hysteresis** | `classify_regime` | a forward single pass; a label switch needs *past* dwell days, never future confirmation |
| **DST/UTC-correct daily close** | ingestion `_daily_to_*close` | build the *naive* wall-clock (Gold 17:00 ET / BTC 00:00 UTC) then localize — never add a Timedelta to a tz-aware midnight (the 2-DST-days-per-year hazard). Guarded by onboarding test A4 |
| **Deep-history warmup drop** | runner `--warmup` | first ~250 rows (indicator warmup NaNs) dropped before backtest |
| **Fixed-seed block bootstrap** | `block_bootstrap_pvalue` | reproducible p-value (`np.random.default_rng(42)`); block=20 keeps ~monthly autocorrelation so the CI isn't falsely tight |
| **Immutable bundles** | `BundlePublisher` | a published `(strategy_id, version, year)` backtest is content-addressed and **never overwritten** — you cannot silently re-tune and reprint the same headline |

> The macro/ETF-flow leakage guard (`published_at`, D+1 lag) lives in migration 052 for when the
> crypto-native features are ingested — see `assets/btcusdt/specs/SPEC-13 §3.2`.

---

## 5. Evaluation — gates, bootstrap, regime attribution, and the honest gate

Each backtest runs the identical evaluation, producing the dashboard-ready `summary` + `trades`:

**5 approval gates (= Vote 1/2, mirror `approval-gates.md`):**

| Gate | Threshold (Gold / BTC) | Meaning |
|------|------------------------|---------|
| `min_return_pct` | > −15% | didn't blow up |
| `min_sharpe_ratio` | > 0 | positive risk-adjusted |
| `max_drawdown_pct` | < 20% / **< 40%** | BTC tolerates deeper DD honestly |
| `min_trades` | ≥ 10 | enough samples (a pure HODL fails this — it is 1 continuous trade, honest) |
| `statistical_significance` | p < 0.05 | bootstrap edge is real |

`recommendation ∈ {PROMOTE, REVIEW, REJECT}` (REJECT iff return or drawdown gate fails);
`backtest_confidence` = fraction of gates passed.

**Statistics:** moving-block bootstrap → one-sided p-value `P(mean daily return ≤ 0)` + 95% annualized
CI. **Regime attribution:** PnL / Sharpe / day-share per regime.

**Trial-aware honesty (2026-07): Deflated Sharpe + true OOS.** Two additions close the "gates were
in-sample / single-test" gap flagged by the metrics audit:
- **Deflated Sharpe Ratio** (Bailey & López de Prado, `services/common/metrics.py::deflated_sharpe_ratio`)
  — each strategy's Probabilistic Sharpe is deflated for the **N strategies tried** per asset (dispersion
  of per-period Sharpe across the trial set). The runners compute it and inject
  `statistical_tests.deflated_sharpe = {sr0, dsr, significant}`; the trial-aware bar is **DSR > 0.95**,
  stricter than a lone bootstrap p < 0.05. This exposed that Gold's `gold_trend_b2` (DSR 0.921) does not
  fully survive the correction while BTC's `btc_trend_b2` (0.999) does.
- **True OOS-2025 slice** — `run_backtest(..., oos_year=2025)` reports metrics + bootstrap on the held-out
  2025 calendar year alone (`summary.oos`), so the gates aren't read only on the span used to pick the
  fixed thresholds. Shared math lives in `services/common/metrics.py` (PSR/DSR/`trial_aware_moments`) —
  the single home the gold/btc backtest twins import (never re-implemented), mirroring the `safe_json_dump`
  pattern.

**Data-quality gate (2026-07).** Every OHLCV seed now passes `src/data_quality/ohlcv_validators.py`
(weekday coverage vs `AssetProfile.session.days`, bars/period, calendar gaps, tz-of-close, OHLC integrity)
before it is written — the gate that would have caught the Gold daily day-shift bug (see §7 / the DS-cycle
spec `_ds-cycle-asbuilt.md §1`).

**The honest gate (the decisive test).** A candidate strategy is only "a strategy" if it beats **both**
baselines B1 (HODL/long-only vol-target) **and** B2 (trend-follower) **out-of-sample on risk-adjusted
terms** (Sharpe *and* Calmar). Otherwise the honest verdict is "the baseline is the floor." The runners
print this verdict explicitly and do not promote a candidate that fails it.

---

## 6. Results per asset (as published to the registry today)

OOS backtest, $10k initial. Numbers are the registry headline (active version). **Never compare across
assets** as if on one clock — each is annualized with its own `N` (§1) and priced in its own units.

### Gold (XAU/USD) — daily, 2004 → 2026 (`scripts/run_gold_pipeline.py`, bundle **v1.1.0**)
Recomputed on the **calendar-corrected** seed (the daily day-shift bug is fixed — see §7). `DSR` =
Deflated Sharpe, deflating each strategy's PSR for the N=3 trials tested (trial-aware bar = DSR > 0.95):
| Strategy | Return | Sharpe | p-value | **DSR** | Rec | Type |
|----------|--------|--------|---------|---------|-----|------|
| `gold_long_only_b1` — HODL vol-targeted (B1) | +187.5% | 0.504 | 0.0106 | 0.981 ✓ | REJECT (DD) | rule_based |
| `gold_regime_gated_v1` — Regime-gated | +116.7% | 0.455 | 0.0196 | 0.968 ✓ | REJECT | hybrid |
| `gold_trend_b2` — Trend-follower (B2) | +55.3% | 0.362 | 0.0412 | 0.921 ✗ | PROMOTE | rule_based |

> **OOS-2025 slice** (`summary.oos`, now published): `gold_trend_b2` +23.8% / Sharpe 2.40 / p=0.0006
> (strong; 2025 was a trending Gold year). Full-history `gold_trend_b2` DSR 0.921 < 0.95 → the edge is
> partly trial-selection; the OOS slice is the honest support.

### Bitcoin (BTC/USDT) — daily, 2018 → 2026 (`scripts/pipeline/run_btc_pipeline.py`)
| Strategy | Return | Sharpe | Calmar | MaxDD | p-value | Rec | Type |
|----------|--------|--------|--------|-------|---------|-----|------|
| **`btc_trend_b2` — Trend-follower (B2)** | **+351.2%** | **1.40** | **1.83** | **−11.0%** | 0.0 | **PROMOTE** | rule_based |
| `btc_hodl_b1` — HODL vol-targeted (B1) | +271.5% | 0.789 | 0.51 | −34.3% | 0.0268 | REVIEW | rule_based |
| `btc_exposure_s3` — Regime-gated (S3) | +124.4% | 0.619 | 0.33 | −31.9% | 0.0602 | REVIEW | hybrid |

**Honest verdicts on record:**
- **BTC:** raw buy-and-hold returned **+552%** but with ~−77% historical drawdowns; the vol-targeted /
  trend strategies trade raw return for risk control (B2's MaxDD is **−11%**). B2 is the standout and
  the published floor. **S3 does NOT beat B2** on Sharpe or Calmar — it needs the on-chain HMM
  (pending crypto-native data) to justify itself. Logged as-is, not inflated.
- **Gold:** B2 is the only PROMOTE (all 5 gates); the regime-gated candidate did not beat B1. RL
  (`design` SPEC-05/08) is deferred behind this floor.
- **COP** (different lineage): `smart_simple_v11` +25.6% / Sharpe 3.35 / p=0.006 (2025, **production**);
  `smart_simple_aggr` +29.5% / Sharpe 4.04 (experimental A/B branch). See `tracks/h5-smart-simple.md`.

### 6.1 Weekly-inference view (`/forecasting`, per pair) — reuse, not a new model

The multi-asset `/forecasting` page shows a **whole-year weekly inference** for Gold & BTC (USD/COP keeps
its 9-model ML zoo). There is **no forecast model** for the rule-based stacks — the "inference" is the
strategy's own **causal weekly positioning**, derived by reusing the exact backtest layer:

```
build_daily_features → classify_regime → build_positions(direction_fn) → compute_returns
   then group the daily position/regime/strat_ret series by time.dt.isocalendar() → per ISO week
```

Per week it emits `direction` (sign of `position` at week start), `exposure` (avg |position|/cap),
`regime`, a `confidence` conviction proxy (`in_market × regime_risk_mult`), `expected_return_pct`
(**transparent edge proxy = position × trailing-20d drift, NOT an ML prediction**), `realized_return_pct`
(strategy), `buyhold_return_pct`, and `hit`. **Methodology (all pairs): trained ≤ Dec-2024, 2025 = backtest
(OOS, default), 2026 = production** — legitimate OOS because the rules are fixed constants and every feature
is causal (rolling + `shift(1)`). Generator: `scripts/pipeline/generate_asset_weekly_forecast.py` (all 3
strategies per asset, `is_primary` = the PROMOTE one); output
`public/forecasting/<asset>/weekly_inference_<year>.json`, wired as the `l5_weekly_forecast` DAG stage.
This is a **view over the published backtest**, not a competing number — the registry bundle headline
figures above are the full-history record.

#### 6.1.1 Weekly-inference results per strategy (as generated, `--year all`)

**Backtest = 2025 (OOS, default view)** · **Production = 2026 (YTD, ≤ W27)**. `strat%` = strategy realized,
`b&h%` = asset buy&hold, `hit%` = directional-positioning accuracy, `wks` = weeks in market / total,
`exp` = avg exposure. These are the honest per-year slices of the same causal strategy (not re-fit).

**Gold (XAU/USD)** — buy&hold 2025 **+65.3%**, 2026-YTD **−3.7%**:

| Strategy | 2025 strat% | hit% | wks | exp | 2026-YTD strat% | hit% |
|----------|-------------|------|-----|-----|-----------------|------|
| `gold_long_only_b1` (HODL) | **+50.8%** | 63.5% | 52/52 | 70% | −2.1% | 51.9% |
| `gold_regime_gated_v1` | +46.9% | 55.8% | 47/52 | 61% | −0.5% | 48.1% |
| `gold_trend_b2` ★ (primary) | +26.9% | 46.2% | 26/52 | 31% | **+8.1%** | 59.3% |

**Bitcoin (BTC/USDT)** — buy&hold 2025 **−5.2%**, 2026-YTD **−27.1%**:

| Strategy | 2025 strat% | hit% | wks | exp | 2026-YTD strat% | hit% |
|----------|-------------|------|-----|-----|-----------------|------|
| `btc_hodl_b1` (HODL) | **+5.4%** | 48.1% | 52/52 | 55% | −2.6% | 44.4% |
| `btc_trend_b2` ★ (primary) | −0.9% | 51.9% | 12/52 | 11% | −2.6% | 55.6% |
| `btc_exposure_s3` (regime) | −2.3% | 44.2% | 41/52 | 43% | **0.0%** (flat all 2026) | 55.6% |

**What the weekly slices show (honest):** the edge is **risk control, not raw upside**. In the flat/down
2025 BTC year the vol-targeted HODL still made **+5.4%** vs buy&hold **−5.2%**; in the **−27.1%** 2026 BTC
drawdown every strategy capped the loss at **≤ −2.6%**, and the regime gate (`btc_exposure_s3`) **sat out
the entire crash (0.0%, 0/27 weeks in market)** — exactly its design intent. Gold's trend-follower gives up
2025 beta (in-market only 26/52 weeks) but is the only one **positive in the 2026 pullback (+8.1%)**. The
full-history registry numbers in §6 remain the promotion record; these year slices are what `/forecasting`
surfaces per pair.

---

## 7. Dynamism & scalability — what you create for a new strategy vs a new asset

The frontend and backend are **fully data-driven** off `registry.json` — publishing a bundle is the
only step that makes a strategy appear. Nothing in the UI is hardcoded per strategy or asset.

### 7.1 Add a NEW STRATEGY to an existing asset (one function + one publish)
1. Add an intent function to that asset's `strategies.py` `STRATEGIES` dict, e.g.
   `"btc_breakout_v1": ("Bitcoin · Donchian breakout", intent_breakout, "rule_based")`.
2. The runner already loops the dict → `build_positions` → `run_backtest` → `BundlePublisher.publish(...)`.
3. `registry.json` auto-refreshes; the dashboard shows a new entry under that asset's dropdown with its
   own KPIs, gates, equity curve, trade table, and replay. **Zero frontend code, zero new DB table.**
- **A/B / a new *version* of the same strategy** = call `publish(strategy_id=same, version="1.1.0")`:
  it coexists immutably; the version dropdown + "Promover a activa" (`/api/registry/promote`) pick the winner.

### 7.2 Add a NEW ASSET (config + data; new science only if the clock differs)
1. `config/assets/<id>.yaml` — the `AssetProfile` keystone (symbol, session, `bars_per_year`, drivers,
   regime thresholds). Onboarding test A1 must pass.
2. Ingestion → seeds (reuse `usdcop_m5_ohlcv` by `symbol` + `asset_daily_ohlcv`; migration only for a
   genuinely new data class, e.g. BTC's crypto-native tables in 052).
3. A science module *only if* the asset's session/annualization/exposure semantics differ enough — BTC
   got its own `src/btc_strategy/` (spot-only, √365, UTC close) rather than bending Gold's. If it's
   close to an existing asset, reuse that stack with a different `AssetProfile`.
4. Run the pipeline → publish → the asset appears as a new group in the selector (chart uses
   `manifest.chart_symbol`, e.g. `BTCUSDT`, never a hardcoded `USDCOP`).

### 7.3 What is created per artifact type (summary)
| You add… | You create | You do NOT touch |
|----------|-----------|------------------|
| A strategy | 1 intent fn in `strategies.py` (+ it flows through the shared runner/publisher) | frontend, registry code, DB |
| A version | nothing new — a `publish(version=…)` call | prior versions (immutable) |
| An asset | `config/assets/<id>.yaml` + ingestion + maybe a science module + maybe a migration | COP/Gold/BTC bundles (additive) |

See `../platform/registry-lifecycle.md` for the bundle/manifest/registry contracts and the verified
A/B recipe, and `_asbuilt-implementation.md §5-8` for the backend/frontend resolution flow.

---

## 8. Reproduce (end-to-end, from seed to web)

```bash
# Gold
python scripts/ingest_asset_ohlcv.py --asset xauusd --daily-start 2004-01-01
python scripts/run_gold_pipeline.py            # backtest B1/B2/regime-gated + publish bundles

# BTC (canonical Binance public data — no API key needed)
python scripts/data/ingest_btc_ohlcv.py --no-db
python scripts/pipeline/run_btc_pipeline.py    # backtest B1/B2/S3 + publish bundles

# → open /dashboard, pick "Gold" or "Bitcoin" in the strategy selector; each renders its own
#   chart (XAUUSD / BTCUSDT), KPIs, gates, equity, trade table, and per-version replay.
```

> **Now DAG-driven** (2026-07-05): the same ingest→backtest+publish flow also runs weekly, unattended,
> via the per-asset DS-cycle DAGs `asset_xauusd_pipeline_weekly` (Sun 01:45 COT) / `asset_btcusdt_pipeline_weekly`
> (Sun 02:00 COT) — emitted by the factory `airflow/dags/asset_pipeline_factory.py` from SSOT
> `config/assets/pipelines.yaml` (CTR-ASSET-PIPELINE-001). Tasks: `l0_ingest` (graceful — a stale feed
> never blocks the science stage; runs on last good seed) → `l4_backtest_publish` → `l6_verify_registry`
> (asserts registry.json has the asset + its 3 bundles). The manual commands above remain valid.

Onboarding + registry guards: `tests/onboarding/test_asset_{xauusd,btcusdt}.py` (A1–C1),
`tests/contracts/test_strategy_registry.py` (R1–R9).
