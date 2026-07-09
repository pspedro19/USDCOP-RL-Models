# SDD Spec: Data-Science Cycle AS-BUILT — lineage, stats, transforms, training, results & conclusions (per asset × strategy)

> **Purpose**: the end-to-end **data-engineering + data-science record** for every tradeable asset and
> sub-strategy — data **lineage**, **descriptive statistics**, **transformations**, exactly **what entered
> training**, the **2025 backtest (replay) + forecasting/inference results**, the **news** per asset, and the
> **conclusions**. It maps every number to where it surfaces in the app.
>
> Companion specs: `_strategy-science.md` (how the strategies are built/sized/judged),
> `dashboard-integration.md` (data contracts), `data-governance.md` (L0 rules),
> `operations/elite-operations.md` (DAG timeline). Numbers below are **as-generated 2026-07** — regenerate
> with the commands in each section; do not hand-edit the figures.

---

## 0. The DS cycle (one shape, every asset)

```
L0 INGEST ─► L2 FEATURES ─► L3 REGIME ─► L4 BACKTEST/EVAL ─► L5 PUBLISH/INFER ─► L6 VERIFY ─► RESULTS/CONCLUSIONS
 raw OHLCV    causal          rule-based    vol-target ×        registry bundle    registry     backtest-2025 +
 (seed)       indicators      4-state       regime → position   + weekly           freshness    forecast + news
                              hysteresis    → returns (costs)    inference JSON                  → this doc §5-9
```

**Methodology (ALL pairs, SSOT)**: train on history **≤ Dec-2024**; **2025 = backtest (OOS, default view +
metrics)**; **2026 = production (YTD)**. Legit OOS because rules are fixed constants and every feature is
**causal** (rolling windows + `shift(1)`) — a 2025 bar only sees data ≤ itself. Gold/BTC follow the
rule-based lineage below; **USD/COP** production is a different lineage (`smart_simple_v11`, H5 Ridge/BR/XGB
+ regime gate — see `tracks/h5-smart-simple.md`) and its `/forecasting` model-zoo is a diagnostic surface.

---

## 1. Data lineage & descriptive statistics (L0)

| | **Gold — XAU/USD** | **Bitcoin — BTC/USDT** | **USD/COP** |
|---|---|---|---|
| Source | TwelveData deep history | **Binance public klines** (no API key) | TwelveData + macro (FRED/BanRep/Investing) |
| Ingest script | `scripts/ingest_asset_ohlcv.py --asset xauusd` | `scripts/data/ingest_btc_ohlcv.py` | `core_l0_*` DAGs + `build_forecasting_dataset_aligned.py` |
| Seed (canonical) | `seeds/latest/xauusd_daily_ohlcv.parquet` | `seeds/latest/btcusdt_daily_ohlcv.parquet` | `seeds/latest/usdcop_daily_ohlcv.parquet` |
| Bar / clock / tz | daily, NY-close, **UTC** | daily, **UTC 00:00** close, 24/7 | daily, COT session |
| Annualization | **√252** | **√365** (24/7) | √252 |
| **Rows** | **5,862** (post-fix) | **3,245** | 1,686 |
| **Date range** | **2004-01-02 → 2026-07-06** | **2017-08-17 → 2026-07-05** | 2020-01-02 → 2026-07-02 |
| Close range (mean) | 374.90 – 5,330.95 (1,524.34) | 3,189 – 124,659 (38,099) | — |
| Daily log-ret mean / std | +0.038% / **1.10%** | +0.083% / **3.57%** | — |
| **Annualized vol** | **17.4%** (post-fix) | **68.1%** | — |
| Skew / excess-kurtosis | −0.39 / 5.8 | −0.96 / 15.7 (fat left tail) | — |
| Data quality | 0 NaN OHLC · 0 dup ts · **Mon-Fri clean** (weekday bug fixed, see note) | 0 NaN OHLC · 0 dup ts · clean 24/7 (~464/weekday) | anti-leakage macro T-1 |
| Macro inputs | — (price-only stack) | — (price-only; on-chain pending mig-052) | **MACRO_DAILY_CLEAN** (10,797 rows): DXY, WTI, VIX, EMBI, Gold, Brent, Coffee, COLCAP, UST10Y, COL10Y |

> Reproduce: `python -c "import pandas as pd; df=pd.read_parquet('seeds/latest/<asset>_daily_ohlcv.parquet'); print(df.describe())"`.
> BTC's fat left tail (kurt 15.7, skew −0.96) is exactly why the vol-targeting/regime overlay exists.
>
> **✅ Gold seed weekday bug — ROOT-CAUSED & FIXED (2026-07).** Pre-fix the XAU/USD seed's day-of-week
> counts were Mon-Thu ≈ 1,173 each but **Fri 151 · Sat 46 · Sun 1,173** — Monday..Friday bars shifted one
> calendar day back onto **Sunday**. Root cause: `ingest_asset_ohlcv.py::_daily_to_nyclose` called
> `.dt.tz_convert("America/New_York").dt.normalize()` on TwelveData daily bars stamped at **00:00 UTC**;
> 00:00 UTC → 19:00 the *previous* ET day, so `normalize()` snapped every bar back a day (Mon→Sun). Fix:
> anchor the trading date in **UTC** (`tz_convert("UTC").normalize()`) before the 17:00-ET close offset
> (correct for both TwelveData 00:00-UTC and Investing 05:00-UTC sources), plus apply `_filter_session`
> to the daily path. Post-fix: **5,862 bars, Mon-Fri only, ~260 bars/yr, ann.vol 17.4%** (was 17.1% on the
> mis-sequenced seed). A new **OHLCV weekday/gap/tz validator** (`src/data_quality/ohlcv_validators.py`,
> wired into every ingest as a hard gate) now blocks this class of bug — it raises on any off-session
> weekday bar. Gold metrics were slightly optimistic pre-fix (`gold_trend_b2` full-history 61.1% → **55.3%**,
> Sharpe 0.38 → **0.362**); verdicts unchanged. BTC's seed was already clean (native Binance openTime, 24/7).

---

## 2. Transformations — feature engineering (L2, `build_daily_features`)

Identical shape for Gold (`src/gold_rl/indicators.py`) and BTC (`src/btc_strategy/indicators.py`); input
requires only `[time, open, high, low, close]` (volume/symbol ignored). All **causal**.

| Feature | Formula | Role |
|---------|---------|------|
| `log_ret` | `ln(close/close₋₁)` | returns base |
| `sma_20/50/100/200` | rolling means | trend / intent gates (SMA100+ADX for B2, SMA200 for regime-gated) |
| `atr_14`, `atr_pct` | **Wilder** ATR | volatility, stop context |
| `adx_14` | **Wilder** ADX | trend-strength gate (`>25`) |
| `realized_vol_20` | `std(log_ret,20) × √N` (N=252 Gold / 365 BTC) | **vol-targeting denominator** |
| `z_sma50` | z-score of close vs SMA50 | regime input |
| `hurst`, `hurst_smooth` | **R/S** rescaled-range (100-bar), 10-bar mean | regime input (slow feature, never a binary switch) |

Then **L3 regime** (`classify_regime`, hysteresis dwell = **4** Gold / **5** BTC) adds `regime` +
`regime_risk_mult`; **risk layer** `vol_target_size` adds `size = clip(target_vol/max(realized_vol_20,floor)
× regime_risk_mult, min, max)`; **`build_positions`** adds `direction`/`intent` + `position =
(dir×size).shift(1)` (causal); **`compute_returns`** adds `ret`, `cost` (2 bps × turnover), `swap`
(2.5%/yr × |exposure|), `strat_ret = position·ret − cost − swap`, `equity`.

**Per-asset risk knobs**: Gold `target_vol=0.10`, cap **1.5×**, vol-floor 6%. BTC `target_vol=0.30`, cap
**1.0** (spot-only, exposure ∈ [0,1]), vol-floor 30%.

---

## 3. What entered training + regime distribution (L2/L3)

| | Gold | BTC |
|---|---|---|
| Warmup drop (indicator burn-in) | 252 bars | 250 bars |
| Modeled rows (post-warmup) | **5,610** (post-fix) | **2,995** |
| **Train (≤ 2024)** | **~5,216** | **2,444** |
| **Backtest 2025 (OOS)** | **261** (≈ 52 wks) | **365** (≈ 52 wks) |
| **Production 2026 (YTD)** | 133 (≈ 27 wks) | 186 (≈ 27 wks) |
| Regime distribution (in-sample) | compression 4,310 · trend 710 · stretched 590 · *(event: 0 fired)* | accumulation 1,493 · markdown 1,055 · markup 235 · distribution 212 |

> Rule-based ⇒ "training" = feature/regime construction on all history; the strategy has **no fitted
> parameters** (thresholds are fixed constants), so the 2025 slice is genuine OOS. Reproduce with
> `scripts/pipeline/generate_asset_weekly_forecast.py` (it runs this exact chain).

---

## 4. Backtest — replay 2025 (`/dashboard`) & full-history registry (L4/L5)

The `/dashboard` **Backtest** page replays the published bundle; the date filters now **default to
2025-01-01 → 2025-12-31** (the OOS backtest year) for every pair, clamped to the trade span. Two number sets:

**(a) Full-history registry bundle** (the promotion record, `registry.json`, **active version 1.1.0**).
Gold recomputed on the calendar-corrected seed (§1). `DSR` = **Deflated Sharpe** (Bailey & López de
Prado), deflating each strategy's PSR for the **N=3 trials** tested per asset — the trial-aware bar is
DSR > 0.95, stricter than a lone bootstrap p < 0.05:

| Asset · strategy | Return | Sharpe | MaxDD | p-value | **DSR** | Rec |
|---|---|---|---|---|---|---|
| Gold `gold_long_only_b1` | +187.5% | 0.504 | −45.5% | 0.0106 | **0.981 ✓** | REJECT (DD) |
| Gold `gold_regime_gated_v1` | +116.7% | 0.455 | −30.5% | 0.0196 | **0.968 ✓** | REJECT |
| Gold `gold_trend_b2` ★ | +55.3% | 0.362 | −15.6% | 0.0412 | **0.921 ✗** | PROMOTE |
| BTC `btc_trend_b2` ★ | **+351.2%** | **1.40** | −11.0% | 0.0 | **0.999 ✓** | PROMOTE |
| BTC `btc_hodl_b1` | +273.1% | 0.791 | −34.3% | 0.0268 | 0.896 ✗ | REVIEW |
| BTC `btc_exposure_s3` | +124.4% | 0.619 | −31.9% | 0.0602 | 0.781 ✗ | REVIEW |

> **Honest read (trial-aware):** after deflating for the 3 strategies tried per asset, Gold's promoted
> `gold_trend_b2` (DSR 0.921) does **not** clear the 0.95 bar — the simpler HODL/regime variants do, but
> they fail the drawdown gate. BTC's `btc_trend_b2` is the one strategy that clears **both** a passing
> gate set **and** DSR > 0.95. This is why the bootstrap p-value alone was insufficient; DSR is now
> published in every bundle's `statistical_tests.deflated_sharpe`.

**(b) 2025 OOS slice** (`summary.oos`, what the replay defaults to; also `/forecasting` §5). Each bundle
now carries a **true held-out 2025** metrics block alongside full-history (the gates were previously
in-sample). Gold 2025 (trending year) is strong OOS — `gold_trend_b2` **+23.8%, Sharpe 2.40, p=0.0006**;
BTC 2025 (choppy) is weak OOS — `btc_trend_b2` **−1.4%, Sharpe −0.05, p=0.62** (the trend edge did not
reproduce out-of-sample that year). Honest OOS ≠ full-history headline.

---

## 5. Forecasting / weekly-inference results per strategy (L5, `/forecasting`)

Whole-year **weekly positioning** (rule-based, no ML). `strat%`=strategy realized, `b&h%`=buy&hold,
`hit%`=directional accuracy, `wks`=weeks in market/total, `exp`=avg exposure. Source:
`public/forecasting/<asset>/weekly_inference_<year>.json`.

**Gold** — buy&hold **2025 +65.3%**, **2026-YTD −3.7%**:

| Strategy | 2025 strat% | hit% | wks | exp | 2026 strat% | hit% |
|---|---|---|---|---|---|---|
| `gold_long_only_b1` | **+50.8%** | 63.5% | 52/52 | 70% | −2.1% | 51.9% |
| `gold_regime_gated_v1` | +46.9% | 55.8% | 47/52 | 61% | −0.5% | 48.1% |
| `gold_trend_b2` ★ | +26.9% | 46.2% | 26/52 | 31% | **+8.1%** | 59.3% |

**BTC** — buy&hold **2025 −5.2%**, **2026-YTD −27.1%**:

| Strategy | 2025 strat% | hit% | wks | exp | 2026 strat% | hit% |
|---|---|---|---|---|---|---|
| `btc_hodl_b1` | **+5.4%** | 48.1% | 52/52 | 55% | −2.6% | 44.4% |
| `btc_trend_b2` ★ | −0.9% | 51.9% | 12/52 | 11% | −2.6% | 55.6% |
| `btc_exposure_s3` | −2.3% | 44.2% | 41/52 | 43% | **0.0%** (flat all 2026) | 55.6% |

---

## 6. USD/COP forecasting model-zoo (9 models, `/forecasting`)

Lineage: 1,686 daily OHLCV (2020→2026) + MACRO_DAILY_CLEAN → **21 SSOT features** (4 price, 4 returns,
3 vol, RSI-14 Wilder, 2 MA-ratios, 3 calendar, **4 macro T-1**) → 9 models × 7 horizons, **walk-forward**
(trains on the fly each run, no artifacts). Whole-year window `--num-weeks 30`.

**2025 backtest (avg direction-accuracy / Sharpe):** `ard` **54.5% / 0.35** (best) · `xgboost` 53.1% ·
`lightgbm` 53.0% · `hybrid_xgboost` 51.7% · `bayesian_ridge` 51.6% · `ridge` 51.2% · `catboost` 48.6% (worst).
**Marginal edge (DA barely > 50%)** — consistent with the project finding that USD/COP alpha is in the
**regime gate + TP/HS mechanics, not the model predictions** (R² < 0 both years). Production = H5
`smart_simple_v11` (+25.6%, Sharpe 3.35, p=0.006, 2025).

---

## 7. News intelligence per asset (`/analysis`)

Per-asset news feeds the weekly `/analysis` view (`public/data/analysis/<asset>/weekly_*.json`,
`news_context` + `news_intelligence.clusters`). Generated by `generate_asset_analysis.py` /
`src/news_engine`.

| Asset | Weekly views | Article-mentions (Σ) | Distinct sources | Notes |
|---|---|---|---|---|
| Gold (xauusd) | 27 | ~1,040 | 131 | AP, ABC, AZ Family, 36Kr… |
| BTC (btcusdt) | 27 | ~1,080 | 102 | CoinDesk, Yahoo Finance, Reuters, Al Jazeera… |
| USD/COP | (COP macro/LLM analysis lineage — Investing.com + Portafolio, W01-W15 LLM digests; see `tracks/news-analysis/`) |

News is **contextual/explanatory** on `/analysis`, not a trading input for the rule-based Gold/BTC stacks
(the BTC on-chain/derivatives feed that *would* be a signal is pending migration 052).

---

## 8. Conclusions per asset & sub-strategy (start → finish)

**The through-line: the edge is RISK CONTROL, not raw upside.** Every strategy trades beta for drawdown
protection; the honest test is whether it survives the bad year, not whether it wins the good one.

- **Gold `gold_trend_b2` ★ (PROMOTE)** — gives up 2025 beta (in-market only 26/52 wks, +26.9% vs b&h
  +65.3%) but is the **only Gold strategy positive in the 2026 pullback (+8.1% vs b&h −3.7%)**. Passes all
  5 gates full-history (**+55.3% on the calendar-corrected seed**, p=0.041; was 61.1% pre-fix). The trend
  filter (SMA100+ADX) is doing its job: out of chop, in for trends. **Caveat (trial-aware):** its
  Deflated Sharpe is **0.921 < 0.95** — the full-history edge does not fully survive the 3-strategy
  multiple-testing correction, though the strong 2025 OOS slice (+23.8%, Sharpe 2.40, p=0.0006) supports it.
- **Gold `gold_long_only_b1`** — highest 2025 (+50.8%) and best hit-rate (63.5%) because 2025 was a strong
  Gold year and HODL rode it vol-targeted; but −2.1% in 2026 → no regime protection. REVIEW.
- **Gold `gold_regime_gated_v1`** — middle ground (+46.9% 2025), but the regime gate did **not** beat B2
  out-of-sample on risk-adjusted terms → REVIEW, not promoted.
- **BTC `btc_exposure_s3` (regime)** — the headline result: **sat out the entire −27.1% 2026 crash (0.0%,
  0/27 weeks in market)** — exactly its design intent. Full-history it does **not** beat B2 (needs the
  on-chain HMM, pending mig-052) → REVIEW.
- **BTC `btc_trend_b2` ★ (PROMOTE)** — the standout full-history (+351%, Sharpe **1.40**, MaxDD only
  **−11%** vs raw b&h ~−77%). Flat-to-slightly-down in the 2025 chop (−0.9%, in-market 12/52) — correctly
  cautious — and caps 2026 at −2.6% vs b&h −27%.
- **BTC `btc_hodl_b1`** — vol-targeting alone beat buy&hold in 2025 (**+5.4% vs −5.2%**) and capped 2026 at
  −2.6%; but −34% historical MaxDD → REVIEW.
- **USD/COP model-zoo** — 9 ML models are **marginal** (DA 48.6-54.5%, R² < 0). Conclusion on record: the
  models are a monitoring/diagnostic surface; the **production edge is the H5 regime gate + TP/HS mechanics**
  (`smart_simple_v11`), not point predictions.

**Honest gate reminder**: a candidate must beat **both** baselines (B1 HODL *and* B2 trend) OOS on Sharpe
**and** Calmar. Today only the B2 trend-followers clear it; the regime-gated candidates are logged as REVIEW,
not inflated.

---

## 9. Results → UI mapping (where each number lives)

| Result | Produced by | File | Surfaced on |
|--------|-------------|------|-------------|
| Full-history bundle metrics | `run_gold_pipeline.py` / `run_btc_pipeline.py` | `strategies/<sid>/backtests/<v>/summary_<year>.json` + `registry.json` | `/dashboard` replay (default **2025** window), registry dropdown |
| 2025 OOS + 2026 weekly inference | `generate_asset_weekly_forecast.py` (`l5_weekly_forecast`) | `forecasting/<asset>/weekly_inference_<year>.json` | `/forecasting` (Gold/BTC) — §5 |
| USD/COP 9-model forecast | `generate_weekly_forecasts.py` (`forecast_weekly_generation`) | `forecasting/bi_dashboard_unified.csv` + PNGs | `/forecasting` (USD/COP) — §6 |
| Backtest chart candles | `export_chart_ohlcv.py` (`l0b_export_chart_ohlcv`) | `data/market/<SYMBOL>_daily.json` | `/dashboard` replay chart |
| News per asset | `generate_asset_analysis.py` + `src/news_engine` | `data/analysis/<asset>/weekly_*.json` | `/analysis` — §7 |

**Regeneration (whole cycle, one asset):** the `asset_<id>_pipeline_weekly` DAG runs
`l0_ingest → l0b_export_chart_ohlcv → l4_backtest_publish → l5_weekly_forecast → l6_verify_registry`
(Sun 06:45 Gold / 07:00 BTC UTC). Manually:
`python scripts/data/ingest_btc_ohlcv.py && python -m scripts.data.export_chart_ohlcv --asset btcusdt && python scripts/pipeline/run_btc_pipeline.py && python -m scripts.pipeline.generate_asset_weekly_forecast --asset btcusdt --year all`.

---

## 10. Systemic conclusions & forward plan (2026-07)

Beyond the per-asset read in §8, the honest system-level conclusions (full detail:
`../audit/STRATEGIC-ASSESSMENT-2026-07.md`):

1. **The edge is risk control, not prediction.** USD/COP models have **R² < 0** / DA ~48–55%; the production
   alpha is the regime gate + TP/HS. Gold/BTC are rule-based. After trial-aware correction, Gold's promoted
   `gold_trend_b2` **does not clear DSR > 0.95** (0.921) and BTC's `btc_trend_b2` is **negative OOS-2025**.
   Call it a **risk-control strategy**, not forecasting. New model claims must clear **DSR > 0.95 + a
   positive OOS slice**, not a lone p-value.
2. **The foundation is thin.** Small samples (H5 2025 = 34 trades, 2026 = 1), small datasets (USD/COP 1,686
   bars), no real Gold intraday history, and a **DB-vs-parquet duality** where `asset_daily_ohlcv` is written
   but read only by the backup job. The 2026-07 OHLCV validator closes one gap; macro/feature gates + a
   single source-of-truth decision remain.
3. **BTC's ceiling is data, not modelling.** Features are **100% price**; the crypto-native tables
   (migration 052: funding/OI/liquidations + on-chain) are **designed but empty** — no extractor exists.
   The unblock is **Fase 1 derivatives extractor first** (`btcusdt/IMPLEMENTATION_ROADMAP.md`).
4. **Infra > signal.** 25+ services / ~29 DAGs (H1 paused, RL deprioritized, MLflow/MinIO/Jaeger under-used)
   on top of a fragile edge. **Narrow the system around what works**; adopt DSR/OOS as the promotion bar;
   feed BTC non-price data. Priorities table: `../audit/STRATEGIC-ASSESSMENT-2026-07.md §6`.
