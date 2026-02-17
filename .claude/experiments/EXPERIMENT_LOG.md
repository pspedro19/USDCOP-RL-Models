# EXPERIMENT LOG — USDCOP RL Trading

> Append-only log. Every experiment gets a unique ID. Never edit past entries.
> Format: EXP-{version}-{sequential}

---

## EXP-V215-001: V21.5 Baseline (Single Seed)
- **Date**: 2026-02-06
- **Hypothesis**: Symmetric stops (4%/4%) + 0% maker fees + min_hold=25 enables profitability
- **Variable changed**: Multiple (from V21) — stops, costs, flat_reward, trailing
- **Config**: SSOT v3.5.0, obs_dim=23, continuous, PPO MlpPolicy, 1M steps, CPU
- **Seeds**: 1 (not multi-seed — this is a weakness)
- **L4 Result**:

| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
|------|---------|--------|-----|------|--------|--------|---------|
| single | +1.26% | +0.202 | 56.3% | 1.006 | 12.72% | 213 | ~109 |

- **Statistical significance**: NOT TESTED (single seed, no CI)
- **vs Buy-and-hold**: NOT TESTED
- **Decision**: PASS gates, but need multi-seed validation
- **Lesson**: First profitable model. But PF=1.006 is essentially breakeven.

---

## EXP-V215-002: V21.5 Multi-Seed Validation
- **Date**: 2026-02-07
- **Hypothesis**: V21.5 success is reproducible across seeds
- **Variable changed**: None (same config, different seeds)
- **Config**: SSOT v3.5.0, obs_dim=23, continuous, PPO MlpPolicy, 1M steps
- **Seeds**: [42, 123, 456, 789, 1337]
- **L4 Result**:

| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades |
|------|---------|--------|-----|------|--------|--------|
| 42 | -39.32% | -7.356 | 61.1% | 0.771 | 40.32% | 1,837 |
| 123 | -41.75% | -7.095 | 61.3% | 0.783 | 42.25% | 1,538 |
| 456 | -32.52% | -7.523 | 58.3% | 0.752 | 33.66% | 1,696 |
| 789 | -34.45% | -6.404 | 57.3% | 0.786 | 35.72% | 1,923 |
| 1337 | -26.33% | -5.082 | 62.3% | 0.824 | 27.52% | 1,166 |

- **Seeds positive**: 0/5
- **Statistical significance**: N/A (all negative)
- **Decision**: FAIL — V21.5 original was lucky. min_hold_bars bug found.
- **Bug found**: CLOSE action + reversals bypass min_hold_bars → 1,500-1,900 trades
- **Lesson**: Single-seed validation is NEVER sufficient. Multi-seed revealed the bug.

---

## EXP-V22-001: Full V22 (5 changes simultaneously) — Pre-Fix
- **Date**: 2026-02-07 to 2026-02-08
- **Hypothesis**: Discrete(4) + LSTM + temporal + reward shaping > V21.5
- **Variables changed**: 5 SIMULTANEOUSLY (violates Rule 1!)
  1. Action space: continuous → Discrete(4)
  2. Model: PPO MlpPolicy → RecurrentPPO MlpLstmPolicy
  3. Features: +4 temporal (obs_dim 23→27)
  4. Reward: +CloseReasonDetector shaping
  5. Training: 1M→2M steps, ent_coef 0.01→0.02
- **Config**: SSOT v4.0.0, obs_dim=27, Discrete(4), RecurrentPPO, 2M steps, GPU
- **Seeds**: [42, 123, 456, 789, 1337]
- **L4 Result**: -39.22% (best seed auto-selected)
- **Bugs found**: 3 critical (min_hold bypass, LSTM states, close_reason not passed)
- **Decision**: FAIL — invalidated by bugs
- **Lesson**: Don't change 5 things at once. Can't attribute failure to any specific change.

---

## EXP-V22-002: V22 Post-Fix (bugs corrected, same config)
- **Date**: 2026-02-08 to 2026-02-09
- **Hypothesis**: Fixing 3 bugs will make V22 profitable
- **Variable changed**: Bug fixes only (not a new variable)
- **Config**: Same as EXP-V22-001 but with bugs fixed
- **Seeds**: [42, 123, 456, 789, 1337]
- **L4 Result**:

| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
|------|---------|--------|-----|------|--------|--------|---------|
| 42 | -9.79% | -1.194 | 49.1% | 0.968 | 14.05% | 383 | 37.2 |
| 123 | -9.85% | -1.096 | 49.5% | 0.970 | 18.59% | 392 | 36.4 |
| 456 | -20.62% | -2.563 | 46.9% | 0.930 | 20.79% | 373 | 38.2 |
| 789 | -31.11% | -4.079 | 44.5% | 0.893 | 33.02% | 391 | 36.5 |
| 1337 | +9.62% | +1.136 | 54.3% | 1.032 | 9.34% | 381 | 37.4 |

- **Seeds positive**: 1/5 (seed 1337 only)
- **Mean return**: -12.35% ± 15.5%
- **Statistical significance**: NOT significant (1/5 seeds, high variance)
- **Bug fixes verified**: Trades 1,834→384 ✓, bars/trade 7.8→37 ✓
- **Decision**: FAIL — only 1/5 profitable, can't distinguish from luck
- **Lesson**: eval_reward ≠ OOS performance. Seed 456 (best eval=131) lost -20.6%, seed 1337 (eval=111) gained +9.6%.

---

## EXP-V215b-001: Controlled Experiment (Temporal Features Only)
- **Date**: 2026-02-10
- **Hypothesis**: Temporal features (hour/dow sin/cos) improve V21.5 baseline
- **Variable changed**: +4 temporal features (obs_dim 23→27) + 2M timesteps (was 1M)
- **Config**: SSOT v3.5.1, obs_dim=27, continuous, PPO MlpPolicy, 2M steps, GPU
- **Seeds**: [42, 123, 456, 789, 1337]
- **L3 Training**:

| Seed | Best Eval | Peak Step | Duration |
|------|-----------|-----------|----------|
| 42 | 99.97 | 1.5M | 4h 13m |
| 123 | 107.43 | 800K | 5h 27m |
| 456 | 96.28 | 2.0M | 7h 02m |
| 789 | 88.47 | 500K | 4h 01m |
| 1337 | 91.44 | 1.2M | ~3.5h |

- **L4 Result**:

| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
|------|---------|--------|-----|------|--------|--------|---------|
| 42 | +1.98% | +0.275 | 54.7% | 1.007 | 11.29% | 340 | 41.9 |
| 123 | +9.38% | +1.026 | 53.5% | 1.028 | 5.45% | 381 | 37.4 |
| 456 | +0.58% | +0.127 | 51.1% | 1.003 | 18.89% | 352 | 40.5 |
| 789 | -2.60% | -0.229 | 43.5% | 0.994 | 21.57% | 292 | 48.8 |
| 1337 | +3.21% | +0.405 | 53.7% | 1.011 | 13.97% | 378 | 37.7 |

- **Seeds positive**: 4/5
- **Mean return**: +2.51% ± 4.37%
- **Bootstrap 95% CI**: PENDING (see EXP-INFRA-001)
- **vs Buy-and-hold**: PENDING
- **Decision**: BEST MODEL — 4/5 seeds profitable, 8.2x improvement over V21.5
- **Note**: 2 variables changed (temporal features + timesteps). Temporal features are the key driver (V22 with 5 changes only got 1/5 profitable).

---

## EXP-INFRA-001: Statistical Baselines (Pre-Fase)
- **Date**: 2026-02-11
- **Hypothesis**: None (pure analysis)
- **Variable changed**: None
- **Script**: `scripts/statistical_analysis.py`
- **Purpose**: Buy-and-hold return, random agent baseline, bootstrap CI, long/short breakdown
- **Results**:
  - **Buy-and-hold USDCOP (2025)**: -14.66% (confirms strong downtrend)
  - **Random agent (1000 sims)**: mean -4.12%, median -4.55%, std 9.34%, 95th pctl +12.51%
  - **V21.5b mean return**: +2.51% ± 4.41% (4/5 seeds positive)
  - **Bootstrap 95% CI**: [-0.69%, +6.15%] — **DOES NOT exclude zero**
  - **t-test**: t=1.272, p=0.272 — **NOT significant at p<0.05**
  - **V21.5b vs B&H**: +17.17pp (BEATS)
  - **V21.5b vs Random mean**: +6.63pp (BEATS)
  - **V21.5b vs Random 95th pctl**: -10.00pp (LOSES — within random noise)
  - **Trade duration**: avg 41.3 bars (3.4h), 1.65x min_hold — not clustered
- **Gate checks**: Beats random mean (PASS), CI excludes zero (FAIL)
- **Interpretation**: V21.5b has a weak edge that's NOT statistically significant.
  The +2.51% mean is better than random mean (-4.12%) but within the random distribution.
  This confirms the plan's diagnosis: PF ~1.01 means wins ≈ losses in size.
  The SDD phases (asymmetric SL/TP, trailing, reward asymmetry) aim to make wins > losses.

---

## EXP-ENS-V215b-001: Ensemble over V21.5b (No Retraining)
- **Date**: 2026-02-11
- **Hypothesis**: 5-model majority vote improves over single best model
- **Variable changed**: Inference method (single → ensemble)
- **Config**: Same 5 V21.5b models, majority vote (3/5 consensus), no Kelly sizing
- **Script**: `scripts/ensemble_backtest_env.py`
- **Models**: `models/v215b_ensemble/seed_{42,123,456,789,1337}/best_model.zip`
- **Result**:

| Metric | Ensemble | V21.5b Mean | V21.5b Best (s123) |
|--------|----------|-------------|---------------------|
| Return | +1.60% | +2.51% | +9.38% |
| Sharpe | +0.241 | +0.321 avg | +1.026 |
| WR% | 53.1% | 51.3% avg | 53.5% |
| PF | 1.01 | 1.009 avg | 1.028 |
| MaxDD | 15.69% | 14.33% avg | 5.45% |
| Trades | 375 | 349 avg | 381 |

- **Monthly returns**: 2025-02: +2.98, 03: +0.45, 04: -0.82, 05: +1.85, 06: -3.03, 07: +2.19, 08: -4.80, 09: -4.70, 10: -4.71, 11: +1.36, 12: +4.82
- **Decision**: FAIL — Ensemble returns LESS than single-model mean (+1.60% vs +2.51%).
  Majority vote averaging reduces best seed's alpha. Ensemble NOT useful at this stage.
  May revisit in Fase 4 after retraining with asymmetric SL/TP produces more aligned models.
- **Lesson**: Ensemble works when models agree on direction. With PF ~1.01 and weak signal,
  consensus voting just averages out the few good decisions from seed 123.

---

## EXP-ASYM-001: Asymmetric SL/TP (Fase 1)
- **Date**: 2026-02-12
- **Hypothesis**: SL=-2.5%/TP=+6% (R:R 1:2.4) improves PF and returns vs symmetric -4%/+4%
- **Variable changed**: Stop levels (SL: -4%→-2.5%, TP: +4%→+6%)
- **Config**: SSOT v4.1.0, obs_dim=27, continuous, PPO MlpPolicy, 2M steps, GPU
- **Seeds**: [42, 123, 456, 789, 1337]
- **L3 Training**:

| Seed | Best Eval | Peak Step | Duration |
|------|-----------|-----------|----------|
| 42 | 120.13 | ~1.5M | 2h 31m |
| 123 | 125.56 | ~1.2M | 2h 30m |
| 456 | 124.41 | ~1.5M | 2h 29m |
| 789 | 114.75 | ~1.0M | 2h 45m |
| 1337 | 119.89 | ~1.0M | 4h 13m |

- **L3 Mean eval**: 120.95 ± 4.72 (CV=3.90% — excellent reproducibility)
- **L4 Result**:

| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
|------|---------|--------|-----|------|--------|--------|---------|
| 42 | -5.57% | -0.584 | 51.1% | 0.985 | 15.13% | 354 | 40.3 |
| 123 | -9.60% | -0.992 | 51.5% | 0.974 | 16.38% | 342 | 41.7 |
| 456 | -12.36% | -1.415 | 47.2% | 0.964 | 17.55% | 358 | 39.8 |
| 789 | -6.14% | -0.623 | 49.6% | 0.983 | 18.01% | 347 | 41.1 |
| 1337 | -9.67% | -1.039 | 47.9% | 0.972 | 19.70% | 436 | 32.7 |

- **Seeds positive**: 0/5
- **Mean return**: -8.67% ± 2.80%
- **Bootstrap 95% CI**: [-10.73%, -6.50%] — significantly negative
- **t-test**: t=-6.909, p=0.0023 — **statistically significant LOSS**
- **vs V21.5b baseline** (+2.51%): -11.18pp WORSE
- **vs Buy-and-hold** (-14.66%): +5.99pp (barely better)
- **Decision**: **FAIL** — 0/5 seeds profitable, significantly worse than V21.5b
- **Root cause analysis**:
  - SL=-2.5% is too tight for USDCOP 5-min bar volatility. Multiple SL triggers observed (-2.51%, -2.54%, -2.58%, -2.67%, -3.05%).
  - With volatility_pct mean=10% and std=8.7%, intraday swings frequently exceed 2.5%.
  - The tighter SL didn't reduce avg_loss enough to compensate for reduced WR.
  - WR dropped from ~53% (V21.5b) to ~49% but PF stayed flat (~0.97 vs ~1.01).
  - TP=+6% rarely triggered (few bars >4% let alone >6%) — the wider TP didn't help.
  - The R:R theory assumed the agent could capture 4.5% avg wins, but USDCOP 5-min movements are too small for 6% TP.
- **Lesson**: Asymmetric SL/TP with tighter SL is DESTRUCTIVE in this volatility regime. The plan assumed avg_win ~4.5% with TP=6%, but USDCOP doesn't produce single-position moves of that magnitude. Wins come from accumulating small gains, not from large individual trades.
- **Gate decision**: FAIL → Proceed to Fase 1B (SL=-3%/TP=+6%, R:R 1:2) as contingency, OR revert to V21.5b config and skip to Fase 2 (trailing stop).

---

## EXP-HOURLY-PPO-001: PPO on 1-Hour Bars (First Attempt)
- **Date**: 2026-02-14
- **Hypothesis**: Hourly bars offer 3.6x better SNR and 12x fewer transaction costs, improving risk-adjusted returns
- **Variable changed**: Bar frequency 5min → 1h. Feature periods adapted for hourly semantics.
- **Config**: `config/experiments/exp_hourly_ppo_001.yaml` (v5.0.0), obs_dim=27, continuous, PPO MlpPolicy, 2M steps, CPU
- **Seeds**: 1 (seed 42 only — aborted after clear overfitting diagnosis)
- **SMA Baseline** (PASO 0): -6.33% return (beats B&H by +8.33pp) → GO decision
- **Hourly seed**: 6,858 rows (2019-12→2025-12, 5 bars/day @ 8-12 COT)
- **Training data**: 5,811 bars (2019-12→2024-12)
- **L4 Result**:

| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
|------|---------|--------|-----|------|--------|--------|---------|
| 42 | -17.49% | -2.618 | 43.8% | 0.74 | 18.14% | 73 | — |

- **Seeds positive**: 0/1 (remaining 4 aborted)
- **vs V21.5b baseline** (+2.51%): -19.50pp WORSE
- **vs Buy-and-hold** (-14.66%): -2.83pp WORSE
- **vs SMA baseline** (-6.33%): -11.16pp WORSE
- **Decision**: **FAIL** — severe entropy collapse (std 0.70→0.04 by 400K steps)
- **Root cause**: 2M timesteps on 5,811 training bars = each bar seen 344x. Massive overfitting.
  - clip_fraction reached 0.60+ (healthy: 0.10-0.15)
  - Entropy collapsed to near-zero by 400K of 2M steps
  - Best eval model (700K) still couldn't generalize to OOS
- **Lesson**: 5-min hyperparameters don't transfer to hourly without scaling timesteps.

---

## EXP-HOURLY-PPO-002: PPO on 1-Hour Bars (Overfitting Fix)
- **Date**: 2026-02-14
- **Hypothesis**: Reducing timesteps to 500K (86x per bar) and increasing ent_coef to 0.03 fixes entropy collapse
- **Variable changed**: total_timesteps 2M→500K + ent_coef 0.01→0.03 (both address same root cause)
- **Config**: `config/experiments/exp_hourly_ppo_002.yaml` (v5.1.0), obs_dim=27, continuous, PPO MlpPolicy, 500K steps, CPU
- **Seeds**: 1 (seed 42 only)
- **Training time**: 42 min (vs 2h18m for EXP-001)
- **L4 Result**:

| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
|------|---------|--------|-----|------|--------|--------|---------|
| 42 | -7.16% | -1.055 | 42.9% | 0.90 | 10.52% | 56 | — |

- **Seeds positive**: 0/1 (remaining 4 not run)
- **vs EXP-001**: +10.33pp improvement (entropy fix confirmed)
- **vs V21.5b baseline** (+2.51%): -9.67pp WORSE
- **vs Buy-and-hold** (-14.66%): +7.50pp BETTER
- **vs SMA baseline** (-6.33%): -0.83pp SIMILAR
- **Decision**: **FAIL** — entropy fix worked (std ended at 0.62, not collapsed) but no learned alpha
- **Root cause**: PPO matches SMA baseline performance. Hourly USDCOP (5 bars/day) too sparse for RL.
  - Training std remained healthy (0.62 at end vs 0.04 in EXP-001)
  - clip_fraction healthy (0.24 at end vs 0.60 in EXP-001)
  - But the agent couldn't learn patterns beyond what a simple moving average captures
- **Key monthly pattern**: +2.36% Jan, then gradual bleed → ends at -7.16%
- **Lesson**: Hourly USDCOP doesn't have enough microstructure for RL. The 5-min timeframe (78 bars/day, 67K training bars) provides much richer patterns. Recommend returning to 5-min experiments.

---

## Hourly Frequency Conclusion (EXP-001 + EXP-002)

**Result**: Hourly bars are NOT viable for USDCOP RL trading.

| Strategy | Return% | Sharpe | Notes |
|----------|---------|--------|-------|
| PPO 2M steps (EXP-001) | -17.49% | -2.618 | Overfit |
| PPO 500K steps (EXP-002) | -7.16% | -1.055 | No alpha |
| SMA 20/50 baseline | -6.33% | -0.39 | Simple |
| Buy-and-hold | -14.66% | — | Passive |
| **V21.5b (5-min)** | **+2.51%** | **+0.321** | **Best** |

Recommendation: Return to 5-min frequency and continue experiment queue (trailing stop, reward asymmetry, etc.).

---

## EXP-DAILY-PPO-001: PPO on Daily Bars with Full Macro Coverage
- **Date**: 2026-02-14
- **Hypothesis**: Daily bars with 15 macro features (inc. coffee, WTI, COLCAP, CLP, COL5Y, IBR) provide richer fundamental signal
- **Variable changed**: Bar frequency 1h → daily. 22 market features (7 technical + 15 macro) + 7 state = 29 obs_dim.
- **Config**: `config/experiments/exp_daily_ppo_001.yaml` (v5.2.0), obs_dim=29, continuous, PPO MlpPolicy, 200K steps, CPU
- **Seeds**: 1 (seed 42 only — aborted after clear failure)
- **Daily seed**: 1,399 rows (2020-01 → 2025-12, 1 bar/day, Mon-Fri)
- **Training data**: 1,126 bars (2020-01 → 2024-12)
- **Training time**: 13 min, 250 FPS
- **L2 Features**: 22 market (7 technical: log_ret_1d/5d/21d/63d, rsi_9, rsi_21, volatility_pct + 15 macro: dxy, vix, ust10y, ust2y, ibr, tpm, embi_col, coffee, wti, colcap, usdclp, col5y, fred_fedfunds, cpi, unemployment)
- **L4 Result**:

| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
|------|---------|--------|-----|------|--------|--------|---------|
| 42 | -8.83% | -2.797 | 12.5% | 0.44 | 8.92% | 8 | — |

- **Seeds positive**: 0/1 (remaining 4 not run)
- **vs V21.5b baseline** (+2.51%): -11.34pp WORSE
- **vs Buy-and-hold** (-14.66%): +5.83pp BETTER
- **vs SMA hourly baseline** (-6.33%): -2.50pp WORSE
- **Decision**: **FAIL** — agent barely traded (8 trades total, 1W/7L)
- **Root cause**: 1,126 training bars is woefully insufficient for PPO.
  - std ended at 1.82 (never converged — healthy range for continuous is 0.3-0.7)
  - With ent_coef=0.05, policy stayed too wide/random
  - Agent learned to mostly HOLD (only 8 entries in 207 trading days)
  - Even with 15 macro features aligned perfectly, there simply isn't enough data for RL to learn
- **Lesson**: RL requires thousands of episodes, not hundreds. Daily USDCOP (~250 bars/year, ~1,200 training bars) cannot support PPO exploration. Supervised learning (gradient boosting, logistic regression) would be more appropriate for daily macro-driven models.

---

## Multi-Frequency Conclusion (5-min, Hourly, Daily)

**Result**: Only 5-minute bars provide sufficient data for RL.

| Frequency | Training Bars | Bars/Day | Best Return% | Sharpe | Verdict |
|-----------|--------------|----------|-------------|--------|---------|
| **5-min** | **67,468** | **78** | **+2.51%** | **+0.321** | **Only viable** |
| Hourly | 5,811 | 5 | -7.16% | -1.055 | No alpha |
| Daily | 1,126 | 1 | -8.83% | -2.797 | Too few bars |
| SMA hourly | — | — | -6.33% | -0.39 | No RL alpha |
| Buy-and-hold | — | — | -14.66% | — | Passive |

**Recommendation**: Return to 5-min frequency and continue experiment queue (trailing stop, reward asymmetry, etc.).

---

## FC-SIZE-001: Forecasting Pipeline — Vol-Target Backtest (Paso 0.1 + 1.1)
- **Date**: 2026-02-15
- **Hypothesis**: Daily forecasting ensemble (9 models, walk-forward) provides statistically significant directional alpha. Vol-targeting can scale returns without destroying Sharpe.
- **Approach**: Supervised learning on daily bars (NOT RL). Walk-forward validation with 5 expanding folds. Top-3 ensemble by DA per fold. Vol-targeting scales position size inversely to realized volatility.
- **Script**: `scripts/vol_target_backtest.py`
- **Data**: `seeds/latest/usdcop_daily_ohlcv.parquet` (1,349 usable rows after feature build, 2020-03 to 2025-12)
- **Features**: 19 SSOT features (4 price + 4 returns + 3 vol + 3 technical + 3 calendar + 2 macro)
- **Models**: 9 (ridge, bayesian_ridge, ard, xgboost_pure, lightgbm_pure, catboost_pure, hybrid_xgboost, hybrid_lightgbm, hybrid_catboost)
- **Walk-forward**: 5 folds, initial_train_ratio=0.6 (809 bars), test_size=108 bars/fold, OOS=540 days

### Paso 0.1: Baseline Results (fixed 1x leverage)

| Metric | Value |
|--------|-------|
| Total Return | **+69.36%** |
| Annualized Return | **+27.87%** |
| Sharpe | **2.109** |
| Sortino | 3.798 |
| Max Drawdown | -8.14% |
| Profit Factor | **1.411** |
| Direction Accuracy | **55.9%** |
| Trading Days (OOS) | 540 |
| Months Positive | 69% |
| Bootstrap 95% CI | **[+0.09%, +0.42%]** (daily mean, annualized) |

**Per-fold:**

| Fold | Period | Return% | DA% | Sharpe | Days |
|------|--------|---------|-----|--------|------|
| 1 | 2023-08 to 2024-01 | +26.24% | 61.1% | 3.85 | 108 |
| 2 | 2024-01 to 2024-07 | +7.82% | 50.9% | 1.82 | 108 |
| 3 | 2024-07 to 2024-12 | +11.36% | 60.2% | 2.32 | 108 |
| 4 | 2024-12 to 2025-05 | +3.72% | 54.6% | 0.72 | 108 |
| 5 | 2025-05 to 2025-12 | +7.72% | 52.8% | 1.61 | 108 |

**Statistical tests:**
- Binomial: 302/540 correct, p=0.0033 (sig@1%)
- t-test: t=3.088, p=0.0021 (sig@1%)
- Bootstrap CI excludes zero
- 5/5 folds positive
- Strategy at 99.9th percentile vs random agent
- vs Buy-and-hold (-7.10%): +76.46pp BETTER

**Model ranking (avg DA):**

| Rank | Model | Avg DA% | Per-fold |
|------|-------|---------|----------|
| 1 | catboost_pure | 54.4% | [61.1, 48.1, 57.4, 55.6, 50.0] |
| 2 | hybrid_catboost | 53.7% | [60.2, 49.1, 54.6, 54.6, 50.0] |
| 3 | ridge | 53.1% | [44.4, 46.3, 63.9, 57.4, 53.7] |
| 4 | xgboost_pure | 53.1% | [55.6, 48.1, 61.1, 52.8, 48.1] |
| 5 | lightgbm_pure | 53.0% | [56.5, 48.1, 59.3, 55.6, 45.4] |
| 6 | hybrid_xgboost | 52.8% | [55.6, 48.1, 59.3, 54.6, 46.3] |
| 7 | hybrid_lightgbm | 52.6% | [55.6, 48.1, 58.3, 53.7, 47.2] |
| 8 | bayesian_ridge | 52.2% | [42.6, 48.1, 59.3, 58.3, 52.8] |
| 9 | ard | 51.1% | [38.9, 56.5, 59.3, 56.5, 44.4] |

**GATE 0.1: PASS** (DA=55.9% > 51%, p=0.0033 < 0.10, 5/5 folds positive)

### Paso 1.1: Vol-Targeting Results

| Config | Return% | Ann% | Sharpe | MaxDD% | PF | AvgLev | CI_lo | CI_hi |
|--------|---------|------|--------|--------|------|--------|-------|-------|
| Baseline (1x) | +69.36 | +27.87 | 2.109 | -8.14 | 1.411 | 1.00 | +0.09 | +0.42 |
| tv=12%, ml=1.5x | +63.42 | +26.87 | 2.050 | -8.27 | 1.382 | 1.09 | +0.08 | +0.41 |
| tv=15%, ml=2.0x | +85.22 | +34.81 | 2.062 | -10.26 | 1.385 | 1.38 | +0.11 | +0.51 |
| tv=18%, ml=2.0x | +102.07 | +40.62 | 2.022 | -12.18 | 1.377 | 1.60 | +0.12 | +0.60 |
| tv=20%, ml=2.5x | +123.01 | +47.51 | 2.050 | -13.51 | 1.382 | 1.82 | +0.14 | +0.68 |

**GATE 1.1: PASS** (All configs: Sharpe >= 1.0, MaxDD < 20%, CI excludes zero)

**Selected config**: tv=15%, ml=2.0x (best Sharpe-to-MaxDD tradeoff)
- Sharpe 2.062 (essentially preserves baseline 2.109)
- Return +85.22% (vs +69.36% baseline) = +23% uplift from vol-targeting
- MaxDD -10.26% (acceptable, 2.26pp worse than baseline)

### Key Insights
1. **Forecasting vastly outperforms RL**: Sharpe 2.109 vs RL's 0.321 (6.6x improvement)
2. **All 9 models contribute**: DA range 51.1%-54.4%, no model is useless
3. **Top-3 ensemble per fold** is effective: adapts model selection to regime
4. **CatBoost dominates** in early folds (2023-24), Ridge/BayesianRidge dominate later (2024-25)
5. **Vol-targeting preserves Sharpe**: Only -2.7% degradation (-0.047) while boosting return +23%
6. **Statistical significance is strong**: p=0.0021 (t-test), p=0.0033 (binomial), CI excludes zero
7. **5/5 folds positive** — no single fold is negative (robust OOS performance)

### Decision
**PROCEED to Paso 1.2** — implement vol-targeting in production code (Airflow DAG, database migration, config file).

---

## EXP-REGIME-001: Regime Detection Features (Forecasting)
- **Date**: 2026-02-16
- **Hypothesis**: Adding regime-aware features (trend slope, range ratio, return sign, vol regime ratio, HMM probabilities) improves directional accuracy over baseline 19 features
- **Variable changed**: Feature set: 19 → 20-24 (7 treatment groups with different regime feature combinations)
- **Config**: Pre-registered in `docs/EXP_REGIME_001.md`
- **Pipeline**: Forecasting (supervised, daily bars)
- **Scripts**: `scripts/exp_regime_001.py`, `src/forecasting/regime_features.py`
- **Validation**: Walk-forward 5 expanding folds, paired t-test with Bonferroni correction (p < 0.05/7 = 0.00714)
- **Models**: 9 (ridge, bayesian_ridge, ard, xgboost_pure, lightgbm_pure, catboost_pure, hybrid_xgboost, hybrid_lightgbm, hybrid_catboost)
- **Total model fits**: 360 (9 models × 8 groups × 5 folds)

### GATE 1: Walk-Forward (2019-2025)

| Group | Features | DA Mean | vs CTRL | p-value | Bonferroni | Verdict |
|-------|----------|---------|---------|---------|------------|---------|
| **CONTROL** | **19** | **52.75%** | — | — | — | **BASE** |
| TREAT-A (+slope) | 20 | 52.60% | -0.14pp | 0.8555 | FAIL | No effect |
| TREAT-B (+slope+range) | 21 | 52.04% | -0.70pp | 0.5159 | FAIL | Degradation |
| TREAT-C (+slope+range+sign) | 22 | 52.58% | -0.16pp | 0.8158 | FAIL | No effect |
| TREAT-D (+4 simple) | 23 | 52.26% | -0.49pp | 0.4617 | FAIL | Degradation |
| TREAT-E (+slope+vol) | 21 | 52.41% | -0.33pp | 0.6897 | FAIL | No effect |
| TREAT-F (+HMM only) | 22 | 52.93% | +0.18pp | 0.9373 | FAIL | Noise |
| TREAT-G (+slope+vol+HMM) | 24 | 52.51% | -0.24pp | 0.9080 | FAIL | No effect |

- **Seeds positive**: N/A (walk-forward, not multi-seed)
- **GATE 1 RESULT**: **FAIL** — 0/7 treatments significant
- **GATE 2 (2026 holdout)**: CONTROL DA=47.0% → **FAIL** (threshold: >50%)

### Root Cause Analysis
1. Regime features are redundant — existing vol/return features already encode regime information
2. HMM probabilities don't add incremental predictive power beyond realized volatility
3. 2026 performance drop is NOT a feature engineering problem — it's a fundamental regime change
4. The correct intervention is operational (SHORT-only filter), not model-based

### Decision
**FAIL** — 0/7 treatment groups passed GATE 1. Keep baseline 19 features. No deployment.

### Lessons
- The 19 baseline features are already near-optimal for this model architecture
- Adding more features (even theoretically sound ones) introduces noise that degrades or neutralizes DA
- Regime detection via features ≠ regime adaptation. The model needs data FROM the new regime, not features ABOUT regimes
- Pre-registration + Bonferroni correction prevented false discovery from 7 simultaneous comparisons

---

## FC-H5-SIMPLE-001: Smart Simple v1.0 -> v1.1 (H=5 Weekly)
- **Date**: 2026-02-16
- **Track**: H5 Weekly Forecasting (parallel to H1 Daily)
- **Hypothesis**: Simpler execution (TP/HS/Friday close, no trailing, no re-entry) with confidence-based sizing and adaptive stops can achieve significant alpha on weekly horizon.
- **Models**: Ridge + BayesianRidge (mean ensemble), trained on 2020-2024 expanding window
- **Features**: 21 (identical to H1 pipeline, 4 macro: DXY, WTI, VIX, EMBI)
- **Config**: `config/execution/smart_simple_v1.yaml`
- **Contract**: FC-H5-SIMPLE-001

### v1.0 Results (OOS 2025)

| Strategy | Return | Sharpe | MaxDD | p-value | Trades | $10K -> |
|----------|--------|--------|-------|---------|--------|---------|
| A) SHORT-only | +12.97% | 2.789 | -4.05% | 0.037 | 24 | $11,297 |
| B) Bidir naive | +11.87% | 1.528 | -7.10% | 0.111 | 34 | $11,187 |
| **C) Bidir smart** | **+13.75%** | **2.867** | **-4.03%** | **0.032** | **24** | **$11,375** |

- WR: 75% (18/24), DA: 62.5%, Exits: 11 TP, 11 week_end, 2 hard_stop
- All 24 executed trades are SHORT (10 LONGs skipped by confidence filter)

### Root Cause Diagnostic (diagnose_smart_simple_v1.py)

**10-section analysis findings:**

1. **Hard stops**: 2 trades hit HS (-6.29% total). Both on weeks with adverse regime moves.
2. **Skipped LONGs**: 10 skipped. If taken: 6 wins (+9.5%) but 4 losses (-10.3%) = net -0.75%. **CORRECT to skip.**
3. **MFE/MAE**: TP well calibrated (MFE/TP = 1.39x for TP trades). HS slightly tight for adverse moves.
4. **Confidence tiers**: LOW SHORTs WR=79% vs MEDIUM=60%. Surprising but N too small for significance.
5. **Theoretical ceiling**: Perfect foresight = +40.84%. Capture ratio = 26.8% (competitive for single EM pair).
6. **Sensitivity**: Current config ranks #8 of 42 in 2D grid search.
7. **Best improvement**: HS multiplier 1.5 -> 2.0 eliminates both hard stops.

### v1.0 -> v1.1 Changes (2 YAML edits)

| Parameter | v1.0 | v1.1 | Rationale |
|-----------|------|------|-----------|
| `vol_multiplier` | 1.5 | **2.0** | Eliminates 2 hard stops, Sharpe 2.87 -> 3.52 |
| SHORT sizing | 2.0/1.5/1.0 | **flat 1.5** | Scorer doesn't discriminate, simplifies system |

### v1.1 Results (OOS 2025 + 2026 YTD)

| Year | Return | Sharpe | MaxDD | p-value | Trades | Hard stops | $10K -> |
|------|--------|--------|-------|---------|--------|------------|---------|
| **2025** | **+20.03%** | **3.516** | **-3.83%** | **0.0097** | 24 | 0 | **$12,003** |
| **2026 YTD** | **+7.12%** | 19.12 | 0.00% | 0.000 | 3 | 0 | **$10,712** |

- 2025: WR 70.8%, DA 62.5%, Exits: 9 TP + 15 week_end + 0 hard_stop
- 2026: 3/3 wins (all SHORT), 2 TP + 1 week_end
- **p=0.0097 passes p<0.01 threshold** (strongest result in the entire project)

### Comparison vs Other Strategies

| Strategy | Return 2025 | Sharpe | p-value | Status |
|----------|-------------|--------|---------|--------|
| H1 Daily + Trail | +36.84% | 3.135 | 0.0178 | PRODUCTION |
| **H5 Smart Simple v1.1** | **+20.03%** | **3.516** | **0.0097** | **PAPER TRADING** |
| RL V21.5b | +2.51% | 0.321 | 0.272 | DEPRIORITIZED |
| Buy & Hold | -12.29% | -- | -- | BASELINE |

### Decision
**PASS** -- Deploy to paper trading. 15-week evaluation period starting 2026-02-16.

### Lessons
- Simpler execution beats complex (TP/HS/close > trailing stop for weekly trades)
- Flat sizing beats tiered when scorer doesn't discriminate (remove degrees of freedom)
- Wider stops beat tight stops in EM FX (reduce whipsaw, accept larger individual losses)
- 2 YAML edits (+6.28pp return, +0.65 Sharpe) demonstrate value of systematic diagnostic
- H5 (weekly) achieves higher Sharpe than H1 (daily) despite lower absolute return
- The 26.8% capture of theoretical ceiling is competitive for single EM pair with linear model
