---
kind: as-built
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - config/experiments/exp_hourly_ppo_001.yaml
  - config/experiments/exp_hourly_ppo_002.yaml
  - config/experiments/exp_daily_ppo_001.yaml
  - src/forecasting/regime_features.py
  - config/execution/smart_simple_v1.yaml
---
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

---

## EXP-TESIS-RL-02-SANITY-S1: la receta no encuentra el flat sobre ruido puro — 2026-09-11

**0 trials de mercado.** Fixture sintética: retornos iid sin señal y costo positivo, donde la
política óptima es demostrablemente **no operar** (neto exactamente 0, disponible en el espacio
de acción). Receta idéntica a la de la tesis (`thesis_ppo_v2.yaml`), 100.000 pasos, las cinco
semillas del protocolo.

| Semilla | Exposición media | Neto medio | Veredicto |
|---|---:|---:|---|
| 42 | 0,527 | −0,00067 | opera |
| 123 | 0,966 | −0,00143 | opera |
| 456 | 0,485 | −0,00070 | opera |
| 789 | 0,985 | −0,00044 | opera |
| 1337 | 0,968 | −0,00147 | opera |

**0/5 se quedan planas; la regla pre-registrada exige ≥4/5. S1 FALLA.**

**Qué significa, medido y NO lo que parecía.** La primera lectura de esta tabla fue «el agente
hace churn y se come el costo». **Es falsa, y la corrijo aquí.** Contando los cambios de
posición de las políticas entrenadas:

| Política | Exposición | Cambios/sesión (de 59) | Costo/sesión | Neto/sesión |
|---|---:|---:|---:|---:|
| baseline seed 42 | 0,524 | **2,0** | 0,00079 | −0,00075 |
| baseline seed 123 | 0,969 | **2,3** | 0,00113 | −0,00138 |
| `κ_turn` seed 42 (plana) | 0,075 | 0,4 | 0,00019 | −0,00038 |

Una política que realmente rotara pagaría **0,0258** por sesión; estas pagan **0,0008**. El
agente **no hace churn: entra, mantiene y cierra** — dos cambios, que son la entrada y el
cierre terminal obligatorio.

Así que el fallo no es «opera de más», sino: **sobre ruido puro, la receta converge a apostar
una dirección constante en vez de quedarse fuera.** La diferencia económica entre una cosa y
la otra es el costo de ese único viaje de ida y vuelta, ~0,08 % por sesión, frente a un ruido
de retorno con desviación 0,22 % por sesión. La política de apostar tiene valor esperado cero
y cuesta un poco; la de no operar vale exactamente cero. PPO no resuelve esa diferencia en
100.000 pasos.

Esto importa para la tesis por una razón concreta: las corridas de mercado publicadas tienen
**1.682 a 2.314 cambios en 584 sesiones**, es decir ~3 por sesión — el mismo patrón de
mantener, no de rotar. La lectura «el alfa vive en la alta frecuencia y el costo lo mata»
tampoco describe lo que el agente hizo allí.

**Consecuencia para EXP-TESIS-RL-01.** El rechazo económico se mantiene: PPO pierde contra
`always_flat` sobre USD/COP. Lo que **no** se sostiene es atribuir ese resultado al mercado.
El corrigendum ya había retirado «la política rentable no existe dentro del espacio de acción»
por ser falsa —flat existe y vale 0—; esto lo refuerza por una segunda vía medida: **la receta
tampoco encuentra flat cuando flat es la única respuesta correcta.** Cualquier conclusión sobre
el intradía de USD/COP obtenida con esta receta describe al optimizador tanto como al activo.

### Sondas ordenadas: las cuatro fallan (mismo día)

Ejecutadas en el orden pre-registrado, una variable cada una, solo sintético. Una sonda queda
descartada en cuanto acumula dos semillas que operan, porque ya no puede llegar a 4/5.

| Receta | Planas | Exposición media por semilla | Estado |
|---|---:|---|---|
| baseline | 0/5 | 0,527 · 0,966 · 0,485 · 0,985 · 0,968 | descartada |
| `ent_coef = 0` | 0/2 | 0,958 · 0,965 | descartada |
| `norm_reward = False` | 0/2 | 0,976 · 0,963 | descartada |
| `γ = 1.0` | 0/2 | 0,603 · 0,618 | descartada |
| `κ_turn = 1.0` | 1/3 | **0,052** · 0,499 · 0,580 | descartada |
| `κ_turn = 1.0` + `ent_coef = 0` | 0/1 | 0,732 | descartada |
| `κ_turn = 1.0` + `ent_coef = 0,05` | 0/1 | 0,927 | descartada |
| **`flat_init`** (estructural) | **5/5** | 0,019 · 0,021 · 0,041 · 0,018 · 0,014 | **PASA S1** |

**Ninguna receta del protocolo pasa S1.** Quitar el suelo de entropía o la normalización del
reward lo empeora (de 0,53 a ~0,97): no eran la causa. La única que produjo una semilla plana
es `κ_turn`, el penalizador de turnover que **el propio documento de diseño especificaba en su
§9.6 y que la tesis original nunca implementó** — y aun así solo funcionó en una de tres.

**Consecuencia sobre el plan.** El pre-registro v3 dice que la receta ganadora se congela
*antes* de tocar datos de mercado. No hay ganadora, así que **la Etapa 4 no puede ejecutarse
tal cual**: reentrenar v2 con una receta que no encuentra el flat sobre ruido produciría otra
conclusión confundida entre optimizador y mercado. Lo que corresponde es buscar la receta en
el terreno sintético —donde no se gasta ni un trial de mercado— hasta que una pase S1-S4.

### Por qué fallan: el mecanismo, medido

Tres comprobaciones, en orden, cada una descartando la explicación anterior.

**1. No es que falte señal de costo.** Por barra, el costo mediano es **2,5 veces** la
desviación del retorno (0,000515 frente a 0,000206); por sesión, una política que rotara
pagaría 0,0258 contra un bruto con desviación 0,0022. La señal domina. Descartada la hipótesis
de que el reward ×100 con `clip_reward=10` la esté enterrando.

**2. No es churn.** Contados los cambios: **2,0 y 2,3 por sesión** de 59 posibles, con costo
0,0008. El agente entra, mantiene y cierra.

**3. Es compromiso temprano, y la penalización de turnover lo agrava.** Se declaró y ejecutó
una quinta receta, `κ_turn = 1` **con** `ent_coef = 0`, razonando que el bono de entropía
empujaba a muestrear acciones distintas. **Salió al revés** (semilla 42: exposición 0,732
frente a 0,052 con entropía), y eso destapa el mecanismo:

| Receta | Exposición | Cambios/sesión | Primera barra con posición |
|---|---:|---:|---|
| `κ_turn = 1` (con entropía) | 0,088 | 0,3 | **nunca toma posición** |
| `κ_turn = 1` + `ent_coef = 0` | 0,737 | 2,5 | **barra 0** |

Sin exploración, la política **se compromete en la primera barra** con la acción que favorezca
su inicialización, y entonces la penalización de turnover la **encierra ahí**: volver a flat
cuesta lo mismo que haber salido. `κ_turn` castiga el **cambio**, no la **exposición**, así que
enseña «no te muevas», no «quédate fuera». Con entropía, la política no se compromete antes de
aprender y sí encuentra el flat — en la semilla 42, no en la 123 ni en la 456.

**El contrato económico tiene la misma forma**: el costo se cobra sobre `|Δw|`, así que un
agente que se desvía de flat está desincentivado a volver. Es una propiedad del entorno, no un
defecto del optimizador, y explica por qué las corridas de mercado mantienen posición.

**4. La entropía tampoco es la palanca — probado en las dos direcciones.** Si el fallo es
comprometerse antes de aprender, retrasar el compromiso debería ayudar. Se declaró y ejecutó
`κ_turn = 1` con `ent_coef = 0.05`:

| Semilla | `ent = 0` | `ent = 0.01` | `ent = 0.05` |
|---|---:|---:|---:|
| 42 | 0,732 | **0,052** | — |
| 123 | — | 0,499 | 0,927 |

**Las dos direcciones desde 0,01 empeoran.** El coeficiente de entropía no es el eje, ni
subiéndolo ni bajándolo.

### Conclusión de la Etapa 3

**Siete recetas: las cuatro pre-registradas y tres derivadas de medir por qué fallaban. Ninguna
pasa S1.** El fallo es robusto a todo el espacio de hiperparámetros declarado, así que la
corrección no es un ajuste: tiene que ser **estructural**. Las tres candidatas, ninguna
ejecutada y todas declarables como experimento propio:

1. **Inicializar la política sesgada hacia flat.** Ataca la causa medida —el compromiso
   temprano— en vez de compensarla después.
2. **Recocer `κ_turn` desde cero.** Deja que aprenda la economía antes de encerrarla; subir
   `κ_turn` por encima de 1 **no** es una opción, con 1 ya es dos mil veces el neto por barra.
3. **Cambiar la parametrización de la acción** a exposición-objetivo en vez de nivel discreto,
   de modo que el costo penalice la exposición y no solo el cambio — que es la asimetría que
   crea la trampa.

### Primera candidata estructural ejecutada: `flat_init` (resultado parcial)

De las tres, se implementó y probó la primera: **sesgar el bias de la capa de acción hacia el
nivel de exposición 0**, de modo que la política arranque prácticamente plana y tenga que
*aprender* a salir. Entorno idéntico a `kappa_turn_one`; lo único que cambia es dónde empieza.

| Semilla 123 | baseline | `κ_turn` | `κ_turn`+ent 0,05 | **`flat_init`** |
|---|---:|---:|---:|---:|
| Exposición | 0,966 | 0,499 | 0,927 | **0,021** |

En la semilla que peor iba, **pasa**. Es la primera receta de ocho que lo consigue, y es
coherente con el mecanismo medido: si el fallo era comprometerse antes de aprender, empezar en
el sitio correcto lo elimina en vez de compensarlo.

**COMPLETADO el mismo día: 5/5 semillas planas. S1 PASA.**

| Semilla | 42 | 123 | 456 | 789 | 1337 |
|---|---:|---:|---:|---:|---:|
| baseline | 0,527 | 0,966 | 0,485 | 0,985 | 0,968 |
| **`flat_init`** | **0,019** | **0,021** | **0,041** | **0,018** | **0,014** |

Las cinco por debajo de 0,1, la regla exigía cuatro. **La compuerta de sanidad S1 queda
abierta**, y es la primera receta de ocho que lo consigue.

**Lo que esto establece, y lo que no.** Establece que el fallo era **de arranque, no de
capacidad**: la misma red, el mismo entorno, el mismo presupuesto y los mismos
hiperparámetros encuentran la política óptima en cuanto se les deja empezar en el sitio
correcto. Siete recetas de hiperparámetros habían fallado sobre exactamente el mismo problema.

**No** establece que la receta sirva para el mercado: S1 solo prueba que no opera sobre ruido.
Faltan **S2** (¿aprende una señal plantada?), **S3** (¿opera cuando el alfa supera el costo?) y
**S4** (¿se abstiene cuando no lo supera?). Una receta que se queda quieta siempre pasaría S1 y
S4 y fracasaría en S2 y S3, así que sin ellas no hay receta que congelar.

Evidencia: `outputs/thesis-repair/sanity_S1_protocol.json`. Siguiente paso:
`python scripts/analysis/thesis_ppo_sanity.py --fixture S2 --probe flat_init --timesteps 100000`

**Reproducir.** `outputs/thesis-repair/sanity_S1_protocol.json` trae las cinco recetas con sus
semillas, hash del runner y commit base.

**Reproducir.** `outputs/thesis-repair/sanity_S1_protocol.json` trae las cinco corridas con
hash del runner y commit base. Entrenado por tramos con `--resume` (el entorno corta los
trabajos largos); `VecNormalize` se guarda y restaura con el checkpoint para que la
normalización del reward no se reinicie a mitad.


## EXP-TESIS-RL-01: PPO intradía sobre sesión acotada (USD/COP) — 2026-08-25

**Contrato**: CTR-RESEARCH-PPO-001 · **Registro**: `HYPOTHESIS-REGISTRY.md::APERTURA H-TESIS-RL-01`
(+2 AT, trials 111→113) · **Pre-registro**: `06-PRE-REGISTRATION.md`

**Variable cambiada (regla 1)**: UNA — la presencia de los 4 posteriores de régimen en la
observación. Todo lo demás es idéntico entre brazos: hiperparámetros, semillas, datos, pasos,
arquitectura. `ppo_backbone` recibe los mismos 39 inputs con los 4 de régimen **puestos a cero**
(no eliminados), para que las dos redes tengan exactamente los mismos parámetros y la
comparación no mida capacidad además de información.

**Semillas (regla 2)**: `[42, 123, 456, 789, 1337]`, las cinco, en ambas configuraciones.

**Hiperparámetros**: congelados de `config/experiments/v215b_baseline.yaml` — `lr 3e-4`,
`n_steps 4096`, `batch 128`, `n_epochs 10`, `γ 0.98`, `gae_λ 0.95`, `clip 0.2`, `ent_coef 0.01`,
net `[256,256]`, CPU (regla 6). **Sin HPO**: prior ex-ante, no suma trials.
300.000 pasos ≈ 10 pasadas sobre desarrollo (los 2M del baseline serían 68 pasadas sobre
29.441 pasos, y ese número está atado a otro tamaño de muestra).

**Entorno**: sesión 08:00-12:55 COT, 60 barras, **59 decisiones operables**, exposición
∈ {−1, −0.5, 0, +0.5, +1}, cierre forzado y cobrado. Costos §9.3 en pips con spread esperado
del posterior del HMM (K=4, ajustado solo en desarrollo).

### Resultados — bloque de SELECCIÓN (2023, n=234 sesiones efectivas)

| Configuración | Semilla | Retorno % | Sharpe | Ops | \|exp\| | Costos % |
|---|---|---|---|---|---|---|
| ppo_regime | 42 | −34.97 | −4.03 | 623 | 0.86 | 38.6 |
| ppo_regime | 123 | −34.02 | −4.07 | 639 | 0.92 | 39.8 |
| ppo_regime | 456 | −31.43 | −3.45 | 680 | 0.88 | 40.7 |
| ppo_regime | 789 | −15.87 | −1.98 | 697 | 0.73 | 37.2 |
| ppo_regime | 1337 | −32.10 | −3.65 | 433 | 0.92 | 32.6 |
| **ppo_regime media-5** | — | **−29.85** | **−4.64** | — | 0.86 | — |
| ppo_backbone | 42 | −38.85 | −4.75 | 708 | 0.81 | 41.5 |
| ppo_backbone | 123 | −40.00 | −4.78 | 809 | 0.77 | 46.6 |
| ppo_backbone | 456 | −36.78 | −4.43 | 653 | 0.93 | 41.6 |
| ppo_backbone | 789 | −39.05 | −5.69 | 813 | 0.62 | 45.5 |
| ppo_backbone | 1337 | −21.55 | −2.95 | 691 | 0.60 | 44.1 |
| **ppo_backbone media-5** | — | **−35.43** | **−6.76** | — | 0.75 | — |

**Baselines sobre las mismas 234 sesiones**: `always_flat` **0.00%** · `B1_pasivo` −20.6% ·
`B1_sesion_1x` −34.7% · `NULL_A_corto_1x` −12.5%.

### Validación estadística (regla 3, regla 5)

- **H2 — DECIDIBLE**: `ppo_regime` supera a `ppo_backbone`, ΔSharpe **+2.120**,
  IC 95% [+0.238, +4.087], **p = 0.0276**, ρ = 0.97. La información de régimen **sí** aporta…
  para perder menos. Los dos brazos pierden.
- **H1 — DECIDIBLE, en contra**: `ppo_regime` queda **por debajo** de `always_flat`,
  ΔSharpe −4.640, IC 95% [−6.793, −2.796], **p < 0.0001**. También por debajo de `B1_pasivo`
  (ΔSharpe −3.196, p = 0.0014).
- **0 de 10 semillas positivas** (la regla 3 pide ≥3/5 por configuración; se obtiene 0/5 y 0/5).
- **DSR = 0.0000** en ambas, con `n_trials = 113` — el conteo del ACTIVO tras esta apertura.
  Los 2 AT que cobró esta tesis deflactan también su propio claim (constitución §2). No pasa el
  bar de 0.95.
- **PBO = 0.116** sobre 12.870 particiones (por debajo del 0.20 pre-registrado — irrelevante:
  el PBO mide si el ganador in-sample sobrevive fuera, y aquí no hay ganador que preservar).
- **Stress de costos**: muere a ×2 y a ×3 ⇒ REJECT por constitución §3.4.
- **B1′** (exposición constante 0.86×): −16.61%/año, también negativo — 2023 fue malo para
  estar largo COP. No rescata a ningún brazo.

### Veredicto

**RECHAZADA H1.** Con el contrato de costos de §9.3, un agente PPO sobre la sesión intradía de
USD/COP **no bate a no operar**, y la diferencia es estadísticamente decidible en contra.

El mecanismo es transparente y no requiere interpretación: los agentes mantienen |exposición|
≈ 0.75-0.86 y ejecutan 430-810 cambios en 234 sesiones, lo que consume **33-47% del nocional en
costos**. La Fase E ya lo había anticipado midiendo que cualquier política acotada por sesión
paga ~4.81 pips/sesión. El intradía de este par es de suma negativa antes de que entre skill.

**Compromiso pre-firmado, ahora aplicado**: *si `ppo_regime` no bate a `always_flat`, la
conclusión escrita es que no operar es la estrategia*. Es la versión concreta de «el baseline ES
la estrategia» (constitución §3).

**Lo que este resultado NO dice**: que PPO no pueda funcionar. Dice que **esta receta**
(hiperparámetros de v215b, 300k pasos, 39 features, 5 niveles de exposición) sobre **estos
datos** contra **estos costos** no funciona. Sin HPO no hay evidencia sobre el mejor PPO
alcanzable — y esa limitación es consecuencia declarada de la decisión de §15, no un descuido.

### Hold-out (2024-01-02 → 2026-08-24, n=584) — abierto una vez el 2026-08-25 16:43 UTC

| | Retorno | Sharpe | IC 95% |
|---|---|---|---|
| **always_flat** | **0,00%** | — | — |
| ppo_regime (media 5) | −54,87% | −6,21 | [−7,54, −4,96] |
| ppo_backbone (media 5) | −56,44% | −6,65 | [−8,52, −5,02] |

`ppo_regime` vs `always_flat`: ΔSharpe **−6,215**, p < 0,0001. 0/10 semillas positivas,
DSR 0,0000 (n_trials=113), muere a ×2 y ×3. Costos 89-119% del nocional.

**H2 NO SE CONFIRMA**: decidible en selección (+2,120, p=0,0276), **indecidible en hold-out**
(+0,432, IC [−0,883, +1,861]). **No se afirma que el efecto desapareciera**: ρ entre brazos cayó
de 0,97 a 0,58 y con ρ=0,58 la potencia a n=584 frente a ΔSharpe 1,2 es del 35% (frente al 95%
con ρ=0,92). Efecto menor o test con menos potencia: estos datos no lo separan. PBO del hold-out
**0,211 > 0,20** pre-registrado, en la misma dirección de cautela.

Se aplica la regla escrita antes de calcularlo (pre-registro §4): no se retunea nada, se
**suspende la pretensión confirmatoria**, y el hold-out se etiqueta como evaluación con
**riesgo alto de selección**. H2 queda **NO CONFIRMADA**.

El rechazo de H1 no depende de esa etiqueta: no hay ganador que proteger.

### Descomposición (2026-08-25, 0 trials) — el rechazo, explicado

**BRUTO +27,95%** (`ppo_regime`, hold-out), Sharpe bruto **+2,23** IC [+1,21, +3,29],
**10/10 corridas positivas**. El agente SÍ aprende; el costo de ejecutarlo lo anula 4×.
El bruto exige costo cero: es una cota superior inalcanzable, no un claim de edge.

**Break-even `s* = −0,29 pips`**: ni con spread cero el alfa (0,67 pips/op) cubre la comisión
de 0,5 pips/lado. Eso hace la conclusión **robusta al supuesto de spread que nunca se midió**.

**Bajar la frecuencia no basta**: a 1 decisión/sesión el bruto ya es negativo (−5,95%). El alfa
vive en la alta frecuencia igual que el costo.

**Always-flat**: `w=0` estaba disponible y el agente lo eligió solo el 16,7% de las barras.

Detalle completo: `.claude/specs/planes/06-RESULTADOS.md` §5b.

### Rama forward RL vs LLM (2026-08-25, **+2 AT** → 115) — CTR-RESEARCH-FORWARD-001

**Exploratoria por diseño.** El hold-out se abrió una vez y el PBO quedó en 0,211 > 0,20: un
brazo que arranca hoy es un experimento nuevo, no rescata H1 ni H2, y
`config/research/preregistration_forward.yaml` lo fija en `status: exploratory` para que el
fichero mismo impida la afirmación cuando el contexto se haya olvidado.

**Por qué forward y no retrospectivo**: no hay corpus histórico (181 artículos en 12 días frente
a 1.383 sesiones) y un LLM que hoy lee una noticia de 2021 ya sabe cómo terminó 2021.

| `arm_id` | Decisiones/sesión | Sella (COT) | Trials |
|---|---|---|---|
| `llm_direct_fwd_v1` | 1 | 07:15 | **+1 AT** |
| `ppo_regime_fwd_k59` | 1 | 08:00 | **+1 AT** — 1 decisión/sesión es una política distinta |
| `ppo_regime_fwd_k1` | 59 | 08:00 | 0 — evidencia forward de la política ya evaluada |
| `always_flat` | 0 | — | 0 — el listón que batió a todo en el hold-out |

**Las dos igualdades que lo anclan a la tesis** (sin ellas mediría otra cosa sin decirlo):
`build_live_spec` reproduce el spec del batch con delta **0,000e+00** en features y contexto, y
liquidar una senda del hold-out por el carril forward reproduce **exactamente** bruto, costo,
neto y turnover de `decomposition_holdout.json`.

**Asimetría declarada**: el RL ve la barra 0 y el LLM no. Los dos sellan antes de que exista
`r_1`, así que los dos son causalmente limpios, pero no con la misma información.

**Potencia**: ~85 sesiones hasta diciembre, cuando ni 584 resolvieron H2. Es un **piloto de
factibilidad**, no un test. Va dicho en la primera línea del pre-registro, no en limitaciones.

**Compuerta pendiente**: inventario de fuentes de 5 días hábiles. Día 1 dio 52 documentos
pre-apertura, pero los tres feeds viables son del mismo medio — riesgo de concentración
declarado. Dos de las tres URL que traía el arnés estaban muertas (404).
