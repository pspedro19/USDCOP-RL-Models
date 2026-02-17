# EXP-REGIME-001: Regime Detection Features for USDCOP Forecasting

## Status: COMPLETED — ALL TREATMENTS FAILED
**Created:** 2026-02-16
**Executed:** 2026-02-16
**Author:** Pedro / Claude
**Pipeline:** Forecasting (L0-L7)
**Result:** 0/7 treatment groups passed GATE 1. Regime features do NOT improve DA.
**Decision:** KEEP baseline 19 features. No deployment.

---

## 1. Motivation

### The Problem

The USDCOP forecasting model (9-model ensemble, DA=56.3%, Sharpe 3.135 in 2025) experienced
a regime change in January-February 2026:

| Metric | 2025 | 2026 (43d) | Change |
|--------|------|------------|--------|
| Overall DA | 56.3% | 46.5% | -9.8pp |
| LONG WR | 57.8% | 27.8% | -30.0pp |
| SHORT WR | 64.8% | 56.0% | -8.8pp |
| Hard stop rate | 1.9% | 11.6% | 6.0x |
| Binomial p-value (HS rate) | -- | 0.0014 | Significant |

The model was trained during a sustained COP depreciation regime (2025: USD/COP -12%).
In 2026, the COP shifted to range-bound/slight appreciation. The model's LONG signals
(predicting further depreciation) collapsed because it applies 2025 patterns to a
fundamentally different market regime.

### The Hypothesis

Adding regime-aware features could help the model learn to **condition its predictions
on the current market regime**, rather than applying a single set of coefficients regardless
of whether the market is trending, mean-reverting, or range-bound.

### Why Features, Not a Separate Regime Detector

A separate regime classifier introduces a second model with its own error rate that
compounds with the forecasting model. Instead, we add features that **encode regime
information directly**, letting the existing 9 models learn the interaction between
regime and prediction. This is simpler, more testable, and avoids the cascading error
problem.

The experiment includes both simple hand-crafted regime features (F1-F4) and HMM-derived
probabilistic features (F5). The HMM is not used as a standalone regime classifier -- its
output probabilities are fed as features to the same 9 forecasting models, preserving the
single-model architecture. This lets us compare whether a statistical model of regimes
(HMM) captures information beyond what simple rolling statistics provide.

---

## 2. Pre-Registered Feature Candidates

### Critical Rule
All features are defined **before** seeing any validation results. No feature will be
added, modified, or removed after the walk-forward begins. This prevents data snooping.

### Feature F1: `trend_slope_60d`

**Definition:** Slope of OLS linear regression on `close` over the last 60 trading days,
normalized by the mean close of the window.

```python
def trend_slope_60d(close: pd.Series, window: int = 60) -> pd.Series:
    """Normalized slope of 60-day linear regression on close prices."""
    def _slope(arr):
        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0]
        return slope / np.mean(arr)  # normalize by level
    return close.rolling(window).apply(_slope, raw=True)
```

**What it captures:**
- Positive = COP depreciating (USD/COP trending up)
- Negative = COP appreciating (USD/COP trending down)
- Near zero = range-bound

**Why 60 days:** Long enough to capture regime (not noise), short enough to adapt within
a quarter. 60 days ~ 3 months of trading ~ one macro cycle.

**Expected interaction with model:**
- Ridge: learns beta_slope that adjusts the intercept -- if slope is negative (appreciation
  regime), it should dampen LONG predictions
- XGBoost: can learn splits like "if slope < -0.001, ignore momentum LONG signals"

**Lookahead risk:** None -- uses only past 60 days of close prices, which are available at T.

**Overfitting risk:** Low -- single parameter (window=60), no threshold to optimize.

---

### Feature F2: `range_ratio_20d`

**Definition:** (High - Low range) / Mean close over a 20-day rolling window.

```python
def range_ratio_20d(high: pd.Series, low: pd.Series, close: pd.Series,
                     window: int = 20) -> pd.Series:
    """Normalized range: high = trending, low = range-bound."""
    rolling_range = high.rolling(window).max() - low.rolling(window).min()
    rolling_mean = close.rolling(window).mean()
    return rolling_range / rolling_mean
```

**What it captures:**
- High value (>0.05) = strong trend (up or down), wide price range
- Low value (<0.02) = range-bound, compressed price range
- This is orthogonal to F1: F1 captures direction, F2 captures magnitude

**Expected interaction:**
- When range_ratio is low (range-bound) AND slope is near zero -> model should reduce
  confidence on directional bets (shrink predictions toward zero)
- When range_ratio is high AND slope is strongly directional -> model should increase
  conviction

**Lookahead risk:** None -- rolling window on past prices only.

**Overfitting risk:** Low -- no thresholds, single window parameter.

---

### Feature F3: `return_20d_sign`

**Definition:** Binary indicator: +1 if 20-day return is positive, -1 if negative.

```python
def return_20d_sign(close: pd.Series) -> pd.Series:
    """Simple regime indicator: +1 = recent depreciation, -1 = appreciation."""
    return np.sign(close.pct_change(20))
```

**What it captures:**
- The simplest possible regime signal: has the COP been depreciating or appreciating
  over the last month?

**Note:** `return_20d` (the continuous version) is already in the 19 features. This binary
version is intentionally redundant -- it tests whether the **sign** alone carries
information that the continuous value doesn't. Tree models may find it easier to split on
a binary feature than to find the optimal threshold on a continuous one.

**Expected interaction:**
- If return_20d_sign = -1 (recent appreciation) AND model predicts LONG -> the regime
  feature should dampen or flip the prediction
- Ridge: beta_sign directly adjusts the prediction mean
- XGBoost: first split on sign, then on other features

**Lookahead risk:** None.

**Overfitting risk:** Minimal -- binary feature with fixed derivation.

---

### Feature F4: `vol_regime_ratio`

**Definition:** Ratio of short-term to long-term volatility.

```python
def vol_regime_ratio(close: pd.Series, short: int = 5, long: int = 60) -> pd.Series:
    """Vol regime: >1 = vol expanding (crisis), <1 = vol contracting (calm)."""
    returns = np.log(close / close.shift(1))
    vol_short = returns.rolling(short).std()
    vol_long = returns.rolling(long).std()
    return vol_short / vol_long
```

**What it captures:**
- >1.5 = volatility spike (potential regime transition, shock event)
- ~1.0 = stable vol regime (model predictions more reliable)
- <0.5 = abnormally calm (potential for breakout)

**Why this matters for 2026:** The 5 hard stops occurred during vol spikes that the model
didn't anticipate. If vol_regime_ratio > 1.5, the model might learn to reduce prediction
magnitude -> lower leverage -> smaller hard stop losses.

**Lookahead risk:** None -- ratio of past volatilities.

**Overfitting risk:** Low -- two standard window parameters, no thresholds.

---

### Feature F5: `hmm_regime_probabilities` (Hidden Markov Model)

**Definition:** Posterior probabilities from a 3-state Gaussian HMM fitted on log-returns
using a rolling 504-day (~2 year) training window. The HMM is re-fitted every 21 trading
days (monthly) to limit computational cost and reduce instability. States are sorted by
ascending volatility to solve the label switching problem.

```python
from hmmlearn import GaussianHMM

def hmm_regime_features(close: pd.Series, n_states: int = 3,
                         train_window: int = 504, refit_every: int = 21) -> pd.DataFrame:
    """
    Rolling HMM regime probabilities.

    Uses ROLLING window (not expanding) to avoid lookahead.
    Re-fits every 21 days to limit computation.
    Outputs probabilities (not states) to avoid label switching.
    States sorted by volatility: 0=calm, 1=normal, 2=crisis.

    Returns DataFrame with columns:
        hmm_prob_calm:   P(state=low_vol | data up to T)
        hmm_prob_crisis: P(state=high_vol | data up to T)
        hmm_entropy:     -sum(p * log(p)) -- regime uncertainty
    """
    log_returns = np.log(close / close.shift(1)).dropna()

    result = pd.DataFrame(index=log_returns.index, columns=[
        'hmm_prob_calm', 'hmm_prob_crisis', 'hmm_entropy'
    ], dtype=float)

    refit_indices = set(range(train_window, len(log_returns), refit_every))
    current_model = None

    for i in range(train_window, len(log_returns)):
        # Re-fit on schedule
        if i in refit_indices or current_model is None:
            train_data = log_returns.iloc[i - train_window:i].values.reshape(-1, 1)

            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            model.fit(train_data)

            # Sort states by volatility (ascending) -- fixes label switching
            order = np.argsort(model.covars_.flatten())
            model.means_ = model.means_[order]
            model.covars_ = model.covars_[order]
            model.transmat_ = model.transmat_[order][:, order]
            model.startprob_ = model.startprob_[order]
            current_model = model

        # Predict using last 60 days of context
        context_start = max(0, i - 60)
        context = log_returns.iloc[context_start:i + 1].values.reshape(-1, 1)
        probs = current_model.predict_proba(context)[-1]  # last day only

        date = log_returns.index[i]
        result.loc[date, 'hmm_prob_calm'] = probs[0]     # state 0 = lowest vol
        result.loc[date, 'hmm_prob_crisis'] = probs[-1]   # state 2 = highest vol

        # Entropy: high = uncertain about regime (transition zone)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        result.loc[date, 'hmm_entropy'] = entropy

    return result
```

**What the 3 output features capture:**

| Feature | Range | What it means |
|---------|-------|---------------|
| `hmm_prob_calm` | [0, 1] | Probability of being in the low-volatility regime. High -> model predictions more reliable, market is stable |
| `hmm_prob_crisis` | [0, 1] | Probability of being in the high-volatility regime. High -> hard stops more likely, model should reduce conviction |
| `hmm_entropy` | [0, log(3)] | Regime uncertainty. High -> market transitioning between regimes, model predictions least reliable |

**Why 3 states (K=3):**
- K=2 (bull/bear) is too coarse -- doesn't distinguish "calm trending" from "volatile trending"
- K=3 captures: calm (low vol, small returns), normal (moderate vol, directional), crisis (high vol, extreme returns)
- K=4+ has insufficient data to reliably estimate transition matrices with 504-day windows

**Why rolling (504d) not expanding:**
- Expanding window means regime parameters are dominated by the longest regime in history
- Rolling window adapts: if the market was in crisis 3 years ago but calm now, the old crisis doesn't distort today's classification
- 504 days ~ 2 years -- long enough for stable HMM estimation, short enough to adapt

**Why re-fit every 21 days (not daily):**
- HMM fitting is stochastic (EM algorithm) -- daily re-fitting introduces noise
- Monthly is sufficient: regime transitions happen over weeks, not days
- Computational: ~120 HMM fits per 2,500-day backtest (manageable)

**Lookahead risk:** None. Rolling window uses only data up to T. The 60-day context
for predict_proba uses only past bars. Monthly re-fit uses data strictly before the
prediction date.

**Overfitting risk:** Medium. The HMM has more parameters than F1-F4 (transition matrix,
emission means/covariances). However, outputting probabilities instead of hard states
provides soft information that tree models can threshold optimally. The 3-feature output
(calm prob, crisis prob, entropy) is compact and interpretable.

**Label switching mitigation:** States are sorted by covariance (ascending) after each
fit. State 0 is ALWAYS the lowest-volatility regime, state 2 is ALWAYS the highest.
This ensures consistent semantics across re-fits and walk-forward folds.

**Computational cost:**
- Per HMM fit: ~0.1s (504 observations, 3 states, 100 EM iterations)
- Per full backtest (2,500 days): ~120 fits x 0.1s = ~12 seconds
- Per walk-forward fold: ~12s (features computed once, then used by all 9 models)
- Total experiment: ~60s for HMM features across 5 folds

---

## 3. Experimental Design

### 3.1 Control: Current 19-Feature Model

The baseline is the existing pipeline with exactly the features currently in production:

```
FEATURES_CONTROL = [
    'close', 'open', 'high', 'low',              # Price (4)
    'return_1d', 'return_5d', 'return_10d', 'return_20d',  # Returns (4)
    'volatility_5d', 'volatility_10d', 'volatility_20d',   # Vol (3)
    'rsi_14d', 'ma_ratio_20d', 'ma_ratio_50d',             # Technical (3)
    'day_of_week', 'month', 'is_month_end',                 # Calendar (3)
    'dxy_close_lag1', 'oil_close_lag1'                      # Macro (2)
]  # Total: 19
```

### 3.2 Treatment Groups

| Group | Features | Total | What's Added |
|-------|----------|-------|--------------|
| CONTROL | 19 baseline | 19 | Nothing |
| TREAT-A | 19 + F1 | 20 | trend_slope_60d only |
| TREAT-B | 19 + F1 + F2 | 21 | slope + range |
| TREAT-C | 19 + F1 + F2 + F3 | 22 | slope + range + sign |
| TREAT-D | 19 + F1 + F2 + F3 + F4 | 23 | All 4 simple regime features |
| TREAT-E | 19 + F1 + F4 | 21 | slope + vol_ratio (minimal simple) |
| TREAT-F | 19 + F5 (3 cols) | 22 | HMM only (prob_calm, prob_crisis, entropy) |
| TREAT-G | 19 + F1 + F4 + F5 (3 cols) | 24 | Best simple (slope + vol_ratio) + HMM |

**Why multiple groups:** We don't know which combination works best. Testing individually
and combined avoids the "kitchen sink" trap where correlated features degrade each other.

TREAT-E is the "minimal simple" group -- just trend direction + vol stability.
TREAT-F is the "pure HMM" group -- tests whether the HMM alone captures regime info.
TREAT-G is the "best of both" group -- combines the simplest hand-crafted features with
the HMM. If TREAT-G wins but TREAT-E or TREAT-F alone also pass, we prefer the simpler
one (parsimony rule).

**Evaluation priority** (most preferred -> least preferred):
1. TREAT-E (2 features, simple) -- if this passes, don't need HMM
2. TREAT-A (1 feature, simplest) -- even more minimal
3. TREAT-F (3 features, HMM only) -- if simple features fail but HMM works
4. TREAT-G (5 features, combined) -- only if neither alone is sufficient
5. TREAT-B/C/D (intermediate combinations) -- kitchen sink options

### 3.3 Validation Methodology

**Walk-Forward Expanding Window** (identical to current pipeline):

```
Data: 2015-01-01 -> 2025-12-31 (~2,500 daily bars)
Initial train: 60% (~1,500 bars)
Gap: 30 days (anti-leakage purging)
Step: 10% per fold
Folds: 5

Fold 1: TRAIN [2015-------2021]  -gap-  TEST [2021-2022]
Fold 2: TRAIN [2015---------2022]  -gap-  TEST [2022-2023]
Fold 3: TRAIN [2015-----------2023]  -gap-  TEST [2023-2024]
Fold 4: TRAIN [2015-------------2024]  -gap-  TEST [2024-2025]
Fold 5: TRAIN [2015---------------2025]  -gap-  TEST [2025-...]
```

**Each fold trains:**
- 9 models x 8 groups = 72 model fits per fold
- 5 folds x 72 = 360 total model fits
- Estimated time: ~45 seconds (models train in ~3.5s per group)
  + ~60 seconds for HMM feature computation across all folds
- Total: ~2 minutes

**Metrics computed per fold per group:**
- Direction Accuracy (DA) -- primary metric
- Sharpe ratio (annualized)
- Maximum Drawdown
- Profit Factor
- Mean daily return
- LONG-specific DA
- SHORT-specific DA

### 3.4 Holdout Validation (Post Day 60)

After the walk-forward on 2019-2025, the winning treatment group is tested on
**2026-01-01 -> 2026-03-15** (~45-50 trading days) as a completely untouched holdout.

This is the ONLY test on 2026 data. One shot, no iteration.

```
Walk-forward (2019-2025): Select best treatment group
          |
Holdout (2026 Jan-Mar): Confirm or reject
          |
If confirmed: Deploy to production
If rejected: Keep CONTROL (current 19 features)
```

---

## 4. GATE Criteria (Pre-Registered)

### GATE 1: Walk-Forward (2019-2025)

All criteria must pass for a treatment group to advance to holdout testing:

| # | Criterion | Threshold | Rationale |
|---|-----------|-----------|-----------|
| G1.1 | DA improvement | TREAT_DA > CONTROL_DA in >= 4/5 folds | Consistent improvement, not one-fold fluke |
| G1.2 | Sharpe non-degradation | TREAT_Sharpe >= 0.95 x CONTROL_Sharpe in >= 4/5 folds | Don't sacrifice risk-adjusted return for DA |
| G1.3 | Paired t-test (daily returns) | p < 0.10 (raw) AND p < 0.014 (Bonferroni-adjusted for 7 groups) | Statistical significance with multiple testing correction |
| G1.4 | LONG DA improvement | TREAT_LONG_DA > CONTROL_LONG_DA in >= 3/5 folds | Specifically addresses the 2026 failure mode |
| G1.5 | No MaxDD degradation | TREAT_MaxDD <= 1.15 x CONTROL_MaxDD | Regime features shouldn't increase tail risk |

**If multiple treatment groups pass:** Select the one with the fewest features (parsimony).
If tied, select the one with highest average DA across folds.

**If no treatment group passes:** STOP. Keep current 19 features. Regime features don't
help this model architecture. Consider alternative approaches (Section 7).

### GATE 2: Holdout (2026 Jan-Mar)

Only the winning treatment group is tested here. One evaluation, no re-runs.

| # | Criterion | Threshold | Rationale |
|---|-----------|-----------|-----------|
| G2.1 | DA | TREAT_DA_2026 > 50% | Better than coin flip OOS |
| G2.2 | DA vs CONTROL | TREAT_DA_2026 > CONTROL_DA_2026 | Treatment beats control on new data |
| G2.3 | LONG WR | TREAT_LONG_WR_2026 > 40% | Improvement over 27.8% baseline |
| G2.4 | Hard stop rate | TREAT_HS_rate < 8% | Reduction from 11.6% baseline |

**Pass = deploy to production.** Replace 19 features with winning feature set.

**Fail = reject.** Keep 19 features, keep SHORT-only filter, re-evaluate at Day 90.

---

## 5. Implementation Plan

### 5.1 Script: `scripts/exp_regime_001.py` (~600 lines)

```python
"""
EXP-REGIME-001: Regime Detection Features Walk-Forward Experiment

Usage:
    python scripts/exp_regime_001.py                    # Full walk-forward
    python scripts/exp_regime_001.py --holdout          # Add 2026 holdout
    python scripts/exp_regime_001.py --group TREAT-A    # Single group

Output: results/exp_regime_001.json
"""
```

### 5.2 Feature Integration (if GATE 2 passes)

Modifications required in production:

```
File 1: src/forecasting/regime_features.py (NEW, ~150 lines)
  - Pure functions: trend_slope_60d(), range_ratio_20d(), etc.
  - HMM wrapper: hmm_regime_features() using hmmlearn
  - No I/O, no side effects
  - Unit tested independently
  - Dependency: pip install hmmlearn (added to requirements.txt)

File 2: airflow/dags/forecast_l5a_weekly_inference.py (MODIFY)
  - Import regime_features
  - Add winning features to _build_features()
  - Retrain models with expanded feature set

File 3: airflow/dags/forecast_l5b_daily_inference.py (MODIFY)
  - Same feature addition in daily inference

File 4: config/forecast_experiments/regime_v1.yaml (NEW)
  - Document which features were added, gate results, experiment reference
```

### 5.3 What Does NOT Change

- L5c (vol-targeting): reads predictions, doesn't care about features
- L7 (smart executor): reads signals, doesn't care about features
- L6 (paper trading): evaluates results, doesn't care about features
- Trailing stop parameters: 0.2% activation, 0.3% trail, 1.5% hard stop
- Ensemble strategy: top-3 by magnitude
- Walk-forward structure: 5 folds, expanding window, 30-day gap

---

## 6. Expected Outcomes

### Optimistic Scenario (25% probability)

TREAT-A, TREAT-E, or TREAT-F passes both gates. DA improves 2-3pp (56% -> 58-59%), LONG WR
improves significantly (58% -> 62%+), and the improvement holds in 2026 holdout
(DA > 52%, LONG WR > 45%). If TREAT-F (HMM) wins, it means regime transitions contain
predictive information beyond simple rolling statistics.

**Action:** Deploy new feature set, reactivate LONG trading, remove SHORT-only filter.

### Neutral Scenario (45% probability)

Some treatment groups pass GATE 1 (walk-forward) but fail GATE 2 (2026 holdout).
The features improve in-sample/cross-validation but don't generalize to the new regime.
This is especially likely for TREAT-F/G -- the HMM may fit historical regimes well but
fail to classify the 2026 regime correctly because it's genuinely novel.

**Action:** Keep 19 features, keep SHORT-only filter. The regime change is too fundamental
for additional features to capture. Consider more structural approaches (Section 7).

### Pessimistic Scenario (30% probability)

No treatment group passes GATE 1. Regime features either don't help or actively degrade
the model. The current 19 features are already capturing what's capturable, and the
HMM adds noise from unstable regime estimation.

**Action:** Keep 19 features. Accept that the model has regime-dependent performance.
Focus on regime-adaptive position sizing or regime-conditional trading rules instead
of trying to fix the prediction layer.

---

## 7. Alternative Approaches (If EXP-REGIME-001 Fails)

If regime features don't solve the problem, these are ordered by complexity:

### 7.1 Regime-Adaptive Leverage (Low complexity)
Instead of fixing the prediction, adapt the sizing:
- If trend_slope_60d > 0 (depreciation): full leverage as usual
- If trend_slope_60d < 0 (appreciation): cap leverage at 0.75x
- If trend_slope_60d ~ 0 (range-bound): cap leverage at 0.5x

This doesn't require retraining -- it's a rule in L5c. But it needs walk-forward
validation to confirm it doesn't degrade 2025 returns significantly.

### 7.2 Regime-Conditional Trading (Medium complexity)
- SHORT always ON (demonstrated alpha in both regimes)
- LONG only when trend_slope_60d > threshold (depreciation confirmed)
- Threshold determined by walk-forward, not by 2026 data

This is a more principled version of the current SHORT-only filter.

### 7.3 Shorter Retraining Window (Medium complexity)
Instead of expanding window (all history), use a rolling window of 2-3 years.
The model "forgets" old regimes and adapts to the current one.

**Risk:** Less data per training fold -> more overfitting for boosting models.
**Experiment needed:** Compare expanding vs rolling(2y) vs rolling(3y) in walk-forward.

### 7.4 Online Learning / Incremental Training (High complexity)
Retrain daily with the latest data point (warm-start for boosting models).
This would let the model adapt to regime changes within days, not weeks.

**Risk:** Catastrophic forgetting, concept drift amplification.
**Not recommended** without extensive simulation first.

---

## 8. Timeline

```
2026-02-16: EXP-REGIME-001 pre-registered (this document)
            SHORT-only filter active, paper trading continues

2026-03-15: Day 60 checkpoint
            Evaluate SHORT WR, decide on capital deployment
            Confirm 2026 holdout period = Jan 1 - Mar 15

2026-03-16 -> 2026-03-20: Execute EXP-REGIME-001
            Run walk-forward (2019-2025) for all 8 groups
            Evaluate GATE 1
            If any group passes -> run GATE 2 on 2026 holdout
            Generate report

2026-03-21: Decision
            GATE 2 PASS -> deploy winning features, reactivate LONGs
            GATE 2 FAIL -> keep 19 features, evaluate alternatives (Section 7)
            Document results regardless of outcome

2026-04-01: Post-deployment monitoring (if deployed)
            20-day check: DA with new features > DA without
            If degradation -> rollback to 19 features
```

---

## 9. Anti-Overfitting Checklist

Before declaring any result "significant":

- [x] Features were defined before any validation run (pre-registered in this document)
- [x] No feature was added, modified, or dropped after seeing fold results
- [x] Walk-forward uses expanding window with 30-day purge gap
- [x] Holdout (2026) was touched exactly once
- [x] Paired comparison uses identical signals (same days, same direction/leverage)
- [x] Multiple testing correction applied (Bonferroni: 7 treatment groups -> p < 0.00714)
- [x] Winning group selected by parsimony, not by best p-value
- [x] Results reported for ALL groups, not just the winner
- [x] Negative results documented with same detail as positive results

---

## 10. Results (Executed 2026-02-16)

### GATE 1: Walk-Forward (2019-2025)

| Group | Features | DA Mean | vs CTRL | p-value (paired t) | Bonferroni (0.00714) | Verdict |
|-------|----------|---------|---------|---------------------|----------------------|---------|
| **CONTROL** | **19** | **52.75%** | — | — | — | **BASE** |
| TREAT-A (+F1) | 20 | 52.60% | -0.14pp | 0.8555 | FAIL | No effect |
| TREAT-B (+F1,F2) | 21 | 52.04% | -0.70pp | 0.5159 | FAIL | Slight degradation |
| TREAT-C (+F1,F2,F3) | 22 | 52.58% | -0.16pp | 0.8158 | FAIL | No effect |
| TREAT-D (+F1-F4) | 23 | 52.26% | -0.49pp | 0.4617 | FAIL | Slight degradation |
| TREAT-E (+F1,F4) | 21 | 52.41% | -0.33pp | 0.6897 | FAIL | No effect |
| TREAT-F (+HMM) | 22 | 52.93% | +0.18pp | 0.9373 | FAIL | Noise |
| TREAT-G (+F1,F4,HMM) | 24 | 52.51% | -0.24pp | 0.9080 | FAIL | No effect |

**GATE 1 RESULT: FAIL** — 0/7 treatment groups show significant improvement.

Best treatment (TREAT-F, +0.18pp) is indistinguishable from random variation (p=0.94).
Worst treatment (TREAT-B, -0.70pp) suggests adding range_ratio_20d introduces noise.

### GATE 2: Holdout (2026 Jan-Feb)

Since GATE 1 failed for all treatments, GATE 2 was run only for CONTROL to measure
2026 baseline performance:

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| CONTROL DA (2026) | 47.0% | > 50% | FAIL |
| Period | 2026-01-02 to 2026-02-14 | — | — |
| Trading days | ~30 | — | — |

**GATE 2 RESULT: FAIL** — CONTROL itself only achieves 47% DA on 2026 data.

### Interpretation

1. **Regime features are redundant with existing features.** The 19 baseline features
   (vol_5d/10d/20d, return_5d/20d) already encode regime information implicitly. Adding
   explicit regime features provides no incremental signal.

2. **The HMM doesn't help.** TREAT-F (+0.18pp) is the closest to an improvement but
   p=0.94 means it's pure noise. The 3-state HMM probabilities don't contain
   predictive information beyond what realized volatility already provides.

3. **2026 performance degradation is NOT a feature engineering problem.** CONTROL DA=47%
   on 2026 means the model's underlying signal has degraded. No amount of regime features
   can fix a model that's trained on depreciation patterns applied to an appreciation regime.

4. **The correct intervention remains operational (SHORT-only filter), not model-based.**

### Scenario Outcome

This matches the **Pessimistic Scenario** (Section 6, estimated 30% probability):
> "No treatment group passes GATE 1. Regime features don't help. The current 19 features
> are already capturing what's capturable."

### Next Steps (from Section 7)

Recommended alternatives, ordered by complexity:
1. **7.2 Regime-Conditional Trading** — already implemented as SHORT-only filter
2. **7.1 Regime-Adaptive Leverage** — cap leverage when trend_slope < 0
3. **7.3 Shorter Retraining Window** — rolling 2-3 years instead of expanding

---

## 11. File Checklist

```
PRE-EXPERIMENT:
  [x] docs/EXP_REGIME_001.md            -- This document (pre-registration + results)
  [x] src/forecasting/regime_features.py -- Pure feature functions (F1-F4 + F5 HMM)
  [x] scripts/exp_regime_001.py          -- Experiment runner
  [x] pyproject.toml                     -- Added hmmlearn>=0.3.0 (forecasting extras)

POST-EXPERIMENT:
  [x] results/exp_regime_001/            -- Raw JSON results (all 8 groups, all folds)
  [x] docs/EXP_REGIME_001.md §10        -- Results documented in this file
  [N/A] results/exp_regime_001_holdout.json -- No holdout needed (GATE 1 failed)

NOT DEPLOYED (all treatments failed):
  [ ] config/forecast_experiments/regime_v1.yaml -- Not created (no winner)
  [ ] Modified L5a + L5b DAGs -- Not modified (keeping 19 features)
```
