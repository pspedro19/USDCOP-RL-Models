# V21 PPO Implementation Plan - Comprehensive Fix for USDCOP Trading

## Executive Summary

**Problem**: V20 PPO model shows catastrophic performance:
- Sharpe Ratio: -5.00
- Max Drawdown: -44.6%
- Win Rate: 13.6% (132 trades OOS)
- LONG Bias: 90%+ trades are LONG

**Root Causes Identified**:
1. **No effective HOLD action** (95% certainty) - Thresholds too narrow (0.10/-0.10)
2. **Insufficient transaction costs** (100% certainty) - 25 bps vs real 50-200 bps for USDCOP
3. **Missing intratrade drawdown penalty** (85% certainty) - Only penalizes end-of-step DD
4. **Entropy too low** (80% certainty) - 0.01 causes premature exploitation
5. **Gamma too high** (70% certainty) - 0.99 overfits to noise in 5-min data
6. **Missing volatility filter** (90% certainty) - No regime detection

---

## Phase 0 (P0) - CRITICAL FIXES

These changes are **mandatory** before the next training run.

### 1. Fix Action Thresholds in Environment

**File**: `src/training/environments/trading_env.py`

**Current** (lines 151-152):
```python
threshold_long: float = 0.10  # Action > 0.10 → LONG
threshold_short: float = -0.10  # Action < -0.10 → SHORT
```

**Problem**: With PPO's continuous output typically centered around 0, a threshold of 0.10 means 80% of the action space triggers a trade. The agent has almost no HOLD zone.

**Fix**: Widen thresholds to create a meaningful HOLD zone.

```python
threshold_long: float = 0.33  # Action > 0.33 → LONG (top 33%)
threshold_short: float = -0.33  # Action < -0.33 → SHORT (bottom 33%)
```

**Also update v20_config.yaml** (lines 21-23):
```yaml
thresholds:
  long: 0.33
  short: -0.33
  confidence_min: 0.6
```

---

### 2. Increase Transaction Costs to Real USDCOP Values

**File**: `src/training/environments/trading_env.py`

**Current** (lines 143-144):
```python
transaction_cost_bps: float = 25.0  # 25 bps = 0.25%
slippage_bps: float = 2.0  # 2 bps slippage
```

**Problem**: Real USDCOP spread is 20-80 pips. At USDCOP ~4200:
- 1 pip = 0.01 COP = ~0.0000024 in percentage
- 40 pips = 0.40 COP = ~0.00952% per side
- Round trip cost = ~100-200 bps for USDCOP (vs 25 bps training)

**Fix**: Increase to realistic values:
```python
transaction_cost_bps: float = 75.0  # 75 bps = 0.75% (conservative estimate)
slippage_bps: float = 15.0  # 15 bps slippage (5-min bars have significant slippage)
```

**Also update v20_config.yaml** (lines 32-33):
```yaml
trading:
  transaction_cost_bps: 75
  slippage_bps: 15
```

---

### 3. Fix Reward Calculator - Add Intratrade Drawdown Penalty

**File**: `src/training/reward_calculator_v20.py`

**Current**: Only penalizes account-level drawdown, not intratrade drawdown.

**Add new config parameter** (after line 47):
```python
# Intratrade drawdown penalty
intratrade_dd_penalty: float = 0.5  # Penalty per % intratrade DD
max_intratrade_dd: float = 0.02  # 2% max intratrade DD threshold
```

**Modify calculate() method** - Add intratrade DD tracking:

After line 99, add parameter:
```python
def calculate(
    self,
    pnl_pct: float,
    position_change: int,
    bars_held: int = 0,
    consecutive_wins: int = 0,
    current_drawdown: float = 0.0,
    intratrade_drawdown: float = 0.0  # NEW: Max DD since position opened
) -> Tuple[float, dict]:
```

After Step 6 (line 170), add Step 7:
```python
# ==========================================================
# STEP 7: Intratrade Drawdown Penalty (NEW V21)
# ==========================================================
intratrade_penalty = 0.0
if intratrade_drawdown > self.config.max_intratrade_dd:
    excess_dd = intratrade_drawdown - self.config.max_intratrade_dd
    intratrade_penalty = excess_dd * self.config.intratrade_dd_penalty * 100
breakdown['intratrade_penalty'] = intratrade_penalty

final_reward = (
    reward_after_cost
    + hold_bonus
    + consistency_bonus
    - drawdown_penalty
    - intratrade_penalty  # ADD THIS
)
```

---

### 4. Fix Hyperparameters

**File**: `config/v20_config.yaml` (create new v21_config.yaml)

**Current**:
```yaml
training:
  ent_coef: 0.01
  gamma: 0.99
```

**Problem**:
- `ent_coef: 0.01` is too low - agent exploits too early without exploring
- `gamma: 0.99` is too high - 5-min bars have high noise, long-term rewards are unreliable

**Fix** (v21_config.yaml):
```yaml
training:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.05  # INCREASED from 0.01 - more exploration
  clip_range: 0.2
  gamma: 0.95  # DECREASED from 0.99 - focus on shorter-term rewards
  gae_lambda: 0.95
```

---

## Phase 1 (P1) - HIGH PRIORITY

These changes significantly improve performance but aren't blocking.

### 5. Add Volatility Filter (Regime Detection)

**File**: `src/feature_store/core.py`

Add new calculator for volatility regime detection.

**Add after ATRPercentCalculator class** (after line 428):
```python
class VolatilityRegimeCalculator(BaseCalculator):
    """
    Volatility regime detector using ATR percentile.

    Returns:
    - 0.0: Low volatility regime (ATR < 25th percentile)
    - 0.5: Normal regime (25th-75th percentile)
    - 1.0: High volatility regime (ATR > 75th percentile)
    """

    def __init__(self, lookback: int = 60):
        super().__init__(
            name="vol_regime",
            requires=["high", "low", "close"],
            window=lookback
        )
        self._lookback = lookback

    def _calculate_impl(self, data: pd.DataFrame, bar_idx: int) -> float:
        if bar_idx < self._lookback:
            return 0.5

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        # Calculate ATR for lookback window
        atr_values = []
        for i in range(bar_idx - self._lookback + 1, bar_idx + 1):
            if i > 0:
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                atr_values.append(tr / close[i] if close[i] > 0 else 0)

        if len(atr_values) < 10:
            return 0.5

        current_atr = atr_values[-1]
        p25 = np.percentile(atr_values, 25)
        p75 = np.percentile(atr_values, 75)

        if current_atr < p25:
            return 0.0  # Low vol
        elif current_atr > p75:
            return 1.0  # High vol
        return 0.5  # Normal

    def _calculate_batch_impl(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_pct = tr / close

        rolling = atr_pct.rolling(window=self._lookback, min_periods=10)
        p25 = rolling.quantile(0.25)
        p75 = rolling.quantile(0.75)

        regime = pd.Series(0.5, index=data.index)
        regime[atr_pct < p25] = 0.0
        regime[atr_pct > p75] = 1.0

        return regime.fillna(0.5)
```

**Register in CalculatorRegistry._initialize_v20()** (line 656):
```python
# Add to V21
"vol_regime": VolatilityRegimeCalculator(lookback=60),
```

---

### 6. Add Volatility Scaling to Position Sizing

**File**: `src/training/environments/trading_env.py`

Modify `_execute_action()` to scale position by volatility regime.

**Add to TradingEnvConfig** (after line 152):
```python
# Volatility scaling
use_volatility_scaling: bool = True
vol_scale_low: float = 1.5  # Scale up in low vol
vol_scale_high: float = 0.5  # Scale down in high vol
vol_regime_feature_idx: int = 15  # Index in observation (for V21)
```

---

### 7. Add Time Decay Penalty for Long Positions

**File**: `src/training/reward_calculator_v20.py`

**Problem**: Agent holds positions too long, hoping for reversal.

**Add to RewardConfig** (after line 50):
```python
# Time decay (penalize holding losing positions)
time_decay_start_bars: int = 24  # Start penalty after 2 hours (24 5-min bars)
time_decay_per_bar: float = 0.0001  # Penalty per bar after threshold
time_decay_losing_multiplier: float = 2.0  # 2x decay for losing positions
```

**Add Step 8 in calculate()** (after Step 7):
```python
# ==========================================================
# STEP 8: Time Decay Penalty (V21)
# ==========================================================
time_decay = 0.0
if bars_held > self.config.time_decay_start_bars:
    excess_bars = bars_held - self.config.time_decay_start_bars
    time_decay = excess_bars * self.config.time_decay_per_bar
    if pnl_pct < 0:  # Losing position
        time_decay *= self.config.time_decay_losing_multiplier
breakdown['time_decay'] = time_decay

final_reward = final_reward - time_decay
```

---

## Phase 2 (P2) - NICE TO HAVE

### 8. Consider Higher Timeframe (15-min or 30-min)

**Problem**: 5-min bars for USDCOP have:
- High noise-to-signal ratio
- High spread relative to move size
- More false breakouts

**Consideration**: If V21 still underperforms, train on 15-min bars.

**Files to modify**:
- `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py` → Create `01_build_15min_datasets.py`
- All technical indicators need period adjustments (RSI 9 → RSI 6, etc.)

---

### 9. Add Colombian Market Features

**Potential additions**:
- TRM (Tasa Representativa del Mercado) daily fixing time
- Colombian holiday calendar
- BanRep intervention probability

**File**: `src/feature_store/core.py` - Add new calculators for Colombia-specific features.

---

### 10. Walk-Forward Validation Improvement

**Current**: Single train/test split
**Suggested**: K-fold walk-forward with purging

**File to modify**: Training orchestration scripts

---

## Complete File Change Summary

| File | Priority | Changes |
|------|----------|---------|
| `src/training/environments/trading_env.py` | P0 | Thresholds, transaction costs, volatility scaling |
| `src/training/reward_calculator_v20.py` | P0 | Intratrade DD penalty, time decay |
| `config/v21_config.yaml` (NEW) | P0 | All hyperparameters (gamma, ent_coef, costs) |
| `src/feature_store/core.py` | P1 | VolatilityRegimeCalculator |
| `config/v21_norm_stats.json` (NEW) | P1 | Add vol_regime stats |
| `src/training/trainers/ppo_trainer.py` | P1 | Update default hyperparams |

---

## V21 Configuration File (CREATE NEW)

**File**: `config/v21_config.yaml`

```yaml
# V21 Configuration
# Contrato: GTR-008
# FIXES: Thresholds, transaction costs, hyperparameters

model:
  name: ppo_v21
  version: "21"
  observation_dim: 15  # Keep same for V21.0, add vol_regime in V21.1
  action_space: 3  # HOLD, BUY, SELL

training:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.05  # INCREASED from 0.01
  clip_range: 0.2
  gamma: 0.95  # DECREASED from 0.99
  gae_lambda: 0.95

thresholds:
  long: 0.33  # INCREASED from 0.10
  short: -0.33  # INCREASED from -0.10
  confidence_min: 0.6

features:
  norm_stats_path: "config/v20_norm_stats.json"  # Reuse V20 stats initially
  clip_range: [-5.0, 5.0]
  warmup_bars: 14

trading:
  initial_capital: 10000
  transaction_cost_bps: 75  # INCREASED from 25
  slippage_bps: 15  # INCREASED from 5
  max_position_size: 1.0

risk:
  max_drawdown_pct: 15.0
  daily_loss_limit_pct: 5.0
  position_limit: 1.0
  volatility_scaling: true

reward:
  loss_penalty_multiplier: 1.5
  hold_bonus_per_bar: 0.0001
  consecutive_win_bonus: 0.001
  drawdown_penalty_threshold: 0.05
  drawdown_penalty_multiplier: 2.0
  intratrade_dd_penalty: 0.5  # NEW
  max_intratrade_dd: 0.02  # NEW
  time_decay_start_bars: 24  # NEW
  time_decay_per_bar: 0.0001  # NEW

dates:
  training_start: "2020-03-01"
  training_end: "2024-12-31"
  validation_start: "2025-01-01"
  validation_end: "2025-06-30"
  test_start: "2025-07-01"
```

---

## Validation Checklist (Before Training)

- [ ] **P0.1**: Thresholds changed to 0.33/-0.33 in env config
- [ ] **P0.2**: Transaction cost increased to 75 bps
- [ ] **P0.3**: Slippage increased to 15 bps
- [ ] **P0.4**: Intratrade DD penalty added to reward calculator
- [ ] **P0.5**: ent_coef increased to 0.05
- [ ] **P0.6**: gamma decreased to 0.95
- [ ] **P0.7**: v21_config.yaml created and referenced

## Validation Checklist (After Training)

- [ ] **T1**: HOLD rate > 30% of decisions
- [ ] **T2**: Long/Short ratio between 0.4 and 0.6
- [ ] **T3**: Average trade duration < 24 bars (2 hours)
- [ ] **T4**: Max intratrade DD per trade < 3%
- [ ] **T5**: OOS Sharpe > 0 (at minimum)
- [ ] **T6**: OOS Max DD < 25%
- [ ] **T7**: Win rate > 35%

---

## Expected Impact

| Metric | V20 (Current) | V21 (Expected) |
|--------|---------------|----------------|
| Sharpe | -5.00 | > 0.5 |
| Max DD | -44.6% | < 25% |
| Win Rate | 13.6% | > 40% |
| HOLD Rate | ~5% | > 30% |
| Avg Trade Duration | Unknown | < 24 bars |
| Transaction Costs | Underestimated | Realistic |

---

## Implementation Order

1. Create `config/v21_config.yaml` (copy from above)
2. Modify `src/training/environments/trading_env.py`:
   - Update `TradingEnvConfig` defaults
   - Or load from v21_config.yaml
3. Modify `src/training/reward_calculator_v20.py`:
   - Add intratrade DD penalty
   - Add time decay
4. Run training with new config
5. Evaluate on OOS data
6. If Sharpe < 0.5, proceed to P1 (volatility filter)

---

## Notes

- **DO NOT** modify V20 files directly. Create V21 versions.
- Keep V20 config frozen for reproducibility.
- Log all changes in training runs.
- Run backtest on same Jan 2025 data to compare V20 vs V21.
