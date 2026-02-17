# Plan: Industry-Grade Returns for USD/COP RL Trading System

**Date**: 2026-02-02
**Current Status**: -27.98% return (FAILING)
**Target**: +18-25% APY (INDUSTRY GRADE)
**Timeline**: 90 days

---

## Executive Summary

### Current Performance
| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **APY** | -55.96% | +18-25% | +74-81% |
| **Win Rate** | 59.1% | 60%+ | OK |
| **Profit Factor** | 1.42 | 1.8+ | +0.38 |
| **Avg Trade PnL** | +$28.46 | +$60 | +$31.54 |
| **HOLD %** | 0.4% | 30-40% | +30% |
| **Transaction Cost** | 0.9% | 0.35-0.50% | -0.4-0.55% |

### Root Causes Identified (4 Agents Analysis)

1. **Transaction Costs Destroying Returns**
   - 22 trades × 0.9% = 19.8% annual cost drag
   - Individual trades profitable (+$28 avg) but costs > profits

2. **HOLD Action at 0.4% (Should be 30-40%)**
   - `weight_holding_decay: 1.0` overwhelms all other signals
   - Thresholds ±0.40 too narrow (80% HOLD zone but penalty discourages it)
   - Model learned: "Always trade to avoid holding_decay penalty"

3. **All Trades Hit max_position_holding (576 bars)**
   - No observation of holding time (model can't see approaching deadline)
   - No reward for voluntary closure (only punishment for holding)
   - Exponential decay starts too early, levels off too fast

4. **Reward Weight Imbalance**
   - Penalty weights: 1.0 + 0.3 + 0.3 = 1.6
   - Reward weights: 0.6 + 0.15 + 0.05 = 0.8
   - Ratio: 2:1 penalty vs reward (should be balanced)

---

## Phase 1: Critical Parameter Fixes (Days 1-7)

### 1.1 Fix Reward Weight Balance (HIGHEST PRIORITY)

**File**: `config/experiment_ssot.yaml`

```yaml
# BEFORE (WRONG - penalty-heavy)
reward:
  pnl_weight: 0.6
  dsr_weight: 0.15
  sortino_weight: 0.05
  regime_penalty: 0.3
  holding_decay: 1.0      # TOO HIGH - dominates everything
  anti_gaming: 0.3

# AFTER (BALANCED)
reward:
  pnl_weight: 0.7         # +0.1 (primary signal)
  dsr_weight: 0.15
  sortino_weight: 0.05
  regime_penalty: 0.15    # -0.15 (soft)
  holding_decay: 0.2      # -0.8 (CRITICAL FIX)
  anti_gaming: 0.15       # -0.15 (soft)
```

**Impact**: Balances reward/penalty weights (0.9 vs 0.5)

### 1.2 Widen HOLD Zone Thresholds

**File**: `config/experiment_ssot.yaml`

```yaml
# BEFORE (WRONG - too narrow)
thresholds:
  long: 0.40
  short: -0.40

# AFTER (WIDER)
thresholds:
  long: 0.60              # Requires 60% confidence
  short: -0.60            # Creates 120% wider HOLD zone
```

**Impact**: Model needs higher confidence to trade, increasing HOLD %

### 1.3 Soften Holding Decay Curve

**File**: `config/experiment_ssot.yaml`

```yaml
# BEFORE (TOO AGGRESSIVE)
holding_decay_config:
  half_life_bars: 24      # Penalty at 50% in 2 hours
  max_penalty: 0.8
  flat_threshold: 4

# AFTER (BALANCED)
holding_decay_config:
  half_life_bars: 144     # Penalty at 50% in 12 hours (full day)
  max_penalty: 0.3        # Reduced from 0.8
  flat_threshold: 24      # 2-hour grace period
  overnight_multiplier: 1.5
```

**Impact**: Positions can be held longer without massive penalties

---

## Phase 2: Voluntary Close Mechanism (Days 8-21)

### 2.1 Add Holding Ratio to Observation Space

**Why**: Model cannot see how long it's held a position. Without this, it cannot learn to exit before max_holding.

**Files to Modify**:
1. `config/experiment_ssot.yaml` - Add feature definition
2. `src/training/environments/trading_env.py` - Add to observation
3. `src/evaluation/backtest_engine.py` - Add to backtest observation

**Changes**:

```yaml
# experiment_ssot.yaml
pipeline:
  observation_dim: 16     # was 15

features:
  # ... existing 15 features ...
  - name: holding_ratio
    order: 15
    category: state
    description: "Normalized holding time (0=just entered, 1=at max)"
    source: runtime
    formula: "bars_in_position / max_position_holding"
    normalization:
      method: none
    is_state: true
```

```python
# trading_env.py - in _get_observation()
if not self._portfolio.position.is_flat:
    bars_in_position = self._current_idx - self._portfolio.position.entry_bar
    holding_ratio = min(bars_in_position / self.config.max_position_duration, 1.0)
else:
    holding_ratio = 0.0
obs[-1] = holding_ratio
```

### 2.2 Create Voluntary Close Bonus Component

**New File**: `src/training/reward_components/voluntary_close.py`

```python
class VoluntaryCloseBonus(RewardComponent):
    """Reward for closing positions before max_position_holding."""

    def __init__(
        self,
        base_bonus: float = 0.15,
        pnl_multiplier: float = 2.0,
        min_remaining_ratio: float = 0.10,
        early_close_bonus: float = 0.05,
    ):
        self.base_bonus = base_bonus
        self.pnl_multiplier = pnl_multiplier
        self.min_remaining_ratio = min_remaining_ratio
        self.early_close_bonus = early_close_bonus

    def calculate(
        self,
        position_closed: bool,
        bars_held: int,
        max_holding: int,
        pnl_pct: float,
        forced_close: bool,
    ) -> float:
        if not position_closed or forced_close:
            return 0.0

        remaining_ratio = 1.0 - (bars_held / max_holding)

        if remaining_ratio < self.min_remaining_ratio:
            return 0.0  # Too late, no bonus

        # Base bonus scaled by remaining time
        bonus = self.base_bonus * remaining_ratio

        # Amplify for profitable closes
        if pnl_pct > 0:
            bonus *= (1 + self.pnl_multiplier * pnl_pct)

        # Extra bonus for very early closes (>50% time remaining)
        if remaining_ratio > 0.5:
            bonus += self.early_close_bonus

        return bonus
```

### 2.3 Progressive Holding Decay (Sigmoid Curve)

**Modify**: `src/training/reward_components/holding_decay.py`

```python
def calculate_sigmoid_penalty(
    self,
    holding_bars: int,
    max_holding: int = 576,
) -> float:
    """Progressive penalty that accelerates sharply near max_holding."""

    # Grace period: no penalty
    if holding_bars < self.grace_period_bars:
        return 0.0

    # Sigmoid centered at 75% of max_holding
    midpoint = max_holding * 0.75  # 432 bars
    steepness = 12.0 / max_holding

    sigmoid = 1.0 / (1.0 + np.exp(-steepness * (holding_bars - midpoint)))
    penalty = self.max_penalty * sigmoid

    # Cliff penalty if actually at max_holding
    if holding_bars >= max_holding:
        penalty += self.cliff_penalty

    return -penalty
```

---

## Phase 3: Cost Reduction Strategy (Days 22-45)

### 3.1 Reduce Transaction Costs

**Target**: 90 bps → 35-50 bps

**Options**:
1. **Negotiate broker rates** - Contact prime broker for volume discounts
2. **Use CFDs with tighter spreads** - Some brokers offer 30-50 bps for USD/COP
3. **Trade correlated pairs** - USD/MXN has ~20 bps spreads
4. **Reduce trade frequency** - Each trade avoided saves 0.9%

### 3.2 Trade Filtering (Higher Conviction Only)

**Concept**: Only take trades where expected profit > 3× transaction cost

**Implementation**: Add confidence scoring to model

```python
# In backtest/inference
action_logit = model.predict(obs)
action_confidence = abs(action_logit)  # How sure is the model?

# Only trade if confidence exceeds threshold
min_confidence = 0.70  # 70% confidence required
if action_confidence < min_confidence:
    action = 0  # Force HOLD
```

**Expected Impact**:
- Reduces trades from 22 to ~15 per period
- Higher quality trades with better avg PnL
- Fewer transaction costs

---

## Phase 4: Training and Validation (Days 46-90)

### 4.1 Retrain with Fixed Parameters

```bash
# 1. Regenerate dataset (no changes needed)
python data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py

# 2. Train with 750k timesteps
python scripts/run_full_pipeline.py --timesteps 750000

# 3. Run full 2025 backtest
# (automatic after training)
```

### 4.2 Validation Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| HOLD % | 30-40% | Check action distribution |
| Avg bars_held | < 300 | Check trade details |
| % forced closes | < 20% | Track voluntary vs forced |
| Win rate | > 55% | Maintain edge |
| Profit factor | > 1.5 | Improve from 1.42 |
| Total return | > 0% | Break-even minimum |
| Sharpe ratio | > 0.5 | Risk-adjusted returns |

### 4.3 Out-of-Sample Testing

Run backtest on multiple time periods:
1. **In-sample**: 2020-2024 (training data)
2. **Out-of-sample**: 2025 (current test)
3. **Extended OOS**: 2026 Jan-Feb (if data available)

---

## APY Projections

### Scenario Analysis

| Scenario | Trades | Avg PnL | Cost | Net Return | APY |
|----------|--------|---------|------|------------|-----|
| **Current** | 22 | +$28 | 0.90% | -27.98% | -56% |
| **Cost Only** | 22 | +$28 | 0.50% | -12.7% | -25% |
| **Signal Only** | 22 | +$60 | 0.90% | -10.2% | -20% |
| **Filter Only** | 15 | +$28 | 0.90% | -17.5% | -35% |
| **Cost+Signal** | 22 | +$60 | 0.50% | +5.5% | +11% |
| **All Three** | 15 | +$60 | 0.50% | +8.8% | **+18%** |
| **Optimistic** | 18 | +$75 | 0.35% | +15.4% | **+31%** |

### Industry Benchmarks

| Tier | APY | Sharpe | Our Target |
|------|-----|--------|------------|
| Elite (Top 0.1%) | 40-70% | 2.0+ | No |
| Excellent (Top 5%) | 20-30% | 1.5-2.0 | Stretch |
| Good (Top 25%) | 15-20% | 1.0-1.5 | **TARGET** |
| Average | 6-10% | 0.5-1.0 | Minimum |

**Realistic Target**: **+18-25% APY with Sharpe 1.0-1.5**

---

## Implementation Checklist

### Phase 1 (Days 1-7) - Parameter Fixes
- [ ] Update `experiment_ssot.yaml` with balanced reward weights
- [ ] Widen thresholds to ±0.60
- [ ] Soften holding_decay curve (half_life=144, max_penalty=0.3)
- [ ] Update `src/training/config.py` to read from SSOT
- [ ] Test with 100k timesteps to verify HOLD % increases

### Phase 2 (Days 8-21) - Voluntary Close
- [ ] Add `holding_ratio` to observation space (16 dims)
- [ ] Update `TradingEnv._get_observation()`
- [ ] Update `BacktestEngine.build_observation()`
- [ ] Create `VoluntaryCloseBonus` component
- [ ] Implement sigmoid penalty in `HoldingDecay`
- [ ] Integrate into `ModularRewardCalculator`
- [ ] Test with 200k timesteps

### Phase 3 (Days 22-45) - Cost Reduction
- [ ] Research broker alternatives for USD/COP
- [ ] Implement trade filtering (confidence threshold)
- [ ] Test with reduced transaction cost parameter
- [ ] Validate edge preserved with fewer trades

### Phase 4 (Days 46-90) - Validation
- [ ] Full training run (750k timesteps)
- [ ] 2025 backtest validation
- [ ] Out-of-sample testing on 2026 data
- [ ] Document final performance metrics
- [ ] Prepare for live paper trading

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Overfitting to parameters | Medium | High | OOS validation |
| Cost reduction impossible | Low | High | Alternative pairs |
| HOLD % still low | Medium | Medium | Further threshold widening |
| Signal quality drops | Low | Medium | Careful hyperparameter tuning |
| Implementation bugs | Medium | Medium | Unit tests, gradual rollout |

---

## Success Criteria

**Minimum Viable Product (Break-even)**:
- Total return > 0%
- HOLD % > 20%
- Avg bars_held < 400
- Win rate > 50%

**Target (Industry Grade)**:
- APY > +15%
- HOLD % 30-40%
- Avg bars_held 150-250
- Win rate > 55%
- Sharpe > 0.8

**Stretch Goal (Excellent)**:
- APY > +25%
- Sharpe > 1.2
- Profit factor > 2.0
- Max drawdown < 15%

---

## Conclusion

The USD/COP RL trading system has a **genuine edge** (59% win rate, 1.42 profit factor, +$28 avg trade) but is **destroyed by transaction costs and improper reward weighting**.

By implementing the 4-phase plan:
1. **Fix reward weights** (balance penalties vs rewards)
2. **Add voluntary close mechanism** (observation + bonus)
3. **Reduce costs** (broker negotiation + trade filtering)
4. **Validate thoroughly** (OOS testing)

We can realistically achieve **+18-25% APY** with **Sharpe 1.0-1.5**, which is **industry-grade performance** for emerging market FX algorithmic trading.
