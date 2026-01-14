# PPO V21 - Model Documentation

**Version**: v21
**Status**: PENDING TRAINING
**Planned Training Date**: 2026-01 (TBD)
**Based On**: V20 with comprehensive fixes from expert analysis

---

## Overview

V21 is the successor to V20, designed to fix all identified issues from the catastrophic V20 performance.
This version incorporates feedback from 4 exploration agents and expert review.

---

## Configuration Files

### Primary Config Path
```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\config\v21_config.yaml
```

### Normalization Stats Path (Reusing V20)
```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\config\v20_norm_stats.json
```

### Implementation Plan
```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\V21_IMPLEMENTATION_PLAN_v2.md
```

---

## Key Changes from V20

| Parameter | V20 | V21 | Rationale |
|-----------|-----|-----|-----------|
| threshold_long | 0.10 | **0.33** | 66% HOLD zone vs 20% |
| threshold_short | -0.10 | **-0.33** | Symmetric HOLD zone |
| transaction_cost_bps | 25 | **75** | Real USDCOP spread |
| slippage_bps | 2-5 | **15** | Realistic for 5-min bars |
| gamma | 0.99 | **0.90** | Shorter-term focus for noisy data |
| ent_coef | 0.01 | **0.05** | More exploration |
| loss_penalty_multiplier | 1.5 | **2.0** | More asymmetric |
| hold_bonus_per_bar | 0.0001 | **0.0** | Disabled (counterproductive) |
| circuit_breaker | None | **5 losses** | Stop after consecutive losses |
| volatility_filter | None | **2x ATR** | Force HOLD in extreme vol |

---

## Training Configuration

```yaml
model:
  name: ppo_v21
  version: "21"
  observation_dim: 15
  action_space: 3  # HOLD, LONG, SHORT

training:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.05          # 5x increase from V20
  clip_range: 0.2
  gamma: 0.90             # Shorter-term focus
  gae_lambda: 0.95
  total_timesteps: 500000

thresholds:
  long: 0.33              # Wider HOLD zone
  short: -0.33
  confidence_min: 0.6

trading:
  initial_capital: 10000
  transaction_cost_bps: 75  # Realistic USDCOP
  slippage_bps: 15
  max_position_size: 1.0

risk:
  max_drawdown_pct: 15.0
  daily_loss_limit_pct: 5.0
  position_limit: 1.0
  volatility_scaling: true
  max_consecutive_losses: 5      # NEW: Circuit breaker
  cooldown_bars_after_losses: 12 # NEW: 1 hour cooldown

reward:
  loss_penalty_multiplier: 2.0   # More asymmetric
  hold_bonus_per_bar: 0.0        # Disabled
  hold_bonus_requires_profit: true
  consecutive_win_bonus: 0.001
  max_consecutive_bonus: 5
  drawdown_penalty_threshold: 0.05
  drawdown_penalty_multiplier: 2.0
  intratrade_dd_penalty: 0.5     # NEW
  max_intratrade_dd: 0.02        # NEW: 2% threshold
  time_decay_start_bars: 24      # NEW: 2 hours
  time_decay_per_bar: 0.0001     # NEW
  time_decay_losing_multiplier: 2.0  # NEW

volatility:
  enable_filter: true            # NEW
  max_atr_multiplier: 2.0        # NEW: Force HOLD if extreme
  force_hold_in_extreme: true    # NEW
  regime_lookback_bars: 60       # NEW
```

---

## Feature Set (15 dimensions - Unchanged from V20)

| Index | Feature | Type | Description |
|-------|---------|------|-------------|
| 0 | log_ret_5m | Price | 5-minute log return |
| 1 | log_ret_1h | Price | 1-hour log return (12 bars) |
| 2 | log_ret_4h | Price | 4-hour log return (48 bars) |
| 3 | rsi_9 | Momentum | RSI with period 9, Wilder's smoothing |
| 4 | atr_pct | Volatility | ATR as % of price, period 10 |
| 5 | adx_14 | Trend | ADX with period 14, Wilder's smoothing |
| 6 | dxy_z | Macro | DXY z-score (60-bar rolling) |
| 7 | dxy_change_1d | Macro | DXY daily change |
| 8 | vix_z | Macro | VIX z-score |
| 9 | embi_z | Macro | EMBI Colombia z-score |
| 10 | brent_change_1d | Macro | Brent crude daily change |
| 11 | rate_spread | Macro | Interest rate spread (Colombia - US) |
| 12 | usdmxn_change_1d | Macro | USD/MXN daily change (proxy) |
| 13 | position | State | Current position (-1, 0, 1) |
| 14 | time_normalized | State | Session progress (0-1) |

**Note**: V21.0 keeps same features. V21.1 may add `vol_regime` as feature 15.

---

## New Components in V21

### 1. Circuit Breaker

**Purpose**: Stop trading after consecutive losses to prevent catastrophic drawdowns.

```python
# Configuration
max_consecutive_losses: 5      # Trigger after 5 losses
cooldown_bars_after_losses: 12 # 1 hour cooldown (12 x 5min)

# Behavior
- Track consecutive losses
- After 5 consecutive losses: Force HOLD for 12 bars
- Reset counter after any winning trade
- Log warning when triggered
```

**Expected Impact**: Prevents sequences of 6-8 losses observed in V20.

### 2. Volatility Hard Stop

**Purpose**: Force HOLD during extreme volatility events.

```python
# Configuration
enable_filter: true
max_atr_multiplier: 2.0  # If current ATR > 2x historical mean
force_hold_in_extreme: true

# Behavior
- Calculate rolling historical ATR mean
- Compare current ATR to historical
- If ATR > 2x mean: Force HOLD regardless of model output
- Protects against events like Trump tariff announcements
```

**Expected Impact**: Avoids $312 loss from Jan 10-11 events.

### 3. Enhanced Reward Shaping

**Purpose**: Better incentive alignment.

```python
# Components
1. Base PnL (scaled)
2. Asymmetric loss penalty (2.0x multiplier)
3. Transaction cost (ADDITIVE)
4. Hold bonus: DISABLED (was counterproductive)
5. Consistency bonus (for winning streaks)
6. Account drawdown penalty (>5% threshold)
7. NEW: Intratrade drawdown penalty (>2% threshold)
8. NEW: Time decay for stale positions (after 24 bars)
```

**Time Decay Logic**:
```python
if bars_held > 24:  # 2 hours
    time_decay = (bars_held - 24) * 0.0001
    if pnl < 0:  # Losing position
        time_decay *= 2.0  # Double penalty
```

---

## Training Data

| Parameter | Value |
|-----------|-------|
| Training Start | 2020-03-01 |
| Training End | 2024-12-31 |
| Validation Start | 2025-01-01 |
| Validation End | 2025-06-30 |
| Test Start | 2025-07-01 |
| Timeframe | 5 minutes |
| Dataset | RL_DS3_MACRO_CORE.csv |
| Total Bars | ~84,671 |

---

## Expected Results

### Target Metrics

| Metric | V20 Actual | V21 Minimum | V21 Target |
|--------|------------|-------------|------------|
| Sharpe Ratio | -5.00 | > 0.0 | 0.3 - 0.5 |
| Max Drawdown | -44.6% | < 30% | < 25% |
| Win Rate | 13.6% | > 35% | 40-45% |
| HOLD Rate | ~5% | > 30% | 35-45% |
| Long/Short Ratio | 90/10 | 40-60% | ~50% |
| Avg Trade Duration | Unknown | < 24 bars | < 18 bars |
| Max Consecutive Losses | 8+ | < 6 | < 5 |

### Behavioral Expectations

1. **More Conservative Trading**
   - Fewer trades due to wider thresholds
   - Higher quality setups only

2. **Better Risk Management**
   - Circuit breaker limits losing streaks
   - Volatility filter avoids extreme events

3. **Balanced Directionality**
   - ~50% LONG, ~50% SHORT instead of 90% LONG
   - Better adaptation to market conditions

4. **Shorter Hold Times**
   - Time decay penalizes stale positions
   - Faster exits from losing trades

---

## Implementation Checklist

### Code Changes Required

- [ ] `src/training/environments/trading_env.py`
  - [ ] threshold_long: 0.10 → 0.33
  - [ ] threshold_short: -0.10 → -0.33
  - [ ] transaction_cost_bps: 25 → 75
  - [ ] slippage_bps: 2 → 15
  - [ ] Add circuit breaker logic
  - [ ] Add volatility filter logic

- [ ] `src/training/reward_calculator_v20.py`
  - [ ] loss_penalty_multiplier: 1.5 → 2.0
  - [ ] hold_bonus_per_bar: 0.0001 → 0.0
  - [ ] Add intratrade_dd_penalty
  - [ ] Add time_decay logic

- [ ] `airflow/dags/l3_model_training.py`
  - [ ] gamma: 0.99 → 0.90
  - [ ] ent_coef: 0.01 → 0.05
  - [ ] transaction_cost_bps: 25 → 75

- [ ] `services/inference_api/core/inference_engine.py`
  - [ ] Remove hardcoded MODEL_THRESHOLDS

- [ ] `services/inference_api/core/trade_simulator.py`
  - [ ] Remove hardcoded MODEL_THRESHOLDS

- [ ] `airflow/dags/services/backtest_factory.py`
  - [ ] Update default thresholds to 0.33/-0.33

### Validation Checklist (Pre-Training)

- [ ] All thresholds consistent across files
- [ ] Transaction costs set to 75 bps everywhere
- [ ] gamma = 0.90 in DAG defaults
- [ ] ent_coef = 0.05 in DAG defaults
- [ ] Circuit breaker added to environment
- [ ] Volatility filter added to environment

### Validation Checklist (Post-Training)

- [ ] HOLD rate > 30%
- [ ] Long/Short ratio between 0.4 and 0.6
- [ ] Average trade duration < 24 bars
- [ ] Max consecutive losses < 6
- [ ] Sharpe ratio > 0.0 on OOS
- [ ] Max drawdown < 30% on OOS

---

## Airflow Variable Configuration

For quick deployment without code changes:

**Key**: `training_config`
**Value**:
```json
{
  "version": "v21",
  "gamma": 0.90,
  "ent_coef": 0.05,
  "transaction_cost_bps": 75.0,
  "slippage_bps": 15.0,
  "threshold_long": 0.33,
  "threshold_short": -0.33,
  "total_timesteps": 500000
}
```

---

## Files Reference

### Configuration
```
config\v21_config.yaml                         # V21 configuration
config\v20_norm_stats.json                     # Normalization stats (reused)
```

### Training Pipeline
```
airflow\dags\l3_model_training.py              # Main training DAG
src\training\environments\trading_env.py       # Trading environment
src\training\reward_calculator_v20.py          # Reward function
src\training\trainers\ppo_trainer.py           # PPO wrapper
src\training\environments\env_factory.py       # Environment factory
```

### Inference Pipeline
```
airflow\dags\l5_multi_model_inference.py       # Inference DAG
services\inference_api\core\inference_engine.py # Inference engine
services\inference_api\core\trade_simulator.py  # Trade simulator
```

### Feature Pipeline
```
src\feature_store\core.py                      # Feature builder (SSOT)
src\core\calculators\regime.py                 # Regime detection (to integrate)
```

### Documentation
```
V21_IMPLEMENTATION_PLAN_v2.md                  # Full implementation plan
versioning\v20\README.md                       # V20 documentation
versioning\v21\README.md                       # This file
```

---

## Results (Pending)

**Training Results**: TBD after training execution

**Out-of-Sample Results**: TBD after backtest on Jan 2025 data

**Production Results**: TBD after deployment

---

## Changelog

### V21.0 (Planned)
- Wider thresholds (0.33/-0.33)
- Realistic transaction costs (75 bps)
- Lower gamma (0.90)
- Higher entropy (0.05)
- More asymmetric loss penalty (2.0x)
- Disabled hold bonus
- Circuit breaker (5 losses)
- Volatility hard stop (2x ATR)

### V21.1 (Future, if needed)
- Add vol_regime as feature 16
- Position sizing by volatility
- Consider 15-min timeframe

---

## Risk Warnings

1. **No Guarantee of Profit**: V21 is designed to fix V20 issues but profitability is not guaranteed
2. **Market Conditions**: USDCOP is an illiquid, emerging market pair with high spreads
3. **Regime Changes**: Model trained on 2020-2024 may not adapt to future regimes
4. **Transaction Costs**: Actual costs may vary from 75 bps estimate
5. **Slippage**: 5-min execution may have higher slippage than modeled

---

## Next Steps

1. Apply all code changes
2. Configure Airflow Variable as backup
3. Execute training DAG
4. Evaluate on OOS (Jan 2025)
5. Compare metrics to targets
6. If acceptable, deploy to production
7. Monitor live performance

---

## Contact

For questions about this model version, refer to:
- Implementation plan: `V21_IMPLEMENTATION_PLAN_v2.md`
- V20 post-mortem: `versioning/v20/README.md`
