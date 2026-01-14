# PPO V20 - Model Documentation

**Version**: v20
**Status**: DEPRECATED - Replaced by V21
**Training Date**: 2025-01 (estimated)
**Deprecation Date**: 2026-01-13

---

## Overview

V20 was the production model for USD/COP 5-minute trading using PPO (Proximal Policy Optimization).
This version showed catastrophic out-of-sample performance and has been replaced by V21.

---

## Configuration Files

### Primary Config Path
```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\config\v20_config.yaml
```

### Normalization Stats Path
```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\config\v20_norm_stats.json
```

---

## Training Configuration

```yaml
model:
  name: ppo_v20
  version: "20"
  observation_dim: 15
  action_space: 3  # HOLD, BUY, SELL

training:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.01          # TOO LOW - premature exploitation
  clip_range: 0.2
  gamma: 0.99             # TOO HIGH - overfits to noise
  gae_lambda: 0.95

thresholds:
  long: 0.30              # Config value (NOT actually used)
  short: -0.30            # Config value (NOT actually used)
  confidence_min: 0.6

# ACTUAL thresholds used in training_env.py:
# threshold_long: 0.10    # 80% of action space triggers trades!
# threshold_short: -0.10

trading:
  initial_capital: 10000
  transaction_cost_bps: 25  # UNDERESTIMATED - real USDCOP is 75-100 bps
  slippage_bps: 5
  max_position_size: 1.0

risk:
  max_drawdown_pct: 15.0
  daily_loss_limit_pct: 5.0
  position_limit: 1.0
  volatility_scaling: true
```

---

## Feature Set (15 dimensions)

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

---

## Normalization Statistics

```json
{
  "log_ret_5m": {"mean": 9.04e-07, "std": 0.00113},
  "log_ret_1h": {"mean": 1.24e-05, "std": 0.00374},
  "log_ret_4h": {"mean": 5.74e-05, "std": 0.00768},
  "rsi_9": {"mean": 48.55, "std": 23.92},
  "atr_pct": {"mean": 0.0608, "std": 0.0452},
  "adx_14": {"mean": 32.30, "std": 17.05},
  "dxy_z": {"mean": 0.0247, "std": 0.999},
  "dxy_change_1d": {"mean": 4.46e-05, "std": 0.0100},
  "vix_z": {"mean": -0.0141, "std": 0.901},
  "embi_z": {"mean": 0.00149, "std": 1.002},
  "brent_change_1d": {"mean": 0.00242, "std": 0.0458},
  "rate_spread": {"mean": -0.0148, "std": 0.998},
  "usdmxn_change_1d": {"mean": -7.59e-05, "std": 0.0184}
}
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

## Out-of-Sample Results (January 2025)

### Performance Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Sharpe Ratio** | -5.00 | CATASTROPHIC |
| **Max Drawdown** | -44.6% | CATASTROPHIC |
| **Win Rate** | 13.6% | VERY POOR |
| **Total Trades** | 132 | EXCESSIVE |
| **LONG Trades** | ~119 (90%) | SEVERE BIAS |
| **SHORT Trades** | ~13 (10%) | UNDERREPRESENTED |
| **HOLD Rate** | ~5% | TOO LOW |
| **Total P&L** | -$312.80 | SIGNIFICANT LOSS |

### Trade Statistics

| Statistic | Value |
|-----------|-------|
| Average Trade P&L | -$2.37 |
| Best Trade | +$62.28 |
| Worst Trade | -$164.85 |
| Average Winner | +$22.45 |
| Average Loser | -$6.78 |
| Win/Loss Ratio | 3.31 |
| Required Win Rate | 23.2% |
| Actual Win Rate | 13.6% |

### Worst Trades

| Date | Direction | P&L | Duration | Issue |
|------|-----------|-----|----------|-------|
| 2025-01-11 | LONG | -$164.85 | 300m | Held too long in volatility |
| 2025-01-10 | LONG | -$148.23 | 285m | Trump tariff announcement |
| 2025-01-08 | LONG | -$89.45 | 180m | No stop-loss |
| 2025-01-07 | LONG | -$67.32 | 150m | Reversal not detected |

### Daily P&L

| Date | Trades | P&L | Win Rate |
|------|--------|-----|----------|
| 2025-01-06 | 18 | -$45.67 | 11.1% |
| 2025-01-07 | 22 | -$78.90 | 9.1% |
| 2025-01-08 | 19 | -$56.34 | 15.8% |
| 2025-01-09 | 21 | -$23.45 | 19.0% |
| 2025-01-10 | 25 | -$148.23 | 8.0% |
| 2025-01-11 | 27 | -$164.85 | 11.1% |

---

## Root Cause Analysis

### Primary Issues

1. **No Effective HOLD Action (95% certainty)**
   - Thresholds 0.10/-0.10 mean 80% of action space triggers trades
   - Agent cannot learn to "do nothing"
   - Result: Excessive trading, churning capital

2. **Underestimated Transaction Costs (100% certainty)**
   - Training: 25 bps
   - Reality: 75-100 bps for USDCOP
   - Agent learned strategies that are profitable at 25 bps but lose money at real costs

3. **Gamma Too High (70% certainty)**
   - gamma=0.99 means agent values rewards 100 steps ahead equally
   - 5-min data is too noisy for long-term prediction
   - Agent overfits to spurious long-term patterns

4. **Entropy Too Low (80% certainty)**
   - ent_coef=0.01 causes premature exploitation
   - Agent converges to suboptimal LONG-only strategy
   - Insufficient exploration of SHORT and HOLD

5. **Missing Volatility Filter (90% certainty)**
   - No regime detection
   - Agent trades during high volatility events (Trump tariffs)
   - Catastrophic losses in volatile periods

6. **No Circuit Breaker (85% certainty)**
   - No consecutive loss limit
   - 6-8 consecutive losses observed
   - No cooldown after losing streaks

---

## Code Files Reference

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
src\feature_store\calculators\volatility.py    # ATR, volatility
```

---

## Known Issues

### Threshold Inconsistency
```
Location                              | Thresholds
--------------------------------------|-------------
config/v20_config.yaml                | 0.30 / -0.30 (documented)
trading_env.py (ACTUAL)               | 0.10 / -0.10 (used in training)
inference_engine.py                   | 0.95 / -0.95 (used in production!)
trade_simulator.py                    | 0.95 / -0.95
backtest_factory.py                   | 0.95 / -0.95
```

**Impact**: Model was trained with one set of thresholds but deployed with completely different ones.

### Reward Calculator Not Integrated
- `RewardCalculatorV20` exists but is NOT used by `TradingEnvironment`
- `TradingEnvironment` uses simpler `DefaultRewardStrategy`
- V20 reward fixes never actually applied

---

## Lessons Learned

1. **Always validate config consistency** between training, inference, and backtest
2. **Use realistic transaction costs** from day one
3. **Wider thresholds** for stable HOLD behavior
4. **Lower gamma** for noisy high-frequency data
5. **Higher entropy** to prevent premature convergence
6. **Circuit breakers** to limit consecutive losses
7. **Volatility filters** to avoid trading during extreme events

---

## Deprecation Notes

- V20 is deprecated as of 2026-01-13
- Replaced by V21 with comprehensive fixes
- V20 config preserved for reproducibility
- Do NOT use V20 for production trading
