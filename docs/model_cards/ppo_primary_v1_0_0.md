# Model Card: ppo_primary

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ppo_primary |
| **Version** | 1.0.0 |
| **Type** | PPO (Proximal Policy Optimization) |
| **Framework** | Stable-Baselines3 |
| **Created** | 2026-01-14 |
| **Author** | Trading Team |
| **Status** | Production |

## Intended Use

### Primary Use Case
Automated trading signal generation for USD/COP currency pair using reinforcement learning.

### Out-of-Scope Uses
- Live trading without human oversight
- Trading other currency pairs without retraining
- High-frequency trading (sub-second decisions)

## Training Data

### Dataset
| Field | Value |
|-------|-------|
| **Dataset Name** | RL_DS3_MACRO_CORE |
| **Date Range** | 2020-03-02 to 2025-10-29 |
| **Rows** | 84,671 |
| **Features** | 15 (13 market + 2 state) |
| **Data Version** | HEAD |

### Feature Set
```
log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14,
dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d,
rate_spread, usdmxn_change_1d, position, time_normalized
```

### Normalization Stats
| Feature | Mean | Std |
|---------|------|-----|
| log_ret_5m | 0.000012 | 0.001580 |
| log_ret_1h | 0.000072 | 0.003860 |
| log_ret_4h | 0.000288 | 0.007720 |
| rsi_9 | 49.850000 | 15.230000 |
| atr_pct | 0.008500 | 0.003200 |
| adx_14 | 22.400000 | 9.800000 |
| dxy_z | 0.000000 | 1.000000 |
| dxy_change_1d | 0.000050 | 0.003500 |
| vix_z | 0.000000 | 1.000000 |
| embi_z | 0.000000 | 1.000000 |
| brent_change_1d | 0.000150 | 0.025000 |
| rate_spread | 7.030000 | 1.410000 |
| usdmxn_change_1d | 0.000080 | 0.008500 |

**Norm Stats Hash**: `a1b2c3d4e5f6`

## Model Architecture

### Network
```
Policy Network (pi):
  - Input: 15 dimensions
  - Hidden: [256, 256]
  - Activation: Tanh
  - Output: 3 (HOLD, BUY, SELL)

Value Network (vf):
  - Input: 15 dimensions
  - Hidden: [256, 256]
  - Activation: Tanh
  - Output: 1 (state value)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| N Steps | 2048 |
| Batch Size | 128 |
| N Epochs | 10 |
| Gamma | 0.90 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coef | 0.05 |

## Performance Metrics

### Backtest Results
| Metric | Value |
|--------|-------|
| Total Return | TBD |
| Sharpe Ratio | TBD |
| Sortino Ratio | TBD |
| Max Drawdown | TBD |
| Win Rate | TBD |
| Profit Factor | TBD |

### Action Distribution
| Action | Percentage |
|--------|------------|
| HOLD | ~60% |
| BUY | ~20% |
| SELL | ~20% |

## Limitations and Biases

### Known Limitations
- Performance may degrade in extreme market conditions (VIX > 40)
- Requires minimum 14 bars warmup before valid predictions
- Trained on 5-minute bars; not suitable for other timeframes
- USD/COP specific; may not generalize to other EM pairs

### Potential Biases
- Training data from 2020-2025 may overweight COVID recovery period
- Model may have learned patterns specific to Colombian market hours

### Failure Modes
- Circuit breaker activates if >20% features are NaN
- Model outputs HOLD during warmup period
- May exhibit stuck behavior in low volatility regimes

## Deployment

### Requirements
- Python 3.9+
- ONNX Runtime 1.15+
- norm_stats.json (exact version used in training)

### Inference Latency
| Metric | Target |
|--------|--------|
| p50 | < 20ms |
| p95 | < 50ms |
| p99 | < 100ms |

### Model Files
| File | Hash |
|------|------|
| model.onnx | `TBD` |
| norm_stats.json | `a1b2c3d4e5f6` |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial production release with 15-feature observation space |

---
*Generated on 2026-01-14 by generate_model_card.py*
