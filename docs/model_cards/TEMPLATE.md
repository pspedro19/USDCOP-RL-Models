# Model Card: [MODEL_NAME]

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | [e.g., ppo_primary_v2.1] |
| **Version** | [e.g., 2.1.0] |
| **Type** | PPO (Proximal Policy Optimization) |
| **Framework** | Stable-Baselines3 |
| **Created** | [YYYY-MM-DD] |
| **Author** | [Team/Person] |
| **Status** | [Development/Staging/Production/Deprecated] |

## Intended Use

### Primary Use Case
[Describe the primary use case for this model]

### Out-of-Scope Uses
[List uses that are not recommended]

## Training Data

### Dataset
| Field | Value |
|-------|-------|
| **Dataset Name** | [e.g., RL_DS3_MACRO_CORE] |
| **Date Range** | [e.g., 2020-03-02 to 2025-10-29] |
| **Rows** | [e.g., 84,671] |
| **Features** | 15 (13 market + 2 state) |
| **Data Version** | [DVC commit hash] |

### Feature Set
```
log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14,
dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d,
rate_spread, usdmxn_change_1d, position, time_normalized
```

### Normalization Stats
| Feature | Mean | Std |
|---------|------|-----|
| log_ret_5m | [value] | [value] |
| rsi_9 | [value] | [value] |
| ... | ... | ... |

**Norm Stats Hash**: [MD5 of norm_stats.json]

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
| Learning Rate | [e.g., 1e-4] |
| N Steps | [e.g., 2048] |
| Batch Size | [e.g., 128] |
| N Epochs | [e.g., 10] |
| Gamma | [e.g., 0.99] |
| GAE Lambda | [e.g., 0.95] |
| Clip Range | [e.g., 0.2] |
| Entropy Coef | [e.g., 0.05] |

## Performance Metrics

### Backtest Results
| Metric | Value | Benchmark |
|--------|-------|-----------|
| Total Return | [e.g., 15.2%] | [e.g., 8.5%] |
| Sharpe Ratio | [e.g., 1.45] | [e.g., 0.85] |
| Sortino Ratio | [e.g., 2.10] | [e.g., 1.20] |
| Max Drawdown | [e.g., -8.3%] | [e.g., -12.1%] |
| Win Rate | [e.g., 54.2%] | [e.g., 50.0%] |
| Profit Factor | [e.g., 1.35] | [e.g., 1.15] |
| Calmar Ratio | [e.g., 1.83] | [e.g., 0.70] |

### Action Distribution
| Action | Percentage |
|--------|------------|
| HOLD | [e.g., 72%] |
| BUY | [e.g., 14%] |
| SELL | [e.g., 14%] |

### Regime Performance
| Regime | Sharpe | Win Rate |
|--------|--------|----------|
| Low Volatility | [value] | [value] |
| Normal | [value] | [value] |
| High Volatility | [value] | [value] |
| Crisis | [value] | [value] |

## Feature Importance

### SHAP Values (Top 5)
| Rank | Feature | Mean |SHAP| |
|------|---------|--------------|
| 1 | [feature] | [value] |
| 2 | [feature] | [value] |
| 3 | [feature] | [value] |
| 4 | [feature] | [value] |
| 5 | [feature] | [value] |

## Limitations and Biases

### Known Limitations
- [List any known limitations]
- [e.g., Performance degrades in crisis conditions]
- [e.g., Requires minimum 14 bars warmup]

### Potential Biases
- [List any potential biases in the training data or model]

### Failure Modes
- [Describe known failure modes]

## Ethical Considerations

- Model is for paper trading / educational purposes
- Not financial advice
- Users should understand risks involved

## Deployment

### Requirements
- Python 3.9+
- ONNX Runtime 1.15+
- norm_stats.json (exact version used in training)

### Inference Latency
| Metric | Target | Actual |
|--------|--------|--------|
| p50 | < 20ms | [value] |
| p95 | < 50ms | [value] |
| p99 | < 100ms | [value] |

### Model Files
| File | Size | Hash |
|------|------|------|
| model.onnx | [size] | [MD5] |
| norm_stats.json | [size] | [MD5] |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.1.0 | YYYY-MM-DD | [Description] |
| 2.0.0 | YYYY-MM-DD | [Description] |
| 1.0.0 | YYYY-MM-DD | Initial release |

## References

- Training notebook: `notebooks/train_v20_production_parity.py`
- Config: `config/feature_config.json`
- Backtest script: `scripts/backtest.py`

---
*Generated on [DATE] using model_card_generator.py*
