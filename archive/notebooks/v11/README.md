# USD/COP RL Trading System V11

## Overview

Version 11 of the USD/COP Reinforcement Learning Trading System with **critical bug fix** for portfolio calculation.

### V11 Critical Fix

| Bug | V10 | V11 | Impact |
|-----|-----|-----|--------|
| **Normalized Returns** | `log_portfolio += pos * normalized_ret` | `log_portfolio += pos * RAW_ret` | Fixed -100% in all folds |

**Problem in V10:**
- `log_ret_5m` was Z-SCORED (normalized)
- A value of 1.0 = 1 std = 0.11% real return, but was interpreted as 100%
- Result: Portfolio lost 99.89% in 100 steps

**Solution in V11:**
- Save RAW returns in separate column `_raw_ret_5m`
- Normalize features for the model
- Use raw returns for portfolio calculation

## Project Structure

```
v11/
├── README.md                 # This file
├── run.py                    # Main execution script
├── requirements.txt          # Python dependencies
│
├── config/
│   ├── __init__.py
│   └── settings.py           # All configuration parameters
│
├── src/
│   ├── __init__.py
│   ├── environment.py        # TradingEnvV11 (Gymnasium environment)
│   ├── backtest.py           # BacktestReporter (40+ metrics)
│   ├── callbacks.py          # Training callbacks
│   ├── utils.py              # Data preprocessing utilities
│   └── logger.py             # Session logging
│
├── notebooks/
│   └── USDCOP_RL_V11.ipynb   # Interactive notebook version
│
├── logs/                     # Session logs
├── models/                   # Saved PPO models
└── outputs/                  # Visualizations and results
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
# Run all 5 walk-forward folds
python run.py

# Run a single fold
python run.py --fold 0

# Run specific folds
python run.py --folds 0 1 2

# Quick test (50k steps per fold)
python run.py --quick
```

### 3. Use as Library

```python
from config import CONFIG, FEATURES_FOR_MODEL, RAW_RETURN_COL
from src import (
    TradingEnvV11,
    BacktestReporter,
    calculate_norm_stats,
    normalize_df_v11
)

# Load and prepare data
df = pd.read_csv("your_data.csv")

# Calculate normalization stats from training data only
norm_stats = calculate_norm_stats(df_train, FEATURES_FOR_MODEL)

# Normalize with raw returns preserved (V11 fix!)
df_normalized = normalize_df_v11(df, norm_stats, FEATURES_FOR_MODEL)

# Create environment
env = TradingEnvV11(
    df=df_normalized,
    features=FEATURES_FOR_MODEL,
    episode_length=288,
    initial_balance=10000,
    raw_ret_col=RAW_RETURN_COL  # Uses raw returns!
)
```

## Features

### V11 Specific
- RAW returns for portfolio calculation (critical fix)
- Normalized features for model input
- Log-space portfolio tracking (numerically stable)

### Inherited from Previous Versions
- Professional backtest reports (40+ metrics)
- Walk-forward validation (5 folds)
- Entropy scheduling during training
- Position distribution monitoring

## Configuration

All parameters are in `config/settings.py`:

```python
# Features for model input
FEATURES_FOR_MODEL = [
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
    'rsi_9', 'atr_pct', 'adx_14', 'bb_position',
    ...
]

# Trading parameters
CONFIG = {
    'initial_balance': 10_000,
    'episode_length': 288,      # 1 day
    'timesteps_per_fold': 300_000,
}

# PPO hyperparameters
PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    ...
}
```

## Walk-Forward Folds

| Fold | Train Period | Test Period |
|------|-------------|-------------|
| 0 | 2020-01 to 2021-06 | 2021-07 to 2021-12 |
| 1 | 2020-04 to 2022-03 | 2022-04 to 2022-10 |
| 2 | 2020-07 to 2022-10 | 2022-11 to 2023-05 |
| 3 | 2021-07 to 2023-06 | 2023-07 to 2023-12 |
| 4 | 2022-01 to 2024-06 | 2024-07 to 2025-05 |

## Metrics

The BacktestReporter calculates:

**Portfolio Metrics:**
- Total return, Sharpe ratio, Max drawdown

**Trade Metrics:**
- Number of trades, Win rate, Profit factor
- Average win/loss, Average trade duration

**Position Metrics:**
- Mean position, Time long/short/neutral

**Walk-Forward:**
- WFE (Walk-Forward Efficiency) = Test Sharpe / Train Sharpe

## Version History

| Version | Issue | Status |
|---------|-------|--------|
| V8 | normalize_df col variable bug | Fixed |
| V8 | Portfolio formula (+=) | Fixed with exp() |
| V8 | Regime on normalized data | Fixed (use RAW) |
| V9 | exp() overflow | Fixed (log-space) |
| V10 | Normalized returns for portfolio | **Fixed in V11** |
| V11 | All fixes applied | Current |

## Data Requirements

Expected CSV columns:
- `timestamp`: Datetime index
- `log_ret_5m`: 5-minute log returns
- `log_ret_1h`, `log_ret_4h`: Hourly returns
- `rsi_9`, `atr_pct`, `adx_14`, `bb_position`: Technical indicators
- `dxy_z`, `vix_z`, `embi_z`: Macro indicators
- `hour_sin`, `hour_cos`: Time features

## License

Internal use only.
