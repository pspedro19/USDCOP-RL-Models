# V17 RAW Dataset - Quick Start Guide

## What You Have

**File:** `RL_DS6_RAW_28F.csv` (20.09 MB, 50,947 bars)

**Date Range:** 2020-03-02 to 2025-12-05 (1,459 trading days)

**Features:** 26 market features + 2 state features (added by RL environment) = 28 total

## Feature Specification

### 5-Minute RAW Features (14)
```python
FEATURES_5MIN_RAW = [
    'log_ret_5m',         # RAW 5min log return (NEED z-score per fold)
    'log_ret_15m',        # RAW 15min log return (NEED z-score per fold)
    'log_ret_30m',        # RAW 30min log return (NEED z-score per fold)
    'momentum_5m',        # close/close[-1] - 1 (NEED z-score per fold)
    'rsi_9',              # RAW RSI 0-100 (Optional: (rsi-50)/50)
    'atr_pct',            # RAW ATR % (NEED z-score per fold)
    'bb_position',        # Already -1 to +1 (NO normalization)
    'adx_14',             # RAW ADX 0-100 (Optional: adx/50-1)
    'ema_cross',          # Already -1 or +1 (NO normalization)
    'high_low_range',     # RAW (high-low)/close (NEED z-score per fold)
    'session_progress',   # Already 0-1 (NO normalization)
    'hour_sin',           # Already -1 to +1 (NO normalization)
    'hour_cos',           # Already -1 to +1 (NO normalization)
    'is_first_hour',      # Already 0 or 1 (NO normalization)
]
```

### Hourly RAW Features - WITH LAG 1H (3)
```python
FEATURES_HOURLY_RAW = [
    'log_ret_1h',         # RAW 1h return, LAG 1H (NEED z-score per fold)
    'log_ret_4h',         # RAW 4h return, LAG 1H (NEED z-score per fold)
    'volatility_1h',      # RAW hourly vol, LAG 1H (NEED z-score per fold)
]
```

**LAG Implementation:** Bar at 9:05 uses hourly data from 8:00-8:55

### Daily RAW Features - WITH LAG 1D (9)
```python
FEATURES_DAILY_RAW = [
    # USD/COP daily (3)
    'usdcop_ret_1d',      # RAW 1-day return, LAG 1D (NEED z-score per fold)
    'usdcop_ret_5d',      # RAW 5-day return, LAG 1D (NEED z-score per fold)
    'usdcop_volatility',  # RAW 20-day vol, LAG 1D (NEED z-score per fold)

    # Macro RAW (6)
    'vix',                # RAW VIX, LAG 1D (NEED z-score per fold)
    'embi',               # RAW EMBI, LAG 1D (NEED z-score per fold)
    'dxy',                # RAW DXY, LAG 1D (NEED z-score per fold)
    'brent',              # RAW Brent, LAG 1D (NEED z-score per fold)
    'rate_spread',        # RAW COL10Y-UST10Y, LAG 1D (NEED z-score per fold)
    'usdmxn_ret_1d',      # RAW USD/MXN return, LAG 1D (NEED z-score per fold)
]
```

**LAG Implementation:** Monday uses Friday's data (yesterday's close)

### State Features (2) - Added by RL Environment
```python
FEATURES_STATE = [
    'position',           # Current position [-1, +1]
    'unrealized_pnl_norm',# Unrealized PnL normalized
]
```

## Normalization Strategy

### Features Already Bounded (6) - NO Normalization Needed
```python
features_no_zscore = [
    'bb_position',      # -1 to +1
    'ema_cross',        # -1 or +1
    'session_progress', # 0 to 1
    'hour_sin',         # -1 to +1
    'hour_cos',         # -1 to +1
    'is_first_hour',    # 0 or 1
]
```

### Features That NEED Z-Score Per Fold (20)
```python
features_need_zscore = [
    # Returns (6)
    'log_ret_5m', 'log_ret_15m', 'log_ret_30m',
    'log_ret_1h', 'log_ret_4h', 'momentum_5m',

    # Volatility (2)
    'volatility_1h', 'usdcop_volatility',

    # Range/ATR (2)
    'atr_pct', 'high_low_range',

    # USD/COP daily (2)
    'usdcop_ret_1d', 'usdcop_ret_5d',

    # Macro (6)
    'vix', 'embi', 'dxy', 'brent',
    'rate_spread', 'usdmxn_ret_1d',

    # Optional: Can normalize RSI and ADX differently
    # 'rsi_9',    # Can use (rsi-50)/50 instead of z-score
    # 'adx_14',   # Can use adx/50-1 instead of z-score
]
```

## How to Use: Per-Fold Z-Score Normalization

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Load RAW dataset
df = pd.read_csv('RL_DS6_RAW_28F.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define feature groups
features_need_zscore = [
    'log_ret_5m', 'log_ret_15m', 'log_ret_30m',
    'momentum_5m', 'atr_pct', 'high_low_range',
    'log_ret_1h', 'log_ret_4h', 'volatility_1h',
    'usdcop_ret_1d', 'usdcop_ret_5d', 'usdcop_volatility',
    'vix', 'embi', 'dxy', 'brent', 'rate_spread', 'usdmxn_ret_1d'
]

features_no_zscore = [
    'bb_position', 'ema_cross', 'session_progress',
    'hour_sin', 'hour_cos', 'is_first_hour'
]

# Optional: Alternative normalization for RSI/ADX
def normalize_rsi_adx(df):
    df['rsi_9_norm'] = (df['rsi_9'] - 50) / 50
    df['adx_14_norm'] = df['adx_14'] / 50 - 1
    return df

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
    print(f"Processing Fold {fold + 1}/5...")

    # Split data
    train_data = df.iloc[train_idx].copy()
    val_data = df.iloc[val_idx].copy()

    # Calculate z-score statistics on TRAINING fold only
    train_mean = train_data[features_need_zscore].mean()
    train_std = train_data[features_need_zscore].std()

    # Apply z-score to training data
    train_data[features_need_zscore] = (
        (train_data[features_need_zscore] - train_mean) / (train_std + 1e-8)
    ).clip(-4, 4)

    # Apply z-score to validation data (using TRAIN statistics!)
    val_data[features_need_zscore] = (
        (val_data[features_need_zscore] - train_mean) / (train_std + 1e-8)
    ).clip(-4, 4)

    # Optional: Normalize RSI and ADX
    train_data = normalize_rsi_adx(train_data)
    val_data = normalize_rsi_adx(val_data)

    # Create RL environments
    train_env = TradingEnv(train_data)
    val_env = TradingEnv(val_data)

    # Train RL agent
    model = PPO("MlpPolicy", train_env, verbose=0)
    model.learn(total_timesteps=100000)

    # Evaluate on validation
    val_returns, val_sharpe = evaluate(model, val_env)

    fold_results.append({
        'fold': fold,
        'sharpe': val_sharpe,
        'train_size': len(train_data),
        'val_size': len(val_data)
    })

    print(f"  Fold {fold + 1} - Sharpe: {val_sharpe:.3f}")

# Average performance across folds
avg_sharpe = np.mean([r['sharpe'] for r in fold_results])
print(f"\nAverage Sharpe Ratio: {avg_sharpe:.3f}")
```

## CRITICAL: Why Per-Fold Normalization?

### WRONG: Global Z-Score (Data Leakage)
```python
# BAD: Uses statistics from ALL data including test set
global_mean = df['vix'].mean()  # Includes future data!
global_std = df['vix'].std()
df['vix_z'] = (df['vix'] - global_mean) / global_std

# Model sees future patterns → overfit → poor production
```

### CORRECT: Per-Fold Z-Score (No Leakage)
```python
# GOOD: Only uses training fold statistics
train_mean = train_data['vix'].mean()  # Only training data
train_std = train_data['vix'].std()

train_data['vix_z'] = (train_data['vix'] - train_mean) / train_std
val_data['vix_z'] = (val_data['vix'] - train_mean) / train_std  # Uses TRAIN stats

# Model learns true patterns → generalizes → good production
```

## Feature Statistics (RAW values)

| Feature | Mean | Std | Min | Max | Need Z-Score? |
|---------|------|-----|-----|-----|---------------|
| log_ret_5m | 0.0000 | 0.0008 | -0.02 | 0.02 | YES |
| rsi_9 | 48.24 | 23.01 | 0.00 | 100.00 | Optional |
| adx_14 | 30.44 | 16.67 | 0.00 | 100.00 | Optional |
| bb_position | -0.01 | 0.56 | -1.00 | 1.00 | NO |
| vix | 21.16 | 7.77 | 0.00 | 82.69 | YES |
| embi | 300.01 | 102.63 | 0.00 | 522.00 | YES |
| dxy | 100.36 | 5.57 | 89.44 | 114.11 | YES |
| brent | 74.52 | 19.14 | 19.33 | 127.98 | YES |

## Files Generated

1. **Dataset**: `RL_DS6_RAW_28F.csv` (20.09 MB)
   - 50,947 bars
   - 26 market features + timestamp
   - RAW values (not normalized)

2. **Statistics**: `STATS_V17_RAW_28F.csv`
   - Mean, std, min, max for each feature
   - Null counts and percentages

3. **README**: `README_V17_RAW_28F.md`
   - Complete documentation
   - Feature descriptions
   - Implementation details

4. **Comparison**: `DATASET_COMPARISON_V17_vs_V17_RAW.md`
   - V17 vs V17 RAW differences
   - Data leakage explanation
   - Migration guide

## Next Steps

1. Load `RL_DS6_RAW_28F.csv`
2. Implement per-fold z-score normalization
3. Train RL agent with TimeSeriesSplit cross-validation
4. Validate that performance is realistic (not overfit)
5. Deploy if Sharpe > 0.6 and max drawdown < 12%

## Summary

- **Use This Dataset**: `RL_DS6_RAW_28F.csv`
- **Apply Z-Score Per Fold**: Not globally
- **Features Already Bounded**: No normalization needed
- **Features That Need Z-Score**: 20 features
- **Expected Benefit**: Realistic validation, better production performance

---

**Generated:** 2025-12-18
**Author:** Claude Code
**Location:** `C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\data\pipeline\07_output\datasets_5min\RL_DS6_RAW_28F.csv`
