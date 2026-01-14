# Dataset Comparison: V17 (Global Z-Score) vs V17 RAW (Per-Fold Z-Score)

## Quick Reference

| Aspect | V17 (Global Z-Score) | V17 RAW (Per-Fold Z-Score) |
|--------|---------------------|---------------------------|
| **File** | `RL_DS5_MULTIFREQ_28F.csv` | `RL_DS6_RAW_28F.csv` |
| **Script** | `03_build_v17_multifreq.py` | `04_build_v17_raw.py` |
| **Normalization** | Pre-computed globally | Computed per fold |
| **Data Leakage** | ❌ YES | ✅ NO |
| **Production Parity** | ❌ Poor | ✅ Excellent |
| **Recommended** | ❌ Deprecated | ✅ Use This |

## Feature Comparison

### V17 (Global Z-Score) - DEPRECATED
```python
# Features with global z-score (uses ALL data including test set)
'log_ret_5m_z'      # Z-scored globally
'rsi_9_norm'        # Normalized: (rsi-50)/50
'adx_14_norm'       # Normalized: adx/50-1
'vix_z'             # Z-scored globally
'embi_z'            # Z-scored globally
# ... etc
```

### V17 RAW (Per-Fold Z-Score) - RECOMMENDED
```python
# Features with RAW values (z-score applied per fold during training)
'log_ret_5m'        # RAW (will be z-scored per fold)
'rsi_9'             # RAW 0-100 (will be normalized per fold)
'adx_14'            # RAW 0-100 (will be normalized per fold)
'vix'               # RAW (will be z-scored per fold)
'embi'              # RAW (will be z-scored per fold)
# ... etc
```

## Data Leakage Explanation

### Problem with V17 (Global Z-Score)

```python
# V17 computes z-score using ALL data (2020-2025)
global_mean = df['vix'].mean()  # Includes 2024 and 2025!
global_std = df['vix'].std()

# When training on 2020-2022:
train_data['vix_z'] = (vix - global_mean) / global_std
# ^ This uses statistics from the future (2023-2025)!

# Result: Model learns patterns that won't exist in production
```

### Solution: V17 RAW (Per-Fold Z-Score)

```python
# V17 RAW stores RAW values, z-score computed PER FOLD
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    # Calculate statistics ONLY on training fold
    train_mean = X.iloc[train_idx]['vix'].mean()
    train_std = X.iloc[train_idx]['vix'].std()
    
    # Apply to both train and validation
    X.iloc[train_idx]['vix_z'] = (X.iloc[train_idx]['vix'] - train_mean) / train_std
    X.iloc[val_idx]['vix_z'] = (X.iloc[val_idx]['vix'] - train_mean) / train_std
    # ^ Uses ONLY training statistics!

# Result: No data leakage, realistic validation metrics
```

## Example: VIX Normalization

### V17 (Global Z-Score) - Shows the Problem

```
Dataset: 2020-2025 (5 years)
VIX range: 10-80
Global mean: 21.16
Global std: 7.77

Training period: 2020-2022
  VIX = 30 → z-score = (30 - 21.16) / 7.77 = +1.14 (slightly high)
  
Validation period: 2023
  VIX = 30 → z-score = (30 - 21.16) / 7.77 = +1.14 (same)

PROBLEM: In reality, VIX=30 in 2020 was EXTREMELY high (COVID),
but global mean makes it look "normal" because 2023-2025 had higher VIX.
Model learns wrong patterns!
```

### V17 RAW (Per-Fold Z-Score) - Correct Approach

```
Training period: 2020-2022
  Train mean: 18.5 (COVID era, high volatility)
  Train std: 8.2
  VIX = 30 → z-score = (30 - 18.5) / 8.2 = +1.40 (appropriately high)

Validation period: 2023 (uses TRAIN statistics)
  VIX = 30 → z-score = (30 - 18.5) / 8.2 = +1.40 (same)

CORRECT: Model correctly learns that VIX=30 is high relative to
training period, matches production behavior.
```

## Feature Categories

### Already Normalized (No Change Needed)
- `bb_position`: -1 to +1
- `ema_cross`: -1 or +1
- `session_progress`: 0 to 1
- `hour_sin`: -1 to +1
- `hour_cos`: -1 to +1
- `is_first_hour`: 0 or 1

### Need Z-Score Per Fold (20 features)
```python
features_need_zscore = [
    # Returns
    'log_ret_5m', 'log_ret_15m', 'log_ret_30m',
    'log_ret_1h', 'log_ret_4h',
    'momentum_5m',
    
    # Volatility
    'volatility_1h', 'usdcop_volatility',
    
    # Range/ATR
    'atr_pct', 'high_low_range',
    
    # USD/COP daily
    'usdcop_ret_1d', 'usdcop_ret_5d',
    
    # Macro
    'vix', 'embi', 'dxy', 'brent',
    'rate_spread', 'usdmxn_ret_1d',
]
```

### Optional Alternative Normalization
```python
# RSI: Can normalize to [-1, +1]
rsi_norm = (rsi_9 - 50) / 50

# ADX: Can normalize to center at 0
adx_norm = adx_14 / 50 - 1
```

## Migration Guide

### If You Were Using V17

```python
# OLD (V17 - Global Z-Score)
df = pd.read_csv('RL_DS5_MULTIFREQ_28F.csv')
# Features already z-scored, just use directly
X = df[['log_ret_5m_z', 'rsi_9_norm', 'vix_z', ...]]

# PROBLEM: Data leakage! ❌
```

### Switch to V17 RAW

```python
# NEW (V17 RAW - Per-Fold Z-Score)
df = pd.read_csv('RL_DS6_RAW_28F.csv')

# Define features that need z-score
features_need_zscore = ['log_ret_5m', 'vix', 'embi', ...]
features_no_zscore = ['bb_position', 'ema_cross', ...]

# Apply z-score PER FOLD
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
    # Calculate on training fold only
    train_mean = df.iloc[train_idx][features_need_zscore].mean()
    train_std = df.iloc[train_idx][features_need_zscore].std()
    
    # Apply to train
    df_train = df.iloc[train_idx].copy()
    df_train[features_need_zscore] = (
        (df_train[features_need_zscore] - train_mean) / (train_std + 1e-8)
    ).clip(-4, 4)
    
    # Apply to validation (using TRAIN statistics!)
    df_val = df.iloc[val_idx].copy()
    df_val[features_need_zscore] = (
        (df_val[features_need_zscore] - train_mean) / (train_std + 1e-8)
    ).clip(-4, 4)
    
    # Train model
    model.fit(df_train, y_train)
    val_score = model.evaluate(df_val, y_val)

# SOLUTION: No data leakage! ✅
```

## Performance Comparison

### Expected Results

| Metric | V17 (Global) | V17 RAW (Per-Fold) |
|--------|-------------|-------------------|
| Validation Sharpe | 0.8 (optimistic) | 0.6 (realistic) |
| Test Sharpe | 0.4 (poor) | 0.6 (good) |
| Production Sharpe | 0.3 (very poor) | 0.6 (matches backtest) |
| Data Leakage | YES | NO |
| Reliability | Low | High |

### Why V17 RAW is Better

1. **No Data Leakage**: Only uses training data for normalization
2. **Realistic Metrics**: Validation performance matches production
3. **Better Generalization**: Model learns true patterns, not artifacts
4. **Industry Standard**: This is the correct way to do cross-validation

## Bottom Line

- **Use V17 RAW (`RL_DS6_RAW_28F.csv`)** for all new training
- **Deprecate V17** (`RL_DS5_MULTIFREQ_28F.csv`) - has data leakage
- **Apply z-score PER FOLD** during training, not globally
- **Expect lower validation scores** but better production performance

---

**Generated:** 2025-12-18
**Author:** Claude Code
