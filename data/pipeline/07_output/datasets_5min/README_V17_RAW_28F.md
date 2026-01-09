# V17 RAW Multi-Frequency Dataset - Exactly 28 Features WITHOUT Z-Score

**Generated:** 2025-12-18 10:42:13
**Script:** `04_build_v17_raw.py`
**Output:** `RL_DS6_RAW_28F.csv`

## CRITICAL CHANGE: RAW Features, NOT Z-Scored

This version outputs **RAW feature values WITHOUT applying z-score normalization**.
Z-score will be computed **PER FOLD** during training to prevent data leakage across
train/validation/test splits.

### Why Per-Fold Normalization?

**Problem with Global Z-Score:**
```python
# BAD: Leaks future information into training
z_score_global = (x - x.mean()) / x.std()  # Uses ALL data including test set
```

**Solution: Per-Fold Z-Score:**
```python
# GOOD: Only uses training fold statistics
train_mean, train_std = X_train.mean(), X_train.std()
X_train_z = (X_train - train_mean) / train_std
X_val_z = (X_val - train_mean) / train_std      # Uses TRAIN stats, not val stats
X_test_z = (X_test - train_mean) / train_std    # Uses TRAIN stats, not test stats
```

## Philosophy

Different market data updates at different frequencies. Respecting these natural
update cycles is CRITICAL for preventing look-ahead bias and ensuring training-production parity.

## Feature Specification (26 market + 2 state = 28 total)

### 1. 5-Minute RAW Features (14) - Change Every Bar

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 0 | `log_ret_5m` | 5-min log return | NEED z-score per fold |
| 1 | `log_ret_15m` | 15-min log return | NEED z-score per fold |
| 2 | `log_ret_30m` | 30-min log return | NEED z-score per fold |
| 3 | `momentum_5m` | close/close[-1] - 1 | NEED z-score per fold |
| 4 | `rsi_9` | RSI 0-100, RAW | Optional: (rsi-50)/50 |
| 5 | `atr_pct` | ATR as % of price, RAW | NEED z-score per fold |
| 6 | `bb_position` | Bollinger position [-1, +1] | Already normalized |
| 7 | `adx_14` | ADX 0-100, RAW | Optional: adx/50-1 |
| 8 | `ema_cross` | +1 if EMA9>EMA21, else -1 | Already normalized |
| 9 | `high_low_range` | (high-low)/close, RAW | NEED z-score per fold |
| 10 | `session_progress` | 0 at 8:00 AM, 1 at 12:55 PM | Already normalized |
| 11 | `hour_sin` | sin(2*pi*session_progress) | Already normalized |
| 12 | `hour_cos` | cos(2*pi*session_progress) | Already normalized |
| 13 | `is_first_hour` | 1.0 if hour==8, else 0.0 | Already normalized |

### 2. Hourly RAW Features (3) - Change Every Hour, USE LAG 1H

| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 14 | `log_ret_1h` | 1-hour log return, **LAG 1H** | NEED z-score per fold |
| 15 | `log_ret_4h` | 4-hour log return, **LAG 1H** | NEED z-score per fold |
| 16 | `volatility_1h` | Intra-hour volatility, **LAG 1H** | NEED z-score per fold |

**LAG Implementation:** Bar at 9:05 uses hourly data from 8:00-8:55

### 3. Daily RAW Features (9) - Change Once Per Day, USE LAG 1D

#### USD/COP Daily (3)
| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 17 | `usdcop_ret_1d` | USD/COP 1-day return, **LAG 1D** | NEED z-score per fold |
| 18 | `usdcop_ret_5d` | USD/COP 5-day return, **LAG 1D** | NEED z-score per fold |
| 19 | `usdcop_volatility` | 20-day rolling vol, **LAG 1D** | NEED z-score per fold |

#### Macro Essentials RAW (6)
| # | Feature | Description | Normalization |
|---|---------|-------------|---------------|
| 20 | `vix` | VIX RAW, **LAG 1D** | NEED z-score per fold |
| 21 | `embi` | EMBI Colombia RAW, **LAG 1D** | NEED z-score per fold |
| 22 | `dxy` | Dollar Index RAW, **LAG 1D** | NEED z-score per fold |
| 23 | `brent` | Brent oil RAW, **LAG 1D** | NEED z-score per fold |
| 24 | `rate_spread` | COL10Y - UST10Y, **LAG 1D** | NEED z-score per fold |
| 25 | `usdmxn_ret_1d` | USD/MXN 1-day return, **LAG 1D** | NEED z-score per fold |

**LAG Implementation:** Monday uses Friday's data (yesterday's close)

### 4. State Features (2) - Added by RL Environment

| # | Feature | Description |
|---|---------|-------------|
| 26 | `position` | Current position [-1, +1] |
| 27 | `unrealized_pnl_norm` | Unrealized PnL normalized |

## Normalization Strategy

### Features that DON'T need normalization (already bounded):
- `bb_position`: -1 to +1
- `ema_cross`: -1 or +1
- `session_progress`: 0 to 1
- `hour_sin`: -1 to +1
- `hour_cos`: -1 to +1
- `is_first_hour`: 0 or 1

### Features that NEED z-score normalization (will be done per fold):
- All returns: `log_ret_5m`, `log_ret_15m`, `log_ret_30m`, `log_ret_1h`, `log_ret_4h`
- Momentum: `momentum_5m`
- Volatility: `volatility_1h`, `usdcop_volatility`
- Range: `high_low_range`, `atr_pct`
- All macro: `vix`, `embi`, `dxy`, `brent`, `rate_spread`, `usdmxn_ret_1d`

### Features with optional alternative normalization:
- `rsi_9`: Can use (rsi-50)/50 to map to [-1, +1]
- `adx_14`: Can use adx/50-1 to center at 0

## Dataset Statistics

- **Total Bars:** 84,671
- **Date Range:** 2020-03-02 13:00:00 to 2025-12-05 14:10:00
- **Trading Days:** 1499
- **Size:** 34.12 MB
- **Market Features:** 26 (in dataset, RAW values)
- **State Features:** 2 (added by environment)
- **Total Features:** 28

## Technical Implementation

### Per-Fold Z-Score Normalization (Apply During Training)

```python
from sklearn.model_selection import TimeSeriesSplit

# Features that need z-score
features_need_zscore = [
    'log_ret_5m', 'log_ret_15m', 'log_ret_30m',
    'momentum_5m', 'atr_pct', 'high_low_range',
    'log_ret_1h', 'log_ret_4h', 'volatility_1h',
    'usdcop_ret_1d', 'usdcop_ret_5d', 'usdcop_volatility',
    'vix', 'embi', 'dxy', 'brent', 'rate_spread', 'usdmxn_ret_1d'
]

# Features that are already normalized
features_no_zscore = [
    'rsi_9', 'bb_position', 'adx_14', 'ema_cross',
    'session_progress', 'hour_sin', 'hour_cos', 'is_first_hour'
]

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]

    # Calculate z-score statistics on TRAINING fold ONLY
    train_mean = X_train_fold[features_need_zscore].mean()
    train_std = X_train_fold[features_need_zscore].std()

    # Apply z-score using TRAINING statistics
    X_train_fold[features_need_zscore] = (
        (X_train_fold[features_need_zscore] - train_mean) / (train_std + 1e-8)
    ).clip(-4, 4)

    X_val_fold[features_need_zscore] = (
        (X_val_fold[features_need_zscore] - train_mean) / (train_std + 1e-8)
    ).clip(-4, 4)

    # Train model on fold
    model.fit(X_train_fold, y_train_fold)
    val_score = model.evaluate(X_val_fold, y_val_fold)
```

### Hourly LAG (1 hour)
```python
# Resample to hourly
df_1h = df.resample('1H').agg({'close': 'last'})

# Calculate hourly features
df_1h['log_ret_1h'] = np.log(df_1h['close'] / df_1h['close'].shift(1))

# SHIFT to create LAG
df_1h_lag = df_1h[hourly_features].shift(1)

# Merge back to 5-min
df['hour_floor'] = df.index.floor('H')
df = df.merge(df_1h_lag, left_on='hour_floor', right_index=True)
```

### Daily LAG (1 day)
```python
# Aggregate to daily
df_daily = df.resample('D').agg({'close': 'last'})

# Calculate daily features
df_daily['usdcop_ret_1d'] = df_daily['close'].pct_change(1)

# SHIFT to create LAG
df_daily_lag = df_daily[daily_features].shift(1)

# Merge back to 5-min
df['date'] = df.index.date
df = df.merge(df_daily_lag, on='date', how='left')
```

## Market Hours

- **Trading Days:** Monday-Friday
- **Trading Hours:** 8:00 AM - 12:55 PM Colombia Time (UTC-5)
- **UTC Hours:** 13:00 - 17:55
- **Bars per Day:** ~60 (5-hour session at 5-min frequency)

## Validation Results

### Feature Statistics (RAW values)

 index           feature      frequency_type          mean        std       min        max
     0        log_ret_5m            5MIN RAW  9.042128e-07   0.001134 -0.049620   0.050000
     1       log_ret_15m            5MIN RAW  2.541190e-06   0.001876 -0.050000   0.050000
     2       log_ret_30m            5MIN RAW  5.057410e-06   0.002642 -0.050000   0.050000
     3       momentum_5m            5MIN RAW  1.532270e-06   0.001134 -0.048409   0.050000
     4             rsi_9            5MIN RAW  4.855248e+01  23.916683  0.000000 100.000000
     5           atr_pct            5MIN RAW  6.081763e-02   0.045250  0.000000   0.776248
     6       bb_position            5MIN RAW -5.039110e-03   0.570597 -1.000000   1.000000
     7            adx_14            5MIN RAW  3.229754e+01  17.046334  0.000000 100.000000
     8         ema_cross            5MIN RAW -1.763296e-02   0.999850 -1.000000   1.000000
     9    high_low_range            5MIN RAW  4.562653e-05   0.000361  0.000000   0.040915
    10  session_progress            5MIN RAW  4.917410e-01   0.288248  0.000000   0.983333
    11          hour_sin            5MIN RAW  9.459097e-04   0.707726 -1.000000   1.000000
    12          hour_cos            5MIN RAW -1.530979e-03   0.706493 -1.000000   1.000000
    13     is_first_hour            5MIN RAW  1.994425e-01   0.399584  0.000000   1.000000
    14        log_ret_1h HOURLY RAW (LAG 1H)  3.126921e-05   0.003481 -0.049763   0.050000
    15        log_ret_4h HOURLY RAW (LAG 1H)  9.903049e-05   0.006371 -0.049763   0.057567
    16     volatility_1h HOURLY RAW (LAG 1H)  8.487867e-04   0.000840  0.000000   0.020338
    17     usdcop_ret_1d  DAILY RAW (LAG 1D)  4.531189e-05   0.007838 -0.043880   0.062422
    18     usdcop_ret_5d  DAILY RAW (LAG 1D)  3.996023e-04   0.016567 -0.060371   0.124103
    19 usdcop_volatility  DAILY RAW (LAG 1D)  6.592169e-03   0.002477  0.001808   0.022433
    20               vix  DAILY RAW (LAG 1D)  2.117045e+01   7.784114  0.000000  82.690000
    21              embi  DAILY RAW (LAG 1D)  3.003528e+02 102.459783  0.000000 522.000000
    22               dxy  DAILY RAW (LAG 1D)  1.003478e+02   5.588427 89.440000 114.110000
    23             brent  DAILY RAW (LAG 1D)  7.458062e+01  19.136592 19.330000 127.980000
    24       rate_spread  DAILY RAW (LAG 1D)  6.607342e+00   1.537197  0.000000  10.970000
    25     usdmxn_ret_1d  DAILY RAW (LAG 1D) -1.926499e-05   0.008005 -0.041789   0.043379

### Data Quality

- **NaNs:** 0 (after warmup removal)
- **Infinities:** 0
- **Zero-variance:** None detected

## Why RAW Features + Per-Fold Z-Score?

### Comparison: Global vs Per-Fold Normalization

| Aspect | Global Z-Score (V17) | Per-Fold Z-Score (V17 RAW) |
|--------|---------------------|---------------------------|
| Data Leakage | YES (uses test stats) | NO (only train stats) |
| Training Bias | Optimistic | Realistic |
| Validation Accuracy | Overestimated | True performance |
| Production Parity | Poor | Excellent |
| Best Practice | Incorrect | Correct |

### Example of Data Leakage

```python
# Dataset: 2020-2024 (5 years)
# Train: 2020-2022, Val: 2023, Test: 2024

# WRONG: Global z-score (V17 style)
global_mean = df['vix'].mean()  # Uses data from 2024!
global_std = df['vix'].std()    # Uses data from 2024!
z_score = (df['vix'] - global_mean) / global_std

# Model learns that VIX=30 is "normal" because it saw 2024 data
# But in 2022, VIX=30 was actually extremely high!
# Result: Overfitted to future regimes

# RIGHT: Per-fold z-score (V17 RAW style)
train_mean = df.loc['2020':'2022', 'vix'].mean()  # Only 2020-2022
train_std = df.loc['2020':'2022', 'vix'].std()
train_z = (df.loc['2020':'2022', 'vix'] - train_mean) / train_std
val_z = (df.loc['2023', 'vix'] - train_mean) / train_std  # Uses TRAIN stats

# Model learns that VIX=30 was high in 2022
# Correctly applies this knowledge to 2023
# Result: True out-of-sample performance
```

## Usage

### Loading Dataset

```python
import pandas as pd
import numpy as np

# Load RAW dataset
df = pd.read_csv('RL_DS6_RAW_28F.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define feature groups
features_5min = ['log_ret_5m', 'log_ret_15m', 'log_ret_30m', 'momentum_5m', 'rsi_9', 'atr_pct', 'bb_position', 'adx_14', 'ema_cross', 'high_low_range', 'session_progress', 'hour_sin', 'hour_cos', 'is_first_hour']
features_hourly = ['log_ret_1h', 'log_ret_4h', 'volatility_1h']
features_daily = ['usdcop_ret_1d', 'usdcop_ret_5d', 'usdcop_volatility', 'vix', 'embi', 'dxy', 'brent', 'rate_spread', 'usdmxn_ret_1d']

# Features that need z-score
features_need_zscore = [
    'log_ret_5m', 'log_ret_15m', 'log_ret_30m',
    'momentum_5m', 'atr_pct', 'high_low_range',
    'log_ret_1h', 'log_ret_4h', 'volatility_1h',
    'usdcop_ret_1d', 'usdcop_ret_5d', 'usdcop_volatility',
    'vix', 'embi', 'dxy', 'brent', 'rate_spread', 'usdmxn_ret_1d'
]

# Features already normalized (don't touch)
features_no_zscore = [
    'rsi_9', 'bb_position', 'adx_14', 'ema_cross',
    'session_progress', 'hour_sin', 'hour_cos', 'is_first_hour'
]
```

### RL Training with Per-Fold Z-Score

```python
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines3 import PPO

# Time series split (e.g., 5 folds)
tscv = TimeSeriesSplit(n_splits=5)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
    print(f"Fold {fold+1}/5")

    # Split data
    train_data = df.iloc[train_idx].copy()
    val_data = df.iloc[val_idx].copy()

    # Calculate z-score parameters on TRAIN only
    train_mean = train_data[features_need_zscore].mean()
    train_std = train_data[features_need_zscore].std()

    # Apply z-score
    train_data[features_need_zscore] = (
        (train_data[features_need_zscore] - train_mean) / (train_std + 1e-8)
    ).clip(-4, 4)

    val_data[features_need_zscore] = (
        (val_data[features_need_zscore] - train_mean) / (train_std + 1e-8)
    ).clip(-4, 4)

    # Create RL environments
    train_env = TradingEnv(train_data)
    val_env = TradingEnv(val_data)

    # Train PPO
    model = PPO("MlpPolicy", train_env, verbose=0)
    model.learn(total_timesteps=100000)

    # Evaluate on validation
    val_returns, val_sharpe = evaluate(model, val_env)
    fold_results.append({'fold': fold, 'sharpe': val_sharpe})

# Average performance across folds
avg_sharpe = np.mean([r['sharpe'] for r in fold_results])
print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
```

## Next Steps

1. **Train RL Agent:** Use PPO/A2C with per-fold z-score normalization
2. **Validate Performance:** Compare per-fold vs global z-score
3. **Backtest:** Use walk-forward analysis with proper normalization
4. **Deploy:** If Sharpe > 0.6 and max drawdown < 12%

## References

- **Source Script:** `04_build_v17_raw.py`
- **Statistics:** `STATS_V17_RAW_28F.csv`
- **Previous Version:** V17 (z-scored globally, data leakage issue)

## Key Differences vs V17

| Feature | V17 (Global Z-Score) | V17 RAW (Per-Fold Z-Score) |
|---------|---------------------|---------------------------|
| Normalization | Pre-computed globally | Computed per fold |
| Data Leakage | YES | NO |
| File Size | Same | Same |
| Feature Count | 26 market + 2 state | 26 market + 2 state |
| Training Complexity | Simpler | Slightly more complex |
| Production Parity | Poor | Excellent |
| Recommended | ❌ | ✅ |

---

*Generated by V17 RAW Multi-Frequency Dataset Builder*
*Author: Claude Code*
*Date: 2025-12-18 10:42:13*
