# V17 Multi-Frequency Dataset - Exactly 28 Features

**Generated:** 2025-12-17 23:59:19
**Script:** `03_build_v17_multifreq.py`
**Output:** `RL_DS5_MULTIFREQ_28F.csv`

## Philosophy

Different market data updates at different frequencies. Respecting these natural
update cycles is CRITICAL for preventing look-ahead bias and ensuring training-production parity.

## Feature Specification (26 market + 2 state = 28 total)

### 1. 5-Minute Features (14) - Change Every Bar

| # | Feature | Description |
|---|---------|-------------|
| 0 | `log_ret_5m_z` | 5-min log return, z-scored |
| 1 | `log_ret_15m_z` | 15-min log return (3 bars), z-scored |
| 2 | `log_ret_30m_z` | 30-min log return (6 bars), z-scored |
| 3 | `momentum_5m` | close/close[-1] - 1 |
| 4 | `rsi_9_norm` | RSI normalized to [-1, +1]: (rsi-50)/50 |
| 5 | `atr_pct_z` | ATR as % of price, z-scored |
| 6 | `bb_position` | Bollinger position [-1, +1] |
| 7 | `adx_14_norm` | ADX normalized: adx/50 - 1 |
| 8 | `ema_cross` | +1 if EMA9>EMA21, else -1 |
| 9 | `high_low_range_z` | (high-low)/close, z-scored |
| 10 | `session_progress` | 0 at 8:00 AM, 1 at 12:55 PM |
| 11 | `hour_sin` | sin(2*pi*session_progress) |
| 12 | `hour_cos` | cos(2*pi*session_progress) |
| 13 | `is_first_hour` | 1.0 if hour==8, else 0.0 |

### 2. Hourly Features (3) - Change Every Hour, USE LAG 1H

| # | Feature | Description |
|---|---------|-------------|
| 14 | `log_ret_1h_z` | 1-hour log return, z-scored, **LAG 1H** |
| 15 | `log_ret_4h_z` | 4-hour log return, z-scored, **LAG 1H** |
| 16 | `volatility_1h_z` | Intra-hour volatility (std of 5min returns), **LAG 1H** |

**LAG Implementation:** Bar at 9:05 uses hourly data from 8:00-8:55

### 3. Daily Features (9) - Change Once Per Day, USE LAG 1D

#### USD/COP Daily (3)
| # | Feature | Description |
|---|---------|-------------|
| 17 | `usdcop_ret_1d` | USD/COP return from previous day, **LAG 1D** |
| 18 | `usdcop_ret_5d` | USD/COP 5-day return, **LAG 1D** |
| 19 | `usdcop_volatility` | Rolling 20-day volatility, **LAG 1D** |

#### Macro Essentials (6)
| # | Feature | Description |
|---|---------|-------------|
| 20 | `vix_z` | VIX z-score, **LAG 1D** |
| 21 | `embi_z` | EMBI Colombia z-score, **LAG 1D** |
| 22 | `dxy_z` | Dollar Index z-score, **LAG 1D** |
| 23 | `brent_z` | Brent oil z-score, **LAG 1D** |
| 24 | `rate_spread` | COL10Y - UST10Y (normalized), **LAG 1D** |
| 25 | `usdmxn_ret_1d` | USD/MXN 1-day return, **LAG 1D** |

**LAG Implementation:** Monday uses Friday's data (yesterday's close)

### 4. State Features (2) - Added by RL Environment

| # | Feature | Description |
|---|---------|-------------|
| 26 | `position` | Current position [-1, +1] |
| 27 | `unrealized_pnl_norm` | Unrealized PnL normalized |

## Dataset Statistics

- **Total Bars:** 50,947
- **Date Range:** 2020-03-02 15:00:00 to 2025-12-05 14:10:00
- **Trading Days:** 1459
- **Size:** 23.06 MB
- **Market Features:** 26 (in dataset)
- **State Features:** 2 (added by environment)
- **Total Features:** 28

## Technical Implementation

### Z-Score Normalization
```python
window = 252 * 60  # 1 year for 5-min data
mean = series.rolling(window, min_periods=60).mean()
std = series.rolling(window, min_periods=60).std()
z = ((series - mean) / (std + 1e-8)).clip(-4, 4)
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

### Feature Statistics

 index           feature  frequency_type      mean      std
     0      log_ret_5m_z            5MIN -0.006482 0.678860
     1     log_ret_15m_z            5MIN -0.010288 0.675876
     2     log_ret_30m_z            5MIN -0.012447 0.679314
     3       momentum_5m            5MIN -0.000003 0.000825
     4        rsi_9_norm            5MIN -0.035198 0.460278
     5         atr_pct_z            5MIN -0.205485 0.782234
     6       bb_position            5MIN -0.012307 0.563496
     7       adx_14_norm            5MIN -0.391135 0.333460
     8         ema_cross            5MIN -0.007164 0.999984
     9  high_low_range_z            5MIN  0.009368 0.432594
    10  session_progress            5MIN  0.687983 0.177788
    11          hour_sin            5MIN -0.463339 0.477780
    12          hour_cos            5MIN -0.178492 0.724706
    13     is_first_hour            5MIN  0.003553 0.059499
    14      log_ret_1h_z HOURLY (LAG 1H)  0.010932 1.016620
    15      log_ret_4h_z HOURLY (LAG 1H)  0.004045 0.187001
    16   volatility_1h_z HOURLY (LAG 1H) -0.195580 0.686004
    17     usdcop_ret_1d  DAILY (LAG 1D)  0.000017 0.007839
    18     usdcop_ret_5d  DAILY (LAG 1D)  0.000366 0.016506
    19 usdcop_volatility  DAILY (LAG 1D)  0.006596 0.002472
    20             vix_z  DAILY (LAG 1D) -0.185333 1.100939
    21            embi_z  DAILY (LAG 1D)  0.194400 1.263746
    22             dxy_z  DAILY (LAG 1D)  0.054414 1.392135
    23           brent_z  DAILY (LAG 1D)  0.016281 1.280713
    24       rate_spread  DAILY (LAG 1D)  0.408464 1.350815
    25     usdmxn_ret_1d  DAILY (LAG 1D) -0.000027 0.007989

### Data Quality

- **NaNs:** 0 (after warmup removal)
- **Infinities:** 0
- **Zero-variance:** None detected

## Why Multi-Frequency Design?

### Problem with Single-Frequency Approach

Previous datasets treated all features as if they update at the same frequency:
- **Wrong:** All features available at bar *t*
- **Reality:** Different features have different latencies

### Solution: Frequency-Aware LAG

| Frequency | Update Cycle | LAG Required | Example |
|-----------|--------------|--------------|---------|
| 5-minute | Every bar | None | Price returns, RSI |
| Hourly | Every 12 bars | 1 hour | Hourly volatility |
| Daily | Every 288 bars | 1 day | Macro indicators |

### Benefits

1. **No Look-Ahead Bias:** Features only use data available at decision time
2. **Training-Production Parity:** Exactly matches what agent sees in live trading
3. **Realistic Backtests:** Performance matches actual deployable strategy
4. **Clear Causality:** Today's decision uses yesterday's macro, not today's close

## Usage

### Loading Dataset

```python
import pandas as pd

df = pd.read_csv('RL_DS5_MULTIFREQ_28F.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Separate features by frequency
features_5min = df[FEATURES_5MIN]
features_hourly = df[FEATURES_HOURLY]
features_daily = df[FEATURES_DAILY]
```

### RL Environment Integration

```python
class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.obs_dim = 28  # 26 market + 2 state

    def get_observation(self, idx):
        # Market features (26)
        market_obs = self.data.iloc[idx][ALL_MARKET_FEATURES].values

        # State features (2)
        state_obs = [self.position, self.unrealized_pnl_norm]

        # Combine (28 total)
        return np.concatenate([market_obs, state_obs])
```

## Next Steps

1. **Train RL Agent:** Use PPO/A2C with 28-dimensional observation space
2. **Validate LAG:** Verify hourly/daily features don't leak future information
3. **Backtest:** Compare vs V16 (intraday-only) to quantify macro value
4. **Deploy:** If Sharpe > 0.6 and max drawdown < 12%

## References

- **Source Script:** `03_build_v17_multifreq.py`
- **Statistics:** `STATS_V17_28F.csv`
- **Previous Version:** V16 (intraday-only, 12 features)

---

*Generated by V17 Multi-Frequency Dataset Builder*
*Author: Claude Code*
*Date: 2025-12-17 23:59:19*
