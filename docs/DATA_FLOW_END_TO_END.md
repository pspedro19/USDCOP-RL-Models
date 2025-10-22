# 🔄 Flujo de Datos End-to-End - Sistema USDCOP Trading

## Arquitectura Completa: Desde API Externa hasta Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         USDCOP TRADING SYSTEM ARCHITECTURE                      │
│                         Storage Registry + Manifest Pattern                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ DATA SOURCES │
└──────────────┘
     │
     ├── TwelveData API (20 keys) ─────┐
     ├── MT5 (opcional) ───────────────┤
     └── WebSocket (real-time) ────────┤
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │    AIRFLOW DAGs        │
                           │   L0 → L1 → ... → L6   │
                           └────────────────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                │                                               │
                ▼                                               ▼
        ┌──────────────┐                              ┌──────────────┐
        │  PostgreSQL  │                              │    MinIO     │
        │  TimescaleDB │                              │ (S3-compat)  │
        └──────────────┘                              └──────────────┘
        Raw OHLCV (L0)                                L1-L6 Features
        Trading Signals                               Models (ONNX)
        Performance Metrics                           Backtest Results
                │                                               │
                └───────────────────┬───────────────────────────┘
                                    │
                           ┌────────────────┐
                           │  FastAPI APIs  │
                           │ (7 services)   │
                           └────────────────┘
                                    │
                           app/deps.py (Repository)
                           config/storage.yaml
                                    │
                                    ▼
                           ┌────────────────┐
                           │  Next.js 15    │
                           │   Dashboard    │
                           │  (13 views)    │
                           └────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │   User Browser │
                           └────────────────┘
```

---

## 📦 Capa 1: Adquisición de Datos (L0)

### **Fuentes de Datos**

#### **TwelveData API** (Principal)
```python
# DAG: usdcop_m5__01_l0_intelligent_acquire.py
# Frecuencia: Cada 5 minutos (Mon-Fri, 08:00-12:55 COT)

ENDPOINTS = {
    "time_series": "https://api.twelvedata.com/time_series",
    "quote": "https://api.twelvedata.com/quote"
}

KEYS = [
    "key_1_through_20"  # 20 API keys para distribución de carga
]
```

**Proceso:**
1. **Gap Detection**: Query PostgreSQL para encontrar fechas faltantes
   ```sql
   SELECT expected_time FROM generate_series(start, end, '5 min')
   WHERE expected_time NOT IN (SELECT datetime FROM market_data)
   ```

2. **Batch Fetch**: Download en lotes para respetar rate limits
   ```python
   for batch in gaps:
       data = fetch_twelvedata(symbol="USDCOP", interval="5min", batch=batch)
   ```

3. **Validation**: Verificar OHLC constraints
   ```python
   assert high >= low
   assert low <= close <= high
   assert low <= open <= high
   ```

4. **Insert to PostgreSQL**:
   ```sql
   INSERT INTO market_data (symbol, datetime, open, high, low, close, volume, source)
   VALUES (...)
   ON CONFLICT (symbol, datetime, timeframe, source) DO NOTHING
   ```

5. **Backup to MinIO**:
   ```python
   s3_hook.load_string(
       string_data=df.to_csv(index=False),
       key="00-raw-usdcop-marketdata/YYYY-MM-DD/data.csv",
       bucket_name="usdcop"
   )
   ```

---

## 🔧 Capa 2: Pipeline de Transformación (L1-L6)

### **L1: Standardize** (Estandarización)

**Input**: PostgreSQL `market_data` table
**Output**: MinIO `01-l1-ds-usdcop-standardize/YYYY-MM-DD/`

**Transformaciones**:
```python
# 1. Timezone Conversion
df['datetime_cot'] = df['datetime_utc'].dt.tz_convert('America/Bogota')

# 2. Session Classification
df['is_trading_session'] = (df['hour'] >= 8) & (df['hour'] <= 12) & (df['minute'] <= 55)

# 3. Holiday Filtering
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=start_date, end=end_date)
df = df[~df['datetime_cot'].dt.date.isin(holidays)]

# 4. Repeated OHLC Detection
df['is_stale'] = (
    (df['open'] == df['open'].shift(1)) &
    (df['high'] == df['high'].shift(1)) &
    (df['low'] == df['low'].shift(1)) &
    (df['close'] == df['close'].shift(1))
)
```

**Output Files**:
```
01-l1-ds-usdcop-standardize/2025-10-20/
├── standardized_data_accepted.parquet
├── standardized_data_rejected.parquet
├── metadata.json
└── READY (signal file)
```

---

### **L2: Prepare** (Limpieza + Indicadores)

**Input**: MinIO L1 standardized data
**Output**: MinIO `02-l2-ds-usdcop-prepare/YYYY-MM-DD/`

**Transformaciones**:

#### **1. Winsorization** (Outlier Clipping)
```python
def winsorize_returns(returns, n_sigma=4):
    """Clip outliers a 4*sigma"""
    mean = returns.mean()
    std = returns.std()
    lower = mean - n_sigma * std
    upper = mean + n_sigma * std

    clipped = returns.clip(lower, upper)
    winsor_rate = ((returns != clipped).sum() / len(returns)) * 100

    return clipped, winsor_rate
```

#### **2. HOD Deseasonalization** (Hour-of-Day)
```python
def hod_deseasonalize(df):
    """Normalize by hour using robust z-score"""
    df['hour'] = df['datetime'].dt.hour

    # Median y MAD por hora
    hod_median = df.groupby('hour')['close'].transform('median')
    hod_mad = df.groupby('hour')['close'].transform(
        lambda x: np.median(np.abs(x - x.median()))
    )

    # Z-score robusto
    df['close_hod_z'] = (df['close'] - hod_median) / hod_mad

    return df
```

#### **3. Technical Indicators** (60+ indicators usando TA-Lib)
```python
import talib

# Momentum
df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
df['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

# Trend
df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=14)

# Moving Averages
for period in [5, 10, 20, 50, 100, 200]:
    df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
    df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)

# MACD
df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

# Bollinger Bands
df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])

# Volatility
df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

# Volume
df['obv'] = talib.OBV(df['close'], df['volume'])
```

**Output Files**:
```
02-l2-ds-usdcop-prepare/2025-10-20/
├── data_premium_strict.parquet   # 08:00-12:55 COT only
├── data_premium_flex.parquet     # Extended hours
├── hod_baselines.json
├── winsor_stats.json
├── metadata.json
└── READY
```

---

### **L3: Feature** (Feature Engineering)

**Input**: MinIO L2 prepared data
**Output**: MinIO `03-l3-ds-usdcop-feature/YYYY-MM-DD/`

**Features** (30 total):
```python
# Multi-timeframe returns
df['ret_5m'] = df['close'].pct_change(1)
df['ret_10m'] = df['close'].pct_change(2)
df['ret_15m'] = df['close'].pct_change(3)
df['ret_30m'] = df['close'].pct_change(6)

# Volatility
df['realized_vol_30m'] = df['ret_5m'].rolling(6).std()
df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000

# Momentum
df['rsi_norm'] = (df['rsi_14'] - 50) / 50  # Normalize to [-1, 1]
df['macd_zscore'] = (df['macd'] - df['macd'].rolling(50).mean()) / df['macd'].rolling(50).std()

# Position in Bollinger Bands
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# EMA Cross Signal
df['ema_cross_signal'] = np.sign(df['ema_10'] - df['ema_20'])

# Volume
df['volume_zscore'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()

# Time Features
df['time_of_day_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['time_of_day_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

**Anti-Leakage Validation** (Forward IC):
```python
def calculate_forward_ic(df, feature, horizons=[1, 5, 10]):
    """
    Calcula Information Coefficient entre feature y retornos futuros.
    IC > 0.10 indica data leakage.
    """
    ic_results = {}

    for h in horizons:
        future_return = df['close'].pct_change(h).shift(-h)
        ic = df[feature].corr(future_return)
        ic_results[f'ic_h{h}'] = ic

    return ic_results

# Ejemplo
ic = calculate_forward_ic(df, 'rsi_norm')
# Si ic_h1 > 0.10 → FAIL (feature tiene información futura)
```

**Output Files**:
```
03-l3-ds-usdcop-feature/2025-10-20/
├── features.parquet
├── feature_specs.json
├── ic_validation.json
├── metadata.json
└── READY
```

---

### **L4: RL-Ready** (Dataset para RL)

**Input**: MinIO L3 features
**Output**: MinIO `04-l4-ds-usdcop-rlready/YYYY-MM-DD/`

**Transformaciones**:

#### **1. Episode Generation** (60 bars/episode)
```python
# Session 08:00-12:55 COT = 60 bars M5
df['episode_id'] = df['datetime'].dt.date
df['t_in_episode'] = df.groupby('episode_id').cumcount()

# Filter complete episodes
episode_lengths = df.groupby('episode_id').size()
complete_episodes = episode_lengths[episode_lengths == 60].index
df = df[df['episode_id'].isin(complete_episodes)]
```

#### **2. Normalization by Hour** (Robust Z-score)
```python
obs_cols = [f'obs_{i:02d}' for i in range(17)]

for col in obs_cols:
    # Por hora (no cross-hour leakage)
    df[col + '_norm'] = df.groupby('hour')[col].transform(
        lambda x: (x - x.median()) / np.median(np.abs(x - x.median()))
    )

    # Clip to [-5, 5]
    df[col + '_norm'] = df[col + '_norm'].clip(-5, 5)
```

#### **3. Train/Val/Test Split** (con embargo)
```python
from sklearn.model_selection import TimeSeriesSplit

# 70% train, 15% val, 15% test
episodes = sorted(df['episode_id'].unique())
n_episodes = len(episodes)

train_end = int(n_episodes * 0.70)
val_end = int(n_episodes * 0.85)

# Embargo: 5 días entre splits
embargo_days = 5

train_episodes = episodes[:train_end - embargo_days]
val_episodes = episodes[train_end + embargo_days:val_end - embargo_days]
test_episodes = episodes[val_end + embargo_days:]

df['split'] = 'train'
df.loc[df['episode_id'].isin(val_episodes), 'split'] = 'val'
df.loc[df['episode_id'].isin(test_episodes), 'split'] = 'test'
```

#### **4. Reward Calculation** (PnL - Costos)
```python
def calculate_reward(df, spread_bps=15):
    """Reward = PnL - transaction costs"""

    # PnL from action
    df['pnl'] = df['action'].shift(1) * df['ret_5m']

    # Transaction cost (spread cuando hay trade)
    df['action_change'] = (df['action'] != df['action'].shift(1)).astype(int)
    df['transaction_cost'] = df['action_change'] * (spread_bps / 10000)

    # Final reward
    df['reward'] = df['pnl'] - df['transaction_cost']

    return df
```

**Output Files**:
```
04-l4-ds-usdcop-rlready/2025-10-20/
├── replay_dataset.parquet        # Episodes completos
├── env_spec.json                 # Observation space spec
├── reward_spec.json              # Reward calculation
├── split_spec.json               # Train/val/test splits
├── obs_clip_rates.json           # Clip rate por feature
├── metadata.json
└── READY
```

**env_spec.json** example:
```json
{
  "observation_space": {
    "type": "Box",
    "shape": [17],
    "dtype": "float32",
    "low": -5.0,
    "high": 5.0
  },
  "action_space": {
    "type": "Discrete",
    "n": 3
  }
}
```

---

### **L5: Serving** (Model Training & Export)

**Input**: MinIO L4 RL-ready dataset
**Output**: MinIO `05-l5-ds-usdcop-serving/YYYY-MM-DD/`

**Proceso**:
```python
# 1. Load L4 data
df_train = load_from_minio("l4/.../replay_dataset.parquet", split="train")

# 2. Train RL model (PPO, DQN, etc.)
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

model.learn(total_timesteps=1_000_000)

# 3. Export to ONNX
import onnx
from onnxruntime import InferenceSession

onnx_path = "model.onnx"
model.policy.save_onnx(onnx_path)

# 4. Validation (latency test)
session = InferenceSession(onnx_path)
obs_sample = np.random.randn(1, 17).astype(np.float32)

import time
latencies = []
for _ in range(1000):
    start = time.time()
    action = session.run(None, {"obs": obs_sample})[0]
    latencies.append((time.time() - start) * 1000)  # ms

p99_latency = np.percentile(latencies, 99)
assert p99_latency <= 20  # Must be <= 20ms

# 5. Create bundle
bundle = {
    "model.onnx": onnx_bytes,
    "model_manifest.json": {...},
    "metrics_summary.json": {...},
    "training_config.json": {...}
}
```

**Output Files**:
```
05-l5-ds-usdcop-serving/2025-10-20/
├── model.onnx                    # ONNX model
├── model_manifest.json           # Model metadata
├── metrics_summary.json          # Training metrics
├── training_config.json          # Hyperparameters
├── metadata.json
└── READY
```

---

### **L6: Backtest** (Performance Evaluation)

**Input**: MinIO L5 model + L4 test data
**Output**: MinIO `usdcop-l6-backtest/YYYY-MM-DD/`

**Proceso**:
```python
# 1. Load model
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")

# 2. Load test data
df_test = load_from_minio("l4/.../replay_dataset.parquet", split="test")

# 3. Run backtest
positions = []
pnl = []

for episode_id in df_test['episode_id'].unique():
    episode = df_test[df_test['episode_id'] == episode_id]

    for t, row in episode.iterrows():
        obs = row[obs_cols].values.reshape(1, -1).astype(np.float32)

        # Get action from model
        action = session.run(None, {"obs": obs})[0][0]

        # Calculate PnL
        ret = row['ret_5m']
        spread_cost = 0.0015 if action != positions[-1] else 0  # 15 bps

        step_pnl = action * ret - spread_cost
        pnl.append(step_pnl)
        positions.append(action)

# 4. Calculate metrics
cumulative_returns = np.cumsum(pnl)
sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252 * 60)  # Annualized
sortino = np.mean(pnl) / np.std([p for p in pnl if p < 0]) * np.sqrt(252 * 60)

# Max Drawdown
cummax = np.maximum.accumulate(cumulative_returns)
drawdown = (cumulative_returns - cummax) / cummax
max_drawdown = drawdown.min()

# Calmar
total_return = cumulative_returns[-1]
calmar = total_return / abs(max_drawdown)

# 5. Save results
results = {
    "model_id": "ppo_v1.2.3",
    "split": "test",
    "sortino": sortino,
    "sharpe": sharpe,
    "calmar": calmar,
    "max_drawdown": max_drawdown,
    "total_return": total_return,
    "trades": len([p for p, prev_p in zip(positions[1:], positions[:-1]) if p != prev_p]),
    "total_costs": sum([0.0015 for p, prev_p in zip(positions[1:], positions[:-1]) if p != prev_p])
}
```

**Output Files**:
```
usdcop-l6-backtest/2025-10-20/
├── trades.parquet                # Individual trades
├── returns.parquet               # Step-by-step returns
├── kpis.json                     # Performance metrics
├── equity_curve.json             # Cumulative PnL
├── metadata.json
└── READY
```

---

## 📡 Capa 3: API Layer (FastAPI + Repository Pattern)

### **Storage Registry** (`config/storage.yaml`)
```yaml
layers:
  l0:
    backend: postgres
    table: market_data

  l2:
    backend: s3
    bucket: usdcop
    prefix: l2

  l4:
    backend: s3
    bucket: usdcop
    prefix: l4
```

### **Manifest System** (`/_meta/`)
```json
// _meta/l4_latest.json
{
  "run_id": "2025-10-20",
  "layer": "l4",
  "path": "l4/2025-10-20/",
  "dataset_hash": "sha256:abc123...",
  "updated_at": "2025-10-20T12:00:00Z"
}

// _meta/l4_2025-10-20_run.json
{
  "run_id": "2025-10-20",
  "layer": "l4",
  "started_at": "2025-10-20T08:00:00Z",
  "completed_at": "2025-10-20T08:15:00Z",
  "status": "success",
  "files": [
    {
      "name": "replay_dataset.parquet",
      "path": "l4/2025-10-20/replay_dataset.parquet",
      "size_bytes": 12345678,
      "row_count": 50000,
      "checksum": "sha256:def456..."
    }
  ]
}
```

### **Repository Pattern** (`app/deps.py`)
```python
def read_layer_data(layer: str, run_id: Optional[str] = None):
    """Unified reader - automatically selects DB or S3"""

    config = get_layer_config(layer)

    if config["backend"] == "postgres":
        # Read from PostgreSQL
        return execute_sql_query(f"SELECT * FROM {config['table']}")

    elif config["backend"] == "s3":
        # Read from MinIO
        manifest = read_latest_manifest(config["bucket"], config["prefix"])
        data_file = manifest["files"][0]
        return read_parquet_dataset(config["bucket"], data_file["path"])
```

### **API Endpoints** (`app/routers/`)

**L0 Router** (`app/routers/l0.py`):
```python
@router.get("/pipeline/l0/extended-statistics")
def get_l0_extended_statistics(days: int = 30):
    """
    Returns:
    - Coverage percentage
    - OHLC violations
    - Duplicates
    - Stale rate
    - Gaps

    GO/NO-GO: Pass if all criteria met
    """
    coverage_pct = calculate_coverage(days)
    violations = count_ohlc_violations(days)

    passed = (
        coverage_pct >= 95 and
        violations == 0
    )

    return {"quality_metrics": {...}, "pass": passed}
```

**L4 Router** (`app/routers/l4.py`):
```python
@router.get("/pipeline/l4/quality-check")
def get_l4_quality_check(run_id: Optional[str] = None):
    """
    Reads from MinIO using manifest system

    Returns:
    - Clip rates per observation
    - Reward reproducibility (RMSE)
    - Split embargo validation
    """
    manifest = read_latest_manifest("usdcop", "l4")
    df = read_parquet_dataset("usdcop", "l4/2025-10-20/replay_dataset.parquet")

    clip_rates = {f"obs_{i:02d}": calculate_clip_rate(df, i) for i in range(17)}

    return {"clip_rates": clip_rates, "pass": max(clip_rates.values()) <= 0.5}
```

---

## 🎨 Capa 4: Frontend (Next.js 15)

### **API Client** (`lib/api.ts`)
```typescript
const API_BASE = "http://localhost:8004";

export async function getL4Contract() {
  const res = await fetch(`${API_BASE}/pipeline/l4/contract`);
  return res.json();
}

export async function getL4QualityCheck(runId?: string) {
  const url = `${API_BASE}/pipeline/l4/quality-check`;
  const params = runId ? `?run_id=${runId}` : "";
  const res = await fetch(url + params);
  return res.json();
}
```

### **Dashboard Components**

**Pipeline Status View** (`components/PipelineStatus.tsx`):
```tsx
export function PipelineStatus() {
  const [l0Stats, setL0Stats] = useState(null);
  const [l4Quality, setL4Quality] = useState(null);

  useEffect(() => {
    async function loadData() {
      const l0 = await fetch('/api/pipeline/l0/extended-statistics?days=30');
      const l4 = await fetch('/api/pipeline/l4/quality-check');

      setL0Stats(await l0.json());
      setL4Quality(await l4.json());
    }

    loadData();
  }, []);

  return (
    <div>
      <Card title="L0 Quality">
        <Metric>Coverage: {l0Stats?.quality_metrics.coverage_pct}%</Metric>
        <Badge color={l0Stats?.pass ? "green" : "red"}>
          {l0Stats?.pass ? "PASS" : "FAIL"}
        </Badge>
      </Card>

      <Card title="L4 RL-Ready">
        <Metric>Max Clip Rate: {l4Quality?.quality_checks.max_clip_rate_pct}%</Metric>
        <Badge color={l4Quality?.pass ? "green" : "red"}>
          {l4Quality?.pass ? "PASS" : "FAIL"}
        </Badge>
      </Card>
    </div>
  );
}
```

---

## 🔍 Cómo se Calcula Cada Métrica

### **Spread (Corwin-Schultz Proxy)**
```python
# NO tenemos bid-ask real, usamos proxy de high-low
def corwin_schultz_spread(high, low):
    """
    Estima spread usando rangos high-low de 2 periodos consecutivos
    Paper: Corwin & Schultz (2012)
    """
    hl_ratio = np.log(high / low)
    hl_prev = hl_ratio.shift(1)

    beta = hl_ratio**2 + hl_prev**2

    max_high = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    min_low = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = (np.log(max_high / min_low))**2

    alpha = (np.sqrt(2*beta) - np.sqrt(beta)) / (3 - 2*np.sqrt(2)) - \
            np.sqrt(gamma / (3 - 2*np.sqrt(2)))

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread_bps = spread * 10000  # En basis points

    return spread_bps
```

### **Session Progress** (08:00-12:55 COT)
```python
def calculate_session_progress():
    """60 barras M5 = 300 minutos = 08:00-12:55"""
    now = datetime.now(pytz.timezone('America/Bogota'))
    session_start = now.replace(hour=8, minute=0)
    session_end = now.replace(hour=12, minute=55)

    if now < session_start:
        status = "PRE_MARKET"
        progress = 0.0
    elif now > session_end:
        status = "POST_MARKET"
        progress = 100.0
    else:
        elapsed = (now - session_start).total_seconds()
        total = (session_end - session_start).total_seconds()
        progress = (elapsed / total) * 100
        status = "ACTIVE"

    bars_elapsed = int(elapsed / 300)  # Cada 5 min

    return {
        "status": status,
        "progress": progress,
        "bars_elapsed": bars_elapsed,
        "bars_total": 60
    }
```

### **Sortino Ratio** (Downside-only volatility)
```python
def sortino_ratio(returns, target=0, periods=252*60):
    """
    Sortino = (Mean Return - Target) / Downside Deviation
    Annualized for M5 data (252 days * 60 bars/day)
    """
    excess = returns - target
    downside = returns[returns < target]
    downside_std = downside.std()

    sortino = (excess.mean() / downside_std) * np.sqrt(periods)

    return sortino
```

### **Max Drawdown**
```python
def max_drawdown(cumulative_returns):
    """Peak-to-trough decline"""
    cummax = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - cummax) / cummax
    return drawdown.min()
```

---

## ✅ Quality Gates (GO/NO-GO)

### **L0 Quality Gates**
```yaml
coverage_pct: >= 95          # At least 95% of expected bars
ohlc_violations: == 0        # No invalid OHLC (high < low, etc.)
duplicates: == 0             # No duplicate timestamps
stale_rate: <= 2%            # Repeated OHLC values
gaps_gt1: == 0               # No gaps > 1 missing bar
```

### **L2 Quality Gates**
```yaml
winsorization_rate: <= 1.0%  # Max 1% outliers clipped
hod_mad: [0.8, 1.2]          # Hour-of-day MAD in range
hod_median_abs: <= 0.05      # Hour median close to 0
nan_rate: <= 0.5%            # Max 0.5% missing values
```

### **L4 Quality Gates**
```yaml
obs_clip_rate: <= 0.5%       # Per feature, max 0.5% clipped
reward_rmse: < 0.01          # Reward reproducibility
reward_std: > 0.1            # Non-degenerate rewards
reward_zero: < 1.0%          # Not all zeros
```

### **L6 Quality Gates**
```yaml
sortino: >= 1.3              # Risk-adjusted return
calmar: >= 0.8               # Return/MaxDD ratio
trades_min: >= 100           # Sufficient sample size
```

---

## 🔄 Flujo Completo de una Métrica (Ejemplo: RL Metrics)

```
1. DATA ACQUISITION (L0)
   TwelveData API → PostgreSQL market_data table
   ↓

2. PIPELINE TRANSFORMATIONS (L1→L4)
   L1: Timezone conversion, session filter
   L2: Winsorization, HOD, 60+ indicators
   L3: Feature engineering (30 features)
   L4: RL-ready (17 obs, splits, rewards)
   ↓
   Each layer writes to MinIO + creates manifest

3. PIPELINE WRITES MANIFEST
   s3://usdcop/_meta/l4_latest.json
   {
     "run_id": "2025-10-20",
     "path": "l4/2025-10-20/",
     "files": ["replay_dataset.parquet", ...]
   }

4. API READS VIA REPOSITORY
   GET /api/analytics/rl-metrics
   ↓
   app/deps.py → read_latest_manifest("usdcop", "l4")
                → read_parquet_dataset("usdcop", "l4/2025-10-20/replay_dataset.parquet")
   ↓
   Calculate metrics:
   - total_episodes = df['episode_id'].nunique()
   - avg_reward = df['reward'].mean()
   - success_rate = (df['reward'] > 0).sum() / len(df)

5. API RETURNS JSON
   {
     "status": "OK",
     "run_id": "2025-10-20",
     "total_episodes": 250,
     "avg_reward": 125.5,
     "success_rate": 0.52,
     "note": "Real data from L4 pipeline"
   }

6. FRONTEND FETCHES
   const data = await fetch('/api/analytics/rl-metrics');

7. REACT COMPONENT RENDERS
   <Card>
     <Metric>{data.total_episodes} Episodes</Metric>
     <Text>Avg Reward: {data.avg_reward}</Text>
   </Card>
```

---

## 📚 Referencias Rápidas

### **Storage Registry**
```bash
# Ver configuración
cat config/storage.yaml

# Layer L0 → PostgreSQL market_data
# Layers L1-L6 → MinIO s3://usdcop/lX/
```

### **Manifests**
```bash
# Último run de L4
aws --endpoint-url http://localhost:9000 s3 cp \
  s3://usdcop/_meta/l4_latest.json -

# Run específico
aws s3 cp s3://usdcop/_meta/l4_2025-10-20_run.json -
```

### **API Testing**
```bash
# L0 Quality
curl http://localhost:8004/api/pipeline/l0/extended-statistics?days=30

# L4 Contract
curl http://localhost:8004/api/pipeline/l4/contract

# L6 Backtest
curl http://localhost:8004/api/backtest/l6/results?model_id=ppo_v1&split=test
```

---

**Última actualización**: 2025-10-21
**Versión**: 2.0 (Storage Registry + Manifest Pattern)
