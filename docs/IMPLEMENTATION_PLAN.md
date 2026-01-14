# USDCOP-RL-Models - Plan de Implementación

**Generado**: 2026-01-11
**Actualizado**: 2026-01-11 (v3.6 - Mejoras de Auditoría Externa)
**Basado en**: Análisis 10-Agentes + Revisión ML Workflow + Auditoría 56-Preguntas
**Objetivo**: Coherencia total + Trazabilidad Production-Ready
**Idioma**: Español (código y términos técnicos en inglés)

> **Nota sobre Coherence Score**: Ver sección [Checklist de Coherencia Verificable](#checklist-de-coherencia-verificable) para criterios objetivos.

---

## Executive Summary

### Audit Results Integration

| Categoría | Críticas (🔴) | Altas (🟠) | Medias (🟡) | Score |
|-----------|---------------|------------|-------------|-------|
| Replay Mode | 2 | 2 | 2 | 4/10 |
| Features | 2 | 2 | 2 | 5/10 |
| Pipelines/DAGs | 2 | 2 | 2 | 4/10 |
| Frontend/Backend | 1 | 2 | 2 | 6/10 |
| Visualizaciones | 1 | 2 | 3 | 6/10 |
| ML/Trading | 2 | 2 | 2 | 5/10 |
| Clean Code | 3 | 2 | 1 | 5/10 |
| Redundancia | 2 | 3 | 2 | 4/10 |
| Consistencia Datos | 2 | 2 | 0 | 5/10 |
| Trazabilidad | 2 | 2 | 0 | 4/10 |
| **TOTAL** | **19** | **21** | **16** | **4.8/10** |

### Plan Coverage

| Priority | Items | Status |
|----------|-------|--------|
| P0 (Critical) | 11 | All audit criticals mapped + merge_asof fix |
| P1 (High) | 13 | All audit highs mapped + bid_ask_spread + Feature Contract Pattern |
| P2 (Medium) | 20 | All audit mediums + archive cleanup explicit |

---

## Priority 0: CRITICAL (Fix Immediately)

### P0-1: Norm Stats Version Mismatch
**Audit Ref**: Related to FE-01, PL-03
**Status**: 🔴 CRITICAL
**Location**: `services/inference_api/config.py:24`
**Issue**: V20 model uses V19 normalization stats.

```python
# CURRENT (WRONG):
norm_stats_path: str = "config/v19_norm_stats.json"

# FIX:
norm_stats_path: str = "config/v20_norm_stats.json"
```

#### P0-1 Addendum: If v20_norm_stats.json Does NOT Exist

```bash
if [ ! -f "config/v20_norm_stats.json" ]; then
    echo "CRITICAL: v20_norm_stats.json missing"
    python notebooks/export_norm_stats.py --version v20
    exit 1
fi
```

---

### P0-2: Action Threshold Mismatch
**Audit Ref**: ML-02
**Status**: 🔴 CRITICAL
**Location**: `services/inference_api/config.py:32-33`

```python
# FIX:
threshold_long: float = 0.30
threshold_short: float = -0.30
```

#### P0-2 Verification Required

```python
# scripts/validate_thresholds.py
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
for t in thresholds:
    metrics = run_backtest(threshold_long=t, period='2025-01-01 to 2025-06-30')
    print(f"{t}: Sharpe={metrics.sharpe:.2f}")
```

---

### P0-3: ADX Hardcoded Placeholder
**Audit Ref**: FE-04
**Status**: 🔴 CRITICAL
**Location**: `airflow/dags/l5_multi_model_inference.py:371-373`

```python
# CURRENT (BROKEN):
return 25.0  # Placeholder

# FIX: Implement real ADX calculation
```

---

### P0-4: Hardcoded Default Passwords
**Audit Ref**: Security
**Status**: 🔴 CRITICAL SECURITY

```python
# FIX: Remove defaults
postgres_password: str = os.environ["POSTGRES_PASSWORD"]
```

---

### P0-5: Missing MIN_TICK_INTERVAL_MS
**Audit Ref**: Code bug
**Status**: 🔴 RUNTIME ERROR
**Location**: `utils/replayPerformance.ts:299`

```typescript
// FIX: Add to PERF_THRESHOLDS
MIN_TICK_INTERVAL_MS: 16,
```

---

### P0-6: Hardcoded Model ID in fetchEquityCurve
**Audit Ref**: FB-02
**Status**: 🔴 CRITICAL
**Location**: `lib/replayApiClient.ts:518`

```typescript
// FIX: Use dynamic modelId
const modelId = options?.modelId || 'ppo_v20';
```

---

### P0-7: ML Workflow Disciplinado
**Audit Ref**: ML-02, ML-05, ML-06
**Status**: 🔴 CRITICAL - P-HACKING RISK

**Acciones**:
1. Auditar cuántas veces se miró validation set
2. Verificar si test set fue contaminado
3. Establecer proceso: Exploration → Single-pass Validation → Test
4. Crear `config/hyperparameter_decisions.json`

---

### P0-8: Features Snapshot NO Se Preserva en BD (NUEVO)
**Audit Ref**: RM-03, TL-01
**Status**: 🔴 CRITICAL
**Issue**: Imposible auditar qué features usó cada trade.

**SQL Migration**:
```sql
-- migrations/002_add_traceability_columns.sql

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS features_snapshot JSONB;

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS model_hash VARCHAR(64);

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS model_version VARCHAR(20);

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS norm_stats_version VARCHAR(20);

ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS config_snapshot JSONB;

-- Index for audit queries
CREATE INDEX idx_trades_model_hash ON trades_history(model_hash);
CREATE INDEX idx_trades_model_version ON trades_history(model_version);

COMMENT ON COLUMN trades_history.features_snapshot IS
  'Complete feature vector used for this trade decision';
COMMENT ON COLUMN trades_history.model_hash IS
  'SHA256 hash of model weights for integrity verification';
```

**Python Implementation**:
```python
# services/inference_api/core/model_hasher.py
import hashlib

def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of model weights for traceability"""
    with open(model_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

# At startup
MODEL_HASH = compute_model_hash(settings.full_model_path)

# When creating trade
trade_record = {
    # ... existing fields ...
    'features_snapshot': json.dumps(observation.tolist()),
    'model_hash': MODEL_HASH,
    'model_version': 'v20',
    'norm_stats_version': 'v20',
    'config_snapshot': json.dumps({
        'threshold_long': settings.threshold_long,
        'threshold_short': settings.threshold_short,
        'slippage_bps': settings.slippage_bps,
    }),
}
```

---

### P0-9: Look-Ahead Bias en Market Regime (NUEVO)
**Audit Ref**: FE-05
**Status**: 🔴 CRITICAL
**Issue**: Percentil de volatilidad puede usar datos futuros.

**Ubicación**: `regime_detector.py`, backtest scripts

**Current Problem**:
```python
# WRONG - uses future data!
df['vol_percentile'] = df['volatility'].rank(pct=True)
```

**Fix**:
```python
# CORRECT - rolling window, closed='left' to exclude current bar
def calculate_regime_safe(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """Calculate market regime WITHOUT look-ahead bias"""

    # Use expanding window for historical percentile
    vol_percentile = df['volatility'].expanding(min_periods=window).apply(
        lambda x: (x.iloc[-1] - x.iloc[:-1].min()) /
                  (x.iloc[:-1].max() - x.iloc[:-1].min() + 1e-8)
    )

    # Or use rolling with closed='left'
    vol_percentile_rolling = df['volatility'].rolling(
        window=window,
        closed='left'  # CRITICAL: exclude current bar
    ).apply(lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100)

    return vol_percentile_rolling

# Validation test
def test_no_lookahead_bias():
    """Verify regime calculation doesn't use future data"""
    df = load_test_data()

    # Calculate regime at each point
    for i in range(100, len(df)):
        # Regime at time i should only use data up to i-1
        regime_at_i = calculate_regime(df.iloc[:i])
        regime_full = calculate_regime(df)

        # These should be identical
        assert regime_at_i.iloc[-1] == regime_full.iloc[i-1], \
            f"Look-ahead bias detected at index {i}"
```

---

### P0-10: Forward-Fill Sin Límite en Macro Data (NUEVO)
**Audit Ref**: CD-03
**Status**: 🔴 CRITICAL
**Issue**: Datos futuros pueden penetrar training period via ffill() sin límite.

**Location**: `01_build_5min_datasets.py:751`

**Current (WRONG)**:
```python
df[col].ffill()  # No limit - can propagate indefinitely!
```

**Fix**:
```python
# Maximum forward-fill: 1 trading day (12 hours * 12 bars/hour = 144 bars for 5min)
MAX_FFILL_LIMIT = 144  # 12 hours of 5-min bars

def safe_ffill(df: pd.DataFrame, columns: list, limit: int = MAX_FFILL_LIMIT) -> pd.DataFrame:
    """Forward-fill with limit and quality tracking"""

    for col in columns:
        # Track how many consecutive NaNs
        nan_groups = df[col].isna().astype(int).groupby(
            df[col].notna().cumsum()
        ).cumsum()

        # Warn if exceeding limit
        if (nan_groups > limit).any():
            logger.warning(
                f"Column {col} has gaps > {limit} bars. "
                f"Max gap: {nan_groups.max()} bars"
            )

        # Apply limited ffill
        df[col] = df[col].ffill(limit=limit)

        # Add quality flag
        df[f'{col}_quality'] = np.where(
            nan_groups == 0, 'fresh',
            np.where(nan_groups <= 12, 'filled_1h',
            np.where(nan_groups <= limit, 'filled_partial', 'stale'))
        )

    return df
```

---

### P0-11: Merge AsOf Tolerance Causes Data Leakage (NUEVO)
**Audit Ref**: CD-03
**Status**: 🔴 CRITICAL
**Issue**: `merge_asof` con `tolerance=pd.Timedelta('1 day')` permite que datos macro del futuro se asocien con OHLCV del pasado.

**Location**: `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py:225-231`

**Current (WRONG)**:
```python
df = pd.merge_asof(
    df_ohlcv.sort_values('datetime'),
    df_macro_subset.sort_values('datetime'),
    on='datetime',
    direction='backward',
    tolerance=pd.Timedelta('1 day')  # ← ALLOWS FUTURE DATA LEAKAGE!
)
```

**Fix**:
```python
def safe_merge_macro(df_ohlcv: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro data WITHOUT tolerance to prevent data leakage.

    Strategy:
    1. For daily macro: forward-fill within trading day only
    2. For intraday: strict backward merge without tolerance
    """

    # Ensure macro data is at day start for daily indicators
    df_macro_daily = df_macro.copy()
    df_macro_daily['datetime'] = pd.to_datetime(
        df_macro_daily['datetime'].dt.date
    )

    # Strict backward merge - NO tolerance
    df = pd.merge_asof(
        df_ohlcv.sort_values('datetime'),
        df_macro_daily.sort_values('datetime'),
        on='datetime',
        direction='backward'
        # NO tolerance parameter - strict match only
    )

    # Log any rows that didn't get macro data
    missing_macro = df[df['dxy'].isna()].shape[0]
    if missing_macro > 0:
        logger.warning(
            f"[MERGE] {missing_macro} OHLCV rows without macro data "
            f"({missing_macro/len(df)*100:.2f}%)"
        )

    return df

# Validation test
def test_no_future_macro_leakage():
    """Verify macro data doesn't come from the future"""
    df = load_merged_dataset()

    for idx, row in df.iterrows():
        ohlcv_date = row['datetime']
        macro_date = row['macro_datetime']  # Need to track this!

        assert macro_date <= ohlcv_date, \
            f"Future leakage at {ohlcv_date}: macro from {macro_date}"
```

**Additional Safeguard** - Track macro source date:
```python
# When merging, keep track of macro data date
df_macro_daily['macro_source_date'] = df_macro_daily['datetime']

# After merge, verify no future data
assert (df['macro_source_date'] <= df['datetime']).all(), \
    "CRITICAL: Future macro data detected!"
```

---

## Priority 1: HIGH (This Week)

### P1-1: ExternalTaskSensor Between L0 and L1 DAGs
**Audit Ref**: PL-01, PL-02

```python
from airflow.sensors.external_task import ExternalTaskSensor

wait_for_l0 = ExternalTaskSensor(
    task_id='wait_for_l0_completion',
    external_dag_id='l0_data_acquisition',
    external_task_id='load_ohlcv',
    poke_interval=30,
    timeout=300,
    mode='reschedule',
    dag=dag,
)
```

---

### P1-2: Capture Model Metadata in Trades (ACTUALIZADO)
**Audit Ref**: RM-03, RM-04, TL-02, FE-03
**Update**: Now includes bid_ask_spread per FE-03 audit finding

```python
trade_record = {
    'entry_confidence': float(action_probs[action]),
    'model_metadata': json.dumps({
        'confidence': float(action_probs[action]),
        'action_probs': action_probs.tolist(),
        'critic_value': float(critic_value),
        'entropy': float(entropy),
        'advantage': float(advantage) if advantage else None,
        'model_version': 'v20',
        'norm_stats_version': 'v20',
    }),
    # UNIFIED FORMAT - Aligned with ARCHITECTURE_CONTRACTS.md
    'features_snapshot': json.dumps({
        # Raw observation vector (15 dims)
        'observation': observation.tolist(),

        # Named features for audit clarity (via FeatureBuilder.export_feature_snapshot)
        'features': {
            'log_ret_5m': float(observation[0]),
            'log_ret_1h': float(observation[1]),
            'log_ret_4h': float(observation[2]),
            'rsi_9': float(observation[3]),
            'atr_pct': float(observation[4]),    # period=10
            'adx_14': float(observation[5]),     # period=14
            'dxy_z': float(observation[6]),
            'dxy_change_1d': float(observation[7]),
            'vix_z': float(observation[8]),
            'embi_z': float(observation[9]),
            'brent_change_1d': float(observation[10]),
            'rate_spread': float(observation[11]),
            'usdmxn_change_1d': float(observation[12]),
            'position': float(observation[13]),
            'time_normalized': float(observation[14]),
        },

        # Market context (NEW per FE-03)
        'market_context': {
            'bid_ask_spread_bps': self._calculate_spread_bps(bid, ask, mid),
            'estimated_slippage_bps': self._estimate_slippage(volatility, hour),
            'execution_price': float(current_price),
            'timestamp_utc': datetime.utcnow().isoformat(),
        },

        # Traceability
        'contract_version': 'v20',
    }),
    'market_regime': self._determine_regime(atr, adx, vix_z),
}

def _calculate_spread_bps(self, bid: float, ask: float, mid: float) -> float:
    """
    Calculate bid-ask spread in basis points.
    If bid/ask not available, estimate from ATR.
    """
    if bid and ask and mid:
        spread = (ask - bid) / mid * 10000
        return round(spread, 2)
    else:
        # Estimate spread from ATR (typical for USD/COP ~10-20 bps)
        # This is a fallback when order book not available
        return 15.0  # Default estimate for SET-FX

def _estimate_slippage(self, volatility: float, hour_utc: int) -> float:
    """Estimate slippage based on volatility and time of day"""
    base_slippage = 5.0  # Base 5 bps per ML-04

    # Time multiplier
    if hour_utc < 13 or hour_utc > 17:  # Outside Colombia hours
        time_mult = 2.0
    else:
        time_mult = 1.0

    # Volatility multiplier
    vol_mult = 1.0 + (volatility / 0.02)

    return round(base_slippage * time_mult * vol_mult, 2)
```

---

### P1-3: Unify Feature Calculation Source ⚠️ SUPERSEDED BY P1-13
**Audit Ref**: FE-01, RD-01, RD-02
**Status**: ⚠️ SUPERSEDED - See P1-13 (Feature Contract Pattern)

~~Create shared `lib/features/` module imported by all components.~~

**Note**: This item is now fully covered by P1-13 which provides a more comprehensive solution with:
- Feature Contract specifications
- Unified FeatureBuilder class
- Model Registry integration
- Parity tests

See `ARCHITECTURE_CONTRACTS.md` for complete implementation details.

---

### P1-4: Remove Dual Equity Filtering
**Audit Ref**: RD-03, FB-03

---

### P1-5: Validate Feature Order at Runtime
**Audit Ref**: FE-01, FE-02

---

### P1-6: Fix EnrichedReplayTrade Data Loss + Confidence Hardcode (ACTUALIZADO)
**Audit Ref**: RM-03, VZ-01, VZ-06
**Update**: Now includes fix for hardcoded confidence in replay signals

**Issue 1**: EnrichedReplayTrade loses metadata during conversion
**Issue 2**: `TradingChartWithSignals.tsx:176` hardcodes `confidence: 75`

**Location**: `usdcop-trading-dashboard/components/charts/TradingChartWithSignals.tsx:168-180`

**Current (WRONG)**:
```typescript
if (isReplayMode && replayTrades && replayTrades.length > 0) {
  return replayTrades.map((trade) => ({
    id: trade.trade_id,
    timestamp: trade.timestamp || trade.entry_time || '',
    type: ['BUY', 'LONG'].includes((trade.side || '').toUpperCase()) ? 'BUY' : 'SELL',
    price: trade.entry_price,
    confidence: 75,  // ← WRONG: Hardcoded!
    stopLoss: null,
    takeProfit: null,
  }))
}
```

**Fix**:
```typescript
if (isReplayMode && replayTrades && replayTrades.length > 0) {
  return replayTrades.map((trade) => ({
    id: trade.trade_id,
    trade_id: trade.trade_id,
    timestamp: trade.timestamp || trade.entry_time || '',
    time: trade.timestamp || trade.entry_time || '',
    type: ['BUY', 'LONG'].includes((trade.side || '').toUpperCase()) ? 'BUY' : 'SELL',
    price: trade.entry_price,
    // FIXED: Use actual confidence from trade, with fallback
    confidence: trade.entry_confidence
      ?? trade.confidence
      ?? (trade.model_metadata?.confidence)
      ?? 75,  // Fallback only if no data available
    // Preserve additional metadata
    stopLoss: trade.stop_loss ?? null,
    takeProfit: trade.take_profit ?? null,
    // Include model info for debugging
    modelVersion: trade.model_version ?? 'unknown',
    entropy: trade.model_metadata?.entropy ?? null,
  }))
}
```

**Also update signal marker rendering** to show confidence visually:
```typescript
// In marker generation
const getMarkerColor = (confidence: number): string => {
  if (confidence >= 90) return '#00C853';  // High confidence - bright green
  if (confidence >= 70) return '#4CAF50';  // Good confidence - green
  if (confidence >= 50) return '#FFC107';  // Medium confidence - yellow
  return '#FF5722';                         // Low confidence - orange
};
```

---

### P1-7: Unificar Fuente de Datos Macro
**Audit Ref**: CD-01, CD-04

```python
def load_macro_from_db(start_date: str, end_date: str) -> pd.DataFrame:
    """Single source of truth: production DB"""
    query = """
        SELECT date, dxy, vix, embi, brent, treasury_10y, usdmxn
        FROM macro_indicators_daily
        WHERE date BETWEEN %s AND %s
    """
    return pd.read_sql(query, get_db_connection(), params=[start_date, end_date])
```

---

### P1-8: Paper Trading Validation
**Audit Ref**: ML-02, ML-06

Run 2-3 months paper trading, compare to backtest.

---

### P1-9: Dataset Registry Table (NUEVO)
**Audit Ref**: CD-02
**Issue**: Imposible auditar reproducibilidad de datasets.

**SQL**:
```sql
-- migrations/003_dataset_registry.sql

CREATE TABLE IF NOT EXISTS dataset_registry (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Content tracking
    checksum_md5 VARCHAR(32) NOT NULL,
    checksum_sha256 VARCHAR(64) NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    columns_json JSONB NOT NULL,

    -- Date ranges
    date_start DATE NOT NULL,
    date_end DATE NOT NULL,

    -- Source tracking
    source_files JSONB,  -- List of source files used
    pipeline_version VARCHAR(20),
    git_commit VARCHAR(40),

    -- Stats for validation
    stats_json JSONB,  -- {col: {mean, std, min, max, nulls}}

    UNIQUE(dataset_name, version)
);

CREATE INDEX idx_dataset_registry_name ON dataset_registry(dataset_name);
CREATE INDEX idx_dataset_registry_checksum ON dataset_registry(checksum_sha256);
```

**Python Registration**:
```python
# lib/dataset_registry.py
import hashlib
import pandas as pd

def register_dataset(df: pd.DataFrame, name: str, version: str, conn) -> dict:
    """Register dataset with checksums for reproducibility"""

    # Compute checksums
    content = df.to_csv(index=False).encode()
    md5 = hashlib.md5(content).hexdigest()
    sha256 = hashlib.sha256(content).hexdigest()

    # Compute stats
    stats = {}
    for col in df.select_dtypes(include='number').columns:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'nulls': int(df[col].isna().sum()),
        }

    # Insert into registry
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO dataset_registry
        (dataset_name, version, checksum_md5, checksum_sha256,
         row_count, column_count, columns_json, date_start, date_end, stats_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (dataset_name, version)
        DO UPDATE SET checksum_sha256 = EXCLUDED.checksum_sha256
        RETURNING id
    """, (name, version, md5, sha256, len(df), len(df.columns),
          json.dumps(list(df.columns)), df.index.min(), df.index.max(),
          json.dumps(stats)))

    conn.commit()
    return {'md5': md5, 'sha256': sha256, 'id': cur.fetchone()[0]}

def verify_dataset(df: pd.DataFrame, name: str, version: str, conn) -> bool:
    """Verify dataset matches registered checksum"""
    content = df.to_csv(index=False).encode()
    sha256 = hashlib.sha256(content).hexdigest()

    cur = conn.cursor()
    cur.execute("""
        SELECT checksum_sha256 FROM dataset_registry
        WHERE dataset_name = %s AND version = %s
    """, (name, version))

    row = cur.fetchone()
    if not row:
        raise ValueError(f"Dataset {name} v{version} not registered")

    if row[0] != sha256:
        raise ValueError(f"Checksum mismatch! Dataset may have changed.")

    return True
```

---

### P1-10: Externalizar Config V20 (NUEVO)
**Audit Ref**: CC-05
**Issue**: Configuración V20 hardcodeada requiere editar código fuente.

**Current (WRONG)** - `train_v20_production_parity.py:55-78`:
```python
# Hardcoded config in code
learning_rate = 3e-4
n_steps = 2048
# ... etc
```

**Fix** - Create `config/v20_config.yaml`:
```yaml
# config/v20_config.yaml
model:
  name: ppo_v20
  version: "20"

training:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.01
  clip_range: 0.2

thresholds:
  long: 0.30
  short: -0.30

features:
  observation_dim: 15
  norm_stats_path: "config/v20_norm_stats.json"

trading:
  initial_capital: 10000
  transaction_cost_bps: 25
  slippage_bps: 5

dates:
  training_end: "2024-12-31"
  validation_start: "2025-01-01"
  validation_end: "2025-06-30"
  test_start: "2025-07-01"
```

**Loader**:
```python
# lib/config_loader.py
import yaml
from pathlib import Path
from pydantic import BaseModel

class ModelConfig(BaseModel):
    name: str
    version: str

class TrainingConfig(BaseModel):
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    ent_coef: float
    clip_range: float

class V20Config(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    # ... other sections

def load_config(version: str = 'v20') -> V20Config:
    config_path = Path(f'config/{version}_config.yaml')
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return V20Config(**data)

# Usage
config = load_config('v20')
print(config.training.learning_rate)  # 3e-4
```

---

### P1-11: Model Hash Registration (NUEVO)
**Audit Ref**: TL-01
**Issue**: Sin validación de integridad del modelo.

```python
# lib/model_integrity.py
import hashlib
from pathlib import Path

def compute_model_hash(model_path: Path) -> str:
    """SHA256 hash of model for integrity verification"""
    with open(model_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def register_model(model_path: Path, conn) -> dict:
    """Register model in database with hash"""
    model_hash = compute_model_hash(model_path)

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO model_registry
        (model_id, model_path, model_hash, created_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (model_id) DO UPDATE SET model_hash = EXCLUDED.model_hash
    """, (model_path.stem, str(model_path), model_hash))
    conn.commit()

    return {'model_id': model_path.stem, 'hash': model_hash}

def verify_model_integrity(model_path: Path, expected_hash: str) -> bool:
    """Verify model hasn't been tampered with"""
    actual_hash = compute_model_hash(model_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Model integrity check FAILED!\n"
            f"Expected: {expected_hash}\n"
            f"Actual: {actual_hash}"
        )
    return True
```

**SQL**:
```sql
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) UNIQUE NOT NULL,
    model_path TEXT NOT NULL,
    model_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    norm_stats_hash VARCHAR(64),
    config_hash VARCHAR(64),
    training_dataset_id INTEGER REFERENCES dataset_registry(id)
);
```

---

### P1-12: Training vs Replay Pipeline Parity Test (NUEVO)
**Audit Ref**: PL-05
**Issue**: Pipelines completamente separados sin validación.

```python
# tests/integration/test_pipeline_parity.py

def test_training_replay_feature_parity():
    """Verify training and replay produce identical features"""

    # Load same timestamp from both pipelines
    test_ts = '2025-03-15 14:30:00'

    # Features from training pipeline (Python)
    training_features = training_pipeline.calculate_features(test_ts)

    # Features from replay/inference API (Python backend)
    inference_features = inference_api.calculate_features(test_ts)

    # Compare each feature
    for i, (train_f, inf_f) in enumerate(zip(training_features, inference_features)):
        assert abs(train_f - inf_f) < 1e-6, \
            f"Feature {i} ({FEATURE_ORDER[i]}) mismatch: " \
            f"training={train_f}, inference={inf_f}"

def test_typescript_python_preprocessing_parity():
    """Verify TypeScript and Python preprocessing match"""

    # Same raw data
    raw_data = load_test_ohlcv()

    # Python preprocessing
    py_features = python_preprocess(raw_data)

    # TypeScript preprocessing (via API call to test endpoint)
    ts_features = call_ts_preprocess_endpoint(raw_data)

    # Compare
    for i in range(len(py_features)):
        assert abs(py_features[i] - ts_features[i]) < 1e-6
```

---

### P1-13: Implementar Feature Contract Pattern (NUEVO)
**Audit Ref**: FE-01, PL-05, RD-01, RD-02, RM-01
**Issue**: 3 implementaciones separadas de features (training, inference, factory) sin contrato común.
**Documento detallado**: Ver `ARCHITECTURE_CONTRACTS.md`

**Estado Actual (Fragmentado)**:
1. `01_build_5min_datasets.py` - Training features con `ta.rsi()`, `ta.atr()`, etc.
2. `observation_builder.py` - Inference features con cálculos manuales
3. `feature_calculator_factory.py` - Factory que existe pero NO se usa

**Entregables** (aligned with ARCHITECTURE_CONTRACTS.md v1.2):

```
lib/features/
├── __init__.py
├── contract.py         # Feature specifications (SINGLE SOURCE OF TRUTH)
├── builder.py          # Unified FeatureBuilder class (includes normalization)
└── calculators/
    ├── __init__.py
    ├── returns.py      # log_ret_5m, log_ret_1h, log_ret_4h
    ├── rsi.py          # rsi_9 (period=9)
    ├── atr.py          # atr_pct (period=10) ← ALIGNED WITH observation_builder.py
    ├── adx.py          # adx_14 (period=14)
    └── macro.py        # dxy_z, vix_z, embi_z, brent_change_1d, rate_spread, usdmxn_change_1d

lib/model_registry.py   # Model → Contract → Hash linking (see P1-11)
```

**Implementación**:

1. **contract.py** - Define especificaciones de features:
```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    type: FeatureType  # TECHNICAL, MACRO, STATE
    dependencies: List[str]
    params: Dict[str, any]
    normalize: bool
    description: str

FEATURE_CONTRACT_V20 = {
    "version": "20",
    "observation_dim": 15,
    "features": [...],
    "feature_order": [...],
    "norm_stats_file": "config/v20_norm_stats.json",
}
```

2. **builder.py** - Único lugar donde se calculan features:
```python
class FeatureBuilder:
    def __init__(self, version: str = "v20"):
        self.contract = get_contract(version)
        self.norm_stats = self._load_norm_stats()

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        position: float,
        timestamp: pd.Timestamp,
    ) -> np.ndarray:
        """Builds observation vector used by ALL components"""
        ...

    def export_feature_snapshot(self, obs: np.ndarray) -> dict:
        """For storing in trades_history.features_snapshot"""
        ...
```

3. **Migrar componentes** para usar FeatureBuilder:
```python
# Training (01_build_5min_datasets.py)
from lib.features.builder import FeatureBuilder
builder = FeatureBuilder(version="v20")

# Inference API (observation_builder.py) - REEMPLAZAR
from lib.features.builder import FeatureBuilder
builder = FeatureBuilder(version="v20")

# Airflow DAGs
from lib.features.builder import FeatureBuilder
```

4. **Tests de paridad**:
```python
def test_training_inference_parity():
    """Features MUST be identical between training and inference"""
    builder = FeatureBuilder(version="v20")

    obs_training = builder.build_observation(...)
    obs_inference = builder.build_observation(...)  # Same inputs

    np.testing.assert_array_almost_equal(obs_training, obs_inference)
```

**Beneficio**:
- Elimina divergencia entre training e inference
- Garantiza que modelo "ve" exactamente lo que fue entrenado
- Facilita agregar nuevas versiones de modelos (v21, v22, etc.)
- Habilita feature snapshots auditables por trade

**Dependencias**:
- P0-8 (features_snapshot column)
- P1-11 (model hash registration)

---

## Priority 1: HIGH - Industry Grade (Week 3)

> **Fuente**: INDUSTRY_GRADE_MLOPS_RECOMMENDATIONS.md v1.0
> **Objetivo**: Elevar el sistema a estándares de hedge funds cuantitativos institucionales

### Madurez Actual vs Target

| Dimensión | Estado Actual | Target Institucional | Gap |
|-----------|---------------|---------------------|-----|
| Inference Latency | ~100-500ms (Python) | <10ms (ONNX) | 🔴 Alto |
| Feature Store | Manual/Ad-hoc | Point-in-time correct | 🔴 Alto |
| Observabilidad | Logs básicos | Full telemetry | 🔴 Alto |
| Risk Management | Básico | Real-time limits | 🔴 Alto |
| Testing | Unit tests | Property + Chaos | 🟠 Medio |

---

### P1-14: ONNX Runtime Conversion
**Ref**: Industry Grade - Section 1.1
**Status**: 🔴 CRITICAL FOR PRODUCTION
**Impact**: 10-50x faster inference (200ms → <10ms)

**Entregables**:
```
lib/inference/
├── __init__.py
├── onnx_converter.py      # Convert SB3 PPO to ONNX
├── onnx_engine.py         # Production inference engine
└── benchmark.py           # Latency benchmarking
```

**Implementación**:
```python
# lib/inference/onnx_converter.py
class ONNXModelConverter:
    """
    Converts Stable-Baselines3 PPO to ONNX format.

    Performance targets:
    - Latency: <10ms (vs ~200ms Python)
    - Throughput: 1000+ inferences/second
    - Memory: ~50MB (vs ~500MB Python)
    """

    def convert_to_onnx(self, model_path: Path, output_path: Path) -> dict:
        # Load SB3 model
        ppo_model = PPO.load(model_path)
        policy = ppo_model.policy
        policy.eval()

        # Create dummy input matching observation space
        dummy_input = torch.randn(1, 15)  # observation_dim

        # Export policy network
        torch.onnx.export(
            policy.mlp_extractor,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            input_names=['observation'],
            output_names=['action_logits', 'value'],
            dynamic_axes={'observation': {0: 'batch_size'}}
        )

        return self._validate_conversion(output_path, dummy_input)

# lib/inference/onnx_engine.py
class ONNXInferenceEngine:
    """Production inference with optimized ONNX Runtime"""

    def __init__(self, model_path: Path, num_threads: int = 4):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = num_threads
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(str(model_path), sess_options)

    def predict(self, observation: np.ndarray) -> tuple:
        """Single observation inference - target <10ms"""
        outputs = self.session.run(None, {'observation': observation.astype(np.float32)})
        return outputs[0], outputs[1]  # action_logits, value

    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> int:
        logits, _ = self.predict(observation)
        return int(np.argmax(logits[0])) if deterministic else self._sample(logits[0])
```

**Integración con Inference Service**:
```python
# services/inference_api/core/model_loader.py
from lib.inference.onnx_engine import ONNXInferenceEngine

class ModelLoader:
    def load_model(self, model_id: str) -> ONNXInferenceEngine:
        # Check for ONNX version first
        onnx_path = self.model_dir / f"{model_id}.onnx"
        if onnx_path.exists():
            logger.info(f"Loading ONNX model: {onnx_path}")
            return ONNXInferenceEngine(onnx_path)

        # Fallback to Python (slower)
        logger.warning(f"ONNX not found, using slow Python inference")
        return self._load_sb3_model(model_id)
```

**Métricas de Performance**:
```python
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Model inference latency',
    ['model_version', 'runtime'],  # runtime: 'onnx' or 'python'
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)
```

**Dependencias**:
- P1-11 (model hash registration) - hash both .zip and .onnx
- P1-13 (Feature Contract) - FeatureBuilder produces observations

---

### P1-15: Circuit Breakers para Trading
**Ref**: Industry Grade - Section 5.2
**Status**: 🔴 CRITICAL SAFETY
**Impact**: Protección automática contra fallas en cascada

**Entregables**:
```
lib/risk/
├── __init__.py
├── circuit_breakers.py    # Circuit breaker implementation
└── trading_breakers.py    # Pre-configured trading breakers
```

**Implementación**:
```python
# lib/risk/circuit_breakers.py
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

class BreakerState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # All trades blocked
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    name: str
    failure_threshold: int    # Failures before opening
    success_threshold: int    # Successes to close from half-open
    timeout_seconds: int      # Time before attempting recovery

class CircuitBreaker:
    """Circuit breaker pattern for trading systems"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = BreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None

    def can_execute(self) -> bool:
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.OPEN:
            if self._should_attempt_recovery():
                self._transition_to_half_open()
                return True
            return False
        if self.state == BreakerState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        return False

    def record_success(self):
        if self.state == BreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        self.failure_count = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()

# lib/risk/trading_breakers.py
class TradingCircuitBreakers:
    """Pre-configured circuit breakers for USDCOP trading"""

    def __init__(self):
        self.breakers = {
            'model_inference': CircuitBreaker(CircuitBreakerConfig(
                name='model_inference',
                failure_threshold=5,
                success_threshold=3,
                timeout_seconds=60
            )),
            'data_pipeline': CircuitBreaker(CircuitBreakerConfig(
                name='data_pipeline',
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=300
            )),
            'execution': CircuitBreaker(CircuitBreakerConfig(
                name='execution',
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=120
            )),
            'drawdown': CircuitBreaker(CircuitBreakerConfig(
                name='drawdown',
                failure_threshold=1,  # Single failure opens
                success_threshold=1,
                timeout_seconds=3600  # 1 hour cooldown
            )),
            'feature_drift': CircuitBreaker(CircuitBreakerConfig(
                name='feature_drift',
                failure_threshold=10,
                success_threshold=5,
                timeout_seconds=1800
            )),
        }

    def can_trade(self) -> tuple[bool, Optional[str]]:
        for name, breaker in self.breakers.items():
            if not breaker.can_execute():
                return False, f"Circuit breaker '{name}' is {breaker.state.value}"
        return True, None
```

**Prometheus Metrics**:
```python
CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 0.5=half-open, 1=open)',
    ['breaker_name']
)
```

**Integración**:
```python
# En inference service antes de cada trade
breakers = TradingCircuitBreakers()

async def make_trade_decision(observation: np.ndarray):
    can_trade, reason = breakers.can_trade()
    if not can_trade:
        logger.warning(f"Trading blocked: {reason}")
        return TradeDecision(action=Action.HOLD, reason=reason)

    try:
        result = await model.predict(observation)
        breakers.breakers['model_inference'].record_success()
        return result
    except Exception as e:
        breakers.breakers['model_inference'].record_failure()
        raise
```

---

### P1-16: Drift Detection System
**Ref**: Industry Grade - Section 4.2
**Status**: 🔴 CRITICAL FOR MODEL HEALTH
**Impact**: Alertas tempranas de degradación del modelo

**Entregables**:
```
lib/observability/
├── __init__.py
├── drift_detector.py       # Feature drift detection
├── prediction_drift.py     # Output drift detection
└── alerts.py               # Alert generation
```

**Implementación**:
```python
# lib/observability/drift_detector.py
from scipy import stats
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class DriftAlert:
    feature_name: str
    drift_type: str      # 'point', 'distribution', 'ks_test'
    severity: str        # 'warning', 'critical'
    current_value: float
    expected_value: float
    zscore: float
    timestamp: datetime

@dataclass
class DriftConfig:
    zscore_warning: float = 2.0
    zscore_critical: float = 3.0
    psi_warning: float = 0.1      # Population Stability Index
    psi_critical: float = 0.2
    window_size: int = 1000

class FeatureDriftDetector:
    """
    Multi-method drift detection for production ML.

    Methods:
    1. Z-score monitoring (point anomalies)
    2. Population Stability Index (distribution shift)
    3. Kolmogorov-Smirnov test (distribution comparison)
    """

    def __init__(self, training_stats: Dict[str, dict], config: DriftConfig = DriftConfig()):
        self.training_stats = training_stats  # From norm_stats.json
        self.config = config
        self.feature_buffers: Dict[str, deque] = {
            name: deque(maxlen=config.window_size)
            for name in training_stats.keys()
        }

    def check_point_drift(self, features: Dict[str, float]) -> List[DriftAlert]:
        """Check if current feature values are anomalous"""
        alerts = []

        for name, value in features.items():
            if name not in self.training_stats:
                continue

            stats = self.training_stats[name]
            mean, std = stats['mean'], stats['std']

            if std == 0:
                continue

            zscore = abs((value - mean) / std)
            self.feature_buffers[name].append(value)

            if zscore >= self.config.zscore_critical:
                alerts.append(DriftAlert(
                    feature_name=name,
                    drift_type='point',
                    severity='critical',
                    current_value=value,
                    expected_value=mean,
                    zscore=zscore,
                    timestamp=datetime.utcnow()
                ))
                FEATURE_DRIFT.labels(feature_name=name).set(zscore)
            elif zscore >= self.config.zscore_warning:
                alerts.append(DriftAlert(..., severity='warning'))

        return alerts

    def calculate_psi(self, feature_name: str, current_data: np.ndarray) -> float:
        """
        Population Stability Index.
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change, investigate
        PSI > 0.2: Significant change, action required
        """
        # Implementation as per INDUSTRY_GRADE doc
        ...
```

**Prometheus Metrics**:
```python
FEATURE_DRIFT = Gauge('feature_drift_zscore', 'Feature drift from training distribution', ['feature_name'])
FEATURE_STALENESS = Gauge('feature_staleness_seconds', 'Age of feature data', ['feature_name'])
```

**Integración con Alerting**:
```yaml
# Prometheus alerting rules
- alert: FeatureDriftCritical
  expr: feature_drift_zscore > 4
  for: 10m
  labels:
    severity: critical
  annotations:
    summary: "Critical feature drift detected"
    description: "Feature {{ $labels.feature_name }} z-score is {{ $value }}"
```

---

### P1-17: Real-Time Risk Engine
**Ref**: Industry Grade - Section 5.1
**Status**: 🔴 CRITICAL SAFETY
**Impact**: Límites de posición y pérdidas en tiempo real

**Entregables**:
```
lib/risk/
├── engine.py           # Risk engine core
├── limits.py           # Risk limit definitions
└── position_manager.py # Position tracking
```

**Implementación**:
```python
# lib/risk/engine.py
from dataclasses import dataclass
from enum import Enum

class RiskAction(Enum):
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK = "block"
    LIQUIDATE = "liquidate"

@dataclass
class RiskLimits:
    # Position limits
    max_position_size_usd: float = 100_000
    max_position_pct_portfolio: float = 0.10  # 10%

    # Loss limits
    max_daily_loss_usd: float = 5_000
    max_daily_loss_pct: float = 0.05  # 5%
    max_drawdown_pct: float = 0.10    # 10%

    # Trade limits
    max_trades_per_hour: int = 10
    max_trades_per_day: int = 50
    min_time_between_trades_seconds: int = 60

    # Circuit breakers
    consecutive_losses_halt: int = 5
    volatility_halt_zscore: float = 4.0

@dataclass
class RiskCheckResult:
    action: RiskAction
    reason: Optional[str]
    adjusted_size: Optional[float]
    limits_triggered: List[str]

class RiskEngine:
    """Real-time risk management for trading systems"""

    def __init__(self, limits: RiskLimits, portfolio_value: float):
        self.limits = limits
        self.portfolio_value = portfolio_value
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.peak_value = portfolio_value

    def check_trade(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        volatility: float
    ) -> RiskCheckResult:
        """Comprehensive pre-trade risk check"""
        limits_triggered = []
        adjusted_size = size_usd

        # 1. Position size check
        if size_usd > self.limits.max_position_size_usd:
            limits_triggered.append('max_position_size')
            adjusted_size = self.limits.max_position_size_usd

        # 2. Portfolio concentration
        max_size_pct = self.portfolio_value * self.limits.max_position_pct_portfolio
        if size_usd > max_size_pct:
            adjusted_size = min(adjusted_size, max_size_pct)

        # 3. Daily loss limit
        if self.daily_pnl <= -self.limits.max_daily_loss_usd:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason="Daily loss limit reached",
                adjusted_size=None,
                limits_triggered=['max_daily_loss']
            )

        # 4. Drawdown limit
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if current_drawdown >= self.limits.max_drawdown_pct:
            return RiskCheckResult(
                action=RiskAction.LIQUIDATE,
                reason=f"Max drawdown ({current_drawdown:.1%}) reached",
                adjusted_size=None,
                limits_triggered=['max_drawdown']
            )

        # 5. Consecutive losses
        if self.consecutive_losses >= self.limits.consecutive_losses_halt:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason=f"Consecutive losses halt ({self.consecutive_losses})",
                adjusted_size=None,
                limits_triggered=['consecutive_losses']
            )

        # 6. High volatility regime
        if volatility > self.limits.volatility_halt_zscore:
            return RiskCheckResult(
                action=RiskAction.REDUCE_SIZE,
                reason=f"High volatility regime (z={volatility:.1f})",
                adjusted_size=adjusted_size * 0.5,
                limits_triggered=['high_volatility']
            )

        return RiskCheckResult(
            action=RiskAction.ALLOW if not limits_triggered else RiskAction.REDUCE_SIZE,
            reason=None,
            adjusted_size=adjusted_size,
            limits_triggered=limits_triggered
        )

    def record_trade(self, pnl: float, is_win: bool):
        """Record completed trade for risk tracking"""
        self.daily_pnl += pnl
        self.daily_trades += 1

        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        self.portfolio_value += pnl
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        # Update Prometheus metrics
        DRAWDOWN_CURRENT.set((self.peak_value - self.portfolio_value) / self.peak_value * 100)
```

**Prometheus Metrics**:
```python
POSITION_SIZE = Gauge('position_size', 'Current position size', ['symbol', 'direction'])
UNREALIZED_PNL = Gauge('unrealized_pnl_usd', 'Unrealized P&L in USD')
DRAWDOWN_CURRENT = Gauge('drawdown_current_pct', 'Current drawdown percentage')
DRAWDOWN_MAX = Gauge('drawdown_max_pct', 'Maximum drawdown (rolling 30d)')
```

---

## Priority 2: MEDIUM (This Month)

### P2-1: Dynamic Slippage Model (ACTUALIZADO)
**Audit Ref**: ML-04
**Issue**: Slippage 2bps insuficiente para Forex 5min. Should be 5-8 bps base.

```python
def calculate_slippage(
    self,
    price: float,
    size: float,
    volatility: float,
    hour_utc: int,
    day_of_week: int
) -> float:
    """Dynamic slippage - base increased to 5 bps per audit"""
    base_slippage_bps = 5.0  # INCREASED from 2.0 per ML-04

    # Time-based multipliers
    if hour_utc < 13 or hour_utc > 21:
        base_slippage_bps *= 3.0
    if day_of_week == 4 and hour_utc > 19:
        base_slippage_bps *= 2.0
    if day_of_week == 0 and hour_utc < 15:
        base_slippage_bps *= 1.5

    vol_multiplier = 1.0 + (volatility / 0.02)
    size_multiplier = 1.0 + (size / 100000)

    effective_bps = min(base_slippage_bps * vol_multiplier * size_multiplier, 50.0)
    return price * (effective_bps / 10000)
```

---

### P2-2: Remove Dead Code, Unused Exports, and Archive Directories (ACTUALIZADO)
**Audit Ref**: RD-03, RD-06
**Update**: Now includes explicit archive cleanup per audit findings

**Phase 1: Identify dead code**
```bash
# Python dead code detection
pip install vulture
vulture . --min-confidence 80 --exclude "archive/,*_backup*"

# TypeScript unused exports
npx ts-prune

# Find unused imports
npx eslint . --rule 'no-unused-vars: error'
```

**Phase 2: Delete Archive Directories (EXPLICIT)**
Audit found ~150 files, ~4 MB of unused code in archives:

```bash
# Backend archives (~100 files, ~3.8 MB)
rm -rf archive/
rm -rf archive/airflow-dags/deprecated/
rm -rf archive/services-deprecated/

# Frontend archives (~50 components)
rm -rf usdcop-trading-dashboard/archive/
rm -rf usdcop-trading-dashboard/archive/components/charts/    # 18 unused chart variants
rm -rf usdcop-trading-dashboard/archive/components/views/     # 15+ unused views
rm -rf usdcop-trading-dashboard/archive/libs/                 # Old architecture

# Full backup (duplicate of entire dashboard)
rm -rf usdcop-trading-dashboard/_backup_20260108/

# Compiled cache files
find . -name "*.old" -path "*/.next/cache/*" -delete  # 149 webpack cache files

# Backup Python files
find . -name "*.backup" -delete
find . -name "*_old.py" -delete
find . -name "*_old.tsx" -delete
```

**Phase 3: Verify no broken imports**
```bash
# After deletion, verify build still works
cd usdcop-trading-dashboard && npm run build
cd .. && python -m pytest tests/ -x

# Check for broken imports
grep -r "from.*archive" --include="*.py" --include="*.ts" --include="*.tsx"
```

**Files to DELETE (explicit list)**:
```
archive/                                           # ~3.8 MB backend deprecated
├── airflow-dags/deprecated/                       # 30+ old DAGs
├── services-deprecated/                           # Old services
└── ...

usdcop-trading-dashboard/archive/                  # ~50 components
├── components/charts/                             # 18 chart variants
│   ├── AdvancedExportCapabilities.tsx
│   ├── AdvancedTechnicalIndicators.tsx
│   ├── AnimatedChart.tsx
│   ├── CandlestickPatternRecognition.tsx
│   ├── CanvasChart.tsx
│   ├── EnhancedTradingDashboard.tsx
│   ├── HighPerformanceVirtualizedChart.tsx (x3)
│   ├── InteractiveChart.tsx
│   ├── MLPredictionOverlay.tsx
│   ├── MultiTimeframeAnalysis.tsx
│   ├── ProfessionalTradingTerminal.tsx (x2)
│   └── VolumeProfile.tsx
├── components/views/                              # 15+ views
│   ├── PipelineStatus.backup.tsx
│   ├── PipelineStatus_old.tsx
│   ├── EnhancedTradingTerminal.tsx
│   └── UnifiedTradingTerminal.tsx
└── libs/                                          # Old shared libs

usdcop-trading-dashboard/_backup_20260108/         # Full duplicate
```

**Estimated cleanup**: ~4.5 MB, ~150+ files

---

### P2-3: Consolidate Constants
**Audit Ref**: RD-05

Create `shared/constants.json` loaded by Python and TypeScript.

---

### P2-4: Feature Parity Tests
**Audit Ref**: FE-01, PL-05

---

### P2-5: Optimize Trade Filtering Performance
**Audit Ref**: VZ-05

Binary search instead of O(n) filter.

---

### P2-6: Model Registry Pattern
**Audit Ref**: TL-01, CC-01

---

### P2-7: Documentar V19 vs V20
**Audit Ref**: ML-05

| Metrica | V19 | V20 | Delta | p-value |
|---------|-----|-----|-------|---------|
| Sharpe | ? | ? | ? | ? |
| Max DD | ? | ? | ? | - |
| Win Rate | ? | ? | ? | - |

---

### P2-8: Reconciliación Datos Históricos
**Audit Ref**: CD-02

---

### P2-9: Pipeline CI/CD
**Audit Ref**: TL-04

---

### P2-10: Performance Benchmarks
**Audit Ref**: VZ-05

---

### P2-11: Structured Logging
**Audit Ref**: TL-02, RM-04

---

### P2-12: Data Quality Flags
**Audit Ref**: CD-03, CD-04

---

### P2-13: Documentation Updates
**Audit Ref**: FE-06, PL-02

---

### P2-14: Drift Detection
**Audit Ref**: FE-01

---

### P2-15: Dataset Checksums
**Audit Ref**: CD-02, TL-03

---

### P2-16: Cold-Start Warmup
**Audit Ref**: RM-01

---

### P2-17: ModelFactory Pattern (NUEVO)
**Audit Ref**: CC-01
**Issue**: Creación ad-hoc de modelos PPO/SAC.

```python
# src/core/factories/model_factory.py
from abc import ABC, abstractmethod
from stable_baselines3 import PPO, SAC

class ModelFactory(ABC):
    @abstractmethod
    def create(self, config: dict) -> object:
        pass

class PPOFactory(ModelFactory):
    def create(self, config: dict) -> PPO:
        return PPO(
            "MlpPolicy",
            config['env'],
            learning_rate=config.get('learning_rate', 3e-4),
            n_steps=config.get('n_steps', 2048),
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            ent_coef=config.get('ent_coef', 0.01),
            verbose=config.get('verbose', 1),
        )

class SACFactory(ModelFactory):
    def create(self, config: dict) -> SAC:
        return SAC(
            "MlpPolicy",
            config['env'],
            learning_rate=config.get('learning_rate', 3e-4),
            buffer_size=config.get('buffer_size', 100000),
            verbose=config.get('verbose', 1),
        )

MODEL_FACTORIES = {
    'ppo': PPOFactory(),
    'sac': SACFactory(),
}

def create_model(model_type: str, config: dict) -> object:
    if model_type not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_FACTORIES[model_type].create(config)
```

---

### P2-18: Dependency Injection / ServiceContainer (NUEVO)
**Audit Ref**: CC-06
**Issue**: Sin inyección de dependencias.

```python
# src/core/container.py
from typing import TypeVar, Callable, Dict, Any

T = TypeVar('T')

class ServiceContainer:
    """Simple DI container for managing dependencies"""

    _services: Dict[str, Any] = {}
    _factories: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, service: Any):
        """Register a singleton service"""
        cls._services[name] = service

    @classmethod
    def register_factory(cls, name: str, factory: Callable):
        """Register a factory for lazy instantiation"""
        cls._factories[name] = factory

    @classmethod
    def get(cls, name: str) -> Any:
        """Get a service by name"""
        if name in cls._services:
            return cls._services[name]
        if name in cls._factories:
            cls._services[name] = cls._factories[name]()
            return cls._services[name]
        raise KeyError(f"Service '{name}' not registered")

    @classmethod
    def reset(cls):
        """Reset container (useful for testing)"""
        cls._services.clear()
        cls._factories.clear()

# Usage
def setup_container():
    from services.inference_api.config import get_settings
    from lib.database import get_connection

    ServiceContainer.register('settings', get_settings())
    ServiceContainer.register_factory('db', get_connection)
    ServiceContainer.register_factory('model', lambda: load_model(
        ServiceContainer.get('settings').full_model_path
    ))

# In code
settings = ServiceContainer.get('settings')
model = ServiceContainer.get('model')
```

---

### P2-19: DAG Versioning (NUEVO)
**Audit Ref**: PL-06
**Issue**: Sin versionamiento de DAGs.

```python
# airflow/dags/versioned_dag.py

DAG_VERSION = "3.2.1"  # Semantic versioning

default_args = {
    'owner': 'usdcop',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'tags': [f'v{DAG_VERSION}', 'l1', 'features'],
}

dag = DAG(
    'l1_feature_refresh',
    default_args=default_args,
    description=f'Feature refresh DAG v{DAG_VERSION}',
    schedule_interval='*/5 13-17 * * 1-5',
    catchup=False,
    tags=['production', f'v{DAG_VERSION}'],
)

# Log version at start
def log_dag_version(**context):
    import logging
    logging.info(f"Running DAG version: {DAG_VERSION}")
    logging.info(f"Execution date: {context['execution_date']}")

start_task = PythonOperator(
    task_id='log_version',
    python_callable=log_dag_version,
    dag=dag,
)
```

---

### P2-20: Trade History Synchronization Fix + Deprecate Dual Filtering (ACTUALIZADO)
**Audit Ref**: VZ-03, VZ-05
**Update**: Now includes deprecation of dual filtering system (replayTrades vs replayVisibleTradeIds)

**Issue 1**: Lag entre visualizaciones y trades no sincronizados
**Issue 2**: Dual filtering system causes inconsistencies between components

```typescript
// hooks/useReplaySynchronization.ts

interface SyncState {
  chartTime: Date;
  tradesTime: Date;
  equityTime: Date;
  isInSync: boolean;
  maxDrift: number;
}

export function useReplaySynchronization(
  chartTime: Date,
  trades: ReplayTrade[],
  equityPoints: EquityPoint[]
): SyncState {
  const [syncState, setSyncState] = useState<SyncState>({
    chartTime,
    tradesTime: chartTime,
    equityTime: chartTime,
    isInSync: true,
    maxDrift: 0,
  });

  useEffect(() => {
    // Find latest trade before chart time
    const latestTrade = trades
      .filter(t => new Date(t.timestamp) <= chartTime)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())[0];

    const tradesTime = latestTrade ? new Date(latestTrade.timestamp) : chartTime;

    // Find latest equity point
    const latestEquity = equityPoints
      .filter(p => new Date(p.timestamp) <= chartTime)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())[0];

    const equityTime = latestEquity ? new Date(latestEquity.timestamp) : chartTime;

    // Calculate drift
    const drifts = [
      Math.abs(chartTime.getTime() - tradesTime.getTime()),
      Math.abs(chartTime.getTime() - equityTime.getTime()),
    ];
    const maxDrift = Math.max(...drifts);

    // Warn if drift > 5 minutes
    const MAX_ACCEPTABLE_DRIFT_MS = 5 * 60 * 1000;
    if (maxDrift > MAX_ACCEPTABLE_DRIFT_MS) {
      console.warn(`[Sync] High drift detected: ${maxDrift}ms`);
    }

    setSyncState({
      chartTime,
      tradesTime,
      equityTime,
      isInSync: maxDrift < MAX_ACCEPTABLE_DRIFT_MS,
      maxDrift,
    });
  }, [chartTime, trades, equityPoints]);

  return syncState;
}
```

**Part 2: Deprecate Dual Filtering System**

**Problem**: Two systems exist for filtering visible trades:
1. `replayTrades` (new) - Used in some components
2. `replayVisibleTradeIds` (legacy) - Used in other components

This causes:
- Potential desync between chart and table
- Confusing codebase with two ways to do same thing
- Market hours filtering applied inconsistently

**Solution: Consolidate to single source (replayTrades)**

```typescript
// hooks/useReplay.ts - SINGLE SOURCE OF TRUTH
export function useReplay() {
  // ...existing code...

  // SINGLE filtered trades array - this is the ONLY source
  const visibleTrades = useMemo(() => {
    if (!replayData?.trades) return [];

    return replayData.trades.filter((trade) => {
      const tradeTime = new Date(trade.timestamp || trade.entry_time || '');

      // Time filter - before current replay time
      if (tradeTime > state.currentDate) return false;

      // Market hours filter (consistent across all components)
      if (!isWithinMarketHours(tradeTime)) return false;

      return true;
    });
  }, [replayData?.trades, state.currentDate]);

  // DEPRECATED: Remove replayVisibleTradeIds
  // const replayVisibleTradeIds = useMemo(() => {
  //   return new Set(visibleTrades.map(t => t.trade_id));
  // }, [visibleTrades]);

  return {
    // ...existing returns...
    visibleTrades,  // Use this everywhere
    // replayVisibleTradeIds,  // REMOVE THIS
  };
}
```

**Files to update**:
```typescript
// 1. TradesTable.tsx - Remove replayVisibleTradeIds usage
// BEFORE:
if (isReplayMode && replayVisibleTradeIds) {
  return data.trades.filter((trade) =>
    replayVisibleTradeIds.has(String(trade.trade_id))
  );
}

// AFTER:
if (isReplayMode && replayTrades) {
  return replayTrades;  // Already filtered by useReplay
}

// 2. TradingChartWithSignals.tsx - Use replayTrades only
// BEFORE:
const signals = useMemo(() => {
  if (isReplayMode && replayTrades) { ... }
  // Later also checks replayVisibleTradeIds
}, [replayTrades, replayVisibleTradeIds]);

// AFTER:
const signals = useMemo(() => {
  if (isReplayMode && replayTrades) {
    // replayTrades already filtered with market hours
    return replayTrades.map(trade => ({...}));
  }
}, [replayTrades]);  // Remove replayVisibleTradeIds dependency

// 3. TradingSummaryCard.tsx - Use replayTrades
// Same pattern as above

// 4. useReplay.ts exports - Remove deprecated export
export interface UseReplayReturn {
  // ...
  visibleTrades: ReplayTrade[];
  // replayVisibleTradeIds: Set<string>;  // REMOVE
}
```

**Verification test**:
```typescript
// tests/e2e/replay-filtering-consistency.spec.ts
test('all components show same trades during replay', async ({ page }) => {
  await page.goto('/dashboard');
  await startReplay(page, '2025-03-15', '2025-03-20');

  // Get trade count from table
  const tableTradeCount = await page.locator('[data-testid="trades-table"] tr').count();

  // Get trade count from chart markers
  const chartMarkerCount = await page.locator('[data-testid="trade-marker"]').count();

  // Get trade count from summary
  const summaryCount = await page.locator('[data-testid="total-trades"]').textContent();

  // All should match
  expect(chartMarkerCount).toBe(tableTradeCount);
  expect(parseInt(summaryCount || '0')).toBe(tableTradeCount);
});
```

---

### P2-21: Prometheus/Grafana Observability Stack
**Ref**: Industry Grade - Section 2.2
**Status**: 🟠 HIGH VALUE
**Impact**: Full system telemetry and alerting

**Entregables**:
```
monitoring/
├── prometheus/
│   ├── prometheus.yml        # Main config
│   ├── alerts/
│   │   ├── trading.rules     # Trading-specific alerts
│   │   ├── model.rules       # ML model alerts
│   │   └── infrastructure.rules
│   └── targets/
│       └── targets.json      # Service discovery
├── grafana/
│   ├── dashboards/
│   │   ├── trading-overview.json
│   │   ├── model-performance.json
│   │   ├── feature-drift.json
│   │   └── system-health.json
│   └── provisioning/
│       └── datasources.yml
└── docker-compose.monitoring.yml
```

**Métricas Críticas**:
```python
# lib/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary

# Trading Metrics
TRADES_TOTAL = Counter(
    'trades_total',
    'Total number of trades',
    ['symbol', 'side', 'status']
)
TRADE_PNL = Histogram(
    'trade_pnl_usd',
    'Trade P&L in USD',
    buckets=[-500, -200, -100, -50, 0, 50, 100, 200, 500, 1000]
)

# Model Metrics
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Model inference latency',
    ['model_version', 'runtime'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
)
MODEL_CONFIDENCE = Histogram(
    'model_confidence',
    'Model action confidence',
    ['model_version', 'action'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# Feature Metrics
FEATURE_VALUE = Gauge(
    'feature_value',
    'Current feature value',
    ['feature_name']
)
FEATURE_DRIFT_SCORE = Gauge(
    'feature_drift_score',
    'Feature drift score (PSI)',
    ['feature_name']
)

# System Metrics
DATA_FRESHNESS = Gauge(
    'data_freshness_seconds',
    'Age of latest market data',
    ['symbol', 'timeframe']
)
```

**Alertas de Trading**:
```yaml
# monitoring/prometheus/alerts/trading.rules
groups:
  - name: trading_alerts
    rules:
      - alert: HighDrawdown
        expr: drawdown_current_pct > 5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Drawdown exceeds 5%"

      - alert: ConsecutiveLosses
        expr: consecutive_losses > 3
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "{{ $value }} consecutive losing trades"

      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, inference_latency_seconds) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model inference p95 latency > 100ms"

      - alert: FeatureDrift
        expr: feature_drift_score > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Feature {{ $labels.feature_name }} showing drift (PSI={{ $value }})"
```

**Dependencias**:
- P1-16 (Drift Detection) - provides drift metrics
- P1-17 (Risk Engine) - provides risk metrics
- Docker/Docker Compose for deployment

---

### P2-22: Feature Store con Point-in-Time Correctness
**Ref**: Industry Grade - Section 3.1
**Status**: 🟠 HIGH VALUE
**Impact**: Elimina training-serving skew, garantiza reproducibilidad

**Entregables**:
```
lib/feature_store/
├── __init__.py
├── store.py              # Main FeatureStore class
├── point_in_time.py      # PIT join logic
├── registry.py           # Feature definitions
└── validators.py         # Data quality validators
```

**Implementación**:
```python
# lib/feature_store/store.py
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy import text

class FeatureStore:
    """
    Point-in-time correct feature store.

    Guarantees:
    - No future data leakage
    - Consistent features between training and inference
    - Full reproducibility with temporal queries
    """

    def __init__(self, db_engine):
        self.engine = db_engine
        self.feature_tables = self._discover_feature_tables()

    def get_features_at(
        self,
        entity_id: str,
        feature_names: List[str],
        as_of: datetime
    ) -> Dict[str, float]:
        """
        Get feature values as they were known at a specific point in time.

        Args:
            entity_id: Entity identifier (e.g., 'USDCOP')
            feature_names: List of feature names to retrieve
            as_of: Point-in-time timestamp

        Returns:
            Dictionary of feature_name -> value
        """
        features = {}

        for feature_name in feature_names:
            table = self.feature_tables.get(feature_name)
            if not table:
                raise ValueError(f"Unknown feature: {feature_name}")

            # Point-in-time query - get latest value BEFORE as_of
            query = text(f"""
                SELECT value
                FROM {table}
                WHERE entity_id = :entity_id
                  AND timestamp <= :as_of
                ORDER BY timestamp DESC
                LIMIT 1
            """)

            result = self.engine.execute(
                query,
                {"entity_id": entity_id, "as_of": as_of}
            ).fetchone()

            features[feature_name] = result[0] if result else None

        return features

    def get_training_dataset(
        self,
        entity_id: str,
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime,
        label_column: str
    ) -> pd.DataFrame:
        """
        Generate training dataset with point-in-time correct features.

        Uses temporal joins to ensure no future data leakage.
        """
        # Get label timestamps
        labels_df = self._get_labels(entity_id, start_date, end_date, label_column)

        # For each label timestamp, get features as-of that time
        rows = []
        for _, label_row in labels_df.iterrows():
            features = self.get_features_at(
                entity_id,
                feature_names,
                label_row['timestamp']
            )
            features['label'] = label_row[label_column]
            features['timestamp'] = label_row['timestamp']
            rows.append(features)

        return pd.DataFrame(rows)

    def validate_no_leakage(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        timestamp_column: str = 'timestamp'
    ) -> Dict[str, bool]:
        """
        Validate that no features contain future information.

        Returns dict of feature_name -> is_valid
        """
        results = {}

        for feature in feature_columns:
            # Check if feature value changes correlate with future events
            # (simplified - actual implementation would be more sophisticated)
            correlation = df[feature].shift(-1).corr(df[feature])
            results[feature] = abs(correlation) < 0.95  # High autocorrelation = potential leakage

        return results
```

**SQL Schema**:
```sql
-- Feature value storage with temporal tracking
CREATE TABLE feature_values (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Composite index for PIT queries
    UNIQUE(entity_id, feature_name, timestamp)
);

CREATE INDEX idx_feature_pit
    ON feature_values(entity_id, feature_name, timestamp DESC);

-- Feature definitions registry
CREATE TABLE feature_definitions (
    feature_name VARCHAR(100) PRIMARY KEY,
    description TEXT,
    data_type VARCHAR(50),
    source_table VARCHAR(100),
    aggregation_type VARCHAR(50),  -- 'latest', 'avg', 'sum', etc.
    lookback_period INTERVAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Dependencias**:
- P1-13 (Feature Contract) - defines canonical feature names
- Database schema migration

---

### P2-23: Walk-Forward Backtesting Framework
**Ref**: Industry Grade - Section 4.1
**Status**: 🟠 HIGH VALUE
**Impact**: Statistically valid out-of-sample testing

**Entregables**:
```
lib/backtesting/
├── __init__.py
├── walk_forward.py       # Walk-forward engine
├── splits.py             # Time-series splits
├── metrics.py            # Performance metrics
└── report.py             # Report generation
```

**Implementación**:
```python
# lib/backtesting/walk_forward.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Callable, Any
import pandas as pd
import numpy as np

@dataclass
class WalkForwardConfig:
    """Walk-forward backtest configuration"""
    train_period_days: int = 252      # 1 year training
    test_period_days: int = 21        # 1 month testing
    step_days: int = 21               # Monthly rolling
    min_train_samples: int = 500      # Minimum training samples
    embargo_days: int = 5             # Gap between train/test

@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_trades: int
    model_params: dict

class WalkForwardEngine:
    """
    Statistically rigorous walk-forward backtesting.

    Key features:
    - No look-ahead bias (strict temporal separation)
    - Embargo period to prevent leakage
    - OOS validation on each fold
    - Aggregate statistics across folds
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        model_trainer: Callable[[pd.DataFrame], Any],
        model_evaluator: Callable[[Any, pd.DataFrame], dict]
    ):
        self.config = config
        self.train_model = model_trainer
        self.evaluate_model = model_evaluator

    def generate_splits(
        self,
        data: pd.DataFrame,
        date_column: str = 'timestamp'
    ) -> List[tuple]:
        """Generate walk-forward train/test splits"""
        splits = []

        dates = pd.to_datetime(data[date_column])
        min_date = dates.min()
        max_date = dates.max()

        train_start = min_date

        while True:
            train_end = train_start + timedelta(days=self.config.train_period_days)
            test_start = train_end + timedelta(days=self.config.embargo_days)
            test_end = test_start + timedelta(days=self.config.test_period_days)

            if test_end > max_date:
                break

            # Check minimum training samples
            train_mask = (dates >= train_start) & (dates < train_end)
            if train_mask.sum() < self.config.min_train_samples:
                train_start += timedelta(days=self.config.step_days)
                continue

            splits.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

            train_start += timedelta(days=self.config.step_days)

        return splits

    def run(self, data: pd.DataFrame) -> List[WalkForwardResult]:
        """Execute walk-forward backtest"""
        splits = self.generate_splits(data)
        results = []

        for fold_id, split in enumerate(splits):
            print(f"Fold {fold_id + 1}/{len(splits)}: "
                  f"{split['test_start'].date()} - {split['test_end'].date()}")

            # Get train/test data
            train_mask = (
                (data['timestamp'] >= split['train_start']) &
                (data['timestamp'] < split['train_end'])
            )
            test_mask = (
                (data['timestamp'] >= split['test_start']) &
                (data['timestamp'] < split['test_end'])
            )

            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()

            # Train model
            model = self.train_model(train_data)

            # Evaluate on train (in-sample)
            train_metrics = self.evaluate_model(model, train_data)

            # Evaluate on test (out-of-sample)
            test_metrics = self.evaluate_model(model, test_data)

            results.append(WalkForwardResult(
                fold_id=fold_id,
                train_start=split['train_start'],
                train_end=split['train_end'],
                test_start=split['test_start'],
                test_end=split['test_end'],
                train_sharpe=train_metrics.get('sharpe', 0),
                test_sharpe=test_metrics.get('sharpe', 0),
                test_return=test_metrics.get('total_return', 0),
                test_trades=test_metrics.get('trade_count', 0),
                model_params=getattr(model, 'params_', {})
            ))

        return results

    def aggregate_results(self, results: List[WalkForwardResult]) -> dict:
        """Aggregate statistics across all folds"""
        test_sharpes = [r.test_sharpe for r in results]
        test_returns = [r.test_return for r in results]

        return {
            'n_folds': len(results),
            'mean_oos_sharpe': np.mean(test_sharpes),
            'std_oos_sharpe': np.std(test_sharpes),
            'median_oos_sharpe': np.median(test_sharpes),
            'mean_oos_return': np.mean(test_returns),
            'win_rate_folds': np.mean([r > 0 for r in test_returns]),
            'total_oos_trades': sum(r.test_trades for r in results),
            't_statistic': self._calculate_t_stat(test_sharpes),
        }

    def _calculate_t_stat(self, values: List[float]) -> float:
        """Calculate t-statistic for mean > 0"""
        if len(values) < 2:
            return 0.0
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        return mean / (std / np.sqrt(len(values))) if std > 0 else 0.0
```

**Uso**:
```python
# Example usage
config = WalkForwardConfig(
    train_period_days=252,  # 1 year
    test_period_days=21,    # 1 month
    embargo_days=5          # 5 day gap
)

engine = WalkForwardEngine(
    config=config,
    model_trainer=train_ppo_model,
    model_evaluator=evaluate_trading_performance
)

results = engine.run(historical_data)
summary = engine.aggregate_results(results)

print(f"OOS Sharpe: {summary['mean_oos_sharpe']:.2f} ± {summary['std_oos_sharpe']:.2f}")
print(f"t-statistic: {summary['t_statistic']:.2f}")
```

**Dependencias**:
- P0-7 (ML Workflow) - fits within disciplined workflow
- P1-9 (Dataset Registry) - versioned data for reproducibility

---

### P2-24: Property-Based Testing con Hypothesis
**Ref**: Industry Grade - Section 4.3
**Status**: 🟠 MEDIUM VALUE
**Impact**: Catch edge cases that unit tests miss

**Entregables**:
```
tests/property/
├── __init__.py
├── conftest.py            # Hypothesis profiles
├── test_features.py       # Feature calculation properties
├── test_observations.py   # Observation builder properties
├── test_trading.py        # Trading logic properties
└── test_risk.py           # Risk engine properties
```

**Implementación**:
```python
# tests/property/test_features.py
from hypothesis import given, strategies as st, settings, assume
import numpy as np

# Feature calculation properties

@given(
    prices=st.lists(
        st.floats(min_value=100, max_value=10000, allow_nan=False),
        min_size=20,
        max_size=1000
    )
)
@settings(max_examples=200)
def test_rsi_bounded(prices):
    """RSI must always be between 0 and 100"""
    prices = np.array(prices)
    rsi = calculate_rsi(prices, period=9)
    assert 0 <= rsi <= 100, f"RSI {rsi} out of bounds"

@given(
    high=st.floats(min_value=100, max_value=10000),
    low=st.floats(min_value=100, max_value=10000),
    close=st.floats(min_value=100, max_value=10000)
)
def test_atr_non_negative(high, low, close):
    """ATR must always be non-negative"""
    assume(high >= low)  # Valid OHLC constraint
    assume(high >= close >= low or True)  # Close can be outside for some edge cases

    atr = calculate_atr_single(high, low, close)
    assert atr >= 0, f"ATR {atr} is negative"

@given(
    features=st.dictionaries(
        keys=st.sampled_from(['log_ret_5m', 'rsi_9', 'atr_pct']),
        values=st.floats(allow_nan=False, allow_infinity=False),
        min_size=1
    )
)
def test_normalization_clips_to_range(features):
    """Normalized features must be in [-5, 5] range"""
    from lib.features import normalize_feature

    for name, value in features.items():
        normalized = normalize_feature(value, name)
        assert -5.0 <= normalized <= 5.0, f"Normalized {name}={normalized} out of range"


# tests/property/test_observations.py
@given(
    position=st.floats(min_value=-1, max_value=1),
    session_progress=st.floats(min_value=0, max_value=1)
)
def test_observation_shape_invariant(position, session_progress):
    """Observation must always have shape (15,)"""
    obs = build_test_observation(position, session_progress)
    assert obs.shape == (15,), f"Shape {obs.shape} != (15,)"

@given(
    position=st.floats(allow_nan=True, allow_infinity=True),
    session_progress=st.floats(allow_nan=True, allow_infinity=True)
)
def test_observation_no_nan(position, session_progress):
    """Observations must never contain NaN or Inf"""
    obs = build_test_observation(position, session_progress)
    assert not np.isnan(obs).any(), "Observation contains NaN"
    assert not np.isinf(obs).any(), "Observation contains Inf"


# tests/property/test_risk.py
@given(
    position_size=st.floats(min_value=0, max_value=1000000),
    volatility=st.floats(min_value=0, max_value=10)
)
def test_risk_check_deterministic(position_size, volatility):
    """Same inputs must always produce same risk decision"""
    result1 = risk_engine.check_trade('USDCOP', 'buy', position_size, volatility)
    result2 = risk_engine.check_trade('USDCOP', 'buy', position_size, volatility)
    assert result1.action == result2.action, "Risk check not deterministic"

@given(size=st.floats(min_value=0))
def test_reduced_size_less_than_original(size):
    """When reducing size, result must be <= original"""
    assume(size > 0)  # Zero doesn't make sense
    result = risk_engine.check_trade('USDCOP', 'buy', size, volatility=2.0)
    if result.adjusted_size is not None:
        assert result.adjusted_size <= size, \
            f"Adjusted {result.adjusted_size} > original {size}"
```

**Configuración Hypothesis**:
```python
# tests/property/conftest.py
from hypothesis import settings, Verbosity, Phase

# Profile for CI (faster)
settings.register_profile("ci", max_examples=50, deadline=None)

# Profile for local development (thorough)
settings.register_profile("dev", max_examples=200, deadline=500)

# Profile for finding edge cases (very thorough)
settings.register_profile("exhaustive",
    max_examples=1000,
    phases=[Phase.generate, Phase.shrink],
    deadline=None
)

# Load based on environment
import os
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

**Integración CI**:
```yaml
# .github/workflows/test.yml
property-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Run property tests
      env:
        HYPOTHESIS_PROFILE: ci
      run: |
        pytest tests/property/ -v --hypothesis-show-statistics
```

**Dependencias**:
- P1-13 (Feature Contract) - test feature contract invariants
- P1-16 (Drift Detection) - test drift detection properties
- P1-17 (Risk Engine) - test risk check properties

---

## Priority 3: LOW - Institutional Features (Future)

> **Objetivo**: Capacidades de nivel institucional para escalar el sistema
> **Timeline**: Post-producción, una vez estabilizado el core

### P3-1: Model Gateway con A/B Testing
**Ref**: Industry Grade - Section 6.1
**Status**: ⚪ FUTURE

**Arquitectura**:
```
┌─────────────────────────────────────────────────────────────┐
│                    Model Gateway                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────────┐    ┌──────────────────┐   │
│  │ Router  │───▶│ Load        │───▶│ Model Pool       │   │
│  │ (A/B)   │    │ Balancer    │    │ (Champion/       │   │
│  │         │    │             │    │  Challengers)    │   │
│  └─────────┘    └─────────────┘    └──────────────────┘   │
│       │                                    │               │
│       ▼                                    ▼               │
│  ┌─────────┐                    ┌──────────────────┐      │
│  │ Shadow  │                    │ Metrics          │      │
│  │ Mode    │                    │ Collector        │      │
│  └─────────┘                    └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**Capacidades**:
- A/B testing con traffic splitting (90/10, 80/20, etc.)
- Shadow mode: nuevo modelo corre sin afectar producción
- Canary deployments con rollback automático
- Champion/Challenger framework

---

### P3-2: TensorRT Optimization
**Ref**: Industry Grade - Section 1.2
**Status**: ⚪ FUTURE (requires GPU)

**Beneficios**:
- 2-5x adicional sobre ONNX
- Sub-millisecond inference
- FP16/INT8 quantization

**Dependencias**: P1-14 (ONNX) primero

---

### P3-3: Chaos Engineering
**Ref**: Industry Grade - Section 5.4
**Status**: ⚪ FUTURE

**Experimentos**:
- Network latency injection
- Service failure simulation
- Database connection drops
- API rate limiting

**Herramientas**: Chaos Monkey, Gremlin, Litmus

---

### P3-4: Kubernetes + GitOps Deployment
**Ref**: Industry Grade - Section 6.2
**Status**: ⚪ FUTURE

**Stack**:
- Kubernetes (EKS/GKE/AKS)
- ArgoCD para GitOps
- Helm charts para packaging
- Istio para service mesh

---

### P3-5: DVC (Data Version Control)
**Ref**: Industry Grade - Section 3.3
**Status**: ⚪ FUTURE

**Capacidades**:
- Version control para datasets grandes
- Reproducibilidad completa de experimentos
- Pipeline DAGs con cache
- Remote storage (S3/GCS)

---

### P3-6: Model Quantization (INT8)
**Ref**: Industry Grade - Section 1.3
**Status**: ⚪ FUTURE

**Beneficios**:
- 2-4x reducción tamaño modelo
- Faster inference en CPU
- Lower memory footprint

**Trade-offs**: Pequeña pérdida de precisión (típicamente <1%)

---

## Rollback Strategy

### Si P0-8 (features_snapshot migration) falla:
```sql
-- Rollback
ALTER TABLE trades_history DROP COLUMN IF EXISTS features_snapshot;
ALTER TABLE trades_history DROP COLUMN IF EXISTS model_hash;
```

### Si P0-9 (look-ahead fix) cambia resultados:
```python
# Keep both calculations temporarily
df['regime_old'] = calculate_regime_old(df)  # Legacy
df['regime_new'] = calculate_regime_safe(df)  # New

# Compare and document differences
diff = (df['regime_old'] != df['regime_new']).sum()
print(f"Regime differences: {diff} ({diff/len(df)*100:.2f}%)")
```

---

## Implementation Order (ACTUALIZADO v3.5)

```
Week 1: Critical Fixes (P0) - 11 items
├── Day 1: P0-7 AUDIT (ML Workflow)
├── Day 2: P0-8 (features_snapshot SQL), P0-9 (look-ahead fix)
├── Day 3: P0-10 (ffill limit), P0-11 (merge_asof fix)
├── Day 4: P0-1 (norm_stats), P0-2 (thresholds), P0-3 (ADX)
└── Day 5: P0-4 (passwords), P0-5 (MIN_TICK), P0-6 (model ID)

Week 2: High Priority (P1) - 13 items (P1-3 superseded)
├── Day 1-2: P1-9 (dataset registry), P1-10 (config YAML), P1-11 (model hash)
├── Day 3: P1-1 (ExternalTaskSensor), P1-2 (metadata + bid_ask_spread), P1-12 (parity test)
├── Day 4: P1-13 (Feature Contract Pattern) ← SUPERSEDES P1-3
├── Day 4: P1-4 (dual filtering)
├── Day 5: P1-5 (feature order), P1-6 (confidence fix)
└── Day 5: P1-7 (macro source), P1-8 (paper trading)

Week 3: Industry Grade (P1) - 4 items 🆕
├── Day 1: P1-14 (ONNX Runtime Conversion) ← CRITICAL
├── Day 2: P1-15 (Circuit Breakers) ← SAFETY
├── Day 3: P1-16 (Drift Detection System)
└── Day 4: P1-17 (Real-Time Risk Engine)

Week 4-5: Medium Priority (P2) - 24 items
├── Day 1: P2-1 (Slippage 5bps base), P2-2 (archive cleanup)
├── Day 2: P2-17 (ModelFactory), P2-18 (ServiceContainer DI)
├── Day 3: P2-19 (DAG versioning), P2-20 (Sync + deprecate dual filter)
├── Day 4: P2-21 (Prometheus/Grafana Observability) 🆕
├── Day 5: P2-22 (Feature Store PIT) 🆕
├── Day 6: P2-23 (Walk-Forward Backtesting) 🆕
├── Day 7: P2-24 (Property-Based Testing) 🆕
└── Days 8-14: Remaining P2 items (P2-3 to P2-16)

Future (P3): Institutional Features - 6 items
├── P3-1: Model Gateway + A/B Testing
├── P3-2: TensorRT Optimization (GPU)
├── P3-3: Chaos Engineering
├── P3-4: Kubernetes + GitOps
├── P3-5: DVC (Data Version Control)
└── P3-6: Model Quantization (INT8)

Ongoing:
├── Paper Trading (3 months)
├── Weekly backtest vs paper comparison
├── Monthly walk-forward validation (P2-23)
└── Quarterly model retraining
```

**Note**: P1-3 is now superseded by P1-13. See ARCHITECTURE_CONTRACTS.md for the comprehensive Feature Contract Pattern implementation.

**Industry Grade Priority**:
1. P1-14 (ONNX) - 10-50x inference speedup, enables real-time trading
2. P1-15 (Circuit Breakers) - Essential safety mechanism
3. P1-16 (Drift Detection) - Early warning for model degradation
4. P1-17 (Risk Engine) - Automated position/drawdown limits

---

## Audit Checklist Mapping

### RM - Replay Mode
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| RM-01 | Feature reproduction | P1-3, P2-16, **P1-13 (Contract)** | ✅ COMPLETE |
| RM-02 | Frozen weights | Already OK | ✅ |
| RM-03 | Features snapshot | **P0-8**, **P1-13 (Contract)** | ✅ COMPLETE |
| RM-04 | Logging train vs data | P2-11 | ✅ Mapped |
| RM-05 | Look-ahead prevention | **P0-9** | ✅ Mapped |
| RM-06 | Backtest vs simulation | P1-12, **P1-13 (Contract)** | ✅ COMPLETE |

### FE - Features
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| FE-01 | Feature identity | P1-3, P2-4, **P1-13 (Contract)** | ✅ COMPLETE |
| FE-02 | Schema validation | P1-5, **P1-13 (Contract)** | ✅ COMPLETE |
| FE-03 | Order book/spread | **P1-2 (updated)** | ✅ NOW MAPPED |
| FE-04 | Market regime | P0-3 | ✅ Mapped |
| FE-05 | Regime look-ahead | **P0-9** | ✅ Mapped |
| FE-06 | v19/v20 docs | P2-7, **P1-13 (Contract)** | ✅ COMPLETE |

### PL - Pipelines
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| PL-01 | Pipeline count | P2-13 | ✅ Mapped |
| PL-02 | Dependency diagram | P2-13 | ✅ Mapped |
| PL-03 | Train/inference align | P0-1, **P1-13 (Contract)** | ✅ COMPLETE |
| PL-04 | Backtest components | P1-12, **P1-13 (Contract)** | ✅ COMPLETE |
| PL-05 | Train vs replay | P1-12, **P1-13 (Contract)** | ✅ COMPLETE |
| PL-06 | DAG versioning | **P2-19** | ✅ Mapped |

### FB - Frontend/Backend
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| FB-01 | FE/BE communication | Already OK | ✅ |
| FB-02 | Model selection | P0-6 | Mapped |
| FB-03 | Duplicate calcs | P1-4 | Mapped |
| FB-04 | API contract | Already OK | ✅ |
| FB-05 | State sync | P2-20 | Mapped |

### VZ - Visualizations
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| VZ-01 | BUY/SELL icons + confidence | **P1-6 (updated)** | ✅ NOW MAPPED |
| VZ-02 | Equity curve | Already OK | ✅ |
| VZ-03 | Trade history sync + dual filter | **P2-20 (updated)** | ✅ NOW MAPPED |
| VZ-04 | Metrics dynamic | Already OK | ✅ |
| VZ-05 | Visualization lag | **P2-20 (updated)** | ✅ Mapped |
| VZ-06 | Icons per model + confidence | **P1-6 (updated)** | ✅ NOW MAPPED |

### ML - ML/Trading
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| ML-01 | Workflow | P0-7 | Mapped |
| ML-02 | Hyperparams OOS | P0-7 | Mapped |
| ML-03 | Data leakage | P0-9, P0-10 | Mapped |
| ML-04 | Slippage forex | P2-1 (5bps) | Mapped |
| ML-05 | v19/v20 metrics | P2-7 | Mapped |
| ML-06 | val/test OOS | P0-7 | Mapped |

### CC - Clean Code
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| CC-01 | ModelFactory | **P2-17** | 🔴 NEW |
| CC-02 | Contracts | Already OK | ✅ |
| CC-03 | SOLID | P2-17, P2-18 | Mapped |
| CC-04 | Layer separation | P2-17 | Mapped |
| CC-05 | Config external | **P1-10** | 🔴 NEW |
| CC-06 | DI | **P2-18** | 🔴 NEW |

### RD - Redundancy
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| RD-01 | Pipeline code dup | P1-3, **P1-13 (Contract)** | ✅ COMPLETE |
| RD-02 | Features replicated | P1-3, **P1-13 (Contract)** | ✅ COMPLETE |
| RD-03 | UI components dup (archive) | **P2-2 (updated)** | ✅ NOW EXPLICIT |
| RD-04 | Trading logic dup | P1-3 | ✅ Mapped |
| RD-05 | Config redundant | P2-3 | ✅ Mapped |
| RD-06 | Dead code + archive cleanup | **P2-2 (updated)** | ✅ NOW EXPLICIT |
| RD-07 | Utils centralized | P1-3, **P1-13 (Contract)** | ✅ COMPLETE |

### CD - Data Consistency
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| CD-01 | Same pipeline | P1-7 | ✅ Mapped |
| CD-02 | Dataset version | **P1-9** | ✅ Mapped |
| CD-03 | No leakage (ffill + merge_asof) | **P0-10, P0-11 (NEW)** | ✅ NOW COMPLETE |
| CD-04 | Deterministic | P1-7, P2-12 | ✅ Mapped |

### TL - Traceability
| ID | Issue | Plan Item | Status |
|----|-------|-----------|--------|
| TL-01 | Model hash | **P0-8, P1-11** | 🔴 NEW |
| TL-02 | Decision logging | P2-11 | Mapped |
| TL-03 | Reproducibility | P2-15 | Mapped |
| TL-04 | Config audit | P2-9 | Mapped |

---

## Success Metrics (ACTUALIZADO)

| Metric | Audit Score | Target |
|--------|-------------|--------|
| Replay Mode | 4/10 | 8/10 |
| Features | 5/10 | 9/10 |
| Pipelines | 4/10 | 8/10 |
| Frontend/Backend | 6/10 | 9/10 |
| Visualizations | 6/10 | 9/10 |
| ML/Trading | 5/10 | 8/10 |
| Clean Code | 5/10 | 8/10 |
| Redundancy | 4/10 | 8/10 |
| Data Consistency | 5/10 | 9/10 |
| Traceability | 4/10 | 9/10 |
| **PROMEDIO** | **4.8/10** | **8.5/10** |

---

## Architecture After All Fixes

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    USDCOP Trading Platform v3 - Production Ready             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    ML WORKFLOW (Disciplined)                          │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   │   │
│  │  │Exploration│───▶│Validation│───▶│  Test    │───▶│Paper Trading │   │   │
│  │  │(20% hold) │    │(1 pass)  │    │(CLEAN!)  │    │(3 months)    │   │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    DATA LAYER (Traceable)                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │   │
│  │  │Dataset       │  │Model         │  │Config                      │ │   │
│  │  │Registry      │  │Registry      │  │(YAML external)             │ │   │
│  │  │(checksums)   │  │(hashes)      │  │                            │ │   │
│  │  └──────────────┘  └──────────────┘  └────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────────────┐    │
│  │   L0 DAG    │───▶│   L1 DAG    │───▶│   L5 Inference DAG           │    │
│  │ (v3.2.1)    │    │ (v3.2.1)    │    │   (Real ADX, 5bps slippage)  │    │
│  │ +Sensor     │    │ +Sensor     │    │   (No look-ahead bias)       │    │
│  └─────────────┘    └─────────────┘    └──────────────────────────────┘    │
│         │                  │                      │                         │
│         ▼                  ▼                      ▼                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              SERVICE CONTAINER (DI)                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │  │
│  │  │ModelFactory │  │FeatureLib   │  │ConfigLoader                 │  │  │
│  │  │(PPO/SAC)    │  │(Shared)     │  │(v20_config.yaml)            │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                  │                      │                         │
│         ▼                  ▼                      ▼                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    trades_history (Full Traceability)                 │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ features_snapshot | model_hash | model_version | config_snapshot│ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                                         │                         │
│         └────────────────────┬────────────────────┘                         │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Next.js Frontend (Synchronized)                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │  │
│  │  │  useReplay  │  │ replayApi   │  │ TradingChart                │  │  │
│  │  │ +SyncHook   │  │ (dynamic)   │  │ (no lag, quality flags)     │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    OBSERVABILITY                                      │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────────┐   │  │
│  │  │Structured│  │Drift Alerts  │  │Data Qual │  │Parity Tests    │   │  │
│  │  │ Logging  │  │              │  │ Flags    │  │(CI/CD)         │   │  │
│  │  └──────────┘  └──────────────┘  └──────────┘  └────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Gap Analysis Summary (v3.1)

### Gaps Fixed in This Update

| Gap | Issue | Fix Applied |
|-----|-------|-------------|
| CD-03 | Merge AsOf tolerance | **P0-11 (NEW)** - Remove 1-day tolerance |
| FE-03 | Order book/spread | **P1-2 (updated)** - Add bid_ask_spread to snapshot |
| VZ-01/06 | Confidence hardcoded | **P1-6 (updated)** - Use actual trade confidence |
| RD-03/06 | Archive not explicit | **P2-2 (updated)** - Explicit directory deletion |
| VZ-03 | Dual filtering | **P2-20 (updated)** - Deprecate replayVisibleTradeIds |

### Final Coverage

| Metric | Value |
|--------|-------|
| Total audit questions | 56 |
| Mapped to plan items | 56 (100%) |
| P0 Critical items | 11 |
| P1 High items | 17 (13 original + 4 industry-grade) |
| P2 Medium items | 24 (20 original + 4 industry-grade) |
| P3 Future items | 6 (institutional features) |
| Total implementation items | **58** |
| Coverage | **100%** |

**Industry-Grade Additions (v3.5)**:
| Priority | Items | Focus |
|----------|-------|-------|
| P1-14 to P1-17 | 4 | Performance, Safety, Monitoring, Risk |
| P2-21 to P2-24 | 4 | Observability, Feature Store, Testing |
| P3-1 to P3-6 | 6 | Institutional/Future capabilities |

---

## Checklist de Coherencia Verificable

### Criterios Objetivos de Coherencia

| # | Criterio | Verificación | Estado |
|---|----------|--------------|--------|
| 1 | Versiones de documentos referenciadas correctamente | IMPL v3.6 ↔ ARCH v1.3 | ✅ |
| 2 | ATR/ADX periods coinciden con código real | `observation_builder.py:120,123` | ✅ |
| 3 | observation_dim = 15 en ambos docs | Verificar líneas específicas | ✅ |
| 4 | feature_order idéntico (15 items, mismo orden) | Comparar arrays | ✅ |
| 5 | norm_stats path consistente | `config/v20_norm_stats.json` | ✅ |
| 6 | Calculators listados coinciden con `lib/features/calculators/` | 5 archivos | ✅ |
| 7 | features_snapshot format unificado | Dict structure matches | ✅ |
| 8 | Model Registry SQL schema idéntico | Comparar DDL | ✅ |
| 9 | Dependencias P0-8 → P1-11 → P1-13 documentadas | Ambos docs | ✅ |
| 10 | Industry-grade items referenciados bidireccionalmente | P1-14 to P2-24 | 🔄 |

**Cálculo del Score**:
- Items ✅: 9/10
- Items 🔄 (pendientes): 1/10

### Coherence Score: **9/10** ⚠️

> **Nota**: Score reducido de 10/10 a 9/10 porque los items industry-grade (P1-14 a P1-17)
> aún requieren detalle completo en ARCHITECTURE_CONTRACTS.md.

---

## Matriz de Coherencia con ARCHITECTURE_CONTRACTS.md

| Item | Este Documento | ARCHITECTURE_CONTRACTS.md | Estado |
|------|----------------|---------------------------|--------|
| Plan Item | P1-13 línea 899-1005 | P1-13 | ✅ |
| ATR Period | 10 (línea 920) | 10 (línea 227) | ✅ |
| ADX Period | 14 (línea 921) | 14 (línea 235) | ✅ |
| observation_dim | 15 | 15 | ✅ |
| feature_order | 15 items (línea 435-450) | 15 items | ✅ |
| norm_stats path | config/v20_norm_stats.json | config/v20_norm_stats.json | ✅ |
| Calculators | returns, rsi, atr, adx, macro | returns, rsi, atr, adx, macro | ✅ |
| features_snapshot format | Unified dict (línea 429-463) | Unified dict (línea 574-611) | ✅ |
| Model Registry | P1-11, P2-6 | Sección integrada | ✅ |
| P1-3 Status | Superseded por P1-13 | P1-13 supersedes P1-3 | ✅ |
| Dependencies | P0-8 → P1-11 → P1-13 | P0-8, P1-11 | ✅ |
| Industry Grade | P1-14 a P1-17, P2-21 a P2-24 | Referenciado en integración | ✅ |
| ONNX Inference | P1-14 (línea 1027-1131) | Pendiente detalle | 🔄 |
| Circuit Breakers | P1-15 (línea 1134-1274) | Pendiente detalle | 🔄 |
| Drift Detection | P1-16 (línea 1278-1397) | Pendiente detalle | 🔄 |
| Risk Engine | P1-17 (línea 1401-1553) | Pendiente detalle | 🔄 |

---

## Progresión de Madurez Industry-Grade

### Niveles con Métricas Intermedias

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         NIVELES DE MADUREZ                                      │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Level 0 (Actual)       Level 0.5 (P0)        Level 1 (P1)       Level 2 (P2) │
│  ──────────────────     ──────────────        ────────────       ──────────── │
│  • Python inference     • Bugs críticos       • ONNX <10ms       • TensorRT   │
│  • Manual monitoring      corregidos          • Circuit breakers • Observabil.│
│  • Basic logging        • Norm stats ok       • Drift detection  • Feat store │
│  • Ad-hoc testing       • Look-ahead fix      • Risk engine      • Property   │
│  • Local deployment     • Passwords ok        • Prometheus       • K8s+GitOps │
│                                                                                │
│  Score: 4.8/10          Target: 6.0/10        Target: 7.5/10     Target: 9.0  │
│                                                                                │
│  ─────────────────────────────────────────────────────────────────────────────│
│                                                                                │
│  Level 3 (P3) - Institucional                                                 │
│  ─────────────────────────────                                                │
│  • Model Gateway A/B    • DVC pipelines       • Chaos engineering             │
│  • INT8 quantization    • Full GitOps         • Sub-ms inference              │
│                                                                                │
│  Target: 9.5/10                                                               │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Métricas por Nivel (Detalladas)

| Nivel | Score | Criterio Principal | Métricas Clave |
|-------|-------|-------------------|----------------|
| 0 (Actual) | 4.8 | Sistema funcional con gaps | Latencia ~200ms, 0 alertas automáticas |
| 0.5 (P0) | 6.0 | Bugs críticos corregidos | Norm stats correcto, sin look-ahead |
| 1 (P1) | 7.5 | Production-ready básico | Latencia <10ms, circuit breakers activos |
| 1.5 (P1+) | 8.0 | Feature Contract completo | Drift monitoring, risk limits |
| 2 (P2) | 9.0 | Observabilidad completa | Grafana dashboards, PIT feature store |
| 3 (P3) | 9.5 | Institucional | A/B testing, chaos engineering |

> **Nota**: Transición de 4.8 → 8.5 requiere completar P0 + P1 completo.
> El salto directo no es realista sin métricas intermedias.

---

## Configuración Centralizada de Defaults

> **Mejora de Auditoría**: Extraer magic numbers a archivo de configuración externo.

### Archivo: `config/defaults.yaml`

```yaml
# config/defaults.yaml
# Configuración centralizada para evitar magic numbers hardcoded
# Referenciado por: IMPLEMENTATION_PLAN.md, ARCHITECTURE_CONTRACTS.md

trading:
  slippage:
    base_bps: 5.0              # Base slippage en basis points (ML-04)
    max_bps: 50.0              # Máximo slippage permitido
    off_hours_multiplier: 3.0  # Multiplicador fuera de horario
    friday_multiplier: 2.0     # Multiplicador viernes tarde
    monday_multiplier: 1.5     # Multiplicador lunes temprano

  market_hours:
    start_utc: 13              # Inicio SET-FX (13:00 UTC)
    end_utc: 19                # Fin SET-FX (19:00 UTC)

  risk:
    max_position_usd: 100000   # Posición máxima por símbolo
    daily_loss_limit_usd: 5000 # Límite pérdida diaria
    max_drawdown_pct: 10.0     # Drawdown máximo permitido
    consecutive_losses_halt: 5 # Halt después de N pérdidas

features:
  observation_dim: 15          # Dimensión del vector de observación
  clip_range: [-5.0, 5.0]      # Rango de clipping para normalización
  atr_period: 10               # Período ATR
  adx_period: 14               # Período ADX
  rsi_period: 9                # Período RSI

inference:
  default_confidence: 0.75     # Confidence por defecto si no disponible
  latency_target_ms: 10        # Target de latencia (ONNX)
  timeout_ms: 300000           # Timeout para backtests largos

alerts:
  drift_psi_threshold: 0.2     # Umbral PSI para alertas de drift
  high_volatility_zscore: 3.0  # Z-score para régimen alta volatilidad
```

### Uso en Código

```python
# Cargar configuración centralizada
import yaml
from pathlib import Path

def load_defaults() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

DEFAULTS = load_defaults()

# Uso
base_slippage = DEFAULTS['trading']['slippage']['base_bps']  # En vez de 5.0 hardcoded
```

---

## Story Points y Grafo de Dependencias

### Estimaciones por Prioridad

| Prioridad | Items | Story Points Total | Esfuerzo Estimado |
|-----------|-------|-------------------|-------------------|
| P0 | 11 | 22 SP | ~1 semana (2 SP/item avg) |
| P1 (core) | 13 | 39 SP | ~2 semanas (3 SP/item avg) |
| P1 (industry) | 4 | 20 SP | ~1 semana (5 SP/item avg) |
| P2 | 24 | 48 SP | ~3 semanas (2 SP/item avg) |
| P3 | 6 | 30 SP | Futuro (5 SP/item avg) |
| **Total** | **58** | **159 SP** | **~7-8 semanas** |

### Story Points por Item Crítico

| Item | SP | Complejidad | Riesgo |
|------|-----|-------------|--------|
| P0-7 (ML Workflow Audit) | 3 | Media | Alto |
| P0-8 (features_snapshot SQL) | 2 | Baja | Medio |
| P0-9 (look-ahead fix) | 3 | Alta | Alto |
| P1-13 (Feature Contract) | 8 | Muy Alta | Medio |
| P1-14 (ONNX Conversion) | 5 | Alta | Medio |
| P1-15 (Circuit Breakers) | 5 | Media | Bajo |
| P1-16 (Drift Detection) | 5 | Alta | Medio |
| P1-17 (Risk Engine) | 5 | Alta | Alto |

### Grafo de Dependencias (DAG)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DEPENDENCY DAG                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────┐                                                                    │
│  │  P0-1   │──┐                                                                 │
│  │norm_stat│  │                                                                 │
│  └─────────┘  │    ┌─────────┐                                                  │
│               ├───▶│  P1-13  │──────────────────────┐                           │
│  ┌─────────┐  │    │FeatureCt│                      │                           │
│  │  P0-8   │──┤    └─────────┘                      │                           │
│  │feat_snap│  │         │                           │                           │
│  └─────────┘  │         ▼                           ▼                           │
│               │    ┌─────────┐    ┌─────────┐  ┌─────────┐                      │
│  ┌─────────┐  │    │  P1-11  │───▶│  P1-14  │  │  P1-16  │                      │
│  │  P0-9   │──┘    │model_hsh│    │  ONNX   │  │  Drift  │                      │
│  │look_ahd │       └─────────┘    └─────────┘  └─────────┘                      │
│  └─────────┘            │              │            │                           │
│                         │              ▼            ▼                           │
│                         │         ┌─────────┐  ┌─────────┐                      │
│                         └────────▶│  P1-15  │  │  P2-21  │                      │
│                                   │CircuitBk│  │Observab │                      │
│                                   └─────────┘  └─────────┘                      │
│                                        │            │                           │
│                                        ▼            ▼                           │
│                                   ┌─────────┐  ┌─────────┐                      │
│                                   │  P1-17  │  │  P2-22  │                      │
│                                   │RiskEng  │  │FeatStore│                      │
│                                   └─────────┘  └─────────┘                      │
│                                                                                 │
│  Leyenda: ──▶ = depende de                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Ejemplos Detallados de Tests

> **Mejora de Auditoría**: Agregar ejemplos concretos de assertions y coverage targets.

### Test de Paridad Feature Contract

```python
# tests/integration/test_feature_contract_parity.py

import pytest
import numpy as np
from lib.features.builder import FeatureBuilder
from lib.features.contract import get_contract

class TestFeatureContractParity:
    """
    Tests que verifican paridad entre training e inference.
    Coverage target: 95% de lib/features/
    """

    @pytest.fixture
    def builder(self):
        return FeatureBuilder(version="v20")

    @pytest.fixture
    def sample_data(self):
        return create_sample_ohlcv_macro(bars=100)

    def test_observation_dimension_matches_contract(self, builder):
        """Observation dim debe coincidir con contrato"""
        contract = get_contract("v20")
        assert builder.get_observation_dim() == contract["observation_dim"]
        assert builder.get_observation_dim() == 15  # Valor esperado

    def test_feature_order_matches_contract(self, builder):
        """Orden de features debe coincidir exactamente"""
        contract = get_contract("v20")
        expected_order = [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            "position", "time_normalized"
        ]
        assert builder.get_feature_names() == expected_order
        assert contract["feature_order"] == expected_order

    def test_observation_no_nan_or_inf(self, builder, sample_data):
        """Observaciones nunca deben contener NaN o Inf"""
        ohlcv, macro = sample_data
        for bar_idx in range(10, len(ohlcv)):
            obs = builder.build_observation(
                ohlcv=ohlcv, macro=macro,
                position=0, timestamp=ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )
            assert not np.isnan(obs).any(), f"NaN en bar {bar_idx}"
            assert not np.isinf(obs).any(), f"Inf en bar {bar_idx}"

    def test_observation_clipped_to_range(self, builder, sample_data):
        """Features normalizadas deben estar en [-5, 5]"""
        ohlcv, macro = sample_data
        obs = builder.build_observation(
            ohlcv=ohlcv, macro=macro,
            position=0, timestamp=ohlcv.index[50],
            bar_idx=50
        )
        # Features 0-12 están normalizadas
        normalized_features = obs[:13]
        assert np.all(normalized_features >= -5.0)
        assert np.all(normalized_features <= 5.0)

    def test_same_input_same_output(self, builder, sample_data):
        """Mismo input debe producir exactamente mismo output (determinismo)"""
        ohlcv, macro = sample_data
        obs1 = builder.build_observation(
            ohlcv=ohlcv, macro=macro,
            position=0.5, timestamp=ohlcv.index[50],
            bar_idx=50
        )
        obs2 = builder.build_observation(
            ohlcv=ohlcv, macro=macro,
            position=0.5, timestamp=ohlcv.index[50],
            bar_idx=50
        )
        np.testing.assert_array_equal(obs1, obs2)


class TestRiskEngineProperties:
    """Tests de propiedades del Risk Engine"""

    def test_reduced_size_never_exceeds_original(self, risk_engine):
        """Tamaño ajustado nunca puede exceder el original"""
        for size in [1000, 10000, 50000, 100000]:
            result = risk_engine.check_trade('USDCOP', 'buy', size, volatility=2.0)
            if result.adjusted_size is not None:
                assert result.adjusted_size <= size

    def test_blocked_trade_has_reason(self, risk_engine):
        """Trade bloqueado siempre debe tener razón"""
        # Forzar bloqueo con posición muy grande
        result = risk_engine.check_trade('USDCOP', 'buy', 1_000_000, volatility=5.0)
        if result.action == RiskAction.BLOCK:
            assert result.reason is not None
            assert len(result.reason) > 0
```

### Coverage Targets

| Módulo | Target | Actual | Estado |
|--------|--------|--------|--------|
| `lib/features/` | 95% | - | 🔄 Pendiente |
| `lib/risk/` | 90% | - | 🔄 Pendiente |
| `lib/inference/` | 85% | - | 🔄 Pendiente |
| `services/inference_api/` | 80% | - | 🔄 Pendiente |

---

*Plan v3.6 - Mejoras de Auditoría Externa*
*Generado: 2026-01-11*
*Fuente: INDUSTRY_GRADE_MLOPS_RECOMMENDATIONS.md v1.0 + Auditoría Externa*
*Audit Score: 4.8/10 → Intermedio: 6.0 → Target: 7.5 → Institucional: 9.5/10*
*58 items totales de implementación (11 P0 + 17 P1 + 24 P2 + 6 P3)*
*Coherence Score: 9/10 (ver checklist verificable)*
*Idioma: Español (términos técnicos en inglés)*
