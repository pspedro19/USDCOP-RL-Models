# USDCOP RL Trading System
## Elite Technical Documentation v2.0

---

# Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Layer 0: Data Acquisition](#layer-0-data-acquisition)
4. [Layer 1: Feature Computation](#layer-1-feature-computation)
5. [Layer 2: Dataset Engineering](#layer-2-dataset-engineering)
6. [Layer 3: Model Training](#layer-3-model-training)
7. [Layer 4: Experiment Orchestration & Validation](#layer-4-experiment-orchestration--validation)
8. [Layer 5: Production Inference](#layer-5-production-inference)
9. [Contract System (SSOT)](#contract-system-ssot)
10. [MLOps Best Practices](#mlops-best-practices)
11. [Near Real-Time (NRT) Architecture](#near-real-time-nrt-architecture)
12. [Code Quality Patterns](#code-quality-patterns)
13. [Deployment Architecture](#deployment-architecture)
14. [Monitoring & Observability](#monitoring--observability)

---

# Executive Summary

## System Purpose
Production-grade **Reinforcement Learning (RL)** trading system for USD/COP currency pair prediction and automated trading execution.

## Key Metrics
| Metric | Value |
|--------|-------|
| Observation Space | 15-dimensional |
| Action Space | 3 (LONG, SHORT, HOLD) |
| Primary Algorithm | PPO (Proximal Policy Optimization) |
| Training Window | 2+ years historical data |
| Inference Latency | < 50ms (p95) |
| Feature Parity | 100% (Training = Inference) |

## Architecture Layers
```
L0 ─────► L1 ─────► L2 ─────► L3 ─────► L4 ─────► L5
 │         │         │         │         │         │
 │         │         │         │         │         └── Production Inference
 │         │         │         │         └── Experiment Validation
 │         │         │         └── Model Training
 │         │         └── Dataset Engineering
 │         └── Feature Computation
 └── Data Acquisition
```

## Core Principles
1. **SSOT (Single Source of Truth)**: All configurations flow from YAML files
2. **Contract-Driven**: Every layer has explicit input/output contracts
3. **Anti-Leakage**: Strict temporal separation (train < val < test)
4. **Two-Vote Promotion**: L4 recommends, Human approves
5. **Canary Deployment**: Gradual traffic shift with automatic rollback

---

# Architecture Overview

## Complete Data Flow
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION TRADING SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐                                   │
│  │ TwelveData   │    │ FRED/DANE    │                                   │
│  │ (OHLCV 5min) │    │ (Macro Data) │                                   │
│  └──────┬───────┘    └──────┬───────┘                                   │
│         │                   │                                            │
│         ▼                   ▼                                            │
│  ┌─────────────────────────────────────┐                                │
│  │           L0: DATA LAYER            │                                │
│  │  • OHLCV Realtime (every 5 min)     │                                │
│  │  • OHLCV Backfill (gap detection)   │                                │
│  │  • Macro Update (hourly)            │                                │
│  └─────────────────┬───────────────────┘                                │
│                    │                                                     │
│                    ▼                                                     │
│  ┌─────────────────────────────────────┐                                │
│  │        L1: FEATURE COMPUTATION       │                                │
│  │  • CanonicalFeatureBuilder (SSOT)   │                                │
│  │  • 13 Market + 2 State = 15 dims    │                                │
│  │  • Wilder's EMA for RSI/ATR/ADX     │                                │
│  └─────────────────┬───────────────────┘                                │
│                    │                                                     │
│         ┌─────────┴─────────┐                                           │
│         ▼                   ▼                                           │
│  ┌─────────────┐    ┌─────────────────────────────┐                     │
│  │ L2: DATASET │    │      L5: INFERENCE          │                     │
│  │  BUILDER    │    │  • ObservationBuilder       │                     │
│  │  • Anti-leak│    │  • Multi-Model Ensemble     │                     │
│  │  • Norm     │    │  • Risk Manager             │                     │
│  │  • Split    │    │  • Paper Trader             │                     │
│  └──────┬──────┘    └──────────────┬──────────────┘                     │
│         │                          │                                     │
│         ▼                          ▼                                     │
│  ┌─────────────┐           ┌──────────────┐                             │
│  │ L3: TRAIN   │           │ Trade Signals│                             │
│  │  • PPO      │           │  → Redis     │                             │
│  │  • MLflow   │           │  → PostgreSQL│                             │
│  │  • Curricul.│           │  → Dashboard │                             │
│  └──────┬──────┘           └──────────────┘                             │
│         │                                                                │
│         ▼                                                                │
│  ┌────────────────────────────────────────┐                             │
│  │     L4: EXPERIMENT VALIDATION          │                             │
│  │  • Backtest on Test Period             │                             │
│  │  • Compare vs Baseline                 │                             │
│  │  • Two-Vote Promotion System           │                             │
│  │    - L4: Automatic recommendation      │                             │
│  │    - Human: Dashboard approval         │                             │
│  └────────────────────────────────────────┘                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Structure Overview
```
USDCOP-RL-Models/
├── airflow/
│   └── dags/
│       ├── l0_ohlcv_realtime.py       # Real-time price ingestion
│       ├── l0_ohlcv_backfill.py       # Gap detection & fill
│       ├── l0_macro_update.py         # Macro data refresh
│       ├── l1_feature_refresh.py      # SSOT feature computation
│       ├── l2_dataset_builder.py      # Dataset engineering
│       ├── l3_model_training.py       # PPO training
│       ├── l4_experiment_runner.py    # Experiment orchestration
│       ├── l4_backtest_promotion.py   # Backtest & promotion
│       ├── l5_multi_model_inference.py # Production inference
│       └── contracts/
│           ├── dag_registry.py        # DAG naming SSOT
│           └── xcom_contracts.py      # Inter-DAG contracts
├── src/
│   ├── core/
│   │   ├── contracts/
│   │   │   ├── feature_contract.py    # CTR-FEAT-001
│   │   │   ├── production_contract.py # Two-vote loading
│   │   │   ├── promotion_contract.py  # L4 proposals
│   │   │   └── experiment_contract.py # Experiment config
│   │   ├── builders/
│   │   │   └── observation_builder.py # 15-dim observation
│   │   └── constants.py               # Global constants
│   ├── feature_store/
│   │   └── builders/
│   │       └── core.py                # CanonicalFeatureBuilder
│   ├── training/
│   │   ├── engine.py                  # TrainingEngine
│   │   ├── config.py                  # Hyperparameters
│   │   └── reward_calculator.py       # Reward shaping
│   ├── inference/
│   │   ├── inference_engine.py        # L5 main loop
│   │   ├── deployment_manager.py      # Canary/rollback
│   │   └── validated_predictor.py     # Contract-aware
│   └── risk/
│       └── risk_manager.py            # Risk checks
├── config/
│   ├── date_ranges.yaml               # SSOT for dates
│   ├── trading_config.yaml            # Market hours, thresholds
│   └── experiments/
│       └── *.yaml                     # Experiment configs
└── database/
    └── migrations/
        ├── 034_promotion_proposals.sql
        ├── 035_approval_audit_log.sql
        └── 036_model_registry_enhanced.sql
```

---

# Layer 0: Data Acquisition

## Purpose
Ingest real-time and historical market data from external sources into PostgreSQL/TimescaleDB with comprehensive gap detection and automatic backfill.

## DAGs

### L0-OHLCV-Realtime
**File:** `airflow/dags/l0_ohlcv_realtime.py`

| Property | Value |
|----------|-------|
| Schedule | `*/5 8-12 * * 1-5` (every 5 min, 8am-12:55pm COT, Mon-Fri) |
| Source | TwelveData API |
| Destination | `usdcop_m5_ohlcv` table |
| Timeout | 5 minutes |

**Key Functions:**
```python
def fetch_ohlcv_data(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    """Fetch OHLCV from TwelveData API with retry logic."""

def insert_ohlcv_data(df: pd.DataFrame, conn: Connection) -> int:
    """Insert with ON CONFLICT DO NOTHING for idempotency."""

def is_bar_in_market_hours(timestamp: datetime, tz: str) -> bool:
    """Validate bar is within 8:00-12:55 COT trading session."""
```

**Best Practices Applied:**
- **Idempotency**: `ON CONFLICT DO NOTHING` prevents duplicates
- **Retry with Backoff**: Exponential backoff on API failures
- **Rate Limiting**: Respects TwelveData API limits
- **Timezone Awareness**: All timestamps in America/Bogota

### L0-OHLCV-Backfill
**File:** `airflow/dags/l0_ohlcv_backfill.py`

| Property | Value |
|----------|-------|
| Schedule | Manual trigger or on startup |
| Function | Detect and fill gaps in historical data |
| Gap Detection | Scans MIN to MAX date range |

**Key Algorithm:**
```python
def detect_gaps() -> List[DateRange]:
    """
    Comprehensive gap detection algorithm:
    1. Query MIN(time), MAX(time) from database
    2. Generate expected bar times (5min intervals, market hours only)
    3. Query existing bars
    4. Compute missing = expected - existing
    5. Group consecutive gaps for efficient backfill
    """

def group_consecutive_gaps(gaps: List[datetime]) -> List[DateRange]:
    """Group consecutive missing bars into date ranges for batch API calls."""
```

**Best Practices Applied:**
- **Holiday Awareness**: Skips Colombian and US market holidays
- **Batch Processing**: Groups gaps to minimize API calls
- **Progress Tracking**: Reports fill rate percentage

### L0-Macro-Update
**File:** `airflow/dags/l0_macro_update.py`

| Property | Value |
|----------|-------|
| Schedule | `0 8-12 * * 1-5` (hourly, 8am-12pm COT, Mon-Fri) |
| Sources | FRED, DANE, Banco de la República |
| Destination | `macro_indicators_daily` table |
| Variables | 30 (18 daily, 8 monthly, 4 quarterly) |

**Variables Tracked:**
```yaml
Daily (18):
  - DXY, VIX, UST10Y, UST2Y, EMBI, BRENT
  - IBR, TPM, TRM, USDMXN, USDBRL
  - S&P500, COLCAP, Oil, Gold

Monthly (8):
  - FEDFUNDS, CPI, CPI_COL, UNEMPLOYMENT_US
  - UNEMPLOYMENT_COL, PMI_US, PMI_COL, M2

Quarterly (4):
  - GDP_US, GDP_COL, BOP, FDI
```

**Best Practices Applied:**
- **Circuit Breaker**: Per-source failure isolation
- **Rewrite Pattern**: Always refreshes last 15 records (no change detection needed)
- **Completeness Flag**: `is_complete` indicates all critical variables present

## Database Schema

```sql
-- OHLCV Data (TimescaleDB hypertable)
CREATE TABLE usdcop_m5_ohlcv (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(20) DEFAULT 'USD/COP',
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    source      VARCHAR(50) DEFAULT 'twelvedata',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('usdcop_m5_ohlcv', 'time');
CREATE INDEX idx_ohlcv_symbol_time ON usdcop_m5_ohlcv(symbol, time DESC);

-- Macro Data
CREATE TABLE macro_indicators_daily (
    fecha           DATE PRIMARY KEY,
    dxy             DOUBLE PRECISION,
    vix             DOUBLE PRECISION,
    embi            DOUBLE PRECISION,
    brent           DOUBLE PRECISION,
    treasury_10y    DOUBLE PRECISION,
    treasury_2y     DOUBLE PRECISION,
    usdmxn          DOUBLE PRECISION,
    -- ... more columns ...
    is_complete     BOOLEAN DEFAULT FALSE,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

---

# Layer 1: Feature Computation

## Purpose
Transform raw OHLCV and macro data into a **15-dimensional observation space** using the **CanonicalFeatureBuilder** as the Single Source of Truth for perfect training/inference parity.

## DAG

### L1-Feature-Refresh
**File:** `airflow/dags/l1_feature_refresh.py`

| Property | Value |
|----------|-------|
| Schedule | Event-driven via `NewOHLCVBarSensor` |
| Fallback | Every 5 minutes |
| Upstream | L0 OHLCV + L0 Macro |
| Destination | `inference_features_5m` + `feature_cache` |

**Task Flow:**
```
task_wait_ohlcv  →  task_validate_contract  →  task_compute  →  task_validate  →  task_mark_processed
      ↓                    ↓                       ↓                ↓
  Wait for new    Validate feature       Compute 15         Validate
  5-min bar       order matches          features           output
```

## CanonicalFeatureBuilder (SSOT)

**File:** `src/feature_store/builders/core.py`

### Feature Set (CTR-FEAT-001)

| Index | Feature | Description | Computation |
|-------|---------|-------------|-------------|
| 0 | `log_ret_5m` | 5-minute log return | `log(close / close[-1])` |
| 1 | `log_ret_1h` | 1-hour log return | `log(close / close[-12])` |
| 2 | `log_ret_4h` | 4-hour log return | `log(close / close[-48])` |
| 3 | `rsi_9` | RSI with Wilder's EMA | `100 - 100/(1 + RS)` where RS = EMA(gains)/EMA(losses) |
| 4 | `atr_pct` | ATR as % of price | `ATR(14) / close * 100` |
| 5 | `adx_14` | ADX trend strength | Wilder's smoothed DI calculation |
| 6 | `dxy_z` | Dollar Index z-score | `(DXY - mean_252) / std_252` |
| 7 | `dxy_change_1d` | DXY daily change | `(DXY - DXY[-1]) / DXY[-1]` |
| 8 | `vix_z` | VIX z-score | `(VIX - mean_252) / std_252` |
| 9 | `embi_z` | EMBI Colombia z-score | `(EMBI - mean_252) / std_252` |
| 10 | `brent_change_1d` | Brent oil daily change | `(BRENT - BRENT[-1]) / BRENT[-1]` |
| 11 | `rate_spread` | Rate differential | `COL_10Y - US_10Y` |
| 12 | `usdmxn_change_1d` | USD/MXN daily change | `(USDMXN - USDMXN[-1]) / USDMXN[-1]` |
| 13 | `position` | Current position state | `-1` (SHORT), `0` (NEUTRAL), `1` (LONG) |
| 14 | `time_normalized` | Hour of day normalized | `hour / 24.0` ∈ [0, 1] |

### Wilder's EMA Implementation
```python
def wilders_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Wilder's Exponential Moving Average (also called SMMA).

    α = 1/period (not 2/(period+1) like standard EMA)

    This is critical for RSI, ATR, and ADX calculations to match
    industry-standard implementations (TradingView, MetaTrader).
    """
    alpha = 1.0 / period
    return data.ewm(alpha=alpha, adjust=False).mean()
```

### Anti-Leakage Merge
```python
def merge_with_anti_leakage(ohlcv: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OHLCV with macro data using T-1 shift.

    For each trading bar at time T:
    - Use macro values from T-1 (yesterday)
    - Forward-fill ONLY within same trading session
    - Never fill across session boundaries

    This prevents future data from leaking into past observations.
    """
    # Shift macro by 1 business day
    macro_shifted = macro.shift(1, freq='B')

    # Merge on date
    merged = ohlcv.merge(
        macro_shifted,
        left_on=ohlcv.index.date,
        right_index=True,
        how='left'
    )

    # Forward-fill within session only
    merged = merged.groupby(merged.index.date).apply(
        lambda x: x.fillna(method='ffill')
    )

    return merged
```

## Feature Contract (CTR-FEAT-001)

**File:** `src/core/contracts/feature_contract.py`

```python
from typing import Tuple
import hashlib

# Immutable feature order - NEVER change without version bump
FEATURE_ORDER: Tuple[str, ...] = (
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    "position", "time_normalized",
)

OBSERVATION_DIM: int = 15
MARKET_FEATURES_COUNT: int = 13
STATE_FEATURES_COUNT: int = 2

# Hash for lineage tracking
FEATURE_ORDER_HASH: str = hashlib.sha256(
    str(FEATURE_ORDER).encode()
).hexdigest()[:16]

def validate_observation(obs: np.ndarray) -> bool:
    """Validate observation matches contract."""
    return (
        obs.shape == (OBSERVATION_DIM,) and
        obs.dtype == np.float32 and
        not np.any(np.isnan(obs)) and
        not np.any(np.isinf(obs))
    )
```

## Best Practices Applied

1. **SSOT Pattern**: Single CanonicalFeatureBuilder used everywhere
2. **Version Tracking**: Builder version + hash stored in database
3. **Idempotency**: Re-running produces identical results
4. **Wilder's EMA**: Industry-standard RSI/ATR/ADX calculations
5. **Anti-Leakage**: Macro T-1 shift + session-only ffill

---

# Layer 2: Dataset Engineering

## Purpose
Transform computed features into RL-ready training datasets with strict anti-leakage guarantees, train-only normalization, and comprehensive lineage tracking.

## DAG

### L2-Dataset-Builder
**File:** `airflow/dags/l2_dataset_builder.py`

| Property | Value |
|----------|-------|
| Schedule | Manual trigger (from L4) |
| Input | Experiment config + date ranges from SSOT |
| Output | train.parquet, val.parquet, test.parquet + norm_stats.json |

**Pipeline Steps:**
```
1. Load OHLCV + Macro from database
           ↓
2. Merge with anti-leakage (macro T-1)
           ↓
3. Calculate 13 market features (CanonicalFeatureBuilder)
           ↓
4. Drop NaN rows
           ↓
5. Compute norm stats (TRAIN ONLY)
           ↓
6. Apply z-score normalization (clip ±10)
           ↓
7. Split by date ranges
           ↓
8. Save outputs + lineage JSON
```

## Date Ranges (SSOT)

**File:** `config/date_ranges.yaml`

```yaml
# AUTHORITATIVE SOURCE FOR ALL DATE RANGES
# All components read from this file

data:
  start: "2020-03-01"  # First available data

training:
  start: "2020-03-01"
  end: "2024-12-31"

validation:
  start: "2025-01-01"
  end: "2025-06-30"

test:
  start: "2025-07-01"
  end: "dynamic"  # Current date
```

## Normalization Statistics

**Computed on TRAINING data only** to prevent data leakage.

```python
def compute_normalization_stats(train_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute per-feature statistics from TRAINING data only.
    These stats are applied to val and test sets.
    """
    stats = {}
    for col in train_df.columns:
        stats[col] = {
            'mean': float(train_df[col].mean()),
            'std': float(train_df[col].std()),
            'min': float(train_df[col].min()),
            'max': float(train_df[col].max()),
            'count': int(train_df[col].count()),
        }
    return stats

def apply_normalization(df: pd.DataFrame, stats: Dict) -> pd.DataFrame:
    """Apply z-score normalization with clipping."""
    normalized = df.copy()
    for col in df.columns:
        if col in stats:
            normalized[col] = (df[col] - stats[col]['mean']) / stats[col]['std']
            normalized[col] = normalized[col].clip(-10, 10)  # Prevent extreme values
    return normalized
```

**Example norm_stats.json:**
```json
{
  "feature_order_hash": "a1b2c3d4e5f6g7h8",
  "computed_at": "2025-01-15T10:30:00Z",
  "row_count": 523456,
  "features": {
    "log_ret_5m": {
      "mean": 0.0000123,
      "std": 0.00456,
      "min": -0.045,
      "max": 0.038,
      "count": 523456
    },
    "rsi_9": {
      "mean": 49.8,
      "std": 18.2,
      "min": 5.0,
      "max": 95.0,
      "count": 523456
    }
  }
}
```

## Output Contract (L2Output)

**File:** `airflow/dags/contracts/xcom_contracts.py`

```python
@dataclass
class L2Output:
    """Contract for L2 → L3 data handoff."""
    dataset_path: str           # Local path to dataset directory
    dataset_uri: Optional[str]  # S3/MinIO URI if using object storage
    dataset_hash: str           # MD5 of concatenated parquet files
    feature_order_hash: str     # Must match CTR-FEAT-001
    row_count: int              # Total rows across splits
    experiment_name: str        # From config
    norm_stats_path: str        # Path to norm_stats.json
    manifest_path: str          # Path to manifest.json

    # Split details
    train_rows: int
    val_rows: int
    test_rows: int
```

## Quality Gates (CTR-DQ-001)

```python
class DataQualityGate:
    """Enforce data quality before L3 training."""

    MIN_ROWS = 100_000
    MAX_NAN_PERCENT = 5.0
    MAX_INF_COUNT = 0
    MIN_FEATURE_COUNT = 13

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []

        if len(df) < self.MIN_ROWS:
            errors.append(f"Insufficient rows: {len(df)} < {self.MIN_ROWS}")

        nan_pct = df.isna().sum().sum() / df.size * 100
        if nan_pct > self.MAX_NAN_PERCENT:
            errors.append(f"Too many NaN: {nan_pct:.2f}% > {self.MAX_NAN_PERCENT}%")

        if np.isinf(df.values).sum() > self.MAX_INF_COUNT:
            errors.append("Contains infinite values")

        if len(df.columns) < self.MIN_FEATURE_COUNT:
            errors.append(f"Missing features: {len(df.columns)} < {self.MIN_FEATURE_COUNT}")

        return len(errors) == 0, errors
```

## Best Practices Applied

1. **Train-Only Normalization**: Stats computed on train set only
2. **Temporal Split**: No shuffling, strict date-based separation
3. **Lineage Tracking**: All hashes recorded in manifest.json
4. **Quality Gates**: Validation before L3 consumption
5. **Parquet Format**: Efficient columnar storage with compression

---

# Layer 3: Model Training

## Purpose
Train PPO (Proximal Policy Optimization) models using Stable-Baselines3 with curriculum learning, reward shaping, and comprehensive MLflow tracking.

## DAG

### L3-Model-Training
**File:** `airflow/dags/l3_model_training.py`

| Property | Value |
|----------|-------|
| Schedule | Manual trigger (from L4) |
| Input | L2Output (dataset) or explicit path |
| Output | Trained model + MLflow artifacts |

**Dataset Resolution Priority:**
```python
def resolve_dataset(context) -> str:
    """3-priority system for dataset resolution."""

    # Priority 1: Explicit path from dag_run.conf
    if 'dataset_path' in context['dag_run'].conf:
        return context['dag_run'].conf['dataset_path']

    # Priority 2: L2 XCom output
    l2_output = context['ti'].xcom_pull(
        dag_id='rl_l2_01_dataset_builder',
        key='l2_output'
    )
    if l2_output:
        return l2_output.dataset_path

    # Priority 3: Config fallback
    return config.dataset_dir / config.dataset_name
```

## TrainingEngine

**File:** `src/training/engine.py`

```python
class TrainingEngine:
    """
    Orchestrates PPO training with full MLflow integration.

    Responsibilities:
    1. Dataset validation (shape, dtype, NaN)
    2. Norm stats loading + validation
    3. Environment creation (TradingEnvironment)
    4. PPO training loop with curriculum
    5. MLflow logging (metrics, params, artifacts)
    6. Model registration + version management
    7. Contract generation
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mlflow_client = MlflowClient()

    def train(self, dataset_path: str) -> TrainingResult:
        """Main training entry point."""

        # 1. Load and validate dataset
        train_df = pd.read_parquet(f"{dataset_path}/train.parquet")
        val_df = pd.read_parquet(f"{dataset_path}/val.parquet")
        norm_stats = self._load_norm_stats(dataset_path)

        self._validate_dataset(train_df)

        # 2. Create environment
        env = self._create_environment(train_df, norm_stats)

        # 3. Setup PPO with configured hyperparameters
        model = PPO(
            policy='MlpPolicy',
            env=env,
            **self.config.ppo_hyperparameters,
            tensorboard_log=self.config.tensorboard_dir,
        )

        # 4. Setup callbacks
        callbacks = [
            EvalCallback(
                eval_env=self._create_environment(val_df, norm_stats),
                eval_freq=self.config.eval_freq,
                best_model_save_path=self.config.checkpoint_dir,
            ),
            CheckpointCallback(
                save_freq=self.config.checkpoint_freq,
                save_path=self.config.checkpoint_dir,
            ),
            MLflowCallback(self.mlflow_client),
        ]

        # 5. Train with MLflow tracking
        with mlflow.start_run():
            mlflow.log_params(self.config.to_dict())

            model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )

            # 6. Save artifacts
            model_path = self._save_model(model)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(f"{dataset_path}/norm_stats.json")

            # 7. Log final metrics
            metrics = self._evaluate_model(model, val_df, norm_stats)
            mlflow.log_metrics(metrics)

        return TrainingResult(
            model_path=model_path,
            metrics=metrics,
            run_id=mlflow.active_run().info.run_id,
        )
```

## PPO Hyperparameters

**File:** `src/training/config.py`

```python
@dataclass
class PPOHyperparameters:
    """PPO hyperparameters with sensible defaults for trading."""

    # Core PPO parameters
    learning_rate: float = 1e-4
    n_steps: int = 2048           # Steps per rollout
    batch_size: int = 64          # Minibatch size
    n_epochs: int = 10            # Epochs per update
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_range: float = 0.2       # PPO clip range
    ent_coef: float = 0.01        # Entropy coefficient
    vf_coef: float = 0.5          # Value function coefficient
    max_grad_norm: float = 0.5    # Gradient clipping

    # Training schedule
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    checkpoint_freq: int = 50_000

    # Network architecture
    policy_layers: List[int] = field(default_factory=lambda: [256, 256])
    value_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = 'Tanh'
```

## Reward Function

**File:** `src/training/reward_calculator.py`

```python
class RewardCalculator:
    """
    Multi-component reward function for RL trading.

    Components:
    1. P&L: Primary signal (realized returns)
    2. Trade Cost: Penalize excessive trading
    3. Drawdown: Penalize drawdown accumulation
    4. Regime: Adjust risk by market regime (VIX)
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def calculate(
        self,
        returns: float,
        trades_made: int,
        drawdown: float,
        vix_z: float,
    ) -> float:
        """Calculate composite reward."""

        # Base return (scaled for stability)
        reward = returns * 100

        # Trade cost penalty
        reward -= trades_made * self.config.trade_cost

        # Drawdown penalty (exponential)
        reward -= drawdown ** 2 * self.config.drawdown_penalty

        # Regime adjustment (reduce risk in high VIX)
        if vix_z > 1.5:  # High volatility regime
            reward *= 0.8  # Reduce reward magnitude

        return float(reward)
```

## TradingEnvironment

**File:** `src/training/environment.py`

```python
class TradingEnvironment(gym.Env):
    """
    OpenAI Gym-compatible trading environment.

    Observation Space: Box(15,) - normalized features
    Action Space: Discrete(3) - LONG, SHORT, HOLD
    """

    def __init__(
        self,
        df: pd.DataFrame,
        norm_stats: Dict,
        config: EnvironmentConfig,
    ):
        super().__init__()

        self.df = df
        self.norm_stats = norm_stats
        self.config = config

        # Spaces
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # 0=SELL, 1=HOLD, 2=BUY

        # State
        self.current_step = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.portfolio_value = config.initial_capital

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return next state."""

        # Get current price
        current_price = self.df.iloc[self.current_step]['close']

        # Calculate P&L if position change
        pnl = self._calculate_pnl(action, current_price)

        # Update position
        self._update_position(action, current_price)

        # Calculate reward
        reward = self.reward_calculator.calculate(
            returns=pnl / self.portfolio_value,
            trades_made=int(action != 1),  # HOLD = no trade
            drawdown=self._get_drawdown(),
            vix_z=self.df.iloc[self.current_step]['vix_z'],
        )

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = self.current_step >= self.config.max_steps

        obs = self._get_observation()
        info = {'pnl': pnl, 'position': self.position}

        return obs, reward, done, truncated, info
```

## MLflow Integration

```python
# MLflow artifact structure
models/<run_id>/
├── model.zip                 # Trained PPO model
├── config.yaml               # Training configuration
├── norm_stats.json           # Normalization statistics
├── training_curve.png        # Learning curve plot
└── metrics/
    ├── episode_reward.csv    # Per-episode rewards
    └── validation_metrics.csv # Eval metrics over time
```

## Output Contract (L3Output)

```python
@dataclass
class L3Output:
    """Contract for L3 → L4 handoff."""
    model_path: str              # Path to model.zip
    model_version: str           # Semantic version
    metrics: Dict[str, float]    # Final metrics
    training_run_id: str         # MLflow run ID
    experiment_name: str         # For lineage
    norm_stats_path: str         # Path to norm_stats
    config_hash: str             # Config MD5
    feature_order_hash: str      # Must match training data
```

## Best Practices Applied

1. **MLflow Tracking**: All experiments fully tracked
2. **Checkpointing**: Regular model saves for recovery
3. **Early Stopping**: Via EvalCallback
4. **Curriculum Learning**: Optional difficulty progression
5. **Reproducibility**: Seed control for all random operations

---

# Layer 4: Experiment Orchestration & Validation

## Purpose
Orchestrate A/B experiment workflows, run out-of-sample backtests, compare against baseline, and coordinate the **Two-Vote Promotion System**.

## DAGs

### L4-Experiment-Runner
**File:** `airflow/dags/l4_experiment_runner.py`

| Property | Value |
|----------|-------|
| Schedule | Manual trigger |
| Function | Orchestrate L2 → L3 pipeline |
| Pattern | TriggerDagRunOperator |

**Workflow:**
```
validate_config
      ↓
trigger_l2_preprocessing
      ↓
[wait for L2 completion]
      ↓
trigger_l3_training
      ↓
[wait for L3 completion]
      ↓
collect_results
      ↓
compare_with_baseline
      ↓
register_experiment
```

### L4-Backtest-Promotion
**File:** `airflow/dags/l4_backtest_promotion.py`

| Property | Value |
|----------|-------|
| Schedule | After L3 completion |
| Function | Run OOS backtest, generate promotion proposal |
| Output | Promotion proposal in database |

**Backtest Engine:**
```python
class BacktestEngine:
    """
    Run out-of-sample backtest for model evaluation.

    Key Features:
    - Uses TEST period only (never seen in training)
    - Simulates L1→L5 pipeline bar-by-bar
    - Calculates comprehensive metrics
    """

    def run_backtest(
        self,
        model_path: str,
        test_df: pd.DataFrame,
        norm_stats: Dict,
    ) -> BacktestResult:
        """Run complete OOS backtest."""

        # Load model
        model = PPO.load(model_path)

        # Simulate trading
        portfolio = Portfolio(initial_capital=10000)

        for i in range(len(test_df)):
            # Build observation (same as L5)
            obs = self._build_observation(test_df.iloc[i], norm_stats)

            # Get action
            action, _ = model.predict(obs, deterministic=True)

            # Execute trade
            portfolio.execute(action, test_df.iloc[i]['close'])

        return BacktestResult(
            total_return=portfolio.total_return,
            sharpe_ratio=portfolio.sharpe_ratio,
            max_drawdown=portfolio.max_drawdown,
            win_rate=portfolio.win_rate,
            profit_factor=portfolio.profit_factor,
            total_trades=portfolio.total_trades,
        )
```

## Two-Vote Promotion System

### First Vote: L4 Automatic Recommendation
```python
def determine_recommendation(
    metrics: BacktestMetrics,
    baseline_metrics: BacktestMetrics,
    criteria: SuccessCriteria,
) -> PromotionRecommendation:
    """
    L4's automatic recommendation based on OOS backtest.

    Returns: PROMOTE, REJECT, or REVIEW
    """

    # Evaluate each criterion
    passed_criteria = []
    for criterion in criteria:
        value = getattr(metrics, criterion.metric)
        threshold = criterion.threshold
        passed = value >= threshold if criterion.direction == 'gte' else value <= threshold
        passed_criteria.append((criterion.name, passed, value, threshold))

    # Calculate weighted score
    total_weight = sum(c.weight for c in criteria)
    passed_weight = sum(
        c.weight for c, (_, passed, _, _) in zip(criteria, passed_criteria) if passed
    )
    score = passed_weight / total_weight

    # Determine recommendation
    if score >= 0.8:  # 80% of weighted criteria passed
        return PromotionRecommendation.PROMOTE
    elif score >= 0.5:  # 50-80%
        return PromotionRecommendation.REVIEW
    else:
        return PromotionRecommendation.REJECT
```

### Second Vote: Human Approval (Dashboard)
```python
# Database table for promotion proposals
class PromotionProposal(BaseModel):
    proposal_id: str          # UUID
    model_id: str             # Model being proposed
    experiment_name: str
    recommendation: str       # PROMOTE, REJECT, REVIEW
    confidence: float         # L4's confidence score
    reason: str               # L4's explanation
    metrics: Dict             # Backtest metrics
    vs_baseline: Dict         # Comparison with current prod
    criteria_results: List    # Per-criterion results
    lineage: Dict             # All hashes for lineage
    status: str               # PENDING_APPROVAL, APPROVED, REJECTED
    reviewer: Optional[str]   # Human reviewer
    reviewed_at: Optional[datetime]
    created_at: datetime
    expires_at: datetime      # Auto-expire if not reviewed
```

## Success Criteria (from YAML)

**File:** `config/experiments/{name}.yaml`

```yaml
success_criteria:
  - metric: sharpe_ratio
    threshold: 1.0
    direction: gte  # greater than or equal
    weight: 3.0

  - metric: max_drawdown
    threshold: 0.15
    direction: lte  # less than or equal
    weight: 2.0

  - metric: win_rate
    threshold: 0.45
    direction: gte
    weight: 1.5

  - metric: profit_factor
    threshold: 1.2
    direction: gte
    weight: 2.0

  - metric: total_trades
    threshold: 50
    direction: gte
    weight: 1.0
```

## Baseline Comparison

```python
def compare_vs_baseline(
    challenger: BacktestMetrics,
    champion: BacktestMetrics,
) -> BaselineComparison:
    """Compare challenger model against current champion."""

    return BaselineComparison(
        return_delta=challenger.total_return - champion.total_return,
        sharpe_delta=challenger.sharpe_ratio - champion.sharpe_ratio,
        drawdown_delta=champion.max_drawdown - challenger.max_drawdown,  # Lower is better
        win_rate_delta=challenger.win_rate - champion.win_rate,

        # Overall assessment
        is_improvement=challenger.sharpe_ratio > champion.sharpe_ratio * 1.05,
    )
```

## Best Practices Applied

1. **Two-Vote System**: Automatic + Human approval
2. **OOS Testing**: Only test period used for evaluation
3. **Expiration**: Proposals expire if not reviewed
4. **Audit Trail**: Full logging of all decisions
5. **Lineage Tracking**: Complete hash chain for reproducibility

---

# Layer 5: Production Inference

## Purpose
Real-time multi-model inference with 15-feature observation space, comprehensive risk management, canary deployment, and automatic rollback.

## DAG

### L5-Multi-Model-Inference
**File:** `airflow/dags/l5_multi_model_inference.py`

| Property | Value |
|----------|-------|
| Schedule | Event-driven via `NewFeatureBarSensor` |
| Fallback | Every 5 minutes during market hours |
| Upstream | L1 features + model registry |
| Output | Signals → Redis + PostgreSQL |

**Architecture:**
```
Feature Sensor (wait for new bar)
          ↓
Load active models (DeploymentManager)
          ↓
Build observation (ObservationBuilder - 15-dim)
          ↓
Multi-model inference (champion + challenger)
          ↓
Risk checks (RiskManager)
          ↓
Paper trade execution (PaperTrader)
          ↓
Multi-destination output
  ├─ Redis Streams (real-time)
  ├─ PostgreSQL (persistence)
  └─ Events table (audit)
          ↓
Model monitoring (drift detection)
```

## ObservationBuilder

**File:** `src/core/builders/observation_builder.py`

```python
class ObservationBuilder:
    """
    Build 15-dimensional observation vector for inference.

    CRITICAL: Must produce identical output to training observations.
    Uses same CanonicalFeatureBuilder as L1/L2.
    """

    def __init__(self, norm_stats: Dict):
        self.norm_stats = norm_stats
        self.state_tracker = StateTracker()

    def build(self, features: pd.Series) -> np.ndarray:
        """
        Build observation from latest feature row.

        Returns:
            np.ndarray: Shape (15,), dtype float32
        """
        # Extract 13 core features
        core_features = [features[f] for f in FEATURE_ORDER[:13]]

        # Add state features
        position = self.state_tracker.get_position()  # -1, 0, 1
        time_norm = datetime.now().hour / 24.0  # [0, 1]

        # Combine
        obs = np.array(
            core_features + [position, time_norm],
            dtype=np.float32
        )

        # Normalize
        obs = self._normalize(obs)

        # Validate
        assert obs.shape == (OBSERVATION_DIM,)
        assert not np.any(np.isnan(obs))

        return obs

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using training stats."""
        normalized = np.zeros_like(obs)
        for i, feature in enumerate(FEATURE_ORDER):
            stats = self.norm_stats['features'][feature]
            normalized[i] = (obs[i] - stats['mean']) / stats['std']
            normalized[i] = np.clip(normalized[i], -10, 10)
        return normalized
```

## DeploymentManager (Canary Pattern)

**File:** `src/inference/deployment_manager.py`

```python
class DeploymentManager:
    """
    Manage canary deployments with automatic promotion/rollback.

    Traffic Flow:
    1. CHAMPION: 100% traffic initially
    2. Deploy CHALLENGER: 10% traffic to challenger
    3. Evaluate: Compare metrics over rolling window
    4. Promote: If challenger wins, gradually shift traffic
    5. Rollback: If challenger underperforms, immediate rollback
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.champion: Optional[ModelDeployment] = None
        self.challenger: Optional[ModelDeployment] = None

    def get_model_for_inference(self) -> Tuple[Model, str]:
        """
        Route inference to appropriate model based on traffic split.

        Returns:
            Tuple of (model, model_id)
        """
        if not self.challenger:
            return self.champion.model, self.champion.model_id

        # Probabilistic routing based on traffic split
        if random.random() < self.challenger.traffic_split:
            return self.challenger.model, self.challenger.model_id
        else:
            return self.champion.model, self.champion.model_id

    def evaluate_and_adjust(self):
        """
        Evaluate challenger performance and adjust traffic or rollback.

        Called periodically (e.g., every hour during market hours).
        """
        if not self.challenger:
            return

        champion_metrics = self._get_rolling_metrics(self.champion.model_id)
        challenger_metrics = self._get_rolling_metrics(self.challenger.model_id)

        # Check for promotion
        if challenger_metrics.sharpe > champion_metrics.sharpe * self.config.promotion_threshold:
            self._increase_traffic()

        # Check for rollback
        elif challenger_metrics.sharpe < champion_metrics.sharpe * self.config.rollback_threshold:
            self._rollback()

    def _increase_traffic(self):
        """Gradually increase challenger traffic: 10% → 50% → 100%"""
        current = self.challenger.traffic_split
        if current < 0.5:
            self.challenger.traffic_split = 0.5
        else:
            self._promote_challenger()

    def _rollback(self):
        """Immediate rollback to champion."""
        self.challenger.status = DeploymentStatus.ROLLED_BACK
        self.challenger = None
        self._log_event('ROLLBACK', 'Challenger underperformed')
```

## RiskManager

**File:** `src/risk/risk_manager.py`

```python
class RiskManager:
    """
    Comprehensive risk management for production trading.

    Checks Applied:
    1. Confidence threshold
    2. Daily loss limit
    3. Drawdown limit
    4. Circuit breaker (consecutive losses)
    5. Trading hours
    6. Max trades per session
    7. Position limits
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None

    def check_all(
        self,
        action: int,
        confidence: float,
        current_position: int,
    ) -> RiskCheckResult:
        """Run all risk checks and return result."""

        checks = [
            self._check_confidence(confidence),
            self._check_daily_loss(),
            self._check_drawdown(),
            self._check_circuit_breaker(),
            self._check_trading_hours(),
            self._check_max_trades(),
            self._check_position_limits(action, current_position),
        ]

        failed = [c for c in checks if not c.passed]

        return RiskCheckResult(
            passed=len(failed) == 0,
            failed_checks=failed,
            action_allowed=action if len(failed) == 0 else 1,  # HOLD if blocked
        )

    def _check_confidence(self, confidence: float) -> RiskCheck:
        """Only trade if confidence exceeds threshold."""
        passed = confidence >= self.limits.confidence_threshold
        return RiskCheck(
            name='confidence',
            passed=passed,
            message=f'Confidence {confidence:.2f} < {self.limits.confidence_threshold}'
        )

    def _check_circuit_breaker(self) -> RiskCheck:
        """Check if circuit breaker is active."""
        if self.circuit_breaker_active:
            if datetime.now() < self.circuit_breaker_until:
                return RiskCheck(
                    name='circuit_breaker',
                    passed=False,
                    message=f'Circuit breaker active until {self.circuit_breaker_until}'
                )
            else:
                self.circuit_breaker_active = False
        return RiskCheck(name='circuit_breaker', passed=True)
```

## Multi-Destination Output

```python
class InferenceOutputManager:
    """Route inference results to multiple destinations."""

    def __init__(self):
        self.redis = Redis()
        self.db = Database()

    async def publish(self, result: InferenceResult):
        """Publish to all destinations concurrently."""

        await asyncio.gather(
            self._publish_redis(result),
            self._publish_postgres(result),
            self._publish_events(result),
        )

    async def _publish_redis(self, result: InferenceResult):
        """Publish to Redis Streams for real-time consumers."""
        await self.redis.xadd(
            'trading:signals:live',
            {
                'timestamp': result.timestamp.isoformat(),
                'model_id': result.model_id,
                'signal': result.signal.name,
                'confidence': str(result.confidence),
                'price': str(result.price),
            }
        )

    async def _publish_postgres(self, result: InferenceResult):
        """Persist to PostgreSQL for historical analysis."""
        await self.db.execute("""
            INSERT INTO inference_signals
            (timestamp, model_id, action, confidence, price, observation)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, result.timestamp, result.model_id, result.signal.value,
            result.confidence, result.price, result.observation.tolist())
```

## ProductionContract Validation

**File:** `src/core/contracts/production_contract.py`

```python
class ProductionContract:
    """
    Two-vote validation for production model loading.

    Both votes must pass:
    1. FEATURE_ORDER_HASH must match
    2. NORM_STATS_HASH must match

    If either fails, fallback to champion model.
    """

    @classmethod
    def validate_and_load(
        cls,
        model_path: str,
        expected_feature_hash: str,
        expected_norm_hash: str,
    ) -> Tuple[Model, bool]:
        """
        Validate model before loading into production.

        Returns:
            Tuple of (model, validation_passed)
        """
        # Load model metadata
        metadata = cls._load_metadata(model_path)

        # Vote 1: Feature order hash
        feature_match = metadata['feature_order_hash'] == expected_feature_hash
        if not feature_match:
            logger.critical(
                f"Feature hash mismatch: {metadata['feature_order_hash']} != {expected_feature_hash}"
            )

        # Vote 2: Norm stats hash
        norm_match = metadata['norm_stats_hash'] == expected_norm_hash
        if not norm_match:
            logger.critical(
                f"Norm stats hash mismatch: {metadata['norm_stats_hash']} != {expected_norm_hash}"
            )

        # Both must pass
        if feature_match and norm_match:
            model = PPO.load(model_path)
            return model, True
        else:
            logger.warning("Validation failed, falling back to champion")
            return cls._load_champion(), False
```

## Best Practices Applied

1. **Two-Vote Validation**: Feature + norm stats hash verification
2. **Canary Deployment**: Gradual traffic shift with monitoring
3. **Circuit Breaker**: Automatic pause after consecutive losses
4. **Multi-Destination**: Redis (real-time) + PostgreSQL (persistence)
5. **Graceful Degradation**: Fallback to champion on any failure

---

# Contract System (SSOT)

## Overview

The contract system ensures **perfect parity** between training and inference through immutable specifications and hash-based lineage tracking.

## Contract Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                 CTR-FEAT-001                            │
│            Feature Contract (SSOT)                      │
│  • 15-dimensional observation space                     │
│  • Immutable feature order                              │
│  • FEATURE_ORDER_HASH for validation                    │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  L2Output   │  │  L3Output   │  │ Production  │
│  Contract   │  │  Contract   │  │  Contract   │
│ • dataset   │  │ • model     │  │ • two-vote  │
│ • norm_stats│  │ • metrics   │  │ • canary    │
│ • hashes    │  │ • hashes    │  │ • rollback  │
└─────────────┘  └─────────────┘  └─────────────┘
```

## CTR-FEAT-001: Feature Contract

```python
# src/core/contracts/feature_contract.py

from typing import Tuple
import hashlib

FEATURE_ORDER: Tuple[str, ...] = (
    # Core market features (0-12)
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    # State features (13-14)
    "position", "time_normalized",
)

OBSERVATION_DIM = 15
FEATURE_ORDER_HASH = hashlib.sha256(str(FEATURE_ORDER).encode()).hexdigest()[:16]
```

## Lineage Tracking

Every artifact includes complete lineage for reproducibility:

```json
{
  "lineage": {
    "config_hash": "a1b2c3d4",
    "feature_order_hash": "e5f6g7h8",
    "dataset_hash": "i9j0k1l2",
    "norm_stats_hash": "m3n4o5p6",
    "model_hash": "q7r8s9t0",
    "reward_config_hash": "u1v2w3x4",
    "training_start": "2022-01-01",
    "training_end": "2024-12-31",
    "created_at": "2025-01-15T10:30:00Z",
    "created_by": "l3_model_training"
  }
}
```

---

# MLOps Best Practices

## 1. Experiment Tracking (MLflow)

```python
# Every training run is fully tracked
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'algorithm': 'PPO',
        'total_timesteps': 500000,
        'learning_rate': 1e-4,
        # ... all hyperparameters
    })

    # Log metrics during training
    for step, reward in enumerate(training_rewards):
        mlflow.log_metric('episode_reward', reward, step=step)

    # Log artifacts
    mlflow.log_artifact('model.zip')
    mlflow.log_artifact('norm_stats.json')
    mlflow.log_artifact('config.yaml')

    # Register model
    mlflow.register_model(
        f"runs:/{run_id}/model",
        "usdcop_ppo"
    )
```

## 2. Data Versioning

```python
# Every dataset is versioned with DVC-compatible hashes
dataset_manifest = {
    'version': '2.0.0',
    'created_at': '2025-01-15T10:30:00Z',
    'files': {
        'train.parquet': {
            'md5': 'abc123...',
            'rows': 400000,
            'size_bytes': 15000000,
        },
        'val.parquet': { ... },
        'test.parquet': { ... },
        'norm_stats.json': { ... },
    },
    'lineage': { ... },
}
```

## 3. Model Registry

```sql
CREATE TABLE model_registry (
    model_id VARCHAR(100) PRIMARY KEY,
    experiment_name VARCHAR(100),
    model_path VARCHAR(500),
    model_hash VARCHAR(64),
    stage VARCHAR(20),  -- 'staging', 'production', 'archived'
    is_active BOOLEAN,
    approved_by VARCHAR(100),
    approved_at TIMESTAMPTZ,
    promoted_at TIMESTAMPTZ,
    metrics JSONB,
    lineage JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 4. Feature Store Pattern

```python
# Features are cached with version + hash
class FeatureCache:
    def store(self, features: pd.DataFrame, metadata: FeatureMetadata):
        """Store computed features with full metadata."""
        self.db.execute("""
            INSERT INTO feature_cache (
                timestamp, features, builder_version,
                feature_order_hash, computed_at
            ) VALUES ($1, $2, $3, $4, $5)
        """, ...)

    def validate(self, expected_hash: str) -> bool:
        """Validate cached features match expected hash."""
        return self.get_latest_hash() == expected_hash
```

## 5. A/B Testing Framework

```yaml
# config/experiments/ab_test_config.yaml
ab_test:
  name: "gamma_sensitivity"
  variants:
    - name: "control"
      config: "baseline_ppo_v1.yaml"
      traffic: 0.5
    - name: "treatment"
      config: "high_gamma_ppo_v1.yaml"
      traffic: 0.5
  metrics:
    primary: "sharpe_ratio"
    secondary: ["max_drawdown", "win_rate"]
  duration_days: 30
```

---

# Near Real-Time (NRT) Architecture

## Unified L1+L5 for Backtest and Production

```
┌────────────────────────────────────────────────────────────────────┐
│                    UNIFIED INFERENCE PIPELINE                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐                                               │
│  │   Data Source   │                                               │
│  │                 │                                               │
│  │ PRODUCTION:     │                                               │
│  │  Real-time bars │                                               │
│  │  (TwelveData)   │                                               │
│  │                 │                                               │
│  │ BACKTEST:       │                                               │
│  │  Historical bars│                                               │
│  │  (PostgreSQL)   │                                               │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                               │
│  │       L1        │  ◄── SAME CODE                                │
│  │  Feature Comp.  │                                               │
│  │  (Canonical     │                                               │
│  │   Builder)      │                                               │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                               │
│  │       L5        │  ◄── SAME CODE                                │
│  │  Inference      │                                               │
│  │  (Model +       │                                               │
│  │   Risk Mgr)     │                                               │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                               │
│  │     Output      │                                               │
│  │                 │                                               │
│  │ PRODUCTION:     │                                               │
│  │  Trade signal   │                                               │
│  │  → Exchange     │                                               │
│  │                 │                                               │
│  │ BACKTEST:       │                                               │
│  │  Equity curve   │                                               │
│  │  → Dashboard    │                                               │
│  └─────────────────┘                                               │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## API Endpoints

```python
# Backtest Replay Mode (bar-by-bar L1+L5)
POST /v1/backtest/replay
{
    "start_date": "2025-07-01",
    "end_date": "2025-12-31",
    "model_id": "ppo_v20_123456",
    "mode": "replay",
    "emit_bar_events": true
}
# Returns: SSE stream with bar + trade events

# Production Inference (real-time L1+L5)
POST /v1/inference
{
    "timestamp": "2025-01-15T14:00:00Z",
    "model_id": "ppo_v20_production"
}
# Returns: Signal (LONG/SHORT/HOLD) + confidence
```

## SSE Event Types

```typescript
// Progress event
{ type: "progress", data: { progress: 0.45, current_bar: 1500, total_bars: 3300 }}

// Bar event (for dynamic equity curve)
{ type: "bar", data: { timestamp: "...", equity: 10234.56, position: "LONG" }}

// Trade event
{ type: "trade", data: { side: "BUY", entry_price: 4250.00, pnl: 15.50 }}

// Result event (backtest complete)
{ type: "result", data: { success: true, total_return: 0.15, sharpe: 1.8 }}
```

---

# Code Quality Patterns

## 1. SOLID Principles

### Single Responsibility
```python
# Each class has one job
class FeatureBuilder:      # Only builds features
class NormalizationEngine: # Only normalizes
class RiskManager:         # Only checks risk
class TradeExecutor:       # Only executes trades
```

### Open/Closed
```python
# Extend via composition, not modification
class BaseRiskCheck(ABC):
    @abstractmethod
    def check(self, context: RiskContext) -> bool:
        pass

class ConfidenceCheck(BaseRiskCheck): ...
class DrawdownCheck(BaseRiskCheck): ...
class CircuitBreakerCheck(BaseRiskCheck): ...

# Adding new checks doesn't modify existing code
```

### Dependency Inversion
```python
# Depend on abstractions, not concretions
class InferenceEngine:
    def __init__(
        self,
        model_loader: ModelLoaderProtocol,      # Interface
        feature_builder: FeatureBuilderProtocol, # Interface
        risk_manager: RiskManagerProtocol,       # Interface
    ):
        self.model_loader = model_loader
        self.feature_builder = feature_builder
        self.risk_manager = risk_manager
```

## 2. Design Patterns

### Factory Pattern
```python
def create_backtest_runner(
    request: BacktestRequest,
    handlers: BacktestEventHandlers,
) -> BacktestRunner:
    """Factory for creating configured backtest runners."""
    return BacktestRunner(
        engine=BacktestEngine(request.model_path),
        event_handlers=handlers,
        config=BacktestConfig.from_request(request),
    )
```

### Strategy Pattern
```python
class RewardStrategy(Protocol):
    def calculate(self, context: RewardContext) -> float: ...

class PnLReward(RewardStrategy): ...
class SharpeReward(RewardStrategy): ...
class RiskAdjustedReward(RewardStrategy): ...

# Switch strategies via configuration
reward_strategy = STRATEGY_MAP[config.reward_type]
```

### Observer Pattern
```python
class InferenceEventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable):
        self.subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event_type: str, data: Any):
        for handler in self.subscribers.get(event_type, []):
            handler(data)

# Usage
event_bus.subscribe('trade_executed', log_trade)
event_bus.subscribe('trade_executed', update_metrics)
event_bus.subscribe('trade_executed', notify_dashboard)
```

## 3. Error Handling

```python
class TradingError(Exception):
    """Base exception for trading system."""
    pass

class FeatureComputationError(TradingError):
    """Raised when feature computation fails."""
    pass

class ModelLoadError(TradingError):
    """Raised when model loading fails."""
    pass

class RiskCheckError(TradingError):
    """Raised when risk check fails."""
    pass

# Structured error handling with recovery
try:
    observation = feature_builder.build(data)
except FeatureComputationError as e:
    logger.error(f"Feature computation failed: {e}")
    metrics.increment('feature_errors')
    # Graceful degradation: use last valid observation
    observation = state_tracker.get_last_observation()
```

## 4. Type Safety

```python
from typing import TypedDict, Protocol, Final
from dataclasses import dataclass
from enum import IntEnum

class Action(IntEnum):
    SELL = 0
    HOLD = 1
    BUY = 2

@dataclass(frozen=True)
class TradeSignal:
    action: Action
    confidence: float
    timestamp: datetime
    model_id: str

class InferenceResult(TypedDict):
    signal: TradeSignal
    observation: np.ndarray
    risk_checks: List[RiskCheck]
```

---

# Deployment Architecture

## Production Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRODUCTION CLUSTER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Airflow    │  │   FastAPI    │  │   Next.js    │          │
│  │   Scheduler  │  │   Inference  │  │   Dashboard  │          │
│  │   (DAGs)     │  │   (L5)       │  │   (UI)       │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                    │
│         └────────┬────────┴────────┬────────┘                   │
│                  │                 │                             │
│         ┌────────▼────────┐ ┌──────▼──────┐                     │
│         │   TimescaleDB   │ │    Redis    │                     │
│         │   (Persistence) │ │  (Real-time)│                     │
│         └─────────────────┘ └─────────────┘                     │
│                                                                  │
│         ┌─────────────────┐ ┌─────────────┐                     │
│         │     MinIO       │ │   MLflow    │                     │
│         │   (Artifacts)   │ │ (Tracking)  │                     │
│         └─────────────────┘ └─────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Container Services

```yaml
# docker-compose.prod.yml
services:
  airflow-scheduler:
    image: usdcop/airflow:2.8
    volumes:
      - ./airflow/dags:/opt/airflow/dags
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor

  inference-api:
    image: usdcop/inference:latest
    ports:
      - "8003:8003"
    environment:
      - MODEL_PATH=/models/production
      - RISK_CONFIG=/config/risk.yaml

  dashboard:
    image: usdcop/dashboard:latest
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://inference-api:8003

  timescaledb:
    image: timescale/timescaledb:latest-pg14
    volumes:
      - timescale_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
    command: server /data

  mlflow:
    image: usdcop/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://...
      - MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts
```

---

# Monitoring & Observability

## Key Metrics

### L0 Data Quality
| Metric | Alert Threshold |
|--------|-----------------|
| OHLCV gap rate | > 1% |
| Macro completeness | < 95% |
| API error rate | > 5% |

### L1 Feature Health
| Metric | Alert Threshold |
|--------|-----------------|
| NaN rate | > 0.1% |
| Outlier rate | > 1% |
| Latency p95 | > 100ms |

### L5 Inference
| Metric | Alert Threshold |
|--------|-----------------|
| Inference latency p95 | > 50ms |
| Risk check fail rate | > 20% |
| Model drift score | > 0.05 (KS statistic) |

### Trading Performance
| Metric | Alert Threshold |
|--------|-----------------|
| Daily loss | > 2% |
| Drawdown | > 15% |
| Consecutive losses | > 5 |

## Grafana Dashboards

```yaml
# Dashboard panels
- L0 Data Ingestion
  - OHLCV bars per minute
  - Macro update status
  - Gap detection alerts

- L1 Feature Health
  - Feature distributions (histograms)
  - NaN/Inf counts
  - Computation latency

- L5 Production
  - Real-time P&L curve
  - Position history
  - Model confidence distribution
  - Risk check outcomes

- Model Performance
  - Sharpe ratio (rolling 30-day)
  - Win rate trend
  - Trade frequency
  - Canary comparison
```

---

# Appendix: Quick Reference

## Start Training Run
```bash
airflow dags trigger rl_l4_01_experiment_runner \
  --conf '{"experiment_name": "my_experiment_v1"}'
```

## Check Production Model
```sql
SELECT model_id, stage, metrics, promoted_at
FROM model_registry
WHERE stage = 'production' AND is_active = TRUE;
```

## Manual Rollback
```sql
UPDATE model_registry SET stage = 'archived' WHERE model_id = 'challenger';
UPDATE model_registry SET stage = 'production', is_active = TRUE WHERE model_id = 'champion';
```

## View Recent Signals
```sql
SELECT timestamp, model_id, action, confidence, price
FROM inference_signals
WHERE timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;
```

---

**Document Version:** 2.0.0
**Last Updated:** 2025-01-31
**Classification:** Internal - Engineering
**Author:** Trading Systems Team
