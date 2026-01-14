# Architecture Consolidation Summary

## Overview

This document describes the consolidated architecture after eliminating duplications and establishing a Single Source of Truth (SSOT) for feature engineering.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE STORE (SSOT)                               │
│                        src/feature_store/                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │    core.py      │  │   contracts.py  │  │       adapters.py           │ │
│  │                 │  │   (Pydantic)    │  │  (Backward Compatibility)   │ │
│  │  - Calculators  │  │                 │  │                             │ │
│  │  - Contracts    │  │  - FeatureSpec  │  │  - InferenceAdapter         │ │
│  │  - Registry     │  │  - FeatureBatch │  │  - TrainingAdapter          │ │
│  │  - Builder      │  │  - NormStats    │  │  - BacktestAdapter          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│                                                                             │
│  Calculators (Wilder's Smoothing):                                          │
│  ├── RSICalculator      (alpha = 1/period)                                  │
│  ├── ATRPercentCalculator (alpha = 1/period)                                │
│  ├── ADXCalculator      (alpha = 1/period)                                  │
│  ├── LogReturnCalculator                                                    │
│  ├── MacroZScoreCalculator (rolling 60-bar window)                          │
│  └── MacroChangeCalculator                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   Training (L3 DAG)     │ │   Backtest (L4 DAG)     │ │   Inference (L5 API)    │
│                         │ │                         │ │                         │
│  Uses: UnifiedBuilder   │ │  Uses: UnifiedBuilder   │ │  Uses: UnifiedBuilder   │
│  Output: norm_stats.json│ │  Input: norm_stats.json │ │  Input: norm_stats.json │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
```

## Key Components

### 1. Feature Store Core (`src/feature_store/core.py`)

The **Single Source of Truth** for all feature calculations.

```python
from feature_store import get_contract, get_feature_builder, UnifiedFeatureBuilder

# Get V20 contract
contract = get_contract("v20")
print(contract.observation_dim)  # 15

# Build features
builder = get_feature_builder("v20")
obs = builder.build_observation(ohlcv, macro, position, timestamp, bar_idx)
```

### 2. Calculator Registry

All calculators use **Wilder's smoothing** (`alpha = 1/period`) consistently:

| Calculator | Smoothing | Period | Output Range |
|------------|-----------|--------|--------------|
| RSICalculator | Wilder's EMA | 9 | [0, 100] |
| ATRPercentCalculator | Wilder's EMA | 10 | [0, ∞) |
| ADXCalculator | Wilder's EMA | 14 | [0, 100] |
| MacroZScoreCalculator | Rolling | 60 | [-3, 3] |

### 3. Contract System

**V20 Contract (15 dimensions)**:
```python
feature_order = (
    "log_ret_5m", "log_ret_1h", "log_ret_4h",  # Returns (3)
    "rsi_9", "atr_pct", "adx_14",               # Technical (3)
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z", # Macro (7)
    "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    "position", "time_normalized"               # State (2)
)
```

## Backward Compatibility Layers

The following modules are **thin wrappers** that delegate to `feature_store`:

| Legacy Module | Delegates To | Status |
|---------------|--------------|--------|
| `src/features/contract.py` | `feature_store.core` | Deprecated |
| `src/features/calculators/*.py` | `feature_store.core` | Deprecated |
| `src/features/builder.py` | Uses calculators directly | Active |

## Files Consolidated

### Eliminated Duplications

| Category | Before | After |
|----------|--------|-------|
| Calculator implementations | 3 locations | 1 (feature_store.core) |
| Contract definitions | 4 locations | 1 (feature_store.core) |
| Config loaders | 3 classes | 1 (src/shared/config_loader.py) |
| Observation builders | 5 implementations | 2 (V20 only) |

### Archived (Legacy)

Files moved to mental archive (not deleted for safety):

- `src/features/feature_builder.py` (V19)
- `src/core/builders/observation_builder_v19.py`
- `services/inference_api/core/observation_builder_v1.py`

## Critical Fixes Applied

### 1. RSI Smoothing Consistency

**Before**: Mixed EMA methods
- Training: `ewm(alpha=1/9)` (Wilder's)
- Inference: `rolling().mean()` (SMA)

**After**: Consistent Wilder's EMA everywhere
```python
alpha = 1.0 / period
ewm(alpha=alpha, adjust=False).mean()
```

### 2. ATR Smoothing Fix

**Before**: Standard EMA
```python
ewm(span=period)  # alpha = 2/(period+1)
```

**After**: Wilder's EMA
```python
alpha = 1.0 / period
ewm(alpha=alpha, adjust=False)
```

### 3. Macro Z-Score Consistency

**Before**:
- Training: Dynamic rolling 60-bar
- Inference: Hardcoded stats from JSON

**After**: Both use rolling 60-bar window (prevents look-ahead bias)

## How to Use

### Training Pipeline
```python
from feature_store import UnifiedFeatureBuilder

builder = UnifiedFeatureBuilder("v20")
obs = builder.build_observation(ohlcv, macro, position, timestamp, bar_idx)
```

### Inference API
```python
from feature_store import get_feature_builder

builder = get_feature_builder("v20")
obs = builder.build_observation(df, macro_df, position, timestamp, bar_idx)
```

### Backtest
```python
from feature_store import UnifiedFeatureBuilder

# Uses same builder as training for perfect parity
builder = UnifiedFeatureBuilder("v20")
```

## Contract Locations (Reference)

| Contract | Location | Purpose |
|----------|----------|---------|
| Feature V20 | `feature_store.core` | Feature specification (SSOT) |
| Backtest | `airflow/dags/contracts/backtest_contracts.py` | Backtest pipeline |
| Model Registry | `services/inference_api/contracts/model_contract.py` | Model management |

## Design Patterns Used

1. **Strategy Pattern**: Calculators are interchangeable strategies
2. **Factory Pattern**: `CalculatorRegistry` creates calculators
3. **Template Method**: `BaseCalculator._calculate_impl()` hook
4. **Singleton**: `CalculatorRegistry.instance()`, `ConfigLoader`
5. **Adapter Pattern**: Backward compatibility wrappers
6. **Frozen Dataclass**: Immutable contracts (`@dataclass(frozen=True)`)

## SOLID Principles

- **S**: Each calculator has one responsibility
- **O**: Extend via registry, not modification
- **L**: All calculators implement `IFeatureCalculator`
- **I**: Minimal `IFeatureCalculator` interface
- **D**: Depend on abstractions (`BaseCalculator`)

## Validation

Run parity tests:
```bash
pytest tests/unit/test_feature_store_parity.py -v
```

Tests verify:
- RSI uses Wilder's smoothing
- ATR uses Wilder's smoothing
- ADX uses Wilder's smoothing
- Macro z-scores use rolling windows
- Feature dimensions match contract

---

**Version**: 2.0.0
**Date**: 2025-01-12
**Author**: Trading Team
