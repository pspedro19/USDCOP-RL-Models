# Code Audit Report - USDCOP-RL-Models
## Date: 2026-01-17

---

## Executive Summary

This audit identifies deprecated code, dead code, duplicate implementations, and redundant files across the USDCOP-RL-Models codebase. The goal is to achieve a professional, clean folder structure with single sources of truth (SSOT) for all major components.

---

## Critical Findings

### 1. FeatureBuilder Implementations (CRITICAL - 5+ duplicates)

| File | Status | Action |
|------|--------|--------|
| `src/feature_store/builders/canonical_feature_builder.py` | **CANONICAL SSOT** | KEEP |
| `src/core/services/feature_builder.py` | Deprecated wrapper | KEEP (delegates to canonical) |
| `src/core/services/feature_builder_refactored.py` | Redundant | DELETE |
| `src/services/backtest_feature_builder.py` | Duplicate logic | REVIEW/DELETE |
| `archive/legacy_v19/feature_builder.py` | Legacy | DELETE (with archive) |

**Recommendation**: Keep only `canonical_feature_builder.py` as SSOT. The wrapper in `feature_builder.py` can stay for backward compatibility but is deprecated.

---

### 2. TradingFlags Implementations (CRITICAL - 2 duplicates)

| File | Lines | Status | Action |
|------|-------|--------|--------|
| `src/config/trading_flags.py` | 876 | **Full SSOT** with TradingFlagsDB | KEEP |
| `src/trading/trading_flags.py` | 313 | Simpler implementation | DELETE |

**Analysis**:
- `src/config/trading_flags.py`: Complete implementation with:
  - TradingMode enum
  - Environment enum
  - Frozen TradingFlags dataclass
  - Thread-safe singleton
  - Database-backed TradingFlagsDB
  - Kill switch activation/deactivation
  - Production validation

- `src/trading/trading_flags.py`: Simpler but redundant:
  - Basic TradingFlags dataclass (not frozen)
  - No TradingMode enum
  - No database support
  - Duplicate functionality

**Recommendation**: DELETE `src/trading/trading_flags.py`, update imports to use `src/config/trading_flags.py`.

---

### 3. InferenceEngine Implementations (HIGH - 4 duplicates)

| File | Context | Action |
|------|---------|--------|
| `services/inference_api/core/inference_engine.py` | FastAPI service | KEEP (service-specific) |
| `src/inference/inference_engine.py` | Core library | REVIEW |
| `src/models/inference_engine.py` | Models package | DELETE if redundant |
| `services/mlops/inference_engine.py` | MLOps service | REVIEW |

**Recommendation**: Consolidate to one core implementation + one service adapter.

---

### 4. Trading API Implementations (HIGH - 3 duplicates)

| File | Purpose | Action |
|------|---------|--------|
| `services/trading_analytics_api.py` | Analytics API | KEEP |
| `services/trading_api_realtime.py` | Realtime trading | DELETE or merge |
| `services/trading_api_multi_model.py` | Multi-model | DELETE or merge |

**Recommendation**: Merge functionality into a single API with feature flags.

---

### 5. Archive Folder (MEDIUM - Should be removed)

The `archive/` folder contains legacy code that should be deleted:

```
archive/
├── airflow-dags/
│   ├── deprecated/          # 50+ deprecated DAG files
│   └── legacy-factories/    # Old DAG factories
├── infrastructure-legacy/   # Old API routers
├── services-deprecated/     # Old services
├── notebooks/v11/           # Legacy notebooks
├── notebooks_legacy/        # More legacy notebooks
├── v16_v17_legacy/          # Dataset builders v16/v17
└── legacy_v19/              # FeatureBuilder v19
```

**Total Files**: ~100+ Python files in archive
**Recommendation**: DELETE entire `archive/` folder (it's already archived)

---

## Files to DELETE Immediately

### Priority 1 - Redundant Duplicates
```
src/trading/trading_flags.py                 # Duplicate of src/config/trading_flags.py
src/core/services/feature_builder_refactored.py  # Redundant
```

### Priority 2 - Deprecated Services
```
services/trading_api_realtime.py             # Merge into main API
services/trading_api_multi_model.py          # Merge into main API
```

### Priority 3 - Archive Cleanup (if not needed for reference)
```
archive/                                      # Entire folder
```

---

## Recommended Folder Structure

```
USDCOP-RL-Models/
├── .github/
│   └── workflows/           # CI/CD pipelines
├── airflow/
│   └── dags/                # Production DAGs only
├── config/
│   ├── feature_config.json  # Feature configuration
│   ├── norm_stats.json      # Normalization statistics
│   └── *.yaml               # Other configs
├── database/
│   ├── alembic/             # Database migrations
│   └── migrations/          # SQL migrations
├── docker/
│   └── Dockerfile.*         # Docker configurations
├── docs/
│   └── *.md                 # Documentation
├── feature_repo/
│   └── *.py                 # Feast feature definitions
├── scripts/
│   └── *.py                 # Utility scripts
├── services/
│   ├── inference_api/       # Main inference API
│   │   ├── core/
│   │   ├── middleware/
│   │   └── routers/
│   ├── mlops/               # MLOps utilities
│   └── shared/              # Shared service utilities
├── src/
│   ├── backtest/            # Backtesting engine
│   ├── config/              # Configuration (SSOT for TradingFlags)
│   │   └── trading_flags.py
│   ├── core/
│   │   ├── constants.py
│   │   ├── container/       # Dependency injection
│   │   ├── contracts/       # Contract definitions
│   │   ├── decorators/
│   │   ├── events/
│   │   ├── logging/
│   │   ├── secrets/
│   │   └── services/        # Core services
│   ├── database/
│   ├── feature_store/
│   │   ├── builders/        # SSOT: CanonicalFeatureBuilder
│   │   │   └── canonical_feature_builder.py
│   │   └── readers/
│   ├── features/
│   ├── inference/           # Inference logic
│   ├── models/              # Model definitions
│   ├── monitoring/          # Drift detection
│   ├── risk/                # Risk management
│   ├── shared/              # Shared utilities
│   ├── trading/             # Trading logic (NOT flags)
│   ├── training/            # Training logic
│   ├── utils/
│   └── validation/          # Validation & testing
├── tests/
│   ├── chaos/
│   ├── contracts/
│   ├── integration/
│   ├── load/
│   ├── regression/
│   └── unit/
└── usdcop-trading-dashboard/
```

---

## Import Updates Required

After deleting `src/trading/trading_flags.py`, update these imports:

### Files using `src.trading.trading_flags`:
```python
# FROM:
from src.trading.trading_flags import TradingFlags, load_trading_flags

# TO:
from src.config.trading_flags import TradingFlags, get_trading_flags
```

### Files using old FeatureBuilder:
```python
# FROM:
from src.core.services.feature_builder import FeatureBuilder

# TO (for new code):
from src.feature_store.builders import CanonicalFeatureBuilder
builder = CanonicalFeatureBuilder.for_inference()
```

---

## Action Items

### Immediate (P0)
1. [x] Create this audit report
2. [ ] Delete `src/trading/trading_flags.py`
3. [ ] Delete `src/core/services/feature_builder_refactored.py`
4. [ ] Update imports in affected files

### Short-term (P1)
5. [ ] Review and consolidate InferenceEngine implementations
6. [ ] Merge trading API implementations
7. [ ] Update `src/trading/__init__.py` exports

### Medium-term (P2)
8. [ ] Delete archive folder (after backup verification)
9. [ ] Clean up docker-compose.yml commented services
10. [ ] Update all __init__.py exports

---

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Python files in archive | ~100 | 0 | -100 |
| FeatureBuilder implementations | 5 | 1 (+ 1 wrapper) | -3 |
| TradingFlags implementations | 2 | 1 | -1 |
| InferenceEngine implementations | 4 | 2 | -2 |
| Trading API files | 3 | 1 | -2 |

---

*Report generated: 2026-01-17*
*Auditor: Claude Code Assistant*
