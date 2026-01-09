# USD/COP Trading System - Test Suite

**Created:** 2025-12-16
**Coverage Goal:** Critical path coverage for feature parity
**Current Status:** 0% → Target coverage with critical tests

## Overview

This test suite ensures **feature parity** between the new FeatureCalculator service and legacy training code. The most critical requirement is that features computed during inference match those used during training to within `1e-6` tolerance.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and pytest config
├── unit/                          # Unit tests (fast, no dependencies)
│   ├── test_feature_builder.py   # Feature calculation logic (~150 lines)
│   ├── test_config_loader.py     # Config validation (~150 lines)
│   └── test_normalization.py     # Normalization logic (~120 lines)
├── integration/                   # Integration tests (require data)
│   ├── test_feature_parity.py    # CRITICAL: Legacy comparison (~180 lines)
│   └── test_observation_space.py # Model compatibility (~150 lines)
└── fixtures/                      # Test data from real project
    ├── sample_ohlcv.csv          # 100 bars of OHLCV data
    └── sample_macro.csv          # 10 days of macro data
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=services --cov-report=html
```

### Specific Test Categories

```bash
# Critical feature parity test
pytest tests/integration/test_feature_parity.py::TestFeatureParityLegacy::test_features_match_legacy -v

# Observation space validation
pytest tests/integration/test_observation_space.py -v

# Normalization tests (including corrected USDMXN clip)
pytest tests/unit/test_normalization.py::TestFeatureClipping::test_usdmxn_clip_corrected -v

# Config validation
pytest tests/unit/test_config_loader.py -v
```

### Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Critical Tests

### 1. Feature Parity (MOST IMPORTANT)

**File:** `tests/integration/test_feature_parity.py`

Tests that new FeatureCalculator produces identical results to legacy code.

```python
def test_features_match_legacy():
    """Compare with training data - tolerance 1e-6"""
    for feature in ['log_ret_5m', 'rsi_9', 'atr_pct', 'adx_14']:
        diff = (legacy[feature] - new[feature]).abs().max()
        assert diff < 1e-6, f"{feature}: max diff = {diff}"
```

**Why critical:** If features don't match training data, model predictions will drift.

### 2. Observation Dimension

**File:** `tests/integration/test_observation_space.py`

```python
def test_observation_is_15_dimensions():
    """Observation must be EXACTLY 15 dimensions"""
    obs = build_observation(features, position=0.0, step=30)
    assert obs.shape == (15,)
```

**Why critical:** Model expects exactly 15 dimensions. Wrong size = runtime error.

### 3. Time Normalization Range

**File:** `tests/unit/test_feature_builder.py`

```python
def test_time_normalized_range():
    """time_normalized in [0, 0.983], NOT [0, 1]"""
    for bar in range(1, 61):
        tn = (bar - 1) / 60
        assert 0 <= tn <= 0.983
    # Bar 60 = 0.983, NOT 1.0
    assert abs((60 - 1) / 60 - 0.983) < 0.001
```

**Why critical:** Matches training environment behavior exactly.

### 4. USDMXN Clip Corrected

**File:** `tests/unit/test_normalization.py`

```python
def test_usdmxn_clip_corrected():
    """usdmxn_ret_1h clip CORRECTED to [-0.10, 0.10]"""
    clipped = calc_pct_change(series, 12, clip_range=(-0.1, 0.1))
    assert clipped.max() <= 0.10  # Was 0.05, corrected to 0.10
```

**Why critical:** Bug fix from config validation - must use corrected values.

## Test Coverage Metrics

| Category | Tests | Lines | Critical? |
|----------|-------|-------|-----------|
| Feature Calculation | 8 | ~150 | ✅ Yes |
| Config Validation | 12 | ~150 | ⚠️ Medium |
| Normalization | 9 | ~120 | ✅ Yes |
| Feature Parity | 7 | ~180 | ⭐ CRITICAL |
| Observation Space | 10 | ~150 | ✅ Yes |
| **TOTAL** | **46** | **~750** | |

## Success Criteria

| Metric | Threshold | Status |
|--------|-----------|--------|
| Feature parity | diff < 1e-6 | To verify |
| Observation dimension | == 15 | ✅ |
| time_normalized range | [0, 0.983] | ✅ |
| Model compatibility | No errors | To verify |
| Config validation | All sections present | ✅ |

## Fixtures

### sample_ohlcv.csv
- **Source:** `data/archive/PASS/OUTPUT_RL/RL_DS3_MACRO_CORE.csv`
- **Rows:** 100 bars (5-minute)
- **Date range:** 2020-03-02 to 2020-03-03
- **Columns:** time, open, high, low, close

### sample_macro.csv
- **Source:** Generated from realistic values
- **Rows:** 10 days
- **Date range:** 2020-03-02 to 2020-03-11
- **Columns:** date, dxy, vix, embi, brent, treasury_2y, treasury_10y, usdmxn

## Common Issues

### 1. Legacy Dataset Not Found

```
pytest.skip("Legacy dataset not available for parity testing")
```

**Solution:** Ensure `data/archive/PASS/OUTPUT_RL/RL_DS3_MACRO_CORE.csv` exists.

### 2. Model Not Available

```
pytest.skip("Model not available for testing")
```

**Solution:** Tests will skip if model file not present. This is OK for unit tests.

### 3. Feature Parity Fails

```
AssertionError: rsi_9: max diff = 5.2e-05 (threshold 1e-6)
```

**Solution:** Check RSI calculation formula matches legacy code exactly.

## Dependencies

```bash
# Core testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Data handling
pandas>=1.5.0
numpy>=1.23.0

# Optional (for model tests)
stable-baselines3>=2.0.0
```

Install:
```bash
pip install -r tests/requirements-test.txt
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r tests/requirements-test.txt
      - name: Run tests
        run: pytest tests/ --cov=services --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest tests/unit/ -v || exit 1
```

## Maintenance

### Adding New Features

1. Add feature to `config/feature_config.json`
2. Implement in `services/feature_calculator.py`
3. Add unit test in `tests/unit/test_feature_builder.py`
4. Add parity test in `tests/integration/test_feature_parity.py`
5. Update observation order test if needed

### Updating Config

1. Modify `config/feature_config.json`
2. Update tests in `tests/unit/test_config_loader.py`
3. Run full test suite to verify compatibility

## References

- **Architecture Doc:** `docs/ARQUITECTURA_INTEGRAL_V3.md` (Section 11.4)
- **Migration Map:** `docs/MAPEO_MIGRACION_BIDIRECCIONAL.md` (Part 8)
- **Feature Config:** `config/feature_config.json`
- **Legacy Data:** `data/archive/PASS/OUTPUT_RL/RL_DS3_MACRO_CORE.csv`

## Contact

For questions or issues with the test suite:
- Author: Pedro @ Lean Tech Solutions
- Date: 2025-12-16
- Related to: OBJETIVO 5 - Testing Suite Creation
