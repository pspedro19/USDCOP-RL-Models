# CHANGELOG: v14 → v15

## Overview

Version 15 fixes a critical **Training-Production Skew** caused by inconsistent z-score normalization between training and production environments.

| Metric | Before (v14) | After (v15) |
|--------|-------------|-------------|
| Z-score correlation | ~5% | ~100% |
| Observations with error > 0.5 std | 75% | <5% |
| Z-score std distribution | ~1.44 (rolling) | ~1.0 (fixed) |

---

## BREAKING CHANGES

### 1. Z-Scores: Rolling → Fixed

**Before (v14 - INCORRECT):**
```python
# Rolling z-score with window=50
z = (x - x.rolling(50).mean()) / x.rolling(50).std()
# Result: std ~1.44, correlated ~5% with production
```

**After (v15 - CORRECT):**
```python
# Fixed z-score with constants from training period (2020-03 to 2025-10)
FIXED_STATS = {
    'dxy':  {'mean': 100.21, 'std': 5.60},
    'vix':  {'mean': 21.16,  'std': 7.89},
    'embi': {'mean': 322.01, 'std': 62.68},
}
z = (x - FIXED_STATS[col]['mean']) / FIXED_STATS[col]['std']
# Result: std ~1.0, correlated ~100% with production
```

**Impact:**
- Features `dxy_z`, `vix_z`, `embi_z` now use fixed normalization
- Training and production now use identical transformations
- Model predictions are now meaningful (not garbage)

### 2. Rate Spread: Yield Curve → Sovereign Spread

**Before (v14 - INCORRECT):**
```python
rate_spread = treasury_10y - treasury_2y  # US yield curve
# Result: Not relevant for emerging market FX
```

**After (v15 - CORRECT):**
```python
rate_spread = 10.0 - treasury_10y  # Sovereign spread (Colombia 10Y - USA 10Y)
# Colombia 10Y hardcoded at 10% (typical EM rate)
# Normalized: (rate_spread - 7.03) / 1.41
```

**Impact:**
- Feature now captures carry trade dynamics
- More relevant for USD/COP price movements

---

## MIGRATION REQUIREMENTS

### 1. Regenerate Dataset (REQUIRED)

```bash
cd data/pipeline/06_rl_dataset_builder
python 01_build_5min_datasets.py
```

**Verify:**
```python
import pandas as pd
df = pd.read_csv('data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv')
for col in ['dxy_z', 'vix_z', 'embi_z']:
    std = df[col].std()
    assert 0.8 < std < 1.2, f"{col} std={std}, expected ~1.0"
print("Dataset verified: Fixed z-scores")
```

### 2. Re-train Model (REQUIRED)

```bash
cd notebooks/pipeline\ entrenamiento
python run.py
```

**Expected output:**
- 5 models: `ppo_usdcop_v15_fold{0-4}.zip`
- Training time: ~4 hours (with GPU)

### 3. Deploy SQL Views (AUTOMATIC)

SQL views in `init-scripts/03-inference-features-views.sql` already updated with:
```sql
-- Fixed z-scores (v15)
LEAST(GREATEST((dxy - 100.21) / 5.60, -4), 4) as dxy_z,
LEAST(GREATEST((vix - 21.16) / 7.89, -4), 4) as vix_z,
LEAST(GREATEST((embi - 322.01) / 62.68, -4), 4) as embi_z,
-- Sovereign spread (v15)
((10.0 - treasury_10y) - 7.03) / 1.41 as rate_spread
```

### 4. Update Model Reference (REQUIRED)

In `config/feature_config.json`:
```json
{
  "_meta": {
    "model_id": "ppo_usdcop_v15"  // Changed from v14
  }
}
```

---

## COMPATIBILITY MATRIX

| Component | v14 Model | v15 Model |
|-----------|-----------|-----------|
| v14 Dataset (rolling z-scores) | Compatible | **INCOMPATIBLE** |
| v15 Dataset (fixed z-scores) | **INCOMPATIBLE** | Compatible |
| v14 SQL Views | Compatible | **INCOMPATIBLE** |
| v15 SQL Views | **INCOMPATIBLE** | Compatible |

**WARNING:** Using a v14 model with v15 production will produce garbage predictions!

---

## FILES MODIFIED

| File | Change |
|------|--------|
| `config/feature_config.json` | Updated to v4.0.0, new norm_stats |
| `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py` | Fixed z-score function |
| `init-scripts/03-inference-features-views.sql` | Updated SQL formulas |
| `init-scripts/04-data-seeding.py` | Renamed from 03- |
| `docker/Dockerfile.data-seeder` | Updated path reference |
| `tests/integration/test_training_production_parity.py` | NEW - 39 parity tests |

---

## VALIDATION

### Run Parity Tests
```bash
pytest tests/integration/test_training_production_parity.py -v
# Expected: 39/39 passed
```

### Verify Z-Score Distribution
```python
# In production SQL
SELECT
    STDDEV(dxy_z) as dxy_std,
    STDDEV(vix_z) as vix_std,
    STDDEV(embi_z) as embi_std
FROM inference_features_5m
WHERE time > NOW() - INTERVAL '30 days';

-- Expected: All std values between 0.8 and 1.2
```

---

## ROLLBACK PROCEDURE

If v15 causes issues, rollback to v14:

1. Restore v14 model:
   ```bash
   cp models/archive/ppo_usdcop_v14_fold0.zip models/ppo_usdcop_v14_fold0.zip
   ```

2. Restore v14 SQL views:
   ```bash
   psql -f init-scripts/archive/03-inference-features-views-v14.sql
   ```

3. Update config:
   ```json
   {"_meta": {"model_id": "ppo_usdcop_v14"}}
   ```

**Note:** This is a temporary fix. The underlying skew issue will persist until v15 is properly deployed.

---

## REFERENCES

- [Training-Production Parity Analysis](../tests/integration/test_training_production_parity.py)
- [Feature Configuration SSOT](../config/feature_config.json)
- [Dataset Builder](../data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py)

---

*Last Updated: 2025-12-17*
*Author: Pedro @ Lean Tech Solutions*
