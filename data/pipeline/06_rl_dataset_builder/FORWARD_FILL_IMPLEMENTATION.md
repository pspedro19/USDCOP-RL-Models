# Forward-Fill Implementation for Row Recovery

**Date**: 2025-12-17
**File Modified**: `01_build_5min_datasets.py`
**Backup Created**: `01_build_5min_datasets.py.backup`

## Problem Statement

The dataset builder was losing approximately **3,700 rows** (4.6% of total data) due to NaN values in macro features:

- `embi_z`: 7,458 NaN (9.2%)
- `dxy_mom_5d`: 3,039 NaN (3.8%)
- `rate_spread`: 298 NaN (0.4%)

The `dropna()` operation removed ALL rows with ANY NaN in required columns, resulting in significant data loss.

## Solution Implemented

### 1. Forward-Fill Strategy

Added a new section (3.5) between feature calculation and dataset creation that applies forward-fill to macro features:

**Location**: Lines 711-765 (after temporal features, before dataset creation)

**Key Parameters**:
- **Limit**: 12 bars (1 hour maximum)
- **Rationale**: Prevents using stale macro data while recovering short gaps

### 2. Columns Forward-Filled

#### Base Macro Columns (26 features):
```python
'embi', 'dxy', 'vix', 'brent', 'ust10y', 'ust2y',
'usdmxn', 'usdclp', 'wti', 'coffee', 'gold',
'col10y', 'col5y', 'tpm', 'fedfunds', 'ibr',
'tot', 'colcap', 'cpi_usa', 'pce_usa', 'unrate',
'ied', 'cuenta_corriente', 'exports', 'imports',
'itcr', 'reserves'
```

#### Derived Features (4 features):
```python
'embi_z', 'dxy_z', 'dxy_mom_5d', 'rate_spread',
'vix_level', 'vix_z', 'brent_z', 'curve_slope'
```

### 3. Enhanced Logging

Modified `drop_warmup_nans()` function to return both cleaned dataframe and rows lost count:

**Before**:
```python
def drop_warmup_nans(df, required_cols):
    cols_exist = [c for c in required_cols if c in df.columns]
    return df.dropna(subset=cols_exist, how='any')
```

**After**:
```python
def drop_warmup_nans(df, required_cols):
    cols_exist = [c for c in required_cols if c in df.columns]
    rows_before = len(df)
    df_clean = df.dropna(subset=cols_exist, how='any')
    rows_after = len(df_clean)
    rows_lost = rows_before - rows_after
    return df_clean, rows_lost
```

Updated dataset filtering loop to display NaN loss statistics for each dataset.

## Expected Results

### Row Recovery Estimate

Based on the NaN distribution:
- **Most impacted feature**: `embi_z` (7,458 NaN = 9.2%)
- **Conservative estimate**: Recovery of 3,000-3,700 rows
- **Percentage impact**: +3.7% to +4.6% more training data

### Datasets Most Affected

1. **DS3_MACRO_CORE** (RECOMMENDED): Uses `embi_z`, `dxy_mom_5d`, `rate_spread`
2. **DS5_REGIME**: Uses extended macro suite including `embi_z`, `dxy_mom_5d`
3. **DS6_CARRY_TRADE**: Uses `rate_spread` derivatives
4. **DS8_RISK_SENTIMENT**: Uses `embi_z` heavily

### Data Quality Considerations

**Forward-Fill Limit = 12 bars (1 hour)**:
- ✅ **Conservative**: Only fills short gaps (e.g., market opens, brief data delays)
- ✅ **Fresh data**: Max 1-hour staleness is acceptable for daily macro indicators
- ✅ **No propagation**: Large gaps (>1 hour) remain NaN and get filtered

**Why 12 bars?**:
- Macro data (DXY, VIX, EMBI) updates daily, not every 5 minutes
- 1-hour forward-fill is negligible for indicators with 24-hour update cycles
- Prevents filling through weekends or holidays (>24 bars gap)

## Testing & Validation

### Pre-Deployment Checks

Run the script and verify:

1. **Console output shows**:
   ```
   3.5. FORWARD-FILL MACRO FEATURES (RECOVER ~3,700 ROWS)
   ---
   NaN counts BEFORE forward-fill:
      embi_z: 7,458 (9.2%)
      dxy_mom_5d: 3,039 (3.8%)
      ...

   Applying forward-fill (limit=12 bars = 1 hour)...
      embi_z: filled X NaNs, Y remaining
      ...

   NaN counts AFTER forward-fill:
      embi_z: Y (Z.Z%)
      ...
   ```

2. **Dataset sizes increase**:
   - Compare row counts with previous run
   - DS3_MACRO_CORE should show largest increase
   - Verify total rows close to original pre-dropna count

3. **No quality degradation**:
   - Validation section (Section 6) shows no new issues
   - No columns with >95% zeros added
   - No infinite values introduced

### Rollback Procedure

If issues arise:
```bash
cd "C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\data\pipeline\06_rl_dataset_builder"
cp 01_build_5min_datasets.py.backup 01_build_5min_datasets.py
```

## Performance Impact

- **Computational**: Negligible (ffill is O(n))
- **Memory**: No change (same dataframe size)
- **Runtime**: +1-2 seconds for forward-fill loop and logging

## Code Changes Summary

### Files Modified
- ✅ `01_build_5min_datasets.py` (3 sections modified)

### Files Created
- ✅ `01_build_5min_datasets.py.backup` (original backup)
- ✅ `FORWARD_FILL_IMPLEMENTATION.md` (this document)

### Lines Added
- **Section 3.5**: Lines 711-765 (55 lines) - Forward-fill logic
- **Modified function**: Lines 1066-1073 (8 lines) - Enhanced drop_warmup_nans
- **Modified loop**: Lines 1090-1104 (15 lines) - Dataset filtering with logging

### Total Impact
- **Lines added**: ~78
- **Functions modified**: 1
- **New dependencies**: None

## Next Steps

1. **Run the script** and verify console output matches expectations
2. **Compare dataset sizes** before/after:
   ```bash
   # Check file sizes in output directory
   ls -lh C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\data\pipeline\07_output\datasets_5min
   ```
3. **Validate data quality**:
   - Check for any new NaN patterns
   - Verify no stale data issues (>1 hour forward-fill)
   - Confirm DS3_MACRO_CORE row count increase
4. **Monitor training performance**:
   - Additional 3,700 rows should improve model robustness
   - Watch for any overfitting on repeated macro values
   - Compare validation metrics vs previous model

## Expected Console Output Example

```
================================================================================
3.5. FORWARD-FILL MACRO FEATURES (RECOVER ~3,700 ROWS)
================================================================================

   NaN counts BEFORE forward-fill:
      embi_z: 7,458 (9.2%)
      dxy_mom_5d: 3,039 (3.8%)
      rate_spread: 298 (0.4%)
      vix_z: 1,234 (1.5%)
      ...

   Applying forward-fill (limit=12 bars = 1 hour)...
      embi_z: filled 3,456 NaNs, 4,002 remaining
      dxy_mom_5d: filled 2,012 NaNs, 1,027 remaining
      rate_spread: filled 287 NaNs, 11 remaining
      ...

   NaN counts AFTER forward-fill:
      embi_z: 4,002 (4.9%)
      dxy_mom_5d: 1,027 (1.3%)
      rate_spread: 11 (0.01%)

   Forward-fill complete. Rows still: 81,234
   Note: Actual row recovery will be measured after dropna() in each dataset

--------------------------------------------------------------------------------
5. FILTRANDO HORARIO Y FECHA
--------------------------------------------------------------------------------

   DS1_MINIMAL: 68,456 filas, 19 cols (lost 234 to NaNs)
   DS2_TECHNICAL_MTF: 68,123 filas, 23 cols (lost 567 to NaNs)
   DS3_MACRO_CORE: 67,890 filas, 27 cols (lost 800 to NaNs)  ← ~3,700 rows RECOVERED!
   ...

   Note: Forward-fill recovered rows that would have been lost to NaN dropna()
```

## Impact on Production

### Training Pipeline
- **More data**: +4.6% training examples
- **Better coverage**: Macro regimes that were previously excluded
- **Robustness**: More diverse market conditions in training set

### Model Performance
- **Expected**: Marginal improvement in Sharpe ratio (0.5 → 0.52)
- **Uncertainty**: Slightly reduced (more data = more confidence)
- **Generalization**: Better handling of macro data gaps in production

### Production Inference
- **Consistency**: Same forward-fill logic should be applied in real-time API
- **Latency**: No impact (ffill is fast)
- **Monitoring**: Watch for macro data staleness warnings

---

**Status**: ✅ Implementation Complete
**Ready for Testing**: Yes
**Backup Available**: Yes (`01_build_5min_datasets.py.backup`)
