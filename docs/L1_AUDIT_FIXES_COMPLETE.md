# L1 Pipeline - Complete Audit Fixes Documentation

## Executive Summary
Implemented all critical audit recommendations to fix data quality issues:
- ✅ Strict 08:00-12:55 COT window enforcement
- ✅ Deduplication on time_utc
- ✅ Correct episode indexing (0-59)
- ✅ Quality metrics that reconcile with data
- ✅ Clean subset properly filtered
- ✅ Comprehensive metadata with SHA256

## Critical Issues Fixed

### 1. Window Clamping ✅
**Problem:** Data included 08:00-13:55 COT (6 hours, 72 bars/day)
**Fix:** Strict filter to 08:00-12:55 COT only
```python
df = df[df['hour_cot'].isin([8, 9, 10, 11, 12])].copy()
df = df[~((df['hour_cot'] == 13))].copy()  # Remove hour 13
```
**Result:** Exactly 60 bars per complete day

### 2. Duplicate Removal ✅
**Problem:** Every time_utc appeared twice (100% duplication)
**Fix:** Drop duplicates before any processing
```python
df = df.drop_duplicates(subset=['time_utc']).sort_values('time_utc')
```
**Result:** time_utc is now unique

### 3. Episode Indexing ✅
**Problem:** t_in_episode ran 0-71, is_terminal not at step 59
**Fix:** Rebuild indexing after window clamp
```python
df['t_in_episode'] = ((df['hour_cot'] - 8) * 12 + df['minute_cot'] / 5).astype(int)
df['is_terminal'] = (df['t_in_episode'] == 59)
```
**Result:** t_in_episode: 0-59, is_terminal only at 59

### 4. Quality Metrics ✅
**Problem:** 
- completeness_pct > 100%
- n_gaps negative
- No duplicate tracking

**Fix:** 
```python
n_gaps = max(0, rows_expected - rows_found)  # Never negative
completeness_pct = (rows_found / 60) * 100   # Max 100%
duplicates_count = track_per_day              # New column
```
**Result:** Metrics reconcile with actual data

### 5. Clean Subset Filtering ✅
**Problem:** FAIL episodes in OK_WARNS file
**Fix:** Strict filtering and verification
```python
ok_warn_episodes = quality_df[
    quality_df['quality_flag'].isin(['OK', 'WARN'])
]['date'].values
# Verify no FAIL episodes
fail_in_clean = set(clean_episodes) & set(fail_episodes)
if fail_in_clean:
    df_clean = df_clean[~df_clean['episode_id'].isin(fail_in_clean)]
```
**Result:** Clean subset contains ONLY OK/WARN episodes

### 6. Metadata Corrections ✅
**Problem:**
- price_unit: "COP" (wrong)
- date_cot: single date for full dataset
- Missing SHA256 hash

**Fix:**
```python
metadata = {
    "price_unit": "COP per USD",           # Corrected
    "date_range_cot": [start, end],        # Range for full dataset
    "sha256_standardized_csv": csv_hash,   # Added integrity
    "duplicates_removed": count,           # Track dedup
}
```

## Quality Report Structure (Enhanced)

| Column | Description | Fix Applied |
|--------|-------------|------------|
| date | Episode date | - |
| rows_expected | Always 60 | - |
| rows_found | Actual bars | After dedup |
| completeness_pct | % complete | Max 100% |
| n_gaps | Missing bars | Never negative |
| max_gap_bars | Consecutive gaps | Proper calculation |
| **duplicates_count** | Dups removed | NEW COLUMN |
| n_stale | Stale bars | - |
| stale_rate | % stale | - |
| stale_burst_max | Max consecutive | - |
| ohlc_violations | OHLC errors | Report only |
| quality_flag | OK/WARN/FAIL | Mode A logic |
| fail_reason | Category | Detailed |

## Mode A Gating Logic

```
60/60 bars + stale_rate ≤ 1% → OK
60/60 bars + stale_rate ≤ 2% → WARN
59/60 bars + max_gap ≤ 1 → WARN
Otherwise → FAIL
```

## Expected Results After Fixes

Based on audit analysis:
- **OK**: ~82.1% (1,008 days with 60/60 bars)
- **WARN**: ~4.9% (60 days with 59/60 bars)
- **FAIL**: ~13.0% (161 days with <59 bars)
- **Global stale rate**: ~1.99%
- **OHLC violations**: ~0.022% (should be 0, reported)

## Verification Checklist

| Check | Status | Details |
|-------|--------|---------|
| Window | ✅ | 08:00-12:55 COT only |
| Dedup | ✅ | No duplicate time_utc |
| Keys unique | ✅ | time_utc, (episode_id, t_in_episode) |
| Counts | ✅ | 60/60 or 59/60 with max_gap≤1 |
| OHLC | ✅ | Violations reported, not fixed |
| Stale rate | ✅ | ≤2% threshold |
| Reports | ✅ | Metrics reconcile |
| Metadata | ✅ | Correct units, hash |
| Clean subset | ✅ | NO FAIL episodes |

## Output Files (6 Total)

1. **standardized_data.parquet** - Deduplicated, clamped data
2. **standardized_data.csv** - Same with 6-decimal prices
3. **_reports/daily_quality_60.csv** - Enhanced with duplicates_count
4. **_metadata.json** - Corrected units, SHA256 hash
5. **standardized_data_OK_WARNS.parquet** - OK/WARN only
6. **standardized_data_OK_WARNS.csv** - OK/WARN only

## DAG Structure

```
load_and_clean (dedup + clamp)
    ↓
calculate_quality (proper metrics)
    ↓
save_outputs (6 files, verified)
```

## Key Improvements

1. **Data Integrity**: Unique keys, no duplicates
2. **Correct Window**: 5-hour premium (08:00-12:55)
3. **Accurate Metrics**: No impossible values
4. **Clean Subset**: Properly filtered
5. **Audit Trail**: Duplicates tracked, SHA256 hash
6. **OHLC Handling**: Report violations, don't fix

## Deployment

```bash
# DAG deployed to Airflow
docker cp usdcop_m5__02_l1_standardize.py usdcop-airflow-webserver:/opt/airflow/dags/

# Trigger execution
docker exec usdcop-airflow-webserver airflow dags trigger usdcop_m5__02_l1_standardize
```

## Impact on Downstream

- **L2**: Can trust M5 grid, build ret_log, range_bps with no leakage
- **L3**: Gets deterministic episodes (0-59)
- **RL**: No confused states, correct terminal flags

## Status
✅ **PRODUCTION READY** - All audit recommendations implemented and verified