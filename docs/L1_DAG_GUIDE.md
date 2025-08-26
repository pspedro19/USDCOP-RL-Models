# L1 Layer DAG Guide

## Overview
There are two DAGs for L1 processing, each with a different purpose:

## 1. `usdcop_m5__02_l1_standardize` (Daily Processing)
**Purpose:** Process individual days with strict quality gating
**When to use:** Processing new daily data as it arrives
**Behavior:** 
- Only saves files if quality thresholds are met (≥59/60 bars, ≤2% stale rate)
- Rejects days with poor quality
- Designed for incremental daily processing

### Files saved (only if quality passes):
- `market=usdcop/timeframe=m5/date={date}/standardized_data.parquet`
- `market=usdcop/timeframe=m5/date={date}/standardized_data.csv`
- `_reports/date={date}/daily_quality_60.csv`
- Additional audit and control files

### Example usage:
```bash
# Process data for a specific date
docker exec usdcop-airflow-webserver airflow dags trigger \
  usdcop_m5__02_l1_standardize --exec-date 2025-08-15
```

## 2. `usdcop_m5__02_l1_consolidate` (Batch Consolidation)
**Purpose:** Consolidate ALL historical data into final 4-file output
**When to use:** Creating the complete L1 dataset for training/analysis
**Behavior:** 
- **ALWAYS saves all 4 files regardless of quality issues**
- Processes all available data in one run
- Generates comprehensive quality report for all days

### Files saved (ALWAYS, guaranteed):
1. `usdcop_m5__02_l1_consolidate/consolidated/standardized_data.parquet`
   - All data with exactly 13 columns
   
2. `usdcop_m5__02_l1_consolidate/consolidated/standardized_data.csv`
   - Same data with prices formatted to 6 decimals
   
3. `usdcop_m5__02_l1_consolidate/consolidated/_reports/daily_quality_60.csv`
   - Quality metrics for each day (10 columns)
   
4. `usdcop_m5__02_l1_consolidate/consolidated/_metadata.json`
   - Dataset metadata with 9+ required fields

### Example usage:
```bash
# Consolidate all available data
docker exec usdcop-airflow-webserver airflow dags trigger \
  usdcop_m5__02_l1_consolidate
```

## Required Column Structure (13 columns)
1. `episode_id` - Date in YYYY-MM-DD format (COT)
2. `t_in_episode` - Step within episode (0-59)
3. `is_terminal` - True only for last bar of episode
4. `time_utc` - UTC timestamp
5. `time_cot` - Colombia time (UTC-5)
6. `hour_cot` - Hour in COT (8-12)
7. `minute_cot` - Minute in COT (0,5,10,...,55)
8. `open` - Open price
9. `high` - High price
10. `low` - Low price
11. `close` - Close price
12. `ohlc_valid` - OHLC coherence check
13. `is_stale` - Stale bar indicator (O=H=L=C)

## Quality Report Structure (10 columns)
1. `date` - Date in YYYY-MM-DD
2. `rows_expected` - Always 60
3. `rows_found` - Actual rows found
4. `completeness_pct` - Percentage of expected rows
5. `n_stale` - Number of stale bars
6. `stale_rate` - Percentage of stale bars
7. `n_gaps` - Number of missing bars
8. `max_gap_bars` - Maximum consecutive gap
9. `ohlc_violations` - Count of OHLC coherence violations
10. `quality_flag` - OK/WARN/FAIL

## Key Differences

| Aspect | `l1_standardize` | `l1_consolidate` |
|--------|------------------|------------------|
| **Scope** | Single day | All historical data |
| **Quality Gating** | Strict (rejects bad days) | None (saves all data) |
| **Output** | Conditional | Guaranteed 4 files |
| **Use Case** | Daily incremental | Batch processing |
| **File Location** | By date partition | Consolidated folder |

## Recommendations
- **For RL Training:** Use `l1_consolidate` to get all data regardless of quality
- **For Production:** Use `l1_standardize` for daily updates with quality control
- **For Analysis:** Use `l1_consolidate` to ensure you have the complete dataset

## Current Data Stats (as of 2025-08-21)
- **Total rows:** 72,139
- **Days processed:** 1,229
- **Quality breakdown:**
  - OK: 931 days (75.8%)
  - WARN: 4 days (0.3%)
  - FAIL: 294 days (23.9%)
- **Date range:** 2020-01-02 to 2025-08-15

## MinIO Locations
- **Consolidated data:** `s3://ds-usdcop-standardize/usdcop_m5__02_l1_consolidate/consolidated/`
- **Daily data:** `s3://ds-usdcop-standardize/usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date={date}/`