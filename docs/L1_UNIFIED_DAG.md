# Unified L1 Standardization DAG

## Overview
Single unified DAG `usdcop_m5__02_l1_standardize` that incorporates all audit recommendations and ALWAYS outputs 5 files regardless of quality issues.

## DAG Name
`usdcop_m5__02_l1_standardize` (single unified version)

## Key Features

### 1. Mode B Padding ✅
- Automatically detects episodes with exactly 59 bars
- Pads single missing slot with placeholder row
- Sets `is_missing=True` flag for padded rows
- Uses NaN for OHLC prices in padded rows

### 2. Enhanced Quality Report ✅
The `daily_quality_60.csv` now includes 13 columns:
1. `date` - Episode date
2. `rows_expected` - Always 60
3. `rows_found` - Actual data rows (excluding padded)
4. `rows_padded` - Number of Mode B padded rows
5. `completeness_pct` - Percentage of actual data
6. `n_stale` - Count of stale bars
7. `stale_rate` - Percentage of stale bars
8. `stale_burst_max` - Maximum consecutive stale bars
9. `n_gaps` - Number of missing bars
10. `max_gap_bars` - Maximum gap size
11. `ohlc_violations` - Count of OHLC coherence failures
12. `quality_flag` - OK/WARN/FAIL
13. `fail_reason` - Categorical failure reason

### 3. Fail Reason Categories ✅
- `PASS` - Meets all quality criteria
- `SINGLE_MISSING_PADDED` - Mode B applied
- `INSUFFICIENT_BARS` - 30-58 bars
- `INSUFFICIENT_BARS_SEVERE` - <30 bars
- `HIGH_STALE_RATE` - 2-10% stale
- `HIGH_STALE_RATE_SEVERE` - >10% stale
- `MODERATE_STALE_RATE` - 1-2% stale
- `NO_DATA` - Episode has no data

### 4. SHA256 Hash ✅
- Added to `_metadata.json` as `data_hash` field
- Provides cryptographic verification of data integrity
- Hash of entire dataset JSON representation

### 5. Clean Subset File ✅
- New file: `standardized_data_OK_WARNS.parquet`
- Contains only OK and WARN quality episodes
- Ideal for production model training
- Excludes all FAIL episodes

## Output Files (ALWAYS 6)

1. **standardized_data.parquet**
   - Full dataset with Mode B padding
   - 14 columns (13 standard + `is_missing`)

2. **standardized_data.csv**
   - CSV version with 6-decimal price precision
   - Same structure as parquet

3. **_reports/daily_quality_60.csv**
   - Enhanced quality report with 13 columns
   - Includes fail reasons and stale burst metrics

4. **_metadata.json**
   - Dataset metadata with SHA256 hash
   - 11+ fields including quality summary

5. **standardized_data_OK_WARNS.parquet**
   - Clean subset for production use
   - Only OK and WARN episodes

6. **standardized_data_OK_WARNS.csv**
   - Clean subset in CSV format
   - 6-decimal price precision
   - Same data as parquet clean subset

## DAG Tasks

```
load_all_data
    ↓
apply_mode_b (padding)
    ↓
calculate_quality (enhanced)
    ↓
save_all_outputs (6 files)
```

## Usage

### Trigger via Airflow UI
1. Navigate to http://localhost:8081
2. Find `usdcop_m5__02_l1_standardize`
3. Click trigger button

### Trigger via CLI
```bash
docker exec usdcop-airflow-webserver airflow dags trigger usdcop_m5__02_l1_standardize
```

### Monitor Progress
```bash
docker exec usdcop-airflow-webserver airflow dags state usdcop_m5__02_l1_standardize <execution_date>
```

## MinIO Output Location
```
s3://ds-usdcop-standardize/usdcop_m5__02_l1_standardize/consolidated/
├── standardized_data.parquet
├── standardized_data.csv
├── standardized_data_OK_WARNS.parquet
├── standardized_data_OK_WARNS.csv
├── _metadata.json
└── _reports/
    └── daily_quality_60.csv
```

## Quality Thresholds
- **OK**: ≥59 bars, ≤1% stale rate
- **WARN**: ≥59 bars, 1-2% stale rate OR Mode B padded
- **FAIL**: <59 bars OR >2% stale rate

## Data Version
`v1.1-audit` - Includes all audit enhancements

## Key Improvements
1. **Guaranteed Output**: Always saves 6 files
2. **Data Recovery**: Mode B padding for single missing slots
3. **Clear Diagnostics**: Fail reasons for every episode
4. **Production Ready**: Clean subset in both parquet and CSV
5. **Data Integrity**: SHA256 hash verification
6. **Comprehensive Metrics**: Stale burst detection

## File Location
```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\airflow\dags\usdcop_m5__02_l1_standardize.py
```

## Status
✅ **PRODUCTION READY** - All audit recommendations implemented in single unified DAG