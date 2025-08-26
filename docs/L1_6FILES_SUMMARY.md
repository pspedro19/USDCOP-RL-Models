# L1 Pipeline - 6 Files Output Summary

## ✅ COMPLETE: Unified DAG with 6 Output Files

### DAG Name
`usdcop_m5__02_l1_standardize` (single unified version)

### Output Files (ALWAYS 6)

| # | File | Format | Description | Rows |
|---|------|--------|-------------|------|
| 1 | `standardized_data.parquet` | Parquet | Full dataset with Mode B padding | 86,272+ |
| 2 | `standardized_data.csv` | CSV | Full dataset, 6-decimal prices | 86,272+ |
| 3 | `_reports/daily_quality_60.csv` | CSV | Enhanced quality report (13 cols) | 1,230 |
| 4 | `_metadata.json` | JSON | Metadata with SHA256 hash | - |
| 5 | `standardized_data_OK_WARNS.parquet` | Parquet | Clean subset (OK/WARN only) | 11,564 |
| 6 | `standardized_data_OK_WARNS.csv` | CSV | Clean subset, 6-decimal prices | 11,564 |

### Key Features Implemented

#### 1. Mode B Padding ✅
- Pads episodes with exactly 59 bars
- Adds `is_missing=True` flag for padded rows
- Uses NaN for OHLC prices in placeholders
- Result: 139 episodes upgraded from FAIL to WARN

#### 2. Enhanced Quality Report ✅
13 columns in `daily_quality_60.csv`:
- `date`, `rows_expected`, `rows_found`, `rows_padded`
- `completeness_pct`, `n_stale`, `stale_rate`, `stale_burst_max`
- `n_gaps`, `max_gap_bars`, `ohlc_violations`
- `quality_flag`, `fail_reason`

#### 3. Fail Reason Categories ✅
- `PASS` - Meets criteria
- `SINGLE_MISSING_PADDED` - Mode B applied
- `INSUFFICIENT_BARS` - 30-58 bars
- `INSUFFICIENT_BARS_SEVERE` - <30 bars
- `HIGH_STALE_RATE` - 2-10% stale
- `HIGH_STALE_RATE_SEVERE` - >10% stale
- `MODERATE_STALE_RATE` - 1-2% stale
- `NO_DATA` - No data for episode

#### 4. Clean Subset (2 formats) ✅
- Parquet format for efficient processing
- CSV format for compatibility
- Only OK and WARN episodes
- 86.6% data reduction
- Ideal for production training

#### 5. SHA256 Hash ✅
- In `_metadata.json` as `data_hash`
- Cryptographic integrity verification
- Hash of entire dataset

### File Sizes
- `standardized_data.parquet`: 3.10 MB
- `standardized_data.csv`: 10.87 MB
- `standardized_data_OK_WARNS.parquet`: 0.46 MB
- `standardized_data_OK_WARNS.csv`: 1.50 MB
- `_reports/daily_quality_60.csv`: 73 KB
- `_metadata.json`: <1 KB

### Quality Summary
- **OK**: 50 episodes (4.1%)
- **WARN**: 139 episodes (11.3%) - includes Mode B
- **FAIL**: 1,040 episodes (84.6%)

### Clean Subset Impact
- Full dataset: 86,272 rows
- Clean subset: 11,564 rows
- Episodes included: 189 (OK + WARN)
- Data reduction: 86.6%

### MinIO Location
```
s3://ds-usdcop-standardize/usdcop_m5__02_l1_standardize/consolidated/
```

### Local Test Outputs
```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_6files\
```

### DAG File
```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\airflow\dags\usdcop_m5__02_l1_standardize.py
```

### Trigger Command
```bash
docker exec usdcop-airflow-webserver airflow dags trigger usdcop_m5__02_l1_standardize
```

## Status
✅ **PRODUCTION READY** - Single unified DAG with 6 guaranteed output files including clean subset in both parquet and CSV formats

## Audit Compliance
All recommendations implemented:
- ✅ Mode B padding
- ✅ Fail reason tracking
- ✅ Stale burst detection
- ✅ Edge-minute hygiene (08:00-12:55 COT)
- ✅ OHLC violations report-only
- ✅ Clean subset in parquet
- ✅ Clean subset in CSV (additional)
- ✅ SHA256 hash
- ✅ Always saves 6 files