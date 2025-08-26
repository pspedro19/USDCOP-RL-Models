# L2 Implementation Plan - PREPARE Layer

## Overview
L2 takes the clean, standardized data from L1 and prepares ML-ready features with deseasonalization and quality controls.

## Input Files from L1
```
From: ds-usdcop-standardize/usdcop_m5__02_l1_standardize/consolidated/
- standardized_data_accepted.parquet  (55,705 rows, 929 episodes)
- _statistics/hod_baseline.csv        (HOD medians/MAD for deseasonalization)
- _metadata.json                       (verification & hashes)
- _reports/daily_quality_60.csv       (optional, for telemetry)
```

## Output Structure for L2
```
To: ds-usdcop-prepare/usdcop_m5__03_l2_prepare/
├── market=usdcop/timeframe=m5/date=YYYY-MM-DD/run_id=UUID/
│   ├── prepared_premium.parquet              # Main dataset (masked, 60 slots)
│   ├── prepared_premium.csv                  # CSV version
│   ├── prepared_premium_strict.parquet       # Only 60/60 episodes
│   ├── prepared_premium_strict.csv
│   └── _control/READY
├── _statistics/date=YYYY-MM-DD/
│   ├── hod_stats.parquet                     # Effective HOD used
│   ├── winsor_params.json                    # Winsorization parameters
│   └── agg_intraday.csv                      # Hour aggregates
├── _reports/date=YYYY-MM-DD/
│   ├── l2_quality_daily.csv                  # Episode-level quality
│   ├── anomaly_report.json                   # Outliers & anomalies
│   └── quality_summary.json                  # Overall summary
├── _audit/date=YYYY-MM-DD/
│   ├── join_hod_coverage.csv                 # HOD baseline coverage
│   └── transform_log.jsonl                   # Processing steps
├── _metadata/
│   └── metadata.json                         # L2 contract
└── latest/                                   # Latest version symlinks
    ├── prepared_premium.parquet
    ├── hod_stats.parquet
    ├── l2_quality_daily.csv
    └── metadata.json
```

## Schema for prepared_premium.parquet

| Column | Type | Description |
|--------|------|-------------|
| episode_id | string | YYYY-MM-DD |
| t_in_episode | int16 | 0-59 |
| is_terminal | bool | True at t=59 |
| time_utc | datetime[UTC] | Unique timestamp |
| time_cot | datetime[COT] | Colombia time |
| hour_cot | int8 | 8-12 |
| minute_cot | int8 | 0,5,10...55 |
| open, high, low, close | float64 | COP per USD (6 decimals) |
| ohlc4 | float64 | (O+H+L+C)/4 |
| ret_log_1 | float64 | log(Ct/Ct-1) within episode |
| range_bps | float32 | (H-L)/C * 10000 |
| atr_14_norm_bps | float32 | ATR(14)/C * 10000 |
| ret_deseason | float64 | (ret_log_1 - median_h)/(1.4826*MAD_h) |
| is_valid_bar | bool | Passes grid+OHLC+not stale |
| is_stale | bool | O=H=L=C |
| is_missing | bool | Bar was absent in L1 |
| quality_flag_l2 | string | OK/WARN/FAIL |
| ingest_run_id | string | UUID |
| dataset_version | string | v2.0-l2 |

## Processing Steps

### 1. Load L1 Data
- Read standardized_data_accepted.parquet
- Read hod_baseline.csv
- Verify row counts match metadata

### 2. Create Base Features
- ohlc4 = (O+H+L+C)/4
- ret_log_1 = log(Ct/Ct-1) within episode
- range_bps = (H-L)/C * 10000
- ATR(14) normalized to bps

### 3. Deseasonalization
- Apply HOD median/MAD from hod_baseline.csv
- ret_deseason = (ret_log_1 - median_h)/(1.4826*MAD_h)
- Track coverage (should be >99%)

### 4. Create Masked vs Strict
- Masked: Always 60 slots, pad missing with is_missing=True
- Strict: Only complete 60/60 episodes

### 5. Apply Winsorization
- Clip returns to [p0.05%, p99.95%]
- Save parameters to winsor_params.json
- Recalculate deseasonalized returns

### 6. Quality Control
- L2 Quality Flags:
  - OK: 60/60, stale_rate ≤ 1%, outliers ≤ 0.5%
  - WARN: 59/60 OR stale_rate ≤ 2%
  - FAIL: Otherwise

### 7. Generate Reports
- l2_quality_daily.csv: Per-episode metrics
- quality_summary.json: Overall statistics
- anomaly_report.json: Outliers detected
- join_hod_coverage.csv: HOD application rate

### 8. Save Outputs
- Main datasets (parquet + CSV)
- Statistics and reports
- Metadata with SHA256 hashes
- READY flag

## Quality Contracts

Before READY:
- prepared_premium has 60 slots/episode (masked)
- prepared_premium_strict has only 60/60 episodes
- time_utc unique
- (episode_id, t_in_episode) unique
- 0 OHLC violations
- stale_rate ≤ 2% (WARN threshold)
- HOD applied to ≥99% of bars
- Parquet rows = CSV rows = metadata counts
- All SHA256 hashes in metadata

## Key Differences from L1
- L1: Standardizes and filters quality
- L2: Creates features and deseasonalizes
- L2 does NOT modify OHLC values
- L2 adds ML-ready features (returns, ranges, volatility)
- L2 provides both masked and strict versions

## Implementation Notes

1. **Deseasonalization is critical**: Use rolling 90-day window
2. **Preserve episode boundaries**: Never calculate returns across episodes
3. **Maintain audit trail**: Keep all transformations logged
4. **Two output modes**: Masked for completeness, Strict for quality
5. **CSV precision**: 6 decimals for prices, 4 for other floats

## Expected Results

From 55,705 L1 rows (929 episodes):
- Masked: 55,740 rows (929 episodes * 60 slots)
- Strict: ~53,400 rows (~890 complete episodes)
- Quality: ~85% OK, ~10% WARN, ~5% gaps/missing

## Success Criteria

✅ All L1 accepted data processed
✅ Features calculated correctly
✅ Deseasonalization applied >99%
✅ Both masked and strict versions created
✅ All reports generated
✅ Metadata complete with hashes
✅ READY flag created
✅ Latest symlinks updated