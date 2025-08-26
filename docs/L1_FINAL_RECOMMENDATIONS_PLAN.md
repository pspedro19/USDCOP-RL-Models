# L1 Pipeline - Final Recommendations Implementation Plan

## Executive Summary
Final sprint to make L1 production-ready by addressing remaining audit findings:
- Remove OHLC violations from accepted data
- Clarify file naming conventions
- Fix metrics calculations
- Standardize output formats
- Improve traceability

## Priority Tasks

### 1. 🚨 CRITICAL: Enforce No OHLC Violations in Accepted Data
**Issue:** 8 rows with `ohlc_valid=False` in accepted subset
**Solution Options:**

#### Option A: Drop Invalid Bars (Recommended)
```python
# In calculate_quality_metrics function
if ohlc_violations > 0:
    # Mark day as WARN if it would be OK
    if quality_flag == 'OK':
        quality_flag = 'WARN'
        fail_reason = 'OHLC_VIOLATIONS_REMOVED'
    
# In save_all_outputs function
# Filter out OHLC violations from clean subset
df_clean = df_clean[df_clean['ohlc_valid'] != False]
```

#### Option B: Replace with Placeholders
```python
# Replace invalid OHLC with NaN and mark as missing
invalid_mask = df_clean['ohlc_valid'] == False
df_clean.loc[invalid_mask, ['open', 'high', 'low', 'close']] = np.nan
df_clean.loc[invalid_mask, 'is_missing'] = True
```

### 2. 📝 Rename Output Files for Clarity
**Current → New:**
- `standardized_data.csv` → `standardized_data_all.csv`
- `standardized_data_OK_WARNS.csv` → `standardized_data_accepted.csv`

```python
# Update file names in save_all_outputs
files = {
    'all_data': {
        'parquet': 'standardized_data_all.parquet',
        'csv': 'standardized_data_all.csv'
    },
    'accepted': {
        'parquet': 'standardized_data_accepted.parquet', 
        'csv': 'standardized_data_accepted.csv'
    }
}
```

### 3. 🔢 Fix Duplicates Count Calculation
**Issue:** Shows constant ~4 when data has no duplicates
**Fix:** Calculate per-day duplicates during initial load

```python
def load_and_clean_data(**context):
    # Track duplicates per episode before dropping
    duplicates_per_day = {}
    for episode_id in df['episode_id'].unique():
        day_df = df[df['episode_id'] == episode_id]
        day_dups = day_df.duplicated(subset=['time_utc']).sum()
        duplicates_per_day[episode_id] = day_dups
    
    # Store for quality calculation
    context['task_instance'].xcom_push(
        key='duplicates_per_day', 
        value=duplicates_per_day
    )
```

### 4. 📊 Standardize CSV Format
**Requirements:**
- 6 decimal precision for prices
- ISO-8601 timestamps
- Consistent encoding

```python
def save_csv_standardized(df, path, s3_hook, bucket, key):
    """Save CSV with standard format"""
    df_csv = df.copy()
    
    # Format prices to 6 decimals
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].map(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    
    # Format timestamps as ISO-8601
    if 'time_utc' in df_csv.columns:
        df_csv['time_utc'] = df_csv['time_utc'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    if 'time_cot' in df_csv.columns:
        df_csv['time_cot'] = df_csv['time_cot'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Save with consistent settings
    buffer = io.BytesIO()
    df_csv.to_csv(
        buffer, 
        index=False,
        float_format='%.6f',
        date_format='%Y-%m-%dT%H:%M:%S'
    )
    
    # Upload to S3
    buffer.seek(0)
    s3_hook.load_bytes(buffer.getvalue(), key=key, bucket_name=bucket, replace=True)
    
    return hashlib.sha256(buffer.getvalue()).hexdigest()
```

### 5. 🏷️ Add Traceability Columns
**New columns:**
- `is_premium`: Boolean flag (always True for L1)
- `dataset_version`: String from metadata

```python
# Add before saving
df['is_premium'] = True
df['dataset_version'] = DATASET_VERSION
```

### 6. 🔧 Fix XCom Serialization
**Issue:** pandas.Timestamp not JSON serializable
**Fix:** Convert all timestamps to strings

```python
def safe_xcom_push(context, key, value):
    """Push to XCom with safe serialization"""
    if isinstance(value, dict):
        # Convert any Timestamp objects to strings
        safe_value = {}
        for k, v in value.items():
            if isinstance(v, pd.Timestamp):
                safe_value[k] = v.isoformat()
            elif isinstance(v, list):
                safe_value[k] = [
                    x.isoformat() if isinstance(x, pd.Timestamp) else x 
                    for x in v
                ]
            else:
                safe_value[k] = v
        value = safe_value
    
    # Ensure JSON serializable
    json_str = json.dumps(value, default=str)
    context['task_instance'].xcom_push(key=key, value=json.loads(json_str))
```

### 7. 📈 Add Failure Reason Categories
**New column in quality report:** `failure_category`

```python
def categorize_failure(row):
    """Categorize failure reasons"""
    if row['quality_flag'] == 'OK':
        return 'PASS'
    elif row['quality_flag'] == 'WARN':
        if 'SINGLE_MISSING' in row['fail_reason']:
            return 'MISSING_BAR'
        elif 'STALE' in row['fail_reason']:
            return 'STALE_RATE'
        elif 'OHLC' in row['fail_reason']:
            return 'OHLC_VIOLATION'
        else:
            return 'OTHER_WARN'
    else:  # FAIL
        if row['rows_found'] == 0:
            return 'NO_DATA'
        elif row['rows_found'] < 30:
            return 'INSUFFICIENT_SEVERE'
        elif row['rows_found'] < 59:
            return 'INSUFFICIENT_BARS'
        elif row['stale_rate'] > 2:
            return 'HIGH_STALE'
        elif row['ohlc_violations'] > 0:
            return 'OHLC_VIOLATIONS'
        else:
            return 'OTHER_FAIL'

quality_df['failure_category'] = quality_df.apply(categorize_failure, axis=1)
```

## Implementation Timeline

| Phase | Task | Priority | Est. Time |
|-------|------|----------|-----------|
| 1 | Fix OHLC violations in accepted data | 🔴 Critical | 1 hour |
| 2 | Fix XCom serialization | 🔴 Critical | 30 min |
| 3 | Fix duplicates_count calculation | 🟡 High | 45 min |
| 4 | Standardize CSV format | 🟡 High | 30 min |
| 5 | Rename output files | 🟢 Medium | 15 min |
| 6 | Add traceability columns | 🟢 Medium | 15 min |
| 7 | Add failure categories | 🟢 Medium | 30 min |
| 8 | Test all changes | 🔴 Critical | 1 hour |
| 9 | Deploy to production | 🔴 Critical | 30 min |

**Total estimated time:** 4.5 hours

## Validation Checklist

### Pre-deployment
- [ ] No OHLC violations in accepted subset
- [ ] Duplicates count accurate per day
- [ ] CSV format standardized (6 decimals, ISO-8601)
- [ ] XCom serialization working
- [ ] File names clarified
- [ ] Traceability columns added
- [ ] Failure categories populated

### Post-deployment
- [ ] Accepted subset has 0 OHLC violations
- [ ] Quality report shows accurate duplicates
- [ ] CSV files parse correctly in pandas
- [ ] Airflow DAG runs without XCom errors
- [ ] L2 can consume standardized_data_accepted.csv
- [ ] Monitoring dashboard shows failure categories

## Expected Results After Implementation

### Data Quality
- **Accepted data:** 57,540 rows, 0 OHLC violations
- **Full data:** 71,717 rows (all episodes)
- **Quality breakdown:** ~65% OK, ~13% WARN, ~22% FAIL

### File Outputs (8 total)
1. `standardized_data_all.parquet` - Complete dataset
2. `standardized_data_all.csv` - Complete dataset (formatted)
3. `standardized_data_accepted.parquet` - OK/WARN only, no violations
4. `standardized_data_accepted.csv` - OK/WARN only (formatted)
5. `_reports/daily_quality_60.csv` - Enhanced quality report
6. `_metadata.json` - Complete metadata with hashes
7. `_reports/failure_summary.csv` - Aggregated failure categories
8. `_reports/ohlc_violations.csv` - List of violation details

## Success Criteria
✅ Accepted subset has exactly 0 OHLC violations
✅ File names clearly indicate content (all vs accepted)
✅ Quality metrics reconcile perfectly with data
✅ CSV format is standardized and parseable
✅ XCom errors eliminated
✅ Complete audit trail with failure categories

## Next Steps
1. Implement changes in priority order
2. Run comprehensive tests locally
3. Deploy to staging environment
4. Validate with sample L2 consumer
5. Deploy to production
6. Monitor for 24 hours
7. Sign off on L1 completion

## Notes
- Keep Mode A (no padding) as default
- Consider Mode B (padding) as future enhancement
- Document all changes in CHANGELOG.md
- Update L2 documentation with new file names