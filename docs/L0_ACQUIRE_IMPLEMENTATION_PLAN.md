# 🎯 L0 ACQUIRE - REAL DATA IMPLEMENTATION PLAN
================================================================================

## ⚠️ CURRENT ISSUES TO FIX
1. **NO SIMULATED DATA** - Must connect to REAL MT5 and TwelveData APIs
2. **Missing Grid Alignment** - Must ensure perfect M5 grid (:00, :05, :10...)
3. **Missing Validation Contracts** - Need grid_ok, ohlc_ok, missing ≤ 0.5%
4. **Incomplete Metadata** - Missing price_precision, quote_convention, pip_size

## 📋 IMPLEMENTATION PLAN

### PHASE 1: REAL DATA CONNECTIONS

#### 1.1 MT5 Connection (MetaTrader5 Python)
```python
# Requirements:
pip install MetaTrader5 pandas pytz

# Connection setup:
import MetaTrader5 as mt5

mt5.initialize()
mt5.login(login=ACCOUNT_ID, password=PASSWORD, server=SERVER)

# Download real USDCOP data:
rates = mt5.copy_rates_range(
    "USDCOP",
    mt5.TIMEFRAME_M5,
    datetime(2020, 1, 1, tzinfo=pytz.UTC),
    datetime.now(pytz.UTC)
)
```

#### 1.2 TwelveData API Connection
```python
# Requirements:
pip install twelvedata pandas

# Connection:
from twelvedata import TDClient

td = TDClient(apikey=TWELVEDATA_API_KEY)

# Download real data:
df = td.time_series(
    symbol="USD/COP",
    interval="5min",
    start_date="2020-01-01",
    end_date="2025-08-20",
    timezone="America/Bogota"
)
```

### PHASE 2: DATA PROCESSING PIPELINE

#### 2.1 Grid Alignment Function
```python
def align_to_m5_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure perfect M5 grid alignment"""
    # Round to nearest 5-minute mark
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.floor('5min')
    
    # Remove any off-grid records
    df = df[df['time'].dt.minute % 5 == 0]
    
    # Remove duplicates keeping first
    df = df.drop_duplicates(subset=['time'], keep='first')
    
    return df.sort_values('time')
```

#### 2.2 OHLC Validation
```python
def validate_ohlc(df: pd.DataFrame) -> dict:
    """Validate OHLC consistency"""
    violations = df[
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ]
    
    return {
        'ohlc_ok': len(violations) == 0,
        'violations_count': len(violations),
        'violation_pct': len(violations) / len(df) * 100
    }
```

#### 2.3 Coverage Analysis
```python
def analyze_coverage(df: pd.DataFrame) -> dict:
    """Analyze data coverage and gaps"""
    # Expected 288 bars per trading day (24h * 12 bars/hour)
    # Premium window: 72 bars (6h * 12 bars/hour)
    
    df = df.sort_values('time')
    df['gap'] = df['time'].diff()
    
    # Identify gaps > 5 minutes
    gaps = df[df['gap'] > pd.Timedelta(minutes=5)]
    
    # Calculate coverage
    date_range = (df['time'].max() - df['time'].min()).days
    expected_bars = date_range * 288 * 0.3  # ~30% trading hours
    actual_bars = len(df)
    
    return {
        'grid_ok': True,  # Set after validation
        'missing_pct': (1 - actual_bars/expected_bars) * 100,
        'duplicates_pct': df.duplicated(subset=['time']).sum() / len(df) * 100,
        'max_gap_minutes': df['gap'].max().total_seconds() / 60,
        'gaps_count': len(gaps),
        'coverage_pct': (actual_bars / expected_bars) * 100
    }
```

### PHASE 3: CONTRACT IMPLEMENTATION

#### 3.1 Validation Report Structure
```json
{
    "validation_report": {
        "grid_ok": true,              // Perfect M5 alignment
        "ohlc_ok": true,              // H≥max(O,C)≥min(O,C)≥L
        "missing_pct": 0.3,           // Must be ≤ 0.5%
        "duplicates_pct": 0.05,       // Must be ≤ 0.1%
        "max_gap_minutes": 15,
        "coverage_stats": {
            "expected_bars": 525600,   // For full period
            "actual_bars": 524024,
            "coverage_pct": 99.7
        },
        "validation_status": "PASSED"  // or "FAILED"
    }
}
```

#### 3.2 Metadata Structure
```json
{
    "metadata": {
        "symbol": "USDCOP",
        "timeframe": "M5",
        "timezone": "UTC",
        "price_precision": 2,
        "quote_convention": "COP per USD",
        "pip_size": 0.01,
        "price_unit": "COP",
        "source": "mt5",
        "broker": "MetaQuotes",
        "account_type": "real",
        "commit_hash": "abc123",
        "ingest_duration_s": 45.3,
        "row_count": 524024,
        "md5_parquet": "d41d8cd98f00b204e9800998ecf8427e",
        "date_range": {
            "start": "2020-01-01T00:00:00Z",
            "end": "2025-08-20T23:55:00Z"
        }
    }
}
```

### PHASE 4: IDEMPOTENT WRITE PATTERN

```python
def write_idempotent(df: pd.DataFrame, s3_hook, bucket: str, final_key: str):
    """Write with tmp/ pattern for idempotency"""
    # 1. Write to tmp/
    tmp_key = final_key.replace('/data/', '/tmp/')
    
    # 2. Validate data
    validation = validate_all_contracts(df)
    
    if validation['status'] == 'PASSED':
        # 3. Write to tmp first
        s3_hook.load_bytes(
            bytes_data=df.to_parquet(),
            key=tmp_key,
            bucket_name=bucket
        )
        
        # 4. Move to final location
        s3_hook.copy_object(
            source_bucket_key=tmp_key,
            dest_bucket_key=final_key,
            source_bucket_name=bucket,
            dest_bucket_name=bucket
        )
        
        # 5. Delete tmp
        s3_hook.delete_objects(bucket=bucket, keys=[tmp_key])
        
        # 6. Write READY signal
        ready_key = final_key.replace('raw_data.parquet', 'READY')
        s3_hook.load_string(
            string_data=json.dumps(validation),
            key=ready_key,
            bucket_name=bucket
        )
    else:
        raise ValueError(f"Validation failed: {validation['errors']}")
```

### PHASE 5: PREMIUM WINDOW VERIFICATION

```python
def verify_premium_coverage(df: pd.DataFrame) -> dict:
    """Verify Premium window (08:00-14:00 COT) coverage"""
    # Convert to COT (UTC-5)
    df['time_cot'] = df['time'] - pd.Timedelta(hours=5)
    df['hour'] = df['time_cot'].dt.hour
    df['dayofweek'] = df['time_cot'].dt.dayofweek
    
    # Premium window: Mon-Fri 08:00-14:00 COT
    premium_mask = (
        (df['dayofweek'] < 5) &  # Monday=0 to Friday=4
        (df['hour'] >= 8) &
        (df['hour'] < 14)
    )
    
    premium_data = df[premium_mask]
    
    # Expected: 72 bars per day (6 hours * 12 bars)
    trading_days = premium_data['time_cot'].dt.date.nunique()
    expected_premium_bars = trading_days * 72
    actual_premium_bars = len(premium_data)
    
    return {
        'premium_coverage_pct': (actual_premium_bars / expected_premium_bars * 100),
        'premium_bars': actual_premium_bars,
        'expected_bars': expected_premium_bars,
        'trading_days': trading_days,
        'premium_quality': 'GOOD' if (actual_premium_bars / expected_premium_bars) > 0.9 else 'NEEDS_REVIEW'
    }
```

## 📊 EXPECTED OUTPUT STRUCTURE

```
ds-usdcop-acquire/
└── usdcop_m5__01_l0_acquire_sync_incremental/
    ├── market=usdcop/
    │   └── timeframe=m5/
    │       └── date=2025-08-20/
    │           └── run_id=20250820-abc123/
    │               ├── raw_data.parquet         # Real MT5/TwelveData
    │               ├── metadata.json            # Complete metadata
    │               ├── validation_report.json   # All contracts
    │               ├── quality_metrics.json     # Coverage analysis
    │               ├── sync_status.json         # Sync tracking
    │               └── READY                    # Signal file
    └── _control/
        └── VALIDATED                        # After all checks pass
```

## 🔑 KEY IMPLEMENTATION POINTS

1. **NO SIMULATED DATA** - Only real API calls to MT5/TwelveData
2. **Grid Alignment** - Perfect M5 grid (:00, :05, :10, :15...)
3. **Validation Contracts** - grid_ok, ohlc_ok, missing ≤ 0.5%
4. **Quote Convention** - Clear "COP per USD" in metadata
5. **Idempotent Writes** - tmp/ pattern with validation
6. **Premium Window** - 08:00-14:00 COT verification
7. **Metadata Complete** - All required fields including MD5
8. **READY Signal** - Only after validation passes

## ⚡ NEXT STEPS

1. Install required packages (MetaTrader5, twelvedata)
2. Configure API credentials in Airflow Variables
3. Implement connection functions
4. Add validation contracts
5. Deploy and test with real data
6. Monitor quality metrics

## 🎯 SUCCESS CRITERIA

- ✅ Real data from MT5 and TwelveData (no simulation)
- ✅ Perfect M5 grid alignment
- ✅ Missing data ≤ 0.5%
- ✅ Duplicates ≤ 0.1%  
- ✅ OHLC consistency 100%
- ✅ Premium window coverage > 90%
- ✅ Complete metadata with MD5 hash
- ✅ Idempotent writes with validation