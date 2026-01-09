# Trading Calendar Usage Guide

## Overview

The `trading_calendar.py` module provides comprehensive Colombian holiday and weekend validation for the USD/COP trading system. It ensures that **NO weekends or Colombian holidays** are used in:

- Data loading to database
- Training data
- Validation data
- Test data
- Production inference

## Installation

```bash
pip install colombian-holidays==0.5.1
pip install holidays==0.37  # For US holidays (optional)
```

## Quick Start

```python
from services.common.trading_calendar import TradingCalendar

# Create calendar instance
cal = TradingCalendar()

# Check if date is a trading day
from datetime import date
is_trading = cal.is_trading_day(date(2025, 1, 1))  # False (New Year)
is_trading = cal.is_trading_day(date(2025, 1, 7))  # True (Tuesday)
```

## Use Cases

### 1. Airflow DAG Integration

#### Option A: Early Exit Pattern

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from services.common.trading_calendar import validate_dag_execution_date

def my_data_pipeline(**context):
    """Only run on trading days."""
    execution_date = context['execution_date']

    # Skip if not a trading day
    if not validate_dag_execution_date(execution_date):
        logging.info(f"Skipping: {execution_date.date()} is not a trading day")
        return

    # Rest of pipeline logic
    logging.info("Processing trading day data...")
    # ... load and process data

dag = DAG('my_dag', ...)
task = PythonOperator(
    task_id='load_ohlcv',
    python_callable=my_data_pipeline,
    provide_context=True
)
```

#### Option B: Branch Operator Pattern

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from services.common.trading_calendar import should_skip_dag_task

with DAG('my_dag', ...) as dag:
    # Check if trading day
    check_trading_day = BranchPythonOperator(
        task_id='check_trading_day',
        python_callable=should_skip_dag_task,
        provide_context=True
    )

    # Real tasks
    process_data = PythonOperator(
        task_id='continue_task',  # Must match return value
        python_callable=process_trading_data
    )

    # Skip placeholder
    skip_task = EmptyOperator(
        task_id='skip_task'  # Must match return value
    )

    check_trading_day >> [process_data, skip_task]
```

#### Example: Real Airflow DAG

```python
"""
DAG: l0_ohlcv_realtime
======================
Only loads data on trading days.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from services.common.trading_calendar import TradingCalendar
import logging

cal = TradingCalendar()

def load_ohlcv_data(**context):
    """Load OHLCV data only on trading days."""
    execution_date = context['execution_date']

    # Validate trading day
    if not cal.is_trading_day(execution_date):
        reason = cal.get_violation_reason(execution_date)
        logging.info(f"Skipping: {execution_date.date()} - {reason}")
        return

    # Load data from TwelveData API
    logging.info(f"Loading OHLCV data for {execution_date.date()}")
    # ... rest of implementation

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'l0_ohlcv_realtime',
    default_args=default_args,
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    catchup=False
) as dag:
    load_task = PythonOperator(
        task_id='load_ohlcv',
        python_callable=load_ohlcv_data,
        provide_context=True
    )
```

### 2. DataFrame Validation & Filtering

#### Filter Trading Days Only

```python
from services.common.trading_calendar import filter_trading_days
import pandas as pd

# Load raw data
df = pd.read_csv('ohlcv_data.csv', parse_dates=['timestamp'])
print(f"Raw data: {len(df)} rows")

# Filter to trading days only
df_clean = filter_trading_days(df, date_col='timestamp')
print(f"Clean data: {len(df_clean)} rows")
print(f"Removed {len(df) - len(df_clean)} non-trading days")
```

#### Validate No Holidays/Weekends

```python
from services.common.trading_calendar import validate_no_holidays

# Check if data is contaminated
is_valid, violations = validate_no_holidays(df, date_col='timestamp')

if not is_valid:
    print(f"WARNING: Found {len(violations)} non-trading days!")
    for v in violations[:5]:
        print(f"  - {v['date']}: {v['reason']}")
else:
    print("✓ All data is from valid trading days")
```

#### One-Step Validate & Filter

```python
from services.common.trading_calendar import validate_and_filter

# Validate and filter in one step
# Option 1: Just filter
df_clean = validate_and_filter(df, date_col='timestamp')

# Option 2: Raise exception if violations found
try:
    df_clean = validate_and_filter(df, date_col='timestamp', raise_on_violations=True)
except ValueError as e:
    print(f"Data validation failed: {e}")
```

### 3. ML Training Pipeline Integration

#### In `run.py` or Training Script

```python
from services.common.trading_calendar import validate_training_data
import pandas as pd

# Load data
df = pd.read_csv('l4_rlready_data.csv', parse_dates=['timestamp'])

# Split into train/test
train_mask = (df.index >= fold['train_start']) & (df.index <= fold['train_end'])
test_mask = (df.index >= fold['test_start']) & (df.index <= fold['test_end'])

df_train_raw = df[train_mask].copy()
df_test_raw = df[test_mask].copy()

# CRITICAL: Validate and clean training data
df_train = validate_training_data(df_train_raw, date_col='timestamp', data_type='training')
df_test = validate_training_data(df_test_raw, date_col='timestamp', data_type='test')

print(f"Train: {len(df_train)} rows (removed {len(df_train_raw) - len(df_train)})")
print(f"Test: {len(df_test)} rows (removed {len(df_test_raw) - len(df_test)})")

# Now safe to train model
model.learn(df_train, ...)
```

#### Complete Training Example

```python
"""
run.py - Walk-forward training with holiday validation
"""
from pathlib import Path
import pandas as pd
from services.common.trading_calendar import validate_training_data, TradingCalendar

# Initialize calendar
cal = TradingCalendar()

def run_fold(fold, df, features, log):
    """Run training for a single fold with holiday validation."""

    # Create data splits
    train_mask = (df.index >= fold['train_start']) & (df.index <= fold['train_end'])
    test_mask = (df.index >= fold['test_start']) & (df.index <= fold['test_end'])

    df_train_raw = df[train_mask].copy()
    df_test_raw = df[test_mask].copy()

    # VALIDATE: Ensure no holidays/weekends in training data
    df_train = validate_training_data(df_train_raw, data_type=f'train_fold_{fold["fold_id"]}')
    df_test = validate_training_data(df_test_raw, data_type=f'test_fold_{fold["fold_id"]}')

    # Check if we lost too much data
    train_loss_pct = (1 - len(df_train)/len(df_train_raw)) * 100
    test_loss_pct = (1 - len(df_test)/len(df_test_raw)) * 100

    log.log(f"Train data: {len(df_train)} rows (lost {train_loss_pct:.1f}%)")
    log.log(f"Test data: {len(df_test)} rows (lost {test_loss_pct:.1f}%)")

    # Verify date ranges are clean
    is_valid, violations = cal.validate_no_holidays(df_train)
    assert is_valid, f"Training data still contaminated after cleaning!"

    # Continue with training...
    return train_model(df_train, df_test, features)
```

### 4. Production Inference Validation

#### Real-time Inference Pipeline

```python
from services.common.trading_calendar import validate_inference_time
from datetime import datetime
import logging

def run_realtime_inference():
    """Run model inference only during valid trading hours."""

    # Check if we should run inference
    is_valid, message = validate_inference_time()

    if not is_valid:
        logging.info(f"Skipping inference: {message}")
        return None

    # Safe to run inference
    logging.info("Running real-time inference...")
    prediction = model.predict(current_observation)
    return prediction
```

#### Scheduled Inference Service

```python
"""
Realtime Inference Service
Runs every 5 minutes, but only executes on trading days during market hours.
"""
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from services.common.trading_calendar import validate_inference_time, TradingCalendar
from datetime import datetime
import logging

app = FastAPI()
cal = TradingCalendar()

def scheduled_inference():
    """Run inference every 5 minutes (but validate first)."""

    # Validate time
    is_valid, message = validate_inference_time()

    if not is_valid:
        logging.debug(f"Skipping: {message}")
        return

    # Run inference
    logging.info(f"Running inference at {datetime.now()}")
    try:
        result = run_model_inference()
        save_to_database(result)
    except Exception as e:
        logging.error(f"Inference failed: {e}")

# Setup scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    scheduled_inference,
    'cron',
    minute='*/5',  # Every 5 minutes
    id='inference_job'
)
scheduler.start()

@app.on_event("startup")
async def startup_event():
    logging.info("Realtime inference service started")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
```

### 5. Data Quality Reports

#### Generate Validation Report

```python
from services.common.trading_calendar import TradingCalendar
import pandas as pd

cal = TradingCalendar()

# Load data
df = pd.read_csv('ohlcv_data.csv', parse_dates=['timestamp'])

# Generate report
report = cal.generate_report(df, date_col='timestamp')
print(report)

# Save report to file
with open('data_validation_report.txt', 'w') as f:
    f.write(report)
```

Example output:
```
======================================================================
TRADING CALENDAR VALIDATION REPORT
======================================================================
DataFrame Shape: (150000, 20)
Date Column: timestamp
Unique Dates: 500
Date Range: 2023-01-01 to 2025-12-31

Status: FAIL
Violations: 25

First 10 violations:
  - 2023-01-01: Colombian Holiday
  - 2023-01-07: Weekend (Saturday)
  - 2023-01-08: Weekend (Sunday)
  - 2023-01-09: Colombian Holiday
  - 2023-01-14: Weekend (Saturday)
  - 2023-01-15: Weekend (Sunday)
  - 2023-01-21: Weekend (Saturday)
  - 2023-01-22: Weekend (Sunday)
  - 2023-01-28: Weekend (Saturday)
  - 2023-01-29: Weekend (Sunday)
  ... and 15 more
======================================================================
```

### 6. Advanced Usage

#### Get Holiday Lists

```python
from services.common.trading_calendar import TradingCalendar
from datetime import date

cal = TradingCalendar()

# Get all holidays in a range
holidays = cal.get_holidays_in_range(
    start=date(2025, 1, 1),
    end=date(2025, 12, 31),
    include_weekends=False  # Only holidays, not weekends
)

print("Colombian holidays in 2025:")
for dt, reason in holidays:
    print(f"  {dt}: {reason}")
```

#### Count Trading Days

```python
# Get all valid trading days
trading_days = cal.get_trading_days_in_range(
    start=date(2025, 1, 1),
    end=date(2025, 12, 31)
)

print(f"Trading days in 2025: {len(trading_days)}")
print(f"First trading day: {trading_days[0]}")
print(f"Last trading day: {trading_days[-1]}")
```

#### Find Next/Previous Trading Day

```python
# Find next trading day after New Year
next_day = cal.get_next_trading_day(date(2025, 1, 1))
print(f"Next trading day after New Year: {next_day}")

# Find previous trading day before a weekend
prev_day = cal.get_previous_trading_day(date(2025, 1, 11))  # Saturday
print(f"Previous trading day: {prev_day}")  # Friday, Jan 10
```

#### Include US Holidays (for Macro Data)

```python
# Create calendar that also checks US holidays
cal_us = TradingCalendar(include_us_holidays=True)

# This will return False if it's a US holiday
is_trading = cal_us.is_trading_day(date(2025, 7, 4))  # July 4th
print(f"July 4th is trading day: {is_trading}")  # False
```

## API Reference

### TradingCalendar Class

```python
class TradingCalendar:
    def __init__(
        self,
        timezone: str = 'America/Bogota',
        include_us_holidays: bool = False,
        market_open_hour: int = 8,
        market_close_hour: int = 12
    )
```

**Methods:**

- `is_trading_day(dt)` - Check if date is a valid trading day
- `is_weekend(dt)` - Check if date is Saturday/Sunday
- `is_colombian_holiday(dt)` - Check if date is Colombian holiday
- `get_violation_reason(dt)` - Get reason why date is not a trading day
- `filter_trading_days(df, date_col)` - Filter DataFrame to trading days
- `validate_no_holidays(df, date_col)` - Validate DataFrame has no holidays
- `get_holidays_in_range(start, end)` - Get holidays in date range
- `get_trading_days_in_range(start, end)` - Get trading days in range
- `get_next_trading_day(dt)` - Find next trading day
- `get_previous_trading_day(dt)` - Find previous trading day
- `generate_report(df, date_col)` - Generate validation report

### Convenience Functions

```python
# Quick access functions (use singleton instance)
is_trading_day(dt) -> bool
filter_trading_days(df, date_col='timestamp') -> pd.DataFrame
validate_no_holidays(df, date_col='timestamp') -> Tuple[bool, List[Dict]]
validate_and_filter(df, date_col='timestamp', raise_on_violations=False) -> pd.DataFrame

# Airflow integration
validate_dag_execution_date(execution_date) -> bool
should_skip_dag_task(**context) -> str  # Returns 'skip_task' or 'continue_task'

# Training integration
validate_training_data(df, date_col='timestamp', data_type='training') -> pd.DataFrame

# Production inference
validate_inference_time(dt=None) -> Tuple[bool, str]
```

## Best Practices

### 1. Always Validate at Load Time

```python
# BAD: Load data without validation
df = pd.read_csv('data.csv')
df.to_sql('ohlcv', engine)

# GOOD: Validate before loading to database
df = pd.read_csv('data.csv')
df_clean = filter_trading_days(df, date_col='timestamp')
df_clean.to_sql('ohlcv', engine)
```

### 2. Validate Before Training

```python
# BAD: Train on potentially contaminated data
model.learn(df_train)

# GOOD: Validate and clean first
df_train = validate_training_data(df_train, data_type='training')
model.learn(df_train)
```

### 3. Check Inference Time

```python
# BAD: Run inference blindly
prediction = model.predict(observation)

# GOOD: Validate time first
is_valid, msg = validate_inference_time()
if is_valid:
    prediction = model.predict(observation)
else:
    logging.info(f"Skipping: {msg}")
```

### 4. Use in Airflow DAGs

```python
# BAD: Run DAG on all days
def my_task(**context):
    load_and_process_data()

# GOOD: Skip non-trading days
def my_task(**context):
    if not validate_dag_execution_date(context['execution_date']):
        logging.info("Skipping: not a trading day")
        return
    load_and_process_data()
```

## Testing

Run the built-in test suite:

```bash
cd services/common
python -m trading_calendar
```

Expected output:
```
======================================================================
USD/COP TRADING CALENDAR - VALIDATION TEST
======================================================================

Library Status:
  colombian-holidays: ✓ Installed
  holidays (US):      ✓ Installed

Testing specific dates:
Date            Expected                       Is Trading?     Reason
--------------------------------------------------------------------------------
2025-01-01      New Year's Day                 ✗ No            Colombian Holiday
2025-01-06      Epiphany (Reyes Magos)         ✗ No            Colombian Holiday
2025-01-07      Tuesday - should be trading    ✓ Yes           Valid trading day
2025-01-11      Saturday - weekend             ✗ No            Weekend (Saturday)
...

Colombian holidays in January 2025:
  - 2025-01-01: Colombian Holiday
  - 2025-01-06: Colombian Holiday

Trading days in January 2025: 21 out of 31 days

======================================================================
TEST COMPLETE
======================================================================
```

## Troubleshooting

### Library Not Installed

If you see warnings about missing libraries:

```python
WARNING: colombian-holidays library not installed. Holiday validation will be incomplete!
```

**Solution:**
```bash
pip install colombian-holidays==0.5.1
```

### Timezone Issues

If you're getting unexpected results, ensure datetime objects are timezone-aware:

```python
from datetime import datetime
import pytz

COT_TZ = pytz.timezone('America/Bogota')

# BAD: Naive datetime
dt = datetime(2025, 1, 1, 8, 0, 0)

# GOOD: Timezone-aware
dt = COT_TZ.localize(datetime(2025, 1, 1, 8, 0, 0))
```

### Performance

The library caches Colombian holidays by year for performance. If processing millions of rows:

```python
# Pre-warm cache for years you'll need
cal = TradingCalendar()
for year in range(2020, 2026):
    cal._get_col_holidays_for_year(year)

# Now filtering is fast
df_clean = cal.filter_trading_days(df)
```

## Migration Guide

If you have existing code using the old `is_trading_day()` from `l0_ohlcv_backfill.py`:

### Before:
```python
def is_trading_day(d: date) -> bool:
    return d.weekday() in [0,1,2,3,4] and d not in HOLIDAYS
```

### After:
```python
from services.common.trading_calendar import is_trading_day

# Same interface!
if is_trading_day(my_date):
    process_data()
```

The new version is more accurate because it uses the `colombian-holidays` library which handles:
- Ley Emiliani (movable holidays)
- Easter-dependent holidays
- Future year holidays automatically

## Summary

The Trading Calendar module provides **defense in depth** against holiday/weekend contamination:

1. **Database Load**: Filter in Airflow DAGs
2. **Training**: Validate in `run.py` before training
3. **Inference**: Check time before prediction
4. **Analytics**: Validate in API endpoints

This ensures your USD/COP trading system NEVER uses non-trading day data, preventing subtle bugs and improving model reliability.
