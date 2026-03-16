# Trading Calendar Integration Examples

## Example 1: Airflow DAG - OHLCV Data Loading

```python
"""
DAG: l0_ohlcv_realtime_validated
=================================
OHLCV data loading with Colombian holiday validation.

BEFORE: Loaded data on all days including weekends/holidays
AFTER: Only loads on valid trading days
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from services.common.trading_calendar import validate_dag_execution_date, TradingCalendar
import psycopg2
import requests
import logging

# Initialize calendar
cal = TradingCalendar()

def load_ohlcv_data(**context):
    """
    Load OHLCV data from TwelveData API.
    CRITICAL: Only runs on trading days!
    """
    execution_date = context['execution_date']

    # VALIDATION: Skip if not a trading day
    if not validate_dag_execution_date(execution_date):
        reason = cal.get_violation_reason(execution_date)
        logging.info(f"⏭️  Skipping: {execution_date.date()} - {reason}")
        return

    logging.info(f"✓ Processing trading day: {execution_date.date()}")

    # Fetch data from TwelveData
    response = requests.get(
        'https://api.twelvedata.com/time_series',
        params={
            'symbol': 'USD/COP',
            'interval': '5min',
            'apikey': os.environ['TWELVEDATA_API_KEY']
        }
    )

    if response.status_code == 200:
        data = response.json()
        # ... process and insert to database
        logging.info(f"✓ Loaded {len(data['values'])} bars")
    else:
        logging.error(f"API error: {response.status_code}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'l0_ohlcv_realtime_validated',
    default_args=default_args,
    schedule_interval='*/5 8-12 * * 1-5',  # Every 5 min, Mon-Fri, 8am-12pm
    catchup=False,
    tags=['layer-0', 'ohlcv', 'realtime', 'validated']
) as dag:

    load_task = PythonOperator(
        task_id='load_ohlcv',
        python_callable=load_ohlcv_data,
        provide_context=True
    )
```

## Example 2: Airflow DAG with BranchOperator

```python
"""
DAG: l1_feature_engineering_validated
======================================
Feature engineering with branch logic for trading day validation.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from services.common.trading_calendar import should_skip_dag_task

def process_features(**context):
    """Process features for valid trading days."""
    logging.info("Processing features...")
    # ... feature engineering logic

def send_completion_alert(**context):
    """Send alert when processing complete."""
    logging.info("Features processed successfully")

with DAG(
    'l1_feature_engineering_validated',
    default_args=default_args,
    schedule_interval='0 13 * * 1-5',  # Daily at 1pm
    catchup=False
) as dag:

    # Check if today is a trading day
    check_trading_day = BranchPythonOperator(
        task_id='check_trading_day',
        python_callable=should_skip_dag_task,
        provide_context=True
    )

    # Continue with processing
    process_task = PythonOperator(
        task_id='continue_task',  # Must match return value
        python_callable=process_features,
        provide_context=True
    )

    alert_task = PythonOperator(
        task_id='send_alert',
        python_callable=send_completion_alert,
        provide_context=True
    )

    # Skip placeholder
    skip_task = EmptyOperator(
        task_id='skip_task'  # Must match return value
    )

    # DAG flow
    check_trading_day >> [process_task, skip_task]
    process_task >> alert_task
```

## Example 3: Training Pipeline (run.py)

```python
"""
run.py - Walk-forward training with holiday validation
=======================================================
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add services to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'services'))

from services.common.trading_calendar import (
    validate_training_data,
    TradingCalendar,
    validate_no_holidays
)

# Initialize
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
cal = TradingCalendar()

def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """
    Load and validate training data.
    CRITICAL: Removes all holidays/weekends.
    """
    logger.info(f"Loading data from {data_path}")

    # Load raw data
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    logger.info(f"Raw data: {len(df):,} rows")

    # VALIDATE: Check for contamination
    is_valid, violations = validate_no_holidays(df, date_col='timestamp')

    if not is_valid:
        logger.warning(
            f"⚠️  DATA CONTAMINATION DETECTED: "
            f"Found {len(violations)} non-trading days"
        )

        # Show first few violations
        for v in violations[:5]:
            logger.warning(f"  - {v['date']}: {v['reason']}")

        if len(violations) > 5:
            logger.warning(f"  ... and {len(violations) - 5} more")

    # CLEAN: Remove holidays/weekends
    df_clean = cal.filter_trading_days(df, date_col='timestamp')

    n_removed = len(df) - len(df_clean)
    logger.info(
        f"✓ Cleaned data: {len(df_clean):,} rows "
        f"(removed {n_removed:,} non-trading days, {n_removed/len(df)*100:.2f}%)"
    )

    return df_clean


def run_fold(fold_id, df, features):
    """
    Run training for a single fold with validation.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold_id}: {fold['train_start']} to {fold['train_end']}")
    logger.info(f"{'='*70}")

    # Create splits
    train_mask = (df.index >= fold['train_start']) & (df.index <= fold['train_end'])
    test_mask = (df.index >= fold['test_start']) & (df.index <= fold['test_end'])

    df_train_raw = df[train_mask].copy()
    df_test_raw = df[test_mask].copy()

    # CRITICAL: Validate training data
    logger.info("Validating training data...")
    df_train = validate_training_data(
        df_train_raw,
        date_col='timestamp',
        data_type=f'train_fold_{fold_id}'
    )

    logger.info("Validating test data...")
    df_test = validate_training_data(
        df_test_raw,
        date_col='timestamp',
        data_type=f'test_fold_{fold_id}'
    )

    # Verify no contamination
    is_valid_train, _ = validate_no_holidays(df_train, date_col='timestamp')
    is_valid_test, _ = validate_no_holidays(df_test, date_col='timestamp')

    assert is_valid_train, "Training data still contaminated after cleaning!"
    assert is_valid_test, "Test data still contaminated after cleaning!"

    logger.info(f"✓ Train: {len(df_train):,} rows (all trading days)")
    logger.info(f"✓ Test: {len(df_test):,} rows (all trading days)")

    # Continue with training
    model = train_model(df_train, df_test, features)
    return model


def main():
    """Main execution."""
    logger.info("="*70)
    logger.info("USD/COP RL TRAINING - WITH HOLIDAY VALIDATION")
    logger.info("="*70)

    # Load and validate data
    data_path = Path('data/l4_rlready_data.csv')
    df = load_and_validate_data(data_path)

    # Show data quality report
    logger.info("\n" + "="*70)
    logger.info("DATA QUALITY REPORT")
    logger.info("="*70)

    report = cal.generate_report(df, date_col='timestamp')
    print(report)

    # Run walk-forward training
    for fold in WALK_FORWARD_FOLDS:
        result = run_fold(fold['fold_id'], df, FEATURES)
        # ... process result

    logger.info("\n✓ Training complete - all data validated")


if __name__ == '__main__':
    main()
```

## Example 4: Real-time Inference Service

```python
"""
realtime_inference_service.py
==============================
Real-time inference with trading day/hour validation.
"""

from fastapi import FastAPI, HTTPException
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from services.common.trading_calendar import validate_inference_time, TradingCalendar
import logging

app = FastAPI()
cal = TradingCalendar()
scheduler = BackgroundScheduler()

# Global state
last_inference_result = None

def run_inference():
    """
    Run model inference.
    CRITICAL: Only runs during trading hours on trading days!
    """
    global last_inference_result

    # VALIDATE: Check if we should run
    is_valid, message = validate_inference_time()

    if not is_valid:
        logging.debug(f"⏭️  Skipping inference: {message}")
        return

    # Run inference
    logging.info(f"✓ Running inference at {datetime.now()}")
    try:
        # Get current market data
        observation = fetch_current_observation()

        # Run model
        prediction = model.predict(observation)

        # Save result
        last_inference_result = {
            'timestamp': datetime.now(),
            'action': prediction['action'],
            'confidence': prediction['confidence']
        }

        # Store in database
        save_to_database(last_inference_result)

        logging.info(f"✓ Inference complete: {prediction['action']}")

    except Exception as e:
        logging.error(f"❌ Inference failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Start scheduled inference."""
    logging.info("Starting inference scheduler...")

    # Schedule inference every 5 minutes
    scheduler.add_job(
        run_inference,
        'cron',
        minute='*/5',
        id='inference_job'
    )
    scheduler.start()

    logging.info("✓ Inference service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop scheduler."""
    scheduler.shutdown()
    logging.info("✓ Inference service stopped")


@app.get("/api/v1/inference/latest")
async def get_latest_inference():
    """Get latest inference result."""
    if last_inference_result is None:
        raise HTTPException(status_code=404, detail="No inference results available")

    # Check if result is still valid (not expired)
    age_minutes = (datetime.now() - last_inference_result['timestamp']).total_seconds() / 60

    if age_minutes > 10:
        raise HTTPException(
            status_code=410,
            detail=f"Latest result is {age_minutes:.1f} minutes old"
        )

    return last_inference_result


@app.get("/api/v1/inference/status")
async def get_inference_status():
    """Check if inference should be running now."""
    is_valid, message = validate_inference_time()

    return {
        'should_run': is_valid,
        'message': message,
        'timestamp': datetime.now(),
        'last_result': last_inference_result
    }
```

## Example 5: Data Quality Monitoring Script

```python
"""
validate_database_data.py
=========================
Script to validate all data in database for holiday contamination.
"""

import psycopg2
import pandas as pd
from services.common.trading_calendar import TradingCalendar, validate_no_holidays
from services.common.database import get_db_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cal = TradingCalendar()

def validate_table(table_name: str, date_col: str = 'timestamp'):
    """Validate a database table for holiday contamination."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Validating table: {table_name}")
    logger.info(f"{'='*70}")

    # Load data
    conn = get_db_connection()
    query = f"SELECT {date_col} FROM {table_name} ORDER BY {date_col}"
    df = pd.read_sql(query, conn, parse_dates=[date_col])
    conn.close()

    logger.info(f"Total rows: {len(df):,}")

    # Validate
    is_valid, violations = validate_no_holidays(df, date_col=date_col)

    if is_valid:
        logger.info(f"✓ PASS: No holidays/weekends found")
    else:
        logger.warning(f"❌ FAIL: Found {len(violations)} non-trading days")

        # Group by reason
        from collections import Counter
        reason_counts = Counter(v['reason'] for v in violations)

        logger.warning("\nViolations by type:")
        for reason, count in reason_counts.items():
            logger.warning(f"  - {reason}: {count}")

        # Show first 10
        logger.warning("\nFirst 10 violations:")
        for v in violations[:10]:
            logger.warning(f"  - {v['date']}: {v['reason']}")

    return is_valid, violations


def main():
    """Validate all critical tables."""
    logger.info("="*70)
    logger.info("DATABASE HOLIDAY VALIDATION REPORT")
    logger.info("="*70)

    tables_to_validate = [
        ('usdcop_m5_ohlcv', 'timestamp'),
        ('usdcop_m5_l2_features', 'timestamp'),
        ('usdcop_m5_l3_features', 'timestamp'),
        ('usdcop_m5_l4_rlready', 'timestamp'),
    ]

    results = {}

    for table_name, date_col in tables_to_validate:
        try:
            is_valid, violations = validate_table(table_name, date_col)
            results[table_name] = {
                'valid': is_valid,
                'violations': len(violations)
            }
        except Exception as e:
            logger.error(f"Error validating {table_name}: {e}")
            results[table_name] = {'valid': False, 'error': str(e)}

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)

    total_tables = len(results)
    valid_tables = sum(1 for r in results.values() if r.get('valid', False))

    logger.info(f"Tables validated: {total_tables}")
    logger.info(f"Clean tables: {valid_tables}")
    logger.info(f"Contaminated tables: {total_tables - valid_tables}")

    for table_name, result in results.items():
        status = "✓ PASS" if result.get('valid', False) else "❌ FAIL"
        logger.info(f"  {table_name}: {status}")

    if valid_tables == total_tables:
        logger.info("\n✓ ALL TABLES CLEAN")
    else:
        logger.warning("\n⚠️  SOME TABLES NEED CLEANING")


if __name__ == '__main__':
    main()
```

## Example 6: Macro Data with US Holidays

```python
"""
load_macro_data.py
==================
Load US macro data and validate against US holidays.
"""

from services.common.trading_calendar import TradingCalendar
import pandas as pd
import logging

# Create calendar that checks BOTH Colombian AND US holidays
cal = TradingCalendar(include_us_holidays=True)

def load_and_validate_macro_data(filepath: str) -> pd.DataFrame:
    """
    Load macro data (SPY, DXY, etc.) and validate.
    These use US market hours, so we check US holidays too.
    """
    df = pd.read_csv(filepath, parse_dates=['date'])

    # Validate - this will check both Colombian and US holidays
    is_valid, violations = cal.validate_no_holidays(df, date_col='date')

    if not is_valid:
        logging.warning(f"Found {len(violations)} invalid dates in macro data")

        # Separate Colombian vs US holidays
        col_holidays = [v for v in violations if 'Colombian' in v['reason']]
        us_holidays = [v for v in violations if 'US' in v['reason']]
        weekends = [v for v in violations if 'Weekend' in v['reason']]

        logging.info(f"  Colombian holidays: {len(col_holidays)}")
        logging.info(f"  US holidays: {len(us_holidays)}")
        logging.info(f"  Weekends: {len(weekends)}")

    # Clean
    df_clean = cal.filter_trading_days(df, date_col='date')

    return df_clean
```

## Example 7: Jupyter Notebook Analysis

```python
"""
In Jupyter Notebook: Data Quality Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
from services.common.trading_calendar import TradingCalendar, validate_no_holidays

# Load data
df = pd.read_csv('usdcop_m5_ohlcv.csv', parse_dates=['timestamp'])

# Initialize calendar
cal = TradingCalendar()

# Validate
is_valid, violations = validate_no_holidays(df, date_col='timestamp')

print(f"Total rows: {len(df):,}")
print(f"Valid: {is_valid}")
print(f"Violations: {len(violations)}")

# Generate and display report
report = cal.generate_report(df, date_col='timestamp')
print("\n" + report)

# Visualize violations by type
if violations:
    from collections import Counter
    reason_counts = Counter(v['reason'] for v in violations)

    plt.figure(figsize=(10, 6))
    plt.bar(reason_counts.keys(), reason_counts.values())
    plt.title('Data Quality Issues by Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Clean data
df_clean = cal.filter_trading_days(df, date_col='timestamp')

print(f"\nCleaned: {len(df_clean):,} rows")
print(f"Removed: {len(df) - len(df_clean):,} rows ({(len(df)-len(df_clean))/len(df)*100:.2f}%)")
```

## Best Practices Summary

1. **Airflow DAGs**: Always validate execution_date at start of task
2. **Training**: Validate data after loading, before training
3. **Inference**: Check time before running prediction
4. **Database Loading**: Filter data before INSERT
5. **Reports**: Generate validation reports for auditing
6. **US Macro Data**: Use `include_us_holidays=True`

## Common Pitfalls to Avoid

### ❌ DON'T: Load data without validation
```python
df = fetch_api_data()
df.to_sql('ohlcv', engine)  # May include holidays!
```

### ✓ DO: Validate before loading
```python
df = fetch_api_data()
df_clean = filter_trading_days(df)
df_clean.to_sql('ohlcv', engine)
```

### ❌ DON'T: Train on raw data
```python
model.learn(df_train)  # May have weekends!
```

### ✓ DO: Validate training data
```python
df_train = validate_training_data(df_train)
model.learn(df_train)
```

### ❌ DON'T: Run inference blindly
```python
prediction = model.predict(obs)  # May run on Sunday!
```

### ✓ DO: Check time first
```python
is_valid, _ = validate_inference_time()
if is_valid:
    prediction = model.predict(obs)
```

## Testing Your Integration

After integrating the trading calendar, verify it works:

```python
# Test script: test_calendar_integration.py
from services.common.trading_calendar import TradingCalendar
from datetime import date

cal = TradingCalendar()

# Test known holidays
test_cases = [
    (date(2025, 1, 1), False, "New Year"),
    (date(2025, 1, 6), False, "Epiphany"),
    (date(2025, 1, 7), True, "Regular Tuesday"),
    (date(2025, 1, 11), False, "Saturday"),
    (date(2025, 5, 1), False, "Labor Day"),
]

print("Testing calendar integration...")
all_passed = True

for test_date, expected, description in test_cases:
    result = cal.is_trading_day(test_date)
    status = "✓" if result == expected else "❌"

    if result != expected:
        all_passed = False

    print(f"{status} {test_date} ({description}): {result} (expected {expected})")

if all_passed:
    print("\n✓ All tests passed!")
else:
    print("\n❌ Some tests failed!")
```
