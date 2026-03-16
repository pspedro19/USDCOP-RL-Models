# Trading Calendar Migration Checklist

## Overview

This checklist helps you integrate the Colombian holiday validation layer into your existing USD/COP trading system.

## Pre-Migration

### 1. Install Dependencies

```bash
# In your Docker containers or virtual environment
pip install colombian-holidays==0.5.1
pip install holidays==0.37
```

### 2. Update requirements.txt

Add to `services/requirements.txt`:
```
# Trading Calendar - Colombian Holidays
colombian-holidays==0.5.1
holidays==0.37
```

### 3. Verify Installation

```bash
cd services/common
python -m trading_calendar
```

Expected output: Test results showing holiday detection working.

## Migration Tasks

### Phase 1: Airflow DAGs (HIGH PRIORITY)

#### 1.1 Update L0 OHLCV Loading DAGs

**Files to modify:**
- `airflow/dags/l0_ohlcv_realtime.py`
- `airflow/dags/l0_ohlcv_backfill.py`

**Changes:**
```python
# Add at top of file
from services.common.trading_calendar import validate_dag_execution_date

# In task function
def load_ohlcv_data(**context):
    # ADD THIS CHECK
    if not validate_dag_execution_date(context['execution_date']):
        logging.info("Skipping: not a trading day")
        return

    # ... rest of function
```

**Test:**
```bash
# Test DAG on a known holiday
airflow dags test l0_ohlcv_realtime 2025-01-01
# Should skip with message "not a trading day"

# Test DAG on a trading day
airflow dags test l0_ohlcv_realtime 2025-01-07
# Should run normally
```

**Status:** [ ] Complete

---

#### 1.2 Update L1 Feature Engineering DAGs

**Files to modify:**
- `airflow/dags/l1_feature_refresh.py`

**Changes:**
```python
from services.common.trading_calendar import validate_dag_execution_date

def refresh_features(**context):
    if not validate_dag_execution_date(context['execution_date']):
        logging.info("Skipping: not a trading day")
        return
    # ... rest
```

**Status:** [ ] Complete

---

#### 1.3 Update Macro Data Loading (if applicable)

**Files to modify:**
- `airflow/dags/l0_macro_daily.py`

**Changes:**
```python
# For US macro data, use US holiday validation
from services.common.trading_calendar import TradingCalendar

cal = TradingCalendar(include_us_holidays=True)

def load_macro_data(**context):
    execution_date = context['execution_date']
    if not cal.is_trading_day(execution_date):
        logging.info("Skipping: not a trading day")
        return
    # ... rest
```

**Status:** [ ] Complete

---

### Phase 2: Training Pipeline (CRITICAL)

#### 2.1 Update run.py or Training Scripts

**Files to modify:**
- `notebooks/pipeline entrenamiento/run.py`
- Any other training scripts

**Changes:**
```python
# Add import
from services.common.trading_calendar import validate_training_data

# In run_fold or training function
def run_fold(fold, df, features, log):
    # ... create splits ...

    df_train_raw = df[train_mask].copy()
    df_test_raw = df[test_mask].copy()

    # ADD VALIDATION HERE
    df_train = validate_training_data(df_train_raw, data_type=f'train_fold_{fold["fold_id"]}')
    df_test = validate_training_data(df_test_raw, data_type=f'test_fold_{fold["fold_id"]}')

    log.log(f"Train: {len(df_train)} rows (removed {len(df_train_raw)-len(df_train)} non-trading days)")
    log.log(f"Test: {len(df_test)} rows (removed {len(df_test_raw)-len(df_test)} non-trading days)")

    # Continue with training...
```

**Test:**
```bash
# Run training on a small fold to verify
python run.py --fold 0
# Check logs for "Validating training data" messages
```

**Status:** [ ] Complete

---

### Phase 3: Production Inference (CRITICAL)

#### 3.1 Update Realtime Inference Service

**Files to modify:**
- `services/trading_api_realtime.py`
- Any scheduled inference scripts

**Changes:**
```python
# Add import
from services.common.trading_calendar import validate_inference_time

# In inference function
def run_inference():
    # ADD THIS CHECK
    is_valid, message = validate_inference_time()
    if not is_valid:
        logging.debug(f"Skipping inference: {message}")
        return

    # ... run model inference
```

**Test:**
```bash
# Test during market hours on a trading day
curl http://localhost:8000/api/v1/inference/run
# Should run

# Test on a weekend or holiday
# Should skip with message
```

**Status:** [ ] Complete

---

### Phase 4: Data Validation (RECOMMENDED)

#### 4.1 Validate Existing Database Data

**Create script:** `scripts/validate_database_holidays.py`

```python
"""
One-time script to check existing data for holiday contamination.
"""
from services.common.trading_calendar import TradingCalendar, validate_no_holidays
from services.common.database import get_db_connection
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cal = TradingCalendar()

def validate_table(table_name, date_col='timestamp'):
    logger.info(f"Validating {table_name}...")

    conn = get_db_connection()
    query = f"SELECT {date_col} FROM {table_name}"
    df = pd.read_sql(query, conn, parse_dates=[date_col])
    conn.close()

    is_valid, violations = validate_no_holidays(df, date_col=date_col)

    if is_valid:
        logger.info(f"✓ {table_name}: CLEAN")
    else:
        logger.warning(f"❌ {table_name}: {len(violations)} violations found")

    return is_valid, violations

# Validate critical tables
tables = [
    ('usdcop_m5_ohlcv', 'timestamp'),
    ('usdcop_m5_l2_features', 'timestamp'),
    ('usdcop_m5_l3_features', 'timestamp'),
    ('usdcop_m5_l4_rlready', 'timestamp'),
]

for table_name, date_col in tables:
    validate_table(table_name, date_col)
```

**Run:**
```bash
python scripts/validate_database_holidays.py > validation_report.txt
```

**Status:** [ ] Complete

---

#### 4.2 Clean Existing Data (if contaminated)

**If validation found issues, create cleanup script:**

```python
"""
scripts/clean_database_holidays.py
"""
from services.common.trading_calendar import TradingCalendar
from services.common.database import get_db_connection
import logging

cal = TradingCalendar()

def clean_table(table_name, date_col='timestamp'):
    """Remove holidays/weekends from table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all dates
    cursor.execute(f"SELECT DISTINCT {date_col} FROM {table_name}")
    dates = cursor.fetchall()

    # Find non-trading days
    to_delete = []
    for (dt,) in dates:
        if not cal.is_trading_day(dt):
            to_delete.append(dt)

    if to_delete:
        logger.warning(f"Deleting {len(to_delete)} non-trading days from {table_name}")

        # Delete non-trading days
        for dt in to_delete:
            cursor.execute(f"DELETE FROM {table_name} WHERE {date_col} = %s", (dt,))

        conn.commit()
        logger.info(f"✓ Cleaned {table_name}")
    else:
        logger.info(f"✓ {table_name} already clean")

    cursor.close()
    conn.close()

# Clean all tables
for table_name, date_col in tables:
    clean_table(table_name, date_col)
```

**⚠️ WARNING:** This deletes data. Back up first!

```bash
# Backup first
pg_dump -h localhost -U admin usdcop_trading > backup_before_cleaning.sql

# Then clean
python scripts/clean_database_holidays.py
```

**Status:** [ ] Complete

---

### Phase 5: API Endpoints (OPTIONAL)

#### 5.1 Add Validation Endpoints

**Add to `services/trading_api_realtime.py`:**

```python
from services.common.trading_calendar import TradingCalendar, validate_inference_time

cal = TradingCalendar()

@app.get("/api/v1/calendar/is-trading-day")
async def check_trading_day(date: str):
    """Check if a date is a trading day."""
    try:
        dt = datetime.fromisoformat(date)
        is_trading = cal.is_trading_day(dt)
        reason = cal.get_violation_reason(dt)

        return {
            'date': date,
            'is_trading_day': is_trading,
            'reason': reason
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")


@app.get("/api/v1/calendar/should-trade-now")
async def should_trade_now():
    """Check if we should be trading right now."""
    is_valid, message = validate_inference_time()

    return {
        'should_trade': is_valid,
        'message': message,
        'timestamp': datetime.now()
    }
```

**Test:**
```bash
curl http://localhost:8000/api/v1/calendar/is-trading-day?date=2025-01-01
curl http://localhost:8000/api/v1/calendar/should-trade-now
```

**Status:** [ ] Complete

---

### Phase 6: Documentation & Testing

#### 6.1 Update README

Add section to main README:

```markdown
## Trading Calendar

This project uses Colombian holiday validation to ensure no weekends or holidays
are used in training, validation, testing, or production inference.

### Key Features
- Automatic Colombian holiday detection using `colombian-holidays` library
- Weekend filtering (Saturday/Sunday)
- US holiday support for macro data
- Integration with Airflow DAGs, training pipelines, and inference services

### Usage
See `docs/TRADING_CALENDAR_USAGE.md` for detailed usage examples.
```

**Status:** [ ] Complete

---

#### 6.2 Create Test Suite

**Create:** `tests/test_trading_calendar.py`

```python
import pytest
from datetime import date
from services.common.trading_calendar import TradingCalendar

def test_known_holidays():
    cal = TradingCalendar()

    # New Year
    assert not cal.is_trading_day(date(2025, 1, 1))

    # Epiphany
    assert not cal.is_trading_day(date(2025, 1, 6))

    # Labor Day
    assert not cal.is_trading_day(date(2025, 5, 1))

def test_weekends():
    cal = TradingCalendar()

    # Saturday
    assert not cal.is_trading_day(date(2025, 1, 11))

    # Sunday
    assert not cal.is_trading_day(date(2025, 1, 12))

def test_regular_trading_days():
    cal = TradingCalendar()

    # Regular Tuesday
    assert cal.is_trading_day(date(2025, 1, 7))

    # Regular Friday
    assert cal.is_trading_day(date(2025, 1, 10))

def test_filter_dataframe():
    import pandas as pd
    from services.common.trading_calendar import filter_trading_days

    # Create test data with holidays
    dates = pd.date_range('2025-01-01', '2025-01-31', freq='D')
    df = pd.DataFrame({'timestamp': dates, 'value': range(len(dates))})

    # Filter
    df_clean = filter_trading_days(df, date_col='timestamp')

    # Should have fewer rows
    assert len(df_clean) < len(df)

    # Should not have Jan 1 (New Year)
    assert date(2025, 1, 1) not in df_clean['timestamp'].dt.date.values

    # Should not have weekends
    for dt in df_clean['timestamp']:
        assert dt.weekday() < 5

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Run:**
```bash
pytest tests/test_trading_calendar.py -v
```

**Status:** [ ] Complete

---

## Verification Checklist

After completing migration, verify:

### Database Layer
- [ ] New data loaded only on trading days
- [ ] Existing data validated and cleaned
- [ ] No holidays/weekends in any tables

### Training Pipeline
- [ ] Training data validated before use
- [ ] Test data validated before use
- [ ] Logs show validation messages
- [ ] Models trained only on trading days

### Production Inference
- [ ] Inference only runs on trading days
- [ ] Inference only runs during market hours
- [ ] Skipped inferences logged properly

### Airflow DAGs
- [ ] DAGs skip execution on holidays/weekends
- [ ] Logs show skip messages
- [ ] Backfill only processes trading days

### Documentation
- [ ] README updated
- [ ] Team trained on new validation
- [ ] Usage examples documented

## Rollback Plan

If issues occur, you can rollback:

### 1. Revert Code Changes
```bash
git checkout HEAD~1 airflow/dags/*.py
git checkout HEAD~1 services/*.py
git checkout HEAD~1 notebooks/*/run.py
```

### 2. Restore Database (if cleaned)
```bash
psql -h localhost -U admin usdcop_trading < backup_before_cleaning.sql
```

### 3. Remove Dependencies
```bash
pip uninstall colombian-holidays holidays
```

## Post-Migration Monitoring

Monitor for 1 week after migration:

### Daily Checks
- [ ] Airflow DAGs running/skipping appropriately
- [ ] Training pipeline logs show validation
- [ ] Inference service respecting market hours
- [ ] No errors in logs related to calendar

### Weekly Checks
- [ ] Database still clean (no new contamination)
- [ ] Model performance stable
- [ ] No unexpected data gaps

## Support

If you encounter issues:

1. Check logs for error messages
2. Review `docs/TRADING_CALENDAR_USAGE.md`
3. Run test suite: `python -m services.common.trading_calendar`
4. Check calendar is detecting holidays correctly

## Summary

This migration ensures your USD/COP trading system maintains data integrity by:

✓ Preventing holiday/weekend data from entering database
✓ Validating training data before model training
✓ Enforcing trading hours in production inference
✓ Providing audit trail of validation

**Estimated Migration Time:** 2-4 hours

**Priority:**
1. Training Pipeline (CRITICAL - affects model quality)
2. Production Inference (CRITICAL - affects live trading)
3. Airflow DAGs (HIGH - prevents contamination)
4. Database Validation (RECOMMENDED - audit existing data)
5. API Endpoints (OPTIONAL - nice to have)
