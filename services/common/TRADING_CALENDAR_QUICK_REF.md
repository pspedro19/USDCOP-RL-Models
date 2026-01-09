# Trading Calendar - Quick Reference

## Installation

```bash
pip install colombian-holidays==0.5.1 holidays==0.37
```

## Basic Usage

```python
from services.common.trading_calendar import is_trading_day, filter_trading_days

# Check if date is a trading day
if is_trading_day(my_date):
    process_data()

# Filter DataFrame to trading days only
df_clean = filter_trading_days(df, date_col='timestamp')
```

## Airflow DAG Integration

```python
from services.common.trading_calendar import validate_dag_execution_date

def my_task(**context):
    if not validate_dag_execution_date(context['execution_date']):
        logging.info("Skipping: not a trading day")
        return
    # ... process data
```

## Training Data Validation

```python
from services.common.trading_calendar import validate_training_data

# In run.py
df_train = validate_training_data(df_train_raw, data_type='training')
df_test = validate_training_data(df_test_raw, data_type='test')
```

## Production Inference

```python
from services.common.trading_calendar import validate_inference_time

is_valid, msg = validate_inference_time()
if is_valid:
    prediction = model.predict(observation)
else:
    logging.info(f"Skipping: {msg}")
```

## Data Validation

```python
from services.common.trading_calendar import validate_no_holidays

is_valid, violations = validate_no_holidays(df, date_col='timestamp')
if not is_valid:
    print(f"Found {len(violations)} non-trading days!")
    for v in violations[:5]:
        print(f"  {v['date']}: {v['reason']}")
```

## Common Patterns

### 1. Database Load (Airflow)
```python
def load_ohlcv(**context):
    if not validate_dag_execution_date(context['execution_date']):
        return
    # ... load data
```

### 2. Training Pipeline
```python
df_train = validate_training_data(df_train, data_type='training')
model.learn(df_train)
```

### 3. Real-time Inference
```python
is_valid, _ = validate_inference_time()
if is_valid:
    run_inference()
```

### 4. Data Cleaning
```python
df = filter_trading_days(df, date_col='timestamp')
```

## Key Functions

| Function | Purpose |
|----------|---------|
| `is_trading_day(dt)` | Check if date is a trading day |
| `filter_trading_days(df)` | Remove holidays/weekends from DataFrame |
| `validate_no_holidays(df)` | Check if DataFrame is clean |
| `validate_training_data(df)` | Clean and validate ML training data |
| `validate_dag_execution_date(dt)` | Check if Airflow should run |
| `validate_inference_time(dt)` | Check if inference should run |

## Testing

```bash
python -m services.common.trading_calendar
```

## What Gets Filtered?

- **Weekends**: Saturday, Sunday
- **Colombian Holidays**: New Year, Reyes Magos, Holy Week, Labor Day, Independence Day, etc.
- **US Holidays** (optional): NYSE holidays for macro data

## Example Output

```python
>>> from services.common.trading_calendar import TradingCalendar
>>> cal = TradingCalendar()
>>> cal.is_trading_day(date(2025, 1, 1))
False  # New Year
>>> cal.get_violation_reason(date(2025, 1, 1))
'Colombian Holiday'
>>> cal.is_trading_day(date(2025, 1, 7))
True  # Regular Tuesday
```

## See Also

- Full documentation: `TRADING_CALENDAR_USAGE.md`
- Source code: `trading_calendar.py`
