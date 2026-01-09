# Trading Calendar Test Suite

## Overview

Comprehensive test suite for holiday and weekend validation in the USD/COP trading system.

**Location:** `tests/test_trading_calendar.py`

**Purpose:** Ensure that training data, live data, and all pipeline stages correctly filter out weekends and Colombian holidays to prevent data contamination.

## Test Coverage

### 1. Weekend Validation (6 tests)
- ✅ Saturdays correctly identified as non-trading days
- ✅ Sundays correctly identified as non-trading days
- ✅ Regular weekdays identified as trading days
- ✅ Friday trading day validation
- ✅ Series filtering removes weekends

### 2. Colombian Holidays 2025 (7 tests)
- ✅ New Year's Day (Jan 1)
- ✅ Epiphany (Jan 6)
- ✅ Labor Day (May 1)
- ✅ Independence Day (Jul 20)
- ✅ Battle of Boyacá (Aug 7)
- ✅ Christmas (Dec 25)
- ✅ All holidays from `trading_calendar.json`

### 3. Easter-Based Holidays (4 tests)
- ✅ Maundy Thursday (Apr 17, 2025)
- ✅ Good Friday (Apr 18, 2025)
- ✅ Ascension Day (Jun 2, 2025 - moved to Monday)
- ✅ Corpus Christi (Jun 23, 2025 - moved to Monday)

### 4. Ley Emiliani (4 tests)
Colombian law that moves certain holidays to Monday:
- ✅ Immaculate Conception (Dec 8)
- ✅ Assumption (Aug 18 - moved from Aug 15)
- ✅ All Saints Day (Nov 3 - moved from Nov 1)
- ✅ Independence of Cartagena (Nov 17 - moved from Nov 11)

### 5. Year Boundary Edge Cases (3 tests)
- ✅ New Year's Eve is trading day
- ✅ January 2nd is trading day
- ✅ Cross-year date range filtering

### 6. DataFrame Filtering (4 tests)
- ✅ Filter removes weekends
- ✅ Filter removes holidays (Christmas)
- ✅ Filter maintains trading hours (8am-2pm COT)
- ✅ All-holiday data returns empty DataFrame

### 7. Training Data Validation (4 tests)
- ✅ Sample data has no weekends
- ✅ Sample data has no holidays
- ✅ Validation function detects contamination
- ✅ Count trading days in month

### 8. Premium Hours Validation (6 tests)
- ✅ 8:00 AM Monday is premium hours
- ✅ 7:59 AM is NOT premium hours
- ✅ 2:00 PM is NOT premium hours
- ✅ 1:59 PM is premium hours
- ✅ Saturday never premium
- ✅ Holiday handling

### 9. Edge Cases (5 tests)
- ✅ Naive datetime handling
- ✅ UTC to COT conversion
- ✅ Leap year February 29
- ✅ Midnight boundary
- ✅ Empty DataFrame handling

### 10. Real Dataset Validation (2 tests)
- ✅ L4 dataset path validation
- ✅ L4 dataset holiday contamination check

**Total: 45+ comprehensive tests**

## Running the Tests

### Run All Trading Calendar Tests
```bash
cd C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models
pytest tests/test_trading_calendar.py -v
```

### Run Specific Test Class
```bash
# Test only weekend validation
pytest tests/test_trading_calendar.py::TestWeekendValidation -v

# Test only Colombian holidays
pytest tests/test_trading_calendar.py::TestColombianHolidays2025 -v

# Test only Ley Emiliani
pytest tests/test_trading_calendar.py::TestLeyEmilianiHolidays -v
```

### Run with Coverage
```bash
pytest tests/test_trading_calendar.py --cov=airflow.dags.utils.datetime_handler --cov-report=html
```

### Run Only Unit Tests
```bash
pytest tests/test_trading_calendar.py -m unit -v
```

## Expected Output

```
tests/test_trading_calendar.py::TestWeekendValidation::test_saturday_not_trading_day PASSED
tests/test_trading_calendar.py::TestWeekendValidation::test_sunday_not_trading_day PASSED
tests/test_trading_calendar.py::TestColombianHolidays2025::test_christmas_not_trading_day PASSED
tests/test_trading_calendar.py::TestLeyEmilianiHolidays::test_assumption_moved_2025 PASSED
...

======================== 45 passed in 2.34s ========================
```

## Integration with Pipeline

### Pre-Training Validation
Before training RL models, validate the dataset:

```python
from airflow.dags.utils.datetime_handler import UnifiedDatetimeHandler
import pandas as pd

# Load L4 training data
df = pd.read_csv('data/pipeline/l4_rl_ready/training_data.csv')

# Validate
calendar = UnifiedDatetimeHandler()
df = calendar.standardize_dataframe_timestamps(df, ['timestamp'])
business_mask = calendar.is_business_day(df['timestamp'])

if not business_mask.all():
    non_business = df[~business_mask]['timestamp'].dt.date.unique()
    raise ValueError(f"Training data contaminated with holidays: {non_business}")
```

### DAG Integration
Add validation to Airflow DAGs:

```python
from utils.datetime_handler import UnifiedDatetimeHandler

def validate_no_holidays(df):
    """Task to validate data has no holidays"""
    calendar = UnifiedDatetimeHandler()
    df = calendar.standardize_dataframe_timestamps(df, ['timestamp'])
    business_mask = calendar.is_business_day(df['timestamp'])

    assert business_mask.all(), "Data contains non-trading days"

validate_task = PythonOperator(
    task_id='validate_no_holidays',
    python_callable=validate_no_holidays,
    op_kwargs={'df': '{{ ti.xcom_pull() }}'}
)
```

## Known Issues & Notes

### Ley Emiliani Complexity
The Ley Emiliani law in Colombia moves certain holidays to the following Monday. This is handled automatically by the `holidays` Python library, but manual verification for 2025 is included in tests.

### Timezone Awareness
All tests use timezone-aware datetimes (America/Bogota). The `UnifiedDatetimeHandler` ensures naive datetimes are properly localized.

### Year-Specific Holidays
Easter-based holidays change each year. The test suite uses 2025 dates. Update annually or use dynamic calculation.

## Test Dependencies

Required packages:
```
pytest>=7.0.0
pandas>=1.3.0
numpy>=1.20.0
pytz>=2021.1
holidays>=0.14
```

Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

## Continuous Integration

Add to CI/CD pipeline (`.github/workflows/tests.yml`):

```yaml
- name: Run Trading Calendar Tests
  run: |
    pytest tests/test_trading_calendar.py -v --junitxml=test-results/trading-calendar.xml
```

## Troubleshooting

### Test Failures on Holidays
If tests fail due to missing holidays, check `config/trading_calendar.json` is up to date.

### Timezone Issues
Ensure system timezone or test environment supports `America/Bogota`:
```python
import pytz
print(pytz.timezone('America/Bogota'))  # Should not raise error
```

### Dataset Not Found
L4 validation tests will skip if dataset not found:
```
tests/test_trading_calendar.py::TestRealDatasetValidation::test_validate_real_l4_dataset_if_exists SKIPPED
```

## Related Files

- **Source:** `airflow/dags/utils/datetime_handler.py` - UnifiedDatetimeHandler implementation
- **Config:** `config/trading_calendar.json` - Holiday definitions
- **Fixtures:** `tests/conftest.py` - Shared test fixtures
- **Integration:** `tests/integration/test_pipeline_api.py` - API-level validation

## Maintenance

### Annual Updates Required
1. Update `config/trading_calendar.json` with next year's holidays
2. Update test cases with new year-specific dates
3. Verify Ley Emiliani dates (moved holidays)
4. Validate Easter-based holidays

### Adding New Holidays
1. Add to `config/trading_calendar.json`
2. Add test case in `TestColombianHolidays20XX`
3. Run full test suite to verify

## Contact

**Author:** Pedro @ Lean Tech Solutions
**Date Created:** 2025-12-17
**Last Updated:** 2025-12-17

For questions or issues, refer to main project README or raise a GitHub issue.
