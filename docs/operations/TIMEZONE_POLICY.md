# Timezone Policy

## Overview

This document defines the timezone handling policy for the USDCOP Trading System.
Consistent timezone handling is critical to prevent data corruption, scheduling issues,
and look-ahead bias.

## Core Policy

| Context | Timezone | Rationale |
|---------|----------|-----------|
| Database Storage | **UTC** | Industry standard, avoids DST issues |
| API Responses | **UTC** | Client converts as needed |
| Logs & Metrics | **UTC** | Consistent across services |
| User Interface | **COT** | Colombian market hours |
| Trading Calendar | **COT** | Market operates in COT |

## Colombian Market Hours

```
Market Open:  08:00 COT (13:00 UTC)
Market Close: 12:55 COT (17:55 UTC)
```

**Important:** The Colombian forex market for USD/COP operates:
- Monday to Friday
- 8:00 AM to 12:55 PM Colombia Time (COT, America/Bogota)
- COT is UTC-5 (no daylight saving time changes)

## Implementation Guidelines

### 1. Database Queries

**Always use explicit timezone conversion in SQL:**

```sql
-- CORRECT: Explicit timezone handling
SELECT * FROM usdcop_m5_ohlcv
WHERE time AT TIME ZONE 'UTC' AT TIME ZONE 'America/Bogota'
    BETWEEN '08:00:00'::time AND '12:55:00'::time;

-- INCORRECT: Ambiguous timezone
SELECT * FROM usdcop_m5_ohlcv
WHERE time::time BETWEEN '08:00:00' AND '12:55:00';
```

### 2. Python Code

**Use the UnifiedDatetimeHandler for all timezone operations:**

```python
from utils.datetime_handler import UnifiedDatetimeHandler

# Convert to COT
cot_time = UnifiedDatetimeHandler.convert_to_cot(utc_timestamp)

# Convert to UTC
utc_time = UnifiedDatetimeHandler.convert_to_utc(cot_timestamp)

# Ensure timezone-aware
aware_dt = UnifiedDatetimeHandler.ensure_timezone_aware(naive_dt, assume_tz='America/Bogota')

# Check if within trading hours
is_trading = UnifiedDatetimeHandler.is_within_trading_hours(timestamp)
```

### 3. API Endpoints

**Always return UTC timestamps in ISO 8601 format with 'Z' suffix:**

```json
{
  "timestamp": "2025-01-14T15:30:00.000Z",
  "data": {...}
}
```

**Accept timestamps in any format, normalize to UTC internally:**

```python
from datetime import datetime
from dateutil.parser import parse

def normalize_timestamp(ts_string: str) -> datetime:
    """Parse timestamp and convert to UTC."""
    parsed = parse(ts_string)
    if parsed.tzinfo is None:
        # Assume UTC if no timezone
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
```

### 4. Logging

**All logs must include UTC timestamp:**

```python
import logging

logging.basicConfig(
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.Formatter.converter = time.gmtime  # Use UTC
```

### 5. Airflow DAGs

**Schedule in UTC, convert for business logic:**

```python
from airflow import DAG
from utils.datetime_handler import UnifiedDatetimeHandler

dag = DAG(
    'my_dag',
    # Schedule in UTC (8:00 AM COT = 13:00 UTC)
    schedule_interval='0 13 * * 1-5',  # Mon-Fri 13:00 UTC
    ...
)

def my_task(**context):
    # Convert execution_date to COT for business logic
    execution_cot = UnifiedDatetimeHandler.convert_to_cot(
        context['execution_date']
    )
```

## Common Pitfalls

### ❌ Don't Mix Naive and Aware Datetimes

```python
# WRONG - Will raise TypeError
naive = datetime.now()
aware = datetime.now(timezone.utc)
diff = aware - naive  # TypeError!

# CORRECT
from utils.datetime_handler import UnifiedDatetimeHandler
dt1 = UnifiedDatetimeHandler.ensure_timezone_aware(datetime.now())
dt2 = datetime.now(timezone.utc)
diff = dt2 - dt1  # Works!
```

### ❌ Don't Assume Timezone from Time String

```python
# WRONG - Assumes local timezone
pd.to_datetime('2025-01-14 08:00:00')

# CORRECT - Explicit timezone
pd.to_datetime('2025-01-14 08:00:00').tz_localize('America/Bogota')
```

### ❌ Don't Use datetime.now() Without Timezone

```python
# WRONG
now = datetime.now()

# CORRECT
from datetime import timezone
now = datetime.now(timezone.utc)

# OR using our handler
now = UnifiedDatetimeHandler.now_utc()
```

## Database Schema Requirements

All timestamp columns should be:

```sql
-- REQUIRED: Use TIMESTAMPTZ (timestamp with time zone)
CREATE TABLE my_table (
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    trade_time TIMESTAMPTZ NOT NULL
);

-- AVOID: Plain TIMESTAMP (no timezone info)
-- This causes ambiguity when querying across timezones
```

## Testing

All timezone-sensitive code must have tests that verify:

1. Correct conversion between UTC and COT
2. Handling of DST transitions (even though COT doesn't have DST, other systems might)
3. Consistent behavior with naive and aware inputs
4. Proper filtering of trading hours

Example test:

```python
def test_trading_hours_filter():
    """Test that trading hours filter works correctly."""
    handler = UnifiedDatetimeHandler()

    # 10:00 AM COT = 15:00 UTC (within trading hours)
    in_hours = datetime(2025, 1, 14, 15, 0, 0, tzinfo=timezone.utc)
    assert handler.is_within_trading_hours(in_hours) == True

    # 2:00 PM COT = 19:00 UTC (after market close)
    after_close = datetime(2025, 1, 14, 19, 0, 0, tzinfo=timezone.utc)
    assert handler.is_within_trading_hours(after_close) == False
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-14 | Initial timezone policy |

## References

- [PostgreSQL Date/Time Types](https://www.postgresql.org/docs/current/datatype-datetime.html)
- [Python pytz Documentation](https://pythonhosted.org/pytz/)
- [ISO 8601 Standard](https://en.wikipedia.org/wiki/ISO_8601)
- [Colombian Market Hours](https://www.banrep.gov.co)
