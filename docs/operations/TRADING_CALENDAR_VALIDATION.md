# Trading Calendar Validation

## Overview

The USD/COP trading API now includes comprehensive holiday and weekend validation to ensure that inference and real-time data endpoints only operate on valid trading days.

**Implementation Date**: 2025-12-17
**Version**: 1.0.0
**Author**: Pedro @ Lean Tech Solutions

---

## Features

### 1. Holiday Detection
- **Colombian Holidays**: All official Colombian public holidays for 2025
- **Movable Holidays**: Proper handling of holidays that move to Monday (Ley Emiliani)
- **Easter-Based Holidays**: Dynamic calculation of holidays relative to Easter Sunday
- **Named Holidays**: Returns the specific name of each holiday

### 2. Weekend Detection
- Identifies Saturday and Sunday as non-trading days
- Timezone-aware weekend checking using Colombian time (COT)

### 3. Trading Day Validation
- Combines holiday and weekend checks
- Provides next trading day calculation
- Clear error messages with specific reasons

---

## API Endpoints

### 1. Check Trading Day Status

**Endpoint**: `GET /api/v1/trading-calendar/is-trading-day`

**Description**: Check if a specific date is a valid trading day.

**Parameters**:
- `date` (optional): Date to check in YYYY-MM-DD format. Defaults to today.

**Response Example** (Trading Day):
```json
{
  "date": "2025-12-17",
  "is_trading_day": true,
  "is_weekend": false,
  "is_holiday": false,
  "reason": null,
  "next_trading_day": null
}
```

**Response Example** (Holiday):
```json
{
  "date": "2025-12-25",
  "is_trading_day": false,
  "is_weekend": false,
  "is_holiday": true,
  "reason": "Navidad",
  "next_trading_day": "2025-12-26"
}
```

**Response Example** (Weekend):
```json
{
  "date": "2025-12-20",
  "is_trading_day": false,
  "is_weekend": true,
  "is_holiday": false,
  "reason": "Weekend",
  "next_trading_day": "2025-12-22"
}
```

**Usage Examples**:
```bash
# Check today
curl http://localhost:8000/api/v1/trading-calendar/is-trading-day

# Check specific date
curl http://localhost:8000/api/v1/trading-calendar/is-trading-day?date=2025-12-25

# Check Independence Day
curl http://localhost:8000/api/v1/trading-calendar/is-trading-day?date=2025-07-20
```

---

### 2. Get Trading Inference

**Endpoint**: `GET /api/v1/inference`

**Description**: Get RL model trading inference. Only available on valid trading days.

**Parameters**: None

**Response Example** (Market Open):
```json
{
  "status": "success",
  "message": "Inference endpoint - implementation pending",
  "timestamp": "2025-12-17T10:30:00-05:00",
  "note": "This endpoint will return RL model predictions when implemented"
}
```

**Response Example** (Market Closed - Holiday):
```json
{
  "status": "market_closed",
  "reason": "Holiday: Navidad",
  "message": "No inference available - market is closed",
  "current_date": "2025-12-25",
  "next_trading_day": "2025-12-26",
  "is_weekend": false,
  "is_holiday": true
}
```

**Response Example** (Market Closed - Weekend):
```json
{
  "status": "market_closed",
  "reason": "Weekend - Market closed on Saturdays and Sundays",
  "message": "No inference available - market is closed",
  "current_date": "2025-12-20",
  "next_trading_day": "2025-12-22",
  "is_weekend": true,
  "is_holiday": false
}
```

**Usage Examples**:
```bash
# Get inference (will check if trading day automatically)
curl http://localhost:8000/api/v1/inference
```

---

### 3. Get Latest Price (Enhanced)

**Endpoint**: `GET /api/latest/{symbol}`

**Description**: Get latest real-time price. Now includes holiday validation in addition to market hours check.

**Parameters**:
- `symbol`: Trading symbol (default: "USDCOP")

**Response Example** (Market Closed - Holiday):
```json
{
  "detail": {
    "error": "Market is closed",
    "message": "Holiday: Día de la Independencia",
    "current_date": "2025-07-20",
    "next_trading_day": "2025-07-21",
    "is_weekend": false,
    "is_holiday": true
  }
}
```

**HTTP Status Codes**:
- `200`: Success - Price data returned
- `425`: Too Early - Market is closed (weekend, holiday, or outside trading hours)
- `404`: No data found for symbol

**Usage Examples**:
```bash
# Get latest price (will check trading day + market hours)
curl http://localhost:8000/api/latest/USDCOP
```

---

## Colombian Holidays 2025

The system recognizes the following Colombian public holidays:

| Date | Holiday Name | Type |
|------|--------------|------|
| 2025-01-01 | Año Nuevo | Fixed |
| 2025-01-06 | Día de los Reyes Magos | Movable |
| 2025-03-24 | Día de San José | Movable |
| 2025-04-17 | Jueves Santo | Easter-relative |
| 2025-04-18 | Viernes Santo | Easter-relative |
| 2025-05-01 | Día del Trabajo | Fixed |
| 2025-06-02 | Ascensión del Señor | Easter-relative |
| 2025-06-23 | Corpus Christi | Easter-relative |
| 2025-06-30 | Sagrado Corazón | Easter-relative |
| 2025-07-20 | Día de la Independencia | Fixed |
| 2025-08-07 | Batalla de Boyacá | Fixed |
| 2025-08-18 | Asunción de la Virgen | Movable |
| 2025-10-13 | Día de la Raza | Movable |
| 2025-11-03 | Día de Todos los Santos | Movable |
| 2025-11-17 | Independencia de Cartagena | Movable |
| 2025-12-08 | Día de la Inmaculada Concepción | Fixed |
| 2025-12-25 | Navidad | Fixed |

---

## Implementation Details

### Module: `services/common/trading_calendar.py`

**Key Classes**:
- `TradingCalendar`: Main calendar management class

**Key Methods**:
- `is_trading_day(date)`: Check if date is a valid trading day
- `is_holiday(date)`: Check if date is a holiday
- `is_weekend(date)`: Check if date is a weekend
- `get_holiday_name(date)`: Get the name of a holiday
- `next_trading_day(date)`: Calculate next trading day
- `previous_trading_day(date)`: Calculate previous trading day

### Integration Points

1. **API Initialization**:
   ```python
   from common.trading_calendar import TradingCalendar
   trading_cal = TradingCalendar()
   ```

2. **Validation Pattern**:
   ```python
   now = datetime.now(COT_TIMEZONE)

   if not trading_cal.is_trading_day(now):
       if trading_cal.is_weekend(now):
           reason = "Weekend - Market closed on Saturdays and Sundays"
       else:
           holiday_name = trading_cal.get_holiday_name(now)
           reason = f"Holiday: {holiday_name}"

       return {
           "status": "market_closed",
           "reason": reason,
           "next_trading_day": trading_cal.next_trading_day(now).isoformat()
       }
   ```

---

## Testing

### Automated Tests

Run the comprehensive test script:
```bash
python scripts/test_trading_calendar_api.py
```

This script tests:
- Trading day validation for various dates
- Holiday detection (Christmas, Independence Day, etc.)
- Weekend detection
- Inference endpoint behavior on non-trading days
- Latest price endpoint behavior on non-trading days

### Manual Testing

**Test Scenarios**:

1. **Regular Trading Day** (e.g., Tuesday):
   ```bash
   curl http://localhost:8000/api/v1/trading-calendar/is-trading-day?date=2025-12-17
   # Should return: is_trading_day: true
   ```

2. **Weekend**:
   ```bash
   curl http://localhost:8000/api/v1/trading-calendar/is-trading-day?date=2025-12-20
   # Should return: is_trading_day: false, reason: "Weekend"
   ```

3. **Holiday**:
   ```bash
   curl http://localhost:8000/api/v1/trading-calendar/is-trading-day?date=2025-12-25
   # Should return: is_trading_day: false, reason: "Navidad"
   ```

4. **Inference on Holiday**:
   ```bash
   # Set system date to December 25, 2025, then:
   curl http://localhost:8000/api/v1/inference
   # Should return: status: "market_closed"
   ```

---

## Error Handling

### HTTP Status Codes

- **200 OK**: Request successful, data returned
- **400 Bad Request**: Invalid date format
- **425 Too Early**: Market is closed (weekend, holiday, or outside hours)
- **500 Internal Server Error**: Server error

### Error Response Format

```json
{
  "detail": {
    "error": "Market is closed",
    "message": "Holiday: Navidad",
    "current_date": "2025-12-25",
    "next_trading_day": "2025-12-26",
    "is_weekend": false,
    "is_holiday": true
  }
}
```

---

## Future Enhancements

1. **Multi-Year Support**: Extend holiday calendar to 2026 and beyond
2. **Market-Specific Holidays**: Add option to check US market holidays separately
3. **Half Trading Days**: Support for shortened trading sessions
4. **Historical Calendar API**: Query past trading days for backtesting
5. **Calendar Subscription**: WebSocket notifications for upcoming holidays

---

## Configuration

Holiday data is stored in: `config/trading_calendar.json`

To update holidays:
1. Edit `config/trading_calendar.json`
2. Add new year's holidays under `holidays_YYYY_colombia`
3. Restart the API service

---

## Support

For issues or questions:
- **Author**: Pedro @ Lean Tech Solutions
- **Documentation**: This file
- **Test Script**: `scripts/test_trading_calendar_api.py`
- **Implementation**: `services/common/trading_calendar.py`

---

## Change Log

### Version 1.0.0 (2025-12-17)
- Initial implementation
- Colombian holidays support for 2025
- Weekend detection
- Trading day validation
- Integration with inference endpoint
- Integration with latest price endpoint
- Comprehensive test suite
