"""
Input Validation Module for USD/COP Trading Services
=====================================================

Provides centralized input validation for API endpoints to ensure
data integrity and prevent invalid requests.

SOLID Compliance:
    - SRP: Single responsibility for validation
    - OCP: Extendable validators without modification
    - DIP: Validators are injectable

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from fastapi import HTTPException, Query
import re


# =============================================================================
# CONSTANTS (from feature_config.json)
# =============================================================================

SUPPORTED_SYMBOLS = ['USDCOP', 'USD/COP']
SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
MAX_LIMIT = 10000
DEFAULT_LIMIT = 1000
MIN_LIMIT = 1
MAX_DATE_RANGE_DAYS = 365
BARS_PER_SESSION = 60


# =============================================================================
# PYDANTIC VALIDATION MODELS
# =============================================================================

class SymbolParam(BaseModel):
    """Validated symbol parameter"""
    symbol: str = Field(..., description="Trading symbol")

    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate and normalize symbol"""
        normalized = v.upper().replace('/', '').replace('-', '').strip()
        if normalized not in ['USDCOP']:
            raise ValueError(f"Unsupported symbol: {v}. Supported: USDCOP, USD/COP")
        return normalized


class TimeframeParam(BaseModel):
    """Validated timeframe parameter"""
    timeframe: str = Field('5m', description="Chart timeframe")

    @validator('timeframe')
    def validate_timeframe(cls, v):
        """Validate timeframe"""
        if v.lower() not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {v}. Supported: {SUPPORTED_TIMEFRAMES}")
        return v.lower()


class DateRangeParam(BaseModel):
    """Validated date range parameters"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @validator('start_date', 'end_date', pre=True)
    def validate_date_format(cls, v):
        """Validate ISO date format"""
        if v is None:
            return v
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid date format: {v}. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

    @root_validator(skip_on_failure=True)
    def validate_date_range(cls, values):
        """Validate date range is not too large"""
        start = values.get('start_date')
        end = values.get('end_date')

        if start and end:
            try:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))

                if start_dt > end_dt:
                    raise ValueError("start_date must be before end_date")

                if (end_dt - start_dt).days > MAX_DATE_RANGE_DAYS:
                    raise ValueError(f"Date range exceeds maximum of {MAX_DATE_RANGE_DAYS} days")
            except ValueError as e:
                if "start_date must be" in str(e) or "Date range exceeds" in str(e):
                    raise e
                pass  # Ignore parsing errors, already validated above

        return values


class LimitParam(BaseModel):
    """Validated limit parameter"""
    limit: int = Field(DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT)


class InferenceRequest(BaseModel):
    """Validated inference request for RL model"""
    observation: List[float] = Field(..., min_items=15, max_items=15)
    position: float = Field(..., ge=-1.0, le=1.0)
    step: int = Field(..., ge=1, le=BARS_PER_SESSION)

    @validator('observation')
    def validate_observation(cls, v):
        """Validate observation values are in expected range"""
        for i, val in enumerate(v):
            if abs(val) > 5.0:
                raise ValueError(f"Observation[{i}] = {val} exceeds expected range [-5, 5]")
        return v


class BarNumberParam(BaseModel):
    """Validated bar number parameter for time_normalized calculation"""
    bar_number: int = Field(..., ge=1, le=BARS_PER_SESSION)


# =============================================================================
# VALIDATION FUNCTIONS (for use in FastAPI endpoints)
# =============================================================================

def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize trading symbol.

    Args:
        symbol: Trading symbol (e.g., 'USDCOP', 'USD/COP')

    Returns:
        Normalized symbol (e.g., 'USDCOP')

    Raises:
        HTTPException: If symbol is not supported
    """
    normalized = symbol.upper().replace('/', '').replace('-', '').strip()
    if normalized not in ['USDCOP']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported symbol: {symbol}. Supported symbols: USDCOP, USD/COP"
        )
    return normalized


def validate_timeframe(timeframe: str) -> str:
    """
    Validate chart timeframe.

    Args:
        timeframe: Timeframe string (e.g., '5m', '1h')

    Returns:
        Validated timeframe

    Raises:
        HTTPException: If timeframe is not supported
    """
    if timeframe.lower() not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported timeframe: {timeframe}. Supported: {SUPPORTED_TIMEFRAMES}"
        )
    return timeframe.lower()


def validate_limit(limit: int) -> int:
    """
    Validate limit parameter.

    Args:
        limit: Maximum number of records

    Returns:
        Validated limit

    Raises:
        HTTPException: If limit is out of range
    """
    if limit < MIN_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be at least {MIN_LIMIT}"
        )
    if limit > MAX_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"limit cannot exceed {MAX_LIMIT}"
        )
    return limit


def validate_date_range(
    start_date: Optional[str],
    end_date: Optional[str]
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Validate date range parameters.

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format

    Returns:
        Tuple of (start_datetime, end_datetime)

    Raises:
        HTTPException: If dates are invalid or range is too large
    """
    start_dt = None
    end_dt = None

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid start_date format: {start_date}. Use ISO format."
            )

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid end_date format: {end_date}. Use ISO format."
            )

    if start_dt and end_dt:
        if start_dt > end_dt:
            raise HTTPException(
                status_code=400,
                detail="start_date must be before end_date"
            )

        if (end_dt - start_dt).days > MAX_DATE_RANGE_DAYS:
            raise HTTPException(
                status_code=400,
                detail=f"Date range cannot exceed {MAX_DATE_RANGE_DAYS} days"
            )

    return start_dt, end_dt


def validate_bar_number(bar_number: int) -> int:
    """
    Validate bar number for episode (1-60 for 5min bars).

    Args:
        bar_number: Current bar in episode

    Returns:
        Validated bar number

    Raises:
        HTTPException: If bar_number is out of range
    """
    if bar_number < 1 or bar_number > BARS_PER_SESSION:
        raise HTTPException(
            status_code=400,
            detail=f"bar_number must be between 1 and {BARS_PER_SESSION}"
        )
    return bar_number


def validate_observation(observation: List[float]) -> List[float]:
    """
    Validate observation vector for RL inference.

    Args:
        observation: List of 15 float values

    Returns:
        Validated observation

    Raises:
        HTTPException: If observation is invalid
    """
    if len(observation) != 15:
        raise HTTPException(
            status_code=400,
            detail=f"Observation must have exactly 15 values, got {len(observation)}"
        )

    for i, val in enumerate(observation):
        if not isinstance(val, (int, float)):
            raise HTTPException(
                status_code=400,
                detail=f"Observation[{i}] must be a number, got {type(val).__name__}"
            )
        if abs(val) > 5.0:
            raise HTTPException(
                status_code=400,
                detail=f"Observation[{i}] = {val} exceeds expected range [-5, 5]"
            )

    return observation


def validate_position(position: float) -> float:
    """
    Validate position value for RL inference.

    Args:
        position: Current position (-1 to 1)

    Returns:
        Validated position

    Raises:
        HTTPException: If position is out of range
    """
    if position < -1.0 or position > 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"Position must be between -1 and 1, got {position}"
        )
    return position


# =============================================================================
# SQL INJECTION PREVENTION
# =============================================================================

def sanitize_identifier(identifier: str) -> str:
    """
    Sanitize SQL identifier to prevent injection.

    Args:
        identifier: SQL identifier (table name, column name)

    Returns:
        Sanitized identifier

    Raises:
        HTTPException: If identifier contains invalid characters
    """
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid identifier: {identifier}"
        )
    return identifier
