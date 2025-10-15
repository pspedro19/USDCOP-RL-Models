"""
Optimized L0 Data Validator with Multi-API Key Support
====================================================

Features:
- Validates historical data from 2020-2025 with 5-minute intervals
- Market hours validation: 8AM-12:55PM COT
- Intelligent API key rotation for 16 keys across 2 groups
- Real-time WebSocket integration during market hours
- Optimized database insertion with batch processing
- Comprehensive data quality validation

Author: USDCOP Trading Team
Version: 3.0.0
"""

import asyncio
import asyncpg
import json
import logging
import os
import pytz
import redis
import websockets
from datetime import datetime, time, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import requests
import time as time_module
from dataclasses import dataclass
from enum import Enum
import aiohttp
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')

# Market hours configuration
MARKET_START_HOUR = 8
MARKET_START_MINUTE = 0
MARKET_END_HOUR = 12
MARKET_END_MINUTE = 55

# Data validation thresholds
MIN_COMPLETENESS = 95.0
MAX_GAP_TOLERANCE_MINUTES = 7.5
EXPECTED_BARS_PER_DAY = 59  # 8:00 AM to 12:55 PM = 4h 55min = 59 bars (5-min intervals)
BATCH_SIZE = 1000  # Records per database batch

class ValidationSeverity(Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    CRITICAL = "CRITICAL"

@dataclass
class APIKeyGroup:
    group_name: str
    keys: List[str]
    current_index: int = 0
    calls_made: int = 0
    rate_limit_per_minute: int = 8
    last_call_time: Optional[datetime] = None

class APIKeyManager:
    """Manages rotation and rate limiting for 16 API keys across 2 groups"""

    def __init__(self):
        self.groups = self._load_api_keys_from_env()
        self.current_group_index = 0
        self._lock = asyncio.Lock()

    def _load_api_keys_from_env(self) -> List[APIKeyGroup]:
        """Load API keys from environment variables"""
        grupo_1_keys = []
        grupo_2_keys = []

        # Load GRUPO 1 API keys
        for i in range(1, 9):
            key = os.getenv(f'API_KEY_G1_{i}')
            if key and key.strip():
                grupo_1_keys.append(key.strip())
            else:
                logger.warning(f"API_KEY_G1_{i} not found or empty in environment")

        # Load GRUPO 2 API keys
        for i in range(1, 9):
            key = os.getenv(f'API_KEY_G2_{i}')
            if key and key.strip():
                grupo_2_keys.append(key.strip())
            else:
                logger.warning(f"API_KEY_G2_{i} not found or empty in environment")

        # Fallback: try the format you provided in the message
        if not grupo_1_keys:
            fallback_keys_g1 = [
                os.getenv('API_KEY_1'), os.getenv('API_KEY_2'), os.getenv('API_KEY_3'), os.getenv('API_KEY_4'),
                os.getenv('API_KEY_5'), os.getenv('API_KEY_6'), os.getenv('API_KEY_7'), os.getenv('API_KEY_8')
            ]
            grupo_1_keys = [key for key in fallback_keys_g1 if key and key.strip()]

        if not grupo_2_keys:
            fallback_keys_g2 = [
                os.getenv('API_KEY_9'), os.getenv('API_KEY_10'), os.getenv('API_KEY_11'), os.getenv('API_KEY_12'),
                os.getenv('API_KEY_13'), os.getenv('API_KEY_14'), os.getenv('API_KEY_15'), os.getenv('API_KEY_16')
            ]
            grupo_2_keys = [key for key in fallback_keys_g2 if key and key.strip()]

        if not grupo_1_keys and not grupo_2_keys:
            logger.error("❌ No API keys found! Please set API keys in environment variables:")
            logger.error("GRUPO 1: API_KEY_G1_1 through API_KEY_G1_8")
            logger.error("GRUPO 2: API_KEY_G2_1 through API_KEY_G2_8")
            raise ValueError("No API keys configured in environment variables")

        groups = []
        if grupo_1_keys:
            groups.append(APIKeyGroup(group_name="GRUPO_1", keys=grupo_1_keys))
            logger.info(f"✅ Loaded {len(grupo_1_keys)} API keys for GRUPO_1")

        if grupo_2_keys:
            groups.append(APIKeyGroup(group_name="GRUPO_2", keys=grupo_2_keys))
            logger.info(f"✅ Loaded {len(grupo_2_keys)} API keys for GRUPO_2")

        logger.info(f"🔑 Total API keys loaded: {sum(len(g.keys) for g in groups)}")
        return groups

    async def get_next_key(self) -> Tuple[str, str]:
        """Get next available API key with intelligent rotation"""
        async with self._lock:
            current_group = self.groups[self.current_group_index]

            # Check if we need to wait for rate limit
            if current_group.last_call_time:
                time_since_last_call = datetime.now() - current_group.last_call_time
                if time_since_last_call.total_seconds() < 8:  # 8 second rate limit
                    wait_time = 8 - time_since_last_call.total_seconds()
                    logger.info(f"Rate limiting {current_group.group_name}: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)

            # Get current key
            current_key = current_group.keys[current_group.current_index]

            # Update counters
            current_group.calls_made += 1
            current_group.last_call_time = datetime.now()

            # Rotate within group if needed (every 8 calls)
            if current_group.calls_made % 8 == 0:
                current_group.current_index = (current_group.current_index + 1) % len(current_group.keys)
                logger.info(f"Rotating to next key in {current_group.group_name}: {current_group.current_index + 1}/8")

            # Rotate between groups every 64 calls (8 keys × 8 calls each)
            if current_group.calls_made % 64 == 0:
                self.current_group_index = (self.current_group_index + 1) % len(self.groups)
                logger.info(f"Switching to {self.groups[self.current_group_index].group_name}")

            return current_key, current_group.group_name

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API key usage statistics"""
        stats = {}
        total_calls = 0

        for group in self.groups:
            stats[group.group_name] = {
                'total_calls': group.calls_made,
                'current_key_index': group.current_index + 1,
                'keys_count': len(group.keys),
                'last_call': group.last_call_time.isoformat() if group.last_call_time else None
            }
            total_calls += group.calls_made

        stats['total_calls'] = total_calls
        stats['active_group'] = self.groups[self.current_group_index].group_name

        return stats

class ColombianHolidayCalendar:
    """Complete Colombian holiday calendar 2020-2025"""

    def __init__(self):
        # Fixed holidays that don't move
        self.fixed_holidays = self._generate_fixed_holidays()

        # Variable holidays (moved to Monday)
        self.variable_holidays = self._generate_variable_holidays()

        # Easter-related holidays
        self.easter_holidays = self._generate_easter_holidays()

        # Combine all holidays
        self.all_holidays = set(self.fixed_holidays + self.variable_holidays + self.easter_holidays)

    def _generate_fixed_holidays(self) -> List[date]:
        """Generate fixed holidays 2020-2025"""
        holidays = []
        for year in range(2020, 2026):
            holidays.extend([
                date(year, 1, 1),   # New Year's Day
                date(year, 5, 1),   # Labor Day
                date(year, 7, 20),  # Independence Day
                date(year, 8, 7),   # Battle of Boyacá
                date(year, 12, 8),  # Immaculate Conception
                date(year, 12, 25), # Christmas Day
            ])
        return holidays

    def _generate_variable_holidays(self) -> List[date]:
        """Generate variable holidays that move to Monday"""
        holidays = []
        for year in range(2020, 2026):
            # These holidays are moved to the following Monday if they fall on a weekend
            base_holidays = [
                date(year, 1, 6),   # Three Kings Day
                date(year, 3, 19),  # St. Joseph's Day
                date(year, 6, 29),  # St. Peter and St. Paul
                date(year, 8, 15),  # Assumption of Mary
                date(year, 10, 12), # Columbus Day
                date(year, 11, 1),  # All Saints' Day
                date(year, 11, 11), # Independence of Cartagena
            ]

            for holiday in base_holidays:
                # Move to Monday if falls on weekend
                if holiday.weekday() >= 5:  # Saturday or Sunday
                    days_to_monday = 7 - holiday.weekday()
                    holiday = holiday + timedelta(days=days_to_monday)
                holidays.append(holiday)

        return holidays

    def _generate_easter_holidays(self) -> List[date]:
        """Generate Easter-related holidays"""
        holidays = []

        # Easter dates for 2020-2025 (pre-calculated)
        easter_dates = {
            2020: date(2020, 4, 12),
            2021: date(2021, 4, 4),
            2022: date(2022, 4, 17),
            2023: date(2023, 4, 9),
            2024: date(2024, 3, 31),
            2025: date(2025, 4, 20)
        }

        for year, easter in easter_dates.items():
            holidays.extend([
                easter - timedelta(days=3),  # Maundy Thursday
                easter - timedelta(days=2),  # Good Friday
                easter + timedelta(days=43), # Ascension Day
                easter + timedelta(days=64), # Corpus Christi
                easter + timedelta(days=71), # Sacred Heart
            ])

        return holidays

    def is_holiday(self, check_date: date) -> bool:
        """Check if a date is a Colombian holiday"""
        return check_date in self.all_holidays

    def get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """Get list of trading days (business days excluding holidays)"""
        trading_days = []
        current_date = start_date

        while current_date <= end_date:
            # Check if it's a weekday and not a holiday
            if current_date.weekday() < 5 and not self.is_holiday(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

class MarketHoursValidator:
    """Validates market hours and session information"""

    def __init__(self):
        self.holiday_calendar = ColombianHolidayCalendar()

    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given timestamp"""
        # Ensure timestamp is in COT
        if timestamp.tzinfo is None:
            timestamp = COT_TZ.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(COT_TZ)

        # Check if it's a weekday
        if timestamp.weekday() >= 5:
            return False

        # Check if it's a holiday
        if self.holiday_calendar.is_holiday(timestamp.date()):
            return False

        # Check time range
        current_time = timestamp.time()
        market_start = time(MARKET_START_HOUR, MARKET_START_MINUTE)
        market_end = time(MARKET_END_HOUR, MARKET_END_MINUTE)

        return market_start <= current_time <= market_end

    def get_market_session_bounds(self, session_date: date) -> Tuple[datetime, datetime]:
        """Get market session start/end times for a given date"""
        start_dt = COT_TZ.localize(datetime.combine(session_date, time(MARKET_START_HOUR, MARKET_START_MINUTE)))
        end_dt = COT_TZ.localize(datetime.combine(session_date, time(MARKET_END_HOUR, MARKET_END_MINUTE)))
        return start_dt, end_dt

    def filter_market_hours_data(self, df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
        """Filter dataframe to only include market hours data"""
        if time_col not in df.columns:
            logger.warning(f"Column '{time_col}' not found in dataframe")
            return df

        # Ensure timezone awareness
        df_copy = df.copy()
        time_series = pd.to_datetime(df_copy[time_col])

        if time_series.dt.tz is None:
            time_series = time_series.dt.tz_localize(COT_TZ)
        else:
            time_series = time_series.dt.tz_convert(COT_TZ)

        # Create mask for market hours
        mask = time_series.apply(self.is_market_open)

        return df_copy[mask].reset_index(drop=True)

@dataclass
class ValidationResult:
    status: ValidationSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime

class OptimizedL0Validator:
    """Optimized L0 validator with comprehensive data quality checks"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.api_manager = APIKeyManager()
        self.market_validator = MarketHoursValidator()
        self.session = None

    async def initialize(self):
        """Initialize HTTP session for API calls"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def validate_historical_data(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Validate historical data for date range 2020-2025

        Returns comprehensive validation report including:
        - Data completeness for each trading day
        - Gap analysis
        - Data quality metrics
        - Missing data identification
        """
        logger.info(f"🔍 Starting historical validation: {start_date} to {end_date}")

        validation_report = {
            'validation_start': datetime.now(COT_TZ).isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'trading_days': [],
            'missing_days': [],
            'data_quality': {},
            'api_usage': {},
            'overall_status': ValidationSeverity.PASS.value
        }

        # Get trading days in range
        trading_days = self.market_validator.holiday_calendar.get_trading_days(start_date, end_date)
        total_trading_days = len(trading_days)

        logger.info(f"📅 Found {total_trading_days} trading days to validate")

        # Check existing data in database
        existing_data_summary = await self._check_existing_data(start_date, end_date)
        validation_report['existing_data'] = existing_data_summary

        # Identify missing data
        missing_data = await self._identify_missing_data(trading_days)
        validation_report['missing_days'] = [d.isoformat() for d in missing_data]

        # Fetch missing data if any
        if missing_data:
            logger.info(f"📥 Fetching {len(missing_data)} missing trading days")
            fetch_results = await self._fetch_missing_data(missing_data)
            validation_report['fetch_results'] = fetch_results

        # Validate data quality for each trading day
        quality_results = await self._validate_daily_data_quality(trading_days)
        validation_report['daily_quality'] = quality_results

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(quality_results)
        validation_report['data_quality'] = overall_metrics

        # Get API usage stats
        validation_report['api_usage'] = self.api_manager.get_usage_stats()

        # Determine overall status
        if overall_metrics['completeness_percentage'] < MIN_COMPLETENESS:
            validation_report['overall_status'] = ValidationSeverity.FAIL.value
        elif overall_metrics['completeness_percentage'] < 98:
            validation_report['overall_status'] = ValidationSeverity.WARNING.value

        validation_report['validation_end'] = datetime.now(COT_TZ).isoformat()

        # Store validation report
        await self._store_validation_report(validation_report)

        logger.info(f"✅ Validation complete: {overall_metrics['completeness_percentage']:.1f}% complete")
        return validation_report

    async def _check_existing_data(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Check what data already exists in database"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_records,
                        COUNT(DISTINCT DATE(datetime AT TIME ZONE 'America/Bogota')) as unique_days,
                        MIN(datetime) as earliest_record,
                        MAX(datetime) as latest_record
                    FROM market_data
                    WHERE symbol = 'USDCOP'
                    AND DATE(datetime AT TIME ZONE 'America/Bogota') BETWEEN $1 AND $2
                    AND source != 'websocket'
                """, start_date, end_date)

                return {
                    'total_records': result['total_records'],
                    'unique_days': result['unique_days'],
                    'earliest_record': result['earliest_record'].isoformat() if result['earliest_record'] else None,
                    'latest_record': result['latest_record'].isoformat() if result['latest_record'] else None
                }

        except Exception as e:
            logger.error(f"Error checking existing data: {e}")
            return {'error': str(e)}

    async def _identify_missing_data(self, trading_days: List[date]) -> List[date]:
        """Identify trading days with missing or incomplete data"""
        missing_days = []

        try:
            async with self.db_pool.acquire() as conn:
                for trading_day in trading_days:
                    # Count records for this day
                    count = await conn.fetchval("""
                        SELECT COUNT(*)
                        FROM market_data
                        WHERE symbol = 'USDCOP'
                        AND DATE(datetime AT TIME ZONE 'America/Bogota') = $1
                        AND source != 'websocket'
                    """, trading_day)

                    # Check if we have sufficient data (at least 80% of expected bars)
                    expected_bars = EXPECTED_BARS_PER_DAY
                    if count < (expected_bars * 0.8):
                        missing_days.append(trading_day)
                        logger.debug(f"Missing data for {trading_day}: {count}/{expected_bars} bars")

        except Exception as e:
            logger.error(f"Error identifying missing data: {e}")
            # If database check fails, assume all days need fetching
            missing_days = trading_days

        return missing_days

    async def _fetch_missing_data(self, missing_days: List[date]) -> Dict[str, Any]:
        """Fetch missing data using API key rotation"""
        fetch_results = {
            'total_days': len(missing_days),
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_records_inserted': 0,
            'failed_days': []
        }

        for day in missing_days:
            try:
                # Get market session bounds for this day
                session_start, session_end = self.market_validator.get_market_session_bounds(day)

                # Fetch data for this day
                data = await self._fetch_day_data(session_start, session_end)

                if data and len(data) > 0:
                    # Insert into database
                    inserted_count = await self._insert_market_data_batch(data)
                    fetch_results['successful_fetches'] += 1
                    fetch_results['total_records_inserted'] += inserted_count
                    logger.info(f"✅ Fetched {len(data)} records for {day}")
                else:
                    fetch_results['failed_fetches'] += 1
                    fetch_results['failed_days'].append(day.isoformat())
                    logger.warning(f"❌ No data received for {day}")

                # Rate limiting between days
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error fetching data for {day}: {e}")
                fetch_results['failed_fetches'] += 1
                fetch_results['failed_days'].append(day.isoformat())

        return fetch_results

    async def _fetch_day_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Fetch data for a specific day using TwelveData API"""
        try:
            api_key, group_name = await self.api_manager.get_next_key()

            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': 'USD/COP',
                'interval': '5min',
                'apikey': api_key,
                'start_date': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'timezone': 'America/Bogota',
                'outputsize': 5000,
                'format': 'JSON'
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if 'values' in data and data['values']:
                        records = []
                        for item in data['values']:
                            record = {
                                'symbol': 'USDCOP',
                                'datetime': pd.to_datetime(item['datetime']).tz_localize(COT_TZ),
                                'open': float(item['open']),
                                'high': float(item['high']),
                                'low': float(item['low']),
                                'close': float(item['close']),
                                'volume': int(item.get('volume', 0)),
                                'source': 'twelvedata_historical',
                                'created_at': datetime.now(COT_TZ)
                            }
                            records.append(record)

                        return records

                elif response.status == 429:
                    logger.warning(f"Rate limit hit for {group_name}, continuing with next key")
                    return []
                else:
                    logger.error(f"API error {response.status}: {await response.text()}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching day data: {e}")
            return []

    async def _insert_market_data_batch(self, records: List[Dict[str, Any]]) -> int:
        """Insert market data records in optimized batches"""
        if not records:
            return 0

        try:
            async with self.db_pool.acquire() as conn:
                # Prepare data for batch insert
                values = []
                for record in records:
                    values.append((
                        record['symbol'],
                        record['datetime'],
                        Decimal(str(record['open'])),
                        Decimal(str(record['high'])),
                        Decimal(str(record['low'])),
                        Decimal(str(record['close'])),
                        record['volume'],
                        record['source'],
                        record['created_at']
                    ))

                # Batch insert with conflict resolution
                insert_query = """
                    INSERT INTO market_data
                    (symbol, datetime, open, high, low, close, volume, source, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (symbol, datetime, source)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        updated_at = NOW()
                """

                await conn.executemany(insert_query, values)

                logger.debug(f"Inserted {len(values)} records into market_data")
                return len(values)

        except Exception as e:
            logger.error(f"Error inserting market data batch: {e}")
            return 0

    async def _validate_daily_data_quality(self, trading_days: List[date]) -> Dict[str, Any]:
        """Validate data quality for each trading day"""
        daily_results = {}

        for day in trading_days:
            try:
                async with self.db_pool.acquire() as conn:
                    # Get data for this day
                    records = await conn.fetch("""
                        SELECT datetime, open, high, low, close, volume
                        FROM market_data
                        WHERE symbol = 'USDCOP'
                        AND DATE(datetime AT TIME ZONE 'America/Bogota') = $1
                        AND source != 'websocket'
                        ORDER BY datetime
                    """, day)

                    if records:
                        df = pd.DataFrame(records)
                        df['datetime'] = pd.to_datetime(df['datetime'])

                        # Validate this day's data
                        day_validation = self._validate_day_quality(df, day)
                        daily_results[day.isoformat()] = day_validation
                    else:
                        daily_results[day.isoformat()] = {
                            'status': ValidationSeverity.FAIL.value,
                            'records_count': 0,
                            'expected_records': EXPECTED_BARS_PER_DAY,
                            'completeness': 0.0,
                            'gaps': [],
                            'issues': ['No data found']
                        }

            except Exception as e:
                logger.error(f"Error validating day {day}: {e}")
                daily_results[day.isoformat()] = {
                    'status': ValidationSeverity.CRITICAL.value,
                    'error': str(e)
                }

        return daily_results

    def _validate_day_quality(self, df: pd.DataFrame, day: date) -> Dict[str, Any]:
        """Validate data quality for a single day"""
        validation = {
            'records_count': len(df),
            'expected_records': EXPECTED_BARS_PER_DAY,
            'completeness': 0.0,
            'gaps': [],
            'issues': [],
            'status': ValidationSeverity.PASS.value
        }

        if len(df) == 0:
            validation['status'] = ValidationSeverity.FAIL.value
            validation['issues'].append('No data available')
            return validation

        # Calculate completeness
        completeness = (len(df) / EXPECTED_BARS_PER_DAY) * 100
        validation['completeness'] = completeness

        # Check for gaps in 5-minute intervals
        gaps = self._detect_gaps(df)
        validation['gaps'] = [
            {
                'start': gap['start'].isoformat(),
                'end': gap['end'].isoformat(),
                'duration_minutes': gap['duration_minutes']
            }
            for gap in gaps
        ]

        # Determine status based on quality
        if completeness < 80:
            validation['status'] = ValidationSeverity.FAIL.value
            validation['issues'].append(f'Low completeness: {completeness:.1f}%')
        elif completeness < 95:
            validation['status'] = ValidationSeverity.WARNING.value
            validation['issues'].append(f'Moderate completeness: {completeness:.1f}%')

        if len(gaps) > 3:
            validation['status'] = ValidationSeverity.WARNING.value
            validation['issues'].append(f'Multiple gaps detected: {len(gaps)}')

        return validation

    def _detect_gaps(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect gaps in 5-minute time series data"""
        if len(df) < 2:
            return []

        gaps = []
        df_sorted = df.sort_values('datetime')

        for i in range(1, len(df_sorted)):
            current_time = df_sorted.iloc[i]['datetime']
            previous_time = df_sorted.iloc[i-1]['datetime']

            # Calculate time difference
            time_diff = current_time - previous_time
            expected_diff = timedelta(minutes=5)

            # Check if gap is larger than expected
            if time_diff > expected_diff * 1.5:  # Allow some tolerance
                gaps.append({
                    'start': previous_time,
                    'end': current_time,
                    'duration_minutes': time_diff.total_seconds() / 60
                })

        return gaps

    def _calculate_overall_metrics(self, daily_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall validation metrics"""
        total_days = len(daily_results)
        if total_days == 0:
            return {'completeness_percentage': 0.0, 'total_days': 0}

        total_records = 0
        total_expected = 0
        pass_count = 0
        warning_count = 0
        fail_count = 0

        for day_result in daily_results.values():
            if 'records_count' in day_result:
                total_records += day_result['records_count']
                total_expected += day_result['expected_records']

                status = day_result.get('status', ValidationSeverity.FAIL.value)
                if status == ValidationSeverity.PASS.value:
                    pass_count += 1
                elif status == ValidationSeverity.WARNING.value:
                    warning_count += 1
                else:
                    fail_count += 1

        completeness_percentage = (total_records / total_expected * 100) if total_expected > 0 else 0

        return {
            'total_days': total_days,
            'total_records': total_records,
            'total_expected': total_expected,
            'completeness_percentage': completeness_percentage,
            'days_passed': pass_count,
            'days_warning': warning_count,
            'days_failed': fail_count,
            'overall_quality_score': (pass_count + warning_count * 0.5) / total_days * 100 if total_days > 0 else 0
        }

    async def _store_validation_report(self, report: Dict[str, Any]):
        """Store validation report in database and cache"""
        try:
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO data_quality_metrics
                    (metric_name, metric_type, value, details, created_at)
                    VALUES ('l0_validation_report', 'comprehensive', $1, $2, $3)
                """,
                    report['data_quality']['completeness_percentage'],
                    json.dumps(report),
                    datetime.now(COT_TZ)
                )

            # Cache in Redis for dashboard
            cache_key = f"l0_validation:latest"
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(report, default=str)
            )

            logger.info("✅ Validation report stored successfully")

        except Exception as e:
            logger.error(f"Error storing validation report: {e}")

class RealtimeWebSocketManager:
    """Manages real-time WebSocket data collection during market hours"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis, api_manager: APIKeyManager):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.api_manager = api_manager
        self.market_validator = MarketHoursValidator()
        self.websocket_connection = None
        self.is_collecting = False

    async def start_realtime_collection(self):
        """Start real-time WebSocket data collection"""
        if not self.market_validator.is_market_open(datetime.now(COT_TZ)):
            logger.info("📅 Market is closed - not starting real-time collection")
            return False

        try:
            api_key, group_name = await self.api_manager.get_next_key()

            ws_url = "wss://ws.twelvedata.com/v1/quotes/price"

            logger.info(f"🔗 Connecting to WebSocket using {group_name}")

            async with websockets.connect(ws_url) as websocket:
                self.websocket_connection = websocket
                self.is_collecting = True

                # Subscribe to USD/COP
                subscribe_message = {
                    "action": "subscribe",
                    "params": {
                        "symbols": "USD/COP",
                        "apikey": api_key
                    }
                }

                await websocket.send(json.dumps(subscribe_message))
                logger.info("✅ Subscribed to USD/COP real-time data")

                # Cache connection status
                self.redis_client.setex(
                    "websocket:usdcop:status",
                    30,  # 30 second TTL
                    json.dumps({
                        'connected': True,
                        'group': group_name,
                        'timestamp': datetime.now(COT_TZ).isoformat()
                    })
                )

                # Listen for messages
                async for message in websocket:
                    if not self.market_validator.is_market_open(datetime.now(COT_TZ)):
                        logger.info("📈 Market closed - stopping real-time collection")
                        break

                    try:
                        data = json.loads(message)
                        await self._process_realtime_tick(data)
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_collecting = False

            # Update error status in cache
            self.redis_client.setex(
                "websocket:usdcop:status",
                300,  # 5 minute TTL for errors
                json.dumps({
                    'connected': False,
                    'error': str(e),
                    'timestamp': datetime.now(COT_TZ).isoformat()
                })
            )

    async def _process_realtime_tick(self, data: Dict[str, Any]):
        """Process incoming real-time tick data"""
        try:
            if not isinstance(data, dict) or 'price' not in data:
                return

            now = datetime.now(COT_TZ)

            # Create normalized tick record
            tick_record = {
                'symbol': 'USDCOP',
                'datetime': now,
                'price': Decimal(str(data['price'])),
                'bid': Decimal(str(data.get('bid', data['price']))),
                'ask': Decimal(str(data.get('ask', data['price']))),
                'volume': int(data.get('volume', 0)),
                'source': 'twelvedata_websocket',
                'created_at': now
            }

            # Store in realtime table
            await self._store_realtime_tick(tick_record)

            # Cache latest price
            await self._cache_latest_price(tick_record)

            # Aggregate to 5-minute bars every 5 minutes
            if now.minute % 5 == 0 and now.second < 10:
                await self._aggregate_to_5min_bar(now)

            logger.debug(f"💱 USDCOP: {tick_record['price']}")

        except Exception as e:
            logger.error(f"Error processing real-time tick: {e}")

    async def _store_realtime_tick(self, tick_record: Dict[str, Any]):
        """Store real-time tick in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO realtime_market_data
                    (symbol, time, bid, ask, last, volume, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    tick_record['symbol'],
                    tick_record['datetime'],
                    tick_record['bid'],
                    tick_record['ask'],
                    tick_record['price'],
                    tick_record['volume'],
                    tick_record['source']
                )

        except Exception as e:
            logger.error(f"Error storing real-time tick: {e}")

    async def _cache_latest_price(self, tick_record: Dict[str, Any]):
        """Cache latest price in Redis"""
        try:
            cache_data = {
                'symbol': tick_record['symbol'],
                'price': float(tick_record['price']),
                'bid': float(tick_record['bid']),
                'ask': float(tick_record['ask']),
                'spread': float(tick_record['ask'] - tick_record['bid']),
                'timestamp': tick_record['datetime'].isoformat(),
                'source': tick_record['source']
            }

            # Cache latest price
            self.redis_client.setex(
                "usdcop:latest_price",
                60,  # 1 minute TTL
                json.dumps(cache_data)
            )

            # Publish to subscribers
            self.redis_client.publish(
                "market_data:realtime:usdcop",
                json.dumps(cache_data)
            )

        except Exception as e:
            logger.error(f"Error caching latest price: {e}")

    async def _aggregate_to_5min_bar(self, current_time: datetime):
        """Aggregate real-time ticks to 5-minute OHLC bar"""
        try:
            # Calculate 5-minute bar boundaries
            bar_start = current_time.replace(minute=(current_time.minute // 5) * 5, second=0, microsecond=0)
            bar_end = bar_start + timedelta(minutes=5)

            async with self.db_pool.acquire() as conn:
                # Get all ticks in this 5-minute window
                result = await conn.fetchrow("""
                    SELECT
                        MIN(time) as bar_time,
                        (array_agg(last ORDER BY time))[1] as open,
                        MAX(last) as high,
                        MIN(last) as low,
                        (array_agg(last ORDER BY time DESC))[1] as close,
                        SUM(volume) as volume,
                        COUNT(*) as tick_count
                    FROM realtime_market_data
                    WHERE symbol = 'USDCOP'
                    AND time >= $1
                    AND time < $2
                    AND source = 'twelvedata_websocket'
                """, bar_start, bar_end)

                if result and result['tick_count'] > 0:
                    # Insert aggregated 5-minute bar
                    await conn.execute("""
                        INSERT INTO market_data
                        (symbol, datetime, open, high, low, close, volume, source, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (symbol, datetime, source)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            updated_at = NOW()
                    """,
                        'USDCOP',
                        bar_start,
                        result['open'],
                        result['high'],
                        result['low'],
                        result['close'],
                        result['volume'] or 0,
                        'realtime_aggregated',
                        current_time
                    )

                    logger.info(f"📊 Aggregated 5-min bar: {result['tick_count']} ticks → OHLC")

        except Exception as e:
            logger.error(f"Error aggregating to 5-minute bar: {e}")

# Main service function
async def run_optimized_l0_validator():
    """Run the optimized L0 validator service"""

    # Initialize connections
    db_pool = await asyncpg.create_pool(
        os.getenv('DATABASE_URL'),
        min_size=5,
        max_size=20,
        command_timeout=60
    )

    redis_client = redis.from_url(os.getenv('REDIS_URL'), decode_responses=True)

    # Initialize validator
    validator = OptimizedL0Validator(db_pool, redis_client)
    await validator.initialize()

    # Initialize WebSocket manager
    websocket_manager = RealtimeWebSocketManager(db_pool, redis_client, validator.api_manager)

    try:
        logger.info("🚀 Starting Optimized L0 Validator Service")

        # Define validation date range (2020-2025)
        start_date = date(2020, 1, 1)
        end_date = date(2025, 12, 31)

        # Run historical validation
        validation_report = await validator.validate_historical_data(start_date, end_date)

        logger.info(f"📋 Historical validation complete: {validation_report['data_quality']['completeness_percentage']:.1f}% complete")

        # Start real-time collection if market is open
        if validator.market_validator.is_market_open(datetime.now(COT_TZ)):
            logger.info("🔴 Market is open - starting real-time collection")
            asyncio.create_task(websocket_manager.start_realtime_collection())

        # Keep service running
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes

            # Check if we should start/stop real-time collection
            current_time = datetime.now(COT_TZ)
            market_open = validator.market_validator.is_market_open(current_time)

            if market_open and not websocket_manager.is_collecting:
                logger.info("🔴 Market opened - starting real-time collection")
                asyncio.create_task(websocket_manager.start_realtime_collection())
            elif not market_open and websocket_manager.is_collecting:
                logger.info("🔘 Market closed - stopping real-time collection")
                websocket_manager.is_collecting = False

    finally:
        await validator.close()
        await db_pool.close()
        redis_client.close()

if __name__ == "__main__":
    asyncio.run(run_optimized_l0_validator())