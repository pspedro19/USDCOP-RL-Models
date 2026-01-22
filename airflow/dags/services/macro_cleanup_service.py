"""
Macro Cleanup Service
=====================

Handles calendar-based cleanup of non-trading days from macro data.
Sets NULL for specific columns based on their trading calendar (USA, Colombia, Global).

Contract: CTR-L0-CLEANUP-001

Operations:
    - cleanup: Apply calendar-specific cleanup to macro_indicators_daily
    - cleanup_by_calendar: Set NULL for columns based on their calendar
    - is_trading_day: Check if a date is a valid trading day for a calendar
    - get_non_trading_days: List non-trading days in a date range

Version: 2.0.0 (Refactored for P1-3/P1-5: Calendar-specific cleanup)
"""

from __future__ import annotations

import logging
import yaml
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Set

from utils.dag_common import get_db_connection

logger = logging.getLogger(__name__)

# Try to import Colombian holidays library
try:
    from colombian_holidays import is_holiday_date as is_colombian_holiday
    COLOMBIAN_HOLIDAYS_AVAILABLE = True
except ImportError:
    COLOMBIAN_HOLIDAYS_AVAILABLE = False
    def is_colombian_holiday(date_obj):
        return False


class MacroCleanupService:
    """
    Service for calendar-based cleanup of non-trading days from macro data.

    Applies different holiday calendars to different columns:
    - USA calendar: DXY, VIX, US Treasuries, WTI
    - Colombia calendar: COLCAP, IBR, Colombian bonds
    - Global calendar: Gold, Brent, EMBI, FX pairs (no holiday cleanup)
    - Monthly calendar: Only weekend cleanup for monthly indicators
    """

    # US Federal Holidays (fixed dates)
    US_FIXED_HOLIDAYS = {
        (1, 1): "New Year's Day",
        (7, 4): "Independence Day",
        (11, 11): "Veterans Day",
        (12, 25): "Christmas Day",
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize cleanup service.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or '/opt/airflow/config/l0_macro_sources.yaml'
        self._config = None
        self._calendar_mapping = None

    @property
    def config(self) -> Dict[str, Any]:
        """Load configuration lazily."""
        if self._config is None:
            try:
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                self._config = {}
        return self._config

    @property
    def calendar_mapping(self) -> Dict[str, str]:
        """
        Build column -> calendar mapping from config.

        Returns:
            Dictionary mapping column names to calendar type
        """
        if self._calendar_mapping is None:
            self._calendar_mapping = {}
            indicator_calendars = self.config.get('indicator_calendars', {})
            for calendar_type, columns in indicator_calendars.items():
                for column in columns:
                    self._calendar_mapping[column] = calendar_type
        return self._calendar_mapping

    def get_columns_by_calendar(self, calendar_type: str) -> List[str]:
        """Get all columns for a specific calendar type."""
        return [col for col, cal in self.calendar_mapping.items() if cal == calendar_type]

    def cleanup(self, **context) -> Dict[str, Any]:
        """
        Apply calendar-specific cleanup to macro_indicators_daily.

        Instead of deleting entire rows, sets NULL for specific columns
        based on their trading calendar:
        - USA columns: NULL on US holidays + weekends
        - Colombia columns: NULL on Colombian holidays + weekends
        - Global columns: NULL only on weekends
        - Monthly columns: No cleanup (monthly data)

        Args:
            **context: Airflow context

        Returns:
            Dictionary with cleanup statistics
        """
        conn = get_db_connection()
        cur = conn.cursor()

        cleanup_stats = {
            'usa_columns_cleaned': 0,
            'colombia_columns_cleaned': 0,
            'weekend_rows_cleaned': 0,
            'us_holiday_updates': 0,
            'col_holiday_updates': 0,
        }

        try:
            # Get all dates in the table
            cur.execute("""
                SELECT DISTINCT fecha
                FROM macro_indicators_daily
                WHERE fecha >= '2020-01-01'
                ORDER BY fecha DESC
            """)
            all_dates = [row[0] for row in cur.fetchall()]

            logger.info(f"[CLEANUP] Processing {len(all_dates)} dates")

            # Get columns by calendar
            usa_columns = self.get_columns_by_calendar('usa')
            col_columns = self.get_columns_by_calendar('colombia')
            global_columns = self.get_columns_by_calendar('global')
            monthly_columns = self.get_columns_by_calendar('monthly')

            logger.info(f"[CLEANUP] Calendar mapping:")
            logger.info(f"  - USA columns: {len(usa_columns)}")
            logger.info(f"  - Colombia columns: {len(col_columns)}")
            logger.info(f"  - Global columns: {len(global_columns)}")
            logger.info(f"  - Monthly columns: {len(monthly_columns)}")

            for fecha in all_dates:
                is_wknd = self.is_weekend(fecha)
                is_us_hol = self.is_us_holiday(fecha)
                is_col_hol = COLOMBIAN_HOLIDAYS_AVAILABLE and is_colombian_holiday(fecha)

                # USA columns: NULL on weekends and US holidays
                if usa_columns and (is_wknd or is_us_hol):
                    for col in usa_columns:
                        cur.execute(f"""
                            UPDATE macro_indicators_daily
                            SET {col} = NULL
                            WHERE fecha = %s AND {col} IS NOT NULL
                        """, [fecha])
                        if cur.rowcount > 0:
                            cleanup_stats['usa_columns_cleaned'] += cur.rowcount
                            if is_us_hol and not is_wknd:
                                cleanup_stats['us_holiday_updates'] += 1

                # Colombia columns: NULL on weekends and Colombian holidays
                if col_columns and (is_wknd or is_col_hol):
                    for col in col_columns:
                        cur.execute(f"""
                            UPDATE macro_indicators_daily
                            SET {col} = NULL
                            WHERE fecha = %s AND {col} IS NOT NULL
                        """, [fecha])
                        if cur.rowcount > 0:
                            cleanup_stats['colombia_columns_cleaned'] += cur.rowcount
                            if is_col_hol and not is_wknd:
                                cleanup_stats['col_holiday_updates'] += 1

                # Global columns: NULL only on weekends (they trade 24/7 otherwise)
                # NOTE: Most global commodities trade even on weekends via crypto/OTC
                # So we keep them as-is unless specifically configured otherwise

                if is_wknd:
                    cleanup_stats['weekend_rows_cleaned'] += 1

            conn.commit()

            logger.info("[CLEANUP] Calendar-based cleanup complete:")
            logger.info(f"  - USA columns cleaned: {cleanup_stats['usa_columns_cleaned']}")
            logger.info(f"  - Colombia columns cleaned: {cleanup_stats['colombia_columns_cleaned']}")
            logger.info(f"  - Weekend rows processed: {cleanup_stats['weekend_rows_cleaned']}")
            logger.info(f"  - US holiday updates: {cleanup_stats['us_holiday_updates']}")
            logger.info(f"  - COL holiday updates: {cleanup_stats['col_holiday_updates']}")

        except Exception as e:
            conn.rollback()
            logger.error(f"[CLEANUP] Error: {e}")
            raise
        finally:
            cur.close()
            conn.close()

        # Push to XCom
        ti = context.get('ti')
        if ti:
            ti.xcom_push(key='cleanup_result', value=cleanup_stats)

        return {
            'status': 'success',
            'stats': cleanup_stats
        }

    def is_weekend(self, date_obj) -> bool:
        """
        Check if a date is a weekend (Saturday=5, Sunday=6).

        Args:
            date_obj: Date to check

        Returns:
            True if weekend
        """
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        return date_obj.weekday() >= 5

    def is_us_holiday(self, date_obj) -> bool:
        """
        Check if a date is a US federal holiday.

        Args:
            date_obj: Date to check

        Returns:
            True if US holiday
        """
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()

        # Check fixed holidays
        if (date_obj.month, date_obj.day) in self.US_FIXED_HOLIDAYS:
            return True

        # Check floating holidays
        floating = self._get_us_floating_holidays(date_obj.year)
        return date_obj in floating

    def _get_us_floating_holidays(self, year: int) -> Set[date]:
        """
        Calculate US floating holidays for a given year.

        Args:
            year: Year to calculate for

        Returns:
            Set of holiday dates
        """
        holidays = set()

        # MLK Day: 3rd Monday of January
        jan_first = date(year, 1, 1)
        days_to_monday = (7 - jan_first.weekday()) % 7
        if jan_first.weekday() == 0:
            days_to_monday = 0
        mlk_day = jan_first + timedelta(days=days_to_monday + 14)
        holidays.add(mlk_day)

        # Presidents Day: 3rd Monday of February
        feb_first = date(year, 2, 1)
        days_to_monday = (7 - feb_first.weekday()) % 7
        if feb_first.weekday() == 0:
            days_to_monday = 0
        presidents_day = feb_first + timedelta(days=days_to_monday + 14)
        holidays.add(presidents_day)

        # Memorial Day: Last Monday of May
        may_31 = date(year, 5, 31)
        days_back = may_31.weekday()
        memorial_day = may_31 - timedelta(days=days_back)
        holidays.add(memorial_day)

        # Labor Day: 1st Monday of September
        sep_first = date(year, 9, 1)
        days_to_monday = (7 - sep_first.weekday()) % 7
        if sep_first.weekday() == 0:
            days_to_monday = 0
        labor_day = sep_first + timedelta(days=days_to_monday)
        holidays.add(labor_day)

        # Columbus Day: 2nd Monday of October
        oct_first = date(year, 10, 1)
        days_to_monday = (7 - oct_first.weekday()) % 7
        if oct_first.weekday() == 0:
            days_to_monday = 0
        columbus_day = oct_first + timedelta(days=days_to_monday + 7)
        holidays.add(columbus_day)

        # Thanksgiving: 4th Thursday of November
        nov_first = date(year, 11, 1)
        days_to_thursday = (3 - nov_first.weekday()) % 7
        thanksgiving = nov_first + timedelta(days=days_to_thursday + 21)
        holidays.add(thanksgiving)

        return holidays

    def is_trading_day(self, date_obj) -> bool:
        """
        Check if a date is a valid trading day.

        Args:
            date_obj: Date to check

        Returns:
            True if trading day
        """
        if self.is_weekend(date_obj):
            return False
        if self.is_us_holiday(date_obj):
            return False
        if COLOMBIAN_HOLIDAYS_AVAILABLE and is_colombian_holiday(date_obj):
            return False
        return True

    def get_non_trading_days(
        self,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """
        Get list of non-trading days in a date range.

        Args:
            start_date: Range start
            end_date: Range end

        Returns:
            List of non-trading day details
        """
        non_trading = []
        current = start_date

        while current <= end_date:
            if not self.is_trading_day(current):
                reason = "unknown"
                if self.is_weekend(current):
                    reason = "weekend"
                elif self.is_us_holiday(current):
                    reason = "us_holiday"
                elif COLOMBIAN_HOLIDAYS_AVAILABLE and is_colombian_holiday(current):
                    reason = "col_holiday"

                non_trading.append({
                    'date': current.strftime('%Y-%m-%d'),
                    'reason': reason,
                })

            current += timedelta(days=1)

        return non_trading


# =============================================================================
# Airflow Task Functions
# =============================================================================

def cleanup_non_trading_days(**context) -> Dict[str, Any]:
    """Airflow task for cleanup."""
    service = MacroCleanupService()
    return service.cleanup(**context)
