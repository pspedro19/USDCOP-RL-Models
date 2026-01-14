"""
Trading Hours Check
===================

Validates trades are within configured trading hours.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

from datetime import datetime, time
from typing import Optional, Set
import pytz

from src.core.interfaces.risk import (
    IRiskCheck,
    RiskContext,
    RiskCheckResult,
    RiskStatus,
    ITradingHoursChecker,
)


class TradingHoursCheck(IRiskCheck, ITradingHoursChecker):
    """
    Check if current time is within trading hours.

    Configurable parameters:
    - start_time: Market open time
    - end_time: Market close time
    - trading_days: Set of weekdays (0=Monday, 6=Sunday)
    - timezone: Timezone for time calculations
    """

    def __init__(
        self,
        start_time: time = time(8, 0),
        end_time: time = time(16, 0),
        trading_days: Optional[Set[int]] = None,
        timezone_str: str = "America/Bogota",
    ):
        self._start_time = start_time
        self._end_time = end_time
        self._trading_days = trading_days or {0, 1, 2, 3, 4}  # Mon-Fri
        self._timezone_str = timezone_str
        self._tz = pytz.timezone(timezone_str)

    @property
    def name(self) -> str:
        return "trading_hours"

    @property
    def order(self) -> int:
        return 10  # Early check - don't process if outside hours

    @property
    def timezone(self) -> str:
        return self._timezone_str

    def is_trading_hours(self) -> tuple[bool, str]:
        """Check if current time is within trading hours."""
        now = datetime.now(self._tz)
        current_time = now.time()
        current_weekday = now.weekday()

        if current_weekday not in self._trading_days:
            return False, f"Not a trading day (weekday={current_weekday})"

        if current_time < self._start_time:
            return False, f"Before market open ({self._start_time})"

        if current_time > self._end_time:
            return False, f"After market close ({self._end_time})"

        return True, "Within trading hours"

    def check(self, context: RiskContext) -> RiskCheckResult:
        """Check trading hours."""
        # Allow skipping this check
        if not context.enforce_trading_hours:
            return RiskCheckResult(
                approved=True,
                status=RiskStatus.APPROVED,
                message="Trading hours check skipped",
            )

        is_within, message = self.is_trading_hours()

        if not is_within:
            return RiskCheckResult(
                approved=False,
                status=RiskStatus.OUTSIDE_TRADING_HOURS,
                message=message,
            )

        return RiskCheckResult(
            approved=True,
            status=RiskStatus.APPROVED,
            message=message,
        )
