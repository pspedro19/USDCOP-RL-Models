"""
Backtest Data Validator - Fail Fast Pattern
Contract: CTR-VALID-001

Validates data completeness BEFORE backtest execution to fail fast
rather than discovering issues mid-backtest.
"""

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, text


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    BLOCKING = "blocking"  # Stops backtest
    WARNING = "warning"    # Logs warning, continues
    INFO = "info"          # Informational only


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation issue found."""
    severity: ValidationSeverity
    category: str
    message: str
    affected_dates: Optional[Tuple[dt.date, dt.date]] = None
    recommendation: Optional[str] = None


@dataclass(frozen=True)
class BacktestDataValidationResult:
    """Complete validation result - immutable for safety."""
    is_valid: bool
    validation_time: dt.datetime
    date_range: Tuple[dt.date, dt.date]

    # Data counts
    ohlcv_rows: int
    macro_rows: int
    feature_rows: int

    # Coverage metrics
    ohlcv_coverage_pct: float
    macro_coverage_pct: float

    # Issues found
    issues: Tuple[ValidationIssue, ...]

    @property
    def blocking_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.BLOCKING]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


class BacktestDataValidator:
    """
    Validates data completeness before backtest execution.

    Follows:
    - Single Responsibility: Only validates, doesn't fix
    - Fail Fast: Raises early on blocking issues
    - Explicit over Implicit: Clear validation rules
    """

    # Critical macro indicators that MUST be present
    CRITICAL_INDICATORS = (
        "fxrt_index_dxy_usa_d_dxy",
        "volt_vix_usa_d_vix",
        "finc_bond_yield10y_usa_d_ust10y",
        "finc_bond_yield2y_usa_d_dgs2",
    )

    # Minimum coverage thresholds
    MIN_OHLCV_COVERAGE = 0.95  # 95% of trading days
    MIN_MACRO_COVERAGE = 0.60  # 60% (weekdays only, holidays excluded)

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def validate(
        self,
        start_date: dt.date,
        end_date: dt.date,
        fail_fast: bool = True
    ) -> BacktestDataValidationResult:
        """
        Validate data for backtest date range.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            fail_fast: If True, raises exception on blocking issues

        Returns:
            BacktestDataValidationResult with all findings

        Raises:
            DataValidationError: If fail_fast=True and blocking issues found
        """
        issues = []

        with self.engine.connect() as conn:
            # 1. Validate OHLCV completeness
            ohlcv_result = self._validate_ohlcv(conn, start_date, end_date)
            issues.extend(ohlcv_result["issues"])

            # 2. Validate macro indicators
            macro_result = self._validate_macro(conn, start_date, end_date)
            issues.extend(macro_result["issues"])

            # 3. Validate critical indicators
            critical_result = self._validate_critical_indicators(conn, start_date, end_date)
            issues.extend(critical_result["issues"])

            # 4. Check for look-ahead bias indicators
            bias_result = self._check_lookahead_bias(conn, start_date, end_date)
            issues.extend(bias_result["issues"])

        result = BacktestDataValidationResult(
            is_valid=len([i for i in issues if i.severity == ValidationSeverity.BLOCKING]) == 0,
            validation_time=dt.datetime.utcnow(),
            date_range=(start_date, end_date),
            ohlcv_rows=ohlcv_result["count"],
            macro_rows=macro_result["count"],
            feature_rows=0,  # Will be populated after feature build
            ohlcv_coverage_pct=ohlcv_result["coverage"],
            macro_coverage_pct=macro_result["coverage"],
            issues=tuple(issues)
        )

        if fail_fast and not result.is_valid:
            raise DataValidationError(result)

        return result

    def _validate_ohlcv(self, conn, start_date: dt.date, end_date: dt.date) -> dict:
        """Validate OHLCV data completeness."""
        query = text("""
            SELECT COUNT(*) as cnt,
                   MIN(timestamp) as min_ts,
                   MAX(timestamp) as max_ts
            FROM ohlcv_5m_usdcop
            WHERE timestamp >= :start AND timestamp < :end
        """)

        row = conn.execute(query, {"start": start_date, "end": end_date}).fetchone()

        # Calculate expected rows (5-min bars, ~12 hours trading, 5 days/week)
        trading_days = self._count_trading_days(start_date, end_date)
        expected_rows = trading_days * 144  # 12 hours * 12 bars/hour

        actual = row.cnt if row else 0
        coverage = actual / expected_rows if expected_rows > 0 else 0

        issues = []
        if coverage < self.MIN_OHLCV_COVERAGE:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.BLOCKING,
                category="OHLCV",
                message=f"OHLCV coverage {coverage:.1%} below minimum {self.MIN_OHLCV_COVERAGE:.1%}",
                affected_dates=(start_date, end_date),
                recommendation="Run OHLCV extraction pipeline to fill gaps"
            ))

        return {"count": actual, "coverage": coverage, "issues": issues}

    def _validate_macro(self, conn, start_date: dt.date, end_date: dt.date) -> dict:
        """Validate macro indicators completeness."""
        query = text("""
            SELECT COUNT(*) as cnt
            FROM macro_indicators_daily
            WHERE fecha >= :start AND fecha <= :end
        """)

        row = conn.execute(query, {"start": start_date, "end": end_date}).fetchone()

        trading_days = self._count_trading_days(start_date, end_date)
        actual = row.cnt if row else 0
        coverage = actual / trading_days if trading_days > 0 else 0

        issues = []
        if coverage < self.MIN_MACRO_COVERAGE:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="MACRO",
                message=f"Macro coverage {coverage:.1%} below expected {self.MIN_MACRO_COVERAGE:.1%}",
                affected_dates=(start_date, end_date),
                recommendation="Check macro pipeline execution logs"
            ))

        return {"count": actual, "coverage": coverage, "issues": issues}

    def _validate_critical_indicators(self, conn, start_date: dt.date, end_date: dt.date) -> dict:
        """Validate critical indicators have data."""
        issues = []

        for indicator in self.CRITICAL_INDICATORS:
            query = text(f"""
                SELECT COUNT(*) as cnt,
                       MIN(fecha) as first_date,
                       MAX(fecha) as last_date
                FROM macro_indicators_daily
                WHERE fecha >= :start AND fecha <= :end
                  AND {indicator} IS NOT NULL
            """)

            row = conn.execute(query, {"start": start_date, "end": end_date}).fetchone()

            if row.cnt == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.BLOCKING,
                    category="CRITICAL_INDICATOR",
                    message=f"Critical indicator {indicator} has NO data in range",
                    affected_dates=(start_date, end_date),
                    recommendation=f"Run macro extraction for {indicator}"
                ))

        return {"issues": issues}

    def _check_lookahead_bias(self, conn, start_date: dt.date, end_date: dt.date) -> dict:
        """Check for potential look-ahead bias in features."""
        issues = []

        # Check if any features have future timestamps
        query = text("""
            SELECT COUNT(*) as future_count
            FROM inference_features_5m
            WHERE timestamp > NOW()
        """)

        row = conn.execute(query).fetchone()
        if row and row.future_count > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.BLOCKING,
                category="LOOKAHEAD_BIAS",
                message=f"Found {row.future_count} features with future timestamps",
                recommendation="Remove features with timestamp > current time"
            ))

        return {"issues": issues}

    def _count_trading_days(self, start_date: dt.date, end_date: dt.date) -> int:
        """Count trading days (weekdays) in range."""
        count = 0
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                count += 1
            current += dt.timedelta(days=1)
        return count


class DataValidationError(Exception):
    """Raised when data validation fails with blocking issues."""

    def __init__(self, result: BacktestDataValidationResult):
        self.result = result
        issues_str = "\n".join(f"  - {i.message}" for i in result.blocking_issues)
        super().__init__(f"Data validation failed with {len(result.blocking_issues)} blocking issues:\n{issues_str}")
