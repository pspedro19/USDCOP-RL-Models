"""
Unit Tests: Backtest Data Validator
Contract: CTR-TEST-VALID-001

Tests validation logic following AAA pattern (Arrange, Act, Assert).
"""

import datetime as dt
import pytest
from unittest.mock import MagicMock, patch

from src.validation.backtest_data_validator import (
    BacktestDataValidator,
    BacktestDataValidationResult,
    DataValidationError,
    ValidationIssue,
    ValidationSeverity,
)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_blocking_issue_creation(self):
        """Test creating a blocking validation issue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.BLOCKING,
            category="OHLCV",
            message="Missing data",
            recommendation="Run extraction"
        )

        assert issue.severity == ValidationSeverity.BLOCKING
        assert issue.category == "OHLCV"
        assert "Missing" in issue.message

    def test_issue_is_immutable(self):
        """Test that ValidationIssue is immutable (frozen dataclass)."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category="MACRO",
            message="Low coverage"
        )

        with pytest.raises(AttributeError):
            issue.message = "Changed"


class TestBacktestDataValidationResult:
    """Tests for validation result."""

    def test_blocking_issues_filter(self):
        """Test filtering blocking issues from result."""
        result = BacktestDataValidationResult(
            is_valid=False,
            validation_time=dt.datetime.utcnow(),
            date_range=(dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
            ohlcv_rows=1000,
            macro_rows=250,
            feature_rows=0,
            ohlcv_coverage_pct=0.95,
            macro_coverage_pct=0.65,
            issues=(
                ValidationIssue(ValidationSeverity.BLOCKING, "A", "msg1"),
                ValidationIssue(ValidationSeverity.WARNING, "B", "msg2"),
                ValidationIssue(ValidationSeverity.BLOCKING, "C", "msg3"),
            )
        )

        assert len(result.blocking_issues) == 2
        assert len(result.warnings) == 1

    def test_result_is_immutable(self):
        """Test that result is immutable."""
        result = BacktestDataValidationResult(
            is_valid=True,
            validation_time=dt.datetime.utcnow(),
            date_range=(dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
            ohlcv_rows=1000,
            macro_rows=250,
            feature_rows=0,
            ohlcv_coverage_pct=0.95,
            macro_coverage_pct=0.65,
            issues=()
        )

        with pytest.raises(AttributeError):
            result.is_valid = False


class TestBacktestDataValidator:
    """Tests for validator logic."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock SQLAlchemy engine."""
        engine = MagicMock()
        return engine

    def test_count_trading_days_weekdays_only(self):
        """Test trading day count excludes weekends."""
        validator = BacktestDataValidator.__new__(BacktestDataValidator)

        # Monday Jan 6 to Friday Jan 10 = 5 trading days
        count = validator._count_trading_days(
            dt.date(2025, 1, 6),
            dt.date(2025, 1, 10)
        )
        assert count == 5

        # Include weekend: Jan 6 (Mon) to Jan 12 (Sun) = 5 trading days
        count = validator._count_trading_days(
            dt.date(2025, 1, 6),
            dt.date(2025, 1, 12)
        )
        assert count == 5

    def test_critical_indicators_constant(self):
        """Test critical indicators are defined."""
        assert "fxrt_index_dxy_usa_d_dxy" in BacktestDataValidator.CRITICAL_INDICATORS
        assert "volt_vix_usa_d_vix" in BacktestDataValidator.CRITICAL_INDICATORS
        assert len(BacktestDataValidator.CRITICAL_INDICATORS) >= 4


class TestDataValidationError:
    """Tests for validation error exception."""

    def test_error_contains_issues(self):
        """Test error message includes blocking issues."""
        result = BacktestDataValidationResult(
            is_valid=False,
            validation_time=dt.datetime.utcnow(),
            date_range=(dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
            ohlcv_rows=0,
            macro_rows=0,
            feature_rows=0,
            ohlcv_coverage_pct=0.0,
            macro_coverage_pct=0.0,
            issues=(
                ValidationIssue(ValidationSeverity.BLOCKING, "OHLCV", "No data found"),
            )
        )

        error = DataValidationError(result)

        assert "No data found" in str(error)
        assert error.result == result


class TestCleanCodePrinciples:
    """Tests verifying clean code principles are followed."""

    def test_validator_has_clear_constants(self):
        """Test validator exposes clear, documented constants."""
        assert hasattr(BacktestDataValidator, "CRITICAL_INDICATORS")
        assert hasattr(BacktestDataValidator, "MIN_OHLCV_COVERAGE")
        assert hasattr(BacktestDataValidator, "MIN_MACRO_COVERAGE")

        # Constants should be reasonable values
        assert 0.0 < BacktestDataValidator.MIN_OHLCV_COVERAGE <= 1.0
        assert 0.0 < BacktestDataValidator.MIN_MACRO_COVERAGE <= 1.0

    def test_validation_result_uses_tuple_for_immutability(self):
        """Test that issues use tuple (immutable) not list."""
        result = BacktestDataValidationResult(
            is_valid=True,
            validation_time=dt.datetime.utcnow(),
            date_range=(dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
            ohlcv_rows=1000,
            macro_rows=250,
            feature_rows=0,
            ohlcv_coverage_pct=0.95,
            macro_coverage_pct=0.65,
            issues=()
        )

        assert isinstance(result.issues, tuple)
        assert isinstance(result.date_range, tuple)
