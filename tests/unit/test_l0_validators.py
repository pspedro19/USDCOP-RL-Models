# -*- coding: utf-8 -*-
"""
Unit Tests for L0 Data Validators
=================================

Tests for:
- SchemaValidator
- RangeValidator
- CompletenessValidator
- LeakageValidator
- FreshnessValidator
- ValidationPipeline

Contract: CTR-L0-VALIDATOR-001
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
DAGS_PATH = PROJECT_ROOT / 'airflow' / 'dags'

for path in [str(DAGS_PATH), str(PROJECT_ROOT / 'src')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from validators.data_validators import (
    SchemaValidator,
    RangeValidator,
    CompletenessValidator,
    LeakageValidator,
    FreshnessValidator,
    ValidationPipeline,
    ValidationResult,
    ValidationReport,
    ValidationSeverity,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_daily_df():
    """Create a valid daily DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'fecha': dates,
        'comm_oil_brent_glb_d_brent': [80.0 + i * 0.1 for i in range(100)],
        'volt_vix_usa_d_vix': [15.0 + i * 0.05 for i in range(100)],
    })


@pytest.fixture
def valid_monthly_df():
    """Create a valid monthly DataFrame for testing."""
    dates = pd.date_range('2022-01-01', periods=24, freq='MS')
    return pd.DataFrame({
        'fecha': dates,
        'polr_fed_funds_usa_m_fedfunds': [4.5 + i * 0.1 for i in range(24)],
    })


@pytest.fixture
def df_with_nulls():
    """Create a DataFrame with NULL values."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    values = [80.0 + i * 0.1 for i in range(100)]
    # Add some nulls
    values[10] = None
    values[20] = None
    values[30] = None
    return pd.DataFrame({
        'fecha': dates,
        'test_var': values,
    })


@pytest.fixture
def df_with_future_dates():
    """Create a DataFrame with future dates (leakage)."""
    today = datetime.now()
    dates = pd.date_range(today - timedelta(days=10), periods=20, freq='D')
    return pd.DataFrame({
        'fecha': dates,
        'test_var': [100.0] * 20,
    })


@pytest.fixture
def df_out_of_range():
    """Create a DataFrame with out-of-range values."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    values = [80.0] * 100
    # Add some out-of-range values (Brent range is 30-150)
    values[5] = 10.0   # Below range
    values[10] = 200.0  # Above range
    values[15] = 5.0    # Below range
    return pd.DataFrame({
        'fecha': dates,
        'comm_oil_brent_glb_d_brent': values,
    })


# =============================================================================
# SCHEMA VALIDATOR TESTS
# =============================================================================

class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_valid_schema_passes(self, valid_daily_df):
        """Test that valid schema passes validation."""
        validator = SchemaValidator()
        result = validator.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert result.passed is True
        assert len(result.errors) == 0
        assert result.validator == 'SchemaValidator'

    def test_empty_dataframe_fails(self):
        """Test that empty DataFrame fails validation."""
        validator = SchemaValidator()
        result = validator.validate(pd.DataFrame(), 'test_var')

        assert result.passed is False
        assert 'empty' in result.errors[0].lower()
        assert result.severity == ValidationSeverity.CRITICAL

    def test_none_dataframe_fails(self):
        """Test that None DataFrame fails validation."""
        validator = SchemaValidator()
        result = validator.validate(None, 'test_var')

        assert result.passed is False
        assert result.severity == ValidationSeverity.CRITICAL

    def test_missing_fecha_column_fails(self):
        """Test that missing fecha column fails validation."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'test_var': range(10),
        })
        validator = SchemaValidator()
        result = validator.validate(df, 'test_var')

        assert result.passed is False
        assert any("'fecha'" in e for e in result.errors)

    def test_missing_variable_column_fails(self):
        """Test that missing variable column fails validation."""
        df = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=10),
            'other_var': range(10),
        })
        validator = SchemaValidator()
        result = validator.validate(df, 'test_var')

        assert result.passed is False
        assert any("'test_var'" in e for e in result.errors)

    def test_non_numeric_variable_fails(self):
        """Test that non-numeric variable column fails validation."""
        df = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=10),
            'test_var': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        })
        validator = SchemaValidator()
        result = validator.validate(df, 'test_var')

        assert result.passed is False
        assert any('numeric' in e.lower() for e in result.errors)

    def test_nulls_generate_warning(self, df_with_nulls):
        """Test that NULL values generate warnings."""
        validator = SchemaValidator()
        result = validator.validate(df_with_nulls, 'test_var')

        # Should pass but with warning (3% nulls is < 50%)
        assert result.passed is True
        assert len(result.warnings) > 0
        assert any('NULL' in w for w in result.warnings)


# =============================================================================
# RANGE VALIDATOR TESTS
# =============================================================================

class TestRangeValidator:
    """Tests for RangeValidator."""

    def test_values_in_range_pass(self, valid_daily_df):
        """Test that values within range pass validation."""
        # Set custom range for Brent: 30-150
        validator = RangeValidator(ranges={
            'comm_oil_brent_glb_d_brent': (30.0, 150.0)
        })
        result = validator.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert result.passed is True
        assert len(result.errors) == 0

    def test_values_out_of_range_fail(self, df_out_of_range):
        """Test that out-of-range values fail validation."""
        validator = RangeValidator(ranges={
            'comm_oil_brent_glb_d_brent': (30.0, 150.0)
        })
        result = validator.validate(df_out_of_range, 'comm_oil_brent_glb_d_brent')

        assert result.passed is False
        assert len(result.errors) > 0
        assert 'outside range' in result.errors[0].lower()

    def test_no_range_defined_passes(self, valid_daily_df):
        """Test that variables without defined range pass with warning."""
        validator = RangeValidator(ranges={})
        result = validator.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert result.passed is True
        assert any('No range defined' in w for w in result.warnings)

    def test_missing_column_fails(self, valid_daily_df):
        """Test that missing column fails validation."""
        validator = RangeValidator(ranges={
            'nonexistent_column': (0, 100)
        })
        result = validator.validate(valid_daily_df, 'nonexistent_column')

        assert result.passed is False
        assert result.severity == ValidationSeverity.CRITICAL

    def test_range_metadata_included(self, valid_daily_df):
        """Test that range metadata is included in result."""
        validator = RangeValidator(ranges={
            'comm_oil_brent_glb_d_brent': (30.0, 150.0)
        })
        result = validator.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert 'range' in result.metadata
        assert result.metadata['range'] == [30.0, 150.0]


# =============================================================================
# COMPLETENESS VALIDATOR TESTS
# =============================================================================

class TestCompletenessValidator:
    """Tests for CompletenessValidator."""

    def test_complete_data_passes(self, valid_daily_df):
        """Test that complete data (no nulls) passes validation."""
        validator = CompletenessValidator()
        result = validator.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert result.passed is True
        assert result.metadata['completeness'] == 1.0

    def test_data_with_nulls_may_pass(self, df_with_nulls):
        """Test that data with some nulls may still pass if above threshold."""
        validator = CompletenessValidator(custom_threshold=0.90)
        result = validator.validate(df_with_nulls, 'test_var')

        # 97% completeness (3 nulls out of 100) should pass 90% threshold
        assert result.passed is True

    def test_incomplete_data_fails(self):
        """Test that very incomplete data fails validation."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = [None] * 50 + [100.0] * 50  # 50% nulls
        df = pd.DataFrame({
            'fecha': dates,
            'test_var': values,
        })

        validator = CompletenessValidator(custom_threshold=0.95)
        result = validator.validate(df, 'test_var')

        assert result.passed is False
        assert 'below threshold' in result.errors[0].lower()

    def test_frequency_inference(self, valid_monthly_df):
        """Test that frequency is correctly inferred from variable name."""
        validator = CompletenessValidator()
        result = validator.validate(valid_monthly_df, 'polr_fed_funds_usa_m_fedfunds')

        # Should infer 'monthly' from '_m_' in variable name
        assert result.metadata['frequency'] == 'monthly'

    def test_empty_dataframe_fails(self):
        """Test that empty DataFrame fails validation."""
        validator = CompletenessValidator()
        result = validator.validate(pd.DataFrame({'fecha': [], 'test_var': []}), 'test_var')

        assert result.passed is False


# =============================================================================
# LEAKAGE VALIDATOR TESTS
# =============================================================================

class TestLeakageValidator:
    """Tests for LeakageValidator."""

    def test_no_future_dates_passes(self, valid_daily_df):
        """Test that data without future dates passes validation."""
        validator = LeakageValidator()
        result = validator.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert result.passed is True
        assert result.metadata['future_rows_count'] == 0

    def test_future_dates_detected(self, df_with_future_dates):
        """Test that future dates are detected as leakage."""
        validator = LeakageValidator()
        result = validator.validate(df_with_future_dates, 'test_var')

        assert result.passed is False
        assert 'LEAKAGE' in result.errors[0]
        assert result.severity == ValidationSeverity.CRITICAL

    def test_custom_reference_date(self, valid_daily_df):
        """Test validation with custom reference date."""
        # Set reference date to before the data
        reference = datetime(2023, 1, 1)
        validator = LeakageValidator(reference_date=reference)
        result = validator.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        # All dates are "in the future" relative to 2023-01-01
        assert result.passed is False
        assert 'LEAKAGE' in result.errors[0]

    def test_missing_fecha_column_fails(self):
        """Test that missing fecha column fails validation."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'test_var': range(10),
        })
        validator = LeakageValidator()
        result = validator.validate(df, 'test_var')

        assert result.passed is False
        assert 'fecha' in result.errors[0].lower()


# =============================================================================
# FRESHNESS VALIDATOR TESTS
# =============================================================================

class TestFreshnessValidator:
    """Tests for FreshnessValidator."""

    def test_fresh_data_passes(self):
        """Test that recent data passes freshness validation."""
        dates = pd.date_range(
            datetime.now() - timedelta(days=5),
            periods=10,
            freq='D'
        )
        df = pd.DataFrame({
            'fecha': dates,
            'test_var': range(10),
        })

        validator = FreshnessValidator()
        result = validator.validate(df, 'test_var')

        assert result.passed is True  # Freshness is a warning, not failure
        assert len(result.warnings) == 0

    def test_stale_data_warns(self):
        """Test that stale data generates warning."""
        dates = pd.date_range(
            datetime.now() - timedelta(days=30),
            periods=10,
            freq='D'
        )
        df = pd.DataFrame({
            'fecha': dates,
            'test_var': range(10),
        })

        validator = FreshnessValidator(max_staleness_days=5)
        result = validator.validate(df, 'test_var')

        assert result.passed is True  # Freshness doesn't fail, just warns
        assert len(result.warnings) > 0
        assert 'stale' in result.warnings[0].lower()

    def test_frequency_specific_threshold(self):
        """Test that frequency affects staleness threshold."""
        # Monthly data can be 45 days old
        dates = pd.date_range(
            datetime.now() - timedelta(days=40),
            periods=5,
            freq='MS'
        )
        df = pd.DataFrame({
            'fecha': dates,
            'polr_fed_funds_usa_m_fedfunds': range(5),
        })

        validator = FreshnessValidator()
        result = validator.validate(df, 'polr_fed_funds_usa_m_fedfunds')

        # Should not warn - 40 days is within 45 day threshold for monthly
        assert len(result.warnings) == 0


# =============================================================================
# VALIDATION PIPELINE TESTS
# =============================================================================

class TestValidationPipeline:
    """Tests for ValidationPipeline."""

    def test_all_validators_run(self, valid_daily_df):
        """Test that all validators run in pipeline."""
        pipeline = ValidationPipeline()
        report = pipeline.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert len(report.results) == 5  # Schema, Range, Completeness, Leakage, Freshness
        assert all(isinstance(r, ValidationResult) for r in report.results)

    def test_overall_passed_when_all_pass(self, valid_daily_df):
        """Test that overall_passed is True when all validators pass."""
        # Use custom ranges to ensure Range validator passes
        pipeline = ValidationPipeline(validators=[
            SchemaValidator(),
            RangeValidator(ranges={'comm_oil_brent_glb_d_brent': (0, 1000)}),
            CompletenessValidator(),
            LeakageValidator(),
        ])
        report = pipeline.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert report.overall_passed is True

    def test_overall_failed_when_any_fails(self, df_with_future_dates):
        """Test that overall_passed is False when any validator fails."""
        pipeline = ValidationPipeline()
        report = pipeline.validate(df_with_future_dates, 'test_var')

        assert report.overall_passed is False
        assert len(report.get_all_errors()) > 0

    def test_fail_fast_mode(self, df_with_future_dates):
        """Test that fail_fast stops on first failure."""
        pipeline = ValidationPipeline(fail_fast=True)
        report = pipeline.validate(df_with_future_dates, 'test_var')

        # Should stop after first failure
        # Leakage validator would fail, but it runs after Schema
        # So we should have at least Schema and possibly Leakage results
        assert report.overall_passed is False
        failed_count = sum(1 for r in report.results if not r.passed)
        assert failed_count >= 1

    def test_custom_validators(self, valid_daily_df):
        """Test pipeline with custom validator list."""
        pipeline = ValidationPipeline(validators=[
            SchemaValidator(),
        ])
        report = pipeline.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        assert len(report.results) == 1
        assert report.results[0].validator == 'SchemaValidator'

    def test_batch_validation(self, valid_daily_df):
        """Test validating multiple variables."""
        pipeline = ValidationPipeline()
        reports = pipeline.validate_batch(
            valid_daily_df,
            ['comm_oil_brent_glb_d_brent', 'volt_vix_usa_d_vix']
        )

        assert len(reports) == 2
        assert 'comm_oil_brent_glb_d_brent' in reports
        assert 'volt_vix_usa_d_vix' in reports

    def test_report_to_dict(self, valid_daily_df):
        """Test ValidationReport serialization."""
        pipeline = ValidationPipeline()
        report = pipeline.validate(valid_daily_df, 'comm_oil_brent_glb_d_brent')

        report_dict = report.to_dict()

        assert 'variable' in report_dict
        assert 'overall_passed' in report_dict
        assert 'errors' in report_dict
        assert 'results_count' in report_dict

    def test_get_critical_failures(self, df_with_future_dates):
        """Test getting only critical failures."""
        pipeline = ValidationPipeline()
        report = pipeline.validate(df_with_future_dates, 'test_var')

        critical = report.get_critical_failures()

        assert len(critical) > 0
        assert all(r.severity == ValidationSeverity.CRITICAL for r in critical)


# =============================================================================
# VALIDATION RESULT/REPORT TESTS
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_error_on_fail(self):
        """Test that default error message is added on failure."""
        result = ValidationResult(
            passed=False,
            validator="TestValidator",
            errors=[],
        )
        assert len(result.errors) == 1
        assert 'failed' in result.errors[0].lower()

    def test_no_default_error_when_provided(self):
        """Test that no default error when errors provided."""
        result = ValidationResult(
            passed=False,
            validator="TestValidator",
            errors=["Custom error"],
        )
        assert len(result.errors) == 1
        assert result.errors[0] == "Custom error"


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_overall_passed_computed(self):
        """Test that overall_passed is computed from results."""
        report = ValidationReport(
            variable="test_var",
            results=[
                ValidationResult(passed=True, validator="A"),
                ValidationResult(passed=True, validator="B"),
            ]
        )
        assert report.overall_passed is True

        report2 = ValidationReport(
            variable="test_var",
            results=[
                ValidationResult(passed=True, validator="A"),
                ValidationResult(passed=False, validator="B", errors=["fail"]),
            ]
        )
        assert report2.overall_passed is False

    def test_get_all_errors_prefixes_validator(self):
        """Test that errors are prefixed with validator name."""
        report = ValidationReport(
            variable="test_var",
            results=[
                ValidationResult(
                    passed=False,
                    validator="TestValidator",
                    errors=["Something failed"]
                ),
            ]
        )

        errors = report.get_all_errors()
        assert len(errors) == 1
        assert "[TestValidator]" in errors[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
