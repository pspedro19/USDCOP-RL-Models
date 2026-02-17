# -*- coding: utf-8 -*-
"""
Unit Tests for Anti-Leakage Validator
=====================================

Tests for:
- AntiLeakageValidator
- Macro T-1 shift validation
- Normalization source validation
- Dataset split validation
- Merge direction validation

Contract: CTR-L0-ANTILEAKAGE-001
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
DAGS_PATH = PROJECT_ROOT / 'airflow' / 'dags'

for path in [str(DAGS_PATH), str(PROJECT_ROOT / 'src')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from validators.anti_leakage_validator import (
    AntiLeakageValidator,
    AntiLeakageReport,
    validate_anti_leakage,
)
from validators.data_validators import ValidationSeverity


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def validator():
    """Create an AntiLeakageValidator instance."""
    return AntiLeakageValidator(strict_mode=True)


@pytest.fixture
def validator_lenient():
    """Create a lenient AntiLeakageValidator instance."""
    return AntiLeakageValidator(strict_mode=False)


@pytest.fixture
def valid_macro_df():
    """Create a valid macro DataFrame with T-1 shifted data."""
    # Data from 30 days ago to yesterday (no today's data)
    yesterday = datetime.now().date() - timedelta(days=1)
    start = yesterday - timedelta(days=30)
    dates = pd.date_range(start, yesterday, freq='D')

    return pd.DataFrame({
        'fecha': dates,
        'IBR_OVERNIGHT': [8.0 + i * 0.01 for i in range(len(dates))],
        'FEDFUNDS': [5.25 + i * 0.005 for i in range(len(dates))],
    })


@pytest.fixture
def leaky_macro_df():
    """Create a DataFrame with same-day macro data (leakage!)."""
    # Include today's data - this is leakage
    today = datetime.now().date()
    start = today - timedelta(days=30)
    dates = pd.date_range(start, today, freq='D')

    return pd.DataFrame({
        'fecha': dates,
        'IBR_OVERNIGHT': [8.0 + i * 0.01 for i in range(len(dates))],
        'FEDFUNDS': [5.25 + i * 0.005 for i in range(len(dates))],
    })


@pytest.fixture
def train_df():
    """Create training DataFrame."""
    dates = pd.date_range('2020-01-01', '2024-06-30', freq='D')
    return pd.DataFrame({
        'fecha': dates,
        'value': range(len(dates)),
    })


@pytest.fixture
def val_df():
    """Create validation DataFrame (no overlap with train)."""
    dates = pd.date_range('2024-07-01', '2024-12-31', freq='D')
    return pd.DataFrame({
        'fecha': dates,
        'value': range(len(dates)),
    })


@pytest.fixture
def test_df():
    """Create test DataFrame (no overlap with val)."""
    dates = pd.date_range('2025-01-01', '2025-06-30', freq='D')
    return pd.DataFrame({
        'fecha': dates,
        'value': range(len(dates)),
    })


@pytest.fixture
def overlapping_val_df():
    """Create validation DataFrame that overlaps with train (leakage!)."""
    dates = pd.date_range('2024-06-01', '2024-09-30', freq='D')  # Overlaps train
    return pd.DataFrame({
        'fecha': dates,
        'value': range(len(dates)),
    })


@pytest.fixture
def valid_norm_stats():
    """Create valid normalization stats with proper metadata."""
    return {
        '_meta': {
            'version': '6.0.0',
            'computed_from': 'train_set_only',
            'date_range': {
                'start': '2020-01-01',
                'end': '2024-06-30',
            },
        },
        'log_ret_5m': {
            'mean': 0.0001,
            'std': 0.001,
            'count': 50000,
        },
    }


@pytest.fixture
def leaky_norm_stats():
    """Create normalization stats computed from all data (leakage!)."""
    return {
        '_meta': {
            'version': '6.0.0',
            'computed_from': 'all_data',  # Leakage!
            'date_range': {
                'start': '2020-01-01',
                'end': '2025-12-31',  # Extends into test period
            },
        },
        'log_ret_5m': {
            'mean': 0.0001,
            'std': 0.001,
            'count': 100000,
        },
    }


# =============================================================================
# MACRO T-1 SHIFT TESTS
# =============================================================================

class TestMacroShiftT1:
    """Tests for macro T-1 shift validation."""

    def test_valid_t1_shift_passes(self, validator, valid_macro_df):
        """Test that properly T-1 shifted data passes validation."""
        result = validator.validate_macro_shift_t1(
            valid_macro_df,
            'IBR_OVERNIGHT',
        )

        assert result.passed is True
        assert len(result.errors) == 0
        assert 'most_recent_date' in result.metadata

    def test_same_day_data_fails_strict(self, validator, leaky_macro_df):
        """Test that same-day macro data fails in strict mode."""
        result = validator.validate_macro_shift_t1(
            leaky_macro_df,
            'IBR_OVERNIGHT',
        )

        assert result.passed is False
        assert any('LEAKAGE' in e or 'leakage' in e.lower() for e in result.errors)
        assert result.severity == ValidationSeverity.CRITICAL

    def test_same_day_data_warns_lenient(self, validator_lenient, leaky_macro_df):
        """Test that same-day macro data only warns in lenient mode."""
        result = validator_lenient.validate_macro_shift_t1(
            leaky_macro_df,
            'IBR_OVERNIGHT',
        )

        # In lenient mode, it should pass but have warnings
        assert result.passed is True
        assert len(result.warnings) > 0

    def test_missing_date_column_fails(self, validator):
        """Test that missing date column fails validation."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),  # Wrong column name
            'IBR': range(10),
        })

        result = validator.validate_macro_shift_t1(df, 'IBR')

        assert result.passed is False
        assert 'fecha' in result.errors[0].lower()

    def test_missing_value_column_fails(self, validator):
        """Test that missing value column fails validation."""
        df = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=10),
            'other_column': range(10),
        })

        result = validator.validate_macro_shift_t1(df, 'IBR')

        assert result.passed is False
        assert 'IBR' in result.errors[0]

    def test_insufficient_data_warns(self, validator):
        """Test that insufficient data returns warning."""
        df = pd.DataFrame({
            'fecha': [datetime.now() - timedelta(days=10)],
            'IBR': [8.0],
        })

        result = validator.validate_macro_shift_t1(df, 'IBR')

        # Should pass but warn about insufficient data
        assert result.passed is True
        assert len(result.warnings) > 0 or result.metadata.get('dates_count', 0) < 2


# =============================================================================
# NORMALIZATION SOURCE TESTS
# =============================================================================

class TestNormalizationSource:
    """Tests for normalization source validation."""

    def test_valid_norm_stats_passes(self, validator, valid_norm_stats):
        """Test that properly sourced norm stats pass validation."""
        result = validator.validate_normalization_source(
            valid_norm_stats,
            train_start='2020-01-01',
            train_end='2024-06-30',
        )

        assert result.passed is True
        assert len(result.errors) == 0

    def test_wrong_source_fails(self, validator, leaky_norm_stats):
        """Test that norm stats from all data fail validation."""
        result = validator.validate_normalization_source(
            leaky_norm_stats,
            train_start='2020-01-01',
            train_end='2024-06-30',
        )

        assert result.passed is False
        assert any('LEAKAGE' in e for e in result.errors)

    def test_date_range_exceeds_train_fails(self, validator, valid_norm_stats):
        """Test that norm stats beyond train period fail."""
        # Modify to have date range beyond train end
        stats = valid_norm_stats.copy()
        stats['_meta'] = valid_norm_stats['_meta'].copy()
        stats['_meta']['date_range'] = {
            'start': '2020-01-01',
            'end': '2025-06-30',  # Beyond train end of 2024-06-30
        }

        result = validator.validate_normalization_source(
            stats,
            train_start='2020-01-01',
            train_end='2024-06-30',
        )

        assert result.passed is False
        assert any('after' in e.lower() for e in result.errors)

    def test_missing_meta_warns(self, validator):
        """Test that missing _meta section generates warning."""
        stats = {
            'log_ret_5m': {'mean': 0.0, 'std': 1.0},
        }

        result = validator.validate_normalization_source(
            stats,
            train_start='2020-01-01',
            train_end='2024-06-30',
        )

        # Should not fail, but should warn
        assert len(result.warnings) > 0
        assert any('_meta' in w for w in result.warnings)


# =============================================================================
# DATASET SPLIT TESTS
# =============================================================================

class TestDatasetSplits:
    """Tests for dataset split validation."""

    def test_valid_splits_pass(self, validator, train_df, val_df, test_df):
        """Test that non-overlapping chronological splits pass."""
        result = validator.validate_dataset_splits(
            train_df, val_df, test_df
        )

        assert result.passed is True
        assert len(result.errors) == 0
        assert 'train_range' in result.metadata
        assert 'val_range' in result.metadata
        assert 'test_range' in result.metadata

    def test_overlapping_train_val_fails(self, validator, train_df, overlapping_val_df, test_df):
        """Test that overlapping train/val splits fail."""
        result = validator.validate_dataset_splits(
            train_df, overlapping_val_df, test_df
        )

        assert result.passed is False
        assert any('overlap' in e.lower() for e in result.errors)
        assert result.severity == ValidationSeverity.CRITICAL

    def test_overlapping_val_test_fails(self, validator, train_df):
        """Test that overlapping val/test splits fail."""
        # Create overlapping val and test
        val_df = pd.DataFrame({
            'fecha': pd.date_range('2024-07-01', '2025-03-31', freq='D'),
            'value': range(274),
        })
        test_df = pd.DataFrame({
            'fecha': pd.date_range('2025-01-01', '2025-06-30', freq='D'),
            'value': range(181),
        })

        result = validator.validate_dataset_splits(train_df, val_df, test_df)

        assert result.passed is False
        assert any('Val/Test overlap' in e for e in result.errors)

    def test_gaps_between_splits_warn(self, validator):
        """Test that gaps between splits generate warnings."""
        train = pd.DataFrame({
            'fecha': pd.date_range('2020-01-01', '2024-06-30', freq='D'),
            'value': range(1643),
        })
        val = pd.DataFrame({
            'fecha': pd.date_range('2024-08-01', '2024-12-31', freq='D'),  # Gap after train
            'value': range(153),
        })
        test = pd.DataFrame({
            'fecha': pd.date_range('2025-01-01', '2025-06-30', freq='D'),
            'value': range(181),
        })

        result = validator.validate_dataset_splits(train, val, test)

        # Should pass but warn about gap
        assert result.passed is True
        assert len(result.warnings) > 0
        assert any('gap' in w.lower() for w in result.warnings)


# =============================================================================
# MERGE DIRECTION TESTS
# =============================================================================

class TestMergeAsofDirection:
    """Tests for merge_asof direction validation."""

    def test_backward_merge_passes(self, validator):
        """Test that backward merge (correct) passes validation."""
        # Create merged DataFrame where right_date <= left_date
        merged = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=10, freq='D'),
            'fecha_macro': pd.date_range('2023-12-31', periods=10, freq='D'),  # One day behind
            'value': range(10),
        })

        result = validator.validate_merge_asof_direction(
            merged,
            left_date_col='fecha',
            right_date_col='fecha_macro',
        )

        assert result.passed is True
        assert result.metadata['forward_merge_count'] == 0

    def test_forward_merge_fails(self, validator):
        """Test that forward merge (leakage!) fails validation."""
        # Create merged DataFrame where right_date > left_date (forward merge!)
        merged = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=10, freq='D'),
            'fecha_macro': pd.date_range('2024-01-02', periods=10, freq='D'),  # One day ahead!
            'value': range(10),
        })

        result = validator.validate_merge_asof_direction(
            merged,
            left_date_col='fecha',
            right_date_col='fecha_macro',
        )

        assert result.passed is False
        assert 'LEAKAGE' in result.errors[0]
        assert result.metadata['forward_merge_count'] == 10

    def test_mixed_merge_fails(self, validator):
        """Test that partial forward merge still fails."""
        # Some rows have forward merge
        merged = pd.DataFrame({
            'fecha': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'fecha_macro': pd.to_datetime(['2023-12-31', '2024-01-03', '2024-01-02']),  # 2nd row is forward
            'value': [1, 2, 3],
        })

        result = validator.validate_merge_asof_direction(
            merged,
            left_date_col='fecha',
            right_date_col='fecha_macro',
        )

        assert result.passed is False
        assert result.metadata['forward_merge_count'] == 1

    def test_missing_columns_fails(self, validator):
        """Test that missing columns fail validation."""
        merged = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'value': range(5),
        })

        result = validator.validate_merge_asof_direction(
            merged,
            left_date_col='fecha',
            right_date_col='fecha_macro',
        )

        assert result.passed is False
        assert 'Missing' in result.errors[0]


# =============================================================================
# NO FUTURE DATA TESTS
# =============================================================================

class TestNoFutureData:
    """Tests for future data detection."""

    def test_historical_data_passes(self, validator):
        """Test that purely historical data passes."""
        yesterday = datetime.now() - timedelta(days=1)
        df = pd.DataFrame({
            'fecha': pd.date_range('2020-01-01', yesterday, freq='D'),
            'value': range(365 * 4),  # ~4 years
        })

        result = validator.validate_no_future_data(df, yesterday)

        assert result.passed is True
        assert result.metadata['future_rows_count'] == 0

    def test_future_data_fails(self, validator):
        """Test that future data is detected and fails."""
        today = datetime.now()
        future = today + timedelta(days=10)
        df = pd.DataFrame({
            'fecha': pd.date_range(today - timedelta(days=5), future, freq='D'),
            'value': range(16),
        })

        result = validator.validate_no_future_data(df, today)

        assert result.passed is False
        assert 'LEAKAGE' in result.errors[0]
        assert result.metadata['future_rows_count'] > 0


# =============================================================================
# FULL VALIDATION TESTS
# =============================================================================

class TestFullValidation:
    """Tests for complete anti-leakage validation."""

    def test_clean_dataset_passes(self, validator, valid_macro_df):
        """Test that a clean dataset passes full validation."""
        report = validator.run_full_validation(
            valid_macro_df,
            reference_date=datetime.now(),
        )

        assert report.passed is True
        assert len(report.get_all_errors()) == 0

    def test_report_structure(self, validator, valid_macro_df):
        """Test that report has expected structure."""
        report = validator.run_full_validation(valid_macro_df)

        assert isinstance(report, AntiLeakageReport)
        assert hasattr(report, 'passed')
        assert hasattr(report, 'checks')
        assert hasattr(report, 'timestamp')

        report_dict = report.to_dict()
        assert 'passed' in report_dict
        assert 'checks' in report_dict
        assert 'error_count' in report_dict

    def test_with_norm_stats_file(self, validator, valid_macro_df, valid_norm_stats):
        """Test validation with norm_stats file."""
        with TemporaryDirectory() as tmpdir:
            norm_path = Path(tmpdir) / 'norm_stats.json'
            with open(norm_path, 'w') as f:
                json.dump(valid_norm_stats, f)

            report = validator.run_full_validation(
                valid_macro_df,
                norm_stats_path=norm_path,
                train_dates=('2020-01-01', '2024-06-30'),
            )

            assert 'normalization_source' in report.checks


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestValidateAntiLeakage:
    """Tests for the convenience function."""

    def test_quick_validation(self, valid_macro_df):
        """Test quick validation function."""
        result = validate_anti_leakage(valid_macro_df)
        assert isinstance(result, bool)

    def test_strict_vs_lenient(self, leaky_macro_df):
        """Test strict vs lenient mode."""
        strict_result = validate_anti_leakage(leaky_macro_df, strict=True)
        lenient_result = validate_anti_leakage(leaky_macro_df, strict=False)

        # Strict should fail, lenient should pass
        assert strict_result is False
        assert lenient_result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
