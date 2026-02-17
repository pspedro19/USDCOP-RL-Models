# -*- coding: utf-8 -*-
"""
Unit Tests for L2 Data Quality Report Generator
================================================

Tests for the L2DataQualityReportGenerator and related classes.

Version: 1.0.0
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_records = 500

    dates = pd.date_range(start='2024-01-01', periods=n_records, freq='D')

    df = pd.DataFrame({
        'fecha': dates,
        'close': np.random.randn(n_records) * 100 + 4200,
        'open': np.random.randn(n_records) * 100 + 4200,
        'high': np.random.randn(n_records) * 100 + 4250,
        'low': np.random.randn(n_records) * 100 + 4150,
        'volume': np.random.exponential(1000, n_records),
        'rsi_14': np.random.uniform(20, 80, n_records),
        'fed_funds': np.random.randn(n_records) * 0.5 + 5.0,
        'dxy': np.random.randn(n_records) * 2 + 104,
    })

    # Introduce some missing values
    df.loc[10:15, 'close'] = np.nan
    df.loc[100:110, 'fed_funds'] = np.nan

    # Introduce some outliers
    df.loc[50, 'close'] = 10000  # Extreme outlier
    df.loc[200, 'rsi_14'] = 150  # Invalid RSI

    return df


@pytest.fixture
def sample_df_with_issues():
    """Create a DataFrame with more quality issues."""
    np.random.seed(42)
    n_records = 200

    dates = pd.date_range(start='2024-01-01', periods=n_records, freq='D')

    df = pd.DataFrame({
        'fecha': dates,
        'close': np.random.randn(n_records) * 100 + 4200,
        'problematic_var': np.random.randn(n_records) * 10 + 50,
    })

    # 30% missing data
    missing_idx = np.random.choice(n_records, size=60, replace=False)
    df.loc[missing_idx, 'problematic_var'] = np.nan

    # Many outliers
    outlier_idx = np.random.choice(n_records, size=20, replace=False)
    df.loc[outlier_idx, 'close'] = np.random.uniform(8000, 10000, 20)

    return df


@pytest.fixture
def norm_stats():
    """Sample normalization statistics."""
    return {
        '_meta': {
            'version': '6.0.0',
            'computed_from': 'train_set_only',
        },
        'features': {
            'close': {'mean': 4200.0, 'std': 100.0},
            'rsi_14': {'mean': 50.0, 'std': 15.0},
            'fed_funds': {'mean': 5.0, 'std': 0.5},
        }
    }


@pytest.fixture
def generator():
    """Create a report generator instance."""
    # Import here to avoid import errors if module not found
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'airflow' / 'dags'))

    from services.l2_data_quality_report import L2DataQualityReportGenerator
    return L2DataQualityReportGenerator()


# =============================================================================
# DATA CLASS TESTS
# =============================================================================

class TestDataClasses:
    """Tests for data classes."""

    def test_variable_report_creation(self):
        """Test VariableReport can be created."""
        from services.l2_data_quality_report import VariableReport

        report = VariableReport(
            variable_name='close',
            count=100,
            mean=4200.0,
            std=100.0,
        )

        assert report.variable_name == 'close'
        assert report.count == 100
        assert report.mean == 4200.0

    def test_variable_report_to_dict(self):
        """Test VariableReport serialization."""
        from services.l2_data_quality_report import VariableReport

        report = VariableReport(
            variable_name='test',
            count=50,
        )
        d = report.to_dict()

        assert isinstance(d, dict)
        assert d['variable_name'] == 'test'
        assert d['count'] == 50

    def test_l2_report_creation(self):
        """Test L2DataQualityReport can be created."""
        from services.l2_data_quality_report import L2DataQualityReport

        report = L2DataQualityReport(
            report_id='test_001',
            total_variables=10,
            total_records=1000,
        )

        assert report.report_id == 'test_001'
        assert report.total_variables == 10
        assert report.total_records == 1000

    def test_anomaly_info_creation(self):
        """Test AnomalyInfo creation."""
        from services.l2_data_quality_report import AnomalyInfo

        anomaly = AnomalyInfo(
            type='outlier_iqr',
            date='2024-01-15',
            value=10000.0,
            expected_range=(4000.0, 4500.0),
            severity='high',
        )

        assert anomaly.type == 'outlier_iqr'
        assert anomaly.value == 10000.0


# =============================================================================
# GENERATOR TESTS
# =============================================================================

class TestL2DataQualityReportGenerator:
    """Tests for the report generator."""

    def test_generator_initialization(self, generator):
        """Test generator initializes with default parameters."""
        assert generator.zscore_threshold == 3.0
        assert generator.iqr_multiplier == 1.5
        assert generator.jump_threshold_pct == 50.0

    def test_generate_report_basic(self, generator, sample_df):
        """Test basic report generation."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        assert report is not None
        assert report.total_records == len(sample_df)
        assert report.total_variables > 0
        assert 'close' in report.variables

    def test_report_contains_all_numeric_columns(self, generator, sample_df):
        """Test report includes all numeric columns."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        expected_vars = ['close', 'open', 'high', 'low', 'volume', 'rsi_14', 'fed_funds', 'dxy']
        for var in expected_vars:
            assert var in report.variables, f"Missing variable: {var}"

    def test_basic_statistics_calculated(self, generator, sample_df):
        """Test basic statistics are calculated correctly."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        close_report = report.variables['close']

        assert close_report.count > 0
        assert close_report.mean != 0
        assert close_report.std > 0
        assert close_report.min < close_report.max

    def test_missing_data_analysis(self, generator, sample_df):
        """Test missing data is detected."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        # close has 6 missing values (indices 10-15)
        close_report = report.variables['close']
        assert close_report.missing.total_missing == 6

        # fed_funds has 11 missing values (indices 100-110)
        fed_report = report.variables['fed_funds']
        assert fed_report.missing.total_missing == 11

    def test_anomaly_detection(self, generator, sample_df):
        """Test anomalies are detected."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        close_report = report.variables['close']
        # We introduced an extreme outlier at index 50
        assert close_report.anomalies.outliers_iqr_count > 0

    def test_temporal_coverage(self, generator, sample_df):
        """Test temporal coverage is calculated."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        close_report = report.variables['close']

        assert close_report.temporal.fecha_min is not None
        assert close_report.temporal.fecha_max is not None
        assert close_report.temporal.actual_records > 0

    def test_distribution_analysis(self, generator, sample_df):
        """Test distribution analysis is performed."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        close_report = report.variables['close']

        assert close_report.distribution.skewness is not None
        assert close_report.distribution.kurtosis is not None

    def test_anti_leakage_verification(self, generator, sample_df):
        """Test anti-leakage verification for macro variables."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
            reference_date=datetime.now() + timedelta(days=365),  # Future reference
        )

        fed_report = report.variables['fed_funds']
        # fed_funds is a macro variable, should be checked
        assert fed_report.anti_leakage is not None

    def test_quality_score_calculation(self, generator, sample_df):
        """Test quality scores are calculated."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        for var_name, var_report in report.variables.items():
            assert 0 <= var_report.quality_score <= 1, f"{var_name} has invalid score"
            assert var_report.quality_level in [
                'excellent', 'good', 'acceptable', 'poor', 'critical'
            ]

    def test_overall_quality_calculation(self, generator, sample_df):
        """Test overall quality is calculated."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        assert 0 <= report.overall_quality_score <= 1
        assert report.overall_quality_level in [
            'excellent', 'good', 'acceptable', 'poor', 'critical'
        ]

    def test_quality_distribution_counts(self, generator, sample_df):
        """Test quality distribution counts sum correctly."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        total_counted = (
            report.variables_excellent +
            report.variables_good +
            report.variables_acceptable +
            report.variables_poor +
            report.variables_critical
        )

        assert total_counted == report.total_variables

    def test_recommendations_generated(self, generator, sample_df_with_issues):
        """Test recommendations are generated for issues."""
        report = generator.generate_report(
            df=sample_df_with_issues,
            date_column='fecha',
        )

        # Should have some recommendations due to missing data and outliers
        assert len(report.recommendations) > 0 or len(report.warnings) > 0

    def test_normalization_info_extracted(self, generator, sample_df, norm_stats):
        """Test normalization info is extracted from norm_stats."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
            norm_stats=norm_stats,
        )

        close_report = report.variables['close']
        assert close_report.transformations.normalization_method == 'zscore'
        assert close_report.transformations.norm_mean == 4200.0
        assert close_report.transformations.norm_std == 100.0

    def test_correlation_analysis(self, generator, sample_df):
        """Test correlation analysis is performed."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        close_report = report.variables['close']
        assert len(close_report.top_correlations) > 0

    def test_empty_dataframe_handling(self, generator):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame({'fecha': [], 'value': []})

        report = generator.generate_report(
            df=empty_df,
            date_column='fecha',
        )

        assert report.total_records == 0
        assert report.total_variables == 0


# =============================================================================
# SAVE REPORT TESTS
# =============================================================================

class TestSaveReport:
    """Tests for saving reports."""

    def test_save_json_report(self, generator, sample_df):
        """Test saving JSON report."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = generator.save_report(report, tmpdir, formats=['json'])

            assert 'json' in saved
            assert saved['json'].exists()

            # Verify JSON is valid
            with open(saved['json']) as f:
                data = json.load(f)
                assert 'report_id' in data
                assert 'variables' in data

    def test_save_csv_report(self, generator, sample_df):
        """Test saving CSV summary."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = generator.save_report(report, tmpdir, formats=['csv'])

            assert 'csv' in saved
            assert saved['csv'].exists()

            # Verify CSV is valid
            df_summary = pd.read_csv(saved['csv'])
            assert 'variable' in df_summary.columns
            assert 'quality_score' in df_summary.columns
            assert len(df_summary) == report.total_variables

    def test_save_html_report(self, generator, sample_df):
        """Test saving HTML report."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = generator.save_report(report, tmpdir, formats=['html'])

            assert 'html' in saved
            assert saved['html'].exists()

            # Verify HTML contains key elements
            with open(saved['html']) as f:
                content = f.read()
                assert 'L2 Data Quality Report' in content
                assert 'close' in content

    def test_save_multiple_formats(self, generator, sample_df):
        """Test saving multiple formats at once."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = generator.save_report(
                report, tmpdir, formats=['json', 'csv', 'html']
            )

            assert len(saved) == 3
            assert all(p.exists() for p in saved.values())


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunction:
    """Tests for the convenience function."""

    def test_generate_l2_report_function(self, sample_df):
        """Test the generate_l2_report convenience function."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'airflow' / 'dags'))

        from services.l2_data_quality_report import generate_l2_report

        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_l2_report(
                df=sample_df,
                output_dir=tmpdir,
                date_column='fecha',
            )

            assert report is not None
            assert report.total_variables > 0

            # Check files were created
            json_files = list(Path(tmpdir).glob('*.json'))
            csv_files = list(Path(tmpdir).glob('*.csv'))
            assert len(json_files) == 1
            assert len(csv_files) == 1


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_record_dataframe(self, generator):
        """Test handling of single record DataFrame."""
        df = pd.DataFrame({
            'fecha': [datetime.now()],
            'value': [100.0],
        })

        report = generator.generate_report(df, date_column='fecha')
        assert report.total_records == 1

    def test_all_missing_column(self, generator):
        """Test handling of column with all missing values."""
        df = pd.DataFrame({
            'fecha': pd.date_range(start='2024-01-01', periods=10),
            'all_missing': [np.nan] * 10,
            'valid': range(10),
        })

        report = generator.generate_report(df, date_column='fecha')
        assert 'all_missing' in report.variables
        assert report.variables['all_missing'].quality_level == 'critical'

    def test_constant_column(self, generator):
        """Test handling of constant column."""
        df = pd.DataFrame({
            'fecha': pd.date_range(start='2024-01-01', periods=100),
            'constant': [42.0] * 100,
        })

        report = generator.generate_report(df, date_column='fecha')
        assert report.variables['constant'].std == 0
        assert report.variables['constant'].variance == 0

    def test_large_numbers(self, generator):
        """Test handling of large numbers."""
        df = pd.DataFrame({
            'fecha': pd.date_range(start='2024-01-01', periods=100),
            'large': np.random.randn(100) * 1e12,
        })

        report = generator.generate_report(df, date_column='fecha')
        assert report.variables['large'].count == 100

    def test_negative_values(self, generator):
        """Test handling of negative values."""
        df = pd.DataFrame({
            'fecha': pd.date_range(start='2024-01-01', periods=100),
            'negative': np.random.randn(100) * 100 - 50,
        })

        report = generator.generate_report(df, date_column='fecha')
        assert report.variables['negative'].min < 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self, sample_df, norm_stats):
        """Test complete workflow from generation to saving."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'airflow' / 'dags'))

        from services.l2_data_quality_report import L2DataQualityReportGenerator

        generator = L2DataQualityReportGenerator(
            zscore_threshold=2.5,
            iqr_multiplier=2.0,
        )

        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
            norm_stats=norm_stats,
            reference_date=datetime(2025, 1, 1),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = generator.save_report(
                report, tmpdir, formats=['json', 'csv']
            )

            # Verify report structure
            assert report.total_variables == 8
            assert report.date_range_start is not None
            assert report.date_range_end is not None

            # Verify all variables have reports
            for var in report.variables.values():
                assert var.count > 0 or var.missing.total_missing > 0
                assert var.quality_score >= 0
                assert var.quality_level is not None

            # Verify saved files
            assert all(p.exists() for p in saved.values())

    def test_report_to_dict_serializable(self, generator, sample_df):
        """Test report can be fully serialized to JSON."""
        report = generator.generate_report(
            df=sample_df,
            date_column='fecha',
        )

        # Should not raise
        json_str = json.dumps(report.to_dict(), default=str)
        assert len(json_str) > 0

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed['report_id'] == report.report_id
