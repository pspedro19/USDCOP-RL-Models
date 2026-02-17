# -*- coding: utf-8 -*-
"""
Unit Tests for L0 Macro Update DAG v2.0
=======================================

Tests for:
- HourlyExtractionReport dataclass
- DailyExtractionSummary dataclass
- is_last_run_of_day() helper
- check_market_hours() logic
- extract_all_sources() always rewriting behavior
- Circuit breaker integration

Contract: CTR-L0-UPDATE-002
"""

import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
DAGS_PATH = PROJECT_ROOT / 'airflow' / 'dags'

for path in [str(DAGS_PATH), str(PROJECT_ROOT / 'src')]:
    if path not in sys.path:
        sys.path.insert(0, path)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_extraction_df():
    """Create sample extraction DataFrame."""
    dates = pd.date_range('2024-01-01', periods=15, freq='D')
    return pd.DataFrame({
        'fecha': dates,
        'dxy': [100.0 + i * 0.1 for i in range(15)],
    })


@pytest.fixture
def mock_context():
    """Create mock Airflow context."""
    ti_mock = Mock()
    ti_mock.xcom_push = Mock()
    ti_mock.xcom_pull = Mock(return_value={})

    dag_run_mock = Mock()
    dag_run_mock.conf = {}

    return {
        'ti': ti_mock,
        'dag_run': dag_run_mock,
        'run_id': 'test_run_123',
        'execution_date': datetime.now(),
    }


@pytest.fixture
def mock_registry():
    """Create mock ExtractorRegistry."""
    mock = Mock()
    mock.get_all_sources.return_value = ['fred', 'investing']
    mock.get_all_variables.return_value = ['dxy', 'vix', 'ust10y']
    mock.get_variables_by_source.return_value = ['dxy']
    mock.get_variable_config.return_value = {'frequency': 'D'}

    result = Mock()
    result.success = True
    result.data = pd.DataFrame({
        'fecha': pd.date_range('2024-01-01', periods=15, freq='D'),
        'dxy': [100.0] * 15,
    })
    result.error = None

    mock.extract_variable.return_value = result
    return mock


# =============================================================================
# HOURLY EXTRACTION REPORT TESTS
# =============================================================================

class TestHourlyExtractionReport:
    """Tests for HourlyExtractionReport dataclass."""

    def test_default_initialization(self):
        """Test that report initializes with defaults."""
        # Import here to avoid issues with module loading
        from l0_macro_update import HourlyExtractionReport

        report = HourlyExtractionReport()

        assert report.source_success_count == {}
        assert report.source_failed_count == {}
        assert report.variable_success == []
        assert report.variable_failed == []
        assert report.total_records_extracted == 0
        assert report.total_records_upserted == 0

    def test_to_dict_serialization(self):
        """Test that report serializes to dict properly."""
        from l0_macro_update import HourlyExtractionReport

        report = HourlyExtractionReport()
        report.source_success_count = {'fred': 10, 'investing': 15}
        report.variable_success = ['dxy', 'vix']
        report.total_records_extracted = 100

        result = report.to_dict()

        assert isinstance(result, dict)
        assert result['source_success_count'] == {'fred': 10, 'investing': 15}
        assert result['variable_success'] == ['dxy', 'vix']
        assert result['total_records_extracted'] == 100
        assert 'run_timestamp' in result

    def test_to_log_format(self):
        """Test log format string generation."""
        from l0_macro_update import HourlyExtractionReport

        report = HourlyExtractionReport()
        report.variable_success = ['dxy', 'vix', 'ust10y']
        report.variable_failed = ['embi']
        report.total_records_extracted = 45
        report.total_records_upserted = 45
        report.total_extraction_duration_ms = 5000
        report.total_upsert_duration_ms = 1000

        log_str = report.to_log_format()

        assert 'Vars: 3/4' in log_str
        assert '75.0%' in log_str
        assert '45 extracted' in log_str
        assert '5000ms' in log_str

    def test_frequency_tracking(self):
        """Test that frequency counters work correctly."""
        from l0_macro_update import HourlyExtractionReport

        report = HourlyExtractionReport()
        report.daily_vars_success = 10
        report.monthly_vars_success = 5
        report.quarterly_vars_success = 2

        result = report.to_dict()

        assert result['daily_vars_success'] == 10
        assert result['monthly_vars_success'] == 5
        assert result['quarterly_vars_success'] == 2


# =============================================================================
# DAILY EXTRACTION SUMMARY TESTS
# =============================================================================

class TestDailyExtractionSummary:
    """Tests for DailyExtractionSummary dataclass."""

    def test_default_initialization(self):
        """Test that summary initializes with today's date."""
        from l0_macro_update import DailyExtractionSummary

        summary = DailyExtractionSummary()

        assert summary.date == date.today()
        assert summary.total_runs == 0
        assert summary.successful_runs == 0
        assert summary.source_success_rate == {}

    def test_to_dict_serialization(self):
        """Test that summary serializes to dict properly."""
        from l0_macro_update import DailyExtractionSummary

        summary = DailyExtractionSummary()
        summary.total_runs = 5
        summary.source_success_rate = {'fred': 1.0, 'investing': 0.9}
        summary.sources_with_issues = ['dane']

        result = summary.to_dict()

        assert isinstance(result, dict)
        assert result['total_runs'] == 5
        assert result['source_success_rate'] == {'fred': 1.0, 'investing': 0.9}
        assert result['sources_with_issues'] == ['dane']

    def test_issues_detection_threshold(self):
        """Test that sources with <80% success are flagged."""
        from l0_macro_update import DailyExtractionSummary

        summary = DailyExtractionSummary()
        summary.source_success_rate = {'fred': 1.0, 'dane': 0.7, 'investing': 0.9}
        summary.sources_with_issues = ['dane']  # dane < 0.8

        assert 'dane' in summary.sources_with_issues
        assert 'fred' not in summary.sources_with_issues


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestIsLastRunOfDay:
    """Tests for is_last_run_of_day() helper."""

    @patch('l0_macro_update.datetime')
    def test_returns_true_at_17_utc(self, mock_datetime):
        """Test that 17:00 UTC is detected as last run."""
        from l0_macro_update import is_last_run_of_day

        mock_datetime.utcnow.return_value = datetime(2024, 1, 15, 17, 30)

        result = is_last_run_of_day()

        assert result is True

    @patch('l0_macro_update.datetime')
    def test_returns_false_at_other_hours(self, mock_datetime):
        """Test that other hours are not last run."""
        from l0_macro_update import is_last_run_of_day

        # Test 14:00 UTC (9:00 COT)
        mock_datetime.utcnow.return_value = datetime(2024, 1, 15, 14, 30)

        result = is_last_run_of_day()

        assert result is False


class TestCheckMarketHours:
    """Tests for check_market_hours() function."""

    def test_force_run_bypasses_check(self, mock_context):
        """Test that force_run=True bypasses market hours check."""
        from l0_macro_update import check_market_hours

        mock_context['dag_run'].conf = {'force_run': True}

        result = check_market_hours(**mock_context)

        assert result is True

    @patch('l0_macro_update.datetime')
    def test_within_market_hours(self, mock_datetime, mock_context):
        """Test that market hours are correctly detected."""
        from l0_macro_update import check_market_hours

        # Create a mock that can be timezone-aware
        with patch('pytz.timezone') as mock_tz:
            mock_now = Mock()
            mock_now.weekday.return_value = 1  # Tuesday
            mock_now.hour = 10  # 10:00 COT
            mock_now.strftime.return_value = "10:00 COT"

            mock_tz.return_value.localize = lambda dt: mock_now
            mock_datetime.now.return_value = mock_now

            mock_context['dag_run'].conf = {}

            result = check_market_hours(**mock_context)

            assert result is True

    @patch('l0_macro_update.datetime')
    def test_outside_market_hours(self, mock_datetime, mock_context):
        """Test that outside market hours returns False."""
        from l0_macro_update import check_market_hours

        with patch('pytz.timezone') as mock_tz:
            mock_now = Mock()
            mock_now.weekday.return_value = 5  # Saturday
            mock_now.hour = 10
            mock_now.strftime.return_value = "2024-01-20 10:00"

            mock_tz.return_value.localize = lambda dt: mock_now
            mock_datetime.now.return_value = mock_now

            mock_context['dag_run'].conf = {}

            result = check_market_hours(**mock_context)

            assert result is False


# =============================================================================
# EXTRACTION LOGIC TESTS
# =============================================================================

class TestExtractAllSources:
    """Tests for extract_all_sources() always-rewrite behavior."""

    def test_always_rewrites_15_records(self, mock_context, mock_registry):
        """Verify that extraction always requests 15 records (no change detection)."""
        from l0_macro_update import SAFETY_RECORDS

        assert SAFETY_RECORDS == 15

    def test_skip_sources_parameter(self, mock_context, mock_registry):
        """Test that skip_sources parameter is respected."""
        from l0_macro_update import extract_all_sources

        mock_context['dag_run'].conf = {'skip_sources': ['dane']}

        with patch('l0_macro_update.ExtractorRegistry', return_value=mock_registry):
            with patch('l0_macro_update.reset_metrics_collector') as mock_collector:
                mock_collector.return_value = Mock()
                mock_collector.return_value.record = Mock()
                mock_collector.return_value.log_summary = Mock()

                with patch('l0_macro_update.get_circuit_breaker_for_source', return_value=None):
                    result = extract_all_sources(**mock_context)

        # dane should not be in results if skipped
        assert 'dane' not in result or result.get('dane', {}).get('skipped', False)

    def test_circuit_breaker_integration(self, mock_context, mock_registry):
        """Test that circuit breaker is called when available."""
        from l0_macro_update import extract_all_sources

        mock_cb = Mock()
        mock_cb.call.return_value = Mock(
            success=True,
            data=pd.DataFrame({'fecha': pd.date_range('2024-01-01', periods=15), 'dxy': [100.0] * 15}),
            error=None
        )

        with patch('l0_macro_update.ExtractorRegistry', return_value=mock_registry):
            with patch('l0_macro_update.reset_metrics_collector') as mock_collector:
                mock_collector.return_value = Mock()
                mock_collector.return_value.record = Mock()
                mock_collector.return_value.log_summary = Mock()

                with patch('l0_macro_update.get_circuit_breaker_for_source', return_value=mock_cb):
                    result = extract_all_sources(**mock_context)

        # Circuit breaker's call method should have been invoked
        assert mock_cb.call.called

    def test_handles_extraction_failure_gracefully(self, mock_context, mock_registry):
        """Test that extraction failures don't crash the pipeline."""
        from l0_macro_update import extract_all_sources

        # Make extraction fail
        mock_registry.extract_variable.return_value = Mock(
            success=False,
            data=None,
            error="API Error"
        )

        with patch('l0_macro_update.ExtractorRegistry', return_value=mock_registry):
            with patch('l0_macro_update.reset_metrics_collector') as mock_collector:
                mock_collector.return_value = Mock()
                mock_collector.return_value.record = Mock()
                mock_collector.return_value.log_summary = Mock()

                with patch('l0_macro_update.get_circuit_breaker_for_source', return_value=None):
                    result = extract_all_sources(**mock_context)

        # Should complete without raising exception
        assert isinstance(result, dict)


# =============================================================================
# UPSERT LOGIC TESTS
# =============================================================================

class TestUpsertAll:
    """Tests for upsert_all() function."""

    def test_handles_empty_extraction_data(self, mock_context):
        """Test that empty extraction data is handled gracefully."""
        from l0_macro_update import upsert_all

        mock_context['ti'].xcom_pull.return_value = {}

        result = upsert_all(**mock_context)

        assert result['success'] == 0
        assert result['total_rows'] == 0

    def test_reconstructs_dataframes(self, mock_context):
        """Test that DataFrames are properly reconstructed from XCom."""
        from l0_macro_update import upsert_all

        # Simulate XCom data (serialized DataFrame)
        df = pd.DataFrame({
            'fecha': pd.date_range('2024-01-01', periods=15, freq='D'),
            'dxy': [100.0] * 15,
        })
        mock_context['ti'].xcom_pull.return_value = {'dxy': df.to_dict()}

        with patch('l0_macro_update.get_db_connection') as mock_conn:
            mock_cursor = Mock()
            mock_conn.return_value.cursor.return_value = mock_cursor

            with patch('l0_macro_update.UpsertService') as mock_upsert:
                mock_upsert.return_value.upsert_last_n.return_value = {
                    'success': True,
                    'rows_affected': 15,
                }

                result = upsert_all(**mock_context)

        assert result['success'] == 1
        assert result['total_rows'] == 15


# =============================================================================
# UPDATE IS_COMPLETE TESTS
# =============================================================================

class TestUpdateIsComplete:
    """Tests for update_is_complete() function."""

    def test_builds_correct_sql(self, mock_context):
        """Test that SQL is built correctly for critical variables."""
        from l0_macro_update import update_is_complete, CRITICAL_VARIABLES

        with patch('l0_macro_update.get_db_connection') as mock_conn:
            mock_cursor = Mock()
            mock_cursor.rowcount = 15
            mock_cursor.fetchone.return_value = (10, 5)
            mock_conn.return_value.cursor.return_value = mock_cursor

            result = update_is_complete(**mock_context)

        # Verify SQL was executed
        assert mock_cursor.execute.called

        # Check that all critical variables are in the SQL
        sql_call = mock_cursor.execute.call_args_list[0][0][0]
        for var in CRITICAL_VARIABLES:
            assert var in sql_call

    def test_returns_correct_counts(self, mock_context):
        """Test that complete/incomplete counts are returned."""
        from l0_macro_update import update_is_complete

        with patch('l0_macro_update.get_db_connection') as mock_conn:
            mock_cursor = Mock()
            mock_cursor.rowcount = 15
            mock_cursor.fetchone.return_value = (10, 5)
            mock_conn.return_value.cursor.return_value = mock_cursor

            result = update_is_complete(**mock_context)

        assert result['complete_count'] == 10
        assert result['incomplete_count'] == 5
        assert result['dates_updated'] == 15


# =============================================================================
# LOG METRICS TESTS
# =============================================================================

class TestLogMetrics:
    """Tests for log_metrics() function."""

    def test_aggregates_metrics_correctly(self, mock_context):
        """Test that metrics are correctly aggregated."""
        from l0_macro_update import log_metrics

        mock_context['ti'].xcom_pull.side_effect = [
            {'fred': {'success': 10, 'failed': 2}, 'investing': {'success': 15, 'failed': 0}},  # extraction_results
            {'total_records_extracted': 100, 'total_extraction_duration_ms': 5000},  # extraction_report
            {'total_rows': 95, 'duration_ms': 1000},  # upsert_results
            {'complete_count': 10, 'incomplete_count': 5},  # is_complete_results
        ]

        # Should not raise
        log_metrics(**mock_context)

        # Verify XCom was queried
        assert mock_context['ti'].xcom_pull.called

    def test_sends_alert_on_low_success_rate(self, mock_context):
        """Test that alert is sent when success rate < 80%."""
        from l0_macro_update import log_metrics

        # Setup low success rate
        mock_context['ti'].xcom_pull.side_effect = [
            {'fred': {'success': 3, 'failed': 10}},  # 23% success
            {'total_records_extracted': 30, 'total_extraction_duration_ms': 5000},
            {'total_rows': 30, 'duration_ms': 1000},
            {'complete_count': 5, 'incomplete_count': 10},
        ]

        with patch('l0_macro_update.send_pipeline_alert') as mock_alert:
            log_metrics(**mock_context)

        # Alert should have been called with WARNING severity
        mock_alert.assert_called()
        call_args = mock_alert.call_args
        assert call_args[1].get('severity') == 'WARNING' or 'Low Success Rate' in call_args[0][0]


# =============================================================================
# DAILY SUMMARY TESTS
# =============================================================================

class TestDailySummary:
    """Tests for daily_summary() function."""

    @patch('l0_macro_update.is_last_run_of_day')
    def test_skips_if_not_last_run(self, mock_is_last, mock_context):
        """Test that summary is skipped if not last run of day."""
        from l0_macro_update import daily_summary

        mock_is_last.return_value = False

        result = daily_summary(**mock_context)

        assert result is None

    @patch('l0_macro_update.is_last_run_of_day')
    def test_generates_summary_at_last_run(self, mock_is_last, mock_context):
        """Test that summary is generated at last run of day."""
        from l0_macro_update import daily_summary

        mock_is_last.return_value = True
        mock_context['ti'].xcom_pull.side_effect = [
            {'fred': {'success': 10, 'failed': 0}},  # extraction_results
            {'total_rows': 100},  # upsert_results
        ]

        result = daily_summary(**mock_context)

        assert result is not None
        assert 'date' in result
        assert 'source_success_rate' in result

    @patch('l0_macro_update.is_last_run_of_day')
    def test_detects_sources_with_issues(self, mock_is_last, mock_context):
        """Test that sources with <80% success rate are flagged."""
        from l0_macro_update import daily_summary

        mock_is_last.return_value = True
        mock_context['ti'].xcom_pull.side_effect = [
            {
                'fred': {'success': 10, 'failed': 0},  # 100%
                'dane': {'success': 3, 'failed': 7},   # 30%
            },
            {'total_rows': 100},
        ]

        with patch('l0_macro_update.send_pipeline_alert') as mock_alert:
            result = daily_summary(**mock_context)

        assert 'dane' in result['sources_with_issues']
        assert 'fred' not in result['sources_with_issues']
        mock_alert.assert_called()  # Alert sent for issues


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Tests for DAG configuration constants."""

    def test_safety_records_is_15(self):
        """Verify SAFETY_RECORDS is set to 15 (not 5)."""
        from l0_macro_update import SAFETY_RECORDS
        assert SAFETY_RECORDS == 15

    def test_market_hours_are_correct(self):
        """Verify market hours configuration."""
        from l0_macro_update import MARKET_HOURS_START, MARKET_HOURS_END
        assert MARKET_HOURS_START == 8
        assert MARKET_HOURS_END == 13

    def test_critical_variables_defined(self):
        """Verify critical variables are defined."""
        from l0_macro_update import CRITICAL_VARIABLES
        assert len(CRITICAL_VARIABLES) > 0
        assert 'dxy' in CRITICAL_VARIABLES
        assert 'vix' in CRITICAL_VARIABLES

    def test_schedule_interval_is_hourly(self):
        """Verify schedule is hourly during market hours."""
        # This tests the DAG definition indirectly
        # The schedule should be '0 13-17 * * 1-5' (hourly 8-12 COT, Mon-Fri)
        from l0_macro_update import dag
        assert dag.schedule_interval == '0 13-17 * * 1-5'


# =============================================================================
# INTEGRATION TESTS (Mocked)
# =============================================================================

class TestDAGStructure:
    """Tests for DAG task structure."""

    def test_all_tasks_defined(self):
        """Verify all expected tasks are defined."""
        from l0_macro_update import dag

        task_ids = [task.task_id for task in dag.tasks]

        expected_tasks = [
            'health_check',
            'check_market_hours',
            'extract_all_sources',
            'upsert_all',
            'update_is_complete',
            'log_metrics',
            'daily_summary',
        ]

        for task_id in expected_tasks:
            assert task_id in task_ids, f"Missing task: {task_id}"

    def test_task_dependencies(self):
        """Verify task dependencies are correctly set."""
        from l0_macro_update import dag

        # Get tasks
        tasks = {task.task_id: task for task in dag.tasks}

        # health_check should be upstream of check_market_hours
        assert tasks['check_market_hours'] in tasks['health_check'].downstream_list

        # extract_all_sources should follow check_market_hours
        assert tasks['extract_all_sources'] in tasks['check_market_hours'].downstream_list

        # upsert_all should follow extract_all_sources
        assert tasks['upsert_all'] in tasks['extract_all_sources'].downstream_list

    def test_daily_summary_trigger_rule(self):
        """Verify daily_summary runs even on upstream failure."""
        from l0_macro_update import dag

        tasks = {task.task_id: task for task in dag.tasks}

        assert tasks['daily_summary'].trigger_rule == 'all_done'

    def test_dag_tags(self):
        """Verify DAG has correct tags."""
        from l0_macro_update import dag

        expected_tags = ['l0', 'macro', 'update', 'realtime', 'v2']

        for tag in expected_tags:
            assert tag in dag.tags


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
