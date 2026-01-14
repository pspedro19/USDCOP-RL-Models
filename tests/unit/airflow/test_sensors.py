"""
Tests for Airflow Custom Sensors
================================

Tests the event-driven sensors that replace fixed schedule patterns.
CLAUDE-T17 | Plan Item: P2-SENSORS

These tests use mocking to work in any environment without Airflow installed.
In production, sensors run inside Docker where Airflow is installed.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "airflow" / "dags"))

# Mock Airflow modules BEFORE importing sensors
# This allows tests to run without actual Airflow installed
mock_base_sensor = MagicMock()
mock_base_sensor.BaseSensorOperator = MagicMock

mock_decorators = MagicMock()
mock_decorators.apply_defaults = lambda f: f  # No-op decorator

sys.modules['airflow'] = MagicMock()
sys.modules['airflow.sensors'] = MagicMock()
sys.modules['airflow.sensors.base'] = mock_base_sensor
sys.modules['airflow.utils'] = MagicMock()
sys.modules['airflow.utils.decorators'] = mock_decorators

# Mock the utils.dag_common module to avoid POSTGRES_PASSWORD requirement
mock_dag_common = MagicMock()
mock_dag_common.get_db_connection = MagicMock(return_value=MagicMock())
mock_dag_common.DB_CONFIG = {'host': 'localhost', 'port': 5432, 'database': 'test'}
sys.modules['utils'] = MagicMock()
sys.modules['utils.dag_common'] = mock_dag_common


class TestNewOHLCVBarSensor:
    """Tests for NewOHLCVBarSensor."""

    @pytest.fixture
    def mock_db_connection(self):
        """Create mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn, mock_cursor

    @pytest.fixture
    def sensor(self):
        """Create sensor instance."""
        # Import after mocking
        from sensors.new_bar_sensor import NewOHLCVBarSensor

        sensor = NewOHLCVBarSensor(
            task_id='test_ohlcv_sensor',
            table_name='usdcop_m5_ohlcv',
            symbol='USD/COP',
            max_staleness_minutes=10,
            poke_interval=30,
            timeout=300,
        )
        return sensor

    def test_sensor_initialization(self, sensor):
        """Sensor should initialize with correct parameters."""
        assert sensor.table_name == 'usdcop_m5_ohlcv'
        assert sensor.symbol == 'USD/COP'
        assert sensor.max_staleness_minutes == 10

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_poke_returns_true_when_fresh_data(self, mock_get_conn, sensor, mock_db_connection):
        """poke() should return True when fresh data is available."""
        mock_conn, mock_cursor = mock_db_connection
        mock_get_conn.return_value = mock_conn

        # Simulate fresh data (1 minute old)
        fresh_time = datetime.utcnow() - timedelta(minutes=1)
        mock_cursor.fetchone.return_value = (fresh_time,)

        # Mock context
        context = {'ti': MagicMock()}
        context['ti'].xcom_pull.return_value = None

        result = sensor.poke(context)

        assert result is True
        mock_cursor.execute.assert_called()

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_poke_returns_false_when_stale_data(self, mock_get_conn, sensor, mock_db_connection):
        """poke() should return False when data is stale."""
        mock_conn, mock_cursor = mock_db_connection
        mock_get_conn.return_value = mock_conn

        # Simulate stale data (15 minutes old)
        stale_time = datetime.utcnow() - timedelta(minutes=15)
        mock_cursor.fetchone.return_value = (stale_time,)

        context = {'ti': MagicMock()}
        context['ti'].xcom_pull.return_value = None

        result = sensor.poke(context)

        assert result is False

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_poke_returns_false_when_no_data(self, mock_get_conn, sensor, mock_db_connection):
        """poke() should return False when no data exists."""
        mock_conn, mock_cursor = mock_db_connection
        mock_get_conn.return_value = mock_conn

        mock_cursor.fetchone.return_value = (None,)

        context = {'ti': MagicMock()}
        context['ti'].xcom_pull.return_value = None

        result = sensor.poke(context)

        assert result is False

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_poke_detects_new_data_since_last_process(self, mock_get_conn, sensor, mock_db_connection):
        """poke() should return False if data hasn't changed since last check."""
        mock_conn, mock_cursor = mock_db_connection
        mock_get_conn.return_value = mock_conn

        # Same timestamp as last processed
        same_time = datetime.utcnow() - timedelta(minutes=2)
        mock_cursor.fetchone.return_value = (same_time,)

        context = {'ti': MagicMock()}
        context['ti'].xcom_pull.return_value = same_time.isoformat()

        result = sensor.poke(context)

        assert result is False

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_poke_pushes_detected_time_to_xcom(self, mock_get_conn, sensor, mock_db_connection):
        """poke() should push detected time to XCom when new data found."""
        mock_conn, mock_cursor = mock_db_connection
        mock_get_conn.return_value = mock_conn

        fresh_time = datetime.utcnow() - timedelta(minutes=1)
        mock_cursor.fetchone.return_value = (fresh_time,)

        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = None
        context = {'ti': mock_ti}

        sensor.poke(context)

        mock_ti.xcom_push.assert_called_once()
        call_args = mock_ti.xcom_push.call_args
        assert call_args[1]['key'] == 'detected_ohlcv_time'


class TestNewFeatureBarSensor:
    """Tests for NewFeatureBarSensor."""

    @pytest.fixture
    def mock_db_connection(self):
        """Create mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn, mock_cursor

    @pytest.fixture
    def sensor(self):
        """Create sensor instance."""
        from sensors.new_bar_sensor import NewFeatureBarSensor

        return NewFeatureBarSensor(
            task_id='test_feature_sensor',
            table_name='inference_features_5m',
            require_complete=True,
            max_staleness_minutes=10,
            poke_interval=30,
            timeout=300,
        )

    def test_sensor_initialization(self, sensor):
        """Sensor should initialize with correct parameters."""
        assert sensor.table_name == 'inference_features_5m'
        assert sensor.require_complete is True
        assert sensor.max_staleness_minutes == 10
        assert len(sensor.critical_features) == 6

    def test_default_critical_features(self, sensor):
        """Sensor should have expected critical features."""
        expected = [
            'log_ret_5m', 'log_ret_1h', 'rsi_9',
            'dxy_z', 'vix_z', 'rate_spread'
        ]
        assert sensor.critical_features == expected

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_poke_returns_true_when_complete_features(self, mock_get_conn, sensor, mock_db_connection):
        """poke() should return True when complete features available."""
        mock_conn, mock_cursor = mock_db_connection
        mock_get_conn.return_value = mock_conn

        fresh_time = datetime.utcnow() - timedelta(minutes=1)
        mock_cursor.fetchone.return_value = (fresh_time,)

        context = {'ti': MagicMock()}
        context['ti'].xcom_pull.return_value = None

        result = sensor.poke(context)

        assert result is True

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_poke_checks_completeness_in_query(self, mock_get_conn, sensor, mock_db_connection):
        """poke() should include completeness checks in SQL when require_complete=True."""
        mock_conn, mock_cursor = mock_db_connection
        mock_get_conn.return_value = mock_conn

        fresh_time = datetime.utcnow() - timedelta(minutes=1)
        mock_cursor.fetchone.return_value = (fresh_time,)

        context = {'ti': MagicMock()}
        context['ti'].xcom_pull.return_value = None

        sensor.poke(context)

        # Verify SQL contains IS NOT NULL checks
        call_args = mock_cursor.execute.call_args[0][0]
        assert 'IS NOT NULL' in call_args
        assert 'log_ret_5m' in call_args


class TestDataFreshnessGuard:
    """Tests for DataFreshnessGuard utility."""

    @pytest.fixture
    def guard(self):
        """Create guard instance."""
        from sensors.new_bar_sensor import DataFreshnessGuard
        return DataFreshnessGuard(max_staleness_minutes=10)

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_is_ohlcv_fresh_returns_true(self, mock_get_conn, guard):
        """is_ohlcv_fresh() should return True for fresh data."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        fresh_time = datetime.utcnow() - timedelta(minutes=5)
        mock_cursor.fetchone.return_value = (fresh_time,)

        result = guard.is_ohlcv_fresh()

        assert result is True

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_is_ohlcv_fresh_returns_false(self, mock_get_conn, guard):
        """is_ohlcv_fresh() should return False for stale data."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        stale_time = datetime.utcnow() - timedelta(minutes=20)
        mock_cursor.fetchone.return_value = (stale_time,)

        result = guard.is_ohlcv_fresh()

        assert result is False

    @patch('sensors.new_bar_sensor.get_db_connection')
    def test_get_data_lag_returns_lag_info(self, mock_get_conn, guard):
        """get_data_lag() should return lag information for all sources."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        now = datetime.utcnow()
        mock_cursor.fetchone.side_effect = [
            (now - timedelta(minutes=2),),  # OHLCV
            (now - timedelta(minutes=3),),  # Features
            (now - timedelta(minutes=5),),  # Signals
        ]

        result = guard.get_data_lag()

        assert 'ohlcv_lag_minutes' in result
        assert 'features_lag_minutes' in result
        assert 'signals_lag_minutes' in result
        assert result['ohlcv_lag_minutes'] == pytest.approx(2, abs=0.1)


class TestSensorIntegration:
    """Integration tests for sensor behavior."""

    def test_sensor_imports_correctly(self):
        """Sensors should import without errors."""
        from sensors.new_bar_sensor import (
            NewOHLCVBarSensor,
            NewFeatureBarSensor,
            DataFreshnessGuard,
        )

        assert NewOHLCVBarSensor is not None
        assert NewFeatureBarSensor is not None
        assert DataFreshnessGuard is not None

    def test_sensor_package_init(self):
        """Sensor package __init__ should export classes."""
        from sensors import NewOHLCVBarSensor, NewFeatureBarSensor

        assert NewOHLCVBarSensor is not None
        assert NewFeatureBarSensor is not None

    def test_sensor_has_poke_method(self):
        """All sensors must implement poke() method."""
        from sensors.new_bar_sensor import NewOHLCVBarSensor, NewFeatureBarSensor

        sensor1 = NewOHLCVBarSensor(task_id='test1')
        sensor2 = NewFeatureBarSensor(task_id='test2')

        assert hasattr(sensor1, 'poke')
        assert callable(sensor1.poke)
        assert hasattr(sensor2, 'poke')
        assert callable(sensor2.poke)

    def test_sensors_have_required_attributes(self):
        """Sensors should have required attributes."""
        from sensors.new_bar_sensor import NewOHLCVBarSensor, NewFeatureBarSensor

        sensor1 = NewOHLCVBarSensor(task_id='test1')
        sensor2 = NewFeatureBarSensor(task_id='test2')

        # OHLCV sensor attributes
        assert hasattr(sensor1, 'table_name')
        assert hasattr(sensor1, 'symbol')
        assert hasattr(sensor1, 'max_staleness_minutes')

        # Feature sensor attributes
        assert hasattr(sensor2, 'table_name')
        assert hasattr(sensor2, 'require_complete')
        assert hasattr(sensor2, 'critical_features')
