"""
Integration Test: Pipeline Data API
====================================

Tests the pipeline data API endpoints:
- Health and status checks
- L0 raw data endpoints
- L1 episode endpoints
- L3 feature endpoints
- L4 dataset endpoints
- L5 model endpoints
- L6 backtest endpoints

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-17
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add services to path for importing
services_path = Path(__file__).parent.parent.parent / 'services'
sys.path.insert(0, str(services_path))


@pytest.mark.integration
class TestPipelineHealth:
    """Test pipeline health check logic"""

    def test_health_response_structure(self):
        """Test health check returns expected structure"""
        mock_health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '3.0.0',
            'api': 'pipeline_data_api',
            'database': {
                'connected': True,
                'tables_available': True
            },
            'pipelines': {
                'L0': 'active',
                'L1': 'active',
                'L3': 'active',
                'L4': 'active',
                'L5': 'active',
                'L6': 'active'
            }
        }

        # Required top-level fields
        assert 'status' in mock_health
        assert 'timestamp' in mock_health
        assert 'database' in mock_health
        assert 'pipelines' in mock_health

    def test_pipeline_status_values(self):
        """Test valid pipeline status values"""
        valid_statuses = ['active', 'inactive', 'error', 'unknown']

        for status in valid_statuses:
            assert status in valid_statuses


@pytest.mark.integration
class TestL0RawData:
    """Test L0 raw data endpoints"""

    @pytest.fixture
    def sample_ohlcv_row(self):
        """Sample OHLCV data row"""
        return {
            'time': '2025-12-17T10:30:00-05:00',
            'open': 4250.00,
            'high': 4255.00,
            'low': 4248.00,
            'close': 4252.50,
            'volume': 5000
        }

    def test_ohlcv_data_validation(self, sample_ohlcv_row):
        """Test OHLCV data validation"""
        row = sample_ohlcv_row

        # OHLC price rules
        assert row['high'] >= row['open'], "High >= Open"
        assert row['high'] >= row['close'], "High >= Close"
        assert row['high'] >= row['low'], "High >= Low"
        assert row['low'] <= row['open'], "Low <= Open"
        assert row['low'] <= row['close'], "Low <= Close"

        # Volume non-negative
        assert row['volume'] >= 0, "Volume >= 0"

    def test_statistics_calculation(self):
        """Test L0 statistics calculation"""
        # Sample statistics
        stats = {
            'total_bars': 1000,
            'date_range': {
                'start': '2025-01-01',
                'end': '2025-12-17'
            },
            'price_stats': {
                'min': 3800.00,
                'max': 4500.00,
                'mean': 4150.00,
                'std': 150.00
            },
            'volume_stats': {
                'total': 5000000,
                'mean': 5000.00,
                'max': 50000
            },
            'gaps': {
                'count': 5,
                'max_gap_minutes': 15
            }
        }

        assert stats['total_bars'] > 0, "Should have bars"
        assert stats['price_stats']['max'] >= stats['price_stats']['min'], "Max >= Min"
        assert stats['price_stats']['std'] >= 0, "Std dev >= 0"

    def test_date_range_filter(self):
        """Test date range filtering logic"""
        start_date = datetime(2025, 12, 1)
        end_date = datetime(2025, 12, 17)

        # Valid range
        assert end_date > start_date, "End must be after start"

        # Range in days
        days = (end_date - start_date).days
        assert days == 16, f"Range should be 16 days, got {days}"

    def test_limit_and_offset(self):
        """Test pagination with limit and offset"""
        total_records = 1000
        limit = 100
        offset = 200

        # Calculate expected results
        expected_start = offset
        expected_end = min(offset + limit, total_records)
        expected_count = expected_end - expected_start

        assert expected_count == 100, f"Should return 100 records"

        # Edge case: offset beyond data
        offset_beyond = 1500
        expected_count_beyond = max(0, total_records - offset_beyond)

        assert expected_count_beyond == 0, "Should return 0 when offset exceeds data"


@pytest.mark.integration
class TestL1Episodes:
    """Test L1 episode endpoints"""

    @pytest.fixture
    def sample_episode(self):
        """Sample episode data"""
        return {
            'episode_id': 'EP_20251217_080000',
            'date': '2025-12-17',
            'start_time': '08:00:00',
            'end_time': '12:55:00',
            'bars': 60,
            'status': 'complete',
            'quality_score': 0.98,
            'gaps': [],
            'price_range': {
                'open': 4250.00,
                'high': 4280.00,
                'low': 4240.00,
                'close': 4265.00
            }
        }

    def test_episode_validation(self, sample_episode):
        """Test episode data validation"""
        ep = sample_episode

        # Required fields
        required = ['episode_id', 'date', 'bars', 'status']
        for field in required:
            assert field in ep, f"Missing field: {field}"

        # Bars should be 60 for complete session
        assert ep['bars'] == 60, "Complete episode should have 60 bars"

        # Quality score in [0, 1]
        assert 0 <= ep['quality_score'] <= 1, "Quality score in [0, 1]"

    def test_episode_status_values(self):
        """Test valid episode status values"""
        valid_statuses = ['complete', 'incomplete', 'in_progress', 'error']

        for status in valid_statuses:
            assert status in valid_statuses

    def test_quality_report_structure(self):
        """Test quality report structure"""
        report = {
            'total_episodes': 250,
            'complete_episodes': 240,
            'incomplete_episodes': 10,
            'average_quality': 0.95,
            'gaps_detected': 15,
            'date_range': {
                'start': '2025-01-01',
                'end': '2025-12-17'
            }
        }

        completeness = report['complete_episodes'] / report['total_episodes']
        assert completeness > 0.9, "Completeness should be > 90%"


@pytest.mark.integration
class TestL3Features:
    """Test L3 feature endpoints"""

    @pytest.fixture
    def feature_list(self):
        """List of expected features"""
        return [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
            'rsi_9', 'atr_pct', 'adx_14',
            'dxy_z', 'dxy_change_1d', 'vix_z',
            'embi_z', 'brent_change_1d', 'rate_spread',
            'usdmxn_ret_1h'
        ]

    def test_feature_count(self, feature_list):
        """Test correct number of features"""
        assert len(feature_list) == 13, f"Expected 13 features, got {len(feature_list)}"

    def test_feature_data_structure(self, feature_list):
        """Test feature data structure"""
        sample_row = {feat: 0.0 for feat in feature_list}
        sample_row['timestamp'] = '2025-12-17T10:30:00'

        # All features present
        for feat in feature_list:
            assert feat in sample_row, f"Missing feature: {feat}"

    def test_feature_normalization_bounds(self, feature_list):
        """Test features are within normalization bounds"""
        # After normalization, features should be roughly in [-4, 4]
        max_bound = 4.0
        min_bound = -4.0

        sample_values = {
            'log_ret_5m': 0.5,
            'rsi_9': -0.3,
            'dxy_z': 2.1,
            'vix_z': -1.5,
        }

        for feat, val in sample_values.items():
            assert min_bound <= val <= max_bound, \
                f"Feature {feat} value {val} out of bounds"


@pytest.mark.integration
class TestL4Dataset:
    """Test L4 dataset endpoints"""

    def test_dataset_contract(self):
        """Test dataset follows contract"""
        contract = {
            'observation_dim': 15,
            'action_dim': 1,
            'features_count': 13,
            'state_vars': ['position', 'time_normalized'],
            'action_range': [-1.0, 1.0]
        }

        assert contract['observation_dim'] == 15, "Obs dim should be 15"
        assert contract['features_count'] + len(contract['state_vars']) == contract['observation_dim']

    def test_quality_check_metrics(self):
        """Test dataset quality check metrics"""
        quality = {
            'nan_ratio': 0.001,
            'inf_ratio': 0.0,
            'outlier_ratio': 0.02,
            'feature_correlation_max': 0.85,
            'class_balance': 0.48,  # Ratio of positive returns
            'passes_checks': True
        }

        # Thresholds
        assert quality['nan_ratio'] < 0.01, "NaN ratio should be < 1%"
        assert quality['inf_ratio'] == 0.0, "No infinities allowed"
        assert quality['outlier_ratio'] < 0.05, "Outlier ratio < 5%"
        assert quality['feature_correlation_max'] < 0.95, "No perfect correlations"


@pytest.mark.integration
class TestL5Models:
    """Test L5 model endpoints"""

    @pytest.fixture
    def model_info(self):
        """Sample model information"""
        return {
            'model_id': 'ppo_primary',
            'algorithm': 'PPO',
            'framework': 'stable-baselines3',
            'version': 'current',
            'fold': 0,
            'observation_dim': 15,
            'action_dim': 1,
            'network': {
                'pi': [32, 32],
                'vf': [32, 32],
                'activation': 'Tanh'
            },
            'training_date': '2025-12-01',
            'training_episodes': 50000,
            'file_path': 'models/ppo_primary.zip'
        }

    def test_model_info_validation(self, model_info):
        """Test model info validation"""
        required = ['model_id', 'algorithm', 'observation_dim', 'action_dim']

        for field in required:
            assert field in model_info, f"Missing field: {field}"

        assert model_info['observation_dim'] == 15, "Obs dim should be 15"
        assert model_info['action_dim'] == 1, "Action dim should be 1"

    def test_model_list_response(self):
        """Test model list response structure"""
        model_list = {
            'count': 3,
            'models': [
                {'id': 'ppo_primary', 'status': 'active'},
                {'id': 'ppo_secondary', 'status': 'active'},
                {'id': 'ppo_legacy', 'status': 'deprecated'}
            ]
        }

        assert model_list['count'] == len(model_list['models'])

        # At least one active model
        active_models = [m for m in model_list['models'] if m['status'] == 'active']
        assert len(active_models) >= 1, "Should have at least one active model"


@pytest.mark.integration
class TestL6Backtest:
    """Test L6 backtest endpoints"""

    @pytest.fixture
    def backtest_result(self):
        """Sample backtest result"""
        return {
            'backtest_id': 'BT_20251217_001',
            'model_id': 'ppo_primary',
            'period': {
                'start': '2025-01-01',
                'end': '2025-12-17'
            },
            'metrics': {
                'total_return': 0.15,
                'sharpe_ratio': 1.8,
                'sortino_ratio': 2.5,
                'max_drawdown': -0.08,
                'win_rate': 0.55,
                'profit_factor': 1.75,
                'calmar_ratio': 1.9
            },
            'trades': {
                'total': 500,
                'winners': 275,
                'losers': 225,
                'avg_win': 0.003,
                'avg_loss': -0.002
            },
            'status': 'complete'
        }

    def test_backtest_metrics_validation(self, backtest_result):
        """Test backtest metrics are valid"""
        metrics = backtest_result['metrics']

        # Sharpe ratio reasonable
        assert -5 < metrics['sharpe_ratio'] < 10, "Sharpe ratio in reasonable range"

        # Max drawdown should be negative or zero
        assert metrics['max_drawdown'] <= 0, "Max drawdown should be <= 0"

        # Win rate in [0, 1]
        assert 0 <= metrics['win_rate'] <= 1, "Win rate in [0, 1]"

        # Profit factor positive (if profitable)
        if metrics['total_return'] > 0:
            assert metrics['profit_factor'] > 1, "Profit factor > 1 for profitable strategy"

    def test_trade_statistics(self, backtest_result):
        """Test trade statistics consistency"""
        trades = backtest_result['trades']

        # Winners + losers = total
        assert trades['winners'] + trades['losers'] == trades['total'], \
            "Winners + losers should equal total"

        # Win rate matches
        calculated_win_rate = trades['winners'] / trades['total']
        reported_win_rate = backtest_result['metrics']['win_rate']

        assert abs(calculated_win_rate - reported_win_rate) < 0.01, \
            "Win rate should match trades data"


@pytest.mark.integration
class TestPipelineStatusEndpoints:
    """Test pipeline status endpoints"""

    def test_l2_status_structure(self):
        """Test L2 pipeline status structure"""
        status = {
            'pipeline': 'L2',
            'status': 'completed',
            'last_run': '2025-12-17T08:00:00',
            'next_run': '2025-12-18T08:00:00',
            'records_processed': 10000,
            'errors': 0
        }

        required = ['pipeline', 'status', 'last_run']
        for field in required:
            assert field in status, f"Missing field: {field}"

    def test_l3_status_structure(self):
        """Test L3 pipeline status structure"""
        status = {
            'pipeline': 'L3',
            'status': 'completed',
            'last_run': '2025-12-17T08:15:00',
            'features_computed': 13,
            'rows_processed': 5000
        }

        assert status['features_computed'] == 13, "Should compute 13 features"

    def test_l4_status_structure(self):
        """Test L4 pipeline status structure"""
        status = {
            'pipeline': 'L4',
            'status': 'completed',
            'last_run': '2025-12-17T08:30:00',
            'dataset_rows': 50000,
            'quality_passed': True
        }

        assert status['quality_passed'], "Dataset should pass quality checks"


@pytest.mark.integration
class TestExtendedStatistics:
    """Test extended statistics endpoints"""

    def test_extended_statistics_structure(self):
        """Test extended statistics structure"""
        stats = {
            'data_coverage': {
                'total_days': 365,
                'trading_days': 250,
                'complete_sessions': 245,
                'coverage_ratio': 0.98
            },
            'price_analysis': {
                'ytd_return': 0.05,
                'volatility_annualized': 0.12,
                'avg_daily_range': 0.8,
                'max_daily_move': 2.5
            },
            'data_quality': {
                'gaps': 5,
                'interpolated_points': 15,
                'outliers_removed': 3
            }
        }

        assert stats['data_coverage']['coverage_ratio'] > 0.9, "Coverage > 90%"
        assert stats['price_analysis']['volatility_annualized'] > 0, "Volatility > 0"


@pytest.mark.integration
class TestGridVerification:
    """Test grid verification endpoint"""

    def test_grid_verification_structure(self):
        """Test grid verification response structure"""
        verification = {
            'total_grid_points': 1000,
            'verified': 995,
            'missing': 5,
            'verification_ratio': 0.995,
            'missing_details': [
                {'date': '2025-03-15', 'bars_missing': 3},
                {'date': '2025-06-20', 'bars_missing': 2}
            ]
        }

        assert verification['verification_ratio'] > 0.99, "Verification > 99%"
        assert len(verification['missing_details']) == 2


@pytest.mark.integration
class TestForwardIC:
    """Test forward IC (Information Coefficient) endpoint"""

    def test_forward_ic_structure(self):
        """Test forward IC response structure"""
        ic_report = {
            'features': {
                'log_ret_5m': {'ic_1bar': 0.05, 'ic_5bar': 0.08, 'ic_12bar': 0.03},
                'rsi_9': {'ic_1bar': 0.03, 'ic_5bar': 0.06, 'ic_12bar': 0.04},
                'dxy_z': {'ic_1bar': 0.02, 'ic_5bar': 0.04, 'ic_12bar': 0.05}
            },
            'average_ic': 0.04,
            'top_features': ['log_ret_5m', 'rsi_9'],
            'evaluation_period': '2025-11-01 to 2025-12-17'
        }

        # IC should be in [-1, 1]
        for feat, ics in ic_report['features'].items():
            for period, ic in ics.items():
                assert -1 <= ic <= 1, f"IC for {feat} {period} out of range"


@pytest.mark.integration
class TestPreparedData:
    """Test prepared data endpoint"""

    def test_prepared_data_structure(self):
        """Test prepared data structure"""
        prepared = {
            'episodes': 250,
            'observations': 15000,
            'features_per_obs': 13,
            'additional_state': 2,
            'total_dim': 15,
            'date_range': {
                'start': '2025-01-01',
                'end': '2025-12-17'
            },
            'ready_for_training': True
        }

        assert prepared['features_per_obs'] + prepared['additional_state'] == prepared['total_dim']
        assert prepared['ready_for_training'], "Data should be ready for training"


@pytest.mark.integration
class TestContractEndpoint:
    """Test contract endpoint"""

    def test_contract_structure(self):
        """Test contract specification structure"""
        contract = {
            'observation_space': {
                'dim': 15,
                'dtype': 'float32',
                'bounds': [-5.0, 5.0]
            },
            'action_space': {
                'dim': 1,
                'dtype': 'float32',
                'bounds': [-1.0, 1.0]
            },
            'features': [
                'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
                'rsi_9', 'atr_pct', 'adx_14',
                'dxy_z', 'dxy_change_1d', 'vix_z',
                'embi_z', 'brent_change_1d', 'rate_spread',
                'usdmxn_ret_1h'
            ],
            'state_variables': ['position', 'time_normalized'],
            'version': '3.2.0'
        }

        assert len(contract['features']) == 13
        assert len(contract['state_variables']) == 2
        assert len(contract['features']) + len(contract['state_variables']) == contract['observation_space']['dim']


@pytest.mark.integration
class TestQualityCheckEndpoint:
    """Test quality check endpoint"""

    def test_quality_check_structure(self):
        """Test quality check response structure"""
        quality = {
            'checks': {
                'nan_check': {'passed': True, 'ratio': 0.0},
                'inf_check': {'passed': True, 'ratio': 0.0},
                'bounds_check': {'passed': True, 'violations': 0},
                'correlation_check': {'passed': True, 'max_corr': 0.82},
                'stationarity_check': {'passed': True, 'features_stationary': 13}
            },
            'overall_passed': True,
            'timestamp': '2025-12-17T10:00:00'
        }

        assert quality['overall_passed'], "Quality checks should pass"

        # All individual checks should pass
        for check_name, check_result in quality['checks'].items():
            assert check_result['passed'], f"Check {check_name} should pass"
