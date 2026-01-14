"""
Unit Tests for Config Loader
==============================

Tests configuration loading and validation:
- Config file parsing
- Required sections present
- Value types correct
- Trading parameters validated

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-16
"""

import pytest
import json
from pathlib import Path


@pytest.mark.unit
class TestConfigLoading:
    """Test configuration file loading"""

    def test_config_file_readable(self, feature_config):
        """Test config file can be read and parsed as JSON"""
        assert feature_config is not None, "Config is None"
        assert isinstance(feature_config, dict), "Config should be a dictionary"

    def test_config_metadata(self, feature_config):
        """Test _meta section has required fields"""
        assert '_meta' in feature_config, "Missing _meta section"

        meta = feature_config['_meta']
        assert 'version' in meta, "Missing version"
        assert 'model_id' in meta, "Missing model_id"
        assert 'created_at' in meta, "Missing created_at"
        assert 'description' in meta, "Missing description"

    def test_config_version_format(self, feature_config):
        """Test version follows semantic versioning"""
        version = feature_config['_meta']['version']
        parts = version.split('.')

        assert len(parts) == 3, f"Version should be X.Y.Z format, got {version}"
        assert all(p.isdigit() for p in parts), "Version parts should be numeric"


@pytest.mark.unit
class TestObservationSpaceConfig:
    """Test observation space configuration for current (15 features)"""

    def test_observation_space_exists(self, feature_config):
        """Test observation_space section exists"""
        assert 'observation_space' in feature_config, "Missing observation_space section"

    def test_observation_dimension(self, feature_config):
        """Test observation dimension is 15 (current format: 13 market + 2 state)"""
        obs_space = feature_config['observation_space']

        # current uses total_dimension instead of dimension
        assert 'total_dimension' in obs_space or 'total_obs_dim' in obs_space, \
            "Missing total_dimension or total_obs_dim"

        total_dim = obs_space.get('total_dimension', obs_space.get('total_obs_dim'))
        assert total_dim == 15, f"Expected total dimension 15 (current), got {total_dim}"

    def test_feature_order(self, feature_config):
        """Test feature order has exactly 15 features (current)"""
        obs_space = feature_config['observation_space']

        assert 'order' in obs_space, "Missing order"
        feature_order = obs_space['order']

        assert len(feature_order) == 15, f"Expected 15 features (current), got {len(feature_order)}"

        # Check some key features are present
        key_features = [
            'position', 'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
            'rsi_9', 'atr_pct', 'adx_14', 'dxy_z', 'vix_z', 'embi_z'
        ]

        for feat in key_features:
            assert feat in feature_order, f"Missing key feature: {feat}"

    def test_state_and_market_features_split(self, feature_config):
        """Test current has state_features and market_features"""
        obs_space = feature_config['observation_space']

        assert 'state_features' in obs_space, "Missing state_features"
        assert 'market_features' in obs_space, "Missing market_features"

        state_count = len(obs_space['state_features'])
        market_count = len(obs_space['market_features'])

        assert state_count == 2, f"Expected 2 state features, got {state_count}"
        assert market_count == 13, f"Expected 13 market features, got {market_count}"
        assert state_count + market_count == 15, "State + market should equal 15"

    def test_position_in_state_features(self, feature_config):
        """Test position is in state_features (current)"""
        obs_space = feature_config['observation_space']
        state_features = obs_space['state_features']

        assert 'position' in state_features, "Missing position in state_features"


@pytest.mark.unit
class TestFeaturesConfig:
    """Test features section configuration for current format"""

    def test_state_features_section_exists(self, feature_config):
        """Test current has state_features section"""
        assert 'state_features' in feature_config, "Missing state_features section"
        assert 'items' in feature_config['state_features'], "state_features missing items"

    def test_market_features_section_exists(self, feature_config):
        """Test current has market_features section with subsections"""
        assert 'market_features' in feature_config, "Missing market_features section"
        # current market_features has subsections: returns, technical, macro_zscore, etc.
        market = feature_config['market_features']
        assert 'returns' in market or 'technical' in market, \
            "market_features should have returns/technical subsections"

    def test_market_features_count(self, feature_config):
        """Test current has multiple market feature subsections"""
        market = feature_config['market_features']
        # Count total items across all subsections
        total_items = 0
        for subsection_name, subsection in market.items():
            if isinstance(subsection, dict) and 'items' in subsection:
                total_items += len(subsection['items'])
        assert total_items == 13, f"Expected 13 market features, got {total_items}"

    def test_state_features_count(self, feature_config):
        """Test current has 2 state features (position, time_normalized)"""
        items = feature_config['state_features']['items']
        assert len(items) == 2, f"Expected 2 state features, got {len(items)}"

    def test_key_market_features_present(self, feature_config):
        """Test key market features are present in current"""
        market = feature_config['market_features']
        # Collect all feature names from all subsections
        feature_names = []
        for subsection_name, subsection in market.items():
            if isinstance(subsection, dict) and 'items' in subsection:
                feature_names.extend([item['name'] for item in subsection['items']])

        key_features = ['log_ret_5m', 'log_ret_1h', 'rsi_9', 'atr_pct', 'adx_14', 'dxy_z', 'vix_z']
        for feat in key_features:
            assert feat in feature_names, f"Missing key market feature: {feat}"

    def test_key_state_features_present(self, feature_config):
        """Test key state features are present in current"""
        items = feature_config['state_features']['items']
        feature_names = [item['name'] for item in items]

        key_features = ['position', 'time_normalized']
        for feat in key_features:
            assert feat in feature_names, f"Missing key state feature: {feat}"


@pytest.mark.unit
class TestTradingConfig:
    """Test trading parameters configuration"""

    def test_trading_section_exists(self, feature_config):
        """Test trading section exists"""
        assert 'trading' in feature_config, "Missing trading section"

    def test_trading_parameters(self, feature_config):
        """Test trading parameters are correctly set (current format)"""
        trading = feature_config['trading']

        # current has signal_thresholds instead of individual threshold params
        assert 'signal_thresholds' in trading, "Missing signal_thresholds"
        thresholds = trading['signal_thresholds']
        assert 'long' in thresholds, "Missing long threshold"
        assert 'short' in thresholds, "Missing short threshold"

        # Cost model is in separate section in current
        assert 'cost_model' in feature_config, "Missing cost_model section"
        cost_model = feature_config['cost_model']
        # current uses basis points (bps) instead of percentage
        assert 'base_spread_bps' in cost_model, \
            "cost_model should have base_spread_bps configuration"

    def test_market_hours_config(self, feature_config):
        """Test market hours configuration"""
        trading = feature_config['trading']

        assert 'market_hours' in trading, "Missing market_hours"
        hours = trading['market_hours']

        assert hours['timezone'] == 'America/Bogota', "Incorrect timezone"
        assert hours['utc_offset'] == -5, "Incorrect UTC offset"
        assert hours['local_start'] == '08:00', "Incorrect local start"
        assert hours['local_end'] == '12:55', "Incorrect local end"

    def test_bars_per_session(self, feature_config):
        """Test bars per session is 60"""
        trading = feature_config['trading']

        assert 'bars_per_session' in trading, "Missing bars_per_session"
        assert trading['bars_per_session'] == 60, \
            f"Expected 60 bars per session, got {trading['bars_per_session']}"


@pytest.mark.unit
class TestEnvironmentConfig:
    """Test environment configuration for current"""

    def test_environment_config_exists(self, feature_config):
        """Test environment_config section exists"""
        assert 'environment_config' in feature_config, "Missing environment_config section"

    def test_cost_model_exists(self, feature_config):
        """Test cost_model section exists"""
        assert 'cost_model' in feature_config, "Missing cost_model section"

    def test_regime_detection_exists(self, feature_config):
        """Test regime_detection section exists"""
        assert 'regime_detection' in feature_config, "Missing regime_detection section"


@pytest.mark.unit
class TestModelConfig:
    """Test model configuration for current"""

    def test_model_section_exists(self, feature_config):
        """Test model section exists"""
        assert 'model' in feature_config, "Missing model section"

    def test_validation_section_exists(self, feature_config):
        """Test validation section exists"""
        assert 'validation' in feature_config, "Missing validation section"

    def test_sources_section_exists(self, feature_config):
        """Test sources section exists for data sources"""
        assert 'sources' in feature_config, "Missing sources section"
