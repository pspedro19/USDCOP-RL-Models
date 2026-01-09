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
    """Test observation space configuration"""

    def test_observation_space_exists(self, feature_config):
        """Test observation_space section exists"""
        assert 'observation_space' in feature_config, "Missing observation_space section"

    def test_observation_dimension(self, feature_config):
        """Test observation dimension is 15"""
        obs_space = feature_config['observation_space']

        assert 'dimension' in obs_space, "Missing dimension"
        assert obs_space['dimension'] == 15, f"Expected dimension 15, got {obs_space['dimension']}"

        assert 'total_obs_dim' in obs_space, "Missing total_obs_dim"
        assert obs_space['total_obs_dim'] == 15, \
            f"Expected total_obs_dim 15, got {obs_space['total_obs_dim']}"

    def test_feature_order(self, feature_config):
        """Test feature order has exactly 13 features"""
        obs_space = feature_config['observation_space']

        assert 'order' in obs_space, "Missing order"
        feature_order = obs_space['order']

        assert len(feature_order) == 13, f"Expected 13 features, got {len(feature_order)}"

        # Check expected features are present
        expected_features = [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
            'rsi_9', 'atr_pct', 'adx_14',
            'dxy_z', 'dxy_change_1d',
            'vix_z', 'embi_z',
            'brent_change_1d', 'rate_spread', 'usdmxn_ret_1h'
        ]

        assert set(feature_order) == set(expected_features), \
            f"Feature order mismatch.\nExpected: {expected_features}\nActual: {feature_order}"

    def test_additional_state_variables(self, feature_config):
        """Test additional_in_env has position and time_normalized"""
        obs_space = feature_config['observation_space']

        assert 'additional_in_env' in obs_space, "Missing additional_in_env"
        additional = obs_space['additional_in_env']

        assert len(additional) == 2, f"Expected 2 additional vars, got {len(additional)}"
        assert 'position' in additional, "Missing position"
        assert 'time_normalized' in additional, "Missing time_normalized"

    def test_time_normalized_config(self, feature_config):
        """Test time_normalized configuration"""
        obs_space = feature_config['observation_space']

        assert 'time_normalized_range' in obs_space, "Missing time_normalized_range"
        range_vals = obs_space['time_normalized_range']

        assert len(range_vals) == 2, "Range should have min and max"
        assert range_vals[0] == 0.0, "Min should be 0.0"
        assert abs(range_vals[1] - 0.983) < 0.001, "Max should be ~0.983, NOT 1.0"


@pytest.mark.unit
class TestFeaturesConfig:
    """Test features section configuration"""

    def test_features_section_exists(self, feature_config):
        """Test features section has required subsections"""
        assert 'features' in feature_config, "Missing features section"

        features = feature_config['features']
        expected_subsections = ['returns', 'technical', 'macro_zscore', 'macro_changes', 'macro_derived']

        for subsection in expected_subsections:
            assert subsection in features, f"Missing subsection: {subsection}"

    def test_returns_features(self, feature_config):
        """Test returns features configuration"""
        returns = feature_config['features']['returns']

        assert 'items' in returns, "Returns missing items"
        items = returns['items']

        assert len(items) == 3, f"Expected 3 return features, got {len(items)}"

        # Check each return feature has required fields
        for item in items:
            assert 'name' in item, "Return feature missing name"
            assert 'formula' in item, "Return feature missing formula"
            assert 'norm_stats' in item, "Return feature missing norm_stats"
            assert 'clip' in item, "Return feature missing clip"

            # Check norm_stats structure
            norm_stats = item['norm_stats']
            assert 'mean' in norm_stats, f"{item['name']} missing mean"
            assert 'std' in norm_stats, f"{item['name']} missing std"

    def test_technical_features(self, feature_config):
        """Test technical features configuration"""
        technical = feature_config['features']['technical']

        assert 'items' in technical, "Technical missing items"
        items = technical['items']

        assert len(items) == 3, f"Expected 3 technical features, got {len(items)}"

        feature_names = [item['name'] for item in items]
        assert 'rsi_9' in feature_names, "Missing rsi_9"
        assert 'atr_pct' in feature_names, "Missing atr_pct"
        assert 'adx_14' in feature_names, "Missing adx_14"

        # Check period configurations
        rsi = next(item for item in items if item['name'] == 'rsi_9')
        assert rsi['period'] == 9, "RSI period should be 9"

        atr = next(item for item in items if item['name'] == 'atr_pct')
        assert atr['period'] == 10, "ATR period should be 10"

        adx = next(item for item in items if item['name'] == 'adx_14')
        assert adx['period'] == 14, "ADX period should be 14"

    def test_macro_zscore_features(self, feature_config):
        """Test macro z-score features configuration"""
        macro_zscore = feature_config['features']['macro_zscore']

        assert 'items' in macro_zscore, "Macro_zscore missing items"
        items = macro_zscore['items']

        assert len(items) == 3, f"Expected 3 macro z-score features, got {len(items)}"

        # Check fixed normalization stats
        dxy = next(item for item in items if item['name'] == 'dxy_z')
        assert dxy['norm_stats']['mean'] == 103.0, "DXY mean should be 103.0"
        assert dxy['norm_stats']['std'] == 5.0, "DXY std should be 5.0"

        vix = next(item for item in items if item['name'] == 'vix_z')
        assert vix['norm_stats']['mean'] == 20.0, "VIX mean should be 20.0"
        assert vix['norm_stats']['std'] == 10.0, "VIX std should be 10.0"

        embi = next(item for item in items if item['name'] == 'embi_z')
        assert embi['norm_stats']['mean'] == 300.0, "EMBI mean should be 300.0"
        assert embi['norm_stats']['std'] == 100.0, "EMBI std should be 100.0"


@pytest.mark.unit
class TestTradingConfig:
    """Test trading parameters configuration"""

    def test_trading_section_exists(self, feature_config):
        """Test trading section exists"""
        assert 'trading' in feature_config, "Missing trading section"

    def test_trading_parameters(self, feature_config):
        """Test trading parameters are correctly set"""
        trading = feature_config['trading']

        assert 'cost_per_trade' in trading, "Missing cost_per_trade"
        assert trading['cost_per_trade'] == 0.0015, \
            f"Expected cost_per_trade 0.0015, got {trading['cost_per_trade']}"

        assert 'weak_signal_threshold' in trading, "Missing weak_signal_threshold"
        assert trading['weak_signal_threshold'] == 0.3, \
            f"Expected weak_signal_threshold 0.3, got {trading['weak_signal_threshold']}"

        assert 'trade_count_threshold' in trading, "Missing trade_count_threshold"
        assert trading['trade_count_threshold'] == 0.3, \
            f"Expected trade_count_threshold 0.3, got {trading['trade_count_threshold']}"

        assert 'min_cost_threshold' in trading, "Missing min_cost_threshold"
        assert trading['min_cost_threshold'] == 0.001, \
            f"Expected min_cost_threshold 0.001, got {trading['min_cost_threshold']}"

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
class TestNormalizationConfig:
    """Test normalization configuration"""

    def test_normalization_method(self, feature_config):
        """Test normalization method is zscore"""
        assert 'normalization' in feature_config, "Missing normalization section"

        norm = feature_config['normalization']
        assert norm['method'] == 'zscore', "Normalization method should be zscore"
        assert norm['formula'] == '(x - mean) / std', "Incorrect normalization formula"

    def test_normalization_clip_range(self, feature_config):
        """Test normalization clip range"""
        norm = feature_config['normalization']

        assert 'clip_after_norm' in norm, "Missing clip_after_norm"
        clip_range = norm['clip_after_norm']

        assert clip_range[0] == -4.0, "Min clip should be -4.0"
        assert clip_range[1] == 4.0, "Max clip should be 4.0"


@pytest.mark.unit
class TestComputeStrategy:
    """Test compute strategy configuration"""

    def test_compute_strategy_exists(self, feature_config):
        """Test compute_strategy section exists"""
        assert 'compute_strategy' in feature_config, "Missing compute_strategy section"

    def test_sql_calculated_features(self, feature_config):
        """Test SQL-calculated features are correctly specified"""
        strategy = feature_config['compute_strategy']

        assert 'sql_calculated' in strategy, "Missing sql_calculated"
        sql_features = strategy['sql_calculated']['features']

        # Should have 9 SQL-calculated features
        assert len(sql_features) == 9, f"Expected 9 SQL features, got {len(sql_features)}"

        expected_sql = [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
            'dxy_z', 'vix_z', 'embi_z',
            'dxy_change_1d', 'brent_change_1d', 'rate_spread'
        ]

        assert set(sql_features) == set(expected_sql), \
            f"SQL features mismatch.\nExpected: {expected_sql}\nActual: {sql_features}"

    def test_python_calculated_features(self, feature_config):
        """Test Python-calculated features are correctly specified"""
        strategy = feature_config['compute_strategy']

        assert 'python_calculated' in strategy, "Missing python_calculated"
        python_features = strategy['python_calculated']['features']

        # Should have 4 Python-calculated features
        assert len(python_features) == 4, f"Expected 4 Python features, got {len(python_features)}"

        expected_python = ['rsi_9', 'atr_pct', 'adx_14', 'usdmxn_ret_1h']

        assert set(python_features) == set(expected_python), \
            f"Python features mismatch.\nExpected: {expected_python}\nActual: {python_features}"

    def test_eliminated_features(self, feature_config):
        """Test eliminated features are documented"""
        strategy = feature_config['compute_strategy']

        assert 'eliminated_v14' in strategy, "Missing eliminated_v14"
        eliminated = strategy['eliminated_v14']['features']

        # Should have 8 eliminated features
        expected_eliminated = [
            'hour_sin', 'hour_cos', 'bb_position', 'dxy_mom_5d',
            'vix_regime', 'brent_vol_5d', 'sma_ratio', 'macd_hist'
        ]

        assert len(eliminated) == 8, f"Expected 8 eliminated features, got {len(eliminated)}"
        assert set(eliminated) == set(expected_eliminated), \
            f"Eliminated features mismatch.\nExpected: {expected_eliminated}\nActual: {eliminated}"
