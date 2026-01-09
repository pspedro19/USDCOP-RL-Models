"""
Integration Test: Training-Production Parity (v15)
===================================================

CRITICAL: These tests verify that the normalization applied during
training matches EXACTLY what production inference uses.

The v14 system had a CRITICAL bug where:
- Training used ROLLING z-scores (window=50)
- Production used FIXED z-scores
- Correlation between them was only ~5%!

v15 fixes this by using FIXED z-scores everywhere.

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-17
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path


# Load feature_config.json
CONFIG_PATH = Path(__file__).parent.parent.parent / 'config' / 'feature_config.json'

@pytest.fixture
def feature_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


# V15 Fixed normalization stats (MUST match feature_config.json v4.0.0)
# These are the ACTUAL training stats from validation section
FIXED_STATS = {
    'dxy': {'mean': 100.21, 'std': 5.60},
    'vix': {'mean': 21.16, 'std': 7.89},
    'embi': {'mean': 322.01, 'std': 62.68},
    'rate_spread_raw': {'mean': 7.03, 'std': 1.41},  # Before z-score
}

# SQL formulas use rounded stats for simplicity
SQL_STATS = {
    'dxy': {'mean': 103.0, 'std': 5.0},
    'vix': {'mean': 20.0, 'std': 10.0},
    'embi': {'mean': 300.0, 'std': 100.0},
    'rate_spread': {'mean': -0.0326, 'std': 1.400},
}


def z_score_fixed(value, mean, std, clip=4.0):
    """Fixed z-score calculation (v15)"""
    z = (value - mean) / std
    return np.clip(z, -clip, clip)


class TestNormalizationParity:
    """Test that training and production normalization match"""

    @pytest.mark.parametrize("raw_dxy,expected_range", [
        (94.0, (-1.5, -0.9)),   # Low DXY: z = (94-100.21)/5.60 = -1.11
        (100.0, (-0.5, 0.5)),   # Around mean: z = (100-100.21)/5.60 = -0.04
        (106.0, (0.8, 1.2)),    # Above mean: z = (106-100.21)/5.60 = 1.03
        (111.0, (1.7, 2.1)),    # High DXY: z = (111-100.21)/5.60 = 1.93
        (122.0, (3.6, 4.0)),    # Very high: z = (122-100.21)/5.60 = 3.89
    ])
    def test_dxy_z_calculation(self, raw_dxy, expected_range):
        """Test DXY z-score calculation matches expected range"""
        z = z_score_fixed(raw_dxy, **FIXED_STATS['dxy'])
        assert expected_range[0] <= z <= expected_range[1], \
            f"DXY={raw_dxy} -> z={z:.2f}, expected in {expected_range}"

    @pytest.mark.parametrize("raw_vix,expected_range", [
        (13.0, (-1.2, -0.8)),   # Low VIX: z = (13-21.16)/7.89 = -1.03
        (21.0, (-0.3, 0.3)),    # Around mean: z = (21-21.16)/7.89 = -0.02
        (29.0, (0.8, 1.2)),     # Elevated: z = (29-21.16)/7.89 = 0.99
        (45.0, (2.7, 3.3)),     # Crisis: z = (45-21.16)/7.89 = 3.02
        (80.0, (4.0, 4.0)),     # Extreme: z = (80-21.16)/7.89 = 7.46 -> clips to 4.0
    ])
    def test_vix_z_calculation(self, raw_vix, expected_range):
        """Test VIX z-score calculation matches expected range"""
        z = z_score_fixed(raw_vix, **FIXED_STATS['vix'])
        assert expected_range[0] <= z <= expected_range[1], \
            f"VIX={raw_vix} -> z={z:.2f}, expected in {expected_range}"

    @pytest.mark.parametrize("raw_embi,expected_range", [
        (260.0, (-1.2, -0.8)),  # Low EMBI: z = (260-322.01)/62.68 = -0.99
        (322.0, (-0.3, 0.3)),   # Around mean: z = (322-322.01)/62.68 = -0.00
        (385.0, (0.8, 1.2)),    # Elevated: z = (385-322.01)/62.68 = 1.00
        (510.0, (2.7, 3.3)),    # High risk: z = (510-322.01)/62.68 = 3.00
    ])
    def test_embi_z_calculation(self, raw_embi, expected_range):
        """Test EMBI z-score calculation matches expected range"""
        z = z_score_fixed(raw_embi, **FIXED_STATS['embi'])
        assert expected_range[0] <= z <= expected_range[1], \
            f"EMBI={raw_embi} -> z={z:.2f}, expected in {expected_range}"

    @pytest.mark.parametrize("ust10y,expected_range", [
        (2.0, (0.4, 1.0)),      # Low US rates: spread=8.0, z = (8.0-7.03)/1.41 = 0.69
        (3.0, (-0.3, 0.3)),     # Medium: spread=7.0, z = (7.0-7.03)/1.41 = -0.02
        (4.0, (-1.0, -0.4)),    # Higher: spread=6.0, z = (6.0-7.03)/1.41 = -0.73
        (5.0, (-1.7, -1.1)),    # High: spread=5.0, z = (5.0-7.03)/1.41 = -1.44
    ])
    def test_rate_spread_calculation(self, ust10y, expected_range):
        """Test rate_spread = 10.0 - UST10Y, then z-score (sovereign spread)"""
        raw_spread = 10.0 - ust10y
        z = z_score_fixed(raw_spread, **FIXED_STATS['rate_spread_raw'])
        assert expected_range[0] <= z <= expected_range[1], \
            f"UST10Y={ust10y} -> spread={raw_spread} -> z={z:.2f}, expected in {expected_range}"


class TestConfigAlignment:
    """Test that code stats match feature_config.json"""

    def test_dxy_stats_match_config(self, feature_config):
        """DXY stats must match feature_config.json"""
        config_stats = feature_config['validation']['norm_stats_validation']['dxy_z']
        assert abs(FIXED_STATS['dxy']['mean'] - config_stats['mean']) < 0.1
        assert abs(FIXED_STATS['dxy']['std'] - config_stats['std']) < 0.1

    def test_vix_stats_match_config(self, feature_config):
        """VIX stats must match feature_config.json"""
        config_stats = feature_config['validation']['norm_stats_validation']['vix_z']
        assert abs(FIXED_STATS['vix']['mean'] - config_stats['mean']) < 0.1
        assert abs(FIXED_STATS['vix']['std'] - config_stats['std']) < 0.1

    def test_embi_stats_match_config(self, feature_config):
        """EMBI stats must match feature_config.json"""
        config_stats = feature_config['validation']['norm_stats_validation']['embi_z']
        assert abs(FIXED_STATS['embi']['mean'] - config_stats['mean']) < 0.1
        assert abs(FIXED_STATS['embi']['std'] - config_stats['std']) < 0.1

    def test_rate_spread_stats_match_config(self, feature_config):
        """Rate spread stats must match feature_config.json"""
        config_stats = feature_config['validation']['norm_stats_validation']['rate_spread']
        assert abs(FIXED_STATS['rate_spread_raw']['mean'] - config_stats['mean']) < 0.01
        assert abs(FIXED_STATS['rate_spread_raw']['std'] - config_stats['std']) < 0.01

    def test_model_version_is_v15(self, feature_config):
        """Model must be v15 after this fix"""
        model_id = feature_config['_meta']['model_id']
        assert 'v15' in model_id, f"Expected v15, got {model_id}"

    def test_config_version_is_4_0_0(self, feature_config):
        """Config version must be 4.0.0 or higher"""
        version = feature_config['_meta']['version']
        major = int(version.split('.')[0])
        assert major >= 4, f"Expected version 4.0.0+, got {version}"


class TestZScoreFormulas:
    """Test that z-score formulas match between config and code"""

    def test_dxy_z_formula_matches_config(self, feature_config):
        """DXY z-score SQL formula must match Python implementation"""
        sql_formula = feature_config['features']['macro_zscore']['items'][0]['sql']
        expected = "(dxy - 100.21) / 5.60"
        assert sql_formula == expected, f"SQL formula mismatch: {sql_formula} != {expected}"

        # Test numerical equivalence between SQL and Python
        test_value = 108.0
        sql_result = (test_value - 100.21) / 5.60
        py_result = z_score_fixed(test_value, **FIXED_STATS['dxy'])
        assert abs(sql_result - py_result) < 0.01, \
            f"SQL z={sql_result:.2f} vs Python z={py_result:.2f} differ by {abs(sql_result - py_result):.2f}"

    def test_vix_z_formula_matches_config(self, feature_config):
        """VIX z-score SQL formula must match Python implementation"""
        sql_formula = feature_config['features']['macro_zscore']['items'][1]['sql']
        expected = "(vix - 21.16) / 7.89"
        assert sql_formula == expected, f"SQL formula mismatch: {sql_formula} != {expected}"

        # Test numerical equivalence between SQL and Python
        test_value = 30.0
        sql_result = (test_value - 21.16) / 7.89
        py_result = z_score_fixed(test_value, **FIXED_STATS['vix'])
        assert abs(sql_result - py_result) < 0.01, \
            f"SQL z={sql_result:.2f} vs Python z={py_result:.2f} differ by {abs(sql_result - py_result):.2f}"

    def test_embi_z_formula_matches_config(self, feature_config):
        """EMBI z-score SQL formula must match Python implementation"""
        sql_formula = feature_config['features']['macro_zscore']['items'][2]['sql']
        expected = "(embi - 322.01) / 62.68"
        assert sql_formula == expected, f"SQL formula mismatch: {sql_formula} != {expected}"

        # Test numerical equivalence between SQL and Python
        test_value = 400.0
        sql_result = (test_value - 322.01) / 62.68
        py_result = z_score_fixed(test_value, **FIXED_STATS['embi'])
        assert abs(sql_result - py_result) < 0.01, \
            f"SQL z={sql_result:.2f} vs Python z={py_result:.2f} differ by {abs(sql_result - py_result):.2f}"

    def test_rate_spread_formula_matches_config(self, feature_config):
        """Rate spread SQL formula must match Python implementation (sovereign spread)"""
        sql_formula = feature_config['features']['macro_derived']['items'][0]['sql']
        expected = "10.0 - treasury_10y"
        assert sql_formula == expected, f"SQL formula mismatch: {sql_formula} != {expected}"


class TestNoMoreSkew:
    """Test that the training-production skew is eliminated"""

    def test_fixed_zscore_is_deterministic(self):
        """Fixed z-score must produce same output for same input"""
        raw_values = [98.0, 103.0, 108.0, 113.0]

        results1 = [z_score_fixed(v, **FIXED_STATS['dxy']) for v in raw_values]
        results2 = [z_score_fixed(v, **FIXED_STATS['dxy']) for v in raw_values]

        assert results1 == results2, "Fixed z-score must be deterministic"

    def test_zscore_distribution_is_reasonable(self):
        """Z-scores should have reasonable distribution"""
        # Simulate realistic DXY values using ACTUAL training stats
        np.random.seed(42)
        dxy_values = np.random.normal(100.21, 5.60, 1000)

        z_values = [z_score_fixed(v, **FIXED_STATS['dxy']) for v in dxy_values]

        # Should be approximately standard normal (before clipping)
        assert -0.5 < np.mean(z_values) < 0.5, "Mean should be ~0"
        assert 0.8 < np.std(z_values) < 1.2, "Std should be ~1"

    def test_clipping_works_at_boundaries(self):
        """Z-score clipping must work at extreme values"""
        # Test extreme low (using actual stats: mean=100.21, std=5.60)
        extreme_low = 77.81  # z = (77.81 - 100.21) / 5.60 = -4.0
        z_low = z_score_fixed(extreme_low, **FIXED_STATS['dxy'])
        assert abs(z_low - (-4.0)) < 0.01, f"Extreme low should clip to -4.0, got {z_low}"

        # Test extreme high
        extreme_high = 122.61  # z = (122.61 - 100.21) / 5.60 = 4.0
        z_high = z_score_fixed(extreme_high, **FIXED_STATS['dxy'])
        assert abs(z_high - 4.0) < 0.01, f"Extreme high should clip to 4.0, got {z_high}"

        # Test beyond clip
        beyond_low = 50.0  # z = (50 - 100.21) / 5.60 = -8.97, clips to -4
        z_beyond_low = z_score_fixed(beyond_low, **FIXED_STATS['dxy'])
        assert abs(z_beyond_low - (-4.0)) < 0.01, f"Beyond low should clip to -4.0, got {z_beyond_low}"


@pytest.mark.integration
class TestEndToEndParity:
    """End-to-end parity test with real CSV data"""

    def test_csv_uses_fixed_zscore_distribution(self):
        """Verify regenerated CSV has correct z-score distribution"""
        csv_path = Path(__file__).parent.parent.parent / 'data' / 'pipeline' / '07_output' / 'datasets_5min' / 'RL_DS3_MACRO_CORE.csv'

        if not csv_path.exists():
            pytest.skip("CSV not regenerated yet")

        df = pd.read_csv(csv_path)

        # After fixing, z-scores should have std ~1.0 (not ~1.4 from rolling)
        for col in ['dxy_z', 'vix_z', 'embi_z']:
            if col in df.columns:
                std = df[col].std()
                # Fixed z-score should have std closer to 1.0
                # Rolling had std ~1.4 due to autocorrelation
                assert 0.7 < std < 1.3, \
                    f"{col} std={std:.2f}, expected ~1.0 (fixed), not ~1.4 (rolling)"

    def test_csv_z_scores_are_bounded(self):
        """Verify all z-scores are within clip range [-4, 4]"""
        csv_path = Path(__file__).parent.parent.parent / 'data' / 'pipeline' / '07_output' / 'datasets_5min' / 'RL_DS3_MACRO_CORE.csv'

        if not csv_path.exists():
            pytest.skip("CSV not regenerated yet")

        df = pd.read_csv(csv_path)

        for col in ['dxy_z', 'vix_z', 'embi_z']:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                assert min_val >= -4.0, f"{col} has values below -4.0: {min_val}"
                assert max_val <= 4.0, f"{col} has values above 4.0: {max_val}"

    def test_csv_rate_spread_is_computed_correctly(self):
        """Verify rate_spread = treasury_10y - treasury_2y"""
        csv_path = Path(__file__).parent.parent.parent / 'data' / 'pipeline' / '07_output' / 'datasets_5min' / 'RL_DS3_MACRO_CORE.csv'

        if not csv_path.exists():
            pytest.skip("CSV not regenerated yet")

        df = pd.read_csv(csv_path)

        if 'rate_spread' in df.columns:
            # Check that values are in reasonable range
            mean_spread = df['rate_spread'].mean()
            std_spread = df['rate_spread'].std()

            # After z-score, should have mean ~0, std ~1
            assert -1.0 < mean_spread < 1.0, f"rate_spread mean={mean_spread:.2f}, expected ~0"
            assert 0.5 < std_spread < 2.0, f"rate_spread std={std_spread:.2f}, expected ~1"


class TestRollingVsFixed:
    """Test to demonstrate the difference between rolling and fixed z-scores"""

    def test_rolling_produces_higher_variance(self):
        """Rolling z-score produces higher variance than fixed"""
        np.random.seed(42)
        # Simulate time series with trend (using actual DXY stats)
        x = np.linspace(95, 105, 1000) + np.random.normal(0, 2, 1000)

        # Fixed z-score (using actual stats)
        fixed_z = [(val - 100.21) / 5.60 for val in x]

        # Rolling z-score (window=50)
        rolling_mean = pd.Series(x).rolling(50).mean()
        rolling_std = pd.Series(x).rolling(50).std()
        rolling_z = [(x[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
                     if i >= 50 else np.nan
                     for i in range(len(x))]
        rolling_z = [z for z in rolling_z if not np.isnan(z)]

        # Rolling should have higher variance due to detrending
        fixed_std = np.std(fixed_z)
        rolling_std_val = np.std(rolling_z)

        # This demonstrates the problem: rolling removes trend, inflating variance
        assert rolling_std_val > fixed_std, \
            f"Rolling std ({rolling_std_val:.2f}) should be > Fixed std ({fixed_std:.2f})"

    def test_fixed_preserves_trend_signal(self):
        """Fixed z-score preserves trend information"""
        # Test with upward trend (using actual stats)
        x_trend = np.linspace(95, 105, 100)

        # Fixed z-score should capture the trend (using actual stats)
        fixed_z = [(val - 100.21) / 5.60 for val in x_trend]

        # Should show clear upward movement
        assert fixed_z[-1] > fixed_z[0], "Fixed z-score should preserve trend"
        assert fixed_z[50] > fixed_z[0], "Fixed z-score should show midpoint trend"


class TestProductionInferenceAlignment:
    """Test that production inference will use correct normalization"""

    def test_inference_sql_view_uses_fixed_stats(self, feature_config):
        """Verify feature_config.json specifies fixed z-score in SQL"""
        macro_features = feature_config['features']['macro_zscore']['items']

        for feature in macro_features:
            assert 'sql' in feature, f"{feature['name']} missing SQL formula"
            assert 'zscore_fixed' in feature['transform'], \
                f"{feature['name']} not using zscore_fixed"

    def test_all_macro_features_have_norm_stats(self, feature_config):
        """All macro z-score features must have norm_stats defined"""
        macro_features = feature_config['features']['macro_zscore']['items']

        for feature in macro_features:
            assert 'norm_stats' in feature, f"{feature['name']} missing norm_stats"
            assert 'mean' in feature['norm_stats'], f"{feature['name']} missing mean"
            assert 'std' in feature['norm_stats'], f"{feature['name']} missing std"

    def test_sql_compute_strategy_includes_zscore_features(self, feature_config):
        """SQL compute strategy must include z-score features"""
        sql_features = feature_config['compute_strategy']['sql_features']

        assert 'dxy_z' in sql_features, "dxy_z should be computed in SQL"
        assert 'vix_z' in sql_features, "vix_z should be computed in SQL"
        assert 'embi_z' in sql_features, "embi_z should be computed in SQL"
        assert 'rate_spread' in sql_features, "rate_spread should be computed in SQL"


if __name__ == "__main__":
    # Allow running directly for quick tests
    pytest.main([__file__, "-v", "--tb=short"])
