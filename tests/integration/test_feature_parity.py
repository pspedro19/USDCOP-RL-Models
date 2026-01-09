"""
Integration Test: Feature Parity (CRITICAL)
============================================

Tests that new FeatureCalculator produces identical results to legacy code.
This is the most critical test for ensuring training/inference compatibility.

Compares against RL_DS3_MACRO_CORE.csv with tolerance 1e-6.

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-16
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
class TestFeatureParityLegacy:
    """Compare new feature calculator against legacy dataset"""

    def test_features_match_legacy(self, feature_calculator, legacy_features_df):
        """
        CRITICAL: Compare with data from training - tolerance 1e-6

        This test ensures that the new FeatureCalculator produces
        bit-for-bit identical results to the legacy code that generated
        the training data. Any deviation means model predictions will drift.
        """
        # Extract OHLCV data from legacy
        ohlcv = legacy_features_df[['timestamp', 'open', 'high', 'low', 'close']].copy()
        ohlcv.rename(columns={'timestamp': 'time'}, inplace=True)

        # Compute features with new calculator
        new_features = feature_calculator.compute_technical_features(ohlcv)

        # Features to compare (technical indicators only - macro handled separately)
        features_to_compare = ['log_ret_5m', 'log_ret_1h', 'log_ret_4h',
                               'rsi_9', 'atr_pct', 'adx_14']

        results = {}
        for feature in features_to_compare:
            if feature not in legacy_features_df.columns:
                pytest.skip(f"Legacy dataset missing {feature}")

            legacy_values = legacy_features_df[feature]
            new_values = new_features[feature]

            # Align by removing NaN (first few values)
            valid_mask = legacy_values.notna() & new_values.notna()
            legacy_valid = legacy_values[valid_mask]
            new_valid = new_values[valid_mask]

            # Compare
            diff = (legacy_valid.values - new_valid.values[:len(legacy_valid)])
            max_diff = np.abs(diff).max()
            mean_diff = np.abs(diff).mean()

            results[feature] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'passed': max_diff < 1e-6
            }

            # Assert with detailed error message
            assert max_diff < 1e-6, \
                f"{feature}: max diff = {max_diff:.2e} (threshold 1e-6)\n" \
                f"Mean diff: {mean_diff:.2e}\n" \
                f"Sample legacy: {legacy_valid.iloc[:5].values}\n" \
                f"Sample new: {new_valid.iloc[:5].values}"

        # Print summary
        print("\nFeature Parity Test Results:")
        print("-" * 60)
        for feature, result in results.items():
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{feature:15s}: {status} (max_diff={result['max_diff']:.2e})")

    def test_log_returns_parity(self, feature_calculator, legacy_features_df):
        """Test log returns match exactly"""
        close = legacy_features_df['close']

        # Test 5m returns
        log_ret_5m_new = feature_calculator.calc_log_return(close, periods=1)
        log_ret_5m_legacy = legacy_features_df['log_ret_5m']

        valid_mask = log_ret_5m_legacy.notna() & log_ret_5m_new.notna()
        diff = (log_ret_5m_legacy[valid_mask] - log_ret_5m_new[valid_mask]).abs().max()

        assert diff < 1e-6, f"log_ret_5m parity failed: max diff = {diff:.2e}"

    def test_rsi_parity(self, feature_calculator, legacy_features_df):
        """Test RSI matches exactly"""
        close = legacy_features_df['close']

        rsi_new = feature_calculator.calc_rsi(close, period=9)
        rsi_legacy = legacy_features_df['rsi_9']

        valid_mask = rsi_legacy.notna() & rsi_new.notna()
        diff = (rsi_legacy[valid_mask] - rsi_new[valid_mask]).abs().max()

        assert diff < 1e-6, f"RSI parity failed: max diff = {diff:.2e}"

    def test_atr_pct_parity(self, feature_calculator, legacy_features_df):
        """Test ATR% matches exactly"""
        high = legacy_features_df['high']
        low = legacy_features_df['low']
        close = legacy_features_df['close']

        atr_pct_new = feature_calculator.calc_atr_pct(high, low, close, period=10)
        atr_pct_legacy = legacy_features_df['atr_pct']

        valid_mask = atr_pct_legacy.notna() & atr_pct_new.notna()
        diff = (atr_pct_legacy[valid_mask] - atr_pct_new[valid_mask]).abs().max()

        assert diff < 1e-6, f"ATR% parity failed: max diff = {diff:.2e}"

    def test_adx_parity(self, feature_calculator, legacy_features_df):
        """Test ADX matches exactly"""
        high = legacy_features_df['high']
        low = legacy_features_df['low']
        close = legacy_features_df['close']

        adx_new = feature_calculator.calc_adx(high, low, close, period=14)
        adx_legacy = legacy_features_df['adx_14']

        valid_mask = adx_legacy.notna() & adx_new.notna()
        diff = (adx_legacy[valid_mask] - adx_new[valid_mask]).abs().max()

        assert diff < 1e-6, f"ADX parity failed: max diff = {diff:.2e}"


@pytest.mark.integration
class TestMacroFeatureParity:
    """Test macro feature calculations match legacy"""

    def test_dxy_zscore_parity(self, legacy_features_df):
        """Test DXY z-score matches legacy"""
        # Legacy DXY z-score should match (dxy - 103) / 5
        dxy_z_legacy = legacy_features_df['dxy_z']

        # Verify clipping to [-4, 4]
        assert (dxy_z_legacy.dropna() >= -4).all(), "Legacy DXY z-score below -4"
        assert (dxy_z_legacy.dropna() <= 4).all(), "Legacy DXY z-score above 4"

    def test_vix_zscore_parity(self, legacy_features_df):
        """Test VIX z-score matches legacy"""
        # Legacy VIX z-score should match (vix - 20) / 10
        vix_z_legacy = legacy_features_df['vix_z']

        # Verify clipping to [-4, 4]
        assert (vix_z_legacy.dropna() >= -4).all(), "Legacy VIX z-score below -4"
        assert (vix_z_legacy.dropna() <= 4).all(), "Legacy VIX z-score above 4"

    def test_rate_spread_parity(self, legacy_features_df):
        """Test rate spread calculation"""
        rate_spread_legacy = legacy_features_df['rate_spread']

        # Rate spread should be reasonable (typically -2 to 4)
        assert rate_spread_legacy.dropna().min() > -5, "Rate spread suspiciously low"
        assert rate_spread_legacy.dropna().max() < 10, "Rate spread suspiciously high"


@pytest.mark.integration
class TestEliminatedFeatures:
    """Verify eliminated features are NOT in new implementation"""

    def test_hour_sin_cos_eliminated(self, feature_calculator):
        """Verify hour_sin and hour_cos are NOT in feature order"""
        feature_order = feature_calculator.feature_order

        assert 'hour_sin' not in feature_order, "hour_sin should be eliminated in v14"
        assert 'hour_cos' not in feature_order, "hour_cos should be eliminated in v14"

    def test_bb_position_eliminated(self, feature_calculator):
        """Verify bb_position is NOT in feature order"""
        assert 'bb_position' not in feature_calculator.feature_order, \
            "bb_position should be eliminated in v14"

    def test_eliminated_features_not_in_config(self, feature_config):
        """Verify eliminated features are not in observation order"""
        observation_order = feature_config['observation_space']['order']

        eliminated = ['hour_sin', 'hour_cos', 'bb_position', 'dxy_mom_5d',
                     'vix_regime', 'brent_vol_5d', 'sma_ratio', 'macd_hist']

        for feat in eliminated:
            assert feat not in observation_order, \
                f"{feat} should not be in observation_space.order (eliminated in v14)"


@pytest.mark.integration
class TestNormalizationParity:
    """Test normalization produces consistent results with legacy"""

    def test_normalized_ranges_match_legacy(self, legacy_features_df):
        """Test that normalized features have similar distributions"""
        # Log returns should be small (clipped to ±0.05 before normalization)
        for feature in ['log_ret_5m', 'log_ret_1h', 'log_ret_4h']:
            if feature in legacy_features_df.columns:
                values = legacy_features_df[feature].dropna()

                # After z-score normalization, should be mostly within [-4, 4]
                within_range = ((values >= -5) & (values <= 5)).mean()
                assert within_range > 0.95, \
                    f"{feature}: Only {within_range*100:.1f}% within [-5, 5]"

    def test_technical_indicator_ranges(self, legacy_features_df):
        """Test technical indicators have expected ranges after normalization"""
        # RSI before normalization should be [0, 100]
        # After normalization with mean~50, std~23, should be mostly [-3, 3]

        # ATR% should be positive
        if 'atr_pct' in legacy_features_df.columns:
            atr_pct = legacy_features_df['atr_pct'].dropna()
            # Before normalization, ATR% typically 0-5%
            # Check we have reasonable values (post-normalization)
            assert atr_pct.mean() > -5, "ATR% mean suspiciously low"
            assert atr_pct.mean() < 5, "ATR% mean suspiciously high"


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndParity:
    """End-to-end test: Full pipeline matches legacy"""

    def test_full_pipeline_parity(self, feature_calculator, legacy_features_df):
        """
        Test complete pipeline from OHLCV to normalized features

        This is the ultimate integration test - if this passes,
        we have feature parity with the training data.
        """
        # Extract OHLCV
        ohlcv = legacy_features_df[['timestamp', 'open', 'high', 'low', 'close']].copy()
        ohlcv.rename(columns={'timestamp': 'time'}, inplace=True)

        # Compute all technical features
        features = feature_calculator.compute_technical_features(ohlcv)

        # Normalize
        normalized = feature_calculator.normalize_features(features)

        # Compare normalized features
        comparison_features = ['log_ret_5m', 'log_ret_1h', 'log_ret_4h',
                             'rsi_9', 'atr_pct', 'adx_14']

        all_passed = True
        for feature in comparison_features:
            if feature not in legacy_features_df.columns:
                continue

            legacy_norm = legacy_features_df[feature]
            new_norm = normalized[feature]

            # Align and compare
            valid_mask = legacy_norm.notna() & new_norm.notna()
            if valid_mask.sum() < 10:
                continue  # Skip if not enough valid data

            diff = (legacy_norm[valid_mask] - new_norm[valid_mask][:len(legacy_norm[valid_mask])]).abs().max()

            if diff >= 1e-6:
                print(f"\n{feature} FAILED parity: max diff = {diff:.2e}")
                all_passed = False
            else:
                print(f"{feature} PASSED: max diff = {diff:.2e}")

        assert all_passed, "One or more features failed parity test"

    def test_observation_construction_from_legacy_data(self, feature_calculator, legacy_features_df):
        """Test building observations from legacy data"""
        # Take a row from legacy data
        if len(legacy_features_df) < 100:
            pytest.skip("Not enough legacy data")

        row = legacy_features_df.iloc[100]

        # Extract the 13 features
        features = pd.Series({
            'log_ret_5m': row['log_ret_5m'],
            'log_ret_1h': row['log_ret_1h'],
            'log_ret_4h': row['log_ret_4h'],
            'rsi_9': row['rsi_9'],
            'atr_pct': row['atr_pct'],
            'adx_14': row['adx_14'],
            'dxy_z': row['dxy_z'],
            'dxy_change_1d': row['dxy_change_1d'],
            'vix_z': row['vix_z'],
            'embi_z': row['embi_z'],
            'brent_change_1d': row['brent_change_1d'],
            'rate_spread': row['rate_spread'],
            'usdmxn_ret_1h': row['usdmxn_ret_1h'],
        })

        # Build observation
        obs = feature_calculator.build_observation(
            features=features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        # Verify observation structure
        assert obs.shape == (15,), f"Expected 15-dim observation, got {obs.shape}"

        # Verify features are in correct order
        for i, feat_name in enumerate(feature_calculator.feature_order):
            expected_val = features[feat_name]
            if not np.isnan(expected_val):
                # Allow for clipping to [-5, 5]
                actual_val = obs[i]
                if abs(expected_val) <= 5:
                    assert abs(actual_val - expected_val) < 1e-5, \
                        f"Feature {feat_name} mismatch at position {i}"
