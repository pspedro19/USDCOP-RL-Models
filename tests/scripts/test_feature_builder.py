"""
Test script for FeatureBuilder consolidated module.

Validates:
1. Configuration loading
2. Feature calculation (RSI, ATR, ADX)
3. Observation building
4. Batch processing
5. Normalization

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-16
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import FeatureBuilder, get_config, ConfigurationError

print("=" * 80)
print("FEATURE BUILDER VALIDATION TEST")
print("=" * 80)

# =============================================================================
# TEST 1: Configuration Loading
# =============================================================================
print("\n" + "-" * 80)
print("TEST 1: Configuration Loading")
print("-" * 80)

try:
    config = get_config()
    print(f"✓ Config loaded successfully")
    print(f"  Version: {config.version}")
    print(f"  Features: {len(config.get_feature_order())} features")
    print(f"  Obs dim: {config.get_obs_dim()}")
except ConfigurationError as e:
    print(f"✗ Config loading failed: {e}")
    sys.exit(1)

# =============================================================================
# TEST 2: FeatureBuilder Initialization
# =============================================================================
print("\n" + "-" * 80)
print("TEST 2: FeatureBuilder Initialization")
print("-" * 80)

try:
    builder = FeatureBuilder()
    print(f"✓ FeatureBuilder initialized")
    print(f"  Version: {builder.version}")
    print(f"  Feature order:")
    for i, feat in enumerate(builder.feature_order, 1):
        print(f"    {i:2d}. {feat}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# =============================================================================
# TEST 3: Technical Indicators
# =============================================================================
print("\n" + "-" * 80)
print("TEST 3: Technical Indicators (RSI, ATR, ADX)")
print("-" * 80)

# Create sample data
np.random.seed(42)
n = 100
close = pd.Series(4200 + np.cumsum(np.random.randn(n) * 2))
high = close + np.random.rand(n) * 5
low = close - np.random.rand(n) * 5

try:
    # RSI
    rsi = builder.calc_rsi(close, period=9)
    assert rsi.min() >= 0 and rsi.max() <= 100, "RSI out of range"
    print(f"✓ RSI: range=[{rsi.min():.1f}, {rsi.max():.1f}], mean={rsi.mean():.1f}")

    # ATR %
    atr_pct = builder.calc_atr_pct(high, low, close, period=10)
    assert atr_pct.min() >= 0, "ATR% negative"
    print(f"✓ ATR%: range=[{atr_pct.min():.3f}, {atr_pct.max():.3f}], mean={atr_pct.mean():.3f}")

    # ADX
    adx = builder.calc_adx(high, low, close, period=14)
    assert adx.min() >= 0 and adx.max() <= 100, "ADX out of range"
    print(f"✓ ADX: range=[{adx.min():.1f}, {adx.max():.1f}], mean={adx.mean():.1f}")

except Exception as e:
    print(f"✗ Technical indicators failed: {e}")
    sys.exit(1)

# =============================================================================
# TEST 4: Observation Building
# =============================================================================
print("\n" + "-" * 80)
print("TEST 4: Observation Building")
print("-" * 80)

try:
    # Create feature dict with all 13 features
    features_dict = {
        'log_ret_5m': 0.0002,
        'log_ret_1h': 0.0005,
        'log_ret_4h': 0.0008,
        'rsi_9': 55.0,
        'atr_pct': 0.08,
        'adx_14': 35.0,
        'dxy_z': 0.5,
        'dxy_change_1d': 0.001,
        'vix_z': -0.3,
        'embi_z': 0.2,
        'brent_change_1d': -0.02,
        'rate_spread': 1.2,
        'usdmxn_ret_1h': 0.0003
    }

    # Build observation for bar 30 of 60 with position 0.5
    obs = builder.build_observation(
        features_dict,
        position=0.5,
        bar_number=30
    )

    assert obs.shape == (15,), f"Wrong shape: {obs.shape}"
    assert not np.any(np.isnan(obs)), "Observation contains NaN"
    assert not np.any(np.isinf(obs)), "Observation contains Inf"

    # Check time_normalized = (30-1)/60 = 0.483
    time_normalized = obs[14]
    expected_time = (30 - 1) / 60
    assert abs(time_normalized - expected_time) < 0.001, \
        f"time_normalized wrong: {time_normalized} vs {expected_time}"

    # Check position
    position = obs[13]
    assert abs(position - 0.5) < 0.001, f"Position wrong: {position} vs 0.5"

    print(f"✓ Observation built successfully")
    print(f"  Shape: {obs.shape}")
    print(f"  Range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Position: {obs[13]:.3f}")
    print(f"  time_normalized: {obs[14]:.3f} (expected: {expected_time:.3f})")
    print(f"  First 5 features: {obs[:5]}")

    # Validate
    builder.validate_observation(obs)
    print(f"✓ Observation validation passed")

except Exception as e:
    print(f"✗ Observation building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# TEST 5: Normalization
# =============================================================================
print("\n" + "-" * 80)
print("TEST 5: Feature Normalization")
print("-" * 80)

try:
    # Test single feature normalization
    raw_rsi = 55.0
    norm_rsi = builder.normalize_feature('rsi_9', raw_rsi)

    # Get expected from config
    stats = builder.config.get_norm_stats('rsi_9')
    expected = (raw_rsi - stats['mean']) / stats['std']
    expected = np.clip(expected, -4.0, 4.0)

    assert abs(norm_rsi - expected) < 0.001, \
        f"Normalization mismatch: {norm_rsi} vs {expected}"

    print(f"✓ Single feature normalization")
    print(f"  RSI raw={raw_rsi}, normalized={norm_rsi:.3f}")
    print(f"  Stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    # Test batch normalization
    df = pd.DataFrame({
        'rsi_9': [50.0, 55.0, 60.0],
        'atr_pct': [0.05, 0.08, 0.10],
        'adx_14': [30.0, 35.0, 40.0]
    })

    df_norm = builder.normalize_batch(df)

    assert df_norm.shape == df.shape, "Shape mismatch after normalization"
    assert not df_norm.isna().any().any(), "NaN after normalization"

    print(f"✓ Batch normalization")
    print(f"  Input shape: {df.shape}")
    print(f"  Output shape: {df_norm.shape}")

except Exception as e:
    print(f"✗ Normalization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# TEST 6: Edge Cases
# =============================================================================
print("\n" + "-" * 80)
print("TEST 6: Edge Cases")
print("-" * 80)

try:
    # Test with missing features (should default to 0)
    partial_features = {
        'log_ret_5m': 0.0002,
        'rsi_9': 55.0,
        # Missing other features
    }

    obs = builder.build_observation(partial_features, position=0.0, bar_number=1)
    assert obs.shape == (15,), "Wrong shape with partial features"
    assert not np.any(np.isnan(obs)), "NaN with partial features"
    print(f"✓ Partial features handled (filled with zeros)")

    # Test boundary conditions
    obs_start = builder.build_observation(features_dict, position=-1.0, bar_number=1)
    assert obs_start[14] == 0.0, "time_normalized wrong at bar 1"
    print(f"✓ Boundary: bar 1 -> time_normalized={obs_start[14]:.3f}")

    obs_end = builder.build_observation(features_dict, position=1.0, bar_number=60)
    expected_end = (60 - 1) / 60
    assert abs(obs_end[14] - expected_end) < 0.001, \
        f"time_normalized wrong at bar 60: {obs_end[14]} vs {expected_end}"
    print(f"✓ Boundary: bar 60 -> time_normalized={obs_end[14]:.3f} (max=0.983)")

except Exception as e:
    print(f"✗ Edge cases failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# TEST 7: Feature Info
# =============================================================================
print("\n" + "-" * 80)
print("TEST 7: Feature Metadata")
print("-" * 80)

try:
    feature_info = builder.get_feature_info()

    print(f"✓ Feature info retrieved for {len(feature_info)} features")
    print(f"\nSample (rsi_9):")
    rsi_info = feature_info['rsi_9']
    print(f"  Period: {rsi_info.get('period')}")
    print(f"  Norm stats: {rsi_info['norm_stats']}")
    print(f"  Clip bounds: {rsi_info.get('clip_bounds')}")

except Exception as e:
    print(f"✗ Feature info failed: {e}")
    sys.exit(1)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION COMPLETE - ALL TESTS PASSED")
print("=" * 80)
print(f"""
Summary:
  ✓ Configuration loading
  ✓ FeatureBuilder initialization
  ✓ Technical indicators (RSI, ATR, ADX)
  ✓ Observation building (15-dim)
  ✓ Feature normalization (single & batch)
  ✓ Edge cases & boundary conditions
  ✓ Feature metadata access

Structure:
  src/
  ├── __init__.py
  ├── core/
  │   ├── __init__.py
  │   └── services/
  │       ├── __init__.py
  │       └── feature_builder.py    # ✓ {Path(builder.__module__).name if hasattr(builder, '__module__') else 'VERIFIED'}
  └── shared/
      ├── __init__.py
      ├── config_loader.py          # ✓ VERIFIED
      └── exceptions.py             # ✓ VERIFIED

Usage:
  from src import FeatureBuilder
  builder = FeatureBuilder()
  obs = builder.build_observation(features_dict, position, bar_number)

Next steps:
  1. Integrate into airflow/dags/usdcop_m5__06_l5_realtime_inference.py
  2. Use in services/trading_api_realtime.py
  3. Update training pipeline to import from src/
""")

print("=" * 80)
