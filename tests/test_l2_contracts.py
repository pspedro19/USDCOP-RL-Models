"""
CI Tests for L2 Contract Enforcement
=====================================
Fail the build if data_premium_strict schema deviates
"""

import pytest
import pandas as pd
import numpy as np
import json
from typing import Dict, List

# FROZEN CONTRACT - DO NOT MODIFY
REQUIRED_COLUMNS = [
    'episode_id', 't_in_episode', 'time_utc', 'time_cot',
    'open', 'high', 'low', 'close',
    'ret_log_5m', 'range_bps',
    'ret_deseason', 'range_norm', 'winsor_flag'
]

OPTIONAL_COLUMNS = [
    'spread_bps', 'is_low_liquidity_episode', 'is_repeated_ohlc'
]

EXPECTED_DTYPES = {
    'episode_id': ['object', 'datetime64[ns]'],  # Can be string or date
    't_in_episode': ['int64'],
    'time_utc': ['datetime64[ns]'],
    'time_cot': ['datetime64[ns]'],
    'open': ['float64'],
    'high': ['float64'],
    'low': ['float64'],
    'close': ['float64'],
    'ret_log_5m': ['float64'],
    'range_bps': ['float64'],
    'ret_deseason': ['float64'],
    'range_norm': ['float64'],
    'winsor_flag': ['bool']
}

def test_schema_compliance(df: pd.DataFrame):
    """Test that data_premium_strict meets schema requirements"""
    
    # Test 1: Required columns present
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    assert len(missing_cols) == 0, f"Missing required columns: {missing_cols}"
    
    # Test 2: No unexpected columns (except optional)
    expected_cols = set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    extra_cols = set(df.columns) - expected_cols
    # Allow some flexibility for debugging columns
    allowed_extra = {'warn_reason', 'debug_flag'}
    unexpected = extra_cols - allowed_extra
    assert len(unexpected) == 0, f"Unexpected columns: {unexpected}"
    
    # Test 3: Data types match
    for col, expected_types in EXPECTED_DTYPES.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            assert actual_dtype in expected_types, \
                f"Column {col} has dtype {actual_dtype}, expected one of {expected_types}"

def test_data_invariants(df: pd.DataFrame):
    """Test that data meets all invariants"""
    
    # Test 1: Episodes have exactly 60 bars
    bars_per_episode = df.groupby('episode_id').size()
    assert (bars_per_episode == 60).all(), \
        f"Episodes with != 60 bars: {bars_per_episode[bars_per_episode != 60].to_dict()}"
    
    # Test 2: t_in_episode ranges from 0 to 59
    assert df['t_in_episode'].min() == 0, f"Min t_in_episode = {df['t_in_episode'].min()}"
    assert df['t_in_episode'].max() == 59, f"Max t_in_episode = {df['t_in_episode'].max()}"
    
    # Test 3: No duplicate (episode_id, t_in_episode)
    assert not df.duplicated(['episode_id', 't_in_episode']).any(), \
        "Duplicate (episode_id, t_in_episode) pairs found"
    
    # Test 4: 300-second grid
    time_diffs = df.sort_values(['episode_id', 't_in_episode']) \
                   .groupby('episode_id')['time_utc'] \
                   .diff().dropna().dt.total_seconds()
    assert (time_diffs == 300).all(), \
        f"Non-300s intervals found: {time_diffs[time_diffs != 300].value_counts().to_dict()}"
    
    # Test 5: OHLC relationships
    assert (df['high'] >= df['low']).all(), "High < Low violations found"
    assert (df['high'] >= df['open']).all(), "High < Open violations found"
    assert (df['high'] >= df['close']).all(), "High < Close violations found"
    assert (df['low'] <= df['open']).all(), "Low > Open violations found"
    assert (df['low'] <= df['close']).all(), "Low > Close violations found"

def test_normalization_ranges(df: pd.DataFrame):
    """Test that normalized values are in expected ranges"""
    
    # Test 1: ret_deseason should be roughly standard normal
    ret_std = df['ret_deseason'].std()
    assert 0.7 <= ret_std <= 1.3, f"ret_deseason std = {ret_std}, outside [0.7, 1.3]"
    
    ret_mean = abs(df['ret_deseason'].mean())
    assert ret_mean <= 0.1, f"ret_deseason mean = {ret_mean}, not centered"
    
    # Test 2: Winsor rate should be low
    winsor_rate = df['winsor_flag'].mean()
    assert winsor_rate <= 0.015, f"Winsor rate = {winsor_rate*100:.2f}%, > 1.5%"
    
    # Test 3: range_norm should be positive
    assert (df['range_norm'] >= 0).all(), "Negative range_norm found"
    
    # Test 4: No extreme outliers after winsorization
    assert df['ret_deseason'].abs().max() <= 10, \
        f"Extreme ret_deseason: {df['ret_deseason'].abs().max()}"

def test_no_repeated_ohlc_strict(df: pd.DataFrame):
    """Test that STRICT dataset has no repeated OHLC"""
    
    # Filter for STRICT (exclude low liquidity if flagged)
    if 'is_low_liquidity_episode' in df.columns:
        df_strict = df[~df['is_low_liquidity_episode']]
    else:
        df_strict = df
    
    # Check each episode
    for episode_id in df_strict['episode_id'].unique():
        episode_df = df_strict[df_strict['episode_id'] == episode_id].sort_values('t_in_episode')
        
        if len(episode_df) > 1:
            ohlc_cols = ['open', 'high', 'low', 'close']
            
            for i in range(1, len(episode_df)):
                curr = episode_df.iloc[i][ohlc_cols].values
                prev = episode_df.iloc[i-1][ohlc_cols].values
                
                assert not np.array_equal(curr, prev), \
                    f"Repeated OHLC found in STRICT episode {episode_id} at bar {i}"

def test_hod_baseline_determinism():
    """Test that HOD baseline calculation is deterministic"""
    
    # Create a small fixture
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    
    fixture_df = pd.DataFrame({
        'time_utc': dates,
        'hour_cot': dates.hour,
        'ret_log_1': np.random.randn(1000) * 0.001,
        'range_bps': np.abs(np.random.randn(1000)) * 10
    })
    
    # Compute HOD baseline twice
    from l2_audit_improvements import apply_frozen_deseasonalization
    
    _, norm_ref1 = apply_frozen_deseasonalization(fixture_df.copy(), '2024-02-01')
    _, norm_ref2 = apply_frozen_deseasonalization(fixture_df.copy(), '2024-02-01')
    
    # Should be identical
    assert json.dumps(norm_ref1, sort_keys=True) == json.dumps(norm_ref2, sort_keys=True), \
        "HOD baseline calculation is not deterministic"

# Pytest fixtures
@pytest.fixture
def sample_l2_data():
    """Load sample L2 data for testing"""
    # In production, this would load from S3/MinIO
    # For testing, create synthetic data
    
    episodes = 10
    bars_per_episode = 60
    
    data = []
    for ep in range(episodes):
        base_time = pd.Timestamp(f'2024-01-{ep+1:02d} 08:00:00', tz='UTC')
        
        for t in range(bars_per_episode):
            time_utc = base_time + pd.Timedelta(minutes=5*t)
            
            data.append({
                'episode_id': f'2024-01-{ep+1:02d}',
                't_in_episode': t,
                'time_utc': time_utc,
                'time_cot': time_utc,
                'open': 3000 + np.random.randn(),
                'high': 3001 + abs(np.random.randn()),
                'low': 2999 - abs(np.random.randn()),
                'close': 3000 + np.random.randn(),
                'ret_log_5m': np.random.randn() * 0.001,
                'range_bps': abs(np.random.randn()) * 10,
                'ret_deseason': np.random.randn(),
                'range_norm': abs(np.random.randn()),
                'winsor_flag': np.random.random() < 0.01
            })
    
    df = pd.DataFrame(data)
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df

def test_full_compliance(sample_l2_data):
    """Run all compliance tests"""
    test_schema_compliance(sample_l2_data)
    test_data_invariants(sample_l2_data)
    test_normalization_ranges(sample_l2_data)
    test_no_repeated_ohlc_strict(sample_l2_data)
    print("✅ All L2 contract tests passed")

if __name__ == "__main__":
    # Run tests
    df = sample_l2_data()
    test_full_compliance(df)
    test_hod_baseline_determinism()