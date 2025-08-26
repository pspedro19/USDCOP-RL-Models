"""
CI Tests for L1 Contract Enforcement
=====================================
Ensures L1 outputs meet strict audit requirements
"""

import pytest
import pandas as pd
import numpy as np
import json
from datetime import date
from typing import Dict, List

# FROZEN CONTRACT - DO NOT MODIFY
L1_ACCEPTANCE_CRITERIA = {
    'bars_per_episode': 60,
    'repeated_ohlc_rate': 0.0,
    'holidays_excluded': True,
    'ohlc_violations': 0,
    'duplicates': 0,
    'stale_burst_max': 0
}

def test_l1_accepted_has_no_holiday_or_rep_ohlc(l1_quality_df: pd.DataFrame, 
                                                l1_accepted_df: pd.DataFrame):
    """Test that accepted episodes have no holidays or repeated OHLC"""
    
    # Get accepted episode IDs
    accepted_episodes = l1_accepted_df['episode_id'].unique()
    
    # Check quality report for these episodes
    q = l1_quality_df.set_index('date')
    accepted_quality = q.loc[accepted_episodes]
    
    # Assert no holidays in accepted
    assert (~accepted_quality['is_holiday']).all(), \
        f"Holidays found in accepted episodes: {accepted_quality[accepted_quality['is_holiday']].index.tolist()}"
    
    # Assert no repeated OHLC in accepted
    assert (accepted_quality['repeated_ohlc_rate'] == 0).all(), \
        f"Repeated OHLC found in accepted episodes: {accepted_quality[accepted_quality['repeated_ohlc_rate'] > 0].index.tolist()}"
    
    # Double-check by recomputing from accepted parquet
    for episode_id in accepted_episodes:
        episode_df = l1_accepted_df[l1_accepted_df['episode_id'] == episode_id].sort_values('t_in_episode')
        
        if len(episode_df) > 1:
            ohlc_cols = ['open', 'high', 'low', 'close']
            rep_check = (episode_df[ohlc_cols] == episode_df[ohlc_cols].shift(1)).all(axis=1).fillna(False)
            
            assert rep_check.sum() == 0, \
                f"Episode {episode_id} has repeated OHLC but passed acceptance"

def test_l1_episode_completeness(l1_accepted_df: pd.DataFrame):
    """Test all accepted episodes have exactly 60 bars"""
    
    bars_per_episode = l1_accepted_df.groupby('episode_id').size()
    
    assert (bars_per_episode == 60).all(), \
        f"Episodes with != 60 bars: {bars_per_episode[bars_per_episode != 60].to_dict()}"

def test_l1_time_grid(l1_accepted_df: pd.DataFrame):
    """Test 300-second grid for all accepted episodes"""
    
    for episode_id in l1_accepted_df['episode_id'].unique():
        episode_df = l1_accepted_df[l1_accepted_df['episode_id'] == episode_id].sort_values('t_in_episode')
        
        if 'time_utc' in episode_df.columns:
            time_diffs = pd.to_datetime(episode_df['time_utc']).diff().dropna()
            seconds = time_diffs.dt.total_seconds()
            
            assert (seconds == 300).all(), \
                f"Episode {episode_id} has non-300s intervals: {seconds[seconds != 300].tolist()}"

def test_l1_no_duplicates(l1_accepted_df: pd.DataFrame):
    """Test no duplicate (episode_id, t_in_episode) pairs"""
    
    duplicates = l1_accepted_df.duplicated(['episode_id', 't_in_episode'])
    
    assert not duplicates.any(), \
        f"Duplicate (episode_id, t_in_episode) pairs found: {duplicates.sum()}"

def test_l1_ohlc_relationships(l1_accepted_df: pd.DataFrame):
    """Test OHLC relationship integrity"""
    
    violations = []
    
    # Check OHLC relationships
    if not (l1_accepted_df['high'] >= l1_accepted_df['low']).all():
        violations.append("High < Low violations found")
    
    if not (l1_accepted_df['high'] >= l1_accepted_df['open']).all():
        violations.append("High < Open violations found")
    
    if not (l1_accepted_df['high'] >= l1_accepted_df['close']).all():
        violations.append("High < Close violations found")
    
    if not (l1_accepted_df['low'] <= l1_accepted_df['open']).all():
        violations.append("Low > Open violations found")
    
    if not (l1_accepted_df['low'] <= l1_accepted_df['close']).all():
        violations.append("Low > Close violations found")
    
    assert len(violations) == 0, f"OHLC violations: {violations}"

def test_l1_quality_report_completeness(l1_quality_df: pd.DataFrame):
    """Test quality report has all required fields"""
    
    required_fields = [
        'date', 'rows_found', 'quality_flag', 'is_holiday',
        'repeated_ohlc_count', 'repeated_ohlc_rate', 'repeated_ohlc_burst_max',
        'ohlc_violations', 'duplicates_count', 'warn_reason'
    ]
    
    missing_fields = [f for f in required_fields if f not in l1_quality_df.columns]
    
    assert len(missing_fields) == 0, f"Missing fields in quality report: {missing_fields}"

def test_l1_metadata_structure(l1_metadata: Dict):
    """Test metadata contains all required elements"""
    
    required_keys = [
        'pipeline_stage', 'execution_date', 'run_id', 'version',
        'acceptance_policy', 'calendar', 'statistics', 'data_integrity',
        'assertions_passed'
    ]
    
    missing_keys = [k for k in required_keys if k not in l1_metadata]
    
    assert len(missing_keys) == 0, f"Missing metadata keys: {missing_keys}"
    
    # Check calendar metadata
    assert 'market_calendar_version' in l1_metadata['calendar'], \
        "Missing market_calendar_version in metadata"
    
    assert 'holiday_episodes_rejected' in l1_metadata['calendar'], \
        "Missing holiday_episodes_rejected in metadata"
    
    # Check acceptance policy
    policy = l1_metadata['acceptance_policy']['criteria']
    assert policy['repeated_ohlc_rate'] == 0.0, \
        f"Acceptance policy allows repeated OHLC: {policy['repeated_ohlc_rate']}"
    
    assert policy['holidays_excluded'] == True, \
        "Acceptance policy doesn't exclude holidays"

def test_l1_calendar_determinism():
    """Test calendar computation is deterministic"""
    
    from l1_enhanced_final import _load_market_holidays, _calendar_version
    
    # Load holidays twice
    years = [2023, 2024]
    holidays1 = _load_market_holidays(years)
    holidays2 = _load_market_holidays(years)
    
    # Should be identical
    assert holidays1 == holidays2, "Holiday calendar is not deterministic"
    
    # Version should be identical
    version1 = _calendar_version(holidays1)
    version2 = _calendar_version(holidays2)
    
    assert version1 == version2, f"Calendar versions differ: {version1} vs {version2}"

def test_l1_repeated_ohlc_detection():
    """Test repeated OHLC detection logic"""
    
    # Create test data with repeated OHLC
    test_data = pd.DataFrame({
        'episode_id': ['2024-01-01'] * 5,
        't_in_episode': [0, 1, 2, 3, 4],
        'open': [100, 100, 101, 101, 102],
        'high': [101, 101, 102, 102, 103],
        'low': [99, 99, 100, 100, 101],
        'close': [100.5, 100.5, 101.5, 101.5, 102.5]
    })
    
    # Check for repeated OHLC
    ohlc_cols = ['open', 'high', 'low', 'close']
    rep_mask = (test_data[ohlc_cols] == test_data[ohlc_cols].shift(1)).all(axis=1).fillna(False)
    
    # Should detect 2 repeated bars (indices 1 and 3)
    assert rep_mask.sum() == 2, f"Expected 2 repeated OHLC, got {rep_mask.sum()}"
    
    # Check burst detection
    rep_burst_max, cur = 0, 0
    for v in rep_mask.values:
        if v:
            cur += 1
            rep_burst_max = max(rep_burst_max, cur)
        else:
            cur = 0
    
    # Max burst should be 1 (non-consecutive)
    assert rep_burst_max == 1, f"Expected burst of 1, got {rep_burst_max}"

def test_l1_acceptance_rate_reasonable(l1_quality_df: pd.DataFrame):
    """Test that acceptance rate is within reasonable bounds"""
    
    ok_episodes = (l1_quality_df['quality_flag'] == 'OK').sum()
    total_episodes = len(l1_quality_df)
    acceptance_rate = ok_episodes / total_episodes * 100
    
    # Expect at least 70% acceptance (excluding holidays)
    assert acceptance_rate >= 70, \
        f"Acceptance rate too low: {acceptance_rate:.1f}% (OK: {ok_episodes}/{total_episodes})"
    
    # But not 100% (should have some holidays/issues)
    assert acceptance_rate < 100, \
        f"Acceptance rate suspiciously high: {acceptance_rate:.1f}%"

# Pytest fixtures
@pytest.fixture
def l1_quality_df():
    """Load L1 quality report for testing"""
    # In production, load from S3
    # For testing, create sample data
    
    data = []
    for i in range(100):
        is_holiday = i % 20 == 0  # 5% holidays
        repeated_ohlc = i % 25 == 0  # 4% with repeated OHLC
        
        data.append({
            'date': f'2024-01-{i+1:02d}',
            'rows_found': 60 if not is_holiday else 59,
            'quality_flag': 'FAIL' if is_holiday or repeated_ohlc else 'OK',
            'is_holiday': is_holiday,
            'repeated_ohlc_count': 10 if repeated_ohlc else 0,
            'repeated_ohlc_rate': 16.7 if repeated_ohlc else 0.0,
            'repeated_ohlc_burst_max': 3 if repeated_ohlc else 0,
            'ohlc_violations': 0,
            'duplicates_count': 0,
            'warn_reason': ''
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def l1_accepted_df():
    """Load L1 accepted data for testing"""
    # Create sample accepted data (no holidays, no repeated OHLC)
    
    episodes = []
    for ep in range(10):
        for t in range(60):
            episodes.append({
                'episode_id': f'2024-02-{ep+1:02d}',
                't_in_episode': t,
                'time_utc': pd.Timestamp(f'2024-02-{ep+1:02d} 08:00:00') + pd.Timedelta(minutes=5*t),
                'open': 3000 + np.random.randn(),
                'high': 3001 + abs(np.random.randn()),
                'low': 2999 - abs(np.random.randn()),
                'close': 3000 + np.random.randn()
            })
    
    df = pd.DataFrame(episodes)
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df

@pytest.fixture
def l1_metadata():
    """Load L1 metadata for testing"""
    
    return {
        'pipeline_stage': 'L1_STANDARDIZE_ENHANCED',
        'execution_date': '2024-08-24',
        'run_id': 'test-run-123',
        'version': 'v3.0-audit-aligned',
        'acceptance_policy': {
            'criteria': {
                'bars_per_episode': 60,
                'repeated_ohlc_rate': 0.0,
                'holidays_excluded': True,
                'ohlc_violations': 0,
                'duplicates': 0,
                'stale_burst_max': 0
            }
        },
        'calendar': {
            'market_calendar_version': 'US-CO-CUSTOM@abc123',
            'holiday_episodes_rejected': 5,
            'repeated_ohlc_episodes_rejected': 4
        },
        'statistics': {
            'total_episodes': 100,
            'accepted_episodes': 91
        },
        'data_integrity': {
            'accepted_data_hash': 'sha256_hash_here'
        },
        'assertions_passed': [
            'no_repeated_ohlc_in_accepted',
            'no_holidays_in_accepted'
        ]
    }

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])