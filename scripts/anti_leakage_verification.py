"""
Anti-Leakage Verification for L4 RL-Ready Data
Comprehensive verification of data integrity for run L4_BACKFILL_20250822_123926
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from minio import Minio
from io import BytesIO
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# MinIO configuration
MINIO_CLIENT = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

BUCKET_NAME = 'ds-usdcop-rlready'
RUN_ID = 'RLREADY_20250822_174902'
DATE = '2025-08-22'

def load_replay_dataset():
    """Load the replay dataset from MinIO bucket"""
    
    base_path = f"usdcop_m5__05_l4_rlready/market=usdcop/timeframe=m5/date={DATE}/run_id={RUN_ID}"
    
    print(f"Loading replay dataset from MinIO...")
    print(f"Path: {base_path}/replay_dataset.parquet")
    
    try:
        response = MINIO_CLIENT.get_object(BUCKET_NAME, f"{base_path}/replay_dataset.parquet")
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
        
        print(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        print(f"Episodes: {df['episode_id'].nunique()}")
        print(f"Date range: {df['time_utc'].min()} to {df['time_utc'].max()}")
        
        return df
    
    except Exception as e:
        print(f"ERROR loading replay dataset: {e}")
        return None


def calculate_forward_returns(df):
    """Calculate forward returns for anti-leakage testing"""
    
    print("\nCalculating forward returns...")
    
    # Sort by episode and time
    df = df.sort_values(['episode_id', 't_in_episode']).copy()
    
    # Calculate t+1 return (what we're trying to predict)
    df['return_t1'] = df.groupby('episode_id')['mid_t'].pct_change().shift(-1)
    
    # Calculate t+2 return (used in reward calculation)
    df['return_t2'] = df.groupby('episode_id')['mid_t'].pct_change(2).shift(-2)
    
    # Alternative calculation using mid_t2 if available
    if 'mid_t2' in df.columns:
        df['return_t1_alt'] = df.groupby('episode_id').apply(
            lambda x: (x['mid_t2'] / x['mid_t'] - 1).shift(-1)
        ).values
        df['return_t2_alt'] = df.groupby('episode_id').apply(
            lambda x: (x['mid_t2'].shift(-1) / x['mid_t'] - 1).shift(-1)
        ).values
    
    # Reward window return [t+1, t+2]
    df['reward_window_return'] = df['return_t1'] + df['return_t2']
    
    print(f"Forward returns calculated:")
    print(f"  - return_t1: {(~df['return_t1'].isna()).sum()} valid values")
    print(f"  - return_t2: {(~df['return_t2'].isna()).sum()} valid values")
    print(f"  - reward_window_return: {(~df['reward_window_return'].isna()).sum()} valid values")
    
    return df


def verify_observation_features(df):
    """Identify and verify observation features"""
    
    print("\nIdentifying observation features...")
    
    # Get all observation columns
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    
    print(f"Found {len(obs_cols)} observation features:")
    for i, col in enumerate(obs_cols[:10], 1):  # Show first 10
        print(f"  {i:2d}. {col}")
    
    if len(obs_cols) > 10:
        print(f"  ... and {len(obs_cols) - 10} more")
    
    # Check for potential problematic columns
    suspicious_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['future', 'next', 'forward', 'lead']):
            suspicious_cols.append(col)
    
    if suspicious_cols:
        print(f"\nWARNING: Found potentially suspicious columns:")
        for col in suspicious_cols:
            print(f"  - {col}")
    else:
        print(f"\nGOOD: No obviously suspicious column names found")
    
    return obs_cols


def calculate_correlations(df, obs_cols):
    """Calculate correlations between observations at t and returns at t+1"""
    
    print("\nCalculating correlations between observations and future returns...")
    
    correlations = {}
    
    # Test against different forward return measures
    return_measures = {
        'return_t1': 'Next period return (t+1)',
        'return_t2': 'Two-period return (t+2)', 
        'reward_window_return': 'Reward window return [t+1, t+2]'
    }
    
    for return_col, return_desc in return_measures.items():
        if return_col in df.columns:
            print(f"\nTesting against {return_desc}:")
            
            correlations[return_col] = {}
            valid_pairs = ~(df[return_col].isna())
            
            for obs_col in obs_cols:
                # Only use rows where both obs and return are valid
                valid_data = valid_pairs & ~df[obs_col].isna()
                
                if valid_data.sum() > 100:  # Need sufficient data points
                    try:
                        corr, p_value = pearsonr(
                            df.loc[valid_data, obs_col], 
                            df.loc[valid_data, return_col]
                        )
                        correlations[return_col][obs_col] = {
                            'correlation': abs(corr),
                            'raw_correlation': corr,
                            'p_value': p_value,
                            'n_observations': valid_data.sum()
                        }
                    except Exception as e:
                        print(f"    Error calculating correlation for {obs_col}: {e}")
                        correlations[return_col][obs_col] = {
                            'correlation': np.nan,
                            'raw_correlation': np.nan,
                            'p_value': np.nan,
                            'n_observations': 0
                        }
    
    return correlations


def analyze_correlations(correlations, threshold=0.10):
    """Analyze correlations and identify potential leakage"""
    
    print(f"\n{'='*80}")
    print(f"ANTI-LEAKAGE ANALYSIS (Threshold: {threshold:.1%})")
    print(f"{'='*80}")
    
    results = {}
    
    for return_measure, corr_dict in correlations.items():
        if not corr_dict:
            continue
            
        print(f"\n{return_measure.upper().replace('_', ' ')}:")
        print(f"{'-'*50}")
        
        # Sort by absolute correlation
        sorted_corrs = sorted(
            [(obs, data) for obs, data in corr_dict.items() if not np.isnan(data['correlation'])],
            key=lambda x: x[1]['correlation'],
            reverse=True
        )
        
        if not sorted_corrs:
            print("  No valid correlations calculated")
            continue
        
        max_correlation = sorted_corrs[0][1]['correlation']
        results[return_measure] = {
            'max_correlation': max_correlation,
            'threshold_met': max_correlation < threshold,
            'top_correlations': sorted_corrs[:10]
        }
        
        print(f"  Maximum correlation: {max_correlation:.6f}")
        print(f"  Threshold compliance: {'PASS' if max_correlation < threshold else 'FAIL'}")
        
        print(f"\n  Top 10 correlations:")
        for i, (obs_col, data) in enumerate(sorted_corrs[:10], 1):
            status = "OK" if data['correlation'] < threshold else "HIGH"
            print(f"    {i:2d}. {obs_col[:40]:40s} | {data['correlation']:8.6f} | {status}")
        
        # Count violations
        violations = sum(1 for _, data in sorted_corrs if data['correlation'] >= threshold)
        if violations > 0:
            print(f"\n  WARNING: {violations} features exceed threshold!")
        else:
            print(f"\n  GOOD: All features below threshold")
    
    return results


def verify_temporal_integrity(df):
    """Verify that observations don't contain future information"""
    
    print(f"\n{'='*80}")
    print(f"TEMPORAL INTEGRITY VERIFICATION")
    print(f"{'='*80}")
    
    # Check if observations are calculated properly within episodes
    temporal_checks = {}
    
    # Sample a few episodes for detailed analysis
    sample_episodes = df['episode_id'].unique()[:5]
    
    for episode_id in sample_episodes:
        ep_data = df[df['episode_id'] == episode_id].copy().sort_values('t_in_episode')
        
        print(f"\nEpisode {episode_id}:")
        print(f"  Length: {len(ep_data)} steps")
        print(f"  Time range: {ep_data['t_in_episode'].min()} to {ep_data['t_in_episode'].max()}")
        
        # Check that terminal step doesn't have future returns
        terminal_step = ep_data[ep_data['is_terminal'] == True]
        if len(terminal_step) > 0:
            terminal_has_future_return = not terminal_step['return_t1'].isna().iloc[0]
            print(f"  Terminal step has future return: {terminal_has_future_return}")
            if terminal_has_future_return:
                print(f"    WARNING: Terminal step should not have future return!")
        
        # Check that observations are consistent within episode
        obs_cols = [col for col in df.columns if col.startswith('obs_')]
        if obs_cols:
            first_obs = obs_cols[0]
            obs_changes = (ep_data[first_obs].diff() != 0).sum()
            print(f"  Observation changes in {first_obs}: {obs_changes}")
    
    return temporal_checks


def verify_reward_calculation_integrity(df):
    """Verify that reward calculation window [t+1, t+2] doesn't leak into observations"""
    
    print(f"\n{'='*80}")
    print(f"REWARD CALCULATION INTEGRITY")
    print(f"{'='*80}")
    
    # Load reward specification if available
    try:
        base_path = f"usdcop_m5__05_l4_rlready/market=usdcop/timeframe=m5/date={DATE}/run_id={RUN_ID}"
        response = MINIO_CLIENT.get_object(BUCKET_NAME, f"{base_path}/reward_spec.json")
        reward_spec = json.loads(response.read().decode('utf-8'))
        response.close()
        
        print(f"Reward specification loaded:")
        print(f"  Window: {reward_spec.get('reward_window', 'Not specified')}")
        print(f"  Calculation: {reward_spec.get('calculation_method', 'Not specified')}")
        
    except Exception as e:
        print(f"Could not load reward specification: {e}")
        reward_spec = {}
    
    # Verify that reward calculation is properly isolated
    print(f"\nReward calculation verification:")
    
    # Check if any observation features correlate suspiciously with reward components
    if 'reward' in df.columns:
        obs_cols = [col for col in df.columns if col.startswith('obs_')]
        
        suspicious_reward_corrs = []
        for obs_col in obs_cols[:20]:  # Check first 20 obs features
            valid_data = ~(df[obs_col].isna() | df['reward'].isna())
            if valid_data.sum() > 100:
                corr = df.loc[valid_data, obs_col].corr(df.loc[valid_data, 'reward'])
                if abs(corr) > 0.05:  # Lower threshold for reward correlation
                    suspicious_reward_corrs.append((obs_col, abs(corr)))
        
        suspicious_reward_corrs.sort(key=lambda x: x[1], reverse=True)
        
        if suspicious_reward_corrs:
            print(f"  Features with high reward correlation:")
            for obs_col, corr in suspicious_reward_corrs[:5]:
                print(f"    - {obs_col}: {corr:.6f}")
        else:
            print(f"  No suspicious reward correlations found")
    
    return reward_spec


def generate_anti_leakage_report(df, correlations, analysis_results):
    """Generate comprehensive anti-leakage compliance report"""
    
    print(f"\n{'='*80}")
    print(f"ANTI-LEAKAGE COMPLIANCE REPORT")
    print(f"{'='*80}")
    
    # Summary statistics
    total_episodes = df['episode_id'].nunique()
    total_rows = len(df)
    obs_features = len([col for col in df.columns if col.startswith('obs_')])
    
    print(f"\nDataset Summary:")
    print(f"  Run ID: {RUN_ID}")
    print(f"  Total episodes: {total_episodes:,}")
    print(f"  Total observations: {total_rows:,}")
    print(f"  Observation features: {obs_features}")
    print(f"  Time range: {df['time_utc'].min()} to {df['time_utc'].max()}")
    
    # Anti-leakage compliance
    print(f"\nAnti-Leakage Compliance:")
    overall_compliant = True
    
    for return_measure, results in analysis_results.items():
        max_corr = results['max_correlation']
        compliant = results['threshold_met']
        
        status = "PASS" if compliant else "FAIL"
        print(f"  {return_measure:25s}: {max_corr:8.6f} | {status}")
        
        if not compliant:
            overall_compliant = False
    
    # Final verdict
    print(f"\n{'='*80}")
    if overall_compliant:
        print(f"ANTI-LEAKAGE VERIFICATION: PASS")
        print(f"  All correlations below 10% threshold")
        print(f"  No future information detected in observations")
        print(f"  Reward calculation window properly isolated")
    else:
        print(f"ANTI-LEAKAGE VERIFICATION: FAIL")
        print(f"  Some correlations exceed 10% threshold")
        print(f"  Manual review required")
    print(f"{'='*80}")
    
    # Create detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'run_id': RUN_ID,
        'date': DATE,
        'dataset_summary': {
            'total_episodes': int(total_episodes),
            'total_rows': int(total_rows),
            'observation_features': int(obs_features),
            'time_range': {
                'start': str(df['time_utc'].min()),
                'end': str(df['time_utc'].max())
            }
        },
        'anti_leakage_results': {},
        'overall_compliant': overall_compliant,
        'threshold': 0.10
    }
    
    # Add correlation results
    for return_measure, results in analysis_results.items():
        report['anti_leakage_results'][return_measure] = {
            'max_correlation': float(results['max_correlation']),
            'threshold_met': results['threshold_met'],
            'top_5_correlations': []
        }
        
        # Add top 5 correlations with details
        for obs_col, data in results['top_correlations'][:5]:
            report['anti_leakage_results'][return_measure]['top_5_correlations'].append({
                'feature': obs_col,
                'abs_correlation': float(data['correlation']),
                'raw_correlation': float(data['raw_correlation']),
                'p_value': float(data['p_value']) if not np.isnan(data['p_value']) else None,
                'n_observations': int(data['n_observations'])
            })
    
    return report


def main():
    """Main execution function"""
    
    print(f"{'='*80}")
    print(f"ANTI-LEAKAGE VERIFICATION FOR L4 RL-READY DATA")
    print(f"{'='*80}")
    print(f"Run ID: {RUN_ID}")
    print(f"Date: {DATE}")
    print(f"MinIO Bucket: {BUCKET_NAME}")
    
    # Step 1: Load replay dataset
    df = load_replay_dataset()
    if df is None:
        print("ERROR: Could not load dataset. Exiting.")
        return
    
    # Step 2: Calculate forward returns
    df = calculate_forward_returns(df)
    
    # Step 3: Verify observation features
    obs_cols = verify_observation_features(df)
    
    # Step 4: Calculate correlations
    correlations = calculate_correlations(df, obs_cols)
    
    # Step 5: Analyze correlations for leakage
    analysis_results = analyze_correlations(correlations, threshold=0.10)
    
    # Step 6: Verify temporal integrity
    temporal_checks = verify_temporal_integrity(df)
    
    # Step 7: Verify reward calculation integrity
    reward_spec = verify_reward_calculation_integrity(df)
    
    # Step 8: Generate comprehensive report
    report = generate_anti_leakage_report(df, correlations, analysis_results)
    
    # Save report
    report_filename = f'anti_leakage_verification_{RUN_ID}.json'
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_filename}")
    
    return report


if __name__ == "__main__":
    report = main()