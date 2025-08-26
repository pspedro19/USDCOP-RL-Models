"""
Anti-Leakage Summary Script - Generate final report
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from minio import Minio
from io import BytesIO
from scipy.stats import pearsonr

# MinIO configuration
MINIO_CLIENT = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

BUCKET_NAME = 'ds-usdcop-rlready'
RUN_ID = 'RLREADY_20250822_174902'  # The actual available run
DATE = '2025-08-22'

def main():
    """Generate anti-leakage verification summary"""
    
    print("="*80)
    print("ANTI-LEAKAGE VERIFICATION SUMMARY")
    print("="*80)
    
    base_path = f"usdcop_m5__05_l4_rlready/market=usdcop/timeframe=m5/date={DATE}/run_id={RUN_ID}"
    
    # Load replay dataset
    try:
        response = MINIO_CLIENT.get_object(BUCKET_NAME, f"{base_path}/replay_dataset.parquet")
        df = pd.read_parquet(BytesIO(response.read()))
        response.close()
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}")
        return
    
    # Basic stats
    total_episodes = df['episode_id'].nunique()
    total_rows = len(df)
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    obs_features = len(obs_cols)
    
    print(f"\nDataset Information:")
    print(f"  Run ID: {RUN_ID}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Observation features: {obs_features}")
    print(f"  Date range: {df['time_utc'].min()} to {df['time_utc'].max()}")
    
    # Calculate forward returns
    df_sorted = df.sort_values(['episode_id', 't_in_episode']).copy()
    df_sorted['return_t1'] = df_sorted.groupby('episode_id')['mid_t'].pct_change().shift(-1)
    df_sorted['return_t2'] = df_sorted.groupby('episode_id')['mid_t'].pct_change(2).shift(-2)
    df_sorted['reward_window_return'] = df_sorted['return_t1'] + df_sorted['return_t2']
    
    # Calculate correlations
    print(f"\nAnti-Leakage Analysis:")
    print(f"Threshold: 10.0%")
    print(f"-" * 40)
    
    return_measures = {
        'Next Period Return (t+1)': 'return_t1',
        'Two-Period Return (t+2)': 'return_t2', 
        'Reward Window Return [t+1,t+2]': 'reward_window_return'
    }
    
    max_correlations = {}
    all_compliant = True
    
    for measure_name, return_col in return_measures.items():
        correlations = []
        valid_data = ~df_sorted[return_col].isna()
        
        for obs_col in obs_cols:
            obs_valid = valid_data & ~df_sorted[obs_col].isna()
            if obs_valid.sum() > 100:
                try:
                    corr, _ = pearsonr(
                        df_sorted.loc[obs_valid, obs_col], 
                        df_sorted.loc[obs_valid, return_col]
                    )
                    correlations.append((obs_col, abs(corr)))
                except:
                    continue
        
        if correlations:
            max_corr = max(correlations, key=lambda x: x[1])[1]
            max_correlations[measure_name] = max_corr
            compliance = max_corr < 0.10
            status = "PASS" if compliance else "FAIL"
            
            print(f"  {measure_name:30s}: {max_corr:.6f} | {status}")
            
            if not compliance:
                all_compliant = False
    
    # Final verdict
    print(f"\n" + "="*80)
    if all_compliant:
        print(f"FINAL VERDICT: ANTI-LEAKAGE VERIFICATION PASS")
        print(f"  - All correlations below 10% threshold")
        print(f"  - Maximum correlation: {max(max_correlations.values()):.6f}")
        print(f"  - No future information detected in observations")
        print(f"  - Reward calculation window properly isolated")
    else:
        print(f"FINAL VERDICT: ANTI-LEAKAGE VERIFICATION FAIL")
        print(f"  - Some correlations exceed 10% threshold")
    print(f"="*80)
    
    # Create summary report
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_id': RUN_ID,
        'dataset_summary': {
            'episodes': total_episodes,
            'total_rows': total_rows,
            'observation_features': obs_features,
            'date_range': f"{df['time_utc'].min()} to {df['time_utc'].max()}"
        },
        'anti_leakage_results': {
            'threshold': 0.10,
            'overall_compliant': all_compliant,
            'max_correlations': max_correlations,
            'max_overall_correlation': max(max_correlations.values()) if max_correlations else 0
        },
        'conclusion': {
            'status': 'PASS' if all_compliant else 'FAIL',
            'future_information_detected': not all_compliant,
            'reward_window_isolated': all_compliant
        }
    }
    
    # Save report
    with open(f'anti_leakage_summary_{RUN_ID}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary report saved: anti_leakage_summary_{RUN_ID}.json")
    
    return summary

if __name__ == "__main__":
    summary = main()