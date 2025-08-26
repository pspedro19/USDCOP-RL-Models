"""
L4 Comprehensive Audit System (2020-2025)
Complete validation of L3→L4 pipeline ensuring all requirements are met
"""

import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
import pytz
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class L4ComprehensiveAuditor:
    """
    Complete L4 auditor that validates:
    - L3→L4 traceability
    - Replay dataset schema and invariants
    - Episodes index consistency
    - Environment specifications
    - Reward and cost models
    - Data quality and coverage
    """
    
    def __init__(self, l3_path: str, l4_path: str):
        self.l3_path = Path(l3_path)
        self.l4_path = Path(l4_path)
        self.audit_results = {}
        self.cot_tz = pytz.timezone('America/Bogota')
        self.utc_tz = pytz.UTC
        
    def run_full_audit(self) -> Dict:
        """Execute complete L4 audit"""
        print("\n" + "="*80)
        print(" L4 COMPREHENSIVE AUDIT SYSTEM (2020-2025)")
        print("="*80)
        
        # 1. Traceability L3→L4
        self.audit_results['traceability'] = self._audit_traceability()
        
        # 2. Replay dataset schema and invariants
        self.audit_results['replay_dataset'] = self._audit_replay_dataset()
        
        # 3. Episodes index
        self.audit_results['episodes_index'] = self._audit_episodes_index()
        
        # 4. Environment specifications
        self.audit_results['env_spec'] = self._audit_env_spec()
        
        # 5. Reward and cost models
        self.audit_results['reward_cost'] = self._audit_reward_cost_models()
        
        # 6. Action and split specifications
        self.audit_results['action_split'] = self._audit_action_split_specs()
        
        # 7. Data quality checks
        self.audit_results['quality'] = self._audit_data_quality()
        
        # 8. Coverage analysis
        self.audit_results['coverage'] = self._audit_coverage()
        
        # 9. Generate final report
        self._generate_final_report()
        
        return self.audit_results
    
    def _audit_traceability(self) -> Dict:
        """Audit 1: L3->L4 traceability"""
        print("\n[1/9] Auditing L3->L4 Traceability...")
        
        results = {
            'status': 'PENDING',
            'checks': {},
            'issues': []
        }
        
        try:
            # Load L4 metadata
            metadata_path = self.l4_path / 'metadata.json'
            if not metadata_path.exists():
                results['issues'].append("metadata.json not found in L4")
                results['status'] = 'FAIL'
                return results
                
            with open(metadata_path, 'r') as f:
                l4_metadata = json.load(f)
            
            # Check L3 references
            if 'l3_inputs' in l4_metadata:
                l3_refs = l4_metadata['l3_inputs']
                
                # Verify L3 files exist
                required_l3_files = ['features.parquet', 'feature_spec.json', 'leakage_gate.json']
                for file in required_l3_files:
                    file_path = self.l3_path / file
                    if file_path.exists():
                        # Calculate hash
                        actual_hash = self._calculate_file_hash(file_path)
                        expected_hash = l3_refs.get(f'{file}_hash', None)
                        
                        if expected_hash and actual_hash != expected_hash:
                            results['issues'].append(f"Hash mismatch for {file}")
                        results['checks'][f'l3_{file}'] = 'OK' if not expected_hash or actual_hash == expected_hash else 'FAIL'
                    else:
                        results['checks'][f'l3_{file}'] = 'MISSING'
                        results['issues'].append(f"L3 file {file} not found")
            
            # Check temporal alignment
            if 'temporal_range' in l4_metadata:
                l4_range = l4_metadata['temporal_range']
                results['checks']['temporal_range'] = {
                    'start': l4_range.get('start'),
                    'end': l4_range.get('end'),
                    'total_days': l4_range.get('total_days')
                }
            
            # Check calendar info
            if 'calendar' in l4_metadata:
                cal = l4_metadata['calendar']
                if cal.get('session') == 'premium' and cal.get('hours') == '08:00-12:55 COT':
                    results['checks']['calendar'] = 'OK'
                else:
                    results['checks']['calendar'] = 'INVALID'
                    results['issues'].append("Invalid calendar specification")
            
            results['status'] = 'PASS' if not results['issues'] else 'WARN'
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error: {str(e)}")
            
        return results
    
    def _audit_replay_dataset(self) -> Dict:
        """Audit 2: Replay dataset schema and invariants"""
        print("\n[2/9] Auditing Replay Dataset...")
        
        results = {
            'status': 'PENDING',
            'schema_checks': {},
            'invariant_checks': {},
            'issues': []
        }
        
        try:
            # Load replay dataset
            replay_file = None
            for ext in ['.parquet', '.csv']:
                path = self.l4_path / f'replay_dataset{ext}'
                if path.exists():
                    replay_file = path
                    break
            
            if not replay_file:
                results['status'] = 'FAIL'
                results['issues'].append("replay_dataset not found")
                return results
            
            df = pd.read_parquet(replay_file) if replay_file.suffix == '.parquet' else pd.read_csv(replay_file)
            
            # Schema checks
            required_cols = [
                'episode_id', 't_in_episode', 'is_terminal', 'time_utc', 'time_cot',
                'open', 'high', 'low', 'close', 'is_blocked'
            ]
            
            for col in required_cols:
                results['schema_checks'][col] = 'OK' if col in df.columns else 'MISSING'
            
            # Check obs_* columns
            obs_cols = [c for c in df.columns if c.startswith('obs_')]
            results['schema_checks']['obs_columns'] = len(obs_cols)
            
            # Invariant checks
            # 1. Unique keys
            unique_keys = df.groupby(['episode_id', 't_in_episode']).size()
            results['invariant_checks']['unique_keys'] = 'OK' if unique_keys.max() == 1 else 'FAIL'
            
            # 2. Grid consistency (300s intervals)
            for episode in df['episode_id'].unique()[:5]:  # Sample check
                ep_data = df[df['episode_id'] == episode].sort_values('t_in_episode')
                if 'time_utc' in ep_data.columns:
                    ep_data['time_utc'] = pd.to_datetime(ep_data['time_utc'])
                    time_diffs = ep_data['time_utc'].diff().dt.total_seconds()
                    if not np.allclose(time_diffs.dropna(), 300, atol=1):
                        results['issues'].append(f"Grid violation in episode {episode}")
            
            # 3. Terminal check
            terminal_checks = df[df['is_terminal'] == True]['t_in_episode'].unique()
            if len(terminal_checks) == 1 and terminal_checks[0] == 59:
                results['invariant_checks']['terminal_step'] = 'OK'
            else:
                results['invariant_checks']['terminal_step'] = 'FAIL'
                results['issues'].append("Terminal flag not exclusively at t=59")
            
            # 4. Data types
            if obs_cols:
                sample_obs = df[obs_cols[0]].dtype
                results['invariant_checks']['obs_dtype'] = 'OK' if sample_obs == np.float32 else f'WARN ({sample_obs})'
            
            # 5. Blocked policy
            blocked_steps = df[df['is_blocked'] == 1]
            results['invariant_checks']['blocked_count'] = len(blocked_steps)
            results['invariant_checks']['blocked_rate'] = f"{100*len(blocked_steps)/len(df):.2f}%"
            
            # Set overall status
            if any('FAIL' in str(v) for v in results['invariant_checks'].values()):
                results['status'] = 'FAIL'
            elif results['issues']:
                results['status'] = 'WARN'
            else:
                results['status'] = 'PASS'
                
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error: {str(e)}")
            
        return results
    
    def _audit_episodes_index(self) -> Dict:
        """Audit 3: Episodes index consistency"""
        print("\n[3/9] Auditing Episodes Index...")
        
        results = {
            'status': 'PENDING',
            'checks': {},
            'issues': []
        }
        
        try:
            # Load episodes index
            episodes_file = None
            for ext in ['.parquet', '.csv']:
                path = self.l4_path / f'episodes_index{ext}'
                if path.exists():
                    episodes_file = path
                    break
            
            if not episodes_file:
                results['status'] = 'FAIL'
                results['issues'].append("episodes_index not found")
                return results
            
            episodes_df = pd.read_parquet(episodes_file) if episodes_file.suffix == '.parquet' else pd.read_csv(episodes_file)
            
            # Check required columns
            required_cols = ['episode_id', 'date_cot', 'n_steps', 'blocked_rate', 'has_gaps', 'quality_flag_episode']
            for col in required_cols:
                results['checks'][f'col_{col}'] = 'OK' if col in episodes_df.columns else 'MISSING'
            
            # Validate n_steps
            valid_steps = episodes_df['n_steps'].isin([59, 60])
            results['checks']['n_steps_valid'] = 'OK' if valid_steps.all() else 'FAIL'
            
            # Check blocked rate thresholds
            high_blocked = episodes_df[episodes_df['blocked_rate'] > 0.10]
            warn_blocked = episodes_df[(episodes_df['blocked_rate'] > 0.05) & (episodes_df['blocked_rate'] <= 0.10)]
            
            results['checks']['episodes_high_blocked'] = len(high_blocked)
            results['checks']['episodes_warn_blocked'] = len(warn_blocked)
            
            # Cross-check with replay dataset if available
            replay_file = self.l4_path / 'replay_dataset.parquet'
            if not replay_file.exists():
                replay_file = self.l4_path / 'replay_dataset.csv'
            
            if replay_file.exists():
                replay_df = pd.read_parquet(replay_file) if replay_file.suffix == '.parquet' else pd.read_csv(replay_file)
                
                # Count steps per episode
                actual_steps = replay_df.groupby('episode_id').size()
                
                for ep_id in episodes_df['episode_id']:
                    expected = episodes_df[episodes_df['episode_id'] == ep_id]['n_steps'].iloc[0]
                    actual = actual_steps.get(ep_id, 0)
                    if expected != actual:
                        results['issues'].append(f"Episode {ep_id}: expected {expected} steps, got {actual}")
            
            # Quality flags distribution
            if 'quality_flag_episode' in episodes_df.columns:
                flag_counts = episodes_df['quality_flag_episode'].value_counts()
                results['checks']['quality_distribution'] = flag_counts.to_dict()
            
            # Set status
            if results['issues'] or any('FAIL' in str(v) for v in results['checks'].values()):
                results['status'] = 'FAIL' if len(results['issues']) > 5 else 'WARN'
            else:
                results['status'] = 'PASS'
                
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error: {str(e)}")
            
        return results
    
    def _audit_env_spec(self) -> Dict:
        """Audit 4: Environment specifications"""
        print("\n[4/9] Auditing Environment Specifications...")
        
        results = {
            'status': 'PENDING',
            'spec_checks': {},
            'issues': []
        }
        
        try:
            env_spec_path = self.l4_path / 'env_spec.json'
            if not env_spec_path.exists():
                results['status'] = 'FAIL'
                results['issues'].append("env_spec.json not found")
                return results
            
            with open(env_spec_path, 'r') as f:
                env_spec = json.load(f)
            
            # Required fields
            required_fields = [
                'framework', 'observation_dim', 'observation_dtype',
                'action_space', 'decision_to_execution', 'reward_window',
                'normalization', 'features_order'
            ]
            
            for field in required_fields:
                if field in env_spec:
                    results['spec_checks'][field] = 'OK'
                    
                    # Specific validations
                    if field == 'framework' and env_spec[field] != 'gymnasium':
                        results['issues'].append(f"Framework should be 'gymnasium', got '{env_spec[field]}'")
                    
                    if field == 'observation_dtype' and env_spec[field] != 'float32':
                        results['issues'].append(f"Observation dtype should be 'float32', got '{env_spec[field]}'")
                    
                    if field == 'decision_to_execution' and 't -> open(t+1)' not in env_spec[field]:
                        results['issues'].append("Decision to execution should specify 't -> open(t+1)'")
                        
                else:
                    results['spec_checks'][field] = 'MISSING'
                    results['issues'].append(f"Missing required field: {field}")
            
            # Cross-check with replay dataset
            replay_file = self.l4_path / 'replay_dataset.parquet'
            if not replay_file.exists():
                replay_file = self.l4_path / 'replay_dataset.csv'
                
            if replay_file.exists() and 'features_order' in env_spec:
                df = pd.read_parquet(replay_file) if replay_file.suffix == '.parquet' else pd.read_csv(replay_file)
                
                # Check feature order matches
                expected_features = env_spec['features_order']
                actual_obs_cols = [c for c in df.columns if c.startswith('obs_')]
                
                if len(expected_features) != len(actual_obs_cols):
                    results['issues'].append(f"Feature count mismatch: spec has {len(expected_features)}, data has {len(actual_obs_cols)}")
                
                # Check observation_dim
                if env_spec.get('observation_dim') != len(actual_obs_cols):
                    results['issues'].append(f"Observation dim mismatch: spec says {env_spec.get('observation_dim')}, data has {len(actual_obs_cols)}")
            
            results['status'] = 'FAIL' if len(results['issues']) > 2 else 'WARN' if results['issues'] else 'PASS'
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error: {str(e)}")
            
        return results
    
    def _audit_reward_cost_models(self) -> Dict:
        """Audit 5: Reward and cost models"""
        print("\n[5/9] Auditing Reward and Cost Models...")
        
        results = {
            'status': 'PENDING',
            'reward_checks': {},
            'cost_checks': {},
            'issues': []
        }
        
        try:
            # Check reward spec
            reward_spec_path = self.l4_path / 'reward_spec.json'
            if reward_spec_path.exists():
                with open(reward_spec_path, 'r') as f:
                    reward_spec = json.load(f)
                
                # Validate reward formula
                if 'formula' in reward_spec:
                    formula = reward_spec['formula']
                    if 't+2' in formula and 'observation' in formula.lower():
                        results['issues'].append("Reward formula references future (t+2) in observation - LEAKAGE!")
                    results['reward_checks']['formula'] = 'OK'
                else:
                    results['reward_checks']['formula'] = 'MISSING'
                
                # Check t=59 handling
                if 't59_handling' in reward_spec or 'terminal_reward' in reward_spec:
                    results['reward_checks']['terminal_handling'] = 'OK'
                else:
                    results['issues'].append("No specification for t=59 reward handling")
            else:
                results['reward_checks']['exists'] = 'FAIL'
                results['issues'].append("reward_spec.json not found")
            
            # Check cost model
            cost_model_path = self.l4_path / 'cost_model.json'
            if cost_model_path.exists():
                with open(cost_model_path, 'r') as f:
                    cost_model = json.load(f)
                
                # Validate spread model
                if 'spread_model' in cost_model:
                    spread_bounds = cost_model.get('spread_bounds_bps', [0, 100])
                    if spread_bounds[1] > 15:
                        results['issues'].append(f"Spread upper bound too high: {spread_bounds[1]} bps")
                    results['cost_checks']['spread_model'] = 'OK'
                else:
                    results['cost_checks']['spread_model'] = 'MISSING'
                
                # Validate slippage model
                if 'slippage_model' in cost_model:
                    results['cost_checks']['slippage_model'] = 'OK'
                
                # Validate fees
                if 'fee_bps' in cost_model:
                    fee = cost_model['fee_bps']
                    if fee < 0 or fee > 10:
                        results['issues'].append(f"Unrealistic fee: {fee} bps")
                    results['cost_checks']['fees'] = 'OK'
            else:
                results['cost_checks']['exists'] = 'FAIL'
                results['issues'].append("cost_model.json not found")
            
            # Validate cost distributions if replay dataset available
            replay_file = self.l4_path / 'replay_dataset.parquet'
            if not replay_file.exists():
                replay_file = self.l4_path / 'replay_dataset.csv'
                
            if replay_file.exists() and 'spread_proxy_bps' in pd.read_csv(replay_file, nrows=1).columns:
                df = pd.read_parquet(replay_file) if replay_file.suffix == '.parquet' else pd.read_csv(replay_file)
                
                if 'spread_proxy_bps' in df.columns:
                    spread_stats = {
                        'p50': df['spread_proxy_bps'].quantile(0.50),
                        'p95': df['spread_proxy_bps'].quantile(0.95),
                        'max': df['spread_proxy_bps'].max()
                    }
                    
                    results['cost_checks']['spread_distribution'] = spread_stats
                    
                    if spread_stats['p95'] > 15:
                        results['issues'].append(f"Spread p95 too high: {spread_stats['p95']:.2f} bps")
                    if spread_stats['p50'] < 3 or spread_stats['p50'] > 8:
                        results['issues'].append(f"Spread p50 unusual: {spread_stats['p50']:.2f} bps")
            
            results['status'] = 'FAIL' if len(results['issues']) > 2 else 'WARN' if results['issues'] else 'PASS'
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error: {str(e)}")
            
        return results
    
    def _audit_action_split_specs(self) -> Dict:
        """Audit 6: Action and split specifications"""
        print("\n[6/9] Auditing Action and Split Specifications...")
        
        results = {
            'status': 'PENDING',
            'action_checks': {},
            'split_checks': {},
            'issues': []
        }
        
        try:
            # Check action spec
            action_spec_path = self.l4_path / 'action_spec.json'
            if action_spec_path.exists():
                with open(action_spec_path, 'r') as f:
                    action_spec = json.load(f)
                
                # Validate action mapping
                if 'action_map' in action_spec:
                    expected_actions = {-1, 0, 1}
                    actual_actions = set(action_spec['action_map'].keys()) if isinstance(action_spec['action_map'], dict) else set()
                    
                    if not expected_actions.issubset(map(int, map(str, actual_actions))):
                        results['issues'].append("Action map doesn't cover {-1, 0, 1}")
                    results['action_checks']['action_map'] = 'OK'
                
                # Check for thresholds
                if 'thresholds' in action_spec or 'inertia' in action_spec:
                    results['action_checks']['thresholds'] = 'OK'
            else:
                results['action_checks']['exists'] = 'MISSING'
                results['issues'].append("action_spec.json not found")
            
            # Check split spec
            split_spec_path = self.l4_path / 'split_spec.json'
            if split_spec_path.exists():
                with open(split_spec_path, 'r') as f:
                    split_spec = json.load(f)
                
                # Validate splits
                if 'splits' in split_spec:
                    splits = split_spec['splits']
                    
                    # Check for overlaps
                    all_episodes = []
                    for split_name, split_info in splits.items():
                        if 'episodes' in split_info:
                            episodes = split_info['episodes']
                            for ep in episodes:
                                if ep in all_episodes:
                                    results['issues'].append(f"Episode {ep} appears in multiple splits")
                                all_episodes.append(ep)
                    
                    results['split_checks']['no_overlaps'] = 'OK' if not any('multiple splits' in i for i in results['issues']) else 'FAIL'
                
                # Check embargo
                if 'embargo_days' in split_spec:
                    embargo = split_spec['embargo_days']
                    if embargo < 3:
                        results['issues'].append(f"Embargo too short: {embargo} days")
                    results['split_checks']['embargo'] = embargo
                
                # Check skip policy
                if 'skip_fail_episodes' in split_spec:
                    results['split_checks']['skip_fail'] = split_spec['skip_fail_episodes']
            else:
                results['split_checks']['exists'] = 'MISSING'
                results['issues'].append("split_spec.json not found")
            
            results['status'] = 'FAIL' if len(results['issues']) > 2 else 'WARN' if results['issues'] else 'PASS'
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error: {str(e)}")
            
        return results
    
    def _audit_data_quality(self) -> Dict:
        """Audit 7: Data quality checks"""
        print("\n[7/9] Auditing Data Quality...")
        
        results = {
            'status': 'PENDING',
            'quality_metrics': {},
            'issues': []
        }
        
        try:
            # Load replay dataset
            replay_file = self.l4_path / 'replay_dataset.parquet'
            if not replay_file.exists():
                replay_file = self.l4_path / 'replay_dataset.csv'
            
            if not replay_file.exists():
                results['status'] = 'FAIL'
                results['issues'].append("replay_dataset not found")
                return results
            
            df = pd.read_parquet(replay_file) if replay_file.suffix == '.parquet' else pd.read_csv(replay_file)
            
            # NaN analysis
            obs_cols = [c for c in df.columns if c.startswith('obs_')]
            if obs_cols:
                # Skip first 10 steps (warmup)
                df_post_warmup = df[df['t_in_episode'] >= 10]
                nan_rates = df_post_warmup[obs_cols].isna().mean()
                
                results['quality_metrics']['nan_rate_mean'] = f"{nan_rates.mean()*100:.3f}%"
                results['quality_metrics']['nan_rate_max'] = f"{nan_rates.max()*100:.3f}%"
                
                if nan_rates.mean() > 0.005:  # 0.5%
                    results['issues'].append(f"High NaN rate post-warmup: {nan_rates.mean()*100:.3f}%")
            
            # OHLC validity
            ohlc_cols = ['open', 'high', 'low', 'close']
            if all(c in df.columns for c in ohlc_cols):
                # Check H >= L
                invalid_hl = df[df['high'] < df['low']]
                if len(invalid_hl) > 0:
                    results['issues'].append(f"Found {len(invalid_hl)} rows with high < low")
                
                # Check H >= O,C and L <= O,C
                invalid_bounds = df[
                    (df['high'] < df['open']) | (df['high'] < df['close']) |
                    (df['low'] > df['open']) | (df['low'] > df['close'])
                ]
                if len(invalid_bounds) > 0:
                    results['issues'].append(f"Found {len(invalid_bounds)} rows with invalid OHLC bounds")
                
                results['quality_metrics']['ohlc_validity'] = 'OK' if len(invalid_hl) == 0 and len(invalid_bounds) == 0 else 'FAIL'
            
            # Blocked rate
            if 'is_blocked' in df.columns:
                blocked_rate = df['is_blocked'].mean()
                results['quality_metrics']['blocked_rate'] = f"{blocked_rate*100:.2f}%"
                
                if blocked_rate > 0.10:
                    results['issues'].append(f"High blocked rate: {blocked_rate*100:.2f}%")
                    results['status'] = 'FAIL'
                elif blocked_rate > 0.05:
                    results['issues'].append(f"Moderate blocked rate: {blocked_rate*100:.2f}%")
            
            # Premium window check (sample)
            if 'time_cot' in df.columns:
                df['time_cot'] = pd.to_datetime(df['time_cot'])
                df['hour_cot'] = df['time_cot'].dt.hour
                df['minute_cot'] = df['time_cot'].dt.minute
                
                # Check if all data is within 08:00-12:55 COT
                outside_premium = df[
                    (df['hour_cot'] < 8) | 
                    (df['hour_cot'] > 12) |
                    ((df['hour_cot'] == 12) & (df['minute_cot'] > 55))
                ]
                
                if len(outside_premium) > 0:
                    results['issues'].append(f"Found {len(outside_premium)} steps outside premium window")
                    results['quality_metrics']['premium_window_violations'] = len(outside_premium)
            
            # Check report if exists
            check_report_path = self.l4_path / 'checks_report.json'
            if check_report_path.exists():
                with open(check_report_path, 'r') as f:
                    checks = json.load(f)
                    
                critical_checks = [
                    'grid_ok', 'keys_unique_ok', 'terminal_step_ok',
                    'no_future_in_obs', 'cost_realism_ok', 'determinism_ok'
                ]
                
                for check in critical_checks:
                    if check in checks:
                        if not checks[check]:
                            results['issues'].append(f"Critical check failed: {check}")
                        results['quality_metrics'][check] = checks[check]
            
            # Set final status
            if not results['status'] == 'FAIL':
                if len(results['issues']) > 3:
                    results['status'] = 'FAIL'
                elif results['issues']:
                    results['status'] = 'WARN'
                else:
                    results['status'] = 'PASS'
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error: {str(e)}")
            
        return results
    
    def _audit_coverage(self) -> Dict:
        """Audit 8: Coverage analysis (2020-2025)"""
        print("\n[8/9] Auditing Coverage (2020-2025)...")
        
        results = {
            'status': 'PENDING',
            'coverage_stats': {},
            'yearly_coverage': {},
            'issues': []
        }
        
        try:
            # Load episodes index
            episodes_file = self.l4_path / 'episodes_index.parquet'
            if not episodes_file.exists():
                episodes_file = self.l4_path / 'episodes_index.csv'
            
            if not episodes_file.exists():
                results['status'] = 'FAIL'
                results['issues'].append("episodes_index not found")
                return results
            
            episodes_df = pd.read_parquet(episodes_file) if episodes_file.suffix == '.parquet' else pd.read_csv(episodes_file)
            
            # Parse dates
            if 'date_cot' in episodes_df.columns:
                episodes_df['date'] = pd.to_datetime(episodes_df['date_cot'])
            elif 'episode_id' in episodes_df.columns:
                # Try to parse from episode_id (usually YYYY-MM-DD format)
                episodes_df['date'] = pd.to_datetime(episodes_df['episode_id'], errors='coerce')
            
            if 'date' not in episodes_df.columns or episodes_df['date'].isna().all():
                results['issues'].append("Cannot parse dates from episodes")
                return results
            
            # Analyze coverage by year
            episodes_df['year'] = episodes_df['date'].dt.year
            
            for year in range(2020, 2026):
                year_data = episodes_df[episodes_df['year'] == year]
                
                if len(year_data) > 0:
                    # Calculate business days (approximate)
                    start_date = year_data['date'].min()
                    end_date = year_data['date'].max()
                    expected_days = pd.bdate_range(start=start_date, end=end_date).shape[0]
                    actual_days = year_data['date'].nunique()
                    
                    coverage = (actual_days / expected_days * 100) if expected_days > 0 else 0
                    
                    results['yearly_coverage'][year] = {
                        'episodes': len(year_data),
                        'days': actual_days,
                        'expected_days': expected_days,
                        'coverage_pct': f"{coverage:.1f}%",
                        'quality_ok': len(year_data[year_data.get('quality_flag_episode', 'OK') == 'OK']) if 'quality_flag_episode' in year_data.columns else 'N/A'
                    }
                    
                    if coverage < 90:
                        results['issues'].append(f"Low coverage for {year}: {coverage:.1f}%")
                else:
                    results['yearly_coverage'][year] = {'episodes': 0, 'coverage_pct': '0%'}
                    results['issues'].append(f"No data for year {year}")
            
            # Overall statistics
            results['coverage_stats']['total_episodes'] = len(episodes_df)
            results['coverage_stats']['date_range'] = f"{episodes_df['date'].min().date()} to {episodes_df['date'].max().date()}"
            results['coverage_stats']['total_days'] = episodes_df['date'].nunique()
            
            # Compare with L3 if available
            l3_features = self.l3_path / 'features.parquet'
            if l3_features.exists():
                l3_df = pd.read_parquet(l3_features, columns=['date'] if 'date' in pd.read_parquet(l3_features, nrows=1).columns else ['timestamp'])
                if 'timestamp' in l3_df.columns:
                    l3_df['date'] = pd.to_datetime(l3_df['timestamp']).dt.date
                else:
                    l3_df['date'] = pd.to_datetime(l3_df['date']).dt.date
                
                l3_days = l3_df['date'].nunique()
                l4_days = episodes_df['date'].dt.date.nunique()
                
                results['coverage_stats']['l3_days'] = l3_days
                results['coverage_stats']['l4_coverage_of_l3'] = f"{100*l4_days/l3_days:.1f}%" if l3_days > 0 else 'N/A'
                
                if l4_days < l3_days * 0.95:
                    results['issues'].append(f"L4 covers only {100*l4_days/l3_days:.1f}% of L3 days")
            
            # Set status
            major_gaps = sum(1 for year, data in results['yearly_coverage'].items() 
                           if isinstance(data, dict) and data.get('episodes', 0) == 0)
            
            if major_gaps >= 2:
                results['status'] = 'FAIL'
            elif results['issues']:
                results['status'] = 'WARN'
            else:
                results['status'] = 'PASS'
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error: {str(e)}")
            
        return results
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _generate_final_report(self):
        """Generate comprehensive audit report"""
        print("\n[9/9] Generating Final Report...")
        
        # Determine overall status
        statuses = [v.get('status', 'UNKNOWN') for v in self.audit_results.values()]
        
        if 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARN' in statuses:
            overall_status = 'WARN'
        elif all(s == 'PASS' for s in statuses):
            overall_status = 'PASS'
        else:
            overall_status = 'INCOMPLETE'
        
        self.audit_results['overall_status'] = overall_status
        self.audit_results['timestamp'] = datetime.now().isoformat()
        
        # Critical issues summary
        all_issues = []
        for section, results in self.audit_results.items():
            if isinstance(results, dict) and 'issues' in results:
                all_issues.extend(results['issues'])
        
        self.audit_results['critical_issues'] = all_issues[:10]  # Top 10 issues
        self.audit_results['total_issues'] = len(all_issues)
        
        # Save report
        report_path = self.l4_path / f'L4_AUDIT_REPORT_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print(" AUDIT SUMMARY")
        print("="*80)
        print(f"Overall Status: {overall_status}")
        print(f"Total Issues Found: {len(all_issues)}")
        
        if overall_status == 'PASS':
            print("\n[PASS] L4 READY - All checks passed!")
        elif overall_status == 'WARN':
            print("\n[WARN] L4 READY WITH WARNINGS - Review issues before proceeding")
        else:
            print("\n[FAIL] L4 NOT READY - Critical issues must be resolved")
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Print critical issues
        if all_issues:
            print("\n[!] Critical Issues to Address:")
            for i, issue in enumerate(all_issues[:5], 1):
                print(f"  {i}. {issue}")

def main():
    """Run comprehensive L4 audit"""
    import argparse
    
    parser = argparse.ArgumentParser(description='L4 Comprehensive Audit System')
    parser.add_argument('--l3-path', type=str, required=True, help='Path to L3 outputs')
    parser.add_argument('--l4-path', type=str, required=True, help='Path to L4 outputs')
    
    args = parser.parse_args()
    
    # Run audit
    auditor = L4ComprehensiveAuditor(args.l3_path, args.l4_path)
    results = auditor.run_full_audit()
    
    # Return exit code based on status
    if results['overall_status'] == 'PASS':
        return 0
    elif results['overall_status'] == 'WARN':
        return 1
    else:
        return 2

if __name__ == "__main__":
    import sys
    sys.exit(main())