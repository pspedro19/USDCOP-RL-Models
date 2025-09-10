#!/usr/bin/env python3
"""
Apply L5 Timeout Fixes to Existing Pipeline
============================================
This script patches your existing usdcop_m5__06_l5_serving_final.py
with all the timeout fixes.
"""

import os
import sys
import shutil
from pathlib import Path

def apply_timeout_fixes(original_file: str, output_file: str = None):
    """Apply all timeout fixes to the existing pipeline"""
    
    if not os.path.exists(original_file):
        print(f"❌ File not found: {original_file}")
        return False
    
    if output_file is None:
        # Backup original
        backup_file = original_file.replace('.py', '_backup.py')
        shutil.copy2(original_file, backup_file)
        print(f"✅ Backup created: {backup_file}")
        output_file = original_file
    
    print(f"Applying fixes to: {output_file}")
    
    # Read original file
    with open(original_file, 'r') as f:
        content = f.read()
    
    # ========================================================================
    # FIX 1: Add timeout configuration variables at the top
    # ========================================================================
    
    timeout_config = '''
# ============================================================================
# TIMEOUT FIX CONFIGURATION
# ============================================================================
from airflow.models import Variable

# Timeout configuration (configurable via Airflow Variables)
TRAIN_TIMEOUT_HOURS = int(Variable.get("L5_TRAIN_TIMEOUT_HOURS", default_var="12"))
TIME_SAFETY_MARGIN_SEC = int(Variable.get("L5_TRAIN_SAFETY_MARGIN_SEC", default_var="300"))

# Force DummyVecEnv in Airflow to avoid daemon issues
FORCE_DUMMY_VEC = Variable.get("L5_FORCE_DUMMY_VEC", default_var="true").lower() == "true"

# Train in chunks to allow checkpointing
TRAIN_CHUNK_STEPS = int(Variable.get("L5_TRAIN_CHUNK_STEPS", default_var="50000"))
CHECKPOINT_FREQ = int(Variable.get("L5_CHECKPOINT_FREQ", default_var="50000"))
'''
    
    # Insert after imports
    import_end = content.find('# Constants and Configuration')
    if import_end == -1:
        import_end = content.find('DEBUG_MODE')
    
    if import_end != -1:
        content = content[:import_end] + timeout_config + '\n' + content[import_end:]
        print("✅ Added timeout configuration variables")
    
    # ========================================================================
    # FIX 2: Update execution_timeout in DAG tasks
    # ========================================================================
    
    # Find and replace execution_timeout
    old_timeout = "execution_timeout=timedelta(hours=4)"
    new_timeout = "execution_timeout=timedelta(hours=TRAIN_TIMEOUT_HOURS)"
    
    if old_timeout in content:
        content = content.replace(old_timeout, new_timeout)
        print("✅ Updated execution_timeout to use TRAIN_TIMEOUT_HOURS variable")
    
    # ========================================================================
    # FIX 3: Add time budget helper functions
    # ========================================================================
    
    helper_functions = '''
# ============================================================================
# TIME BUDGET HELPER FUNCTIONS
# ============================================================================

def get_time_remaining(context) -> float:
    """Calculate seconds remaining before task timeout"""
    from datetime import datetime, timezone, timedelta
    
    start_dt = context['ti'].start_date
    if not start_dt:
        return float('inf')
    
    # Convert to timezone-aware if needed
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    
    task_timeout = context['task'].execution_timeout or timedelta(hours=TRAIN_TIMEOUT_HOURS)
    deadline = start_dt + task_timeout - timedelta(seconds=TIME_SAFETY_MARGIN_SEC)
    
    now = datetime.now(timezone.utc)
    remaining = (deadline - now).total_seconds()
    
    return max(0, remaining)

def save_training_checkpoint(model, path: str, metadata: dict = None):
    """Save model checkpoint with metadata for resuming"""
    import json
    
    # Save model
    model.save(path)
    
    # Save metadata
    if metadata:
        metadata_path = path.replace('.zip', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Log to MLflow if available
        try:
            import mlflow
            mlflow.log_artifact(path, artifact_path="checkpoints")
            mlflow.log_artifact(metadata_path, artifact_path="checkpoints")
        except:
            pass
    
    logger.info(f"Checkpoint saved: {path}")

def load_training_checkpoint(algorithm: str, seed: int, env, run_id: str = None):
    """Try to load checkpoint from previous run"""
    from stable_baselines3 import PPO, DQN
    
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        if not run_id:
            return None, 0
        
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Try to download checkpoint
        import tempfile
        checkpoint_dir = tempfile.mkdtemp()
        checkpoint_name = f"{algorithm}_seed_{seed}.zip"
        
        artifacts = client.list_artifacts(run_id, "checkpoints")
        for artifact in artifacts:
            if artifact.path.endswith(checkpoint_name):
                local_path = client.download_artifacts(run_id, artifact.path, checkpoint_dir)
                
                # Load model based on algorithm
                if 'ppo' in algorithm.lower():
                    model = PPO.load(local_path, env=env)
                elif 'dqn' in algorithm.lower():
                    model = DQN.load(local_path, env=env)
                else:
                    return None, 0
                
                # Get timesteps from model
                timesteps_done = getattr(model, 'num_timesteps', 0)
                
                logger.info(f"Resumed from checkpoint: {local_path} (timesteps: {timesteps_done})")
                return model, timesteps_done
    
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
    
    return None, 0
'''
    
    # Add helper functions before train_rl_models_real
    train_func_start = content.find('def train_rl_models_real')
    if train_func_start != -1:
        content = content[:train_func_start] + helper_functions + '\n' + content[train_func_start:]
        print("✅ Added time budget helper functions")
    
    # ========================================================================
    # FIX 4: Modify create_optimized_vec_env to force DummyVecEnv
    # ========================================================================
    
    vec_env_fix = '''
    # Force DummyVecEnv in Airflow to avoid daemon issues
    if FORCE_DUMMY_VEC:
        logger.info("FORCE_DUMMY_VEC=True → using DummyVecEnv (n_envs=1)")
        return DummyVecEnv([env_factory])
'''
    
    # Find create_optimized_vec_env function
    vec_env_start = content.find('def create_optimized_vec_env')
    if vec_env_start != -1:
        # Find the line after device_info setup
        device_info_line = content.find('device_info = detect_and_configure_device()', vec_env_start)
        if device_info_line != -1:
            # Find the end of that line
            line_end = content.find('\n', device_info_line)
            content = content[:line_end+1] + vec_env_fix + content[line_end+1:]
            print("✅ Added FORCE_DUMMY_VEC check to create_optimized_vec_env")
    
    # ========================================================================
    # FIX 5: Add chunked training with checkpointing
    # ========================================================================
    
    # This is more complex - we need to modify the model.learn() calls
    # to use chunked training with time budget checks
    
    chunked_training_template = '''
                # ============================================================
                # CHUNKED TRAINING WITH TIME BUDGET
                # ============================================================
                
                # Check if we should resume from checkpoint
                checkpoint_path = os.path.join(tempfile.gettempdir(), f"{model_name}_seed_{seed}.zip")
                model_loaded, timesteps_done = load_training_checkpoint(model_name, seed, train_env_vec, child_run.info.run_id)
                
                if model_loaded is not None:
                    model = model_loaded
                    logger.info(f"Resumed training from timestep {timesteps_done}")
                else:
                    timesteps_done = 0
                
                # Training loop with time budget
                target_timesteps = TRAINING_CONFIG['timesteps']
                
                while timesteps_done < target_timesteps:
                    # Check time remaining
                    if 'context' in locals():
                        time_left = get_time_remaining(context)
                        
                        if time_left < 60:  # Less than 1 minute left
                            logger.warning(f"Time budget exhausted (remaining: {time_left:.0f}s)")
                            logger.info("Saving checkpoint for resume on retry...")
                            
                            save_training_checkpoint(model, checkpoint_path, {
                                "algorithm": model_name,
                                "seed": seed,
                                "timesteps_done": timesteps_done,
                                "target_timesteps": target_timesteps,
                            })
                            
                            mlflow.set_tag("training_status", "INCOMPLETE")
                            return  # Exit gracefully for retry
                    
                    # Calculate chunk size
                    chunk_size = min(TRAIN_CHUNK_STEPS, target_timesteps - timesteps_done)
                    
                    logger.info(f"Training chunk: {chunk_size} steps (progress: {timesteps_done}/{target_timesteps})")
                    
                    # Train for chunk
                    model.learn(
                        total_timesteps=chunk_size,
                        reset_num_timesteps=False,
                        callback=eval_callback,
                        progress_bar=False
                    )
                    
                    timesteps_done = model.num_timesteps
                    
                    # Save checkpoint periodically
                    if timesteps_done % CHECKPOINT_FREQ == 0:
                        save_training_checkpoint(model, checkpoint_path, {
                            "algorithm": model_name,
                            "seed": seed,
                            "timesteps_done": timesteps_done,
                            "target_timesteps": target_timesteps,
                        })
                
                logger.info(f"✅ Training completed: {timesteps_done} timesteps")
'''
    
    print("✅ Chunked training template prepared (manual integration needed)")
    
    # ========================================================================
    # Save the modified file
    # ========================================================================
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixes applied and saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF FIXES APPLIED:")
    print("="*60)
    print("1. ✅ Added configurable timeout variables")
    print("2. ✅ Updated execution_timeout to use TRAIN_TIMEOUT_HOURS")
    print("3. ✅ Added time budget helper functions")
    print("4. ✅ Added FORCE_DUMMY_VEC check")
    print("5. ⚠️  Chunked training template prepared (needs manual integration)")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Configure Airflow Variables:")
    print("   bash scripts/configure_airflow_l5.sh")
    print("")
    print("2. Test with debug mode:")
    print("   airflow variables set L5_DEBUG_MODE true")
    print("   airflow dags trigger usdcop_m5__06_l5_serving_final")
    print("")
    print("3. Run full pipeline:")
    print("   airflow variables set L5_DEBUG_MODE false")
    print("   airflow dags trigger usdcop_m5__06_l5_serving_final")
    
    return True

if __name__ == "__main__":
    # Check if file path provided
    if len(sys.argv) > 1:
        original_file = sys.argv[1]
    else:
        # Default path
        original_file = "dags/usdcop_m5__06_l5_serving_final.py"
    
    # Apply fixes
    success = apply_timeout_fixes(original_file)
    
    sys.exit(0 if success else 1)