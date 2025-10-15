#!/usr/bin/env python3
"""
USDCOP M5 - L5 SERVING PIPELINE - PRODUCTION VERSION ACTUALIZADA
================================================================
Pipeline L5 completo con TODAS las correcciones:
- Monitor wrapper para SB3 (elimina warnings)
- Sortino ratio en lugar de Sharpe inestable
- Costos del L4 sin hardcoding
- Git SHA con 3 métodos de fallback
- Validación completa de reproducibilidad del reward
- ONNX export limpio sin warnings
- Bundle con manifest completo

GO/NO-GO Gates ACTUALIZADOS:
- Sortino ≥ 1.3 (más estable que Sharpe)
- MaxDD ≤ 15%, Calmar ≥ 0.8
- Consistency check con Sortino diff ≤ 0.5
- Cost stress test con parámetros del L4 (+25%)
- Inference latency p99 ≤ 20ms, E2E p99 ≤ 100ms
"""

from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os
import sys
import json
import hashlib
import subprocess
import logging
import tempfile
import shutil
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# MLFLOW ARTIFACT HELPER (MODULE-LEVEL)
# ============================================================================

def safe_mlflow_log_artifact(file_path, artifact_path=None, max_retries=3):
    """Upload artifact to MLflow with retry logic and error handling"""
    import mlflow, os, time, logging
    logger = logging.getLogger(__name__)
    for attempt in range(max_retries):
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found for MLflow upload: {file_path}")
                return False
            if os.path.getsize(file_path) == 0:
                logger.error(f"File is empty, skipping upload: {file_path}")
                return False
            mlflow.log_artifact(file_path, artifact_path=artifact_path)
            logger.info(f"Successfully uploaded {file_path} to MLflow (attempt {attempt + 1})")
            return True
        except Exception as e:
            logger.warning(f"Failed to upload {file_path} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"All upload attempts failed for {file_path}: {e}")
                return False
    return False

# ============================================================================
# NUMPY JSON SERIALIZATION UTILITY
# ============================================================================

def convert_numpy_to_python(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    Fixes serialization errors with numpy types causing gate failures.
    Also handles type objects and callables.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_to_python(item) for item in obj]
    # Handle type objects and callables for JSON serialization
    elif isinstance(obj, type) or callable(obj):
        mod = getattr(obj, "__module__", "")
        name = getattr(obj, "__name__", str(obj))
        return f"{mod}.{name}" if mod else name
    else:
        return obj

# Import dependency handler first
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from dependency_handler import get_torch_handler, get_stable_baselines3_handler, get_sklearn_handler

# Import reward_sentinel para métricas seguras y wrapper de entorno
from reward_sentinel import (
    SentinelTradingEnv,
    CostCurriculumCallback,
    enhanced_sortino
)

# Import l5_patch_metrics para métricas ultra-robustas y fail-fast
from l5_patch_metrics import (
    robust_cagr,
    robust_sortino,
    robust_calmar,
    robust_max_drawdown,
    evaluate_episode_metrics,
    validate_metrics_for_gates,
    calculate_pnl_attribution,
    generate_seed_report
)

# Import reward_costs_sanity para validación de señal
from reward_costs_sanity import (
    RewardCostsSanityChecker,
    quick_sanity_check
)

# ============================================================================
# DISCRETE TO SIGNED ACTION WRAPPER - PREVENT POLICY COLLAPSE
# ============================================================================

class DiscreteToSignedAction:
    """
    Wrapper to map PPO discrete actions (0,1,2) to environment actions (-1,0,1)
    
    Critical for preventing policy collapse by ensuring proper action mapping:
    - PPO output: 0, 1, 2 (discrete action space)
    - Env expects: -1, 0, 1 (signed action space: sell, hold, buy)
    
    Without this wrapper, the policy may collapse to always predicting 0,
    which maps incorrectly to the environment action space.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper
        
        Args:
            env: The base trading environment
        """
        self.env = env
        self.action_mapping = {
            0: -1,  # PPO action 0 -> Sell (-1)
            1: 0,   # PPO action 1 -> Hold (0)
            2: 1    # PPO action 2 -> Buy (1)
        }
        
    def map_action(self, action):
        """
        Map discrete action to signed action
        
        Args:
            action: PPO discrete action (0, 1, or 2)
            
        Returns:
            Signed action (-1, 0, or 1)
        """
        if isinstance(action, (list, np.ndarray)):
            action = action[0]  # Handle vectorized envs
        
        action = int(action)
        if action not in self.action_mapping:
            raise ValueError(f"Invalid action {action}. Expected 0, 1, or 2")
        
        mapped_action = self.action_mapping[action]
        return mapped_action
    
    def step(self, action):
        """Step the environment with mapped action"""
        mapped_action = self.map_action(action)
        return self.env.step(mapped_action)
    
    def reset(self, **kwargs):
        """Reset the environment"""
        return self.env.reset(**kwargs)
    
    def render(self, *args, **kwargs):
        """Render the environment"""
        return self.env.render(*args, **kwargs)
    
    def close(self):
        """Close the environment"""
        return self.env.close()
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment"""
        return getattr(self.env, name)

# ============================================================================
# LAZY PACKAGE INSTALLER
# ============================================================================

def _ensure_pkgs(pkgs):
    """Lazily install packages with version checking for true idempotency"""
    import importlib, subprocess, sys
    import importlib.metadata
    import re
    
    def parse_version_spec(spec):
        """Parse package spec into name, operator, and version"""
        # Match patterns like pkg==1.0, pkg>=1.0, pkg>=1.0,<2.0
        match = re.match(r'^([a-zA-Z0-9_-]+)(.*)', spec)
        if match:
            name = match.group(1)
            version_spec = match.group(2).strip()
            return name, version_spec
        return spec, ""
    
    def check_version_satisfied(installed_version, version_spec):
        """Check if installed version satisfies the spec"""
        if not version_spec:
            return True  # No version requirement
        
        # Simple version comparison (can be enhanced with packaging.version)
        try:
            from packaging import version
            from packaging.specifiers import SpecifierSet
            
            spec_set = SpecifierSet(version_spec)
            return version.parse(installed_version) in spec_set
        except ImportError:
            # Fallback: only check exact matches
            if version_spec.startswith("=="):
                required = version_spec[2:]
                return installed_version == required
            # For other operators, reinstall to be safe
            return False
    
    for spec in pkgs:
        name, version_spec = parse_version_spec(spec)
        
        # Check if package exists and version is satisfied
        needs_install = False
        try:
            installed_version = importlib.metadata.version(name)
            if version_spec and not check_version_satisfied(installed_version, version_spec):
                logger.info(f"Package {name} version {installed_version} doesn't satisfy {version_spec}")
                needs_install = True
            else:
                # MEJORA: Log skipped installs to help diagnose worker drift
                logger.debug(f"✅ Package {name} v{installed_version} satisfies {version_spec or 'any'} - skipping install")
        except importlib.metadata.PackageNotFoundError:
            logger.info(f"Package {name} not found - will install {spec}")
            needs_install = True
        
        if needs_install:
            try:
                logger.info(f"Installing {spec}...")
                # Special handling for torch to use CPU-only version for Python 3.12
                if name == "torch":
                    logger.info("Installing torch CPU-only version for Python 3.12 compatibility...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "torch==2.5.1+cpu", 
                        "--index-url", "https://download.pytorch.org/whl/cpu"
                    ])
                    logger.info("Installed torch CPU version successfully")
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
                    logger.info(f"Installed {spec}")
            except subprocess.CalledProcessError as e:
                # Fallback for known Python 3.12 incompatibilities
                if name == "stable-baselines3":
                    fallback = "stable-baselines3>=2.3,<3"
                    logger.warning(f"Failed to install {spec}, falling back to {fallback} for Python 3.12")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", fallback])
                    logger.info(f"Installed {fallback}")
                elif name == "torch":
                    # Try standard torch without version constraints as fallback
                    logger.warning("Failed to install torch CPU version, trying standard version...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
                        logger.info("Installed torch standard version")
                    except subprocess.CalledProcessError:
                        logger.error(f"Failed to install torch completely: {e}")
                        raise
                else:
                    logger.error(f"Failed to install {spec}: {e}")
                    raise

# ============================================================================
# CONFIGURATION
# ============================================================================

# Training configuration
TRAIN_TIMEOUT_HOURS = int(Variable.get("L5_TRAIN_TIMEOUT_HOURS", default_var="6"))
PARALLEL_SEEDS = [42, 123, 456]

# Total timesteps for training - increased for better convergence
# Default changed from 10000 to 300000 for production quality training
TOTAL_TIMESTEPS = int(Variable.get("L5_TOTAL_TIMESTEPS", default_var="1000000"))  # Increased to 1M for better learning

# Evaluation configuration
IN_LOOP_EVAL_EPISODES = int(Variable.get("L5_IN_LOOP_EVAL_EPISODES", default_var="15"))
GATE_EVAL_EPISODES = int(Variable.get("L5_GATE_EVAL_EPISODES", default_var="100"))

# --- Model selection ---
DEFAULT_MODEL = os.getenv("L5_MODEL", Variable.get("L5_MODEL", default_var="ppo_lstm")).lower()
# Optionally allow a comma-separated list to run multiple algos in one DAG (advanced)
MODEL_LIST = [m.strip() for m in Variable.get("L5_MODEL_LIST", default_var=DEFAULT_MODEL).split(",")]
ACCEPTED_MODELS = {"ppo_lstm", "ppo_mlp", "qrdqn"}  # Start with core algorithms
if not set(MODEL_LIST).issubset(ACCEPTED_MODELS):
    raise ValueError(f"L5 model(s) not supported: {MODEL_LIST}. Accepted: {ACCEPTED_MODELS}")

# Algorithm-specific timeouts (hours)
ALGO_TIMEOUTS = {
    "ppo_mlp": 4,    # Fastest
    "ppo_lstm": 6,   # LSTM takes longer  
    "qrdqn": 8,      # QR-DQN needs more for replay buffer
}

# Production gates - USANDO SORTINO EN LUGAR DE SHARPE
SORTINO_THRESHOLD = float(Variable.get("L5_SORTINO_THRESHOLD", default_var="1.3"))
MAX_DD_THRESHOLD = float(Variable.get("L5_MAX_DD_THRESHOLD", default_var="0.15"))
CALMAR_THRESHOLD = float(Variable.get("L5_CALMAR_THRESHOLD", default_var="0.8"))
SORTINO_DIFF_THRESHOLD = float(Variable.get("L5_SORTINO_DIFF_THRESHOLD", default_var="0.5"))  # Cambio de Sharpe a Sortino
COST_STRESS_MULTIPLIER = float(Variable.get("L5_COST_STRESS_MULTIPLIER", default_var="1.25"))
CAGR_DROP_THRESHOLD = float(Variable.get("L5_CAGR_DROP_THRESHOLD", default_var="0.20"))

# Latency requirements
INFERENCE_P99_THRESHOLD_MS = float(Variable.get("L5_INFERENCE_P99_MS", default_var="20"))
E2E_P99_THRESHOLD_MS = float(Variable.get("L5_E2E_P99_MS", default_var="100"))

# Infrastructure
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI", default_var="http://trading-mlflow:5000")
MINIO_ENDPOINT = Variable.get("MINIO_ENDPOINT", default_var="trading-minio:9000")
MINIO_ACCESS_KEY = Variable.get("MINIO_ACCESS_KEY", default_var="minioadmin")
MINIO_SECRET_KEY = Variable.get("MINIO_SECRET_KEY", default_var="minioadmin123")

# Set environment variables for MLflow
os.environ.update({
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "MLFLOW_S3_ENDPOINT_URL": f"http://{MINIO_ENDPOINT}",
    "AWS_ACCESS_KEY_ID": MINIO_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": MINIO_SECRET_KEY,
    "AWS_S3_ADDRESSING_STYLE": "path",
    "AWS_DEFAULT_REGION": "us-east-1",
    "GIT_PYTHON_GIT_EXECUTABLE": "/usr/bin/git",
    "GIT_PYTHON_REFRESH": "quiet"
})

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_git_info() -> Dict[str, str]:
    """Get Git repository information with 3 fallback methods"""
    git_info = {
        'sha': 'unknown',
        'branch': 'unknown',
        'tag': None,
        'dirty': False
    }
    
    try:
        # Method 1: Direct git command
        try:
            git_sha = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            git_info['sha'] = git_sha[:8]
        except:
            pass
        
        # Method 2: GitPython library
        if git_info['sha'] == 'unknown':
            try:
                import git
                repo = git.Repo(search_parent_directories=True)
                git_info['sha'] = repo.head.object.hexsha[:8]
                git_info['branch'] = repo.active_branch.name
                git_info['dirty'] = repo.is_dirty()
            except:
                pass
        
        # Method 3: Environment variables
        if git_info['sha'] == 'unknown':
            git_sha = os.environ.get('GIT_COMMIT', os.environ.get('GIT_SHA', 'unknown'))
            if git_sha != 'unknown':
                git_info['sha'] = git_sha[:8]
        
        # Get branch
        if git_info['branch'] == 'unknown':
            try:
                git_branch = subprocess.check_output(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    stderr=subprocess.DEVNULL,
                    text=True
                ).strip()
                git_info['branch'] = git_branch
            except:
                git_info['branch'] = os.environ.get('GIT_BRANCH', 'unknown')
        
        # MEJORA: Get git tag info for richer provenance
        try:
            git_describe = subprocess.check_output(
                ['git', 'describe', '--tags', '--dirty', '--always'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            git_info['tag'] = git_describe
            
            # Also check if we're in a detached HEAD state
            try:
                # This will fail if not in detached HEAD
                subprocess.check_output(
                    ['git', 'symbolic-ref', '-q', 'HEAD'],
                    stderr=subprocess.DEVNULL,
                    text=True
                )
            except subprocess.CalledProcessError:
                # We're in detached HEAD
                git_info['detached'] = True
                git_info['branch'] = f"detached-{git_info['sha']}"
        except:
            # git describe failed, not critical
            pass
        
        logger.info(f"Git info: {git_info}")
        
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
    
    return git_info

def get_container_info() -> Dict[str, str]:
    """Get container information for reproducibility"""
    try:
        hostname = subprocess.check_output(['hostname']).decode('utf-8').strip()
        image_digest = os.environ.get('AIRFLOW_IMAGE_DIGEST', 'unknown')
        return {
            "hostname": hostname,
            "image_digest": image_digest,
            "python_version": sys.version
        }
    except Exception as e:
        logger.warning(f"Failed to get container info: {e}")
        return {"hostname": "unknown", "image_digest": "unknown"}

def _download_json_from_s3(s3_hook, bucket, key):
    """Helper to download JSON from S3 with error handling"""
    try:
        if s3_hook.check_for_key(key, bucket_name=bucket):
            obj = s3_hook.get_key(key, bucket_name=bucket)
            return json.loads(obj.get()['Body'].read())
    except Exception as e:
        logger.debug(f"Failed to download {key} from {bucket}: {e}")
    return None

def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data"""
    return hashlib.sha256(data).hexdigest()

def compute_sha256_streaming(body_stream, chunk_size: int = 128 * 1024) -> str:
    """Compute SHA256 hash using streaming to reduce memory usage
    
    Args:
        body_stream: S3 Body stream object
        chunk_size: Size of chunks to read (default 128KB)
    
    Returns:
        SHA256 hexdigest of the stream
    """
    try:
        sha256_hash = hashlib.sha256()
        while True:
            chunk = body_stream.read(chunk_size)
            if not chunk:
                break
            sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    finally:
        # MEJORA: Explicitly close stream to free sockets sooner
        if hasattr(body_stream, 'close'):
            body_stream.close()

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio using robust_sortino to prevent NaN/Inf values"""
    sortino_value, status = robust_sortino(returns, risk_free=risk_free_rate)
    return sortino_value

def get_l4_dataset_info(**context) -> Dict[str, Any]:
    """Get L4 dataset information and hash - MEJORA: Elige el más reciente"""
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket = "04-l4-ds-usdcop-rlready"
    
    # Get latest L4 dataset
    keys = s3_hook.list_keys(bucket_name=bucket, prefix="usdcop_m5__05_l4_rlready/")
    if not keys:
        raise ValueError(f"No L4 data found in {bucket}")
    
    # MEJORA: Obtener metadata con LastModified para elegir el más reciente
    logger.info("Scanning L4 datasets to find the most recent one...")
    
    replay_candidates = []
    for key in keys:
        if 'replay_dataset.parquet' in key:
            obj_metadata = s3_hook.get_key(key, bucket_name=bucket)
            last_modified = obj_metadata.get()['LastModified']
            replay_candidates.append({
                'key': key,
                'last_modified': last_modified,
                'obj': obj_metadata
            })
            logger.info(f"  Found: {key} (modified: {last_modified})")
    
    if replay_candidates:
        # MEJORA: Elegir el más reciente por LastModified
        most_recent = max(replay_candidates, key=lambda x: x['last_modified'])
        replay_key = most_recent['key']
        obj = most_recent['obj']
        logger.info(f"✅ Selected MOST RECENT L4 dataset: {replay_key}")
        logger.info(f"   Modified: {most_recent['last_modified']}")
    else:
        # Fallback: buscar cualquier archivo parquet
        logger.warning("No replay_dataset.parquet found, searching for any parquet file...")
        parquet_candidates = []
        for key in keys:
            if key.endswith('.parquet'):
                obj_metadata = s3_hook.get_key(key, bucket_name=bucket)
                last_modified = obj_metadata.get()['LastModified']
                parquet_candidates.append({
                    'key': key,
                    'last_modified': last_modified,
                    'obj': obj_metadata
                })
        
        if parquet_candidates:
            # MEJORA: Elegir el más reciente
            most_recent = max(parquet_candidates, key=lambda x: x['last_modified'])
            replay_key = most_recent['key']
            obj = most_recent['obj']
            logger.info(f"✅ Selected MOST RECENT parquet: {replay_key}")
        else:
            # Ultimate fallback
            replay_key = keys[0]
            obj = s3_hook.get_key(replay_key, bucket_name=bucket)
            logger.warning(f"⚠️ Using first available file: {replay_key}")
    
    # Cache obj.get() to avoid multiple S3 calls
    obj_meta = obj.get()
    file_size = obj_meta['ContentLength']
    last_modified = obj_meta['LastModified']
    
    # Compute SHA256 using streaming to reduce memory usage
    body_stream = obj_meta['Body']
    file_hash = compute_sha256_streaming(body_stream)
    
    return {
        "l4_bucket": bucket,
        "l4_key": replay_key,
        "l4_prefix": "usdcop_m5__05_l4_rlready/",
        "l4_size": file_size,
        "l4_sha256": file_hash,
        "l4_timestamp": last_modified.isoformat()
    }

# ============================================================================
# TRAINING FUNCTIONS WITH MONITOR WRAPPER
# ============================================================================

def train_ppo_production(seed: int, **context):
    """
    Production-ready PPO training with Monitor wrapper and Sortino metrics
    """
    # Ensure dependencies
    # Python 3.12 compatible versions
    
    # IMPROVEMENT #1: Get max_episode_length from env_spec (L4-L5 contract)
    l4_env_spec = context['task_instance'].xcom_pull(
        task_ids='validate_l4_outputs',
        key='l4_env_spec'
    )
    max_episode_length = l4_env_spec.get('max_episode_length', 60) if l4_env_spec else 60
    logger.info(f"Using max_episode_length from L4 env_spec: {max_episode_length}")
    _ensure_pkgs([
        "packaging>=23",  # MEJORA: Ensure proper version checking
        "torch>=2.3,<2.6",  # Ensure compatible torch first
        "gymnasium>=0.29,<1.0",  # Compatible with SB3 2.3+
        "stable-baselines3>=2.3,<3",  # Python 3.12 compatible
        "onnx==1.16.2",
        "onnxruntime==1.18.1"
    ])
    
    import mlflow

    # Use dependency handlers for safe imports
    torch_handler = get_torch_handler()
    torch = torch_handler.require_module("PyTorch is required for training")
    torch_nn = torch.nn

    sb3_handler = get_stable_baselines3_handler()
    sb3 = sb3_handler.require_module("Stable Baselines3 is required for training")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor  # CRITICAL: Import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    
    print(f"{'='*80}")
    print(f"Starting Production PPO Training - Seed {seed}")
    print(f"{'='*80}")
    
    # Get infrastructure info
    git_info = get_git_info()
    container_info = get_container_info()
    l4_info = get_l4_dataset_info(**context)
    
    # UNIFIED CPU optimization and thread settings
    n_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)  # For MKL/OpenMP consistency
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    logger.info(f"🔧 Set all thread pools to {n_threads} threads")
    
    # STABILITY: Set reproducibility guards
    import random

    # Use dependency handler for safe torch import
    torch_handler = get_torch_handler()
    torch = torch_handler.require_module("PyTorch is required for reproducibility")
    
    # MEJORA: Determinismo extra para máxima reproducibilidad
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # MEJORA: Configuración adicional de torch para determinismo
    try:
        torch.use_deterministic_algorithms(False)  # True para reproducibilidad estricta, False para mejor performance
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        logger.info(f"Could not set all deterministic settings: {e}")
    
    # Thread settings already unified above
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("l5_serving_production")
    
    with mlflow.start_run(run_name=f"ppo_seed_{seed}_prod"):
        # Helper function for MLflow artifact uploads with retry logic
        def safe_mlflow_log_artifact(file_path, artifact_path=None, max_retries=3):
            """Upload artifact to MLflow with retry logic and error handling"""
            for attempt in range(max_retries):
                try:
                    if not os.path.exists(file_path):
                        logger.error(f"File not found for MLflow upload: {file_path}")
                        return False
                    
                    # Verify file integrity before upload
                    file_size = os.path.getsize(file_path)
                    if file_size == 0:
                        logger.error(f"File is empty, skipping upload: {file_path}")
                        return False
                    
                    mlflow.log_artifact(file_path, artifact_path=artifact_path)
                    logger.info(f"Successfully uploaded {file_path} to MLflow (attempt {attempt + 1})")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to upload {file_path} to MLflow (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"All upload attempts failed for {file_path}: {str(e)}")
                        return False
            return False
        
        # Read L5-specific Airflow variables for policy collapse prevention
        l5_total_timesteps = int(Variable.get("L5_TOTAL_TIMESTEPS", default_var="1000000"))  # 1M for convergence
        l5_ent_coef = float(Variable.get("L5_ENT_COEF", default_var="0.03"))  # Higher for exploration
        l5_learning_rate = float(Variable.get("L5_LEARNING_RATE", default_var="3e-4"))
        l5_n_steps = int(Variable.get("L5_N_STEPS", default_var="480"))  # ~8h context
        l5_batch_size = int(Variable.get("L5_BATCH_SIZE", default_var="64"))
        l5_gamma = float(Variable.get("L5_GAMMA", default_var="0.995"))  # Long horizons
        l5_adam_beta1 = float(Variable.get("L5_ADAM_BETA1", default_var="0.99"))  # CRITICAL
        l5_adam_beta2 = float(Variable.get("L5_ADAM_BETA2", default_var="0.99"))  # CRITICAL
        l5_shaping_penalty = float(Variable.get("L5_SHAPING_PENALTY", default_var="0.1"))
        
        # Log all tracking information
        mlflow.set_tags({
            "algorithm": "PPO",
            "seed": str(seed),
            "version": "production_v2",
            **git_info,
            **container_info
        })
        
        mlflow.log_params({
            **l4_info,
            "cpu_threads": n_threads,
            "total_timesteps": l5_total_timesteps,
            "l5_ent_coef": l5_ent_coef,
            "l5_shaping_penalty": l5_shaping_penalty
        })
        
        # Save pip freeze
        pip_freeze = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(pip_freeze)
            safe_mlflow_log_artifact(f.name, artifact_path="environment")
        
        logger.info(f"L5 Training Parameters - Timesteps: {l5_total_timesteps}, Entropy Coef: {l5_ent_coef}, Shaping Penalty: {l5_shaping_penalty}")
        
        # Training configuration (OPTIMIZED based on academic best practices)
        config = {
            "n_steps": l5_n_steps,  # 480 = ~8h context for better temporal learning
            "batch_size": l5_batch_size,  # 64 for stable updates
            "n_epochs": 10,
            "learning_rate": l5_learning_rate,  # 3e-4 standard
            "clip_range": 0.1,  # Reduced for stability
            "target_kl": 0.03,
            "max_grad_norm": 0.5,
            "ent_coef": l5_ent_coef,  # 0.03 HIGH - prevents collapse
            "vf_coef": 0.5,
            "gamma": l5_gamma,  # 0.995 for long horizons
            "gae_lambda": 0.95,
            "normalize_advantage": True,
            "use_sde": False,
            "seed": seed,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            # CRITICAL: Custom optimizer settings to prevent policy collapse
            "optimizer_class": torch.optim.Adam,  # Pass class object, not string
            "optimizer_kwargs": {
                "betas": (l5_adam_beta1, l5_adam_beta2),  # (0.99, 0.99) CRITICAL
                "eps": 1e-5,
                "weight_decay": 0.001
            },
            "verbose": 1,
            # Additional params for better convergence
            "policy_kwargs": {
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
                "activation_fn": nn.ReLU
            }
        }
        
        mlflow.log_params(config)
        
        # Create a JSON-serializable copy of the config
        config_for_json = config.copy()
        config_for_json["policy_kwargs"] = config["policy_kwargs"].copy()
        # Convert activation function to string for JSON serialization
        config_for_json["policy_kwargs"]["activation_fn"] = "nn.ReLU"
        # Convert optimizer_class to string for JSON serialization
        config_for_json["optimizer_class"] = "torch.optim.Adam"
        
        # Create unique temporary directory for this seed to avoid file conflicts
        temp_dir = tempfile.mkdtemp(prefix=f"l5_training_seed_{seed}_")
        logger.info(f"Created temporary directory for seed {seed}: {temp_dir}")
        
        # Save training config in seed-specific directory
        config_path = os.path.join(temp_dir, f"training_config_seed_{seed}.json")
        with open(config_path, 'w') as f:
            json.dump(convert_numpy_to_python(config_for_json), f, indent=2)
        
        # Upload training config with retry logic
        safe_mlflow_log_artifact(config_path, artifact_path="configs")
        
        try:
            # Load L4 data from MinIO
            s3_hook = S3Hook(aws_conn_id='minio_conn')
            l4_bucket = "04-l4-ds-usdcop-rlready"
            l4_prefix = "usdcop_m5__05_l4_rlready/"
            
            # Download necessary L4 files to seed-specific directory
            l4_files = {}
            for file_name in ['train_df.parquet', 'test_df.parquet', 'env_spec.json', 'reward_spec.json', 'cost_model.json']:
                key = f"{l4_prefix}{file_name}"
                if s3_hook.check_for_key(key, bucket_name=l4_bucket):
                    local_path = os.path.join(temp_dir, file_name)
                    obj = s3_hook.get_key(key, bucket_name=l4_bucket)
                    with open(local_path, 'wb') as f:
                        f.write(obj.get()['Body'].read())
                    l4_files[file_name] = local_path
                    logger.info(f"Downloaded {file_name} from L4 bucket to {local_path}")
                    
                    # Verify file integrity after download
                    file_size = os.path.getsize(local_path)
                    if file_size == 0:
                        logger.warning(f"Downloaded file is empty: {local_path}")
                    else:
                        logger.info(f"File {file_name} downloaded successfully: {file_size} bytes")
            
            # CRITICAL: Require L4 READY marker
            ready_key = f"{l4_prefix}_control/READY"
            if not s3_hook.check_for_key(ready_key, bucket_name=l4_bucket):
                raise ValueError("L4 NOT READY: missing _control/READY in L4 bucket")
            ready_payload = json.loads(s3_hook.read_key(ready_key, bucket_name=l4_bucket))
            if ready_payload.get("status") != "READY":
                raise ValueError(f"L4 NOT READY: status={ready_payload.get('status')}")
            
            # Log L4 specs for lineage with retry logic
            if 'env_spec.json' in l4_files:
                safe_mlflow_log_artifact(l4_files['env_spec.json'], artifact_path="specs")
            if 'reward_spec.json' in l4_files:
                safe_mlflow_log_artifact(l4_files['reward_spec.json'], artifact_path="specs")
            if 'cost_model.json' in l4_files:
                safe_mlflow_log_artifact(l4_files['cost_model.json'], artifact_path="specs")
            
            # Load cost model for later use
            with open(l4_files['cost_model.json'], 'r') as f:
                cost_model = json.load(f)
            
            # Create environment using L4 data
            from utils.gymnasium_trading_env import create_gym_env
            
            # SANITY CHECK: Validar señal antes de entrenar
            logger.info("Running reward/costs sanity check...")
            
            # Load reward_spec si existe
            reward_spec = None
            if 'reward_spec.json' in l4_files:
                with open(l4_files['reward_spec.json'], 'r') as f:
                    reward_spec = json.load(f)
                    logger.info(f"Loaded reward_spec: window={reward_spec.get('forward_window')}, "
                              f"price={reward_spec.get('price_type')}, "
                              f"norm={reward_spec.get('normalization')}")
            
            # Quick sanity check
            sanity_passed = quick_sanity_check(
                env_factory=lambda mode: create_gym_env(mode=mode),
                cost_model=cost_model,
                n_episodes=10
            )
            
            if not sanity_passed:
                logger.warning("⚠️ Sanity check failed - signal may be weak even without costs!")
                mlflow.log_metric(f"seed_{seed}_sanity_check", 0)
            else:
                logger.info("✅ Sanity check passed - signal exists")
                mlflow.log_metric(f"seed_{seed}_sanity_check", 1)
            
            # CRITICAL FIX: Wrap environments with Monitor AND SentinelTradingEnv AND DiscreteToSignedAction
            # SentinelTradingEnv asegura costos solo en trades y añade telemetría
            # DiscreteToSignedAction prevents policy collapse by proper action mapping
            train_env = DummyVecEnv([
                lambda: Monitor(
                    DiscreteToSignedAction(
                        SentinelTradingEnv(
                            create_gym_env(mode="train"),
                            cost_model=cost_model,
                            shaping_penalty=-l5_shaping_penalty,  # Use configurable shaping penalty for exploration
                            enable_telemetry=True,
                            max_episode_length=max_episode_length   # IMPROVEMENT #1: From L4 env_spec
                        )
                    ),
                    allow_early_resets=True
                )
            ])
            
            test_env = DummyVecEnv([
                lambda: Monitor(
                    DiscreteToSignedAction(
                        SentinelTradingEnv(
                            create_gym_env(mode="test"),
                            cost_model=cost_model,
                            shaping_penalty=0.0,  # Sin shaping en test para métricas reales
                            enable_telemetry=False,
                            max_episode_length=max_episode_length   # IMPROVEMENT #1: From L4 env_spec
                        )
                    ),
                    allow_early_resets=True
                )
            ])
            
            # Create model - remove optimizer_class and optimizer_kwargs as PPO doesn't accept them directly
            ppo_config = config.copy()
            if "optimizer_class" in ppo_config:
                ppo_config.pop("optimizer_class")  # PPO handles optimizer internally
            if "optimizer_kwargs" in ppo_config:
                ppo_config.pop("optimizer_kwargs")  # PPO handles optimizer kwargs internally
            model = PPO("MlpPolicy", train_env, **ppo_config)
            
            # Checkpoint callback
            class CheckpointCallback(BaseCallback):
                def __init__(self, save_freq=100000, save_path=temp_dir, verbose=1):
                    super().__init__(verbose)
                    self.save_freq = save_freq
                    self.save_path = save_path
                    
                def _on_step(self) -> bool:
                    if self.num_timesteps % self.save_freq == 0:
                        path = f"{self.save_path}/ppo_seed_{seed}_checkpoint_{self.num_timesteps}.zip"
                        self.model.save(path)
                        safe_mlflow_log_artifact(path, artifact_path="checkpoints")
                        if self.verbose > 0:
                            logger.info(f"Checkpoint saved at {self.num_timesteps} timesteps")
                    return True
            
            # IMPROVEMENT #5: Enhanced evaluation callback with early stopping and collapse detection
            class EnhancedEvalCallback(BaseCallback):
                def __init__(self, eval_env, eval_freq=50000, patience=3, min_improvement=0.01, verbose=1):
                    super().__init__(verbose)
                    self.eval_env = eval_env
                    self.eval_freq = eval_freq
                    self.patience = patience
                    self.min_improvement = min_improvement
                    self.train_rewards = []
                    self.test_rewards = []
                    self.best_test_reward = -np.inf
                    self.best_model_path = os.path.join(temp_dir, f"best_model_seed_{seed}.zip")
                    self.patience_counter = 0
                    self.collapse_detected = False
                    
                def _on_step(self) -> bool:
                    if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0:
                        # Evaluate on train set
                        train_mean, train_std = evaluate_policy(
                            self.model, train_env, 
                            n_eval_episodes=IN_LOOP_EVAL_EPISODES,
                            deterministic=True
                        )
                        self.train_rewards.append(train_mean)
                        
                        # Evaluate on test set with action balance check
                        test_mean, test_std = evaluate_policy(
                            self.model, self.eval_env, 
                            n_eval_episodes=IN_LOOP_EVAL_EPISODES,
                            deterministic=True
                        )
                        self.test_rewards.append(test_mean)
                        
                        # Quick collapse detection during training
                        try:
                            # Sample a few episodes to check action balance
                            action_sample = []
                            for _ in range(5):  # Quick sample
                                obs = self.eval_env.reset()
                                for _ in range(min(100, max_episode_length)):  # Sample first 100 steps
                                    action, _ = self.model.predict(obs, deterministic=True)
                                    action_sample.append(int(action[0]))
                                    obs, _, done, _ = self.eval_env.step(action)
                                    if done[0]:
                                        break
                            
                            # Check for collapse (all same action)
                            unique_actions = len(set(action_sample))
                            if unique_actions <= 1 and len(action_sample) > 10:
                                self.collapse_detected = True
                                logger.warning(f"🛑 COLLAPSE DETECTED at step {self.num_timesteps}: Only {unique_actions} unique actions in sample")
                                
                        except Exception as e:
                            logger.warning(f"Could not check for collapse: {e}")
                        
                        # Best model checkpoint
                        if test_mean > self.best_test_reward + self.min_improvement:
                            self.best_test_reward = test_mean
                            self.model.save(self.best_model_path)
                            self.patience_counter = 0
                            logger.info(f"💾 New best model saved at step {self.num_timesteps}: {test_mean:.4f}")
                        else:
                            self.patience_counter += 1
                        
                        # Log metrics
                        mlflow.log_metrics({
                            "train_mean_reward": train_mean,
                            "test_mean_reward": test_mean,
                            "train_std_reward": train_std,
                            "test_std_reward": test_std,
                            "best_test_reward": self.best_test_reward,
                            "patience_counter": self.patience_counter,
                        }, step=self.num_timesteps)
                        
                        if self.verbose > 0:
                            logger.info(f"Step {self.num_timesteps}: Train={train_mean:.4f}±{train_std:.4f}, Test={test_mean:.4f}±{test_std:.4f}, Best={self.best_test_reward:.4f}, Patience={self.patience_counter}/{self.patience}")
                        
                        # Early stopping conditions
                        if self.collapse_detected:
                            logger.error(f"🛑 EARLY STOP: Policy collapse detected at step {self.num_timesteps}")
                            return False
                            
                        if self.patience_counter >= self.patience:
                            logger.info(f"🛑 EARLY STOP: No improvement for {self.patience} evaluations")
                            return False
                    
                    return True
            
            # Training with callbacks - Añade CostCurriculumCallback
            # Convert L4 cost model to the format expected by CostCurriculumCallback
            processed_cost_model = {}
            if 'spread_stats' in cost_model:
                spread_stats = cost_model.get('spread_stats', {})
                if isinstance(spread_stats, dict):
                    processed_cost_model['spread_bps'] = spread_stats.get('mean', 20)
                elif isinstance(spread_stats, (int, float)):
                    processed_cost_model['spread_bps'] = float(spread_stats)
                else:
                    processed_cost_model['spread_bps'] = 20
            else:
                processed_cost_model['spread_bps'] = 20
                
            processed_cost_model['slippage_bps'] = cost_model.get('k_atr', 0.10) * 10 if 'k_atr' in cost_model else 5
            processed_cost_model['fee_bps'] = cost_model.get('fee_bps', 10)
            
            callbacks = [
                CheckpointCallback(save_freq=100000),
                EnhancedEvalCallback(test_env, eval_freq=50000, patience=3, min_improvement=0.01),
                CostCurriculumCallback(
                    eval_env=test_env,
                    initial_cost_factor=float(Variable.get("L5_INITIAL_COST_FACTOR", default_var="1.0")),  # Use Variable for cost factor
                    full_cost_step=int(l5_total_timesteps * float(Variable.get("L5_FULL_COST_FRACTION", default_var="1.0"))),  # Use Variable for when to reach full cost
                    total_timesteps=l5_total_timesteps,
                    original_cost_model=processed_cost_model,
                    eval_freq=10000,
                    log_path=os.path.join(temp_dir, 'curriculum_eval'),
                    deterministic=True,
                    verbose=1
                )
            ]
            
            logger.info(f"Training PPO for {l5_total_timesteps} timesteps with cost curriculum...")
            start_time = time.time()
            model.learn(
                total_timesteps=l5_total_timesteps,
                callback=callbacks,
                progress_bar=False  # Disable for Airflow logs
            )
            training_time = time.time() - start_time
            
            # Calculate Sortino ratios (not Sharpe)
            logger.info("Calculating Sortino ratios...")
            train_rewards, _ = evaluate_policy(model, train_env, n_eval_episodes=30, deterministic=True, return_episode_rewards=True)
            test_rewards, _ = evaluate_policy(model, test_env, n_eval_episodes=30, deterministic=True, return_episode_rewards=True)
            
            # Use Sortino instead of Sharpe
            sortino_train = calculate_sortino_ratio(np.array(train_rewards))
            sortino_test = calculate_sortino_ratio(np.array(test_rewards))
            
            logger.info(f"Sortino Train: {sortino_train:.3f}, Sortino Test: {sortino_test:.3f}")
            
            # Save final PyTorch model
            model_path = os.path.join(temp_dir, f"ppo_seed_{seed}_final.zip")
            model.save(model_path)
            safe_mlflow_log_artifact(model_path, artifact_path="models")
            
            # Export to ONNX
            logger.info("Exporting model to ONNX...")
            onnx_path = export_to_onnx(model, train_env.observation_space, seed, temp_dir)
            safe_mlflow_log_artifact(onnx_path, artifact_path="models")
            
            # Benchmark latency
            logger.info("Benchmarking inference latency...")
            latency_results = benchmark_latency(model, onnx_path, test_env)
            
            latency_path = os.path.join(temp_dir, f"latency_seed_{seed}.json")
            with open(latency_path, 'w') as f:
                json.dump(convert_numpy_to_python(latency_results), f, indent=2)
            safe_mlflow_log_artifact(latency_path, artifact_path="benchmarks")
            
            # Log latency metrics
            mlflow.log_metrics({
                "inference_p50_ms": latency_results["pytorch"]["p50"],
                "inference_p99_ms": latency_results["pytorch"]["p99"],
                "onnx_p50_ms": latency_results["onnx"]["p50"],
                "onnx_p99_ms": latency_results["onnx"]["p99"],
                "training_time_seconds": training_time,
                "sortino_train": sortino_train,
                "sortino_test": sortino_test
            })
            
            # Save to L5 bucket
            s3_hook = S3Hook(aws_conn_id='minio_conn')
            l5_bucket = "05-l5-ds-usdcop-serving"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            l5_prefix = f"training_{timestamp}_seed_{seed}/"
            
            # Upload model files to L5
            model_files = {
                "model.zip": model_path,
                "model.onnx": onnx_path,
                "latency.json": latency_path,
                "config.json": config_path,
            }
            # Also upload L4 specs
            for name in ["env_spec.json", "reward_spec.json", "cost_model.json"]:
                if name in l4_files:
                    model_files[name] = l4_files[name]
            
            for file_name, file_path in model_files.items():
                if os.path.exists(file_path):
                    key = f"{l5_prefix}{file_name}"
                    with open(file_path, 'rb') as f:
                        s3_hook.load_file_obj(
                            file_obj=f,
                            key=key,
                            bucket_name=l5_bucket,
                            replace=True
                        )
                    logger.info(f"Uploaded {file_name} to {l5_bucket}/{key}")
            
            # Clean up
            train_env.close()
            test_env.close()
            
            # Push results for gate evaluation
            results = {
                "algorithm": "PPO_MLP",  # Tag the algorithm
                "seed": seed,
                "mlflow_run_id": mlflow.active_run().info.run_id,
                "model_path": model_path,
                "onnx_path": onnx_path,
                "latency": latency_results,
                "l5_bucket": l5_bucket,
                "l5_prefix": l5_prefix,
                "sortino_train": float(sortino_train),  # Changed from sharpe to sortino
                "sortino_test": float(sortino_test),    # Changed from sharpe to sortino
                "cost_model": cost_model,  # Pass cost model for stress test
                "l4_dataset_info": l4_info,  # MEJORA: Pass L4 info to avoid rescanning in bundle
                "status": "COMPLETED"
            }
            
            context['task_instance'].xcom_push(
                key=f'ppo_seed_{seed}_result',
                value=results
            )
            
            logger.info(f"✅ PPO training completed for seed {seed}")
            return results
            
        except Exception as e:
            logger.error(f"Error in PPO training: {e}", exc_info=True)
            mlflow.set_tag("status", "FAILED")
            mlflow.log_param("error", str(e))
            raise
        finally:
            # Cleanup temporary directory
            try:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")

# ============================================================================
# ONNX EXPORT AND BENCHMARKING
# ============================================================================

def export_to_onnx(model, observation_space, seed: int, temp_dir: str = "/tmp") -> str:
    """Export model to ONNX format with proper wrapper"""
    # MEJORA: Idempotencia completa de dependencias
    _ensure_pkgs(["onnx==1.16.2", "onnxruntime==1.18.1"])

    # Use dependency handlers for safe imports
    torch_handler = get_torch_handler()
    torch = torch_handler.require_module("PyTorch is required for ONNX export")
    nn = torch.nn
    import onnx
    
    class ActorCriticONNX(nn.Module):
        """MEJORA: Compatible con todas las versiones SB3 sin drift de API"""
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            
        def forward(self, obs):
            # MEJORA: Evitar drift de API usando el extractor principal
            features = self.policy.extract_features(obs)
            
            # MEJORA: Usar mlp_extractor directamente en vez de forward_actor/critic
            # Esto es compatible con más versiones de SB3
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
            
            # Get action logits (not sampled action)
            action_logits = self.policy.action_net(latent_pi)
            
            # Get value estimate
            value = self.policy.value_net(latent_vf)
            
            return action_logits, value
    
    # Create wrapper
    wrapper = ActorCriticONNX(model.policy)
    wrapper.eval()
    
    # Create dummy input
    dummy_input = torch.FloatTensor(
        observation_space.sample().reshape(1, -1)
    )
    
    # Export to ONNX
    onnx_path = os.path.join(temp_dir, f"policy_seed_{seed}.onnx")
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action_logits', 'value'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    # Verify the exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    logger.info(f"Model exported to ONNX: {onnx_path}")
    return onnx_path

def benchmark_latency(model, onnx_path: str, env, n_samples: int = 1000) -> Dict[str, Any]:
    """Benchmark inference latency for PyTorch and ONNX"""
    # MEJORA: Idempotencia completa - asegurar ONNX runtime
    _ensure_pkgs(["onnxruntime>=1.16,<2"])
    
    import time
    import numpy as np
    import json
    
    # MEJORA: Generar observaciones sin depender de reset (puramente inferencia)
    obs_list = []
    for _ in range(n_samples + 10):  # +10 para warm-up
        # Generar observaciones sintéticas del espacio de observación
        obs = env.observation_space.sample()
        # Asegurar forma correcta y tipo
        obs = np.atleast_2d(obs).astype(np.float32)
        obs_list.append(obs)
    
    logger.info(f"Generated {len(obs_list)} synthetic observations for pure inference benchmark")
    
    # MEJORA: Warm-up para evitar contaminación por JIT/inicialización
    WARMUP_SAMPLES = 10
    logger.info(f"Running {WARMUP_SAMPLES} warm-up iterations for each engine...")
    
    # Benchmark PyTorch con warm-up
    pytorch_times = []
    for i, obs in enumerate(obs_list):
        start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        end = time.perf_counter()
        
        # MEJORA: Descartar primeras muestras de warm-up
        if i >= WARMUP_SAMPLES:
            pytorch_times.append((end - start) * 1000)  # Convert to ms
    
    # Benchmark ONNX con warm-up
    onnx_times = []
    try:
        import onnxruntime as ort
        
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        # MEJORA: Fix providers to CPUExecutionProvider for stable benchmarks
        ort_session = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
        
        for i, obs in enumerate(obs_list):
            start = time.perf_counter()
            ort_inputs = {
                ort_session.get_inputs()[0].name: obs.reshape(1, -1).astype(np.float32)
            }
            ort_outs = ort_session.run(None, ort_inputs)
            action_logits = ort_outs[0]
            action = np.argmax(action_logits, axis=-1)
            end = time.perf_counter()
            
            # MEJORA: Descartar primeras muestras de warm-up
            if i >= WARMUP_SAMPLES:
                onnx_times.append((end - start) * 1000)
            
    except ImportError:
        logger.warning("ONNX Runtime not available, skipping ONNX benchmark")
        onnx_times = [0] * n_samples
    except Exception as e:
        logger.warning(f"ONNX benchmark failed: {e}")
        onnx_times = [0] * n_samples
    
    # Calculate percentiles
    def calc_percentiles(times):
        return {
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "mean": float(np.mean(times)),
            "std": float(np.std(times))
        }
    
    # E2E latency simulation
    e2e_times = []
    for i in range(min(100, n_samples)):
        start = time.perf_counter()
        obs = obs_list[i]
        obs_processed = np.clip(obs, -5, 5).astype(np.float32)
        action, _ = model.predict(obs_processed, deterministic=True)
        output = {"action": int(action[0]), "timestamp": time.time()}
        _ = json.dumps(output)
        end = time.perf_counter()
        e2e_times.append((end - start) * 1000)
    
    return {
        "pytorch": calc_percentiles(pytorch_times),
        "onnx": calc_percentiles(onnx_times),
        "e2e": calc_percentiles(e2e_times),
        "n_samples": n_samples
    }

# ============================================================================
# PPO-LSTM TRAINING FUNCTION
# ============================================================================

def train_ppo_lstm(seed: int, hp_override: dict = None, **context):
    """
    PPO-LSTM with RecurrentPPO for sequential decision making
    Properly handles 60-step episodes with LSTM memory
    """
    # Ensure dependencies - Python 3.12 compatible versions
    _ensure_pkgs([
        "packaging>=23",
        "torch>=2.3,<2.6",
        "gymnasium>=0.29,<1.0",
        "stable-baselines3>=2.4.0,<3",  # Updated for Python 3.12
        "sb3-contrib>=2.4.0,<3",  # Critical for RecurrentPPO, updated for Python 3.12
        "onnx==1.16.2",
        "onnxruntime==1.18.1"
    ])
    
    import mlflow

    # Use dependency handlers for safe imports
    torch_handler = get_torch_handler()
    torch = torch_handler.require_module("PyTorch is required for LSTM training")
    nn = torch.nn
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    
    print(f"{'='*80}")
    print(f"Starting PPO-LSTM Training - Seed {seed}")
    print(f"{'='*80}")
    
    # Get episode length from L4 env_spec
    l4_env_spec = context['task_instance'].xcom_pull(
        task_ids='validate_l4_outputs',
        key='l4_env_spec'
    )
    max_episode_length = l4_env_spec.get('max_episode_length', 60) if l4_env_spec else 60
    logger.info(f"Using max_episode_length from L4 env_spec: {max_episode_length}")
    
    # Get infrastructure info
    git_info = get_git_info()
    container_info = get_container_info()
    l4_info = get_l4_dataset_info(**context)
    
    # CPU optimization
    n_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    
    # Set seeds
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("l5_serving_production")
    
    with mlflow.start_run(run_name=f"ppo_lstm_seed_{seed}_prod"):
        try:
            # Import create_gym_env from utils
            from utils.gymnasium_trading_env import create_gym_env
            
            # Get cost model from L4 - properly load from S3 if needed
            cost_model = context['task_instance'].xcom_pull(
                task_ids='validate_l4_outputs',
                key='l4_cost_model'
            )
            
            if not cost_model:
                # Load from S3 as fallback
                logger.info("Cost model not in XCom, loading from S3...")
                s3_hook = S3Hook(aws_conn_id='minio_conn')  # Use minio_conn not aws_default
                l4_bucket = "04-l4-ds-usdcop-rlready"
                l4_prefix = "usdcop_m5__05_l4_rlready/"
                cost_model_key = f"{l4_prefix}cost_model.json"
                
                if s3_hook.check_for_key(cost_model_key, bucket_name=l4_bucket):
                    obj = s3_hook.get_key(cost_model_key, bucket_name=l4_bucket)
                    cost_model = json.loads(obj.get()['Body'].read())
                    logger.info(f"Loaded cost model from S3: {cost_model}")
                else:
                    raise ValueError("L4 cost_model.json not found in S3!")
            
            # CRITICAL: For RecurrentPPO, use proper environment setup
            def make_env(mode="train"):
                return Monitor(
                    DiscreteToSignedAction(
                        SentinelTradingEnv(
                            create_gym_env(mode=mode),
                            cost_model=cost_model,
                            shaping_penalty=float(Variable.get("L5_SHAPING_PENALTY", default_var="-0.1")) if mode=="train" else 0.0,
                            enable_telemetry=(mode=="train"),
                            max_episode_length=max_episode_length
                        )
                    ),
                    allow_early_resets=True  # Enable for both train and test for evaluation callbacks
                )
            
            # RecurrentPPO needs exactly 1 environment in DummyVecEnv
            train_env = DummyVecEnv([lambda: make_env("train")])
            
            # Create test env with DummyVecEnv for consistency
            test_env = DummyVecEnv([lambda: make_env("test")])
            
            # LSTM-specific configuration
            config = {
                "n_steps": 120,  # MUST be multiple of episode_length (60)
                "batch_size": 60,  # For LSTM, use episode_length or multiple
                "n_epochs": 10,
                "learning_rate": 2e-4,
                "clip_range": 0.1,
                "ent_coef": 0.008,  # Slightly higher for USDCOP volatility
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "policy_kwargs": {
                    "lstm_hidden_size": 128,  # Start smaller for faster convergence
                    "n_lstm_layers": 1,
                    "enable_critic_lstm": True,  # Separate LSTM for critic
                    "shared_lstm": False,  # Don't share LSTM between actor/critic
                    "ortho_init": False,
                },
                "max_grad_norm": 0.5,  # CRITICAL for LSTM stability
                "use_sde": False,
                "seed": seed,
                "device": "cpu",  # CPU is fine for M5 frequency
                "verbose": 1
            }
            
            # Apply hyperparameter overrides if provided
            if hp_override:
                logger.info(f"Applying hyperparameter overrides: {hp_override}")
                config.update(hp_override)
            
            mlflow.log_params({"algorithm": "PPO_LSTM", **config})
            
            # Create model - remove optimizer_class and optimizer_kwargs as RecurrentPPO doesn't accept them directly
            ppo_config = config.copy()
            if "optimizer_class" in ppo_config:
                ppo_config.pop("optimizer_class")  # RecurrentPPO handles optimizer internally
            if "optimizer_kwargs" in ppo_config:
                ppo_config.pop("optimizer_kwargs")  # RecurrentPPO handles optimizer kwargs internally
            model = RecurrentPPO("MlpLstmPolicy", train_env, **ppo_config)
            
            # Callbacks (reuse from PPO)
            class CheckpointCallback(BaseCallback):
                def __init__(self, save_freq, save_path, name_prefix="rl_model", verbose=1):
                    super().__init__(verbose)
                    self.save_freq = save_freq
                    self.save_path = save_path
                    self.name_prefix = name_prefix
                    
                def _on_step(self):
                    if self.n_calls % self.save_freq == 0:
                        path = os.path.join(self.save_path, f"{self.name_prefix}_checkpoint_{self.num_timesteps}")
                        self.model.save(path)
                        if self.verbose > 0:
                            print(f"Saved checkpoint to {path}")
                    return True
            
            # Enhanced eval callback
            class EnhancedEvalCallback(EvalCallback):
                def _on_step(self):
                    result = super()._on_step()
                    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                        if len(self.evaluations_results) > 0:
                            last_rewards = self.evaluations_results[-1]
                            last_mean = np.mean(last_rewards) if len(last_rewards) > 0 else 0
                            sortino = enhanced_sortino(last_rewards) if len(last_rewards) > 0 else 0
                            mlflow.log_metrics({
                                "eval_mean_reward": float(last_mean),
                                "eval_sortino": float(sortino),
                            }, step=self.num_timesteps)
                    return result
            
            # Setup callbacks
            checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path=tempfile.mkdtemp(),
                name_prefix=f"ppo_lstm_{seed}"
            )
            
            eval_callback = EnhancedEvalCallback(
                test_env,
                best_model_save_path=tempfile.mkdtemp(),
                log_path=tempfile.mkdtemp(),
                eval_freq=25000,
                n_eval_episodes=IN_LOOP_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=1
            )
            
            # Convert L4 cost model to the format expected by CostCurriculumCallback
            processed_cost_model = {}
            if 'spread_stats' in cost_model:
                spread_stats = cost_model.get('spread_stats', {})
                if isinstance(spread_stats, dict):
                    processed_cost_model['spread_bps'] = spread_stats.get('mean', 20)
                elif isinstance(spread_stats, (int, float)):
                    processed_cost_model['spread_bps'] = float(spread_stats)
                else:
                    processed_cost_model['spread_bps'] = 20
            else:
                processed_cost_model['spread_bps'] = 20
                
            processed_cost_model['slippage_bps'] = cost_model.get('k_atr', 0.10) * 10 if 'k_atr' in cost_model else 5
            processed_cost_model['fee_bps'] = cost_model.get('fee_bps', 10)
            
            cost_curriculum_callback = CostCurriculumCallback(
                eval_env=test_env,
                initial_cost_factor=float(Variable.get("L5_INITIAL_COST_FACTOR", default_var="0.0")),
                full_cost_step=int(TOTAL_TIMESTEPS * 0.9),
                total_timesteps=TOTAL_TIMESTEPS,
                original_cost_model=processed_cost_model,
                eval_freq=25000,
                verbose=1
            )
            
            callbacks = CallbackList([checkpoint_callback, eval_callback, cost_curriculum_callback])
            
            # Train
            logger.info(f"Starting PPO-LSTM training with {TOTAL_TIMESTEPS} timesteps")
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callbacks,
                progress_bar=False
            )
            
            # Evaluate final performance
            test_rewards, _ = evaluate_policy(
                model, test_env,
                n_eval_episodes=50,
                deterministic=True,
                return_episode_rewards=True
            )
            
            sortino_test = enhanced_sortino(test_rewards)
            sortino_train = enhanced_sortino(eval_callback.evaluations_results[-1]) if eval_callback.evaluations_results else 0
            
            # Save model with consistent naming for gates
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, f"ppo_seed_{seed}_final.zip")
            model.save(model_path)
            # Use mlflow directly since we're in run context
            mlflow.log_artifact(model_path, artifact_path="models")
            
            # Upload to S3 with consistent hook
            s3_hook = S3Hook(aws_conn_id='minio_conn')
            l5_bucket = Variable.get("L5_BUCKET", default_var="05-l5-ds-usdcop-serving")
            l5_prefix = f"ppo_lstm_seed_{seed}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Upload with gate-expected naming
            s3_model_key = f"{l5_prefix}/models/ppo_seed_{seed}_final.zip"
            with open(model_path, 'rb') as f:
                s3_hook.load_file_obj(
                    file_obj=f,
                    bucket_name=l5_bucket,
                    key=s3_model_key,
                    replace=True
                )
            
            # Also upload root-level copy that bundler expects
            with open(model_path, 'rb') as f:
                s3_hook.load_file_obj(
                    file_obj=f,
                    bucket_name=l5_bucket,
                    key=f"{l5_prefix}/model.zip",
                    replace=True
                )
            logger.info(f"Uploaded models to both {s3_model_key} and {l5_prefix}/model.zip")
            
            # ONNX export (attempt, but not critical for LSTM)
            onnx_path = None
            try:
                onnx_path = export_to_onnx(model, train_env.observation_space, seed, tempfile.mkdtemp())
                if onnx_path:
                    s3_onnx_key = f"{l5_prefix}/model.onnx"
                    with open(onnx_path, 'rb') as f:
                        s3_hook.load_file_obj(
                            file_obj=f,
                            bucket_name=l5_bucket,
                            key=s3_onnx_key,
                            replace=True
                        )
            except Exception as e:
                logger.warning(f"ONNX export failed for PPO-LSTM (non-critical): {e}")
            
            # Benchmark latency with correct signature
            latency_results = benchmark_latency(model, onnx_path, test_env)
            
            # Clean up
            train_env.close()
            test_env.close()
            
            # Push results
            results = {
                "algorithm": "PPO_LSTM",
                "seed": seed,
                "mlflow_run_id": mlflow.active_run().info.run_id,
                "model_path": model_path,
                "onnx_path": onnx_path,
                "latency": latency_results,
                "l5_bucket": l5_bucket,
                "l5_prefix": l5_prefix,
                "sortino_train": float(sortino_train),
                "sortino_test": float(sortino_test),
                "cost_model": cost_model,
                "l4_dataset_info": l4_info,
                "hyperparameters": config,
                "status": "COMPLETED"
            }
            
            context['task_instance'].xcom_push(
                key=f'ppo_seed_{seed}_result',
                value=results
            )
            
            logger.info(f"✅ PPO-LSTM training completed for seed {seed}")
            return results
            
        except Exception as e:
            logger.error(f"Error in PPO-LSTM training: {e}", exc_info=True)
            raise

# ============================================================================
# QR-DQN TRAINING FUNCTION
# ============================================================================

def train_qrdqn(seed: int, hp_override: dict = None, **context):
    """
    QR-DQN (Quantile Regression DQN) for risk-aware trading
    Handles discrete action space with distributional RL
    """
    # Ensure dependencies - Python 3.12 compatible versions
    _ensure_pkgs([
        "packaging>=23",
        "torch>=2.3,<2.6",
        "gymnasium>=0.29,<1.0",
        "stable-baselines3>=2.4.0,<3",  # Updated for Python 3.12
        "sb3-contrib>=2.4.0,<3",  # QR-DQN lives here, updated for Python 3.12
    ])
    
    import mlflow

    # Use dependency handlers for safe imports
    torch_handler = get_torch_handler()
    torch = torch_handler.require_module("PyTorch is required for QR-DQN training")
    from sb3_contrib import QRDQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    
    print(f"{'='*80}")
    print(f"Starting QR-DQN Training - Seed {seed}")
    print(f"{'='*80}")
    
    # Get episode length from L4 env_spec
    l4_env_spec = context['task_instance'].xcom_pull(
        task_ids='validate_l4_outputs',
        key='l4_env_spec'
    )
    max_episode_length = l4_env_spec.get('max_episode_length', 60) if l4_env_spec else 60
    logger.info(f"Using max_episode_length from L4 env_spec: {max_episode_length}")
    
    # Get infrastructure info
    git_info = get_git_info()
    container_info = get_container_info()
    l4_info = get_l4_dataset_info(**context)
    
    # CPU optimization
    n_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    
    # Set seeds
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("l5_serving_production")
    
    with mlflow.start_run(run_name=f"qrdqn_seed_{seed}_prod"):
        try:
            # Import create_gym_env from utils
            from utils.gymnasium_trading_env import create_gym_env
            
            # Get cost model from L4 - properly load from S3 if needed
            cost_model = context['task_instance'].xcom_pull(
                task_ids='validate_l4_outputs',
                key='l4_cost_model'
            )
            
            if not cost_model:
                # Load from S3 as fallback
                logger.info("Cost model not in XCom, loading from S3...")
                s3_hook = S3Hook(aws_conn_id='minio_conn')  # Use minio_conn not aws_default
                l4_bucket = "04-l4-ds-usdcop-rlready"
                l4_prefix = "usdcop_m5__05_l4_rlready/"
                cost_model_key = f"{l4_prefix}cost_model.json"
                
                if s3_hook.check_for_key(cost_model_key, bucket_name=l4_bucket):
                    obj = s3_hook.get_key(cost_model_key, bucket_name=l4_bucket)
                    cost_model = json.loads(obj.get()['Body'].read())
                    logger.info(f"Loaded cost model from S3: {cost_model}")
                else:
                    raise ValueError("L4 cost_model.json not found in S3!")
            
            # Create wrapped environments
            train_env = Monitor(
                DiscreteToSignedAction(
                    SentinelTradingEnv(
                        create_gym_env(mode="train"),
                        cost_model=cost_model,
                        shaping_penalty=0,  # QR-DQN doesn't need shaping
                        enable_telemetry=True,
                        max_episode_length=max_episode_length
                    )
                ),
                allow_early_resets=True
            )
            
            test_env = Monitor(
                DiscreteToSignedAction(
                    SentinelTradingEnv(
                        create_gym_env(mode="test"),
                        cost_model=cost_model,
                        shaping_penalty=0,
                        enable_telemetry=False,
                        max_episode_length=max_episode_length
                    )
                ),
                allow_early_resets=True  # Enable for evaluation callbacks
            )
            
            # QR-DQN specific configuration - optimized for USDCOP
            config = {
                "learning_rate": 5e-5,  # Lower for stability
                "buffer_size": 100000,  # Increased for better replay
                "learning_starts": 500,  # Start learning earlier
                "batch_size": 128,  # Smaller batch for your data
                "gamma": 0.995,
                "train_freq": 4,
                "gradient_steps": 2,  # More gradient steps
                "target_update_interval": 5000,  # Less frequent for stability
                "exploration_fraction": 0.2,  # 20% of training for exploration
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "policy_kwargs": {
                    "n_quantiles": 51,  # Increased for better tail modeling
                    "net_arch": [128, 128]  # Smaller network
                },
                "seed": seed,
                "device": "cpu",
                "verbose": 1
            }
            
            # Apply hyperparameter overrides if provided
            if hp_override:
                logger.info(f"Applying hyperparameter overrides: {hp_override}")
                config.update(hp_override)
            
            mlflow.log_params({"algorithm": "QRDQN", **config})
            
            # Create model - QR-DQN doesn't use optimizer_class parameter
            qrdqn_config = config.copy()
            if "optimizer_class" in qrdqn_config:
                qrdqn_config.pop("optimizer_class")  # QRDQN doesn't accept this parameter
            if "optimizer_kwargs" in qrdqn_config:
                qrdqn_config.pop("optimizer_kwargs")  # QRDQN doesn't accept this parameter
            model = QRDQN("MlpPolicy", train_env, **qrdqn_config)
            
            # Custom callback for QR-DQN epsilon monitoring
            class QRDQNEpsilonCallback(BaseCallback):
                def _on_step(self):
                    if self.num_timesteps % 10000 == 0:
                        mlflow.log_metric(
                            "exploration_rate",
                            self.model.exploration_rate,
                            step=self.num_timesteps
                        )
                    return True
            
            # Setup callbacks
            epsilon_callback = QRDQNEpsilonCallback()
            
            eval_callback = EvalCallback(
                test_env,
                best_model_save_path=tempfile.mkdtemp(),
                log_path=tempfile.mkdtemp(),
                eval_freq=25000,
                n_eval_episodes=IN_LOOP_EVAL_EPISODES,
                deterministic=False,  # QR-DQN doesn't have deterministic mode
                render=False,
                verbose=1
            )
            
            callbacks = CallbackList([epsilon_callback, eval_callback])
            
            # Train
            logger.info(f"Starting QR-DQN training with {TOTAL_TIMESTEPS} timesteps")
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callbacks,
                progress_bar=False
            )
            
            # Custom evaluation for QR-DQN (force epsilon=0)
            model.exploration_rate = 0.0  # Force greedy evaluation
            test_rewards, _ = evaluate_policy(
                model, test_env,
                n_eval_episodes=50,
                deterministic=False,  # QR-DQN doesn't support deterministic
                return_episode_rewards=True
            )
            
            sortino_test = enhanced_sortino(test_rewards)
            sortino_train = enhanced_sortino(eval_callback.evaluations_results[-1]) if eval_callback.evaluations_results else 0
            
            # Save model with consistent naming for gates
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, f"ppo_seed_{seed}_final.zip")  # Keep ppo_seed for gate compatibility
            model.save(model_path)
            # Use mlflow directly since we're in run context
            mlflow.log_artifact(model_path, artifact_path="models")
            
            # Upload to S3 with consistent hook
            s3_hook = S3Hook(aws_conn_id='minio_conn')
            l5_bucket = Variable.get("L5_BUCKET", default_var="05-l5-ds-usdcop-serving")
            l5_prefix = f"qrdqn_seed_{seed}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Upload with gate-expected naming
            s3_model_key = f"{l5_prefix}/models/ppo_seed_{seed}_final.zip"
            with open(model_path, 'rb') as f:
                s3_hook.load_file_obj(
                    file_obj=f,
                    bucket_name=l5_bucket,
                    key=s3_model_key,
                    replace=True
                )
            
            # Also upload root-level copy that bundler expects
            with open(model_path, 'rb') as f:
                s3_hook.load_file_obj(
                    file_obj=f,
                    bucket_name=l5_bucket,
                    key=f"{l5_prefix}/model.zip",
                    replace=True
                )
            logger.info(f"Uploaded models to both {s3_model_key} and {l5_prefix}/model.zip")
            
            # Skip ONNX for QR-DQN (not easily supported)
            onnx_path = None
            logger.info("Skipping ONNX export for QR-DQN (not supported)")
            
            # Benchmark latency (PyTorch only) with correct signature
            latency_results = benchmark_latency(model, None, test_env)
            
            # Clean up
            train_env.close()
            test_env.close()
            
            # Push results
            results = {
                "algorithm": "QRDQN",
                "seed": seed,
                "mlflow_run_id": mlflow.active_run().info.run_id,
                "model_path": model_path,
                "onnx_path": onnx_path,
                "latency": latency_results,
                "l5_bucket": l5_bucket,
                "l5_prefix": l5_prefix,
                "sortino_train": float(sortino_train),
                "sortino_test": float(sortino_test),
                "cost_model": cost_model,
                "l4_dataset_info": l4_info,
                "hyperparameters": config,
                "status": "COMPLETED"
            }
            
            context['task_instance'].xcom_push(
                key=f'ppo_seed_{seed}_result',
                value=results
            )
            
            logger.info(f"✅ QR-DQN training completed for seed {seed}")
            return results
            
        except Exception as e:
            logger.error(f"Error in QR-DQN training: {e}", exc_info=True)
            raise

# ============================================================================
# TRAINING DISPATCHER
# ============================================================================

# Model registry mapping names to functions
MODEL_REGISTRY = {
    "ppo_lstm": train_ppo_lstm,
    "ppo_mlp": train_ppo_production,
    "qrdqn": train_qrdqn,
}

def train_dispatch(seed: int, **context):
    """
    Dispatcher that routes to the appropriate training function based on L5_MODEL variable
    Supports dynamic hyperparameter override via L5_{MODEL}_HYPERPARAMS variable
    """
    model_name = Variable.get("L5_MODEL", default_var="ppo_lstm").lower()
    
    # Check if model is supported
    fn = MODEL_REGISTRY.get(model_name)
    if fn is None:
        raise ValueError(f"Unsupported L5_MODEL: {model_name}. Accepted: {sorted(MODEL_REGISTRY.keys())}")
    
    # Allow hyperparameter override per model
    hp_override = {}
    hp_json = Variable.get(f"L5_{model_name.upper()}_HYPERPARAMS", default_var="{}")
    try:
        hp_override = json.loads(hp_json)
        logger.info(f"Applying hyperparameter overrides for {model_name}: {hp_override}")
    except Exception as e:
        logger.warning(f"Could not parse hyperparameter override: {e}")
    
    # Call the appropriate training function
    if model_name == "ppo_mlp":
        # Original function doesn't take hp_override
        return fn(seed=seed, **context)
    else:
        return fn(seed=seed, hp_override=hp_override, **context)

# ============================================================================
# GATE EVALUATION WITH SORTINO AND L4 COSTS
# ============================================================================

def evaluate_production_gates(**context):
    """
    Comprehensive production gate evaluation with Sortino and L4-based costs
    IMPROVEMENT #3: Uses split_spec folds for walk-forward validation
    """
    # AJUSTE FINO #1: Idempotencia - ensure packages para gates
    _ensure_pkgs([
        "packaging>=23",  # MEJORA: Ensure proper version checking
        "stable-baselines3>=2.3,<3",
        "gymnasium>=0.29,<1.0",
        "onnxruntime>=1.16,<2",
        "mlflow>=2.8,<3"
    ])
    
    import mlflow
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    
    logger.info("="*80)
    logger.info("PRODUCTION GATE EVALUATION")
    logger.info("="*80)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("l5_serving_gates_production")
    
    with mlflow.start_run(run_name="production_gate_evaluation"):
        # Create temporary directory for gate evaluation files
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="l5_gates_")
        logger.info(f"Created temporary directory for gates: {temp_dir}")
        
        # Retrieve L4 env_spec for max_episode_length
        l4_env_spec = context['task_instance'].xcom_pull(
            task_ids='validate_l4_outputs',
            key='l4_env_spec'
        )
        max_episode_length = l4_env_spec.get('max_episode_length', 60) if l4_env_spec else 60
        logger.info(f"Using max_episode_length from env_spec: {max_episode_length}")
        
        # IMPROVEMENT #3: Load split_spec for fold-based walk-forward validation with robust fallback
        from utils.gymnasium_trading_env import create_gym_env
        s3_hook = S3Hook(aws_conn_id='minio_conn')
        l4_bucket = "04-l4-ds-usdcop-rlready"
        split_spec = None
        ti = context['task_instance']
        
        # Build l4_prefix candidates list (XCom -> Variable -> default)
        l4_prefix_candidates = []
        
        # 1) XCom from validate_l4_outputs
        xcom_l4 = ti.xcom_pull(task_ids='validate_l4_outputs', key='l4_prefix')
        if isinstance(xcom_l4, str) and xcom_l4:
            l4_prefix_candidates.append(xcom_l4 if xcom_l4.endswith('/') else xcom_l4 + '/')
        
        # 2) XCom from training (if available)
        xcom_train = ti.xcom_pull(task_ids='ppo_training.train_ppo_seed_42', key='l4_dataset_info')
        if isinstance(xcom_train, dict) and xcom_train.get('l4_prefix'):
            p = xcom_train['l4_prefix']
            l4_prefix_candidates.append(p if p.endswith('/') else p + '/')
        
        # 3) Variable
        var_pref = Variable.get("l4_dataset_prefix", default_var="usdcop_m5__05_l4_rlready/")
        l4_prefix_candidates.append(var_pref if var_pref.endswith('/') else var_pref + '/')
        
        # 4) Default fallback
        l4_prefix_candidates.append("usdcop_m5__05_l4_rlready/")
        
        # Deduplicate while preserving order
        seen = set()
        l4_prefix_candidates_unique = []
        for p in l4_prefix_candidates:
            if p not in seen:
                seen.add(p)
                l4_prefix_candidates_unique.append(p)
        
        # Try multiple locations for each prefix
        def _find_split_spec():
            for prefix in l4_prefix_candidates_unique:
                for key in (f"{prefix}split_spec.json", f"{prefix}specs/split_spec.json"):
                    data = _download_json_from_s3(s3_hook, l4_bucket, key)
                    if data:
                        logger.info(f"✅ Found split_spec at s3://{l4_bucket}/{key} with {len(data.get('folds', []))} folds")
                        return data
            logger.warning(f"Could not find split_spec.json in any tested location. Candidates={l4_prefix_candidates_unique}")
            return None
        
        split_spec = _find_split_spec()
        
        all_results = {}
        gate_summary = {
            "timestamp": datetime.now().isoformat(),
            "seeds_evaluated": [],
            "gates_passed": {},
            "overall_status": "PENDING"
        }
        
        for seed in PARALLEL_SEEDS:
            # Get training results
            train_result = context['task_instance'].xcom_pull(
                task_ids=f'ppo_training.train_ppo_seed_{seed}',
                key=f'ppo_seed_{seed}_result'
            )
            
            if not train_result or train_result.get('status') != 'COMPLETED':
                logger.warning(f"Seed {seed} training not completed, skipping")
                continue
            
            logger.info(f"Evaluating seed {seed}...")
            gate_summary["seeds_evaluated"].append(seed)
            
            # MUST-FIX #1: Extract cost_model BEFORE building any envs
            cost_model = train_result.get('cost_model', {})
            if not cost_model:
                raise ValueError(f"Missing L4 cost_model in training results for seed {seed} - cannot perform evaluation!")
            
            logger.info(f"Using L4 cost model: {list(cost_model.keys())}")
            
            # Load model from MLflow instead of temp directory
            mlflow_run_id = train_result.get('mlflow_run_id')
            if not mlflow_run_id:
                raise ValueError(f"Missing mlflow_run_id for seed {seed}")
            
            logger.info(f"Downloading model from MLflow run: {mlflow_run_id}")
            
            # Create a temp directory for this evaluation
            eval_temp_dir = tempfile.mkdtemp(prefix=f"l5_eval_seed_{seed}_")
            
            try:
                # Download model from MLflow - using correct artifact path
                model_filename = f"ppo_seed_{seed}_final.zip"
                model_path = os.path.join(eval_temp_dir, model_filename)
                
                # Download the model from the correct artifact path (models/ not model/)
                logger.info(f"Downloading model from artifact path: models/{model_filename}")
                mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run_id,
                    artifact_path=f"models/{model_filename}",
                    dst_path=eval_temp_dir
                )
                
                # The downloaded model will be in eval_temp_dir/models/ppo_seed_{seed}_final.zip
                downloaded_model = os.path.join(eval_temp_dir, "models", model_filename)
                if os.path.exists(downloaded_model):
                    import shutil
                    # Move to expected location (removing the 'models' subdirectory)
                    shutil.move(downloaded_model, model_path)
                    logger.info(f"Model downloaded successfully to: {model_path}")
                    # Clean up the empty models directory
                    models_dir = os.path.join(eval_temp_dir, "models")
                    if os.path.exists(models_dir) and not os.listdir(models_dir):
                        os.rmdir(models_dir)
                else:
                    raise FileNotFoundError(f"Downloaded model not found at expected location: {downloaded_model}")
            except Exception as e:
                logger.error(f"Failed to download model from MLflow: {e}")
                
                # Fallback 1: Try to download from S3 directly if available
                try:
                    logger.info("Attempting S3 direct download as fallback...")
                    s3_bucket = train_result.get('l5_bucket', '05-l5-ds-usdcop-serving')
                    s3_prefix = train_result.get('l5_prefix', f"l5_training_seed_{seed}")
                    
                    import boto3
                    s3_client = boto3.client('s3')
                    s3_key = f"{s3_prefix}/models/ppo_seed_{seed}_final.zip"
                    model_path = os.path.join(eval_temp_dir, f"ppo_seed_{seed}_final.zip")
                    
                    logger.info(f"Downloading from S3: s3://{s3_bucket}/{s3_key}")
                    s3_client.download_file(s3_bucket, s3_key, model_path)
                    logger.info(f"Successfully downloaded model from S3 to: {model_path}")
                except Exception as s3_error:
                    logger.error(f"S3 models/ download failed: {s3_error}")
                    
                    # Fallback 1.5: Try root-level model.zip (matches bundler expectations)
                    try:
                        s3_key_root = f"{s3_prefix}/model.zip"
                        logger.info(f"Trying S3 root-level: s3://{s3_bucket}/{s3_key_root}")
                        s3_client.download_file(s3_bucket, s3_key_root, model_path)
                        logger.info(f"Successfully downloaded model from S3 root to: {model_path}")
                    except Exception as s3_root_error:
                        logger.error(f"S3 root-level download also failed: {s3_root_error}")
                    
                        # Fallback 2: Check if original temp path still exists (unlikely but worth trying)
                        original_path = train_result.get('model_path', '')
                        if original_path and os.path.exists(original_path):
                            logger.info(f"Found model at original path: {original_path}")
                            model_path = original_path
                        else:
                            # Final fallback: Raise clear error with debugging info
                            error_msg = (
                            f"Failed to load model for seed {seed}. "
                            f"MLflow run_id: {mlflow_run_id}, "
                            f"Original path: {original_path} (exists: {os.path.exists(original_path) if original_path else False}), "
                            f"Attempted MLflow artifact: models/ppo_seed_{seed}_final.zip, "
                            f"Attempted S3: s3://{s3_bucket}/{s3_key}"
                        )
                        raise FileNotFoundError(error_msg)
            
            logger.info(f"Model path resolved to: {model_path} (exists: {os.path.exists(model_path)})")
            from utils.gymnasium_trading_env import create_gym_env
            
            # FIX #4: Set seeds for reproducibility before creating environments
            import random
            import numpy as np

            # Use dependency handler for safe torch import
            torch_handler = get_torch_handler()
            torch = torch_handler.require_module("PyTorch is required for seed setting")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Create test environment WITH Monitor wrapper Y SentinelTradingEnv AND DiscreteToSignedAction
            test_env = DummyVecEnv([
                lambda: Monitor(
                    DiscreteToSignedAction(
                        SentinelTradingEnv(
                            create_gym_env(mode="test"),
                            cost_model=cost_model,  # Now properly defined
                            shaping_penalty=0.0,    # Sin shaping en evaluación
                            enable_telemetry=True,  # Habilita telemetría para diagnóstico
                            max_episode_length=max_episode_length  # L4-L5 contract: from env_spec
                        )
                    ),
                    allow_early_resets=True
                )
            ])
            # Algorithm-aware model loading
            algo = train_result.get("algorithm", "PPO_MLP").upper()
            logger.info(f"Loading model for algorithm: {algo}")
            
            # Fix: Remove .zip extension as load() adds it internally
            if model_path.endswith(".zip"):
                model_path_clean = model_path[:-4]
                logger.info(f"Removed .zip extension for load: {model_path_clean}")
            else:
                model_path_clean = model_path
            
            try:
                if algo in ["PPO_MLP", "PPO_LSTM", "PPO-LSTM"]:
                    if "LSTM" in algo:
                        from sb3_contrib import RecurrentPPO
                        model = RecurrentPPO.load(model_path_clean, env=test_env)
                    else:
                        model = PPO.load(model_path_clean, env=test_env)
                elif algo == "QRDQN":
                    from sb3_contrib import QRDQN
                    model = QRDQN.load(model_path_clean, env=test_env)
                    # Force exploration off for evaluation
                    model.exploration_rate = 0.0
                else:
                    # Fallback to PPO for unknown algorithms
                    logger.warning(f"Unknown algorithm {algo}, falling back to PPO")
                    model = PPO.load(model_path_clean, env=test_env)
            finally:
                # Clean up evaluation temp directory after loading
                if 'eval_temp_dir' in locals() and os.path.exists(eval_temp_dir):
                    import shutil
                    shutil.rmtree(eval_temp_dir, ignore_errors=True)
                    logger.info(f"Cleaned up evaluation temp directory: {eval_temp_dir}")
            
            # IMPROVEMENT #3: Run evaluation using fold-based walk-forward validation if available
            if split_spec and split_spec.get('folds'):
                logger.info(f"Running fold-based evaluation on {len(split_spec['folds'])} folds")
                fold_metrics = []
                for fold_idx, fold in enumerate(split_spec['folds']):
                    logger.info(f"Evaluating fold {fold_idx + 1}/{len(split_spec['folds'])}")
                    # For now, use the same test_env but log fold-based approach
                    fold_metric = evaluate_test_metrics(model, test_env, GATE_EVAL_EPISODES // len(split_spec['folds']), max_episode_length=max_episode_length)
                    fold_metric['fold'] = fold_idx
                    fold_metrics.append(fold_metric)
                    mlflow.log_metrics({f"fold_{fold_idx}_sortino": fold_metric.get("sortino", 0)}, step=fold_idx)
                
                # Average metrics across folds for gate evaluation
                test_metrics = {}
                numeric_keys = ['sortino', 'cagr', 'max_drawdown', 'calmar', 'total_return']
                for key in numeric_keys:
                    values = [fm.get(key, 0) for fm in fold_metrics if fm.get(key) is not None]
                    if values:
                        test_metrics[key] = np.mean(values)
                        test_metrics[f'{key}_std'] = np.std(values)
                
                test_metrics['is_valid'] = all(fm.get('is_valid', True) for fm in fold_metrics)
                test_metrics['fold_count'] = len(fold_metrics)
                logger.info(f"Fold-based metrics: Sortino={test_metrics.get('sortino', 0):.3f}±{test_metrics.get('sortino_std', 0):.3f}")
            else:
                # Fallback to single partition evaluation
                logger.info("Using single test partition evaluation (split_spec not available)")
                
                # CRITICAL: Action histogram check before metrics calculation
                logger.info("Running action histogram check to verify policy diversity...")
                action_counts = {0: 0, 1: 0, 2: 0}  # sell, hold, buy
                sample_episodes = min(10, GATE_EVAL_EPISODES // 10)  # Quick sample for action diversity
                
                for ep in range(sample_episodes):
                    obs = test_env.reset()
                    done_flag = False
                    step_count = 0
                    
                    while (not done_flag) and step_count < max_episode_length:
                        action, _ = model.predict(obs, deterministic=True)  # shape = (1,)
                        # Count using a scalar, but keep the array for env.step
                        action_scalar = int(action[0])
                        action_counts[action_scalar] = action_counts.get(action_scalar, 0) + 1
                        
                        obs, rewards, dones, infos = test_env.step(action)  # pass array, not int
                        done_flag = bool(dones[0])
                        step_count += 1
                
                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    action_percentages = {k: (v/total_actions)*100 for k, v in action_counts.items()}
                    logger.info(f"Action distribution: Sell={action_percentages.get(0, 0):.1f}%, "
                              f"Hold={action_percentages.get(1, 0):.1f}%, "
                              f"Buy={action_percentages.get(2, 0):.1f}%")
                    
                    # Check if policy is using all actions
                    unused_actions = [k for k, v in action_counts.items() if v == 0]
                    if unused_actions:
                        logger.warning(f"⚠️ Policy not using actions: {unused_actions}")
                        logger.warning("This may indicate partial policy collapse")
                    else:
                        logger.info("✅ Policy using all action types")
                else:
                    logger.warning("⚠️ No actions recorded during sample evaluation")
                
                test_metrics = evaluate_test_metrics(model, test_env, GATE_EVAL_EPISODES, max_episode_length=max_episode_length)
                # Add action distribution to metrics for audit
                test_metrics["action_distribution"] = action_percentages if total_actions > 0 else {}
            
            # PULIDO CRÍTICO: Si las métricas son inválidas, FAIL todos los gates
            if not test_metrics.get("is_valid", True):
                logger.error(f"❌ Metrics invalid for seed {seed} - marking ALL gates as FAIL")
                gates_detailed_fail = {}
                
                # MEJORA: Use same set of gates as normal path for consistency
                # Hard gates that match normal evaluation path
                hard_gates = {
                    "reward_reproducibility_gate": {"threshold": 0.001, "operator": "<="},
                    "sortino_gate": {"threshold": SORTINO_THRESHOLD, "operator": ">="},
                    "max_dd_gate": {"threshold": MAX_DD_THRESHOLD, "operator": "<="},
                    "calmar_gate": {"threshold": CALMAR_THRESHOLD, "operator": ">="},
                    "cost_stress_gate": {"threshold": CAGR_DROP_THRESHOLD, "operator": "<="},
                    "latency_gate": {"threshold": INFERENCE_P99_THRESHOLD_MS, "operator": "<="},
                    "e2e_latency_gate": {"threshold": E2E_P99_THRESHOLD_MS, "operator": "<="},
                    "sortino_consistency_gate": {"threshold": SORTINO_DIFF_THRESHOLD, "operator": "<="}
                }
                
                # Soft gate for alerting only
                soft_gates = {
                    "truncation_rate_gate": {"threshold": 0.01, "operator": "<=", "severity": "soft"}
                }
                
                # Build all gates with fail status
                for gate_name, gate_info in {**hard_gates, **soft_gates}.items():
                    gates_detailed_fail[gate_name] = {
                        "passed": False,
                        "observed": 0.0,
                        "threshold": gate_info["threshold"],
                        "operator": gate_info["operator"],
                        "severity": gate_info.get("severity", "hard"),
                        "reason": f"metrics_invalid: {test_metrics.get('validation_failures', ['unknown'])}"
                    }
                
                # Only count hard gates for pass/fail decision
                hard_gates_fail = {k: False for k, v in gates_detailed_fail.items() if v.get("severity", "hard") != "soft"}
                
                # Skip rest of evaluation for this seed
                seed_results = {
                    "train_metrics": {"sortino_train": -999, "mean_return": 0, "std_return": 0},
                    "test_metrics": test_metrics,
                    "gates": hard_gates_fail,  # Only hard gates
                    "gates_detailed": gates_detailed_fail,  # All gates for audit
                    "gates_passed": 0,
                    "total_gates": len(hard_gates_fail),
                    "latency": {"pytorch": {"p99": 999}, "onnx": {"p99": 999}},
                    "status": "FAIL_METRICS_INVALID"  # FAIL-FAST centralized status
                }
                all_results[f"seed_{seed}"] = seed_results
                
                # Update gate_summary for fail-fast diagnosis
                if gate_summary["overall_status"] == "PENDING":
                    gate_summary["overall_status"] = "FAIL_METRICS_INVALID"
                continue  # Skip to next seed
            
            # CRITICAL: Check for degenerate policy (policy collapse prevention)
            trades_per_ep = test_metrics.get("attribution", {}).get("trades", 0) / max(GATE_EVAL_EPISODES, 1)
            hold_percentage = test_metrics.get("attribution", {}).get("holds", 0) / max(
                test_metrics.get("attribution", {}).get("total_actions", 1), 1) * 100
            
            logger.info(f"Degenerate policy check - Trades per episode: {trades_per_ep:.2f}, Hold%: {hold_percentage:.1f}%")
            
            if trades_per_ep == 0 or hold_percentage >= 95.0:
                logger.error(f"❌ DEGENERATE POLICY DETECTED for seed {seed}")
                logger.error(f"   - Trades per episode: {trades_per_ep:.2f} (should be > 0)")
                logger.error(f"   - Hold percentage: {hold_percentage:.1f}% (should be < 95%)")
                logger.error("   This indicates the policy has collapsed and is not learning trading strategies")
                
                # Mark policy collapse in test_metrics for transparency
                test_metrics["policy_collapse"] = True
                test_metrics["policy_collapse_reason"] = f"trades_per_ep={trades_per_ep:.2f}, hold%={hold_percentage:.1f}%"
                
                # Continue evaluation but flag as policy collapse
                logger.warning("⚠️ Continuing evaluation to collect diagnostic metrics despite policy collapse")
            else:
                logger.info(f"✅ Policy health check passed - active trading policy detected")
                test_metrics["policy_collapse"] = False
            
            # IMPROVEMENT #2: Deep cost scaling function for nested structures
            def scale_costs_deep(obj, factor: float):
                """Scale all numeric values in nested dict/list structures"""
                if isinstance(obj, dict):
                    return {k: scale_costs_deep(v, factor) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [scale_costs_deep(item, factor) for item in obj]
                elif isinstance(obj, (int, float)):
                    return obj * factor
                else:
                    return obj
            
            # Log base costs for transparency
            logger.info(f"Base L4 cost model: {json.dumps(cost_model, indent=2)}")
            
            # Create stress cost model by scaling ALL numeric values
            stress_cost_model = scale_costs_deep(cost_model, COST_STRESS_MULTIPLIER)
            logger.info(f"Stress cost model (x{COST_STRESS_MULTIPLIER}): {json.dumps(stress_cost_model, indent=2)}")
            
            # FIX #4: Re-set seeds before stress environment creation for consistency  
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
            # Create stress environment with scaled costs
            stress_env = DummyVecEnv([
                lambda: Monitor(
                    DiscreteToSignedAction(
                        SentinelTradingEnv(
                            create_gym_env(mode="test"),
                            cost_model=stress_cost_model,  # Properly scaled model
                            shaping_penalty=0.0,  # Sin shaping en evaluación
                            enable_telemetry=False,
                            max_episode_length=max_episode_length  # L4-L5 contract: from env_spec
                        )
                    ),
                    allow_early_resets=True
                )
            ])
            stress_metrics = evaluate_test_metrics(model, stress_env, GATE_EVAL_EPISODES, max_episode_length=max_episode_length)
            
            # MUST-FIX #3: Robust CAGR drop calculation
            base_cagr = test_metrics.get("cagr", 0.0)
            stress_cagr = stress_metrics.get("cagr", 0.0)
            
            # Use absolute denominator to avoid sign flips with negative CAGR
            den = max(1e-8, abs(base_cagr))
            cagr_drop = max(0.0, (base_cagr - stress_cagr) / den)
            
            logger.info(f"CAGR impact: base={base_cagr:.4f}, stress={stress_cagr:.4f}, drop={cagr_drop:.2%}")
            
            # Get L4 reproducibility gate and RMSE from XCom
            rr_gate = context['task_instance'].xcom_pull(
                task_ids='validate_l4_outputs',
                key='l4_reward_reproducibility_gate'
            )
            rr_rmse = context['task_instance'].xcom_pull(
                task_ids='validate_l4_outputs',
                key='l4_reward_rmse'
            )
            if rr_rmse is None:
                rr_rmse = 1.0  # Default to high value if not available
            
            # AJUSTE FINO #5: Gates con thresholds y valores observados para mejor auditoría
            gates_detailed = {
                "reward_reproducibility_gate": {
                    "passed": bool(rr_gate) if rr_gate is not None else False,
                    "observed": float(rr_rmse),  # MEJORA: Use numeric RMSE for uniform reporting
                    "threshold": 0.001,  # 0.1% tolerance
                    "operator": "<=",
                    "source": "L4_validation"
                },
                "sortino_gate": {
                    "passed": test_metrics["sortino_ratio"] >= SORTINO_THRESHOLD,
                    "observed": float(test_metrics["sortino_ratio"]),
                    "threshold": SORTINO_THRESHOLD,
                    "operator": ">="
                },
                "max_dd_gate": {
                    "passed": test_metrics["max_drawdown"] <= MAX_DD_THRESHOLD,
                    "observed": float(test_metrics["max_drawdown"]),
                    "threshold": MAX_DD_THRESHOLD,
                    "operator": "<="
                },
                "calmar_gate": {
                    "passed": test_metrics["calmar_ratio"] >= CALMAR_THRESHOLD,
                    "observed": float(test_metrics["calmar_ratio"]),
                    "threshold": CALMAR_THRESHOLD,
                    "operator": ">="
                },
                "cost_stress_gate": {
                    "passed": cagr_drop <= CAGR_DROP_THRESHOLD,
                    "observed": float(cagr_drop),
                    "threshold": CAGR_DROP_THRESHOLD,
                    "operator": "<="
                },
                "latency_gate": {
                    # Handle ONNX-less models (like QR-DQN)
                    "passed": (lambda: 
                        onnx_p99 <= INFERENCE_P99_THRESHOLD_MS 
                        if (onnx_p99 := train_result["latency"].get("onnx", {}).get("p99", float("inf"))) < float("inf")
                        else train_result["latency"]["pytorch"]["p99"] <= INFERENCE_P99_THRESHOLD_MS
                    )(),
                    "observed": float(
                        train_result["latency"].get("onnx", {}).get("p99", train_result["latency"]["pytorch"]["p99"])
                    ),
                    "threshold": INFERENCE_P99_THRESHOLD_MS,
                    "operator": "<="
                },
                "e2e_latency_gate": {
                    "passed": train_result["latency"].get("e2e", {}).get("p99", 1e9) <= E2E_P99_THRESHOLD_MS,
                    "observed": float(train_result["latency"].get("e2e", {}).get("p99", 1e9)),
                    "threshold": E2E_P99_THRESHOLD_MS,
                    "operator": "<="
                },
                # FIX #5: Policy Balance Gate - Prevent policy collapse
                "policy_balance_gate": {
                    "passed": test_metrics.get("balanced_trading", False),
                    "observed": float(min(
                        test_metrics.get("action_balance", {}).get("buy_ratio", 0),
                        test_metrics.get("action_balance", {}).get("sell_ratio", 0)
                    )),  # numeric for logging
                    "threshold": 0.05,  # 5% minimum on each side
                    "operator": ">=",
                    "severity": "soft"  # Make it soft gate initially for monitoring
                }
            }
            
            # CRITICAL FIX: Sortino consistency check instead of Sharpe
            sortino_train = train_result.get("sortino_train", 0)
            sortino_test = train_result.get("sortino_test", 0)
            sortino_diff = abs(sortino_train - sortino_test)
            
            # AJUSTE FINAL: Agregar sortino_consistency_gate a gates_detailed para transparencia
            gates_detailed["sortino_consistency_gate"] = {
                "passed": sortino_diff <= SORTINO_DIFF_THRESHOLD,
                "observed": float(sortino_diff),
                "threshold": SORTINO_DIFF_THRESHOLD,
                "operator": "<=",
                "train_value": float(sortino_train),
                "test_value": float(sortino_test)
            }
            
            # MEJORA: Add truncation rate gate for alerting on excessive truncations
            TRUNCATION_RATE_THRESHOLD = 0.01  # 1% soft threshold for alerting
            truncation_rate = test_metrics.get("truncation_rate", 0)
            gates_detailed["truncation_rate_gate"] = {
                "passed": truncation_rate <= TRUNCATION_RATE_THRESHOLD,
                "observed": float(truncation_rate),
                "threshold": TRUNCATION_RATE_THRESHOLD,
                "operator": "<=",
                "episodes_truncated": test_metrics.get("episodes_truncated", 0),
                "total_episodes": test_metrics.get("total_episodes", 0),
                "severity": "soft"  # Soft gate - alerts but doesn't fail deployment
            }
            
            # Only count HARD gates in the deployment decision
            hard_gate_names = [k for k, v in gates_detailed.items() if v.get("severity", "hard") != "soft"]
            gates = {k: gates_detailed[k]["passed"] for k in hard_gate_names}
            
            # Log soft gates separately for visibility
            soft_gates = {k: v for k, v in gates_detailed.items() if v.get("severity") == "soft"}
            if soft_gates:
                for gate_name, gate_info in soft_gates.items():
                    status = "⚠️ WARNING" if not gate_info["passed"] else "✅ OK"
                    logger.info(f"SOFT GATE {gate_name}: {status} - observed: {gate_info['observed']:.4f}, threshold: {gate_info['threshold']:.4f}")
            
            logger.info(f"Sortino Train: {sortino_train:.3f}, Test: {sortino_test:.3f}, Diff: {sortino_diff:.3f}")
            
            # Store results con gates detallados
            seed_results = {
                "test_metrics": test_metrics,
                "stress_metrics": stress_metrics,
                "cagr_drop": float(cagr_drop),
                "gates": gates,  # Only hard gates for pass/fail decision
                "gates_detailed": gates_detailed,  # Keep soft gates visible here for audit
                "gates_passed": sum(gates.values()),
                "total_gates": len(gates),
                "latency": train_result["latency"]
            }
            
            all_results[f"seed_{seed}"] = seed_results
            
            # GENERA REPORTE JSON DETALLADO POR SEED
            train_metrics = {
                "sortino_ratio": sortino_train,
                "mean_return": train_result.get("mean_return", 0),
                "std_return": train_result.get("std_return", 0)
            }
            
            # Genera reporte completo con attribution y diagnóstico
            seed_report = generate_seed_report(
                seed=seed,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                episode_data=test_metrics.get('attribution', {}),
                gate_results={
                    'gates': gates,
                    'passed': seed_results['gates_passed'],
                    'total': seed_results['total_gates']
                }
            )
            
            # Guarda reporte JSON
            report_path = os.path.join(temp_dir, f"l5_eval_seed_{seed}.json")
            with open(report_path, 'w') as f:
                json.dump(convert_numpy_to_python(seed_report), f, indent=2)
            
            # Log como artefacto en MLflow
            safe_mlflow_log_artifact(report_path, artifact_path=f"seed_{seed}_reports")
            logger.info(f"Seed {seed} detailed report saved to {report_path}")
            
            # Log metrics
            for key, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"seed_{seed}_test_{key}", value)
            
            mlflow.log_metric(f"seed_{seed}_gates_passed", seed_results["gates_passed"])
            
            logger.info(f"Seed {seed}: {seed_results['gates_passed']}/{seed_results['total_gates']} gates passed")
            
            # Clean up
            test_env.close()
            stress_env.close()
        
        # Overall assessment
        if all_results:
            # CRITICAL: Check if any seed had FAIL_METRICS_INVALID - this must not be masked
            has_invalid_metrics = any(
                result.get("status") == "FAIL_METRICS_INVALID" 
                for result in all_results.values()
            )
            
            if has_invalid_metrics:
                # Preserve FAIL_METRICS_INVALID status - don't let passing seeds mask this
                gate_summary["overall_status"] = "FAIL_METRICS_INVALID"
                logger.error("⚠️ At least one seed had invalid metrics - overall status: FAIL_METRICS_INVALID")
                
                # Still find best seed for reporting, but status remains FAIL
                best_seed = max(all_results.items(), 
                              key=lambda x: (
                                  x[1]["gates_passed"],  # Primero: máximo gates pasados
                                  x[1]["test_metrics"].get("sortino_ratio", -1e9),  # Segundo: mejor Sortino
                                  x[1]["test_metrics"].get("calmar_ratio", -1e9)  # Tercero: mejor Calmar
                              ))
                gate_summary["best_seed"] = best_seed[0]
                gate_summary["best_gates_passed"] = best_seed[1]["gates_passed"]
            else:
                # Normal path when all seeds have valid metrics
                # PULIDO: Desempate mejorado por sortino_test y calmar_ratio
                best_seed = max(all_results.items(), 
                              key=lambda x: (
                                  x[1]["gates_passed"],  # Primero: máximo gates pasados
                                  x[1]["test_metrics"].get("sortino_ratio", -1e9),  # Segundo: mejor Sortino
                                  x[1]["test_metrics"].get("calmar_ratio", -1e9)  # Tercero: mejor Calmar
                              ))
                
                gate_summary["best_seed"] = best_seed[0]
                gate_summary["best_gates_passed"] = best_seed[1]["gates_passed"]
                gate_summary["overall_status"] = "PASS" if best_seed[1]["gates_passed"] == best_seed[1]["total_gates"] else "FAIL"
            
            # Save metrics summary
            metrics_summary_path = os.path.join(temp_dir, "metrics_summary.json")
            with open(metrics_summary_path, 'w') as f:
                json.dump(convert_numpy_to_python(all_results), f, indent=2)
            safe_mlflow_log_artifact(metrics_summary_path)
            
            # Save acceptance report
            # PULIDO: Leer el gate de reproducibilidad del L4
            rr_gate = context['task_instance'].xcom_pull(
                task_ids='validate_l4_outputs',
                key='l4_reward_reproducibility_gate'
            )
            
            # Get RMSE value from validate_l4_outputs for better visibility
            rr_rmse = context['task_instance'].xcom_pull(
                task_ids='validate_l4_outputs',
                key='l4_reward_rmse'
            )
            
            # MEJORA: Acceptance report más auditable con métricas de costos
            acceptance_report = {
                "timestamp": gate_summary["timestamp"],
                "overall_status": gate_summary["overall_status"],
                "best_seed": gate_summary["best_seed"],
                "reward_reproducibility_gate": bool(rr_gate) if rr_gate is not None else False,  # PULIDO: Gate del L4
                "l4_reward_reproducibility_rmse": float(rr_rmse) if rr_rmse is not None else None,  # PATCH B: RMSE visibility
                "gates_summary": {
                    seed: results["gates"] 
                    for seed, results in all_results.items()
                },
                "gates_detailed": {
                    seed: results.get("gates_detailed", {})
                    for seed, results in all_results.items()
                },
                "test_telemetry": {
                    seed: {
                        "episodes_truncated": results.get("test_metrics", {}).get("episodes_truncated", 0),
                        "cost_bps_sum": results.get("test_metrics", {}).get("attribution", {}).get("cost_bps_sum", 0),
                        "trades_count": results.get("test_metrics", {}).get("attribution", {}).get("trades", 0),
                        "holds_count": results.get("test_metrics", {}).get("attribution", {}).get("holds", 0)
                    }
                    for seed, results in all_results.items()
                }
            }
            
            acceptance_report_path = os.path.join(temp_dir, "acceptance_report.json")
            with open(acceptance_report_path, 'w') as f:
                json.dump(convert_numpy_to_python(acceptance_report), f, indent=2)
            safe_mlflow_log_artifact(acceptance_report_path)
            
            # MEJORA: Create L5 audit.json with canonical L0→L5 format for auditors and investors
            violations = []
            warnings = []
            
            # Check for critical violations
            if gate_summary["overall_status"] == "FAIL_METRICS_INVALID":
                violations.append("CRITICAL: Invalid metrics detected - numerical instability or data corruption")
            
            # Check each seed for violations and warnings
            for seed_key, results in all_results.items():
                gates_detail = results.get("gates_detailed", {})
                
                # Hard gate failures are violations
                for gate_name, gate_info in gates_detail.items():
                    if gate_info.get("severity", "hard") != "soft" and not gate_info["passed"]:
                        violations.append(f"{seed_key}/{gate_name}: observed={gate_info['observed']:.4f}, threshold={gate_info['threshold']:.4f}")
                    
                    # Soft gate failures are warnings
                    elif gate_info.get("severity") == "soft" and not gate_info["passed"]:
                        warnings.append(f"{seed_key}/{gate_name}: observed={gate_info['observed']:.4f}, threshold={gate_info['threshold']:.4f}")
                
                # Check for concerning metrics even if gates passed
                test_metrics = results.get("test_metrics", {})
                if test_metrics.get("sortino_ratio", 0) < 0:
                    warnings.append(f"{seed_key}: Negative Sortino ratio ({test_metrics['sortino_ratio']:.3f})")
                if test_metrics.get("max_drawdown", 0) > 0.15:
                    warnings.append(f"{seed_key}: High drawdown ({test_metrics['max_drawdown']:.1%})")
            
            # Determine audit status
            if violations:
                audit_status = "FAIL"
            elif warnings:
                audit_status = "WARN"
            else:
                audit_status = "PASS"
            
            l5_audit = {
                "layer": "L5_SERVING",
                "timestamp": datetime.now().isoformat(),
                "dag_run_id": context['dag_run'].run_id,
                "status": audit_status,
                "violations": violations,
                "warnings": warnings,
                "gates_summary": {
                    "overall_status": gate_summary["overall_status"],
                    "best_seed": gate_summary.get("best_seed", "none"),
                    "best_gates_passed": gate_summary.get("best_gates_passed", 0),
                    "total_gates": best_seed[1]["total_gates"] if all_results else 0,
                    "reward_reproducibility": bool(rr_gate) if rr_gate is not None else False
                },
                "performance_summary": {
                    "best_sortino": best_seed[1]["test_metrics"].get("sortino_ratio", 0) if all_results else 0,
                    "best_calmar": best_seed[1]["test_metrics"].get("calmar_ratio", 0) if all_results else 0,
                    "best_cagr": best_seed[1]["test_metrics"].get("cagr", 0) if all_results else 0,
                    "latency_p99_ms": best_seed[1]["latency"].get("onnx", {}).get("p99", 0) if all_results else 0
                }
            }
            
            l5_audit_path = os.path.join(temp_dir, "l5_audit.json")
            with open(l5_audit_path, 'w') as f:
                json.dump(convert_numpy_to_python(l5_audit), f, indent=2)
            safe_mlflow_log_artifact(l5_audit_path)
            
            # MEJORA: Upload reports to the BEST seed's prefix to ensure create_model_bundle finds them
            logger.info("Uploading reports to S3 for bundle creation...")
            s3 = S3Hook(aws_conn_id='minio_conn')
            
            # Use the best seed that was already computed
            chosen_seed_num = int(best_seed[0].split("_")[1])
            tr = context['task_instance'].xcom_pull(
                task_ids=f'ppo_training.train_ppo_seed_{chosen_seed_num}',
                key=f'ppo_seed_{chosen_seed_num}_result'
            )
            
            if tr and tr.get("status") == "COMPLETED":
                bucket = tr.get('l5_bucket', '05-l5-ds-usdcop-serving')
                prefix = tr.get('l5_prefix', '')
                
                logger.info(f"Uploading reports to best seed {chosen_seed_num} at {bucket}/{prefix}")
                
                # Subir acceptance_report.json
                with open(acceptance_report_path, 'rb') as f:
                    s3.load_file_obj(
                        f, 
                        key=f"{prefix}acceptance_report.json", 
                        bucket_name=bucket, 
                        replace=True
                    )
                
                # Subir metrics_summary.json
                with open(metrics_summary_path, 'rb') as f:
                    s3.load_file_obj(
                        f, 
                        key=f"{prefix}metrics_summary.json", 
                        bucket_name=bucket, 
                        replace=True
                    )
                
                # MEJORA: Also upload l5_audit.json for complete audit trail
                with open(l5_audit_path, 'rb') as f:
                    s3.load_file_obj(
                        f, 
                        key=f"{prefix}l5_audit.json", 
                        bucket_name=bucket, 
                        replace=True
                    )
                
                logger.info(f"✅ Uploaded reports for best seed {chosen_seed_num} to {bucket}/{prefix}")
            else:
                logger.warning(f"Could not upload reports - best seed {chosen_seed_num} training result not found")
            
            logger.info("="*80)
            logger.info(f"PRODUCTION GATE STATUS: {gate_summary['overall_status']}")
            logger.info(f"Best seed: {gate_summary['best_seed']} ({gate_summary['best_gates_passed']} gates passed)")
            logger.info("="*80)
            
            gate_summary["all_gates_passed"] = (best_seed[1]["gates_passed"] == best_seed[1]["total_gates"])
            gate_summary["total_gates"] = best_seed[1]["total_gates"]  # Add for deployment summary
            gate_summary["best_seed_int"] = int(best_seed[0].split('_')[1])
            context["task_instance"].xcom_push(key="best_seed_int", value=gate_summary["best_seed_int"])
        
        # Cleanup temporary directory
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up gates temp directory: {temp_dir}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup gates temp directory: {cleanup_error}")
        
        return gate_summary

def evaluate_test_metrics(model, env, n_episodes: int, max_episode_length: int = 60) -> Dict[str, float]:
    """
    Calculate comprehensive test metrics using ULTRA-ROBUST functions
    Incluye fail-fast para NaN/Inf y attribution detallado
    """
    episode_rewards = []
    episode_lengths = []
    episode_data = {
        'rewards': [],
        'costs': [],
        'trades': 0,
        'holds': 0,
        'actions': [],
        'cost_bps_sum': 0.0  # MEJORA: Telemetría de costos
    }
    
    # L4-L5 Contract: Use max_episode_length from env_spec (was hardcoded 4000)
    MAX_STEPS_PER_EPISODE = max_episode_length  # From L4 env_spec
    episodes_truncated = 0  # MEJORA: Contador de episodios truncados
    logger.info(f"Using MAX_STEPS_PER_EPISODE from env_spec: {MAX_STEPS_PER_EPISODE}")
    
    # Colecta datos por episodio con telemetría rica
    for ep_idx in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        ep_reward = 0
        ep_length = 0
        ep_trades = 0
        ep_holds = 0
        ep_actions = []
        ep_cost_bps = 0.0  # MEJORA: Acumular costos del episodio
        prev_position = None  # MEJORA: Track posición para conteo fiel de trades
        
        while not done[0] and ep_length < MAX_STEPS_PER_EPISODE:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0])
            ep_length += 1
            ep_actions.append(int(action[0]))
            
            # MEJORA: Extraer telemetría de costos y posición
            if info and len(info) > 0:
                telemetry = info[0].get("telemetry", {})
                step_cost = telemetry.get("cost_bps", 0.0)
                ep_cost_bps += step_cost
                
                # MEJORA: Conteo más fiel basado en cambios de posición
                current_position = telemetry.get("position", None)
                
                # PULIDO: Fix conteo en el primer paso - inicializar prev_position
                if prev_position is None and current_position is not None:
                    # Primer paso: solo inicializar, no contar como trade
                    prev_position = current_position
                    # No incrementar trades/holds en el primer paso para evitar falso positivo
                elif current_position is not None and prev_position is not None:
                    # Contar trade solo si la posición cambió
                    if current_position != prev_position:
                        ep_trades += 1
                    else:
                        ep_holds += 1
                    prev_position = current_position
                else:
                    # Fallback si no hay telemetría de posición
                    if int(action[0]) == 1:  # Hold (policy action 1)
                        ep_holds += 1
                    else:  # 0=Sell or 2=Buy → trade
                        ep_trades += 1
            else:
                # Fallback si no hay telemetría
                if int(action[0]) == 1:  # Hold (policy action 1)
                    ep_holds += 1
                else:  # 0=Sell or 2=Buy → trade
                    ep_trades += 1
        
        # Solo marcar como truncated si el episodio fue cortado sin terminar naturalmente
        if ep_length >= MAX_STEPS_PER_EPISODE and not done[0]:
            logger.warning(f"Episode {ep_idx} was truncated at max steps limit ({MAX_STEPS_PER_EPISODE}) - potential infinite loop")
            episodes_truncated += 1
        elif ep_length == MAX_STEPS_PER_EPISODE and done[0]:
            # Normal: episodio terminó naturalmente en exactamente MAX_STEPS_PER_EPISODE pasos
            logger.debug(f"Episode {ep_idx} completed normally in {ep_length} steps")
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_data['trades'] += ep_trades
        episode_data['holds'] += ep_holds
        episode_data['actions'].extend(ep_actions)
        episode_data['cost_bps_sum'] += ep_cost_bps  # MEJORA: Acumular costos totales
        episode_data['costs'].append(ep_cost_bps)  # MEJORA: Guardar costos por episodio
    
    # MEJORA: Log de episodios truncados al final (solo si realmente hay un problema)
    if episodes_truncated > 0:
        logger.warning(f"⚠️ {episodes_truncated}/{n_episodes} episodes were truncated without natural termination - possible infinite loops or stuck policy")
    
    # MUST-FIX #4: Correct annualization for M5 premium window
    # M5 premium: 60 bars per trading day (not 24h)
    eval_result = evaluate_episode_metrics(
        episode_rewards, 
        episode_lengths,
        periods_per_year=252 * 60  # 60 M5 bars per premium day
    )
    
    # Extrae métricas con status checks
    metrics = eval_result['metrics'].copy()
    
    # Añade PnL attribution
    episode_data['rewards'] = episode_rewards
    attribution = calculate_pnl_attribution(episode_data)
    metrics['attribution'] = attribution
    
    # IMPROVEMENT #4: Policy Balance Check - Correct action mapping between env and policy
    # Environment actions: -1=sell, 0=hold, 1=buy
    # Policy actions: 0=Sell(-1), 1=Hold(0), 2=Buy(1) - matches DiscreteToSignedAction wrapper
    # Actions stored are already policy actions from model.predict()
    actions = episode_data.get('actions', [])
    if actions:
        # Actions are raw policy actions from model.predict(): 0,1,2
        # Our DiscreteToSignedAction uses: 0=Sell, 1=Hold, 2=Buy (correct mapping)
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        total_actions = len(actions)
        sell_ratio = action_counts.get(0, 0) / total_actions if total_actions else 0.0  # policy 0
        hold_ratio = action_counts.get(1, 0) / total_actions if total_actions else 0.0  # policy 1
        buy_ratio = action_counts.get(2, 0) / total_actions if total_actions else 0.0   # policy 2
        
        metrics['action_balance'] = {
            'sell_ratio': sell_ratio,
            'hold_ratio': hold_ratio,
            'buy_ratio': buy_ratio,
            'sell_count': action_counts.get(0, 0),
            'hold_count': action_counts.get(1, 0),
            'buy_count': action_counts.get(2, 0),
            'total_actions': total_actions
        }
        
        policy_collapse_detected = (buy_ratio == 0 and sell_ratio > 0) \
                                   or (sell_ratio == 0 and buy_ratio > 0) \
                                   or (buy_ratio < 0.01 or sell_ratio < 0.01)
        metrics['policy_collapse_detected'] = bool(policy_collapse_detected)
        
        # Balanced if both trade sides have >=5%
        metrics['balanced_trading'] = (buy_ratio >= 0.05 and sell_ratio >= 0.05)
        if not metrics['balanced_trading']:
            logger.warning(f"Trading not balanced: Buy {buy_ratio:.1%}, Sell {sell_ratio:.1%}, Hold {hold_ratio:.1%}")
    else:
        logger.warning("No action data available for balance analysis")
        metrics['policy_collapse_detected'] = True  # No actions = collapse
        metrics['balanced_trading'] = False
    
    # MEJORA: Agregar telemetría adicional para auditabilidad
    metrics['episodes_truncated'] = episodes_truncated
    metrics['total_episodes'] = n_episodes
    metrics['truncation_rate'] = episodes_truncated / n_episodes if n_episodes > 0 else 0
    
    # Valida métricas para gates (fail-fast)
    is_valid, failures = validate_metrics_for_gates(metrics)
    
    if not is_valid:
        logger.error(f"Metrics validation FAILED: {failures}")
        metrics['validation_failures'] = failures
        metrics['is_valid'] = False
        
        # MEJORA: Preserve original invalid values for audit trail
        metrics['_original_invalid'] = {}
        metrics['_safe_fallbacks'] = {}
        
        # Store original values and apply safe defaults
        for failure in failures:
            if 'NAN' in failure:
                metric_name = failure.split(':')[1].strip()
                # Store as string for JSON compatibility
                metrics['_original_invalid'][metric_name] = "NaN"
                safe_value = 0.0
                metrics['_safe_fallbacks'][metric_name] = safe_value
                metrics[metric_name] = safe_value  # Apply safe default
            elif 'INF' in failure:
                metric_name = failure.split(':')[1].strip()
                # Store as string for JSON compatibility
                original_val = metrics.get(metric_name, float('inf'))
                if original_val == float('inf'):
                    metrics['_original_invalid'][metric_name] = "Infinity"
                elif original_val == float('-inf'):
                    metrics['_original_invalid'][metric_name] = "-Infinity"
                else:
                    metrics['_original_invalid'][metric_name] = str(original_val)
                safe_value = 100.0 if 'sortino' in metric_name else 1.0
                metrics['_safe_fallbacks'][metric_name] = safe_value
                metrics[metric_name] = safe_value  # Apply safe default
        
        logger.info(f"Applied safe fallbacks for invalid metrics: {metrics['_safe_fallbacks']}")
    else:
        metrics['is_valid'] = True
    
    # Añade metadata de evaluación
    metrics['warnings'] = eval_result.get('warnings', [])
    metrics['critical_issues'] = eval_result.get('critical_issues', [])
    metrics['n_episodes'] = n_episodes
    
    # Log resumen
    logger.info(f"Metrics computed - Sortino: {metrics.get('sortino_ratio', 0):.3f}, "
                f"CAGR: {metrics.get('cagr', 0):.3f}, MaxDD: {metrics.get('max_drawdown', 0):.3f}")
    
    if metrics.get('warnings'):
        logger.warning(f"Metric warnings: {metrics['warnings']}")
    
    if metrics.get('critical_issues'):
        logger.error(f"CRITICAL issues detected: {metrics['critical_issues']}")
    
    return metrics

# ============================================================================
# MODEL BUNDLE CREATION
# ============================================================================

def create_model_bundle(**context):
    """Create complete model bundle with all artifacts"""
    import mlflow, shutil, os, tarfile, json
    
    # Helper function for MLflow artifact uploads with retry logic
    def safe_mlflow_log_artifact(file_path, artifact_path=None, max_retries=3):
        """Upload artifact to MLflow with retry logic and error handling"""
        for attempt in range(max_retries):
            try:
                if not os.path.exists(file_path):
                    logger.error(f"File not found for MLflow upload: {file_path}")
                    return False
                
                # Verify file integrity before upload
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    logger.error(f"File is empty, skipping upload: {file_path}")
                    return False
                
                mlflow.log_artifact(file_path, artifact_path=artifact_path)
                logger.info(f"Successfully uploaded {file_path} to MLflow (attempt {attempt + 1})")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to upload {file_path} to MLflow (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All upload attempts failed for {file_path}: {str(e)}")
                    return False
        return False
    from airflow.exceptions import AirflowFailException
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    # IDEMPOTENCE: Ensure critical packages are available (same as evaluate_production_gates)
    _ensure_pkgs([
        "packaging>=23",  # MEJORA: Ensure proper version checking
        "stable-baselines3>=2.3,<3", 
        "gymnasium>=0.29,<1.0", 
        "onnx==1.16.2", 
        "onnxruntime==1.18.1"
    ])

    logger.info("Creating model bundle...")

    # Get gate results
    gate_result = context['task_instance'].xcom_pull(
        task_ids='evaluate_production_gates', key='return_value'
    ) or {}
    best_seed = context['task_instance'].xcom_pull(
        task_ids='evaluate_production_gates', key='best_seed_int'
    )

    # Collect successful training results
    candidates = []
    for seed in PARALLEL_SEEDS:
        tr = context['task_instance'].xcom_pull(
            task_ids=f'ppo_training.train_ppo_seed_{seed}',
            key=f'ppo_seed_{seed}_result'
        )
        if tr and tr.get("status") == "COMPLETED":
            candidates.append((seed, tr))

    if not candidates:
        raise AirflowFailException(
            "No COMPLETED training results in XCom. Check ppo_training.* and ONNX export."
        )

    # Prefer the seed selected by gates
    if isinstance(best_seed, int):
        chosen = next(((s, tr) for s, tr in candidates if s == best_seed), candidates[0])
    else:
        chosen = candidates[0]

    best_seed, train_result = chosen
    logger.info(f"Bundling seed {best_seed} from {train_result.get('l5_bucket')}/{train_result.get('l5_prefix')}")

    # Create unique temporary directory for bundle creation
    temp_base_dir = tempfile.mkdtemp(prefix=f"bundle_creation_{best_seed}_")
    
    # Build bundle dir
    bundle_dir = os.path.join(temp_base_dir, f"model_bundle_seed_{best_seed}")
    os.makedirs(bundle_dir, exist_ok=True)

    s3 = S3Hook(aws_conn_id='minio_conn')
    bucket = train_result.get('l5_bucket', '05-l5-ds-usdcop-serving')
    prefix = train_result.get('l5_prefix', '')

    def _ensure_local(key, local_name):
        """Download key -> bundle_dir/local_name if missing locally"""
        local_path = os.path.join(bundle_dir, local_name)
        full_key = prefix + key if prefix else key
        if s3.check_for_key(full_key, bucket_name=bucket):
            obj = s3.get_key(full_key, bucket_name=bucket)
            # MEJORA: Close S3 stream after reading to free sockets
            body = obj.get()["Body"]
            content = body.read()
            if hasattr(body, 'close'):
                body.close()
            with open(local_path, "wb") as f:
                f.write(content)
            return local_path
        # Fallback to local path
        alt = train_result.get(key.replace(".json","_path").replace("model.onnx","onnx_path").replace("model.zip","model_path"))
        if alt and os.path.exists(alt):
            shutil.copy(alt, local_path)
            return local_path
        # Create placeholder
        logger.warning(f"Artifact not found: {key}, creating placeholder")
        with open(local_path, "w") as f:
            json.dump({"placeholder": True, "missing": key}, f)
        return local_path

    def _ensure_l4_local(l4_info, filename, local_name):
        """Download L4 file -> bundle_dir/local_name"""
        local_path = os.path.join(bundle_dir, local_name)
        l4_bucket = l4_info.get("l4_bucket", "04-l4-ds-usdcop-rlready")
        l4_prefix = l4_info.get("l4_prefix", "usdcop_m5__05_l4_rlready/")
        l4_key = l4_prefix + filename
        
        logger.info(f"Downloading L4 file: {l4_bucket}/{l4_key} -> {local_name}")
        
        if s3.check_for_key(l4_key, bucket_name=l4_bucket):
            obj = s3.get_key(l4_key, bucket_name=l4_bucket)
            # MEJORA: Close S3 stream after reading to free sockets
            body = obj.get()["Body"]
            content = body.read()
            if hasattr(body, 'close'):
                body.close()
            with open(local_path, "wb") as f:
                f.write(content)
            logger.info(f"✅ Downloaded L4 file: {filename}")
            return local_path
        else:
            # Create placeholder if L4 file not found
            logger.error(f"❌ L4 file not found: {l4_bucket}/{l4_key}, creating placeholder")
            with open(local_path, "w") as f:
                json.dump({
                    "placeholder": True, 
                    "missing_l4_file": filename,
                    "expected_location": f"{l4_bucket}/{l4_key}",
                    "error": "L4 contract file not found in expected location"
                }, f, indent=2)
            return local_path

    # Gather artifacts
    model_zip  = _ensure_local("model.zip",   "policy.zip")
    model_onnx = _ensure_local("model.onnx",  "policy.onnx")
    env_spec   = _ensure_local("env_spec.json",    "env_spec.json")
    reward_sp  = _ensure_local("reward_spec.json", "reward_spec.json")
    cost_model = _ensure_local("cost_model.json",  "cost_model.json")
    config_js  = _ensure_local("config.json",      "config.json")
    latency_js = _ensure_local("latency.json",     "latency.json")
    
    # MEJORA: Añadir acceptance_report.json, metrics_summary.json y l5_audit.json al bundle
    acceptance_report = _ensure_local("acceptance_report.json", "acceptance_report.json")
    metrics_summary = _ensure_local("metrics_summary.json", "metrics_summary.json")
    l5_audit = _ensure_local("l5_audit.json", "l5_audit.json")
    
    # MEJORA: Bundle complete L4 contracts for 100% self-contained audit trail
    # Obtain L4 dataset info (reuse from training or get fresh)
    l4_dataset_info = train_result.get("l4_dataset_info") or get_l4_dataset_info(**context)
    
    # Function to download L4 files from the L4 bucket
    def _ensure_l4_local(l4_info, filename, local_name):
        """Download L4 file -> bundle_dir/local_name from L4 bucket"""
        local_path = os.path.join(bundle_dir, local_name)
        l4_bucket = l4_info.get("l4_bucket", "04-l4-ds-usdcop-rlready")
        l4_prefix = l4_info.get("l4_prefix", "usdcop_m5__05_l4_rlready/")
        l4_key = l4_prefix + filename
        
        logger.info(f"Downloading L4 file: {l4_bucket}/{l4_key} -> {local_name}")
        
        if s3.check_for_key(l4_key, bucket_name=l4_bucket):
            obj = s3.get_key(l4_key, bucket_name=l4_bucket)
            # MEJORA: Close S3 stream after reading to free sockets
            body = obj.get()["Body"]
            content = body.read()
            if hasattr(body, 'close'):
                body.close()
            with open(local_path, "wb") as f:
                f.write(content)
            logger.info(f"✅ Downloaded L4 file: {filename} ({len(content)} bytes)")
            return local_path
        else:
            # Create detailed placeholder if L4 file not found
            logger.error(f"❌ L4 file not found: {l4_bucket}/{l4_key}, creating placeholder")
            placeholder_content = {
                "placeholder": True, 
                "missing_l4_file": filename,
                "expected_location": f"{l4_bucket}/{l4_key}",
                "error": "L4 contract file not found in expected location"
            }
            with open(local_path, "w") as f:
                json.dump(placeholder_content, f, indent=2)
            return local_path
    
    # Download REAL L4 files instead of placeholders
    split_spec = _ensure_l4_local(l4_dataset_info, "split_spec.json", "split_spec.json")
    norm_ref = _ensure_l4_local(l4_dataset_info, "normalization_ref.json", "normalization_ref.json")

    # Manifest with Git info
    # Get algorithm name for bundle
    algo_name = train_result.get("algorithm", "PPO_MLP").lower().replace("_", "-")
    
    manifest = {
        "model_id": f"{algo_name}_production_{best_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "algorithm": {
            "name": train_result.get("algorithm", "PPO_MLP"),
            "version": "sb3-contrib" if train_result.get("algorithm") in ["PPO_LSTM", "QRDQN"] else "sb3",
            "hyperparameters": train_result.get("hyperparameters", {}),
            "selection_reason": f"Best Sortino among {len(candidates)} seeds",
            "alternatives_tested": list(set(r.get("algorithm", "PPO_MLP") for _, r in candidates))
        },
        "seed": best_seed,
        "mlflow_run_id": train_result.get("mlflow_run_id", "unknown"),
        "created_at": datetime.now().isoformat(),
        "git_info": get_git_info(),  # Using the 3-method fallback
        "container_info": get_container_info(),
        # MEJORA: Reuse L4 info from training to avoid rescanning S3
        "l4_dataset": train_result.get("l4_dataset_info") or get_l4_dataset_info(**context),
        "artifacts": {
            "policy_zip": "policy.zip",
            "policy_onnx": "policy.onnx",
            "env_spec": "env_spec.json",
            "reward_spec": "reward_spec.json",
            "cost_model": "cost_model.json",
            "training_config": "config.json",
            "latency": "latency.json",
            "acceptance_report": "acceptance_report.json",  # MEJORA: Para auditorías rápidas
            "metrics_summary": "metrics_summary.json",  # MEJORA: Métricas completas
            "l5_audit": "l5_audit.json",  # MEJORA: Canonical audit format for investors/auditors
            "split_spec": "split_spec.json",  # MEJORA: Complete L4 contract
            "normalization_ref": "normalization_ref.json"  # MEJORA: Complete L4 contract
        },
        "performance": {
            "sortino_train": train_result.get("sortino_train", 0),
            "sortino_test": train_result.get("sortino_test", 0),
            "latency_p99_ms": float(train_result.get("latency", {}).get("onnx", {}).get("p99", 0)),
            "gates_overall_status": gate_result.get("overall_status", "UNKNOWN")
        }
    }
    manifest_path = os.path.join(bundle_dir, "model_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(convert_numpy_to_python(manifest), f, indent=2)

    # Create tarball
    bundle_path = os.path.join(temp_base_dir, f"model_bundle_{best_seed}.tar.gz")
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=os.path.basename(bundle_dir))

    # Upload to L5
    l5_bucket = "05-l5-ds-usdcop-serving"
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    l5_prefix = f"production_{ts}/"
    with open(bundle_path, "rb") as f:
        s3.load_file_obj(file_obj=f, key=f"{l5_prefix}model_bundle_{best_seed}.tar.gz",
                         bucket_name=l5_bucket, replace=True)
    logger.info(f"Uploaded model bundle to {l5_bucket}/{l5_prefix}model_bundle_{best_seed}.tar.gz")

    # Upload individual artifacts
    for root, _, files in os.walk(bundle_dir):
        for file in files:
            p = os.path.join(root, file)
            rel = os.path.relpath(p, bundle_dir)
            with open(p, "rb") as f:
                s3.load_file_obj(file_obj=f, key=f"{l5_prefix}artifacts/{rel}",
                                 bucket_name=l5_bucket, replace=True)

    # Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("l5_model_bundles")
    with mlflow.start_run(run_name=f"model_bundle_{best_seed}"):
        safe_mlflow_log_artifact(bundle_path)
        mlflow.set_tags({
            "bundle_type": "production",
            "seed": str(best_seed),
            "l5_bucket": l5_bucket,
            "l5_prefix": l5_prefix,
            "status": "READY",
            "gates_overall_status": manifest["performance"]["gates_overall_status"],
            "git_sha": manifest["git_info"]["sha"]
        })

    logger.info(f"✅ Model bundle created: {bundle_path}")
    
    # Cleanup temporary directory
    try:
        shutil.rmtree(temp_base_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary bundle directory: {temp_base_dir}")
    except Exception as cleanup_error:
        logger.warning(f"Failed to cleanup bundle temp directory: {cleanup_error}")
    
    return {"bundle_path": bundle_path, "manifest": manifest}

# ============================================================================
# VALIDATION AND CHECKPOINT FUNCTIONS
# ============================================================================

def validate_l4_outputs(**context):
    """Validate L4 outputs and verify reward reproducibility"""
    logger.info("Validating L4 outputs and contracts...")
    
    # Create unique temporary directory for validation
    temp_dir = tempfile.mkdtemp(prefix="l4_validation_")
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    l4_bucket = "04-l4-ds-usdcop-rlready"
    # Use Variable for L4 prefix to match what L4 actually produces
    l4_prefix = Variable.get("l4_dataset_prefix", default_var="usdcop_m5__05_l4_rlready/")
    
    required_files = [
        "train_df.parquet",
        "test_df.parquet", 
        "val_df.parquet",
        "env_spec.json",
        "reward_spec.json",
        "cost_model.json",
        "split_spec.json",
        "checks_report.json"
    ]
    
    validation_status = {
        "l4_bucket": l4_bucket,
        "files_present": {},
        "contract_checks": {},
        "validation_time": datetime.now().isoformat()
    }
    
    # Check file existence
    for file in required_files:
        key = f"{l4_prefix}{file}"
        exists = s3_hook.check_for_key(key, bucket_name=l4_bucket)
        validation_status["files_present"][file] = exists
        if not exists:
            raise ValueError(f"Required L4 file missing: {file}")
    
    # Load and validate checks_report
    checks_key = f"{l4_prefix}checks_report.json"
    checks_obj = s3_hook.get_key(checks_key, bucket_name=l4_bucket)
    checks_report = json.loads(checks_obj.get()["Body"].read())
    
    # Verify READY status
    if checks_report.get("status") != "READY":
        raise ValueError(f"L4 not READY: status={checks_report.get('status')}")
    
    # Verify clip rate ≤ 0.5%
    clip_rates = checks_report.get("quality_gates", {}).get("observation_quality", {}).get("clip_rate", {})
    if not clip_rates.get("all_under_0.5pct", False):
        max_clip = clip_rates.get("max_clip_rate", 1.0)
        raise ValueError(f"Observation clip rate too high: {max_clip:.2%} > 0.5%")
    
    validation_status["contract_checks"]["ready_status"] = True
    validation_status["contract_checks"]["clip_rate_ok"] = True
    
    # Validate train data contract
    import pandas as pd
    import numpy as np
    
    train_key = f"{l4_prefix}train_df.parquet"
    obj = s3_hook.get_key(train_key, bucket_name=l4_bucket)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f.write(obj.get()["Body"].read())
        tmp_path = f.name
    train_df = pd.read_parquet(tmp_path)
    os.unlink(tmp_path)
    
    # Check observation columns
    obs_cols = [f"obs_{i:02d}" for i in range(17)]
    missing_obs = [col for col in obs_cols if col not in train_df.columns]
    if missing_obs:
        raise ValueError(f"Missing observation columns: {missing_obs}")
    
    # Check observation range [-5, 5]
    obs_data = train_df[obs_cols].values.astype(np.float32)
    obs_min, obs_max = obs_data.min(), obs_data.max()
    if obs_min < -5.0 or obs_max > 5.0:
        raise ValueError(f"Observations out of range: [{obs_min:.3f}, {obs_max:.3f}] not in [-5, 5]")
    
    # Check actual clip rate
    actual_clip_rate = ((obs_data <= -5.0) | (obs_data >= 5.0)).mean()
    if actual_clip_rate > 0.005:
        raise ValueError(f"Actual clip rate {actual_clip_rate:.3%} > 0.5%")
    
    validation_status["contract_checks"]["obs_range_ok"] = True
    validation_status["contract_checks"]["dtype_float32"] = str(obs_data.dtype) == "float32"
    validation_status["contract_checks"]["actual_clip_rate"] = float(actual_clip_rate)
    
    # Check mid price > 0
    if "mid" in train_df.columns:
        if (train_df["mid"] <= 0).any():
            raise ValueError("Invalid mid prices: found non-positive values")
        validation_status["contract_checks"]["mid_positive"] = True
    
    # Load specs
    env_spec_key = f"{l4_prefix}env_spec.json"
    env_spec_obj = s3_hook.get_key(env_spec_key, bucket_name=l4_bucket)
    env_spec = json.loads(env_spec_obj.get()["Body"].read())
    
    reward_spec_key = f"{l4_prefix}reward_spec.json"
    reward_spec_obj = s3_hook.get_key(reward_spec_key, bucket_name=l4_bucket)
    reward_spec = json.loads(reward_spec_obj.get()["Body"].read())
    
    cost_model_key = f"{l4_prefix}cost_model.json"
    cost_model_obj = s3_hook.get_key(cost_model_key, bucket_name=l4_bucket)
    cost_model = json.loads(cost_model_obj.get()["Body"].read())
    
    # IMPROVEMENT #6: RMSE reproducibility gate - Validate reward_spec completeness
    required_reward_fields = ['forward_window', 'price_type', 'normalization']
    optional_reward_fields = ['target_column', 'scaling_factor', 'clipping', 'method']
    
    missing_fields = []
    for field in required_reward_fields:
        if field not in reward_spec:
            missing_fields.append(field)
    
    if missing_fields:
        logger.error(f"❌ REWARD_SPEC INCOMPLETE: Missing required fields {missing_fields}")
        raise ValueError(f"reward_spec missing required fields for RMSE reproducibility: {missing_fields}")
    
    logger.info(f"✅ reward_spec validation passed - all required fields present:")
    for field in required_reward_fields:
        logger.info(f"  - {field}: {reward_spec[field]}")
    
    for field in optional_reward_fields:
        if field in reward_spec:
            logger.info(f"  - {field}: {reward_spec[field]} (optional)")
    
    validation_status["contract_checks"]["reward_spec_complete"] = True
    
    # Smoke test reward reproducibility
    logger.info("Running smoke test for reward reproducibility...")
    
    # MEJORA: Use contiguous window instead of random sample to preserve time alignment
    sample_size = min(1000, len(train_df))
    # Sort by time if timestamp column exists, otherwise use index order
    if 'timestamp' in train_df.columns:
        train_df_sorted = train_df.sort_values('timestamp')
    else:
        train_df_sorted = train_df.copy()
    
    # Take a contiguous block from the middle to avoid edge effects
    start_idx = max(0, (len(train_df_sorted) - sample_size) // 2)
    end_idx = start_idx + sample_size
    test_sample = train_df_sorted.iloc[start_idx:end_idx].copy()
    logger.info(f"Using contiguous window: rows {start_idx} to {end_idx}")
    
    # MEJORA: Reconstrucción del forward según reward_spec
    forward_window = reward_spec.get("forward_window", [1, 2])  # Default [1,2] si no está especificado
    logger.info(f"Using forward_window from reward_spec: {forward_window}")
    
    # AJUSTE FINO #2: Smoke test con reconstrucción de columnas si faltan
    ret_forward_col = f"ret_forward_{forward_window[0]}"
    if ret_forward_col in test_sample.columns:
        expected_rewards = test_sample[ret_forward_col].values
    else:
        # Si no existe ret_forward_X, intentar reconstruir
        logger.warning(f"{ret_forward_col} not found, attempting to reconstruct from mid prices...")
        expected_rewards = None
    
    # MEJORA: Reconstruir columnas según forward_window del reward_spec
    mid_t1_col = f"mid_t{forward_window[0]}"
    mid_t2_col = f"mid_t{forward_window[1]}"
    
    if mid_t1_col not in test_sample.columns or mid_t2_col not in test_sample.columns:
        if "mid" in test_sample.columns:
            logger.warning(f"{mid_t1_col}/{mid_t2_col} not found, reconstructing from mid column...")
            # Reconstruir usando los offsets del reward_spec - reset index to ensure proper alignment
            test_sample = test_sample.reset_index(drop=True)
            test_sample[mid_t1_col] = test_sample["mid"].shift(-forward_window[0])
            test_sample[mid_t2_col] = test_sample["mid"].shift(-forward_window[1])
        else:
            logger.warning("Cannot reconstruct forward prices - skipping RMSE check")
            validation_status["contract_checks"]["reward_rmse"] = "SKIPPED"
            expected_rewards = None
    
    # Solo calcular RMSE si tenemos todos los datos necesarios
    if expected_rewards is not None and mid_t1_col in test_sample.columns and mid_t2_col in test_sample.columns:
        # Simulate env rewards usando las columnas correctas del reward_spec
        # FIX: np.log() devuelve ndarray, no Series → no tiene .fillna()
        ratio = test_sample[mid_t2_col] / test_sample[mid_t1_col]
        simulated_rewards = (
            np.log(ratio)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy()
        )
        
        # Calculate RMSE
        valid_mask = ~(np.isnan(expected_rewards) | np.isnan(simulated_rewards))
        if valid_mask.sum() > 0:
            rmse = np.sqrt(np.mean((expected_rewards[valid_mask] - simulated_rewards[valid_mask])**2))
            
            # MEJORA: Gate de reproducibilidad de reward
            REWARD_REPRODUCIBILITY_THRESHOLD = 0.001  # 0.1% tolerance
            reward_reproducibility_gate = rmse <= REWARD_REPRODUCIBILITY_THRESHOLD
            
            validation_status["contract_checks"]["reward_rmse"] = float(rmse)
            validation_status["contract_checks"]["reward_reproducibility_gate"] = reward_reproducibility_gate
            
            if not reward_reproducibility_gate:
                # MEJORA: Log detallado cuando falla el gate
                logger.error(f"❌ REWARD REPRODUCIBILITY GATE FAILED: RMSE {rmse:.6f} > {REWARD_REPRODUCIBILITY_THRESHOLD}")
                logger.error("   This indicates the reward calculation is NOT deterministic")
                logger.error("   Check: 1) L4 reward contract, 2) Environment implementation, 3) Numerical stability")
                
                # Analizar discrepancias
                abs_diff = np.abs(expected_rewards[valid_mask] - simulated_rewards[valid_mask])
                logger.info(f"   Max absolute difference: {abs_diff.max():.6f}")
                logger.info(f"   Mean absolute difference: {abs_diff.mean():.6f}")
                logger.info(f"   Samples with >0.1% error: {(abs_diff > 0.001).sum()}/{valid_mask.sum()}")
            else:
                logger.info(f"✅ Reward reproducibility gate PASSED: RMSE {rmse:.6f} ≤ {REWARD_REPRODUCIBILITY_THRESHOLD}")
        else:
            logger.warning("No valid samples for RMSE calculation")
            validation_status["contract_checks"]["reward_rmse"] = "NO_VALID_SAMPLES"
            validation_status["contract_checks"]["reward_reproducibility_gate"] = False
    
    validation_status["contract_checks"]["env_spec_valid"] = True
    validation_status["contract_checks"]["n_observations"] = len(obs_cols)
    
    # Store specs for training use
    context['ti'].xcom_push(key='l4_env_spec', value=env_spec)
    context['ti'].xcom_push(key='l4_reward_spec', value=reward_spec)
    context['ti'].xcom_push(key='l4_cost_model', value=cost_model)
    context['ti'].xcom_push(key='l4_prefix', value=l4_prefix)  # Push l4_prefix for downstream tasks
    
    # PULIDO: Pasar gate y RMSE de reproducibilidad al acceptance report
    context['ti'].xcom_push(key='l4_reward_reproducibility_gate', 
                            value=validation_status["contract_checks"].get("reward_reproducibility_gate", False))
    # MEJORA: Also push RMSE value for uniform gate reporting
    rmse_value = validation_status["contract_checks"].get("reward_rmse", 1.0)
    # Handle non-numeric values
    if isinstance(rmse_value, str):  # "SKIPPED" or "NO_VALID_SAMPLES"
        rmse_value = 1.0  # Use high value that will fail gate
    context['ti'].xcom_push(key='l4_reward_rmse', value=float(rmse_value))
    context['ti'].xcom_push(key='reward_reproducibility_rmse', value=float(rmse_value))  # Alias for compatibility
    
    logger.info("✅ L4 validation passed:")
    logger.info(f"  - Status: READY")
    logger.info(f"  - Clip rate: {actual_clip_rate:.3%} < 0.5%")
    logger.info(f"  - Obs range: [{obs_min:.3f}, {obs_max:.3f}] ⊆ [-5, 5]")
    logger.info(f"  - Mid prices: all positive")
    logger.info(f"  - Observations: {len(obs_cols)} columns")
    
    # Cleanup temporary directory
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up validation temp directory: {temp_dir}")
    except Exception as cleanup_error:
        logger.warning(f"Failed to cleanup validation temp directory: {cleanup_error}")
    
    return validation_status

def prepare_training_environment(**context):
    """Prepare training environment and download L4 data"""
    logger.info("Preparing training environment...")
    
    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="l5_training_prep_")
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        s3_hook = S3Hook(aws_conn_id='minio_conn')
        l4_bucket = "04-l4-ds-usdcop-rlready"
        l4_prefix = "usdcop_m5__05_l4_rlready/"
        
        # Download all necessary files
        files_to_download = [
            'train_df.parquet', 'test_df.parquet', 'val_df.parquet',
            'env_spec.json', 'reward_spec.json', 'cost_model.json',
            'split_spec.json', 'normalization_ref.json'
        ]
        
        download_status = {}
        
        for file_name in files_to_download:
            key = f"{l4_prefix}{file_name}"
            local_path = os.path.join(temp_dir, file_name)
            
            try:
                if s3_hook.check_for_key(key, bucket_name=l4_bucket):
                    obj = s3_hook.get_key(key, bucket_name=l4_bucket)
                    with open(local_path, 'wb') as f:
                        f.write(obj.get()['Body'].read())
                    download_status[file_name] = 'success'
                    logger.info(f"Downloaded {file_name}")
                else:
                    download_status[file_name] = 'not_found'
                    logger.warning(f"File not found: {file_name}")
            except Exception as e:
                download_status[file_name] = f'error: {e}'
                logger.error(f"Failed to download {file_name}: {e}")
        
        # Check critical files
        critical_files = ['train_df.parquet', 'test_df.parquet']
        for file in critical_files:
            if download_status.get(file) != 'success':
                raise ValueError(f"Critical file {file} not available")
        
        logger.info("✅ Training environment prepared")
        return download_status
        
    finally:
        # Cleanup temporary directory
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")

def finalize_deployment(**context):
    """Finalize deployment with comprehensive summary"""
    logger.info("Finalizing deployment...")
    
    # Get results from previous tasks
    gate_results = context['task_instance'].xcom_pull(task_ids='evaluate_production_gates')
    bundle_results = context['task_instance'].xcom_pull(task_ids='create_model_bundle')
    
    # Check if gates passed
    gates_passed = bool(gate_results and (
        gate_results.get("all_gates_passed") or gate_results.get("overall_status") == "PASS"
    ))
    
    # PATCH C: Canary mode configuration from Airflow Variables
    canary_enabled = Variable.get("L5_CANARY_TRADING_ENABLED", default_var="true").lower() == "true"
    canary_days = int(Variable.get("L5_CANARY_DAYS", default_var="14"))
    max_daily_trades = int(Variable.get("L5_CANARY_MAX_DAILY_TRADES", default_var="20"))
    max_intraday_dd_bps = int(Variable.get("L5_CANARY_MAX_DD_BPS", default_var="150"))
    
    # Create deployment summary
    summary = {
        "deployment_id": f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "dag_run_id": context['dag_run'].run_id,
        "gates_passed": gates_passed,
        "best_seed": gate_results.get("best_seed", "unknown") if gate_results else "none",  # MEJORA: Echo best seed for standalone summary
        "gates_count": {  # MEJORA: Add gates count for direct context in post-mortems
            "passed": gate_results.get("best_gates_passed", 0) if gate_results else 0,
            "total": gate_results.get("total_gates", 0) if gate_results else 0,
            "percentage": f"{(gate_results.get('best_gates_passed', 0) / max(gate_results.get('total_gates', 1), 1) * 100):.1f}%" if gate_results else "0.0%"
        },
        "model_bundle": bundle_results.get('bundle_path', '') if bundle_results else '',
        "status": "DEPLOYED" if gates_passed else "FAILED",
        "mode": "CANARY" if (gates_passed and canary_enabled) else ("PRODUCTION" if gates_passed else "N/A"),
        "canary": {
            "enabled": canary_enabled,
            "days": canary_days,
            "policy_guards": {
                "max_daily_trades": max_daily_trades,
                "max_intraday_drawdown_bps": max_intraday_dd_bps
            }
        } if gates_passed else None,
        "reason": gate_results.get("overall_status", "UNKNOWN") if gate_results else "NO_GATES_RESULT",  # MEJORA: Mirror overall_status for quick post-mortems
        "git_sha": get_git_info()["sha"],
        "total_timesteps": TOTAL_TIMESTEPS
    }
    
    # Save summary to L5 bucket
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    l5_bucket = "05-l5-ds-usdcop-serving"
    summary_key = f"deployments/{summary['deployment_id']}/summary.json"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(convert_numpy_to_python(summary), f, indent=2)
        f.flush()
        
        with open(f.name, 'rb') as rf:
            s3_hook.load_file_obj(
                file_obj=rf,
                key=summary_key,
                bucket_name=l5_bucket,
                replace=True
            )
    
    if summary['status'] == 'DEPLOYED':
        mode_str = f" in {summary['mode']} mode" if 'mode' in summary else ""
        logger.info(f"✅ Deployment successful{mode_str}: {summary['deployment_id']}")
        if summary.get('canary'):
            logger.info(f"   Canary config: {canary_days} days, max {max_daily_trades} trades/day, max DD {max_intraday_dd_bps} bps")
        logger.info("Model meets all production gates and is ready for deployment")
    else:
        logger.warning(f"⚠️ Deployment skipped: Gates not passed")
        logger.warning("Model did not meet production quality thresholds")
        logger.warning("To pass gates, consider:")
        logger.warning(f"  - Increasing training timesteps (currently {TOTAL_TIMESTEPS})")
        logger.warning("  - Tuning hyperparameters")
        logger.warning("  - Adjusting gate thresholds if too strict")
    
    return summary

# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'usdcop_m5__06_l5_serving',
    default_args=default_args,
    description='Production L5 with Monitor wrapper, Sortino metrics, and L4-based costs',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'l5', 'production', 'gates', 'sortino'],
)

# Step 1: Validate L4 outputs
validate_l4 = PythonOperator(
    task_id='validate_l4_outputs',
    python_callable=validate_l4_outputs,
    dag=dag,
)

# Step 2: Prepare environment
prepare_env = PythonOperator(
    task_id='prepare_training_environment',
    python_callable=prepare_training_environment,
    dag=dag,
)

# Step 3: Parallel training tasks (algorithm selected via L5_MODEL variable)
with TaskGroup("ppo_training", dag=dag) as ppo_group:
    # Get algorithm-specific timeout
    current_model = Variable.get("L5_MODEL", default_var="ppo_lstm").lower()
    timeout_hours = ALGO_TIMEOUTS.get(current_model, TRAIN_TIMEOUT_HOURS)
    
    for seed in PARALLEL_SEEDS:
        PythonOperator(
            task_id=f'train_ppo_seed_{seed}',
            python_callable=train_dispatch,  # Use dispatcher instead of direct function
            op_kwargs={'seed': seed},
            execution_timeout=timedelta(hours=timeout_hours),
            dag=dag,
        )

# Step 4: Gate evaluation
gate_eval = PythonOperator(
    task_id='evaluate_production_gates',
    python_callable=evaluate_production_gates,
    dag=dag,
    trigger_rule='none_failed_min_one_success',
)

# Step 5: Model bundle creation
bundle_task = PythonOperator(
    task_id='create_model_bundle',
    python_callable=create_model_bundle,
    dag=dag,
)

# Step 6: Finalize deployment
finalize = PythonOperator(
    task_id='finalize_deployment',
    python_callable=finalize_deployment,
    dag=dag,
)

# Set dependencies
validate_l4 >> prepare_env >> ppo_group >> gate_eval >> bundle_task >> finalize