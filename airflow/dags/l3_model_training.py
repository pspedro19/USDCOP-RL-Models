"""
DAG: v3.l3_model_training
=========================
USD/COP Trading System - Model Training Pipeline

This is a THIN WRAPPER around the TrainingEngine.
All training logic is in src/training/engine.py (DRY principle).

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  L2 Preprocessing (upstream)                            │
    │  - Generates RL datasets                                │
    │  - Pushes dataset_path, hash via XCom                   │
    └─────────────────────────────────────────────────────────┘
                              ↓ XCom (L2XComKeysEnum)
    ┌─────────────────────────────────────────────────────────┐
    │  Airflow DAG (this file)                                │
    │  - Orchestration only                                   │
    │  - XCom passing (using contracts)                       │
    │  - Alerting                                             │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │  TrainingEngine (src/training/engine.py)                │
    │  - ALL training logic                                   │
    │  - Dataset validation                                   │
    │  - Norm stats generation                                │
    │  - Contract creation                                    │
    │  - PPO training                                         │
    │  - MLflow logging                                       │
    │  - Model registration                                   │
    └─────────────────────────────────────────────────────────┘
                              ↓ XCom (L3XComKeysEnum)
    ┌─────────────────────────────────────────────────────────┐
    │  L4 Experiment Runner (downstream)                      │
    │  - Receives model_path, metrics                         │
    │  - Compares with baseline                               │
    └─────────────────────────────────────────────────────────┘

Author: Trading Team
Version: 2.1.0 (L2 XCom integration with contracts)
Date: 2026-01-18
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
import json
import logging
import os
import sys

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

# XCom Contracts - SSOT for inter-DAG communication
try:
    from airflow.dags.contracts.xcom_contracts import (
        L2XComKeysEnum,
        L3XComKeysEnum,
        L2_DAG_ID,
        L3Output,
        pull_l2_output,
        compute_file_hash,
    )
    XCOM_CONTRACTS_AVAILABLE = True
    logging.info("[SSOT] XCom contracts loaded successfully")
except ImportError as e:
    XCOM_CONTRACTS_AVAILABLE = False
    logging.warning(f"[SSOT] XCom contracts not available: {e}")

try:
    from src.training.engine import TrainingEngine, TrainingRequest
    from src.training.config import PPO_HYPERPARAMETERS, RewardConfig, REWARD_CONFIG
    ENGINE_AVAILABLE = True
    logging.info(f"[SSOT] TrainingEngine loaded successfully")
except ImportError as e:
    ENGINE_AVAILABLE = False
    REWARD_CONFIG = None
    RewardConfig = None
    logging.error(f"[SSOT] TrainingEngine not available: {e}")

try:
    from src.core.contracts.feature_contract import FEATURE_ORDER, FEATURE_ORDER_HASH
except ImportError:
    FEATURE_ORDER = None
    FEATURE_ORDER_HASH = None

# =============================================================================
# EXPERIMENT SSOT - Single Source of Truth for L2 + L3
# =============================================================================
try:
    from src.config.experiment_loader import (
        load_experiment_config as load_experiment_ssot,
        ExperimentConfig,
        get_training_config as get_ssot_training_config,
        get_reward_config as get_ssot_reward_config,
    )
    EXPERIMENT_SSOT = load_experiment_ssot()
    EXPERIMENT_SSOT_AVAILABLE = True
    logging.info(f"[SSOT] Loaded experiment SSOT v{EXPERIMENT_SSOT.version}")
except ImportError as e:
    EXPERIMENT_SSOT = None
    EXPERIMENT_SSOT_AVAILABLE = False
    logging.warning(f"[SSOT] Experiment SSOT not available: {e}")

# Fallback to old experiment config system
try:
    from src.experiments import load_experiment_config, ExperimentConfig as OldExperimentConfig
    EXPERIMENT_CONFIG_AVAILABLE = True
except ImportError:
    EXPERIMENT_CONFIG_AVAILABLE = False
    logging.warning("[SSOT] Legacy ExperimentConfig not available")

# GAP 1, 2: DVC and Lineage tracking
try:
    from src.core.services import (
        DVCService,
        DVCTag,
        LineageTracker,
        create_dvc_service,
        create_lineage_tracker,
    )
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False
    logging.warning("[SSOT] LineageTracker not available")

# MinIO-First Architecture: ExperimentManager for storage
try:
    from src.ml_workflow.experiment_manager import ExperimentManager
    EXPERIMENT_MANAGER_AVAILABLE = True
except ImportError:
    EXPERIMENT_MANAGER_AVAILABLE = False
    logging.warning("[SSOT] ExperimentManager not available - using local storage")

# =============================================================================
# CONFIGURATION
# =============================================================================
try:
    from contracts.dag_registry import RL_L3_MODEL_TRAINING
    DAG_ID = RL_L3_MODEL_TRAINING
except ImportError:
    DAG_ID = "rl_l3_01_model_training"
PROJECT_ROOT = Path('/opt/airflow')

# Build DEFAULT_CONFIG from SSOT when available
if EXPERIMENT_SSOT_AVAILABLE and EXPERIMENT_SSOT is not None:
    _ssot_training = EXPERIMENT_SSOT.training
    _ssot_output = EXPERIMENT_SSOT.pipeline
    DEFAULT_CONFIG = {
        "version": "auto",
        "experiment_name": EXPERIMENT_SSOT.logging.experiment_name,
        "dataset_name": f"{_ssot_output.output_prefix}_train.parquet",
        "dataset_dir": "5min",
        "total_timesteps": _ssot_training.total_timesteps,
        "learning_rate": _ssot_training.learning_rate,
        "n_steps": _ssot_training.n_steps,
        "batch_size": _ssot_training.batch_size,
        "n_epochs": _ssot_training.n_epochs,
        "gamma": _ssot_training.gamma,
        "gae_lambda": _ssot_training.gae_lambda,
        "clip_range": _ssot_training.clip_range,
        "ent_coef": _ssot_training.ent_coef,
        "vf_coef": _ssot_training.vf_coef,
        "max_grad_norm": _ssot_training.max_grad_norm,
        "mlflow_enabled": True,
        "auto_register": True,
        "experiment_config_path": None,
        "dvc_enabled": True,
        "dvc_auto_tag": True,
    }
    logging.info(f"[SSOT] Using experiment SSOT config: ent_coef={_ssot_training.ent_coef}")
else:
    DEFAULT_CONFIG = {
        "version": "auto",
        "experiment_name": "ppo_usdcop",
        "dataset_name": "DS_default_train.parquet",
        "dataset_dir": "5min",
        "total_timesteps": PPO_HYPERPARAMETERS.total_timesteps if ENGINE_AVAILABLE else 500_000,
        "mlflow_enabled": True,
        "auto_register": True,
        "experiment_config_path": None,
        "dvc_enabled": True,
        "dvc_auto_tag": True,
    }
    logging.warning("[SSOT] Using fallback config (SSOT not available)")


def get_training_config(**context) -> Dict[str, Any]:
    """Get training configuration from Airflow Variables or DAG run conf.

    IMPORTANT: Force reloads SSOT at runtime to pick up any changes made
    after DAG import time. This ensures hyperparameter tuning takes effect.
    """
    # Force reload SSOT at runtime to get fresh values
    try:
        from src.config.experiment_loader import load_experiment_config
        ssot = load_experiment_config(force_reload=True)
        _ssot_training = ssot.training
        _ssot_output = ssot.pipeline
        config = {
            "version": "auto",
            "experiment_name": ssot.logging.experiment_name,
            "dataset_name": f"{_ssot_output.output_prefix}_train.parquet",
            "dataset_dir": "5min",
            "total_timesteps": _ssot_training.total_timesteps,
            "learning_rate": _ssot_training.learning_rate,
            "n_steps": _ssot_training.n_steps,
            "batch_size": _ssot_training.batch_size,
            "n_epochs": _ssot_training.n_epochs,
            "gamma": _ssot_training.gamma,
            "gae_lambda": _ssot_training.gae_lambda,
            "clip_range": _ssot_training.clip_range,
            "ent_coef": _ssot_training.ent_coef,
            "vf_coef": _ssot_training.vf_coef,
            "max_grad_norm": _ssot_training.max_grad_norm,
            "mlflow_enabled": True,
            "auto_register": True,
            "experiment_config_path": None,
            "dvc_enabled": True,
            "dvc_auto_tag": True,
        }
        logging.info(f"[SSOT-RUNTIME] Loaded fresh config: lr={_ssot_training.learning_rate}, ent_coef={_ssot_training.ent_coef}")
    except (ImportError, FileNotFoundError) as e:
        logging.warning(f"[SSOT-RUNTIME] Failed to reload SSOT, using cached config: {e}")
        config = DEFAULT_CONFIG.copy()

    # Override from Variable
    try:
        var_config = Variable.get("training_config", default_var=None)
        if var_config:
            config.update(json.loads(var_config))
    except Exception as e:
        logging.warning(f"Could not load training_config Variable: {e}")

    # Override from dag_run.conf
    if context.get('dag_run') and context['dag_run'].conf:
        config.update(context['dag_run'].conf)

    # Auto-increment version if needed
    if config.get("version") == "auto":
        config["version"] = _get_next_version()

    return config


def _get_next_version() -> str:
    """Get next model version from database."""
    try:
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(CAST(REGEXP_REPLACE(model_version, '[^0-9]', '', 'g') AS INTEGER))
            FROM model_registry WHERE model_version ~ '^v?[0-9]+$'
        """)
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result and result[0]:
            return f"v{result[0] + 1}"
        return "v1"
    except Exception as e:
        logging.warning(f"Auto-version failed: {e}")
        return f"v{datetime.now().strftime('%Y%m%d')}"


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def run_training(**context) -> Dict[str, Any]:
    """
    Main training task - delegates to TrainingEngine.

    This is the ONLY place that calls the engine.
    All training logic is centralized in engine.py.

    Dataset Resolution Priority (SSOT):
        1. dag_run.conf['dataset_path'] - Explicit path from L4 trigger
        2. L2 XCom - Pull from L2 preprocessing DAG
        3. Config fallback - Build from dataset_dir/dataset_name

    GAP 2: Now supports experiment_config_path parameter for YAML-based config.
    GAP 4: Pushes config_hash to XCom for lineage tracking.
    GAP 1: Integrates DVC for dataset versioning.
    """
    if not ENGINE_AVAILABLE:
        raise RuntimeError("TrainingEngine not available. Check imports.")

    config = get_training_config(**context)
    ti = context['ti']

    logging.info("=" * 60)
    logging.info(f"Starting training v{config['version']}")
    logging.info("=" * 60)

    # GAP 2: Load experiment config from YAML if provided
    experiment_config = None
    config_hash = None
    experiment_config_path = config.get('experiment_config_path')

    if experiment_config_path and EXPERIMENT_CONFIG_AVAILABLE:
        exp_config_full_path = PROJECT_ROOT / experiment_config_path
        if exp_config_full_path.exists():
            logging.info(f"Loading experiment config: {exp_config_full_path}")
            experiment_config = load_experiment_config(exp_config_full_path)

            # Compute config hash for lineage
            import hashlib
            with open(exp_config_full_path, 'rb') as f:
                config_hash = hashlib.sha256(f.read()).hexdigest()[:16]

            # Override config values from experiment YAML
            if experiment_config.training.total_timesteps:
                config['total_timesteps'] = experiment_config.training.total_timesteps
            if experiment_config.experiment.name:
                config['experiment_name'] = experiment_config.experiment.name
            if experiment_config.experiment.version:
                config['version'] = f"v{experiment_config.experiment.version.replace('.', '_')}"

            logging.info(f"Experiment: {experiment_config.experiment.name}")
            logging.info(f"Config hash: {config_hash}")
        else:
            logging.warning(f"Experiment config not found: {exp_config_full_path}")

    # =========================================================================
    # DATASET RESOLUTION - SSOT Priority (MinIO-First Architecture)
    # =========================================================================
    dataset_path = None
    dataset_hash = None
    l2_output = None
    experiment_id = None
    dataset_version = None
    is_minio_first = False

    # Priority 1: Explicit dataset_path from dag_run.conf (L4 trigger)
    if config.get('dataset_path'):
        dataset_path = Path(config['dataset_path'])
        dataset_hash = config.get('dataset_hash')
        logging.info(f"[SSOT] Dataset from dag_run.conf: {dataset_path}")
        if dataset_hash:
            logging.info(f"[SSOT] Expected hash: {dataset_hash}")

    # Priority 2: Pull from L2 XCom (if L2 ran before us)
    elif XCOM_CONTRACTS_AVAILABLE:
        l2_output = pull_l2_output(ti)
        if l2_output:
            # Check if L2 used MinIO-First architecture
            if hasattr(l2_output, 'is_minio_first') and l2_output.is_minio_first():
                is_minio_first = True
                experiment_id = l2_output.experiment_id
                dataset_version = l2_output.version
                logging.info(f"[MinIO-First] L2 used MinIO storage")
                logging.info(f"[MinIO-First] Dataset URI: {l2_output.dataset_uri}")
                logging.info(f"[MinIO-First] Experiment ID: {experiment_id}")
                logging.info(f"[MinIO-First] Version: {dataset_version}")

                # Load dataset from MinIO
                if EXPERIMENT_MANAGER_AVAILABLE:
                    try:
                        manager = ExperimentManager(experiment_id)
                        df = manager.load_dataset(version=dataset_version)
                        logging.info(f"[MinIO-First] Loaded dataset from MinIO: {len(df)} rows")

                        # Save locally for training (TrainingEngine expects local file)
                        local_cache_dir = PROJECT_ROOT / "data" / "cache" / experiment_id
                        local_cache_dir.mkdir(parents=True, exist_ok=True)
                        dataset_path = local_cache_dir / f"{dataset_version}_train.parquet"
                        df.to_parquet(dataset_path, index=False)
                        logging.info(f"[MinIO-First] Cached locally at: {dataset_path}")

                        dataset_hash = l2_output.dataset_hash
                    except Exception as e:
                        logging.error(f"[MinIO-First] Failed to load from MinIO: {e}")
                        logging.warning("[MinIO-First] Falling back to local path")
                        is_minio_first = False
                else:
                    logging.warning("[MinIO-First] ExperimentManager not available, using local path")
                    is_minio_first = False

            # Fall back to local path from L2 if MinIO not available
            if not is_minio_first and l2_output.dataset_path:
                dataset_path = Path(l2_output.dataset_path)
                dataset_hash = l2_output.dataset_hash
                experiment_id = l2_output.experiment_name

            logging.info(f"[SSOT] Dataset from L2 XCom: {dataset_path}")
            logging.info(f"[SSOT] L2 hash: {dataset_hash}")
            logging.info(f"[SSOT] L2 row count: {l2_output.row_count}")
            logging.info(f"[SSOT] L2 experiment: {l2_output.experiment_name}")

    # Priority 3: Config fallback (legacy behavior)
    if dataset_path is None:
        # L2 builder outputs to "5min/" not "datasets_5min/"
        dataset_path = (
            PROJECT_ROOT / "data" / "pipeline" / "07_output" /
            config['dataset_dir'] / config['dataset_name']
        )
        logging.info(f"[SSOT] Dataset from config fallback: {dataset_path}")

    # Validate dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Validate hash if provided
    if dataset_hash and XCOM_CONTRACTS_AVAILABLE:
        actual_hash = compute_file_hash(dataset_path)
        if actual_hash != dataset_hash:
            logging.warning(
                f"[SSOT] Dataset hash mismatch! "
                f"Expected: {dataset_hash}, Actual: {actual_hash}"
            )
            # Don't fail - just log warning for now
        else:
            logging.info(f"[SSOT] Dataset hash verified: {actual_hash}")

    # Get DB connection string
    db_connection = os.environ.get("DATABASE_URL")

    # Get reward configuration from dag_run.conf or use default SSOT
    reward_config = None
    reward_contract_id = config.get('reward_contract_id', 'v1.0.0')
    enable_curriculum = config.get('enable_curriculum', True)

    if RewardConfig is not None:
        # FIX 2026-02-01: Always get defaults from SSOT, not hardcoded values
        # This ensures experiment_ssot.yaml is the single source of truth
        ssot_defaults = {
            'pnl': 0.80,           # SSOT experiment_ssot.yaml
            'dsr': 0.15,           # SSOT experiment_ssot.yaml
            'sortino': 0.05,       # SSOT experiment_ssot.yaml
            'regime_penalty': 0.3, # SSOT experiment_ssot.yaml
            'holding_decay': 0.2,  # SSOT experiment_ssot.yaml
            'anti_gaming': 0.3,    # SSOT experiment_ssot.yaml
        }

        # Try to load from SSOT at runtime for fresh values
        if EXPERIMENT_SSOT_AVAILABLE:
            try:
                from src.config.experiment_loader import load_experiment_config
                _ssot = load_experiment_config(force_reload=True)
                ssot_defaults = {
                    'pnl': _ssot.reward.pnl_weight,
                    'dsr': _ssot.reward.dsr_weight,
                    'sortino': _ssot.reward.sortino_weight,
                    'regime_penalty': _ssot.reward.regime_penalty,
                    'holding_decay': _ssot.reward.holding_decay,
                    'anti_gaming': _ssot.reward.anti_gaming,
                }
                logging.info(f"[REWARD-SSOT] Loaded from experiment_ssot.yaml: pnl={ssot_defaults['pnl']}")
            except Exception as e:
                logging.warning(f"[REWARD-SSOT] Failed to load SSOT, using hardcoded SSOT values: {e}")

        # Check for custom reward weights in config (overrides SSOT)
        if 'reward_weights' in config:
            weights = config['reward_weights']
            reward_config = RewardConfig(
                weight_pnl=weights.get('pnl', ssot_defaults['pnl']),
                weight_dsr=weights.get('dsr', ssot_defaults['dsr']),
                weight_sortino=weights.get('sortino', ssot_defaults['sortino']),
                weight_regime_penalty=weights.get('regime_penalty', ssot_defaults['regime_penalty']),
                weight_holding_decay=weights.get('holding_decay', ssot_defaults['holding_decay']),
                weight_anti_gaming=weights.get('anti_gaming', ssot_defaults['anti_gaming']),
                enable_normalization=config.get('enable_reward_normalization', True),
                enable_curriculum=enable_curriculum,
            )
            logging.info(f"[REWARD] Using custom reward weights (with SSOT defaults): {weights}")
        elif REWARD_CONFIG is not None:
            # Use SSOT default config
            reward_config = REWARD_CONFIG
            logging.info(f"[REWARD] Using SSOT default reward config: contract={reward_contract_id}")
        else:
            # Create RewardConfig with SSOT defaults
            reward_config = RewardConfig(
                weight_pnl=ssot_defaults['pnl'],
                weight_dsr=ssot_defaults['dsr'],
                weight_sortino=ssot_defaults['sortino'],
                weight_regime_penalty=ssot_defaults['regime_penalty'],
                weight_holding_decay=ssot_defaults['holding_decay'],
                weight_anti_gaming=ssot_defaults['anti_gaming'],
            )
            logging.info(f"[REWARD] Using SSOT-based RewardConfig: pnl={ssot_defaults['pnl']}, contract={reward_contract_id}")

    # Create request with experiment config
    request = TrainingRequest(
        version=config['version'],
        dataset_path=dataset_path,
        total_timesteps=config.get('total_timesteps'),
        experiment_name=config.get('experiment_name'),
        mlflow_enabled=config.get('mlflow_enabled', True),
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        db_connection_string=db_connection,
        auto_register=config.get('auto_register', True),
        feature_columns=list(FEATURE_ORDER) if FEATURE_ORDER else None,
        # Reward system integration (CTR-REWARD-SNAPSHOT-001)
        reward_config=reward_config,
        reward_contract_id=reward_contract_id,
        enable_curriculum=enable_curriculum,
    )

    # Run training via engine
    engine = TrainingEngine(project_root=PROJECT_ROOT)
    result = engine.run(request)

    if not result.success:
        raise RuntimeError(f"Training failed: {result.errors}")

    # GAP 1: DVC versioning for dataset
    dvc_tag = None
    if config.get('dvc_enabled', True) and LINEAGE_AVAILABLE:
        try:
            dvc_service = create_dvc_service(project_root=PROJECT_ROOT)
            dvc_tag_obj = DVCTag.for_experiment(
                experiment_name=config['experiment_name'],
                version=config['version'],
            )
            dvc_tag = str(dvc_tag_obj)
            logging.info(f"DVC tag created: {dvc_tag}")
        except Exception as e:
            logging.warning(f"DVC tagging skipped: {e}")

    # =========================================================================
    # MinIO-First: SAVE MODEL TO OBJECT STORAGE
    # =========================================================================
    model_snapshot = None
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use experiment_id from L2 if available, otherwise from config
    effective_experiment_id = experiment_id or config.get('experiment_name', 'default')

    if is_minio_first and EXPERIMENT_MANAGER_AVAILABLE:
        logging.info(f"[MinIO-First] Saving model to MinIO for experiment: {effective_experiment_id}")

        try:
            manager = ExperimentManager(effective_experiment_id)

            # Load norm_stats from local path (generated by TrainingEngine)
            norm_stats_local_path = result.model_path.parent / 'norm_stats.json'
            norm_stats = {}
            if norm_stats_local_path.exists():
                import json as json_module
                with open(norm_stats_local_path) as f:
                    norm_stats = json_module.load(f)

            # Load config from local path if exists
            config_local_path = result.model_path.parent / 'config.yaml'
            training_config_dict = {}
            if config_local_path.exists():
                import yaml
                with open(config_local_path) as f:
                    training_config_dict = yaml.safe_load(f)
            else:
                training_config_dict = {
                    'total_timesteps': config.get('total_timesteps'),
                    'experiment_name': config.get('experiment_name'),
                    'version': config['version'],
                }

            # Lineage info
            lineage_info = {
                'dataset_hash': dataset_hash,
                'dataset_version': dataset_version,
                'mlflow_run_id': result.mlflow_run_id,
                'training_duration_seconds': result.training_duration_seconds,
                'best_mean_reward': result.best_mean_reward,
                'dag_run_id': context.get('dag_run').run_id if context.get('dag_run') else None,
            }

            # Save model to MinIO - returns ModelSnapshot
            model_snapshot = manager.save_model(
                model_path=str(result.model_path),
                version=model_version,
                norm_stats=norm_stats,
                config=training_config_dict,
                lineage_info=lineage_info,
            )

            logging.info(f"[MinIO-First] Model saved to: {model_snapshot.storage_uri}")
            logging.info(f"[MinIO-First] Model hash: {model_snapshot.model_hash}")
            logging.info(f"[MinIO-First] Norm stats URI: {model_snapshot.norm_stats_uri}")

        except Exception as e:
            logging.error(f"[MinIO-First] Failed to save model to MinIO: {e}")
            logging.warning("[MinIO-First] Model remains at local path only")
            model_snapshot = None

    # =========================================================================
    # XCOM PUSH - Using contracts for inter-DAG communication
    # =========================================================================

    # Legacy keys (backward compatibility)
    ti.xcom_push(key='training_result', value=result.to_dict())
    ti.xcom_push(key='model_path', value=str(result.model_path))
    ti.xcom_push(key='model_hash', value=result.model_hash)
    ti.xcom_push(key='model_id', value=result.model_id)
    ti.xcom_push(key='mlflow_run_id', value=result.mlflow_run_id)
    ti.xcom_push(key='config_hash', value=config_hash)
    ti.xcom_push(key='feature_order_hash', value=FEATURE_ORDER_HASH)
    ti.xcom_push(key='dvc_tag', value=dvc_tag)
    ti.xcom_push(key='experiment_config_path', value=experiment_config_path)

    # MinIO-First: Push S3 URIs
    if model_snapshot:
        ti.xcom_push(key='model_uri', value=model_snapshot.storage_uri)
        ti.xcom_push(key='norm_stats_uri', value=model_snapshot.norm_stats_uri)
        ti.xcom_push(key='model_version', value=model_version)
        ti.xcom_push(key='experiment_id', value=effective_experiment_id)

    # Contract-based keys (SSOT for L4 consumption)
    if XCOM_CONTRACTS_AVAILABLE:
        if model_snapshot is not None:
            # MinIO-First: Use S3 URIs from snapshot
            l3_output = L3Output(
                # S3 URIs (MinIO-First)
                model_uri=model_snapshot.storage_uri,
                norm_stats_uri=model_snapshot.norm_stats_uri,
                config_uri=model_snapshot.config_uri,
                lineage_uri=model_snapshot.lineage_uri,
                experiment_id=effective_experiment_id,
                version=model_version,
                feature_order_hash=FEATURE_ORDER_HASH,
                observation_dim=model_snapshot.observation_dim,
                # Dataset snapshot from L2
                dataset_snapshot=l2_output if l2_output else None,
                # Legacy fields
                model_path=str(result.model_path),
                model_hash=result.model_hash,
                mlflow_run_id=result.mlflow_run_id or "",
                training_duration=result.training_duration_seconds,
                best_reward=result.best_mean_reward,
                norm_stats_hash=result.to_dict().get('norm_stats_hash', ""),
                dataset_hash=dataset_hash or result.to_dict().get('dataset_hash', ""),
                config_hash=config_hash,
                experiment_name=config.get('experiment_name'),
                # Reward system integration (CTR-REWARD-SNAPSHOT-001)
                reward_contract_id=result.reward_contract_id,
                reward_config_hash=result.reward_config_hash,
                reward_config_uri=result.reward_config_uri,
                curriculum_final_phase=result.curriculum_final_phase,
                reward_weights=result.reward_weights,
            )
            logging.info(f"[MinIO-First] Pushing S3 URIs to XCom")
            logging.info(f"[REWARD] Curriculum final phase: {result.curriculum_final_phase}")
        else:
            # Local storage: use file paths
            l3_output = L3Output(
                model_path=str(result.model_path),
                model_hash=result.model_hash,
                mlflow_run_id=result.mlflow_run_id or "",
                training_duration=result.training_duration_seconds,
                best_reward=result.best_mean_reward,
                norm_stats_hash=result.to_dict().get('norm_stats_hash', ""),
                dataset_hash=dataset_hash or result.to_dict().get('dataset_hash', ""),
                config_hash=config_hash,
                experiment_name=config.get('experiment_name'),
                # Reward system integration (CTR-REWARD-SNAPSHOT-001)
                reward_contract_id=result.reward_contract_id,
                reward_config_hash=result.reward_config_hash,
                reward_config_uri=result.reward_config_uri,
                curriculum_final_phase=result.curriculum_final_phase,
                reward_weights=result.reward_weights,
            )
        l3_output.push_to_xcom(ti)
        logging.info(f"[SSOT] L3Output pushed to XCom via contracts")

    # Build lineage record for XCom (legacy + enhanced)
    lineage_xcom = {
        'run_id': result.model_id,
        'config_hash': config_hash,
        'dataset_hash': dataset_hash or result.to_dict().get('dataset_hash'),
        'dataset_path': str(dataset_path),
        'feature_order_hash': FEATURE_ORDER_HASH,
        'dvc_tag': dvc_tag,
        'mlflow_run_id': result.mlflow_run_id,
        'stage': 'L3_training',
        # L2 provenance (if available)
        'l2_experiment_name': l2_output.experiment_name if l2_output else None,
        'l2_row_count': l2_output.row_count if l2_output else None,
        # Reward system lineage (CTR-REWARD-SNAPSHOT-001)
        'reward_contract_id': result.reward_contract_id,
        'reward_config_hash': result.reward_config_hash,
        'curriculum_final_phase': result.curriculum_final_phase,
    }
    ti.xcom_push(key='lineage', value=lineage_xcom)

    logging.info("=" * 60)
    logging.info(f"Training complete: {result.model_path}")
    logging.info(f"Dataset: {dataset_path}")
    logging.info(f"Dataset hash: {dataset_hash or 'N/A'}")
    logging.info(f"Duration: {result.training_duration_seconds/60:.1f} min")
    logging.info(f"Best reward: {result.best_mean_reward:.2f}")
    logging.info(f"Config hash: {config_hash}")
    logging.info(f"Feature order hash: {FEATURE_ORDER_HASH}")
    logging.info(f"DVC tag: {dvc_tag}")
    if l2_output:
        logging.info(f"L2 Provenance: experiment={l2_output.experiment_name}, rows={l2_output.row_count}")
    if model_snapshot:
        logging.info(f"[MinIO-First] Model S3 URI: {model_snapshot.storage_uri}")
        logging.info(f"[MinIO-First] Norm Stats S3 URI: {model_snapshot.norm_stats_uri}")
    # Reward system summary
    logging.info(f"[REWARD] Contract ID: {result.reward_contract_id or 'N/A'}")
    logging.info(f"[REWARD] Config hash: {result.reward_config_hash or 'N/A'}")
    logging.info(f"[REWARD] Curriculum phase: {result.curriculum_final_phase or 'N/A'}")
    if result.reward_weights:
        logging.info(f"[REWARD] Weights: {result.reward_weights}")
    logging.info("=" * 60)

    return result.to_dict()


def training_summary(**context) -> Dict[str, Any]:
    """Generate training summary."""
    ti = context['ti']
    result = ti.xcom_pull(key='training_result', task_ids='train_model') or {}

    logging.info("=" * 60)
    logging.info("TRAINING SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Version: {result.get('version', 'N/A')}")
    logging.info(f"  Model ID: {result.get('model_id', 'N/A')}")
    logging.info(f"  Model Path: {result.get('model_path', 'N/A')}")
    logging.info(f"  Duration: {result.get('training_duration_seconds', 0)/60:.1f} min")
    logging.info(f"  Best Reward: {result.get('best_mean_reward', 0):.2f}")
    logging.info(f"  MLflow Run: {result.get('mlflow_run_id', 'N/A')}")
    logging.info("=" * 60)

    return result


def on_failure_callback(context):
    """Handle task failures."""
    task = context['task_instance']
    error = context.get('exception')

    logging.error(
        f"TRAINING FAILED\n"
        f"Task: {task.task_id}\n"
        f"Error: {error}"
    )


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': on_failure_callback,
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Model training pipeline using TrainingEngine',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l3', 'training', 'ppo', 'mlflow'],
    params={
        "version": "auto",
        "total_timesteps": 500_000,
    },
)

with dag:
    # Single training task - all logic in engine
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=run_training,
        provide_context=True,
        execution_timeout=timedelta(hours=8),  # INCREASED: 4h → 8h for full training
    )

    # Summary task
    summary_task = PythonOperator(
        task_id='training_summary',
        python_callable=training_summary,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    train_task >> summary_task
