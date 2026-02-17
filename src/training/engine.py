"""
Training Engine - Unified Training Orchestration
=================================================

This is the SINGLE SOURCE OF TRUTH for training orchestration.
All training entry points (DAG, CLI) MUST use this engine.

Design Principles:
    - DRY: Zero duplication of training logic
    - Single Responsibility: One engine, one purpose
    - Dependency Injection: All dependencies passed in
    - Clean Code: Explicit, testable, documented

Usage:
    from src.training.engine import TrainingEngine, TrainingRequest

    # Create request
    request = TrainingRequest(
        version="v1",
        dataset_path=Path("data/training.csv"),
    )

    # Run training
    engine = TrainingEngine(project_root=Path("."))
    result = engine.run(request)

    if result.success:
        print(f"Model: {result.model_path}")

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union

import numpy as np
import pandas as pd

from .config import (
    PPO_HYPERPARAMETERS,
    ENVIRONMENT_CONFIG,
    DATA_SPLIT_CONFIG,
    INDICATOR_CONFIG,
    TrainingConfig,
    RewardConfig,
    get_training_config,
    get_ppo_hyperparameters,
    get_environment_config,
)
from .utils.reproducibility import set_reproducible_seeds

# =============================================================================
# GAP 1, 2: DVC and Lineage Tracking imports
# =============================================================================
try:
    from src.core.services import (
        DVCService,
        DVCTag,
        DVCResult,
        create_dvc_service,
    )
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False
    DVCService = None
    DVCTag = None
    create_dvc_service = None

try:
    from src.core.services import (
        LineageTracker,
        LineageRecord,
        create_lineage_tracker,
    )
    LINEAGE_AVAILABLE = True
except ImportError:
    LINEAGE_AVAILABLE = False
    LineageTracker = None
    create_lineage_tracker = None

# GAP 3: Feature order hash for contract validation
# Try experiment SSOT first, then fall back to feature_contract
FEATURE_ORDER_HASH = None
FEATURE_ORDER = None
try:
    from src.config.experiment_loader import (
        load_experiment_config as _load_ssot,
        get_feature_order_hash as _get_ssot_hash,
    )
    _ssot_config = _load_ssot()
    FEATURE_ORDER_HASH = _ssot_config.feature_order_hash
    FEATURE_ORDER = _ssot_config.feature_order
except (ImportError, FileNotFoundError):
    try:
        from src.core.contracts.feature_contract import FEATURE_ORDER_HASH, FEATURE_ORDER
    except ImportError:
        pass

logger = logging.getLogger(__name__)


# =============================================================================
# Request / Response DTOs
# =============================================================================

@dataclass
class TrainingRequest:
    """
    Training request - all inputs needed to run training.

    This is the ONLY way to request training.
    """
    # Required
    version: str
    dataset_path: Path

    # Optional overrides (defaults from SSOT)
    total_timesteps: Optional[int] = None
    learning_rate: Optional[float] = None

    # Reproducibility - CRITICAL for consistent training results
    seed: int = 42  # Fixed seed for reproducibility

    # Output configuration
    output_dir: Optional[Path] = None
    experiment_name: Optional[str] = None

    # Feature configuration
    feature_columns: Optional[List[str]] = None

    # MLflow
    mlflow_tracking_uri: Optional[str] = None
    mlflow_enabled: bool = True

    # Database registration
    db_connection_string: Optional[str] = None
    auto_register: bool = True

    # Callbacks
    on_progress: Optional[Callable[[str, int, int], None]] = None

    # GAP 1: DVC integration for dataset versioning
    dvc_enabled: bool = True
    dvc_remote: Optional[str] = "minio"  # DVC remote name

    # GAP 2: Lineage tracking
    lineage_enabled: bool = True

    # GAP 5: Experiment config path for artifact logging
    experiment_config_path: Optional[Path] = None

    # Reward system configuration (CTR-REWARD-SNAPSHOT-001)
    reward_config: Optional["RewardConfig"] = None
    reward_contract_id: str = "v1.0.0"  # Default contract version
    enable_curriculum: bool = True  # Enable curriculum learning phases


@dataclass
class TrainingResult:
    """
    Training result - all outputs from training.

    Immutable record of training run.
    """
    success: bool
    version: str

    # Paths
    model_path: Optional[Path] = None
    norm_stats_path: Optional[Path] = None
    contract_path: Optional[Path] = None

    # Hashes (for reproducibility)
    model_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    norm_stats_hash: Optional[str] = None
    config_hash: Optional[str] = None

    # Metrics
    training_duration_seconds: float = 0.0
    total_timesteps: int = 0
    best_mean_reward: float = 0.0
    final_mean_reward: float = 0.0

    # Reproducibility tracking
    training_seed: int = 42  # Seed used for this training run

    # MLflow
    mlflow_run_id: Optional[str] = None

    # Database
    model_id: Optional[str] = None

    # GAP 1: DVC versioning
    dvc_tag: Optional[str] = None
    dvc_pushed: bool = False

    # GAP 2, 3: Lineage and contract tracking
    feature_order_hash: Optional[str] = None
    lineage_record: Optional[Dict[str, Any]] = None

    # Reward system tracking (CTR-REWARD-SNAPSHOT-001)
    reward_contract_id: Optional[str] = None
    reward_config_hash: Optional[str] = None
    reward_config_uri: Optional[str] = None
    curriculum_final_phase: Optional[str] = None
    reward_weights: Optional[Dict[str, float]] = None

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "version": self.version,
            "model_path": str(self.model_path) if self.model_path else None,
            "norm_stats_path": str(self.norm_stats_path) if self.norm_stats_path else None,
            "contract_path": str(self.contract_path) if self.contract_path else None,
            "model_hash": self.model_hash,
            "dataset_hash": self.dataset_hash,
            "norm_stats_hash": self.norm_stats_hash,
            "config_hash": self.config_hash,
            "training_duration_seconds": self.training_duration_seconds,
            "total_timesteps": self.total_timesteps,
            "best_mean_reward": self.best_mean_reward,
            "final_mean_reward": self.final_mean_reward,
            "training_seed": self.training_seed,
            "mlflow_run_id": self.mlflow_run_id,
            "model_id": self.model_id,
            # GAP 1, 2: DVC and lineage
            "dvc_tag": self.dvc_tag,
            "dvc_pushed": self.dvc_pushed,
            "feature_order_hash": self.feature_order_hash,
            "lineage_record": self.lineage_record,
            # Reward system (CTR-REWARD-SNAPSHOT-001)
            "reward_contract_id": self.reward_contract_id,
            "reward_config_hash": self.reward_config_hash,
            "reward_config_uri": self.reward_config_uri,
            "curriculum_final_phase": self.curriculum_final_phase,
            "reward_weights": self.reward_weights,
            "errors": self.errors,
        }


# =============================================================================
# Training Engine
# =============================================================================

class TrainingEngine:
    """
    Unified training engine.

    This is the ONLY class that orchestrates training.
    All entry points (DAG, CLI) use this engine.

    Responsibilities:
        1. Dataset validation
        2. Norm stats generation
        3. Contract creation
        4. Environment creation
        5. PPO training
        6. MLflow logging
        7. Model registration
    """

    def __init__(self, project_root: Path):
        """
        Initialize training engine.

        Args:
            project_root: Project root directory
        """
        self.project_root = Path(project_root)
        self._mlflow = None
        self._mlflow_run_id = None

        # GAP 1: DVC service
        self._dvc_service = None
        self._dvc_tag = None

        # GAP 2: Lineage tracker
        self._lineage_tracker = None
        self._lineage_record = None

    def _get_date_ranges(self) -> Dict[str, str]:
        """
        Get date ranges from SSOT config for train/test split.

        Returns:
            Dict with train_end, test_start dates from experiment_ssot.yaml
        """
        try:
            from src.config.experiment_loader import load_experiment_config
            config = load_experiment_config()
            date_ranges = config.pipeline.date_ranges if hasattr(config.pipeline, 'date_ranges') else {}

            # Handle both dict and object access
            if hasattr(date_ranges, 'train_end'):
                return {
                    'train_end': date_ranges.train_end,
                    'test_start': date_ranges.test_start,
                }
            elif isinstance(date_ranges, dict):
                return {
                    'train_end': date_ranges.get('train_end', '2024-12-31'),
                    'test_start': date_ranges.get('test_start', '2025-01-01'),
                }
            else:
                return {
                    'train_end': '2024-12-31',
                    'test_start': '2025-01-01',
                }
        except Exception as e:
            logger.warning(f"Could not load date_ranges from SSOT: {e}")
            return {
                'train_end': '2024-12-31',
                'test_start': '2025-01-01',
            }

    def run(self, request: TrainingRequest) -> TrainingResult:
        """
        Execute complete training pipeline.

        This is the SINGLE entry point for all training.

        Args:
            request: Training request with all parameters

        Returns:
            TrainingResult with all outputs
        """
        start_time = time.time()
        errors = []

        logger.info("=" * 60)
        logger.info(f"TRAINING ENGINE - Starting v{request.version}")
        logger.info("=" * 60)

        # REPRODUCIBILITY: Set seeds BEFORE any stochastic operations
        set_reproducible_seeds(request.seed)
        logger.info(f"[REPRODUCIBILITY] Seed={request.seed} set for all RNGs")

        try:
            # Step 1: Validate dataset
            self._report_progress(request, "validate_dataset", 1, 9)
            df, dataset_hash = self._validate_dataset(request)

            # Step 1b: GAP 1 - DVC versioning of dataset
            dvc_tag_str = None
            dvc_pushed = False
            if request.dvc_enabled and DVC_AVAILABLE:
                dvc_result = self._version_dataset_with_dvc(request, dataset_hash)
                if dvc_result:
                    dvc_tag_str = dvc_result.get("tag")
                    dvc_pushed = dvc_result.get("pushed", False)

            # Step 2: Generate norm stats
            self._report_progress(request, "generate_norm_stats", 2, 9)
            norm_stats_path, norm_stats_hash = self._generate_norm_stats(
                df, request
            )

            # Step 3: Create contract
            self._report_progress(request, "create_contract", 3, 9)
            contract_path, config_hash = self._create_contract(
                request, norm_stats_path, norm_stats_hash, len(df)
            )

            # Step 4: Initialize MLflow
            self._report_progress(request, "init_mlflow", 4, 9)
            if request.mlflow_enabled:
                self._init_mlflow(request, dataset_hash, norm_stats_hash, config_hash, dvc_tag_str)

            # Step 5: Train model
            self._report_progress(request, "train_model", 5, 9)
            train_result = self._train_model(
                request, df, norm_stats_path
            )

            # Step 6: Register model
            self._report_progress(request, "register_model", 6, 9)
            model_id = None
            if request.auto_register:
                model_id = self._register_model(request, train_result)

            # Step 7: GAP 2 - Create lineage record
            self._report_progress(request, "track_lineage", 7, 9)
            lineage_record_dict = None
            if request.lineage_enabled and LINEAGE_AVAILABLE:
                lineage_record_dict = self._track_lineage(
                    request, dataset_hash, norm_stats_hash, config_hash,
                    train_result, dvc_tag_str
                )

            # Step 8: GAP 5 - Log experiment config to MLflow
            self._report_progress(request, "log_artifacts", 8, 9)
            if request.mlflow_enabled and self._mlflow:
                self._log_experiment_config_artifact(request)

            # Step 9: Finalize
            self._report_progress(request, "finalize", 9, 9)
            if request.mlflow_enabled and self._mlflow:
                self._finalize_mlflow(train_result)

            duration = time.time() - start_time

            # Compute reward config hash
            reward_config_hash = None
            reward_weights = None
            curriculum_final_phase = None
            if request.reward_config:
                import hashlib
                import json
                from dataclasses import asdict
                try:
                    config_json = json.dumps(asdict(request.reward_config), sort_keys=True)
                    reward_config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]
                except:
                    pass

            # Get curriculum phase and weights from train_result if available
            if hasattr(train_result, 'curriculum_final_phase'):
                curriculum_final_phase = train_result.curriculum_final_phase
            if hasattr(train_result, 'reward_weights'):
                reward_weights = train_result.reward_weights

            result = TrainingResult(
                success=True,
                version=request.version,
                model_path=train_result.model_path,
                norm_stats_path=norm_stats_path,
                contract_path=contract_path,
                model_hash=train_result.model_hash,
                dataset_hash=dataset_hash,
                norm_stats_hash=norm_stats_hash,
                config_hash=config_hash,
                training_duration_seconds=duration,
                total_timesteps=train_result.total_timesteps,
                best_mean_reward=train_result.best_mean_reward,
                final_mean_reward=train_result.final_mean_reward,
                training_seed=request.seed,
                mlflow_run_id=self._mlflow_run_id,
                model_id=model_id,
                # GAP 1, 2: DVC and lineage
                dvc_tag=dvc_tag_str,
                dvc_pushed=dvc_pushed,
                feature_order_hash=FEATURE_ORDER_HASH,
                lineage_record=lineage_record_dict,
                # Reward system (CTR-REWARD-SNAPSHOT-001)
                reward_contract_id=request.reward_contract_id,
                reward_config_hash=reward_config_hash,
                curriculum_final_phase=curriculum_final_phase,
                reward_weights=reward_weights,
                errors=errors,
            )

            # Save result
            self._save_result(request, result)

            logger.info("=" * 60)
            logger.info(f"TRAINING ENGINE - Complete in {duration/60:.1f}min")
            logger.info(f"  Model: {result.model_path}")
            logger.info(f"  Best reward: {result.best_mean_reward:.2f}")
            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            errors.append(str(e))

            if request.mlflow_enabled and self._mlflow:
                try:
                    self._mlflow.set_tag("status", "FAILED")
                    self._mlflow.set_tag("error", str(e)[:250])
                    self._mlflow.end_run()
                except:
                    pass

            return TrainingResult(
                success=False,
                version=request.version,
                training_duration_seconds=time.time() - start_time,
                errors=errors,
            )

    # =========================================================================
    # Private Methods - Each step of the pipeline
    # =========================================================================

    def _report_progress(
        self,
        request: TrainingRequest,
        stage: str,
        current: int,
        total: int
    ) -> None:
        """Report progress via callback."""
        logger.info(f"[{current}/{total}] {stage}")
        if request.on_progress:
            request.on_progress(stage, current, total)

    def _validate_dataset(
        self,
        request: TrainingRequest
    ) -> tuple[pd.DataFrame, str]:
        """
        Validate dataset exists and compute hash.

        Returns:
            Tuple of (dataframe, dataset_hash)
        """
        if not request.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {request.dataset_path}")

        # Support both CSV and Parquet formats
        if str(request.dataset_path).endswith('.parquet'):
            df = pd.read_parquet(request.dataset_path)
        else:
            df = pd.read_csv(request.dataset_path)

        # Get feature columns from SSOT or request
        feature_columns = request.feature_columns
        if not feature_columns:
            # Use SSOT feature order (already loaded at module level)
            if FEATURE_ORDER:
                feature_columns = list(FEATURE_ORDER)
            else:
                raise ValueError("No feature_columns provided and SSOT not available")

        # Exclude runtime state features (added by environment, not in dataset)
        # Read state features from SSOT or use defaults
        try:
            from src.config.experiment_loader import load_experiment_config
            ssot = load_experiment_config()
            state_features = [f.name for f in ssot.features if f.is_state]
        except (ImportError, FileNotFoundError):
            state_features = ["position", "unrealized_pnl", "time_normalized"]
        market_features = [c for c in feature_columns if c not in state_features]

        # Validate only market columns exist (state features are added at runtime)
        missing = [c for c in market_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Compute hash
        with open(request.dataset_path, 'rb') as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()

        logger.info(f"Dataset validated: {len(df):,} rows, hash={dataset_hash[:12]}...")

        return df, dataset_hash

    def _find_l2_norm_stats(self, dataset_path: Path) -> Optional[Path]:
        """
        Find L2-generated norm_stats adjacent to dataset.

        L2 dataset builder now generates {dataset_prefix}_norm_stats.json
        alongside each dataset (e.g., DS_v3_close_only_norm_stats.json).

        Search order:
        1. {dataset_dir}/{base}_norm_stats.json (stripping _train/_val/_test suffix)
        2. {dataset_dir}/{dataset_stem}_norm_stats.json (exact match)
        3. {dataset_dir}/{dataset_stem.replace('RL_', '')}_norm_stats.json
        4. {dataset_dir}/norm_stats.json

        Args:
            dataset_path: Path to the dataset file

        Returns:
            Path to norm_stats.json if found, None otherwise
        """
        dataset_dir = dataset_path.parent
        dataset_stem = dataset_path.stem

        # Strip _train/_val/_test suffix to get base name
        # E.g., DS_v3_close_only_train -> DS_v3_close_only
        base_stem = dataset_stem
        for suffix in ['_train', '_val', '_test']:
            if base_stem.endswith(suffix):
                base_stem = base_stem[:-len(suffix)]
                break

        candidates = []

        # Pattern 1: Base name without _train/_val/_test (MOST COMMON for L2)
        # E.g., DS_v3_close_only_train.parquet -> DS_v3_close_only_norm_stats.json
        candidates.append(dataset_dir / f"{base_stem}_norm_stats.json")

        # Pattern 2: RL_ prefix stripping
        if dataset_stem.startswith('RL_'):
            candidates.append(dataset_dir / f"{dataset_stem[3:]}_norm_stats.json")

        # Pattern 3: Direct match with full dataset stem
        candidates.append(dataset_dir / f"{dataset_stem}_norm_stats.json")

        # Pattern 4: Generic norm_stats.json in same directory
        candidates.append(dataset_dir / "norm_stats.json")

        for p in candidates:
            if p.exists():
                logger.info(f"[L2->L3] Found L2 norm_stats: {p}")
                return p

        logger.debug(f"No L2 norm_stats found for {dataset_path}. Searched: {[str(c) for c in candidates]}")
        return None

    def _generate_norm_stats(
        self,
        df: pd.DataFrame,
        request: TrainingRequest
    ) -> tuple[Path, str]:
        """
        Generate normalization statistics.

        FIX: First check if L2 already provides norm_stats to avoid double normalization.
        L2 computes stats from raw data, so we should use those instead of recomputing
        from potentially pre-normalized data.

        Returns:
            Tuple of (norm_stats_path, norm_stats_hash)
        """
        # Get feature columns (exclude state features)
        feature_columns = request.feature_columns
        if not feature_columns and FEATURE_ORDER:
            feature_columns = list(FEATURE_ORDER)

        # Get state features from SSOT
        try:
            from src.config.experiment_loader import load_experiment_config
            ssot = load_experiment_config()
            state_features = [f.name for f in ssot.features if f.is_state]
        except (ImportError, FileNotFoundError):
            state_features = ["position", "unrealized_pnl", "time_normalized"]
        market_features = [c for c in feature_columns if c not in state_features]

        # FIX: Use new method to find L2 norm_stats
        l2_norm_stats_path = self._find_l2_norm_stats(request.dataset_path)

        if l2_norm_stats_path is not None and l2_norm_stats_path.exists():
            logger.info(f"[L2->L3] Using L2 norm_stats from: {l2_norm_stats_path}")
            with open(l2_norm_stats_path, 'r') as f:
                l2_stats = json.load(f)

            # Convert L2 format to engine format
            norm_stats = {}
            # L2 format can have features:
            #   1. Nested under "features" key (L2DatasetBuilder v1.0+)
            #   2. Directly at top level (legacy format)
            if "features" in l2_stats and isinstance(l2_stats["features"], dict):
                # New format: features nested under "features" key
                l2_features = l2_stats["features"]
                logger.info(f"[L2->L3] Using nested 'features' key from L2 norm_stats")
            else:
                # Legacy format: features directly at top level
                # Skip metadata keys (start with "_")
                l2_features = {k: v for k, v in l2_stats.items() if not k.startswith("_")}

            for col in market_features:
                if col in l2_features:
                    stats = l2_features[col]
                    norm_stats[col] = {
                        "mean": float(stats.get("mean", 0.0)),
                        "std": float(stats.get("std", 1.0)),
                        "min": float(stats.get("min", -5.0)),
                        "max": float(stats.get("max", 5.0)),
                    }
                elif col in df.columns:
                    # Fallback for features not in L2 stats
                    logger.warning(f"Feature '{col}' not in L2 norm_stats, computing from data")
                    values = df[col].dropna()
                    norm_stats[col] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }

            logger.info(f"[L2->L3] Loaded {len(norm_stats)} features from L2 norm_stats (avoiding double normalization)")
        else:
            logger.warning(f"[L2->L3] L2 norm_stats not found, computing from data (this may cause normalization mismatch!)")
            # Calculate stats from data (original behavior)
            norm_stats = {}
            for col in market_features:
                if col in df.columns:
                    values = df[col].dropna()
                    norm_stats[col] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }

        # Add metadata
        norm_stats["_metadata"] = {
            "version": request.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(market_features),
            "sample_count": len(df),
        }

        # Determine output path
        output_dir = request.output_dir or (
            self.project_root / "models" / f"ppo_{request.version}_production"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        norm_stats_path = output_dir / "norm_stats.json"

        # Also save to config/ for compatibility
        config_path = self.project_root / "config" / f"{request.version}_norm_stats.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        for path in [norm_stats_path, config_path]:
            with open(path, 'w') as f:
                json.dump(norm_stats, f, indent=2)

        # Compute hash
        canonical = json.dumps(norm_stats, sort_keys=True, separators=(',', ':'))
        norm_stats_hash = hashlib.sha256(canonical.encode()).hexdigest()

        logger.info(f"Norm stats generated: {len(market_features)} features, hash={norm_stats_hash[:12]}...")

        return norm_stats_path, norm_stats_hash

    def _create_contract(
        self,
        request: TrainingRequest,
        norm_stats_path: Path,
        norm_stats_hash: str,
        sample_count: int
    ) -> tuple[Path, str]:
        """
        Create feature contract.

        Returns:
            Tuple of (contract_path, config_hash)
        """
        feature_columns = request.feature_columns
        if not feature_columns and FEATURE_ORDER:
            feature_columns = list(FEATURE_ORDER)

        contract = {
            "version": request.version,
            "observation_dim": len(feature_columns),
            "feature_order": feature_columns,
            "norm_stats_path": str(norm_stats_path.relative_to(self.project_root)),
            "model_path": f"models/ppo_{request.version}_production/final_model.zip",
            "rsi_period": INDICATOR_CONFIG.rsi_period,
            "atr_period": INDICATOR_CONFIG.atr_period,
            "adx_period": INDICATOR_CONFIG.adx_period,
            "warmup_bars": INDICATOR_CONFIG.warmup_bars,
            "trading_hours_start": "13:00",
            "trading_hours_end": "17:55",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "norm_stats_hash": norm_stats_hash,
            "sample_count": sample_count,
        }

        # Compute hash
        canonical = json.dumps(contract, sort_keys=True, separators=(',', ':'))
        config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        contract["contract_hash"] = config_hash

        # Save
        contracts_dir = self.project_root / "config" / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)
        contract_path = contracts_dir / f"{request.version}_contract.json"

        with open(contract_path, 'w') as f:
            json.dump(contract, f, indent=2)

        logger.info(f"Contract created: {contract_path}")

        return contract_path, config_hash

    def _init_mlflow(
        self,
        request: TrainingRequest,
        dataset_hash: str,
        norm_stats_hash: str,
        config_hash: str,
        dvc_tag: Optional[str] = None
    ) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow
            self._mlflow = mlflow

            tracking_uri = request.mlflow_tracking_uri or os.environ.get(
                "MLFLOW_TRACKING_URI", "http://localhost:5000"
            )
            mlflow.set_tracking_uri(tracking_uri)

            experiment_name = request.experiment_name or "usdcop-rl-training"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

            # Start run
            run_name = f"train_{request.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = mlflow.start_run(run_name=run_name)
            self._mlflow_run_id = run.info.run_id

            # Log parameters - use fresh hyperparams from SSOT
            hyperparams = get_ppo_hyperparameters(force_reload=True)
            params = {
                "version": request.version,
                "learning_rate": request.learning_rate or hyperparams.learning_rate,
                "gamma": hyperparams.gamma,
                "ent_coef": hyperparams.ent_coef,
                "total_timesteps": request.total_timesteps or hyperparams.total_timesteps,
                "dataset_hash": dataset_hash[:16],
                "norm_stats_hash": norm_stats_hash[:16],
                "config_hash": config_hash,
            }

            # GAP 2: Log dvc_tag to MLflow for lineage tracing
            if dvc_tag:
                params["dvc_tag"] = dvc_tag
                mlflow.set_tag("dvc.tag", dvc_tag)

            # GAP 3: Log feature_order_hash for contract validation
            if FEATURE_ORDER_HASH:
                params["feature_order_hash"] = FEATURE_ORDER_HASH[:16]
                mlflow.set_tag("contract.feature_order_hash", FEATURE_ORDER_HASH)

            mlflow.log_params(params)

            # Log git info
            self._log_git_info()

            logger.info(f"MLflow initialized: run_id={self._mlflow_run_id}")
            if dvc_tag:
                logger.info(f"[GAP 2] DVC tag logged to MLflow: {dvc_tag}")

        except ImportError:
            logger.warning("MLflow not available")
            self._mlflow = None
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}")
            self._mlflow = None

    def _log_git_info(self) -> None:
        """Log git info to MLflow."""
        if not self._mlflow:
            return

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=str(self.project_root)
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                self._mlflow.log_param("git_commit", commit[:12])
                self._mlflow.set_tag("git.commit", commit)
        except:
            pass

    def _train_model(
        self,
        request: TrainingRequest,
        df: pd.DataFrame,
        norm_stats_path: Path
    ) -> Any:
        """
        Execute PPO training.

        Returns:
            TrainingResult from PPOTrainer
        """
        from .environments import EnvironmentFactory, TradingEnvConfig
        from .environments.trading_env import ModularRewardStrategyAdapter
        from .trainers import PPOTrainer, PPOConfig

        # Get feature columns from SSOT or request
        feature_columns = request.feature_columns
        if not feature_columns and FEATURE_ORDER:
            feature_columns = list(FEATURE_ORDER)

        # Create output directory
        output_dir = request.output_dir or (
            self.project_root / "models" / f"ppo_{request.version}_production"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create environment factory
        env_factory = EnvironmentFactory(project_root=self.project_root)

        # Create reward strategy with modular reward system
        reward_config = request.reward_config or RewardConfig()
        reward_strategy = ModularRewardStrategyAdapter(
            reward_config=reward_config,
            enable_curriculum=request.enable_curriculum,
        )
        self._reward_strategy = reward_strategy  # Store for later access

        # Log reward config info
        logger.info(f"Reward system: contract={request.reward_contract_id}, "
                   f"curriculum={'enabled' if request.enable_curriculum else 'disabled'}")

        # Environment config - force reload from SSOT to get latest settings
        env_ssot = get_environment_config(force_reload=True)
        logger.info(f"[SSOT] Environment config: max_episode_steps={env_ssot.max_episode_steps}")

        env_config = TradingEnvConfig(
            observation_dim=len(feature_columns),
            initial_balance=getattr(env_ssot, 'initial_capital', 10_000.0),
            transaction_cost_bps=env_ssot.transaction_cost_bps,
            episode_length=env_ssot.max_episode_steps,
        )

        # Get date ranges from SSOT for proper train/test split
        date_ranges = self._get_date_ranges()
        logger.info(f"[SSOT] Date ranges: train_end={date_ranges['train_end']}, test_start={date_ranges['test_start']}")

        # Create environments with date-based splits
        env_dict = env_factory.create_train_eval_envs(
            dataset_path=request.dataset_path,
            norm_stats_path=norm_stats_path,
            config=env_config,
            date_ranges=date_ranges,  # NEW: date-based split from SSOT
            train_ratio=DATA_SPLIT_CONFIG.train_ratio,  # Fallback if date_ranges fails
            val_ratio=DATA_SPLIT_CONFIG.val_ratio,
            n_train_envs=1,
            n_eval_envs=1,
        )

        train_env = env_dict["train"]
        eval_env = env_dict["val"]

        logger.info(
            f"Environments: train={env_dict['splits']['train_size']} bars, "
            f"val={env_dict['splits']['val_size']} bars"
        )

        # PPO config - DRY: Use from_ssot() to read all defaults from SSOT
        total_timesteps = request.total_timesteps or get_ppo_hyperparameters().total_timesteps

        # Build overrides from request
        overrides = {
            "total_timesteps": total_timesteps,
            "eval_freq": max(total_timesteps // 20, 10000),
            "n_eval_episodes": 5,
            "checkpoint_freq": max(total_timesteps // 10, 25000),
            "tensorboard_log": False,  # Disabled until tensorboard is properly installed
            "verbose": 1,
        }
        if request.learning_rate:
            overrides["learning_rate"] = request.learning_rate

        ppo_config = PPOConfig.from_ssot(**overrides)

        # Resolve algorithm name from pipeline SSOT if available
        try:
            from src.training.algorithm_factory import resolve_algorithm_name
            from src.config.pipeline_config import load_pipeline_config
            pipeline_cfg = load_pipeline_config()
            ppo_config.algorithm_name = resolve_algorithm_name(pipeline_cfg)
        except Exception:
            pass  # Keep default "ppo"

        logger.info(f"[SSOT] PPO hyperparameters: lr={ppo_config.learning_rate}, ent_coef={ppo_config.ent_coef}")
        logger.info(f"[SSOT] Early stopping: enabled={ppo_config.early_stopping_enabled}, patience={ppo_config.early_stopping_patience}")
        logger.info(f"[SSOT] LR decay: enabled={ppo_config.lr_decay_enabled}")

        # Create trainer
        trainer = PPOTrainer(
            train_env=train_env,
            eval_env=eval_env,
            config=ppo_config,
            output_dir=output_dir,
            experiment_name=request.experiment_name or f"ppo_{request.version}",
        )

        # Train
        logger.info(f"Training for {total_timesteps:,} timesteps...")
        result = trainer.train()

        # Cleanup
        train_env.close()
        eval_env.close()

        # Log to MLflow
        if self._mlflow and result.success:
            self._mlflow.log_metrics({
                "training_duration_seconds": result.training_duration_seconds,
                "best_mean_reward": result.best_mean_reward,
                "final_mean_reward": result.final_mean_reward,
                "total_timesteps": result.total_timesteps,
            })

            if result.model_path and result.model_path.exists():
                self._mlflow.log_artifact(str(result.model_path))

            # Log reward system configuration
            self._mlflow.set_tag("reward.contract_id", request.reward_contract_id)
            self._mlflow.set_tag("reward.curriculum_enabled", str(request.enable_curriculum))

            # Log curriculum final phase if available
            if hasattr(self, '_reward_strategy') and self._reward_strategy:
                curriculum_phase = self._reward_strategy.curriculum_phase
                if curriculum_phase:
                    self._mlflow.set_tag("reward.curriculum_final_phase", curriculum_phase)

        # Store curriculum phase in result for downstream use
        if hasattr(self, '_reward_strategy') and self._reward_strategy:
            result.curriculum_final_phase = self._reward_strategy.curriculum_phase
            result.reward_weights = {
                "pnl": reward_config.weight_pnl,
                "dsr": reward_config.weight_dsr,
                "sortino": reward_config.weight_sortino,
                "regime_penalty": reward_config.weight_regime_penalty,
                "holding_decay": reward_config.weight_holding_decay,
                "anti_gaming": reward_config.weight_anti_gaming,
            }

        return result

    def _register_model(
        self,
        request: TrainingRequest,
        train_result: Any
    ) -> Optional[str]:
        """Register model in database."""
        if not request.db_connection_string:
            logger.info("No DB connection, skipping registration")
            return None

        try:
            import psycopg2

            model_hash = train_result.model_hash or "unknown"
            model_id = f"ppo_{request.version}_{model_hash[:8]}"

            conn = psycopg2.connect(request.db_connection_string)
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO model_registry (
                    model_id, model_version, model_path, model_hash,
                    status, created_at
                ) VALUES (%s, %s, %s, %s, 'registered', NOW())
                ON CONFLICT (model_id) DO UPDATE SET
                    model_path = EXCLUDED.model_path,
                    status = 'registered'
            """, (
                model_id,
                request.version,
                str(train_result.model_path),
                model_hash,
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Model registered: {model_id}")
            return model_id

        except Exception as e:
            logger.warning(f"Registration failed: {e}")
            return None

    def _finalize_mlflow(self, train_result: Any) -> None:
        """Finalize MLflow run."""
        if not self._mlflow:
            return

        try:
            status = "FINISHED" if train_result.success else "FAILED"
            self._mlflow.set_tag("status", status)
            self._mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow finalize failed: {e}")

    # =========================================================================
    # GAP 1, 2, 5: DVC, Lineage, and Artifact Methods
    # =========================================================================

    def _version_dataset_with_dvc(
        self,
        request: TrainingRequest,
        dataset_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        GAP 1: Version dataset with DVC and push to remote.

        Creates a semantic DVC tag and pushes the dataset to MinIO.

        Args:
            request: Training request
            dataset_hash: SHA256 hash of the dataset

        Returns:
            Dict with tag and push status, or None on failure
        """
        if not DVC_AVAILABLE or not create_dvc_service:
            logger.debug("DVC not available, skipping versioning")
            return None

        try:
            dvc_service = create_dvc_service(project_root=self.project_root)

            # Create semantic tag
            experiment_name = request.experiment_name or f"train_{request.version}"
            dvc_tag = DVCTag.for_experiment(
                experiment_name=experiment_name,
                version=request.version,
            )

            # Add and push dataset
            result = dvc_service.add_and_push(
                path=request.dataset_path,
                tag=dvc_tag,
                message=f"Training dataset for {request.version}",
                remote=request.dvc_remote,
            )

            if result.success:
                logger.info(f"[GAP 1] Dataset versioned with DVC: tag={str(dvc_tag)}")
                return {
                    "tag": str(dvc_tag),
                    "pushed": result.pushed,
                    "dvc_file": str(result.dvc_file) if result.dvc_file else None,
                }
            else:
                logger.warning(f"[GAP 1] DVC versioning failed: {result.error}")
                return None

        except Exception as e:
            logger.warning(f"[GAP 1] DVC versioning error: {e}")
            return None

    def _track_lineage(
        self,
        request: TrainingRequest,
        dataset_hash: str,
        norm_stats_hash: str,
        config_hash: str,
        train_result: Any,
        dvc_tag: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        GAP 2: Track complete lineage for this training run.

        Creates a lineage record that connects:
        - Dataset hash
        - Norm stats hash
        - Config hash
        - Model hash
        - DVC tag
        - Feature order hash
        - MLflow run ID

        Args:
            request: Training request
            dataset_hash: Hash of the dataset
            norm_stats_hash: Hash of norm_stats.json
            config_hash: Hash of the training config/contract
            train_result: Result from PPO training
            dvc_tag: DVC tag if created

        Returns:
            Lineage record as dict, or None on failure
        """
        if not LINEAGE_AVAILABLE or not create_lineage_tracker:
            logger.debug("LineageTracker not available, skipping")
            return None

        try:
            tracker = create_lineage_tracker(project_root=self.project_root)

            lineage = tracker.track_experiment(
                experiment_name=request.experiment_name or f"train_{request.version}",
                config_path=request.experiment_config_path,
                dataset_path=request.dataset_path,
                version=request.version,
                stage="L3_training",
            )

            # Enrich with training results
            lineage.model_hash = train_result.model_hash if hasattr(train_result, 'model_hash') else None
            lineage.mlflow_run_id = self._mlflow_run_id
            lineage.dvc_tag = dvc_tag
            lineage.dataset_hash = dataset_hash
            lineage.norm_stats_hash = norm_stats_hash
            lineage.config_hash = config_hash
            lineage.feature_order_hash = FEATURE_ORDER_HASH

            # Log to MLflow if available
            if self._mlflow and request.mlflow_enabled:
                tracker.log_to_mlflow(lineage, log_artifacts=True)

            logger.info(f"[GAP 2] Lineage tracked: run_id={lineage.run_id}")
            return lineage.to_dict()

        except Exception as e:
            logger.warning(f"[GAP 2] Lineage tracking error: {e}")
            return None

    def _log_experiment_config_artifact(self, request: TrainingRequest) -> None:
        """
        GAP 5: Log experiment_config.yaml as MLflow artifact.

        This ensures the experiment configuration is preserved
        alongside the model for reproducibility.

        Args:
            request: Training request
        """
        if not self._mlflow:
            return

        if not request.experiment_config_path:
            logger.debug("[GAP 5] No experiment_config_path provided, skipping")
            return

        try:
            config_path = Path(request.experiment_config_path)
            if not config_path.is_absolute():
                config_path = self.project_root / config_path

            if config_path.exists():
                self._mlflow.log_artifact(str(config_path), artifact_path="config")
                logger.info(f"[GAP 5] Experiment config logged as artifact: {config_path.name}")

                # Also log the config hash as a tag
                with open(config_path, 'rb') as f:
                    exp_config_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                self._mlflow.set_tag("experiment.config_hash", exp_config_hash)
                self._mlflow.set_tag("experiment.config_path", str(config_path))
            else:
                logger.warning(f"[GAP 5] Experiment config not found: {config_path}")

        except Exception as e:
            logger.warning(f"[GAP 5] Error logging experiment config: {e}")

    def _save_result(self, request: TrainingRequest, result: TrainingResult) -> None:
        """Save training result to JSON."""
        output_dir = request.output_dir or (
            self.project_root / "models" / f"ppo_{request.version}_production"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = output_dir / "training_result.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Result saved: {result_path}")


# =============================================================================
# Convenience Function
# =============================================================================

def run_training(
    project_root: Path,
    version: str,
    dataset_path: Path,
    total_timesteps: Optional[int] = None,
    mlflow_enabled: bool = True,
    db_connection_string: Optional[str] = None,
    **kwargs
) -> TrainingResult:
    """
    Convenience function to run training.

    Example:
        result = run_training(
            project_root=Path("."),
            version="v1",
            dataset_path=Path("data/training.csv"),
            total_timesteps=1_000_000,
        )
    """
    request = TrainingRequest(
        version=version,
        dataset_path=Path(dataset_path),
        total_timesteps=total_timesteps,
        mlflow_enabled=mlflow_enabled,
        db_connection_string=db_connection_string,
        **kwargs
    )

    engine = TrainingEngine(project_root=project_root)
    return engine.run(request)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TrainingEngine",
    "TrainingRequest",
    "TrainingResult",
    "run_training",
]
