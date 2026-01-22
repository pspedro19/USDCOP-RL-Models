"""
Experiment Manager
==================

Unified orchestration for ML experiments using MinIO-first architecture.

This module provides:
- ExperimentManager: High-level experiment lifecycle management
- Dataset versioning and storage
- Model versioning and storage
- Backtest result storage
- A/B comparison storage
- Lineage tracking

Contract: CTR-EXP-MANAGER-001
- All artifacts stored in MinIO (s3://experiments/)
- Immutable snapshots for all versions
- Full lineage tracking

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.core.contracts.storage_contracts import (
    DatasetSnapshot,
    ModelSnapshot,
    BacktestSnapshot,
    ABComparisonSnapshot,
    LineageRecord,
    RewardConfigSnapshot,
)
from src.core.factories.storage_factory import (
    StorageFactory,
    get_dataset_repository,
    get_model_repository,
    get_backtest_repository,
    get_ab_comparison_repository,
)
from src.core.interfaces.storage import (
    IDatasetRepository,
    IModelRepository,
    IBacktestRepository,
    IABComparisonRepository,
    ObjectNotFoundError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXPERIMENT MANAGER
# =============================================================================


class ExperimentManager:
    """
    Unified experiment lifecycle manager.

    Provides high-level API for:
    - Creating and managing experiments
    - Versioning datasets and models
    - Running and storing backtests
    - A/B comparisons
    - Production promotion

    Example:
        >>> manager = ExperimentManager("baseline_v2")
        >>>
        >>> # Save dataset
        >>> dataset = manager.save_dataset(df, metadata={"date_range_start": "2023-01-01"})
        >>>
        >>> # Save model
        >>> model = manager.save_model(
        ...     model_path="models/ppo.zip",
        ...     norm_stats=norm_stats,
        ...     config=training_config,
        ... )
        >>>
        >>> # Save backtest
        >>> backtest = manager.save_backtest(
        ...     model_version=model.version,
        ...     result=backtest_result,
        ...     trades=trades_df,
        ...     equity_curve=equity_df,
        ... )
        >>>
        >>> # Promote to production
        >>> model_id = manager.promote_model(model.version)
    """

    def __init__(
        self,
        experiment_id: str,
        storage_factory: Optional[StorageFactory] = None,
    ):
        """
        Initialize experiment manager.

        Args:
            experiment_id: Unique experiment identifier
            storage_factory: Optional custom storage factory
        """
        self._experiment_id = experiment_id
        self._factory = storage_factory or StorageFactory.get_instance()

        # Repositories (lazy-loaded)
        self._dataset_repo: Optional[IDatasetRepository] = None
        self._model_repo: Optional[IModelRepository] = None
        self._backtest_repo: Optional[IBacktestRepository] = None
        self._ab_repo: Optional[IABComparisonRepository] = None

        # Current user for lineage
        self._user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))

        logger.info(f"ExperimentManager initialized for: {experiment_id}")

    @property
    def experiment_id(self) -> str:
        """Get experiment identifier."""
        return self._experiment_id

    @property
    def dataset_repo(self) -> IDatasetRepository:
        """Get dataset repository (lazy-loaded)."""
        if self._dataset_repo is None:
            self._dataset_repo = self._factory.create_dataset_repository()
        return self._dataset_repo

    @property
    def model_repo(self) -> IModelRepository:
        """Get model repository (lazy-loaded)."""
        if self._model_repo is None:
            self._model_repo = self._factory.create_model_repository()
        return self._model_repo

    @property
    def backtest_repo(self) -> IBacktestRepository:
        """Get backtest repository (lazy-loaded)."""
        if self._backtest_repo is None:
            self._backtest_repo = self._factory.create_backtest_repository()
        return self._backtest_repo

    @property
    def ab_repo(self) -> IABComparisonRepository:
        """Get A/B comparison repository (lazy-loaded)."""
        if self._ab_repo is None:
            self._ab_repo = self._factory.create_ab_comparison_repository()
        return self._ab_repo

    # =========================================================================
    # DATASET OPERATIONS
    # =========================================================================

    def save_dataset(
        self,
        data: pd.DataFrame,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None,
    ) -> DatasetSnapshot:
        """
        Save dataset to MinIO.

        Args:
            data: DataFrame to save
            version: Version string (auto-generated if None)
            metadata: Additional metadata
            parent_version: Previous version for lineage

        Returns:
            DatasetSnapshot with all URIs and hashes
        """
        if version is None:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        metadata = metadata or {}
        metadata["parent_version"] = parent_version

        # Create lineage record
        lineage = LineageRecord(
            artifact_type="dataset",
            artifact_id=f"{self._experiment_id}/{version}",
            parent_id=f"{self._experiment_id}/{parent_version}" if parent_version else None,
            parent_type="dataset" if parent_version else None,
            source_uri=None,
            source_hash=None,
            transform_name="l2_preprocessing",
            transform_params=tuple(metadata.items()) if metadata else None,
            created_at=datetime.utcnow(),
            created_by=self._user,
        )
        metadata["lineage"] = lineage.to_dict()

        snapshot = self.dataset_repo.save_dataset(
            experiment_id=self._experiment_id,
            data=data,
            version=version,
            metadata=metadata,
        )

        logger.info(f"Saved dataset {self._experiment_id}/{version}: {len(data)} rows")
        return snapshot

    def load_dataset(self, version: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from MinIO.

        Args:
            version: Specific version or None for latest

        Returns:
            DataFrame
        """
        return self.dataset_repo.load_dataset(self._experiment_id, version)

    def get_dataset_snapshot(self, version: Optional[str] = None) -> DatasetSnapshot:
        """
        Get dataset snapshot (metadata without data).

        Args:
            version: Specific version or None for latest

        Returns:
            DatasetSnapshot
        """
        return self.dataset_repo.get_snapshot(self._experiment_id, version)

    def list_dataset_versions(self) -> List[DatasetSnapshot]:
        """
        List all dataset versions.

        Returns:
            List of DatasetSnapshot, newest first
        """
        return self.dataset_repo.list_versions(self._experiment_id)

    def get_norm_stats(self, version: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get normalization statistics.

        Args:
            version: Specific version or None for latest

        Returns:
            Normalization stats dictionary
        """
        return self.dataset_repo.get_norm_stats(self._experiment_id, version)

    # =========================================================================
    # MODEL OPERATIONS
    # =========================================================================

    def save_model(
        self,
        model_path: Union[str, Path],
        norm_stats: Dict[str, Any],
        config: Dict[str, Any],
        dataset_version: Optional[str] = None,
        version: Optional[str] = None,
    ) -> ModelSnapshot:
        """
        Save trained model to MinIO.

        Args:
            model_path: Path to model file
            norm_stats: Normalization statistics
            config: Training configuration
            dataset_version: Version of dataset used for training
            version: Model version (auto-generated if None)

        Returns:
            ModelSnapshot with all URIs and hashes
        """
        if version is None:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Get dataset snapshot if version provided
        dataset_snapshot = None
        if dataset_version:
            try:
                dataset_snapshot = self.get_dataset_snapshot(dataset_version)
            except ObjectNotFoundError:
                logger.warning(f"Dataset version {dataset_version} not found for lineage")

        # Create lineage record
        lineage = LineageRecord(
            artifact_type="model",
            artifact_id=f"{self._experiment_id}/models/{version}",
            parent_id=f"{self._experiment_id}/datasets/{dataset_version}" if dataset_version else None,
            parent_type="dataset" if dataset_version else None,
            source_uri=dataset_snapshot.storage_uri if dataset_snapshot else None,
            source_hash=dataset_snapshot.data_hash if dataset_snapshot else None,
            transform_name="l3_training",
            transform_params=tuple(
                (k, str(v)) for k, v in config.items()
                if k in ["algorithm", "total_timesteps", "learning_rate"]
            ),
            created_at=datetime.utcnow(),
            created_by=self._user,
        )

        snapshot = self.model_repo.save_model(
            experiment_id=self._experiment_id,
            model_path=model_path,
            norm_stats=norm_stats,
            config=config,
            lineage=lineage,
            version=version,
        )

        logger.info(f"Saved model {self._experiment_id}/{version}")
        return snapshot

    def load_model(self, version: Optional[str] = None) -> bytes:
        """
        Load model bytes from MinIO.

        Args:
            version: Specific version or None for latest

        Returns:
            Raw model bytes
        """
        return self.model_repo.load_model(self._experiment_id, version)

    def get_model_snapshot(self, version: Optional[str] = None) -> ModelSnapshot:
        """
        Get model snapshot (metadata without model).

        Args:
            version: Specific version or None for latest

        Returns:
            ModelSnapshot
        """
        return self.model_repo.get_snapshot(self._experiment_id, version)

    def list_model_versions(self) -> List[ModelSnapshot]:
        """
        List all model versions.

        Returns:
            List of ModelSnapshot, newest first
        """
        return self.model_repo.list_versions(self._experiment_id)

    def promote_model(
        self,
        version: str,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Promote model to production bucket.

        This copies the model to s3://production/models/{model_id}/
        and returns the model_id for PostgreSQL registration.

        Args:
            version: Model version to promote
            model_id: Optional custom model ID

        Returns:
            Generated or provided model_id
        """
        model_id = self.model_repo.promote_to_production(
            self._experiment_id,
            version,
            model_id,
        )
        logger.info(f"Promoted model {self._experiment_id}/{version} to production as {model_id}")
        return model_id

    # =========================================================================
    # REWARD CONFIGURATION OPERATIONS
    # =========================================================================

    def save_reward_config(
        self,
        reward_config: "RewardConfig",
        contract_id: str,
        model_version: Optional[str] = None,
    ) -> RewardConfigSnapshot:
        """
        Save reward configuration to MinIO for lineage tracking.

        This method serializes the complete reward configuration used during
        training to enable:
        - Full reproducibility
        - Experiment comparison
        - Reward function auditing
        - A/B testing with different reward configs

        Args:
            reward_config: RewardConfig instance from training
            contract_id: Reward contract version (e.g., "v1.0.0")
            model_version: Optional version to link to model

        Returns:
            RewardConfigSnapshot with storage URI and hash
        """
        import hashlib
        import json
        from dataclasses import asdict

        # Serialize config
        config_dict = {
            "contract_id": contract_id,
            "weights": {
                "pnl": reward_config.weight_pnl,
                "dsr": reward_config.weight_dsr,
                "sortino": reward_config.weight_sortino,
                "regime_penalty": reward_config.weight_regime_penalty,
                "holding_decay": reward_config.weight_holding_decay,
                "anti_gaming": reward_config.weight_anti_gaming,
            },
            "flags": {
                "enable_normalization": reward_config.enable_normalization,
                "enable_curriculum": reward_config.enable_curriculum,
                "enable_banrep_detection": reward_config.banrep.enabled if hasattr(reward_config, 'banrep') else False,
                "enable_oil_tracking": reward_config.oil.enabled if hasattr(reward_config, 'oil') else False,
            },
            "components": {},
        }

        # Serialize each component config
        component_configs = []
        for component_name in ["dsr", "sortino", "regime", "market_impact",
                               "holding_decay", "anti_gaming", "curriculum",
                               "normalizer", "banrep", "oil"]:
            if hasattr(reward_config, component_name):
                comp_config = getattr(reward_config, component_name)
                try:
                    comp_dict = asdict(comp_config)
                    config_dict["components"][component_name] = comp_dict
                    component_configs.append((component_name, json.dumps(comp_dict)))
                except Exception:
                    # Handle non-dataclass components
                    pass

        # Compute hash
        config_json = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

        # Build version string
        version = model_version or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Save to MinIO
        key = f"{self._experiment_id}/reward_configs/{version}/reward_config.json"
        config_bytes = config_json.encode()

        from src.core.factories.storage_factory import StorageFactory
        storage = StorageFactory.get_instance().create_object_storage()
        storage.put_object("experiments", key, config_bytes, {
            "experiment_id": self._experiment_id,
            "version": version,
            "contract_id": contract_id,
            "config_hash": config_hash,
        })

        # Create snapshot
        snapshot = RewardConfigSnapshot(
            experiment_id=self._experiment_id,
            version=version,
            storage_uri=f"s3://experiments/{key}",
            reward_contract_id=contract_id,
            reward_config_hash=config_hash,
            weight_pnl=reward_config.weight_pnl,
            weight_dsr=reward_config.weight_dsr,
            weight_sortino=reward_config.weight_sortino,
            weight_regime_penalty=reward_config.weight_regime_penalty,
            weight_holding_decay=reward_config.weight_holding_decay,
            weight_anti_gaming=reward_config.weight_anti_gaming,
            enable_normalization=reward_config.enable_normalization,
            enable_curriculum=reward_config.enable_curriculum,
            enable_banrep_detection=reward_config.banrep.enabled if hasattr(reward_config, 'banrep') else False,
            enable_oil_tracking=reward_config.oil.enabled if hasattr(reward_config, 'oil') else False,
            curriculum_phase_1_steps=reward_config.curriculum.phase_1_steps if hasattr(reward_config, 'curriculum') else 0,
            curriculum_phase_2_steps=reward_config.curriculum.phase_2_steps if hasattr(reward_config, 'curriculum') else 0,
            curriculum_phase_3_steps=reward_config.curriculum.phase_3_steps if hasattr(reward_config, 'curriculum') else 0,
            component_configs=tuple(component_configs),
            created_at=datetime.utcnow(),
        )

        logger.info(f"Saved reward config {self._experiment_id}/{version}: hash={config_hash}")
        return snapshot

    def load_reward_config(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load reward configuration from MinIO.

        Args:
            version: Specific version or None for latest

        Returns:
            Reward configuration dictionary
        """
        import json

        from src.core.factories.storage_factory import StorageFactory
        storage = StorageFactory.get_instance().create_object_storage()

        if version is None:
            # Find latest version
            prefix = f"{self._experiment_id}/reward_configs/"
            objects = storage.list_objects("experiments", prefix)
            if not objects:
                raise ObjectNotFoundError(f"No reward configs found for {self._experiment_id}")
            # Get latest by sorting versions
            versions = sorted([obj.artifact_id.split("/")[-2] for obj in objects], reverse=True)
            version = versions[0]

        key = f"{self._experiment_id}/reward_configs/{version}/reward_config.json"
        config_bytes = storage.get_object("experiments", key)
        return json.loads(config_bytes.decode())

    def get_reward_config_snapshot(self, version: str) -> RewardConfigSnapshot:
        """
        Get reward config snapshot by version.

        Args:
            version: Version string

        Returns:
            RewardConfigSnapshot
        """
        config = self.load_reward_config(version)

        # Rebuild component configs tuple
        component_configs = []
        if "components" in config:
            import json
            for name, comp_dict in config["components"].items():
                component_configs.append((name, json.dumps(comp_dict)))

        return RewardConfigSnapshot(
            experiment_id=self._experiment_id,
            version=version,
            storage_uri=f"s3://experiments/{self._experiment_id}/reward_configs/{version}/reward_config.json",
            reward_contract_id=config.get("contract_id", "unknown"),
            reward_config_hash=config.get("config_hash", "unknown"),
            weight_pnl=config.get("weights", {}).get("pnl", 0.0),
            weight_dsr=config.get("weights", {}).get("dsr", 0.0),
            weight_sortino=config.get("weights", {}).get("sortino", 0.0),
            weight_regime_penalty=config.get("weights", {}).get("regime_penalty", 0.0),
            weight_holding_decay=config.get("weights", {}).get("holding_decay", 0.0),
            weight_anti_gaming=config.get("weights", {}).get("anti_gaming", 0.0),
            enable_normalization=config.get("flags", {}).get("enable_normalization", True),
            enable_curriculum=config.get("flags", {}).get("enable_curriculum", True),
            enable_banrep_detection=config.get("flags", {}).get("enable_banrep_detection", False),
            enable_oil_tracking=config.get("flags", {}).get("enable_oil_tracking", False),
            curriculum_phase_1_steps=config.get("components", {}).get("curriculum", {}).get("phase_1_steps", 0),
            curriculum_phase_2_steps=config.get("components", {}).get("curriculum", {}).get("phase_2_steps", 0),
            curriculum_phase_3_steps=config.get("components", {}).get("curriculum", {}).get("phase_3_steps", 0),
            component_configs=tuple(component_configs),
            created_at=datetime.fromisoformat(config.get("created_at", datetime.utcnow().isoformat())),
        )

    def list_reward_configs(self) -> List[str]:
        """
        List all reward config versions.

        Returns:
            List of version strings, newest first
        """
        from src.core.factories.storage_factory import StorageFactory
        storage = StorageFactory.get_instance().create_object_storage()

        prefix = f"{self._experiment_id}/reward_configs/"
        objects = storage.list_objects("experiments", prefix)

        # Extract unique versions
        versions = set()
        for obj in objects:
            parts = obj.artifact_id.split("/")
            if len(parts) >= 3:
                versions.add(parts[-2])

        return sorted(list(versions), reverse=True)

    # =========================================================================
    # BACKTEST OPERATIONS
    # =========================================================================

    def save_backtest(
        self,
        model_version: str,
        result: Dict[str, Any],
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        backtest_id: Optional[str] = None,
    ) -> BacktestSnapshot:
        """
        Save backtest results to MinIO.

        Args:
            model_version: Version of model used
            result: Backtest metrics dictionary
            trades: DataFrame of trades
            equity_curve: DataFrame of equity curve
            backtest_id: Optional custom ID

        Returns:
            BacktestSnapshot with all URIs
        """
        snapshot = self.backtest_repo.save_backtest(
            experiment_id=self._experiment_id,
            model_version=model_version,
            result=result,
            trades=trades,
            equity_curve=equity_curve,
            backtest_id=backtest_id,
        )

        logger.info(
            f"Saved backtest {self._experiment_id}/{snapshot.backtest_id}: "
            f"Sharpe={snapshot.sharpe_ratio:.2f}, Return={snapshot.total_return:.2%}"
        )
        return snapshot

    def load_backtest(self, backtest_id: str) -> Dict[str, Any]:
        """
        Load backtest result.

        Args:
            backtest_id: Backtest identifier

        Returns:
            Backtest result dictionary
        """
        return self.backtest_repo.load_backtest(self._experiment_id, backtest_id)

    def list_backtests(self) -> List[BacktestSnapshot]:
        """
        List all backtests.

        Returns:
            List of BacktestSnapshot, newest first
        """
        return self.backtest_repo.list_backtests(self._experiment_id)

    # =========================================================================
    # A/B COMPARISON OPERATIONS
    # =========================================================================

    def save_ab_comparison(
        self,
        baseline_experiment_id: str,
        baseline_version: str,
        treatment_version: str,
        result: Dict[str, Any],
        shadow_trades: Optional[pd.DataFrame] = None,
    ) -> ABComparisonSnapshot:
        """
        Save A/B comparison results.

        Args:
            baseline_experiment_id: Experiment ID of baseline model
            baseline_version: Version of baseline model
            treatment_version: Version of treatment model (from this experiment)
            result: Comparison metrics
            shadow_trades: Optional shadow trades DataFrame

        Returns:
            ABComparisonSnapshot
        """
        # Get model snapshots
        baseline_repo = ExperimentManager(baseline_experiment_id, self._factory).model_repo
        baseline_model = baseline_repo.get_snapshot(baseline_experiment_id, baseline_version)
        treatment_model = self.model_repo.get_snapshot(self._experiment_id, treatment_version)

        snapshot = self.ab_repo.save_comparison(
            experiment_id=self._experiment_id,
            baseline_model=baseline_model,
            treatment_model=treatment_model,
            result=result,
            shadow_trades=shadow_trades,
        )

        logger.info(
            f"Saved A/B comparison: {baseline_version} vs {treatment_version}, "
            f"recommendation={snapshot.recommendation}"
        )
        return snapshot

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_latest_dataset_uri(self) -> str:
        """Get S3 URI of latest dataset."""
        snapshot = self.get_dataset_snapshot()
        return snapshot.storage_uri

    def get_latest_model_uri(self) -> str:
        """Get S3 URI of latest model."""
        snapshot = self.get_model_snapshot()
        return snapshot.storage_uri

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary of experiment state.

        Returns:
            Dictionary with counts and latest versions
        """
        datasets = self.list_dataset_versions()
        models = self.list_model_versions()
        backtests = self.list_backtests()

        return {
            "experiment_id": self._experiment_id,
            "dataset_count": len(datasets),
            "model_count": len(models),
            "backtest_count": len(backtests),
            "latest_dataset": datasets[0].to_xcom_dict() if datasets else None,
            "latest_model": models[0].to_xcom_dict() if models else None,
            "latest_backtest": backtests[0].to_dict() if backtests else None,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_experiment(experiment_id: str) -> ExperimentManager:
    """
    Create new experiment manager.

    Args:
        experiment_id: Unique experiment identifier

    Returns:
        ExperimentManager instance
    """
    return ExperimentManager(experiment_id)


def get_experiment(experiment_id: str) -> ExperimentManager:
    """
    Get existing experiment manager.

    Args:
        experiment_id: Experiment identifier

    Returns:
        ExperimentManager instance
    """
    return ExperimentManager(experiment_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ExperimentManager",
    "create_experiment",
    "get_experiment",
]
