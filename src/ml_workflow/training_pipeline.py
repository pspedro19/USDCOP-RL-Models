"""
End-to-End Training Pipeline
============================
Orchestrates the complete training flow with automatic registration.

SOLID Principles:
- Single Responsibility: Each stage has one job
- Open/Closed: New stages via registration, not code changes
- Dependency Inversion: Depends on abstractions (contracts, registries)

Design Patterns:
- Pipeline Pattern: Sequential stages with clear interfaces
- Factory Pattern: Creates contracts and builders dynamically
- Observer Pattern: Callbacks for progress tracking
- Strategy Pattern: Different training strategies

This pipeline automates:
1. Dataset loading and validation
2. Norm stats calculation
3. Feature contract creation
4. Model training with callbacks
5. Model registration with hash verification
6. Backtest validation (optional)
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for a training run"""
    # Version and naming
    version: str  # e.g., "current"
    experiment_name: str = ""  # Optional descriptive name

    # Data
    dataset_path: Path = None
    feature_columns: List[str] = field(default_factory=list)
    state_features: List[str] = field(default_factory=lambda: ["position", "time_normalized"])

    # Technical indicators
    rsi_period: int = 9
    atr_period: int = 10
    adx_period: int = 14

    # Trading hours
    trading_hours_start: str = "13:00"
    trading_hours_end: str = "17:55"

    # Training hyperparameters
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.90  # From config/trading_config.yaml SSOT
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01

    # Environment
    initial_capital: float = 10_000.0
    transaction_cost_bps: float = 25.0

    # Output paths (auto-generated if not specified)
    model_output_dir: Path = None
    norm_stats_output_path: Path = None
    contract_output_path: Path = None

    # Database
    db_connection_string: str = None

    # Options
    auto_register: bool = True
    run_backtest_validation: bool = False
    backtest_start_date: str = None
    backtest_end_date: str = None

    def __post_init__(self):
        """Set default paths based on version"""
        if not self.experiment_name:
            self.experiment_name = f"ppo_{self.version}_{datetime.now().strftime('%Y%m%d')}"


@dataclass
class PipelineResult:
    """Result of a training pipeline run"""
    success: bool
    version: str
    model_id: Optional[str] = None
    model_path: Optional[Path] = None
    norm_stats_path: Optional[Path] = None
    contract_path: Optional[Path] = None

    # Hashes for integrity
    model_hash: Optional[str] = None
    norm_stats_hash: Optional[str] = None
    contract_hash: Optional[str] = None

    # Metrics
    training_duration_seconds: float = 0.0
    total_timesteps: int = 0
    best_mean_reward: float = 0.0

    # Backtest metrics (if run)
    backtest_sharpe: Optional[float] = None
    backtest_max_drawdown: Optional[float] = None
    backtest_win_rate: Optional[float] = None

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "version": self.version,
            "model_id": self.model_id,
            "model_path": str(self.model_path) if self.model_path else None,
            "norm_stats_path": str(self.norm_stats_path) if self.norm_stats_path else None,
            "contract_path": str(self.contract_path) if self.contract_path else None,
            "model_hash": self.model_hash,
            "norm_stats_hash": self.norm_stats_hash,
            "contract_hash": self.contract_hash,
            "training_duration_seconds": self.training_duration_seconds,
            "total_timesteps": self.total_timesteps,
            "best_mean_reward": self.best_mean_reward,
            "backtest_sharpe": self.backtest_sharpe,
            "backtest_max_drawdown": self.backtest_max_drawdown,
            "backtest_win_rate": self.backtest_win_rate,
            "errors": self.errors,
        }


# =============================================================================
# Pipeline Stages (Strategy Pattern)
# =============================================================================

class PipelineStage:
    """Base class for pipeline stages"""

    def __init__(self, name: str):
        self.name = name

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stage and return updated context"""
        raise NotImplementedError


class DatasetValidationStage(PipelineStage):
    """Validate dataset exists and has required columns"""

    def __init__(self):
        super().__init__("dataset_validation")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import pandas as pd

        config: TrainingConfig = context["config"]

        if not config.dataset_path or not config.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

        df = pd.read_csv(config.dataset_path)

        # Validate columns
        missing = [c for c in config.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        context["dataset"] = df
        context["sample_count"] = len(df)

        logger.info(f"✓ Dataset validated: {len(df)} samples, {len(config.feature_columns)} features")
        return context


class NormStatsGenerationStage(PipelineStage):
    """Calculate and save normalization statistics"""

    def __init__(self, project_root: Path):
        super().__init__("norm_stats_generation")
        self.project_root = project_root

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from .dynamic_contract_factory import NormStatsCalculator

        config: TrainingConfig = context["config"]
        df = context["dataset"]

        calculator = NormStatsCalculator()
        norm_stats = calculator.calculate_from_dataframe(
            df=df,
            feature_columns=config.feature_columns,
            exclude_state_features=config.state_features,
        )

        # Save
        output_path = config.norm_stats_output_path or (
            self.project_root / "config" / f"{config.version}_norm_stats.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        norm_stats_hash = calculator.save_to_json(norm_stats, output_path)

        context["norm_stats"] = norm_stats
        context["norm_stats_path"] = output_path
        context["norm_stats_hash"] = norm_stats_hash

        logger.info(f"✓ Norm stats generated: {len(norm_stats)} features")
        return context


class ContractCreationStage(PipelineStage):
    """Create feature contract"""

    def __init__(self, project_root: Path):
        super().__init__("contract_creation")
        self.project_root = project_root

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from .dynamic_contract_factory import DynamicFeatureContract

        config: TrainingConfig = context["config"]
        norm_stats_path = context["norm_stats_path"]

        contract = DynamicFeatureContract(
            version=config.version,
            observation_dim=len(config.feature_columns),
            feature_order=tuple(config.feature_columns),
            norm_stats_path=str(norm_stats_path.relative_to(self.project_root)),
            model_path=f"models/ppo_{config.version}_production/final_model.zip",
            rsi_period=config.rsi_period,
            atr_period=config.atr_period,
            adx_period=config.adx_period,
            warmup_bars=max(config.rsi_period, config.atr_period, config.adx_period),
            trading_hours_start=config.trading_hours_start,
            trading_hours_end=config.trading_hours_end,
            created_from_dataset=str(config.dataset_path),
            sample_count=context["sample_count"],
        )

        # Save contract
        contracts_dir = self.project_root / "config" / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)
        contract_path = contracts_dir / f"{config.version}_contract.json"

        with open(contract_path, 'w') as f:
            json.dump(contract.to_dict(), f, indent=2)

        context["contract"] = contract
        context["contract_path"] = contract_path
        context["contract_hash"] = contract.contract_hash

        logger.info(f"✓ Contract created: {config.version} ({contract.observation_dim} dims)")
        return context


class ModelTrainingStage(PipelineStage):
    """
    Train PPO model using the professional training infrastructure.

    Uses:
    - EnvironmentFactory: Creates vectorized training environments
    - PPOTrainer: Wraps Stable Baselines 3 PPO with clean interface
    - TradingEnvConfig: Environment configuration

    Design:
    - Single Responsibility: Only handles PPO training orchestration
    - Dependency Injection: Receives config through context
    - Factory Pattern: Uses EnvironmentFactory for env creation
    """

    def __init__(self, project_root: Path):
        super().__init__("model_training")
        self.project_root = project_root

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config: TrainingConfig = context["config"]
        contract = context["contract"]
        norm_stats_path = context["norm_stats_path"]

        # Create model output directory
        model_dir = config.model_output_dir or (
            self.project_root / "models" / f"ppo_{config.version}_production"
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating training environment with {contract.observation_dim} dims...")

        # Run training with real infrastructure
        training_result = self._run_training(
            config=config,
            dataset_path=config.dataset_path,
            norm_stats_path=norm_stats_path,
            output_dir=model_dir,
            observation_dim=contract.observation_dim,
        )

        # Extract results
        context["model_path"] = training_result.model_path
        context["training_duration_seconds"] = training_result.training_duration_seconds
        context["total_timesteps"] = training_result.total_timesteps
        context["best_mean_reward"] = training_result.best_mean_reward
        context["model_hash"] = training_result.model_hash

        if training_result.success:
            logger.info(
                f"✓ Model trained in {training_result.training_duration_seconds:.1f}s, "
                f"best_reward={training_result.best_mean_reward:.2f}"
            )
        else:
            logger.error(f"✗ Training failed: {training_result.error_message}")
            raise RuntimeError(f"Training failed: {training_result.error_message}")

        return context

    def _run_training(
        self,
        config: TrainingConfig,
        dataset_path: Path,
        norm_stats_path: Path,
        output_dir: Path,
        observation_dim: int,
    ):
        """
        Run actual model training using the professional training infrastructure.

        Uses:
        - EnvironmentFactory to create train/eval environments
        - PPOTrainer for clean training workflow
        - TradingEnvConfig for environment configuration
        """
        # Import training infrastructure
        try:
            from src.training import (
                EnvironmentFactory,
                TradingEnvConfig,
                PPOTrainer,
                PPOConfig,
            )
        except ImportError:
            # Try relative import
            from ..training import (
                EnvironmentFactory,
                TradingEnvConfig,
                PPOTrainer,
                PPOConfig,
            )

        # Create environment factory
        env_factory = EnvironmentFactory(project_root=self.project_root)

        # Configure environment
        env_config = TradingEnvConfig(
            observation_dim=observation_dim,
            initial_capital=config.initial_capital,
            transaction_cost_bps=config.transaction_cost_bps,
            random_episode_start=True,
            max_episode_steps=2000,
        )

        # Create train/eval environments with data split
        env_dict = env_factory.create_train_eval_envs(
            dataset_path=dataset_path,
            norm_stats_path=norm_stats_path,
            config=env_config,
            train_ratio=0.7,
            val_ratio=0.15,
            n_train_envs=1,
            n_eval_envs=1,
        )

        train_env = env_dict["train"]
        eval_env = env_dict["val"]

        logger.info(
            f"Environments created: "
            f"train={env_dict['splits']['train_size']} bars, "
            f"val={env_dict['splits']['val_size']} bars"
        )

        # Configure PPO hyperparameters
        ppo_config = PPOConfig(
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            total_timesteps=config.total_timesteps,
            eval_freq=max(config.total_timesteps // 20, 10000),  # ~20 evals
            n_eval_episodes=5,
            checkpoint_freq=max(config.total_timesteps // 10, 25000),
            tensorboard_log=True,
            verbose=1,
        )

        # Create trainer
        trainer = PPOTrainer(
            train_env=train_env,
            eval_env=eval_env,
            config=ppo_config,
            output_dir=output_dir,
            experiment_name=config.experiment_name,
        )

        # Run training
        logger.info(f"Starting PPO training for {config.total_timesteps:,} timesteps...")
        result = trainer.train()

        # Cleanup environments
        train_env.close()
        eval_env.close()

        return result


class ModelRegistrationStage(PipelineStage):
    """Register model in database"""

    def __init__(self, project_root: Path):
        super().__init__("model_registration")
        self.project_root = project_root

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config: TrainingConfig = context["config"]

        if not config.auto_register:
            logger.info("⏭ Auto-registration disabled, skipping")
            return context

        if not config.db_connection_string:
            logger.warning("⚠ No DB connection string, skipping registration")
            return context

        # Import model registry
        try:
            from .model_registry import ModelRegistry
        except ImportError:
            logger.warning("ModelRegistry not available, skipping registration")
            return context

        contract = context["contract"]
        model_path = context["model_path"]
        norm_stats_path = context["norm_stats_path"]

        # Generate model_id
        model_hash_short = context.get("model_hash", "unknown")[:8]
        model_id = f"ppo_{config.version}_{model_hash_short}"

        # Register
        try:
            registry = ModelRegistry(config.db_connection_string)
            registry.register_model(
                model_id=model_id,
                model_version=config.version,
                model_path=str(model_path),
                model_hash=context.get("model_hash", ""),
                norm_stats_hash=context.get("norm_stats_hash", ""),
                config_hash=context.get("contract_hash", ""),
                observation_dim=contract.observation_dim,
                action_space=3,  # LONG, SHORT, HOLD
                feature_order=list(contract.feature_order),
                training_metadata={
                    "duration_seconds": context.get("training_duration_seconds", 0),
                    "total_timesteps": context.get("total_timesteps", 0),
                    "best_mean_reward": context.get("best_mean_reward", 0),
                },
            )
            context["model_id"] = model_id
            logger.info(f"✓ Model registered: {model_id}")
        except Exception as e:
            logger.error(f"✗ Registration failed: {e}")
            context.setdefault("errors", []).append(f"Registration failed: {e}")

        return context


class BacktestValidationStage(PipelineStage):
    """Run backtest to validate model"""

    def __init__(self, project_root: Path):
        super().__init__("backtest_validation")
        self.project_root = project_root

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config: TrainingConfig = context["config"]

        if not config.run_backtest_validation:
            logger.info("⏭ Backtest validation disabled, skipping")
            return context

        if not config.backtest_start_date or not config.backtest_end_date:
            logger.warning("⚠ Backtest dates not specified, skipping validation")
            return context

        # Run backtest via inference API
        logger.info(f"Running backtest validation: {config.backtest_start_date} to {config.backtest_end_date}")

        # Placeholder - in production, call the inference API
        # result = await backtest_orchestrator.run(
        #     start_date=config.backtest_start_date,
        #     end_date=config.backtest_end_date,
        #     model_id=context.get("model_id"),
        # )

        logger.warning("BacktestValidationStage is a placeholder. Implement with actual backtest.")

        return context


# =============================================================================
# Training Pipeline Orchestrator
# =============================================================================

class TrainingPipeline:
    """
    Orchestrates the complete training pipeline.

    Usage:
        config = TrainingConfig(
            version="v1",
            dataset_path=Path("data/training.csv"),
            feature_columns=[...],
            total_timesteps=500_000,
        )

        pipeline = TrainingPipeline(project_root=Path("."))
        result = pipeline.run(config)

        if result.success:
            print(f"Model registered: {result.model_id}")
    """

    def __init__(
        self,
        project_root: Path,
        custom_stages: List[PipelineStage] = None
    ):
        self.project_root = project_root

        # Default pipeline stages
        self.stages = custom_stages or [
            DatasetValidationStage(),
            NormStatsGenerationStage(project_root),
            ContractCreationStage(project_root),
            ModelTrainingStage(project_root),
            ModelRegistrationStage(project_root),
            BacktestValidationStage(project_root),
        ]

    def run(
        self,
        config: TrainingConfig,
        on_progress: Callable[[str, int, int], None] = None
    ) -> PipelineResult:
        """
        Run the complete training pipeline.

        Args:
            config: Training configuration
            on_progress: Optional callback (stage_name, current, total)

        Returns:
            PipelineResult with all outputs and metrics
        """
        import time
        start_time = time.time()

        context = {"config": config}
        errors = []

        total_stages = len(self.stages)

        for i, stage in enumerate(self.stages):
            if on_progress:
                on_progress(stage.name, i + 1, total_stages)

            logger.info(f"[{i + 1}/{total_stages}] Running stage: {stage.name}")

            try:
                context = stage.execute(context)
            except Exception as e:
                error_msg = f"Stage '{stage.name}' failed: {str(e)}"
                logger.error(f"✗ {error_msg}")
                errors.append(error_msg)

                # Return early on critical failures
                if stage.name in ["dataset_validation", "model_training"]:
                    return PipelineResult(
                        success=False,
                        version=config.version,
                        errors=errors,
                    )

        # Build result
        result = PipelineResult(
            success=len(errors) == 0,
            version=config.version,
            model_id=context.get("model_id"),
            model_path=context.get("model_path"),
            norm_stats_path=context.get("norm_stats_path"),
            contract_path=context.get("contract_path"),
            model_hash=context.get("model_hash"),
            norm_stats_hash=context.get("norm_stats_hash"),
            contract_hash=context.get("contract_hash"),
            training_duration_seconds=context.get("training_duration_seconds", 0),
            total_timesteps=context.get("total_timesteps", 0),
            best_mean_reward=context.get("best_mean_reward", 0),
            backtest_sharpe=context.get("backtest_sharpe"),
            backtest_max_drawdown=context.get("backtest_max_drawdown"),
            backtest_win_rate=context.get("backtest_win_rate"),
            errors=errors,
        )

        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.1f}s - Success: {result.success}")

        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def run_training(
    project_root: Path,
    version: str,
    dataset_path: Path,
    feature_columns: List[str],
    total_timesteps: int = 500_000,
    db_connection_string: str = None,
    **kwargs
) -> PipelineResult:
    """
    Convenience function to run training pipeline.

    Example:
        result = run_training(
            project_root=Path("."),
            version="current",
            dataset_path=Path("data/RL_DS3_MACRO_CORE.csv"),
            feature_columns=[
                "log_ret_5m", "log_ret_1h", "log_ret_4h",
                "rsi_9", "atr_pct", "adx_14",
                "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
                "brent_change_1d", "rate_spread", "usdmxn_change_1d",
                "position", "time_normalized"
            ],
            total_timesteps=1_000_000,
            db_connection_string=os.environ.get("DATABASE_URL"),
        )

        if result.success:
            print(f"Model registered: {result.model_id}")
            print(f"Contract: {result.contract_path}")
            print(f"Norm stats: {result.norm_stats_path}")
    """
    config = TrainingConfig(
        version=version,
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        total_timesteps=total_timesteps,
        db_connection_string=db_connection_string,
        **kwargs
    )

    pipeline = TrainingPipeline(project_root)
    return pipeline.run(config)


# =============================================================================
# Standard Feature Order (for reference)
# =============================================================================

FEATURE_ORDER = [
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    "position", "time_normalized"
]
