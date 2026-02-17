"""
Multi-Seed Trainer - Training with Multiple Seeds for Robustness
================================================================

Addresses the RL variance problem by training with multiple seeds
and selecting the best model based on validation performance.

Problem:
    Same config + different seeds = very different results
    Example: seed=None gave +13.64% return, seed=42 gave -6.24%

Solution:
    Train N models with different seeds, select best, report variance.
    If CV (coefficient of variation) > threshold, alert high variance.

Contract: CTR-MULTI-SEED-001
Version: 1.0.0
Date: 2026-02-04

Usage:
    from src.training.multi_seed_trainer import MultiSeedTrainer, MultiSeedConfig

    config = MultiSeedConfig(seeds=[42, 123, 456, 789, 1337])
    trainer = MultiSeedTrainer(project_root=Path("."))
    result = trainer.train(base_request, config)

    print(f"Best seed: {result.best_seed}")
    print(f"Variance CV: {result.cv_reward:.2%}")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .engine import TrainingEngine, TrainingRequest, TrainingResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultiSeedConfig:
    """
    Configuration for multi-seed training.

    Attributes:
        seeds: List of seeds to train with
        selection_metric: Metric to select best model ("best_mean_reward" or "final_mean_reward")
        max_cv_threshold: Maximum coefficient of variation before warning (0.30 = 30%)
        save_all_models: Whether to save all models or just the best
        parallel: Whether to train in parallel (future enhancement)
    """
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337])
    selection_metric: str = "best_mean_reward"
    max_cv_threshold: float = 0.30  # 30% CV threshold
    save_all_models: bool = True  # V22: Need ALL 5 for ensemble voting
    parallel: bool = False  # Future: parallel training


@dataclass
class MultiSeedResult:
    """
    Result from multi-seed training.

    Captures all individual results and aggregate statistics.
    """
    success: bool

    # Best model info
    best_seed: int = 0
    best_model_path: Optional[Path] = None
    best_model_hash: Optional[str] = None
    best_mean_reward: float = 0.0

    # Aggregate statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    cv_reward: float = 0.0  # Coefficient of variation
    min_reward: float = 0.0
    max_reward: float = 0.0

    # V22: All model paths for ensemble
    all_model_paths: List[Path] = field(default_factory=list)
    all_model_rewards: Dict[int, float] = field(default_factory=dict)

    # Individual results
    seed_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Warnings
    high_variance_warning: bool = False
    warnings: List[str] = field(default_factory=list)

    # Timing
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "best_seed": self.best_seed,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "best_model_hash": self.best_model_hash,
            "best_mean_reward": self.best_mean_reward,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "cv_reward": self.cv_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "high_variance_warning": self.high_variance_warning,
            "warnings": self.warnings,
            "total_duration_seconds": self.total_duration_seconds,
            "seed_results": {
                str(seed): {
                    "seed": seed,
                    "success": r.get("success", False),
                    "best_mean_reward": r.get("best_mean_reward", 0.0),
                    "model_hash": r.get("model_hash"),
                }
                for seed, r in self.seed_results.items()
            },
        }


# =============================================================================
# Multi-Seed Trainer
# =============================================================================

class MultiSeedTrainer:
    """
    Orchestrates training with multiple seeds.

    This trainer runs the same configuration with different random seeds
    to assess model variance and select the best performing seed.

    Benefits:
        1. Quantifies training variance (is the result reproducible?)
        2. Selects best seed (higher chance of production success)
        3. Alerts on high variance (indicates unstable training)
        4. Provides confidence intervals for expected performance
    """

    def __init__(self, project_root: Path):
        """
        Initialize multi-seed trainer.

        Args:
            project_root: Project root directory
        """
        self.project_root = Path(project_root)
        self.engine = TrainingEngine(project_root=project_root)

    def train(
        self,
        base_request: TrainingRequest,
        config: Optional[MultiSeedConfig] = None,
    ) -> MultiSeedResult:
        """
        Train with multiple seeds and select best model.

        Args:
            base_request: Base training request (seed will be overwritten)
            config: Multi-seed configuration

        Returns:
            MultiSeedResult with best model and variance statistics
        """
        config = config or MultiSeedConfig()
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("MULTI-SEED TRAINER")
        logger.info("=" * 70)
        logger.info(f"Seeds: {config.seeds}")
        logger.info(f"Selection metric: {config.selection_metric}")
        logger.info(f"Max CV threshold: {config.max_cv_threshold:.0%}")

        # Train with each seed
        results: Dict[int, TrainingResult] = {}

        for i, seed in enumerate(config.seeds):
            logger.info("-" * 50)
            logger.info(f"Training with seed {seed} ({i+1}/{len(config.seeds)})")
            logger.info("-" * 50)

            # Create request with this seed
            seed_request = self._create_seed_request(base_request, seed, i)

            # Run training
            result = self.engine.run(seed_request)
            results[seed] = result

            if result.success:
                logger.info(f"  Seed {seed}: best_reward={result.best_mean_reward:.4f}")
            else:
                logger.warning(f"  Seed {seed}: FAILED - {result.errors}")

        # Analyze results
        multi_result = self._analyze_results(results, config)
        multi_result.total_duration_seconds = time.time() - start_time

        # Save summary
        self._save_summary(base_request, multi_result, config)

        # Log summary
        logger.info("=" * 70)
        logger.info("MULTI-SEED TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best seed: {multi_result.best_seed}")
        logger.info(f"Best reward: {multi_result.best_mean_reward:.4f}")
        logger.info(f"Mean reward: {multi_result.mean_reward:.4f} +/- {multi_result.std_reward:.4f}")
        logger.info(f"CV: {multi_result.cv_reward:.2%}")

        if multi_result.high_variance_warning:
            logger.warning(f"HIGH VARIANCE DETECTED! CV={multi_result.cv_reward:.2%} > {config.max_cv_threshold:.0%}")
            logger.warning("Consider: adjusting hyperparameters, more timesteps, or different architecture")

        return multi_result

    def _create_seed_request(
        self,
        base_request: TrainingRequest,
        seed: int,
        index: int,
    ) -> TrainingRequest:
        """Create training request for specific seed."""
        # Create unique output directory for this seed
        base_output = base_request.output_dir or (
            self.project_root / "models" / f"ppo_{base_request.version}_production"
        )
        seed_output = base_output.parent / f"{base_output.name}_seed{seed}"

        # Create unique experiment name
        base_exp = base_request.experiment_name or f"ppo_{base_request.version}"
        seed_exp = f"{base_exp}_seed{seed}"

        # Use dataclasses.replace for immutable update
        return replace(
            base_request,
            seed=seed,
            output_dir=seed_output,
            experiment_name=seed_exp,
        )

    def _analyze_results(
        self,
        results: Dict[int, TrainingResult],
        config: MultiSeedConfig,
    ) -> MultiSeedResult:
        """Analyze results from all seeds."""
        # Filter successful results
        successful = {
            seed: r for seed, r in results.items() if r.success
        }

        if not successful:
            return MultiSeedResult(
                success=False,
                warnings=["All seed trainings failed"],
                seed_results={
                    seed: {"success": False, "errors": r.errors}
                    for seed, r in results.items()
                },
            )

        # Extract rewards based on selection metric
        if config.selection_metric == "best_mean_reward":
            rewards = {seed: r.best_mean_reward for seed, r in successful.items()}
        else:
            rewards = {seed: r.final_mean_reward for seed, r in successful.items()}

        # Find best seed
        best_seed = max(rewards, key=lambda s: rewards[s])
        best_result = successful[best_seed]

        # Calculate statistics
        reward_values = list(rewards.values())
        mean_reward = float(np.mean(reward_values))
        std_reward = float(np.std(reward_values))
        cv_reward = std_reward / abs(mean_reward) if abs(mean_reward) > 1e-8 else float('inf')

        # Check for high variance
        high_variance = cv_reward > config.max_cv_threshold
        warnings = []
        if high_variance:
            warnings.append(
                f"High variance detected: CV={cv_reward:.2%} > threshold={config.max_cv_threshold:.0%}"
            )

        # Failed seeds warning
        failed_seeds = [s for s, r in results.items() if not r.success]
        if failed_seeds:
            warnings.append(f"Seeds {failed_seeds} failed to train")

        # V22: Collect all model paths for ensemble
        all_paths = []
        all_rewards = {}
        for seed, r in successful.items():
            if r.model_path:
                # Check for best_model.zip in the model directory
                model_dir = Path(r.model_path).parent
                best_path = model_dir / "best_model.zip"
                if best_path.exists():
                    all_paths.append(best_path)
                elif Path(r.model_path).exists():
                    all_paths.append(Path(r.model_path))
                all_rewards[seed] = float(rewards[seed])

        return MultiSeedResult(
            success=True,
            best_seed=best_seed,
            best_model_path=best_result.model_path,
            best_model_hash=best_result.model_hash,
            best_mean_reward=float(rewards[best_seed]),
            mean_reward=mean_reward,
            std_reward=std_reward,
            cv_reward=cv_reward,
            min_reward=float(min(reward_values)),
            max_reward=float(max(reward_values)),
            all_model_paths=all_paths,
            all_model_rewards=all_rewards,
            high_variance_warning=high_variance,
            warnings=warnings,
            seed_results={
                seed: {
                    "success": r.success,
                    "best_mean_reward": r.best_mean_reward,
                    "final_mean_reward": r.final_mean_reward,
                    "model_hash": r.model_hash,
                    "model_path": str(r.model_path) if r.model_path else None,
                    "training_seed": r.training_seed,
                }
                for seed, r in results.items()
            },
        )

    def _save_summary(
        self,
        base_request: TrainingRequest,
        result: MultiSeedResult,
        config: MultiSeedConfig,
    ) -> None:
        """Save multi-seed training summary."""
        output_dir = base_request.output_dir or (
            self.project_root / "models" / f"ppo_{base_request.version}_production"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "multi_seed_training": True,
            "config": {
                "seeds": config.seeds,
                "selection_metric": config.selection_metric,
                "max_cv_threshold": config.max_cv_threshold,
            },
            "result": result.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": base_request.version,
        }

        summary_path = output_dir / "multi_seed_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Multi-seed summary saved: {summary_path}")


# =============================================================================
# Convenience Function
# =============================================================================

def train_with_multiple_seeds(
    project_root: Path,
    version: str,
    dataset_path: Path,
    seeds: Optional[List[int]] = None,
    total_timesteps: Optional[int] = None,
    **kwargs
) -> MultiSeedResult:
    """
    Convenience function for multi-seed training.

    Args:
        project_root: Project root directory
        version: Model version string
        dataset_path: Path to training dataset
        seeds: List of seeds (default: [42, 123, 456, 789, 1337])
        total_timesteps: Total training timesteps
        **kwargs: Additional TrainingRequest kwargs

    Returns:
        MultiSeedResult with best model and statistics

    Example:
        result = train_with_multiple_seeds(
            project_root=Path("."),
            version="v1",
            dataset_path=Path("data/train.parquet"),
            seeds=[42, 123, 456],
        )
        print(f"Best seed: {result.best_seed}, CV: {result.cv_reward:.2%}")
    """
    request = TrainingRequest(
        version=version,
        dataset_path=Path(dataset_path),
        total_timesteps=total_timesteps,
        **kwargs,
    )

    config = MultiSeedConfig(seeds=seeds or [42, 123, 456, 789, 1337])

    trainer = MultiSeedTrainer(project_root=project_root)
    return trainer.train(request, config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MultiSeedTrainer",
    "MultiSeedConfig",
    "MultiSeedResult",
    "train_with_multiple_seeds",
]
