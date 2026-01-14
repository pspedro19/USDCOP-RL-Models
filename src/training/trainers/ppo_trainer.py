"""
PPO Trainer - Professional PPO Training Wrapper
===============================================
Clean wrapper around Stable Baselines 3 PPO.

SOLID Principles:
- Single Responsibility: Only handles PPO training
- Open/Closed: Extensible via callbacks
- Dependency Inversion: Depends on VecEnv abstraction

Design Patterns:
- Facade Pattern: Simplifies SB3 PPO interface
- Builder Pattern: Fluent configuration
- Observer Pattern: Callbacks for monitoring

Clean Code:
- Explicit configuration (no hidden defaults)
- Comprehensive logging
- Type hints throughout
"""

import json
import logging
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.vec_env import VecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    VecEnv = None

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PPOConfig:
    """
    PPO hyperparameter configuration.

    All hyperparameters are explicit with sensible defaults.
    """
    # Core PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps per environment before update
    batch_size: int = 64  # Minibatch size
    n_epochs: int = 10  # Epochs per update
    gamma: float = 0.90  # shorter-term focus for noisy 5-min data
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO clip range
    clip_range_vf: Optional[float] = None  # Value function clip (None = no clip)

    # Entropy and value coefficients
    ent_coef: float = 0.05  # more exploration
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Network architecture
    net_arch: List[int] = field(default_factory=lambda: [256, 256])  # Match production training
    activation_fn: str = "tanh"  # "tanh", "relu", "elu"

    # Training settings
    total_timesteps: int = 500_000
    eval_freq: int = 25_000  # Steps between evaluations
    n_eval_episodes: int = 5  # Episodes per evaluation
    checkpoint_freq: int = 50_000  # Steps between checkpoints

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"

    # Logging
    tensorboard_log: bool = True
    verbose: int = 1

    def to_sb3_kwargs(self) -> Dict[str, Any]:
        """Convert to Stable Baselines 3 kwargs"""
        activation_map = {
            "tanh": torch.nn.Tanh if SB3_AVAILABLE else None,
            "relu": torch.nn.ReLU if SB3_AVAILABLE else None,
            "elu": torch.nn.ELU if SB3_AVAILABLE else None,
        }

        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "policy_kwargs": {
                "net_arch": self.net_arch,
                "activation_fn": activation_map.get(self.activation_fn),
            },
            "device": self.device,
            "verbose": self.verbose,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class TrainingResult:
    """Result of a training run"""
    success: bool
    model_path: Optional[Path] = None
    model_hash: Optional[str] = None

    # Timing
    training_duration_seconds: float = 0.0
    total_timesteps: int = 0

    # Metrics
    final_mean_reward: float = 0.0
    best_mean_reward: float = 0.0
    final_mean_ep_length: float = 0.0

    # Evaluation results
    eval_rewards: List[float] = field(default_factory=list)
    eval_timesteps: List[int] = field(default_factory=list)

    # Errors
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_hash": self.model_hash,
            "training_duration_seconds": self.training_duration_seconds,
            "total_timesteps": self.total_timesteps,
            "final_mean_reward": self.final_mean_reward,
            "best_mean_reward": self.best_mean_reward,
            "final_mean_ep_length": self.final_mean_ep_length,
            "eval_rewards": self.eval_rewards,
            "eval_timesteps": self.eval_timesteps,
            "error_message": self.error_message,
        }


# =============================================================================
# Custom Callbacks
# =============================================================================

class ActionDistributionCallback(BaseCallback):
    """Monitor action distribution during training"""

    def __init__(
        self,
        log_freq: int = 5000,
        threshold_long: float = 0.33,  # From config/trading_config.yaml SSOT
        threshold_short: float = -0.33,  # From config/trading_config.yaml SSOT
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.actions: List[float] = []

    def _on_step(self) -> bool:
        if "actions" in self.locals:
            action = self.locals["actions"][0]
            if isinstance(action, np.ndarray):
                action = float(action[0])
            self.actions.append(action)

        if self.n_calls % self.log_freq == 0 and len(self.actions) > 0:
            recent_actions = np.array(self.actions[-self.log_freq:])

            long_pct = np.mean(recent_actions > self.threshold_long) * 100
            short_pct = np.mean(recent_actions < self.threshold_short) * 100
            hold_pct = 100 - long_pct - short_pct

            if self.verbose > 0:
                logger.info(
                    f"[Step {self.n_calls}] Actions: "
                    f"LONG={long_pct:.1f}%, HOLD={hold_pct:.1f}%, SHORT={short_pct:.1f}%"
                )

            # Log to tensorboard if available
            if self.logger:
                self.logger.record("actions/long_pct", long_pct)
                self.logger.record("actions/hold_pct", hold_pct)
                self.logger.record("actions/short_pct", short_pct)

        return True


class MetricsCallback(BaseCallback):
    """Collect training metrics for result reporting"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.eval_rewards: List[float] = []
        self.eval_timesteps: List[int] = []
        self.best_mean_reward: float = float("-inf")

    def _on_step(self) -> bool:
        # Check if eval callback has run
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            mean_reward = np.mean(rewards)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

        return True

    def update_eval_metrics(self, mean_reward: float, timestep: int):
        """Called by EvalCallback to record eval results"""
        self.eval_rewards.append(mean_reward)
        self.eval_timesteps.append(timestep)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward


class ProgressCallback(BaseCallback):
    """Progress reporting callback"""

    def __init__(
        self,
        total_timesteps: int,
        report_freq: int = 10000,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.report_freq = report_freq
        self.on_progress = on_progress
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.report_freq == 0:
            progress = self.n_calls / self.total_timesteps
            elapsed = time.time() - self.start_time if self.start_time else 0

            if self.on_progress:
                self.on_progress(self.n_calls, self.total_timesteps, elapsed)

            if self.verbose > 0:
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                logger.info(
                    f"Progress: {progress*100:.1f}% "
                    f"({self.n_calls:,}/{self.total_timesteps:,}) "
                    f"ETA: {eta/60:.1f}min"
                )

        return True


# =============================================================================
# PPO Trainer
# =============================================================================

class PPOTrainer:
    """
    Professional PPO trainer wrapper.

    Provides:
    - Clean configuration interface
    - Automatic callback setup
    - Checkpoint management
    - Metrics collection
    - Model hashing

    Usage:
        trainer = PPOTrainer(
            train_env=train_env,
            eval_env=eval_env,
            config=PPOConfig(total_timesteps=1_000_000),
            output_dir=Path("models/ppo"),
        )

        result = trainer.train()

        if result.success:
            print(f"Model saved: {result.model_path}")
            print(f"Best reward: {result.best_mean_reward}")
    """

    def __init__(
        self,
        train_env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        config: Optional[PPOConfig] = None,
        output_dir: Optional[Path] = None,
        experiment_name: str = "ppo_training",
        custom_callbacks: Optional[List[BaseCallback]] = None,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            train_env: Vectorized training environment
            eval_env: Optional evaluation environment
            config: PPO configuration
            output_dir: Output directory for model and logs
            experiment_name: Name for this training run
            custom_callbacks: Additional callbacks
            on_progress: Progress callback (current, total, elapsed_seconds)
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 required. Install with: pip install stable-baselines3"
            )

        self.train_env = train_env
        self.eval_env = eval_env
        self.config = config or PPOConfig()
        self.output_dir = output_dir or Path("models/ppo_training")
        self.experiment_name = experiment_name
        self.custom_callbacks = custom_callbacks or []
        self.on_progress = on_progress

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model will be created in train()
        self.model: Optional[PPO] = None
        self._metrics_callback: Optional[MetricsCallback] = None

        logger.info(
            f"PPOTrainer initialized: "
            f"timesteps={self.config.total_timesteps:,}, "
            f"output_dir={self.output_dir}"
        )

    def train(self) -> TrainingResult:
        """
        Run PPO training.

        Returns:
            TrainingResult with model path, metrics, and status
        """
        start_time = time.time()

        try:
            # Create model
            logger.info("Creating PPO model...")
            self.model = self._create_model()

            # Create callbacks
            callbacks = self._create_callbacks()

            # Log configuration
            self._save_config()

            # Train
            logger.info(f"Starting training for {self.config.total_timesteps:,} timesteps...")
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=self.config.verbose > 0,
            )

            # Save final model
            model_path = self.output_dir / "final_model.zip"
            self.model.save(model_path)
            logger.info(f"Model saved: {model_path}")

            # Compute model hash
            model_hash = self._compute_model_hash(model_path)

            # Collect metrics
            training_duration = time.time() - start_time

            # Get final metrics
            final_mean_reward = 0.0
            final_mean_ep_length = 0.0
            if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
                final_mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                final_mean_ep_length = np.mean([ep["l"] for ep in self.model.ep_info_buffer])

            result = TrainingResult(
                success=True,
                model_path=model_path,
                model_hash=model_hash,
                training_duration_seconds=training_duration,
                total_timesteps=self.config.total_timesteps,
                final_mean_reward=float(final_mean_reward),
                best_mean_reward=float(
                    self._metrics_callback.best_mean_reward
                    if self._metrics_callback else final_mean_reward
                ),
                final_mean_ep_length=float(final_mean_ep_length),
                eval_rewards=self._metrics_callback.eval_rewards if self._metrics_callback else [],
                eval_timesteps=self._metrics_callback.eval_timesteps if self._metrics_callback else [],
            )

            # Save result
            self._save_result(result)

            logger.info(
                f"Training complete: "
                f"duration={training_duration/60:.1f}min, "
                f"best_reward={result.best_mean_reward:.2f}"
            )

            return result

        except Exception as e:
            training_duration = time.time() - start_time
            logger.error(f"Training failed: {e}")

            return TrainingResult(
                success=False,
                training_duration_seconds=training_duration,
                error_message=str(e),
            )

    def _create_model(self) -> PPO:
        """Create PPO model with configuration"""
        sb3_kwargs = self.config.to_sb3_kwargs()

        # Setup tensorboard logging
        tensorboard_log = None
        if self.config.tensorboard_log:
            tensorboard_log = str(self.output_dir / "tensorboard")

        model = PPO(
            policy="MlpPolicy",
            env=self.train_env,
            tensorboard_log=tensorboard_log,
            **sb3_kwargs,
        )

        logger.info(
            f"PPO model created: "
            f"policy=MlpPolicy, "
            f"net_arch={self.config.net_arch}, "
            f"device={model.device}"
        )

        return model

    def _create_callbacks(self) -> CallbackList:
        """Create training callbacks"""
        callbacks = []

        # Metrics callback (always)
        self._metrics_callback = MetricsCallback(verbose=self.config.verbose)
        callbacks.append(self._metrics_callback)

        # Progress callback
        if self.on_progress or self.config.verbose > 0:
            callbacks.append(ProgressCallback(
                total_timesteps=self.config.total_timesteps,
                report_freq=10000,
                on_progress=self.on_progress,
                verbose=self.config.verbose,
            ))

        # Action distribution monitoring
        callbacks.append(ActionDistributionCallback(
            log_freq=10000,
            verbose=self.config.verbose,
        ))

        # Evaluation callback
        if self.eval_env is not None:
            callbacks.append(EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.output_dir),
                log_path=str(self.output_dir / "eval_logs"),
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True,
                verbose=self.config.verbose,
            ))

        # Checkpoint callback
        if self.config.checkpoint_freq > 0:
            callbacks.append(CheckpointCallback(
                save_freq=self.config.checkpoint_freq,
                save_path=str(self.output_dir / "checkpoints"),
                name_prefix=self.experiment_name,
                verbose=self.config.verbose,
            ))

        # Custom callbacks
        callbacks.extend(self.custom_callbacks)

        return CallbackList(callbacks)

    def _compute_model_hash(self, model_path: Path) -> str:
        """Compute SHA256 hash of model file"""
        sha256 = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _save_config(self) -> None:
        """Save training configuration"""
        config_path = self.output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "ppo_config": self.config.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            }, f, indent=2)
        logger.info(f"Config saved: {config_path}")

    def _save_result(self, result: TrainingResult) -> None:
        """Save training result"""
        result_path = self.output_dir / "training_result.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Result saved: {result_path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def train_ppo(
    train_env: VecEnv,
    eval_env: Optional[VecEnv] = None,
    total_timesteps: int = 500_000,
    output_dir: Optional[Path] = None,
    **ppo_kwargs,
) -> TrainingResult:
    """
    Convenience function to train PPO model.

    Args:
        train_env: Training environment
        eval_env: Evaluation environment
        total_timesteps: Total training timesteps
        output_dir: Output directory
        **ppo_kwargs: Additional PPO config kwargs

    Returns:
        TrainingResult
    """
    config = PPOConfig(total_timesteps=total_timesteps, **ppo_kwargs)

    trainer = PPOTrainer(
        train_env=train_env,
        eval_env=eval_env,
        config=config,
        output_dir=output_dir,
    )

    return trainer.train()
