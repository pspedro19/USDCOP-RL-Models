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

    DRY Principle: Use PPOConfig.from_ssot() to load defaults from SSOT.
    Direct instantiation uses fallback defaults for backward compatibility.

    SSOT: config/experiment_ssot.yaml
    """
    # Core PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None

    # Entropy and value coefficients
    ent_coef: float = 0.04  # Match SSOT default
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"

    # Training settings
    total_timesteps: int = 500_000
    eval_freq: int = 25_000
    n_eval_episodes: int = 5
    checkpoint_freq: int = 50_000

    # Overfitting prevention (SSOT: experiment_ssot.yaml)
    lr_decay_enabled: bool = True
    lr_decay_final: float = 0.00003
    early_stopping_enabled: bool = False  # Match SSOT default
    early_stopping_patience: int = 10     # Match SSOT default
    early_stopping_min_improvement: float = 0.01

    # Reproducibility
    seed: int = 42  # Fixed seed for reproducibility

    # Algorithm selection (Phase 1 factory)
    algorithm_name: str = "ppo"  # "ppo" | "recurrent_ppo" | "sac"

    # V22 P3: LSTM configuration
    use_lstm: bool = False
    lstm_hidden_size: int = 128   # NOT 256 — reduces overfitting
    n_lstm_layers: int = 1

    # Device
    device: str = "auto"

    # Logging
    tensorboard_log: bool = False  # Match SSOT default (disabled)
    verbose: int = 1

    @classmethod
    def from_ssot(cls, **overrides) -> "PPOConfig":
        """
        Create PPOConfig from SSOT (Single Source of Truth).

        DRY: This is the preferred way to create PPOConfig.
        All defaults come from config/experiment_ssot.yaml.

        Args:
            **overrides: Override any SSOT values

        Returns:
            PPOConfig with SSOT defaults and any overrides applied

        Example:
            config = PPOConfig.from_ssot(total_timesteps=100_000)
        """
        try:
            from src.training.config import get_ppo_hyperparameters
            from src.config import get_early_stopping_config, get_lr_decay_config

            # Load from SSOT
            hyperparams = get_ppo_hyperparameters(force_reload=True)
            early_stop = get_early_stopping_config()
            lr_decay = get_lr_decay_config()

            # Build config from SSOT
            ssot_values = {
                "learning_rate": hyperparams.learning_rate,
                "n_steps": hyperparams.n_steps,
                "batch_size": hyperparams.batch_size,
                "n_epochs": hyperparams.n_epochs,
                "gamma": hyperparams.gamma,
                "gae_lambda": hyperparams.gae_lambda,
                "clip_range": hyperparams.clip_range,
                "ent_coef": hyperparams.ent_coef,
                "vf_coef": hyperparams.vf_coef,
                "max_grad_norm": hyperparams.max_grad_norm,
                "total_timesteps": hyperparams.total_timesteps,
                "eval_freq": getattr(hyperparams, 'eval_freq', 25_000),
                # LR decay
                "lr_decay_enabled": lr_decay.enabled,
                "lr_decay_final": lr_decay.final_lr,
                # Early stopping
                "early_stopping_enabled": early_stop.enabled,
                "early_stopping_patience": early_stop.patience,
                "early_stopping_min_improvement": early_stop.min_improvement,
            }

            # Apply overrides
            ssot_values.update(overrides)

            logger.debug(f"[PPOConfig] Loaded from SSOT: lr={ssot_values['learning_rate']}, ent_coef={ssot_values['ent_coef']}")

            return cls(**ssot_values)

        except ImportError as e:
            logger.warning(f"[PPOConfig] SSOT not available, using defaults: {e}")
            return cls(**overrides)

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
    """
    Monitor action distribution during training.

    FASE 4: Detects and alerts on action bias (e.g., 86% SHORT).
    Logs to tensorboard and MLflow for tracking.
    """

    def __init__(
        self,
        log_freq: int = 5000,
        threshold_long: float = 0.50,  # SSOT: experiment_ssot.yaml thresholds.long
        threshold_short: float = -0.50,  # SSOT: experiment_ssot.yaml thresholds.short
        bias_alert_threshold: float = 0.70,  # Alert if any action > 70%
        target_hold_range: tuple = (0.30, 0.50),  # Target HOLD range: 30-50%
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.bias_alert_threshold = bias_alert_threshold
        self.target_hold_range = target_hold_range
        self.actions: List[float] = []

        # Cumulative tracking
        self._total_long = 0
        self._total_short = 0
        self._total_hold = 0
        self._bias_alerts = 0

    def _on_step(self) -> bool:
        if "actions" in self.locals:
            action = self.locals["actions"][0]
            if isinstance(action, np.ndarray):
                action = float(action[0])
            self.actions.append(action)

            # Track cumulative
            if action > self.threshold_long:
                self._total_long += 1
            elif action < self.threshold_short:
                self._total_short += 1
            else:
                self._total_hold += 1

        if self.n_calls % self.log_freq == 0 and len(self.actions) > 0:
            recent_actions = np.array(self.actions[-self.log_freq:])

            long_pct = np.mean(recent_actions > self.threshold_long) * 100
            short_pct = np.mean(recent_actions < self.threshold_short) * 100
            hold_pct = 100 - long_pct - short_pct

            # Calculate cumulative percentages
            total = self._total_long + self._total_short + self._total_hold
            if total > 0:
                cum_long_pct = self._total_long / total * 100
                cum_short_pct = self._total_short / total * 100
                cum_hold_pct = self._total_hold / total * 100
            else:
                cum_long_pct = cum_short_pct = cum_hold_pct = 0

            # FASE 4: Bias detection
            bias_detected = False
            bias_type = None
            if long_pct > self.bias_alert_threshold * 100:
                bias_detected = True
                bias_type = "LONG"
            elif short_pct > self.bias_alert_threshold * 100:
                bias_detected = True
                bias_type = "SHORT"

            if bias_detected:
                self._bias_alerts += 1
                logger.warning(
                    f"[FASE-4 BIAS ALERT] Step {self.n_calls}: "
                    f"Extreme {bias_type} bias detected! "
                    f"LONG={long_pct:.1f}%, HOLD={hold_pct:.1f}%, SHORT={short_pct:.1f}%"
                )

            # Check if HOLD is in target range
            hold_in_range = self.target_hold_range[0] * 100 <= hold_pct <= self.target_hold_range[1] * 100

            if self.verbose > 0:
                status = "OK" if hold_in_range and not bias_detected else "!"
                logger.info(
                    f"[Step {self.n_calls}] Actions: "
                    f"LONG={long_pct:.1f}%, HOLD={hold_pct:.1f}%, SHORT={short_pct:.1f}% "
                    f"(cum: L={cum_long_pct:.1f}%, H={cum_hold_pct:.1f}%, S={cum_short_pct:.1f}%) [{status}]"
                )

            # Log to tensorboard if available
            if self.logger:
                self.logger.record("actions/long_pct", long_pct)
                self.logger.record("actions/hold_pct", hold_pct)
                self.logger.record("actions/short_pct", short_pct)
                self.logger.record("actions/cum_long_pct", cum_long_pct)
                self.logger.record("actions/cum_short_pct", cum_short_pct)
                self.logger.record("actions/cum_hold_pct", cum_hold_pct)
                self.logger.record("actions/bias_alerts", self._bias_alerts)
                self.logger.record("actions/hold_in_target", float(hold_in_range))

            # Log to MLflow if available
            try:
                import mlflow
                if mlflow.active_run():
                    mlflow.log_metrics({
                        "actions_long_pct": long_pct,
                        "actions_hold_pct": hold_pct,
                        "actions_short_pct": short_pct,
                        "actions_bias_alerts": self._bias_alerts,
                    }, step=self.n_calls)
            except ImportError:
                pass
            except Exception:
                pass  # MLflow not active, skip

        return True

    @property
    def action_distribution(self) -> Dict[str, float]:
        """Get cumulative action distribution."""
        total = self._total_long + self._total_short + self._total_hold
        if total == 0:
            return {"long": 0, "hold": 0, "short": 0}
        return {
            "long": self._total_long / total * 100,
            "hold": self._total_hold / total * 100,
            "short": self._total_short / total * 100,
        }

    @property
    def has_severe_bias(self) -> bool:
        """Check if there's severe action bias."""
        dist = self.action_distribution
        return any(v > self.bias_alert_threshold * 100 for v in [dist["long"], dist["short"]])


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
                progress_bar=False,  # Disabled until tqdm/rich properly installed
            )

            # Save final model
            # FIX: Save policy weights only if full save fails due to pickle issues
            # (environment may contain non-picklable objects like PyCapsule from C extensions)
            model_path = self.output_dir / "final_model.zip"
            policy_path = self.output_dir / "policy_weights.pt"
            save_success = False

            # Try standard SB3 save first
            try:
                self.model.save(model_path)
                logger.info(f"Model saved: {model_path}")
                save_success = True
            except Exception as e:
                logger.warning(f"Standard model save failed: {e}")

                # Try saving without environment data
                try:
                    # Method 1: Use exclude parameter (SB3 2.0+)
                    self.model.save(model_path, exclude=["env", "_vec_normalize_env"])
                    logger.info(f"Model saved (excluded env): {model_path}")
                    save_success = True
                except Exception as e2:
                    logger.warning(f"Excluded save failed: {e2}")

                    # Method 2: Save policy weights only using PyTorch
                    try:
                        import torch
                        state_dict = {
                            "policy": self.model.policy.state_dict(),
                            "optimizer": self.model.policy.optimizer.state_dict() if hasattr(self.model.policy, 'optimizer') else None,
                        }
                        torch.save(state_dict, policy_path)
                        logger.info(f"Policy weights saved: {policy_path}")
                        # Create empty placeholder for model_path
                        model_path.touch()
                        save_success = True
                    except Exception as e3:
                        logger.error(f"Policy save failed completely: {e3}")
                        # Continue anyway - model is trained, just can't save
                        save_success = False
                        model_path = None

            # Compute model hash (handle None model_path)
            model_hash = None
            if model_path and model_path.exists() and model_path.stat().st_size > 0:
                model_hash = self._compute_model_hash(model_path)
            elif policy_path.exists():
                model_hash = self._compute_model_hash(policy_path)

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
            import traceback
            logger.error(f"Training failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

            return TrainingResult(
                success=False,
                training_duration_seconds=training_duration,
                error_message=str(e),
            )

    def _create_model(self):
        """Create model using algorithm factory.

        Supports PPO, RecurrentPPO, and SAC via the factory pattern.
        Backward compatible: use_lstm=True → recurrent_ppo.
        """
        from src.training.algorithm_factory import create_algorithm

        sb3_kwargs = self.config.to_sb3_kwargs()

        # Setup tensorboard logging
        tensorboard_log = None
        if self.config.tensorboard_log:
            tensorboard_log = str(self.output_dir / "tensorboard")

        # Resolve algorithm name (backward compat: use_lstm → recurrent_ppo)
        algo_name = self.config.algorithm_name
        if algo_name == "ppo" and self.config.use_lstm:
            algo_name = "recurrent_ppo"

        # Build adapter kwargs
        adapter_kwargs = {}
        if algo_name == "recurrent_ppo":
            adapter_kwargs["lstm_hidden_size"] = self.config.lstm_hidden_size
            adapter_kwargs["n_lstm_layers"] = self.config.n_lstm_layers

        adapter = create_algorithm(algo_name, **adapter_kwargs)

        # Build creation kwargs
        create_kwargs = dict(sb3_kwargs)
        create_kwargs["tensorboard_log"] = tensorboard_log
        create_kwargs["seed"] = self.config.seed

        model = adapter.create(self.train_env, **create_kwargs)

        logger.info(
            f"{algo_name} model created: "
            f"net_arch={self.config.net_arch}, "
            f"seed={self.config.seed}, "
            f"device={model.device}, "
            f"recurrent={adapter.is_recurrent()}"
        )

        return model

    def _create_callbacks(self) -> CallbackList:
        """Create training callbacks including FASE 2 overfitting prevention."""
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

        # =====================================================================
        # FASE 2: Overfitting Prevention Callbacks
        # =====================================================================

        # LR Decay callback
        if self.config.lr_decay_enabled:
            try:
                from src.ml_workflow.training_callbacks import LearningRateDecayCallback
                callbacks.append(LearningRateDecayCallback(
                    initial_lr=self.config.learning_rate,
                    final_lr=self.config.lr_decay_final,
                    total_timesteps=self.config.total_timesteps,
                    log_freq=self.config.eval_freq,
                    verbose=self.config.verbose,
                ))
                logger.info(
                    f"[FASE-2] LR decay enabled: {self.config.learning_rate:.6f} -> "
                    f"{self.config.lr_decay_final:.6f}"
                )
            except ImportError as e:
                logger.warning(f"[FASE-2] LR decay not available: {e}")

        # Early stopping callback (only if we have eval_env)
        if self.config.early_stopping_enabled and self.eval_env is not None:
            try:
                from src.ml_workflow.training_callbacks import ValidationEarlyStoppingCallback
                self._early_stopping_callback = ValidationEarlyStoppingCallback(
                    eval_env=self.eval_env,
                    eval_freq=self.config.eval_freq,
                    n_eval_episodes=self.config.n_eval_episodes,
                    patience=self.config.early_stopping_patience,
                    min_improvement=self.config.early_stopping_min_improvement,
                    best_model_save_path=str(self.output_dir),  # Enable best_model saving
                    verbose=self.config.verbose,
                )
                callbacks.append(self._early_stopping_callback)
                logger.info(
                    f"[FASE-2] Early stopping enabled: patience={self.config.early_stopping_patience}, "
                    f"min_improvement={self.config.early_stopping_min_improvement:.2%}"
                )
            except ImportError as e:
                logger.warning(f"[FASE-2] Early stopping not available: {e}")

        # Standard Evaluation callback (if early stopping not enabled or as backup)
        if self.eval_env is not None and not self.config.early_stopping_enabled:
            # Create eval_logs directory explicitly to avoid potential Windows path issues
            eval_logs_dir = self.output_dir / "eval_logs"
            eval_logs_dir.mkdir(parents=True, exist_ok=True)
            # Enable best_model saving - CRITICAL for reproducibility
            # If pickle fails, we'll save policy weights as fallback after training
            callbacks.append(EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.output_dir),  # Save best_model.zip here
                log_path=str(eval_logs_dir),
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True,
                verbose=self.config.verbose,
            ))

        # Checkpoint callback - DISABLED due to PyCapsule pickle issues
        # The environment contains non-picklable objects that prevent model.save()
        # from working. The final model will be saved using PyTorch state_dict fallback.
        # if self.config.checkpoint_freq > 0:
        #     checkpoints_dir = self.output_dir / "checkpoints"
        #     checkpoints_dir.mkdir(parents=True, exist_ok=True)
        #     callbacks.append(CheckpointCallback(
        #         save_freq=self.config.checkpoint_freq,
        #         save_path=str(checkpoints_dir),
        #         name_prefix=self.experiment_name,
        #         verbose=self.config.verbose,
        #     ))
        # Create checkpoints dir anyway for consistency
        if self.config.checkpoint_freq > 0:
            checkpoints_dir = self.output_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

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
