"""
Algorithm Factory - Multi-Algorithm Support via SSOT Config
===========================================================
Replaces hardcoded PPO/RecurrentPPO dispatch with a single factory.

Usage:
    from src.training.algorithm_factory import create_algorithm, get_algorithm_kwargs

    adapter = create_algorithm("ppo")
    model = adapter.create(env, **kwargs)
    loaded = adapter.load(path, device="cpu")

Supported algorithms:
    - "ppo": PPO with MlpPolicy (CPU preferred)
    - "recurrent_ppo": RecurrentPPO with MlpLstmPolicy (GPU OK)
    - "sac": SAC with MlpPolicy (off-policy, replay buffer)

Contract: CTR-ALGORITHM-FACTORY-001
Version: 1.0.0
Date: 2026-02-12
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

# =============================================================================
# ALGORITHM ADAPTER INTERFACE
# =============================================================================


class AlgorithmAdapter(ABC):
    """Abstract adapter wrapping an SB3 algorithm."""

    @abstractmethod
    def create(self, env, **kwargs) -> Any:
        """Create a new model instance.

        Args:
            env: Vectorized environment (DummyVecEnv)
            **kwargs: Algorithm-specific hyperparameters

        Returns:
            SB3 model instance
        """
        ...

    @abstractmethod
    def load(self, path: str, device: str = "cpu") -> Any:
        """Load a saved model from disk.

        Args:
            path: Path to saved model (.zip)
            device: Device to load onto

        Returns:
            Loaded SB3 model
        """
        ...

    @abstractmethod
    def is_recurrent(self) -> bool:
        """Whether this algorithm uses recurrent (LSTM) states."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Algorithm name for logging."""
        ...


# =============================================================================
# PPO ADAPTER
# =============================================================================


class PPOAdapter(AlgorithmAdapter):
    """Adapter for Stable-Baselines3 PPO."""

    def create(self, env, **kwargs) -> Any:
        from stable_baselines3 import PPO

        policy_kwargs = kwargs.pop("policy_kwargs", {"net_arch": [256, 256]})
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )
        logger.info(
            f"PPO model created: net_arch={policy_kwargs.get('net_arch')}, "
            f"seed={kwargs.get('seed')}, device={model.device}"
        )
        return model

    def load(self, path: str, device: str = "cpu") -> Any:
        from stable_baselines3 import PPO

        logger.info(f"Loading PPO model from {path}")
        return PPO.load(path, device=device)

    def is_recurrent(self) -> bool:
        return False

    def name(self) -> str:
        return "ppo"


# =============================================================================
# RECURRENT PPO ADAPTER
# =============================================================================


class RecurrentPPOAdapter(AlgorithmAdapter):
    """Adapter for sb3-contrib RecurrentPPO (LSTM)."""

    def __init__(self, lstm_hidden_size: int = 128, n_lstm_layers: int = 1):
        self._lstm_hidden_size = lstm_hidden_size
        self._n_lstm_layers = n_lstm_layers

    def create(self, env, **kwargs) -> Any:
        from sb3_contrib import RecurrentPPO

        policy_kwargs = kwargs.pop("policy_kwargs", {"net_arch": [256, 256]})
        # Inject LSTM params into policy_kwargs
        policy_kwargs["lstm_hidden_size"] = self._lstm_hidden_size
        policy_kwargs["n_lstm_layers"] = self._n_lstm_layers

        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )
        logger.info(
            f"RecurrentPPO model created: lstm_hidden={self._lstm_hidden_size}, "
            f"n_layers={self._n_lstm_layers}, seed={kwargs.get('seed')}, "
            f"device={model.device}"
        )
        return model

    def load(self, path: str, device: str = "cpu") -> Any:
        from sb3_contrib import RecurrentPPO

        logger.info(f"Loading RecurrentPPO model from {path}")
        return RecurrentPPO.load(path, device=device)

    def is_recurrent(self) -> bool:
        return True

    def name(self) -> str:
        return "recurrent_ppo"


# =============================================================================
# SAC ADAPTER
# =============================================================================


class SACAdapter(AlgorithmAdapter):
    """Adapter for Stable-Baselines3 SAC (off-policy, continuous actions)."""

    def create(self, env, **kwargs) -> Any:
        from stable_baselines3 import SAC

        policy_kwargs = kwargs.pop("policy_kwargs", {"net_arch": [256, 256]})
        model = SAC(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )
        logger.info(
            f"SAC model created: net_arch={policy_kwargs.get('net_arch')}, "
            f"seed={kwargs.get('seed')}, device={model.device}"
        )
        return model

    def load(self, path: str, device: str = "cpu") -> Any:
        from stable_baselines3 import SAC

        logger.info(f"Loading SAC model from {path}")
        return SAC.load(path, device=device)

    def is_recurrent(self) -> bool:
        return False

    def name(self) -> str:
        return "sac"


# =============================================================================
# FACTORY REGISTRY
# =============================================================================

_ALGORITHM_REGISTRY: Dict[str, Type[AlgorithmAdapter]] = {
    "ppo": PPOAdapter,
    "recurrent_ppo": RecurrentPPOAdapter,
    "sac": SACAdapter,
}


def register_algorithm(name: str, adapter_cls: Type[AlgorithmAdapter]) -> None:
    """Register a new algorithm adapter.

    Args:
        name: Algorithm name (used in SSOT config)
        adapter_cls: AlgorithmAdapter subclass
    """
    _ALGORITHM_REGISTRY[name] = adapter_cls
    logger.info(f"Registered algorithm: {name}")


def create_algorithm(name: str, **adapter_kwargs) -> AlgorithmAdapter:
    """Create an algorithm adapter by name.

    Args:
        name: Algorithm name from SSOT ("ppo", "recurrent_ppo", "sac")
        **adapter_kwargs: Adapter-specific constructor args (e.g., lstm_hidden_size)

    Returns:
        AlgorithmAdapter instance

    Raises:
        ValueError: If algorithm name is not registered
    """
    if name not in _ALGORITHM_REGISTRY:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")

    adapter_cls = _ALGORITHM_REGISTRY[name]

    # Only pass kwargs that the adapter constructor accepts
    import inspect
    sig = inspect.signature(adapter_cls.__init__)
    valid_kwargs = {
        k: v for k, v in adapter_kwargs.items()
        if k in sig.parameters and k != "self"
    }

    return adapter_cls(**valid_kwargs)


def get_algorithm_kwargs(config) -> Dict[str, Any]:
    """Extract algorithm-specific hyperparameters from SSOT config.

    Args:
        config: PipelineConfig instance

    Returns:
        Dict of kwargs to pass to adapter.create()
    """
    algorithm = resolve_algorithm_name(config)

    if algorithm == "sac":
        return _get_sac_kwargs(config)
    else:
        return _get_ppo_kwargs(config)


def resolve_algorithm_name(config) -> str:
    """Resolve algorithm name from SSOT config with backward compat.

    Priority:
    1. Explicit training.algorithm field
    2. lstm.enabled=true -> "recurrent_ppo"
    3. Default -> "ppo"
    """
    raw = config._raw.get("training", {})
    explicit = raw.get("algorithm")

    if explicit:
        return explicit

    # Backward compat: derive from lstm.enabled
    if config.lstm.enabled:
        return "recurrent_ppo"

    return "ppo"


def _get_ppo_kwargs(config) -> Dict[str, Any]:
    """Build PPO/RecurrentPPO kwargs from SSOT config."""
    ppo = config.ppo
    net_arch = list(
        config._raw.get("training", {}).get("network", {}).get("policy_layers", [256, 256])
    )

    kwargs = {
        "learning_rate": ppo.learning_rate,
        "n_steps": ppo.n_steps,
        "batch_size": ppo.batch_size,
        "n_epochs": ppo.n_epochs,
        "gamma": ppo.gamma,
        "gae_lambda": ppo.gae_lambda,
        "clip_range": ppo.clip_range,
        "ent_coef": ppo.ent_coef,
        "vf_coef": ppo.vf_coef,
        "max_grad_norm": ppo.max_grad_norm,
        "policy_kwargs": {"net_arch": net_arch},
    }
    return kwargs


def _get_sac_kwargs(config) -> Dict[str, Any]:
    """Build SAC kwargs from SSOT config."""
    sac_raw = config._raw.get("training", {}).get("sac", {})
    net_arch = list(
        config._raw.get("training", {}).get("network", {}).get("policy_layers", [256, 256])
    )

    # Handle string "auto" for ent_coef
    ent_coef = sac_raw.get("ent_coef", "auto")

    kwargs = {
        "learning_rate": sac_raw.get("learning_rate", 1e-4),
        "buffer_size": sac_raw.get("buffer_size", 200_000),
        "learning_starts": sac_raw.get("learning_starts", 10_000),
        "batch_size": sac_raw.get("batch_size", 512),
        "tau": sac_raw.get("tau", 0.005),
        "gamma": sac_raw.get("gamma", 0.99),
        "ent_coef": ent_coef,
        "train_freq": sac_raw.get("train_freq", 4),
        "gradient_steps": sac_raw.get("gradient_steps", 4),
        "policy_kwargs": {"net_arch": net_arch},
    }

    # target_entropy: only pass if not "auto"
    target_entropy = sac_raw.get("target_entropy", "auto")
    if target_entropy != "auto":
        kwargs["target_entropy"] = target_entropy

    return kwargs
