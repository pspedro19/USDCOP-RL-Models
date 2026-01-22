"""
ModelLoaderFactory - Factory Pattern for Model Loaders
======================================================

Creates appropriate model loader instances based on model format.
Supports ONNX, SB3 (stable-baselines3), and TorchScript formats.

Design Patterns:
- Factory Pattern: Centralizes loader creation
- Strategy Pattern: Loaders are interchangeable strategies
- Protocol (Structural Subtyping): Defines loader interface

Author: Trading Team
Version: 1.0.0
Date: 2025-01-16
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class LoaderType(str, Enum):
    """Supported model loader types."""
    ONNX = "onnx"
    SB3 = "sb3"
    TORCHSCRIPT = "torchscript"


@dataclass
class ModelLoaderConfig:
    """
    Configuration for model loaders.

    Attributes:
        providers: Execution providers for ONNX (e.g., CPUExecutionProvider)
        device: Device for model inference (cpu, cuda)
        warmup_iterations: Number of warmup iterations
        optimize: Whether to enable model optimization
        num_threads: Number of inference threads
    """
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    device: str = "cpu"
    warmup_iterations: int = 10
    optimize: bool = True
    num_threads: int = 2


# =============================================================================
# Protocol Definition
# =============================================================================

@runtime_checkable
class IModelLoaderProtocol(Protocol):
    """
    Common interface for model loaders.

    Uses Python's Protocol for structural subtyping, allowing any class
    that implements these methods to be used as a loader.
    """

    @property
    def name(self) -> str:
        """Get loader/model name identifier."""
        ...

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        ...

    def load(self, path: str, providers: Optional[List[str]] = None) -> bool:
        """
        Load model from path.

        Args:
            path: Path to model file
            providers: Optional execution providers

        Returns:
            True if loaded successfully
        """
        ...

    def warmup(self, iterations: int = 10) -> None:
        """
        Warm up model with dummy inference.

        Args:
            iterations: Number of warmup iterations
        """
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        ...

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Run inference on observation.

        Args:
            observation: Input observation array

        Returns:
            Model output array
        """
        ...


# =============================================================================
# Factory Implementation
# =============================================================================

class ModelLoaderFactory:
    """
    Factory for creating model loaders.

    Implements the Factory Pattern to centralize loader creation.
    Supports registration of custom loader types for extensibility.

    Usage:
        # Register loaders (done at application startup)
        ModelLoaderFactory.register(LoaderType.ONNX, ONNXModelLoader)
        ModelLoaderFactory.register(LoaderType.SB3, SB3ModelLoader)

        # Create loader
        config = ModelLoaderConfig(device="cpu", warmup_iterations=5)
        loader = ModelLoaderFactory.create(LoaderType.ONNX, "my_model", config)

        # Load and use model
        if loader.load("model.onnx"):
            loader.warmup()
            output = loader.predict(observation)
    """

    _loaders: Dict[LoaderType, Type[IModelLoaderProtocol]] = {}
    _instances: Dict[str, IModelLoaderProtocol] = {}

    @classmethod
    def register(cls, loader_type: LoaderType, loader_class: Type[IModelLoaderProtocol]) -> None:
        """
        Register a loader class for a given type.

        Args:
            loader_type: Type of model loader
            loader_class: Loader class implementing IModelLoaderProtocol
        """
        cls._loaders[loader_type] = loader_class
        logger.info(f"Registered loader '{loader_class.__name__}' for type '{loader_type.value}'")

    @classmethod
    def unregister(cls, loader_type: LoaderType) -> bool:
        """
        Unregister a loader for a given type.

        Args:
            loader_type: Loader type to unregister

        Returns:
            True if unregistered, False if not found
        """
        if loader_type in cls._loaders:
            del cls._loaders[loader_type]
            logger.info(f"Unregistered loader for type '{loader_type.value}'")
            return True
        return False

    @classmethod
    def create(
        cls,
        loader_type: LoaderType,
        name: str,
        config: Optional[ModelLoaderConfig] = None,
    ) -> IModelLoaderProtocol:
        """
        Create a model loader of the specified type.

        Args:
            loader_type: Type of loader to create
            name: Model name identifier
            config: Loader configuration

        Returns:
            IModelLoaderProtocol instance

        Raises:
            ValueError: If loader type is not registered

        Example:
            >>> config = ModelLoaderConfig(device="cpu")
            >>> loader = ModelLoaderFactory.create(LoaderType.ONNX, "ppo_v1", config)
        """
        if loader_type not in cls._loaders:
            available = [t.value for t in cls._loaders.keys()]
            raise ValueError(
                f"Unknown loader type: '{loader_type.value}'. "
                f"Available types: {available}"
            )

        config = config or ModelLoaderConfig()
        loader_class = cls._loaders[loader_type]

        try:
            loader = loader_class(name=name, config=config)
            logger.info(f"Created {loader_type.value} loader for model '{name}'")
            return loader
        except Exception as e:
            logger.error(f"Failed to create {loader_type.value} loader: {e}")
            raise

    @classmethod
    def create_from_path(
        cls,
        path: str,
        name: Optional[str] = None,
        config: Optional[ModelLoaderConfig] = None,
    ) -> IModelLoaderProtocol:
        """
        Create a model loader based on file extension.

        Args:
            path: Path to model file
            name: Optional model name (defaults to filename)
            config: Loader configuration

        Returns:
            IModelLoaderProtocol instance

        Example:
            >>> loader = ModelLoaderFactory.create_from_path("models/ppo_v1.onnx")
        """
        # Determine loader type from extension
        ext = os.path.splitext(path)[1].lower()
        type_map = {
            ".onnx": LoaderType.ONNX,
            ".zip": LoaderType.SB3,  # SB3 models are zipped
            ".pt": LoaderType.TORCHSCRIPT,
            ".pth": LoaderType.TORCHSCRIPT,
        }

        loader_type = type_map.get(ext)
        if loader_type is None:
            raise ValueError(f"Unknown model format: {ext}")

        if name is None:
            name = os.path.splitext(os.path.basename(path))[0]

        return cls.create(loader_type, name, config)

    @classmethod
    def get_singleton(
        cls,
        loader_type: LoaderType,
        name: str,
        config: Optional[ModelLoaderConfig] = None,
    ) -> IModelLoaderProtocol:
        """
        Get or create a singleton loader instance.

        Args:
            loader_type: Type of loader
            name: Model name identifier
            config: Loader configuration (ignored if exists)

        Returns:
            IModelLoaderProtocol instance
        """
        key = f"{loader_type.value}:{name}"

        if key not in cls._instances:
            cls._instances[key] = cls.create(loader_type, name, config)

        return cls._instances[key]

    @classmethod
    def clear_singletons(cls) -> None:
        """Clear all singleton instances."""
        cls._instances.clear()
        logger.info("Cleared all singleton loader instances")

    @classmethod
    def get_registered_types(cls) -> List[LoaderType]:
        """Get list of all registered loader types."""
        return list(cls._loaders.keys())

    @classmethod
    def is_registered(cls, loader_type: LoaderType) -> bool:
        """Check if a loader type is registered."""
        return loader_type in cls._loaders


# =============================================================================
# Default Loader Implementations
# =============================================================================

class BaseModelLoader:
    """
    Base class for model loaders with common functionality.
    """

    def __init__(self, name: str, config: ModelLoaderConfig):
        self._name = name
        self._config = config
        self._loaded = False
        self._input_shape: Tuple[int, ...] = ()
        self._model: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str, providers: Optional[List[str]] = None) -> bool:
        raise NotImplementedError

    def warmup(self, iterations: int = 10) -> None:
        raise NotImplementedError

    def predict(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ONNXLoader(BaseModelLoader):
    """
    ONNX model loader.

    Loads and runs inference on ONNX format models using onnxruntime.
    """

    def __init__(self, name: str, config: ModelLoaderConfig):
        super().__init__(name, config)
        self._session = None
        self._input_name: str = ""
        self._output_names: List[str] = []

    def load(self, path: str, providers: Optional[List[str]] = None) -> bool:
        """Load ONNX model from path."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            return False

        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False

        providers = providers or self._config.providers

        try:
            # Session options for optimization
            sess_options = ort.SessionOptions()
            if self._config.optimize:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = self._config.num_threads
            sess_options.inter_op_num_threads = self._config.num_threads

            self._session = ort.InferenceSession(
                path,
                sess_options=sess_options,
                providers=providers
            )

            # Extract metadata
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]

            # Get input shape
            input_info = self._session.get_inputs()[0]
            self._input_shape = tuple(
                d if isinstance(d, int) else 1
                for d in input_info.shape
            )

            self._loaded = True

            logger.info(f"Loaded ONNX model '{self._name}' from {path}")
            logger.debug(f"  Input: {self._input_name}, shape: {self._input_shape}")
            logger.debug(f"  Outputs: {self._output_names}")

            return True

        except Exception as e:
            logger.error(f"Failed to load ONNX model '{self._name}': {e}")
            return False

    def warmup(self, iterations: int = 10) -> None:
        """Warm up model with dummy inference."""
        if not self._loaded:
            raise RuntimeError(f"Model '{self._name}' not loaded")

        iterations = iterations or self._config.warmup_iterations

        logger.info(f"Warming up ONNX model '{self._name}'...")

        obs_dim = self._input_shape[-1] if self._input_shape else 45
        dummy_input = np.random.randn(1, obs_dim).astype(np.float32)

        for _ in range(iterations):
            self._session.run(None, {self._input_name: dummy_input})

        logger.info(f"Warmup complete for '{self._name}'")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on observation."""
        if not self._loaded:
            raise RuntimeError(f"Model '{self._name}' not loaded")

        # Ensure correct shape and dtype
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        observation = observation.astype(np.float32)

        outputs = self._session.run(None, {self._input_name: observation})
        return outputs[0]

    @property
    def session(self) -> Any:
        """Get underlying ONNX session."""
        return self._session


class SB3Loader(BaseModelLoader):
    """
    Stable-Baselines3 model loader.

    Loads and runs inference on SB3 format models.
    """

    def __init__(self, name: str, config: ModelLoaderConfig):
        super().__init__(name, config)
        self._algorithm = None

    def load(self, path: str, providers: Optional[List[str]] = None) -> bool:
        """Load SB3 model from path."""
        try:
            from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
        except ImportError:
            logger.error("stable-baselines3 not installed. Install with: pip install stable-baselines3")
            return False

        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False

        # Determine algorithm from filename or try common ones
        algorithms = [PPO, SAC, TD3, A2C, DQN]
        filename = os.path.basename(path).lower()

        # Try to infer algorithm from filename
        for algo in algorithms:
            if algo.__name__.lower() in filename:
                algorithms = [algo]
                break

        device = self._config.device

        for algo_class in algorithms:
            try:
                self._model = algo_class.load(path, device=device)
                self._algorithm = algo_class.__name__

                # Get input shape from observation space
                obs_space = self._model.observation_space
                if hasattr(obs_space, 'shape'):
                    self._input_shape = obs_space.shape

                self._loaded = True

                logger.info(f"Loaded SB3 {self._algorithm} model '{self._name}' from {path}")
                return True

            except Exception:
                continue

        logger.error(f"Failed to load SB3 model '{self._name}' from {path}")
        return False

    def warmup(self, iterations: int = 10) -> None:
        """Warm up model with dummy inference."""
        if not self._loaded:
            raise RuntimeError(f"Model '{self._name}' not loaded")

        logger.info(f"Warming up SB3 model '{self._name}'...")

        obs_dim = self._input_shape[0] if self._input_shape else 45
        dummy_input = np.random.randn(obs_dim).astype(np.float32)

        for _ in range(iterations):
            self._model.predict(dummy_input, deterministic=True)

        logger.info(f"Warmup complete for '{self._name}'")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference on observation."""
        if not self._loaded:
            raise RuntimeError(f"Model '{self._name}' not loaded")

        observation = observation.astype(np.float32)
        if observation.ndim == 2 and observation.shape[0] == 1:
            observation = observation.squeeze(0)

        action, _ = self._model.predict(observation, deterministic=True)
        return np.atleast_1d(action)


# =============================================================================
# Auto-register default loaders
# =============================================================================

ModelLoaderFactory.register(LoaderType.ONNX, ONNXLoader)
ModelLoaderFactory.register(LoaderType.SB3, SB3Loader)
