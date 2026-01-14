"""
Model Loader
============

Single Responsibility: Load and initialize model artifacts.

Author: Trading Team
Version: 1.0.0
Date: 2025-01-14
"""

import os
import logging
from typing import Optional, List, Tuple, Any

import numpy as np

from src.core.interfaces.inference import IModelLoader

logger = logging.getLogger(__name__)


class ONNXModelLoader(IModelLoader):
    """
    ONNX model loader.

    Single Responsibility: Load ONNX models and prepare for inference.
    """

    def __init__(self, name: str = "default"):
        """
        Args:
            name: Model identifier
        """
        self._name = name
        self._session = None
        self._input_name: str = ""
        self._output_names: List[str] = []
        self._input_shape: Tuple[int, ...] = ()
        self._loaded = False
        self._providers: List[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str, providers: Optional[List[str]] = None) -> bool:
        """
        Load ONNX model from path.

        Args:
            path: Path to .onnx file
            providers: Execution providers (default: CPUExecutionProvider)

        Returns:
            True if loaded successfully
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime not installed")
            return False

        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False

        providers = providers or ['CPUExecutionProvider']
        self._providers = providers

        try:
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2

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

            logger.info(f"Loaded model '{self._name}' from {path}")
            logger.debug(f"  Input: {self._input_name}, shape: {self._input_shape}")
            logger.debug(f"  Outputs: {self._output_names}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model '{self._name}': {e}")
            return False

    def warmup(self, iterations: int = 10) -> None:
        """
        Warm up model with dummy inference.

        Args:
            iterations: Number of warmup iterations
        """
        if not self._loaded:
            raise RuntimeError(f"Model '{self._name}' not loaded")

        logger.info(f"Warming up model '{self._name}'...")

        # Determine observation dimension
        obs_dim = self._input_shape[-1] if self._input_shape else 45
        dummy_input = np.random.randn(1, obs_dim).astype(np.float32)

        for _ in range(iterations):
            self._session.run(None, {self._input_name: dummy_input})

        logger.info(f"Warmup complete for '{self._name}'")

    @property
    def session(self) -> Any:
        """Get underlying ONNX session."""
        return self._session

    @property
    def input_name(self) -> str:
        """Get input tensor name."""
        return self._input_name

    @property
    def output_names(self) -> List[str]:
        """Get output tensor names."""
        return self._output_names

    @property
    def providers(self) -> List[str]:
        """Get execution providers."""
        return self._providers
