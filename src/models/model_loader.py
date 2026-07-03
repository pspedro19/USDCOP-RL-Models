"""
Model Loader for loading SB3 models from various sources.

This module handles the actual loading of reinforcement learning models
from local files, MinIO object storage, and ONNX format.
"""

import os
import tempfile
import logging
from typing import Optional, Union, Type
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Import SB3 algorithms with graceful fallback
try:
    from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
    from stable_baselines3.common.base_class import BaseAlgorithm
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseAlgorithm = object
    logger.warning("stable-baselines3 not installed. Model loading will be limited.")

# Import ONNX Runtime with graceful fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None
    logger.warning("onnxruntime not installed. ONNX model loading will be unavailable.")

# Import MinIO client with graceful fallback
try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    logger.warning("minio not installed. MinIO model loading will be unavailable.")


class ModelLoader:
    """
    Loads SB3 models from various sources.

    Supports loading from:
    - Local .zip files (SB3 format)
    - MinIO object storage
    - ONNX format for optimized inference

    Attributes:
        SUPPORTED_ALGORITHMS: List of supported SB3 algorithm names
    """

    SUPPORTED_ALGORITHMS = ['PPO', 'SAC', 'TD3', 'A2C', 'DQN']

    def __init__(self, minio_config: Optional[dict] = None):
        """
        Initialize the ModelLoader.

        Args:
            minio_config: Optional MinIO configuration dict with keys:
                - endpoint: MinIO server endpoint
                - access_key: Access key
                - secret_key: Secret key
                - secure: Whether to use HTTPS (default: True)
        """
        self._algorithm_map = {}
        if SB3_AVAILABLE:
            self._algorithm_map = {
                'PPO': PPO,
                'SAC': SAC,
                'TD3': TD3,
                'A2C': A2C,
                'DQN': DQN
            }

        self._minio_client = None
        if minio_config and MINIO_AVAILABLE:
            self._minio_client = Minio(
                minio_config['endpoint'],
                access_key=minio_config['access_key'],
                secret_key=minio_config['secret_key'],
                secure=minio_config.get('secure', True)
            )

    def _get_algorithm_class(self, algorithm: str) -> Type[BaseAlgorithm]:
        """
        Get the algorithm class for a given algorithm name.

        Args:
            algorithm: Algorithm name (PPO, SAC, TD3, A2C, DQN)

        Returns:
            Algorithm class

        Raises:
            ValueError: If algorithm is not supported
            RuntimeError: If stable-baselines3 is not installed
        """
        if not SB3_AVAILABLE:
            raise RuntimeError(
                "stable-baselines3 is required for loading models. "
                "Install with: pip install stable-baselines3"
            )

        algorithm_upper = algorithm.upper()
        if algorithm_upper not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {self.SUPPORTED_ALGORITHMS}"
            )

        return self._algorithm_map[algorithm_upper]

    def load_from_file(
        self,
        path: str,
        algorithm: str,
        device: str = 'auto',
        custom_objects: Optional[dict] = None
    ) -> BaseAlgorithm:
        """
        Load model from local .zip file.

        Args:
            path: Path to the .zip model file
            algorithm: Algorithm type (PPO, SAC, TD3, A2C, DQN)
            device: Device to load model on ('auto', 'cpu', 'cuda')
            custom_objects: Optional dict of custom objects for loading

        Returns:
            Loaded SB3 BaseAlgorithm model

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If algorithm is not supported
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        algorithm_class = self._get_algorithm_class(algorithm)

        logger.info(f"Loading {algorithm} model from {path}")

        try:
            model = algorithm_class.load(
                str(path),
                device=device,
                custom_objects=custom_objects
            )
            logger.info(f"Successfully loaded {algorithm} model from {path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise

    def load_from_minio(
        self,
        bucket: str,
        key: str,
        algorithm: str,
        device: str = 'auto'
    ) -> BaseAlgorithm:
        """
        Load model from MinIO object storage.

        Args:
            bucket: MinIO bucket name
            key: Object key (path within bucket)
            algorithm: Algorithm type (PPO, SAC, TD3, A2C, DQN)
            device: Device to load model on ('auto', 'cpu', 'cuda')

        Returns:
            Loaded SB3 BaseAlgorithm model

        Raises:
            RuntimeError: If MinIO client is not configured
            ValueError: If algorithm is not supported
        """
        if not MINIO_AVAILABLE:
            raise RuntimeError(
                "minio is required for loading from MinIO. "
                "Install with: pip install minio"
            )

        if self._minio_client is None:
            raise RuntimeError(
                "MinIO client not configured. Pass minio_config to ModelLoader."
            )

        logger.info(f"Downloading model from MinIO: {bucket}/{key}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Download from MinIO
            self._minio_client.fget_object(bucket, key, tmp_path)
            logger.info(f"Downloaded model to temporary file: {tmp_path}")

            # Load the model
            model = self.load_from_file(tmp_path, algorithm, device)

            return model

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_onnx(
        self,
        path: str,
        providers: Optional[list] = None
    ) -> 'ort.InferenceSession':
        """
        Load ONNX-exported model for faster inference.

        ONNX models provide significantly faster inference compared to
        native SB3 models, especially on CPU.

        Args:
            path: Path to the .onnx model file
            providers: Optional list of execution providers
                      (default: ['CPUExecutionProvider'])

        Returns:
            ONNX Runtime InferenceSession

        Raises:
            RuntimeError: If onnxruntime is not installed
            FileNotFoundError: If model file doesn't exist
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is required for loading ONNX models. "
                "Install with: pip install onnxruntime"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {path}")

        if providers is None:
            providers = ['CPUExecutionProvider']

        logger.info(f"Loading ONNX model from {path}")

        try:
            session = ort.InferenceSession(
                str(path),
                providers=providers
            )

            # Log model info
            input_info = session.get_inputs()
            output_info = session.get_outputs()
            logger.info(
                f"ONNX model loaded. "
                f"Inputs: {[i.name for i in input_info]}, "
                f"Outputs: {[o.name for o in output_info]}"
            )

            return session

        except Exception as e:
            logger.error(f"Failed to load ONNX model from {path}: {e}")
            raise

    def load_onnx_from_minio(
        self,
        bucket: str,
        key: str,
        providers: Optional[list] = None
    ) -> 'ort.InferenceSession':
        """
        Load ONNX model from MinIO object storage.

        Args:
            bucket: MinIO bucket name
            key: Object key (path within bucket)
            providers: Optional list of execution providers

        Returns:
            ONNX Runtime InferenceSession
        """
        if not MINIO_AVAILABLE:
            raise RuntimeError(
                "minio is required for loading from MinIO. "
                "Install with: pip install minio"
            )

        if self._minio_client is None:
            raise RuntimeError(
                "MinIO client not configured. Pass minio_config to ModelLoader."
            )

        logger.info(f"Downloading ONNX model from MinIO: {bucket}/{key}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as tmp_file:
            tmp_path = tmp_file.name

        try:
            self._minio_client.fget_object(bucket, key, tmp_path)
            return self.load_onnx(tmp_path, providers)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def validate_model(
        self,
        model: Union[BaseAlgorithm, 'ort.InferenceSession'],
        feature_dim: int,
        action_dim: Optional[int] = None
    ) -> bool:
        """
        Validate model input/output dimensions.

        Args:
            model: SB3 model or ONNX session to validate
            feature_dim: Expected input feature dimension
            action_dim: Optional expected action dimension

        Returns:
            True if dimensions match, False otherwise
        """
        try:
            if ONNX_AVAILABLE and isinstance(model, ort.InferenceSession):
                return self._validate_onnx_model(model, feature_dim, action_dim)
            else:
                return self._validate_sb3_model(model, feature_dim, action_dim)

        except Exception as e:
            logger.warning(f"Model validation failed with error: {e}")
            return False

    def _validate_sb3_model(
        self,
        model: BaseAlgorithm,
        feature_dim: int,
        action_dim: Optional[int] = None
    ) -> bool:
        """Validate SB3 model dimensions."""
        try:
            # Get observation space dimension
            obs_shape = model.observation_space.shape
            if len(obs_shape) == 1:
                actual_feature_dim = obs_shape[0]
            else:
                actual_feature_dim = np.prod(obs_shape)

            if actual_feature_dim != feature_dim:
                logger.warning(
                    f"Feature dimension mismatch. "
                    f"Expected: {feature_dim}, Actual: {actual_feature_dim}"
                )
                return False

            # Validate action dimension if provided
            if action_dim is not None:
                action_shape = model.action_space.shape
                if len(action_shape) == 0:
                    # Discrete action space
                    actual_action_dim = model.action_space.n
                else:
                    actual_action_dim = action_shape[0]

                if actual_action_dim != action_dim:
                    logger.warning(
                        f"Action dimension mismatch. "
                        f"Expected: {action_dim}, Actual: {actual_action_dim}"
                    )
                    return False

            logger.debug(
                f"Model validated. Feature dim: {actual_feature_dim}, "
                f"Action space: {model.action_space}"
            )
            return True

        except Exception as e:
            logger.warning(f"SB3 model validation error: {e}")
            return False

    def _validate_onnx_model(
        self,
        session: 'ort.InferenceSession',
        feature_dim: int,
        action_dim: Optional[int] = None
    ) -> bool:
        """Validate ONNX model dimensions."""
        try:
            inputs = session.get_inputs()
            if not inputs:
                logger.warning("ONNX model has no inputs")
                return False

            # Check input dimension
            input_shape = inputs[0].shape
            # Handle dynamic dimensions (usually first dim is batch size)
            actual_feature_dim = input_shape[-1] if len(input_shape) > 1 else input_shape[0]

            # Skip validation if dimension is dynamic
            if isinstance(actual_feature_dim, str):
                logger.debug("ONNX model has dynamic input dimension, skipping validation")
                return True

            if actual_feature_dim != feature_dim:
                logger.warning(
                    f"Feature dimension mismatch. "
                    f"Expected: {feature_dim}, Actual: {actual_feature_dim}"
                )
                return False

            logger.debug(f"ONNX model validated. Input shape: {input_shape}")
            return True

        except Exception as e:
            logger.warning(f"ONNX model validation error: {e}")
            return False

    def export_to_onnx(
        self,
        model: BaseAlgorithm,
        output_path: str,
        opset_version: int = 11
    ) -> str:
        """
        Export SB3 model to ONNX format.

        Args:
            model: SB3 model to export
            output_path: Path for the output .onnx file
            opset_version: ONNX opset version (default: 11)

        Returns:
            Path to the exported ONNX model
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for ONNX export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to ONNX: {output_path}")

        try:
            # Get policy network
            policy = model.policy

            # Create dummy input
            obs_shape = model.observation_space.shape
            dummy_input = torch.zeros(1, *obs_shape)

            # Export to ONNX
            torch.onnx.export(
                policy,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=['observation'],
                output_names=['action'],
                dynamic_axes={
                    'observation': {0: 'batch_size'},
                    'action': {0: 'batch_size'}
                }
            )

            logger.info(f"Successfully exported model to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            raise

    def get_model_info(self, model: BaseAlgorithm) -> dict:
        """
        Get information about a loaded model.

        Args:
            model: Loaded SB3 model

        Returns:
            Dictionary containing model information
        """
        if not SB3_AVAILABLE:
            return {'error': 'stable-baselines3 not available'}

        try:
            info = {
                'algorithm': model.__class__.__name__,
                'observation_space': {
                    'shape': model.observation_space.shape,
                    'dtype': str(model.observation_space.dtype)
                },
                'action_space': {
                    'type': model.action_space.__class__.__name__,
                },
                'device': str(model.device),
                'num_timesteps': model.num_timesteps,
            }

            # Add action space details
            if hasattr(model.action_space, 'shape'):
                info['action_space']['shape'] = model.action_space.shape
            if hasattr(model.action_space, 'n'):
                info['action_space']['n'] = model.action_space.n
            if hasattr(model.action_space, 'low'):
                info['action_space']['low'] = model.action_space.low.tolist()
                info['action_space']['high'] = model.action_space.high.tolist()

            # Add policy architecture if available
            if hasattr(model.policy, 'net_arch'):
                info['policy_net_arch'] = model.policy.net_arch

            return info

        except Exception as e:
            logger.warning(f"Error getting model info: {e}")
            return {'error': str(e)}
