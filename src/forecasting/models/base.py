"""
Base Model Interface
====================

Abstract base class for all forecasting models.
Implements Template Method pattern for training/prediction lifecycle.

@version 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for forecasting models.

    All models must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - get_params(): Get model parameters
    - set_params(): Set model parameters

    Optional overrides:
    - _preprocess(): Custom preprocessing
    - _postprocess(): Custom postprocessing
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize model.

        Args:
            name: Model identifier
            params: Model hyperparameters
        """
        self.name = name
        self.params = params or {}
        self._model = None
        self._is_fitted = False
        self._feature_names: Optional[list] = None
        self._training_metrics: Dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Train the model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass

    @abstractmethod
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        pass

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess features before training/prediction.
        Override in subclasses for custom preprocessing.

        Args:
            X: Raw features

        Returns:
            Preprocessed features
        """
        return X

    def _postprocess(self, predictions: np.ndarray) -> np.ndarray:
        """
        Postprocess predictions.
        Override in subclasses for custom postprocessing.

        Args:
            predictions: Raw predictions

        Returns:
            Postprocessed predictions
        """
        return predictions

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    @property
    def requires_scaling(self) -> bool:
        """
        Whether this model requires feature scaling.
        Override in subclasses that need scaling (e.g., linear models).
        """
        return False

    @property
    def supports_early_stopping(self) -> bool:
        """
        Whether this model supports early stopping.
        Override in subclasses that support it (e.g., boosting models).
        """
        return False

    @property
    def feature_names(self) -> Optional[list]:
        """Get feature names if available."""
        return self._feature_names

    @feature_names.setter
    def feature_names(self, names: list):
        """Set feature names."""
        self._feature_names = names

    @property
    def training_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return self._training_metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        Override in subclasses that support feature importance.

        Returns:
            Dict mapping feature names to importance scores, or None
        """
        return None

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save to
        """
        import joblib
        joblib.dump({
            'name': self.name,
            'params': self.params,
            'model': self._model,
            'is_fitted': self._is_fitted,
            'feature_names': self._feature_names,
            'training_metrics': self._training_metrics,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load model from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded model instance
        """
        import joblib
        data = joblib.load(path)

        instance = cls.__new__(cls)
        instance.name = data['name']
        instance.params = data['params']
        instance._model = data['model']
        instance._is_fitted = data['is_fitted']
        instance._feature_names = data.get('feature_names')
        instance._training_metrics = data.get('training_metrics', {})

        logger.info(f"Model loaded from {path}")
        return instance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self.is_fitted})"
