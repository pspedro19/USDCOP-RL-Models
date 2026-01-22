# pipeline_limpio_regresion/core/base.py
"""
Abstract base classes following SOLID principles.

- Single Responsibility: Each class has one job
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: Subclasses are substitutable
- Interface Segregation: Specific interfaces
- Dependency Inversion: Depend on abstractions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    """
    Standard result container for all models.

    Ensures consistent output format across different model types.
    """
    model_name: str
    model_type: str
    horizon: int
    train_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    model: Any = None
    params: Dict = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None


class BaseModel(ABC):
    """
    Abstract base class for all regression models.

    Implements Template Method pattern for training workflow.
    """

    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self._model = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance. Template method."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Fit the model to training data.

        Args:
            X: Features array
            y: Target array

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features array

        Returns:
            Predictions array
        """
        pass

    def get_params(self) -> Dict:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        self.params.update(params)
        return self


class BaseStatisticalModel(ABC):
    """
    Abstract base class for statistical time series models.

    Different interface than ML models since they work with time series directly.
    """

    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self._model = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, series: pd.Series) -> 'BaseStatisticalModel':
        """
        Fit to time series.

        Args:
            series: Time series data

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def forecast(self, horizon: int) -> np.ndarray:
        """
        Generate forecast for given horizon.

        Args:
            horizon: Number of steps ahead

        Returns:
            Forecast array
        """
        pass


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    Implements Strategy pattern for different training strategies.
    """

    def __init__(self, config: Any = None):
        self.config = config
        self.results: List[ModelResult] = []
        self.trained_models: Dict[str, Dict[int, Any]] = {}

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        horizon: int
    ) -> ModelResult:
        """
        Train model and return results.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            horizon: Prediction horizon

        Returns:
            ModelResult with metrics and predictions
        """
        pass

    def get_results(self) -> List[ModelResult]:
        """Get all training results."""
        return self.results

    def get_results_df(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        rows = []
        for r in self.results:
            row = {
                'model': r.model_name,
                'type': r.model_type,
                'horizon': r.horizon,
                'train_time': r.train_time,
                **r.metrics
            }
            rows.append(row)
        return pd.DataFrame(rows)


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluators.

    Separates evaluation logic from training.
    """

    @abstractmethod
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate predictions.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        pass

    @abstractmethod
    def compare_models(
        self,
        results: List[ModelResult]
    ) -> pd.DataFrame:
        """
        Compare multiple model results.

        Args:
            results: List of model results

        Returns:
            Comparison DataFrame
        """
        pass
