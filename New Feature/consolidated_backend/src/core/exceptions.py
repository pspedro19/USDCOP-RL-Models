# pipeline_limpio_regresion/core/exceptions.py
"""
Custom exceptions for the pipeline.

Provides specific error types for better error handling and debugging.
"""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataValidationError(PipelineError):
    """Raised when data validation fails."""

    def __init__(self, message: str, issues: list = None):
        super().__init__(message, {'issues': issues or []})
        self.issues = issues or []


class ModelTrainingError(PipelineError):
    """Raised when model training fails."""

    def __init__(self, message: str, model_name: str = None, horizon: int = None):
        super().__init__(message, {'model': model_name, 'horizon': horizon})
        self.model_name = model_name
        self.horizon = horizon


class ForecastingError(PipelineError):
    """Raised when forecasting fails."""

    def __init__(self, message: str, model_name: str = None, horizon: int = None):
        super().__init__(message, {'model': model_name, 'horizon': horizon})
        self.model_name = model_name
        self.horizon = horizon


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, param: str = None):
        super().__init__(message, {'param': param})
        self.param = param


class FeatureEngineeringError(PipelineError):
    """Raised when feature engineering fails."""

    def __init__(self, message: str, feature_name: str = None):
        super().__init__(message, {'feature': feature_name})
        self.feature_name = feature_name
