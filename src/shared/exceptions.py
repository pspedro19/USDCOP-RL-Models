"""
Custom exceptions for USD/COP Trading System.

Defines specific exception types for error handling across the system.

Author: Pedro @ Lean Tech Solutions
Version: 2.0.0
Date: 2025-12-16
"""


class USDCOPError(Exception):
    """Base exception for all USD/COP Trading System errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(USDCOPError):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str, config_path: str = None, missing_key: str = None):
        details = {}
        if config_path:
            details["config_path"] = config_path
        if missing_key:
            details["missing_key"] = missing_key
        super().__init__(message, details)


class FeatureCalculationError(USDCOPError):
    """Exception raised when feature calculation fails."""

    def __init__(self, message: str, feature_name: str = None, error_type: str = None):
        details = {}
        if feature_name:
            details["feature_name"] = feature_name
        if error_type:
            details["error_type"] = error_type
        super().__init__(message, details)


class ValidationError(USDCOPError):
    """Exception raised when observation validation fails."""

    def __init__(self, message: str, expected_shape: tuple = None, actual_shape: tuple = None):
        details = {}
        if expected_shape:
            details["expected_shape"] = expected_shape
        if actual_shape:
            details["actual_shape"] = actual_shape
        super().__init__(message, details)


class ObservationDimensionError(ValidationError):
    """Exception raised when observation dimension doesn't match expected."""

    def __init__(self, expected: int, actual: int):
        message = f"Observation dimension mismatch: expected {expected}, got {actual}"
        super().__init__(
            message,
            expected_shape=(expected,),
            actual_shape=(actual,)
        )
        self.expected = expected
        self.actual = actual


class FeatureMissingError(USDCOPError):
    """Exception raised when a required feature is missing."""

    def __init__(self, feature_name: str):
        message = f"Required feature '{feature_name}' is missing from observation data"
        details = {"feature_name": feature_name}
        super().__init__(message, details)
        self.feature_name = feature_name


# Legacy aliases for backward compatibility
class FeatureBuilderError(USDCOPError):
    """Legacy alias for USDCOPError - use USDCOPError instead."""
    pass


class NormalizationError(FeatureCalculationError):
    """Exception raised when feature normalization fails."""

    def __init__(self, message: str, feature_name: str = None, raw_value: float = None):
        details = {}
        if feature_name:
            details["feature_name"] = feature_name
        if raw_value is not None:
            details["raw_value"] = raw_value
        super().__init__(message, feature_name=feature_name, error_type="normalization")
        if raw_value is not None:
            self.details["raw_value"] = raw_value
