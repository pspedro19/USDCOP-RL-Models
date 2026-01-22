"""
Protocol Interfaces for Dependency Injection
=============================================

Defines Protocol-based interfaces for the ApplicationContext DI container.
These protocols enable compile-time type checking and runtime duck typing.

Design Principles:
- Interface Segregation Principle (ISP): Small, focused interfaces
- Dependency Inversion Principle (DIP): Depend on abstractions
- Protocol Pattern: Structural subtyping for flexibility

Author: Trading Team
Version: 1.0.0
Date: 2026-01-16
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)
from dataclasses import dataclass
from enum import Enum

import numpy as np


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar('T')
EventT = TypeVar('EventT', bound='Event')


# =============================================================================
# BASE EVENT TYPE
# =============================================================================

@dataclass
class Event:
    """Base class for all domain events."""
    event_type: str
    timestamp: str
    payload: Dict[str, Any]


# =============================================================================
# FEATURE BUILDING PROTOCOLS
# =============================================================================

@runtime_checkable
class IFeatureBuilder(Protocol):
    """
    Protocol for building observation vectors from raw market data.

    Implementations should handle:
    - Technical indicator calculation (RSI, ATR, ADX)
    - Macro feature integration
    - Normalization and clipping
    - State variable integration (position, time)
    """

    def build_observation(
        self,
        ohlcv: Dict[str, Any],
        macro: Optional[Dict[str, Any]],
        position: float,
        bar_idx: int,
    ) -> np.ndarray:
        """
        Build observation vector for model inference.

        Args:
            ohlcv: OHLCV data dict with keys: open, high, low, close, volume
            macro: Optional macro data dict (dxy, vix, embi, etc.)
            position: Current position (-1 to 1)
            bar_idx: Current bar index in episode (1-based)

        Returns:
            Observation vector of shape (obs_dim,) ready for model
        """
        ...

    @property
    def observation_dim(self) -> int:
        """Get total observation dimension."""
        ...

    @property
    def feature_order(self) -> List[str]:
        """Get ordered list of feature names."""
        ...


@runtime_checkable
class IFeatureCalculator(Protocol):
    """
    Protocol for calculating individual features.

    Single Responsibility: Calculate one specific feature.
    """

    @property
    def name(self) -> str:
        """Feature name identifier."""
        ...

    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate feature value from input data.

        Args:
            data: Input data dictionary

        Returns:
            Calculated feature value
        """
        ...


@runtime_checkable
class INormalizer(Protocol):
    """
    Protocol for normalizing feature values.

    Implementations may include:
    - Z-score normalization
    - Min-max scaling
    - Clipping
    """

    def normalize(self, value: float, feature_name: str) -> float:
        """
        Normalize a single feature value.

        Args:
            value: Raw feature value
            feature_name: Name of the feature

        Returns:
            Normalized value
        """
        ...


# =============================================================================
# MODEL LOADING PROTOCOLS
# =============================================================================

@runtime_checkable
class IModelLoader(Protocol):
    """
    Protocol for loading model artifacts.

    Supports various model formats: ONNX, SavedModel, pickle, etc.
    """

    def load(self, path: str) -> bool:
        """
        Load model from file path.

        Args:
            path: Path to model file

        Returns:
            True if loaded successfully
        """
        ...

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        ...

    def warmup(self, iterations: int = 5) -> None:
        """
        Warm up model with dummy inference.

        Args:
            iterations: Number of warmup iterations
        """
        ...

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        ...

    @property
    def model_name(self) -> str:
        """Get model name identifier."""
        ...


# =============================================================================
# PREDICTION PROTOCOLS
# =============================================================================

class SignalType(str, Enum):
    """Trading signal types."""
    HOLD = "HOLD"
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class PredictionResult:
    """Result of model prediction."""
    action: int
    confidence: float
    action_probs: Dict[str, float]
    latency_ms: float


@runtime_checkable
class IPredictor(Protocol):
    """
    Protocol for running model predictions.

    Single Responsibility: Execute inference on observations.
    """

    def predict(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Run inference on observation.

        Args:
            observation: Feature vector (1D array)

        Returns:
            Tuple of (action, confidence)
        """
        ...

    def predict_with_probs(self, observation: np.ndarray) -> PredictionResult:
        """
        Run inference with full probability distribution.

        Args:
            observation: Feature vector (1D array)

        Returns:
            PredictionResult with action, confidence, and probabilities
        """
        ...

    @property
    def model_name(self) -> str:
        """Get model name identifier."""
        ...


# =============================================================================
# REPOSITORY PROTOCOLS
# =============================================================================

@runtime_checkable
class ITradeRepository(Protocol):
    """
    Protocol for trade persistence operations.

    Supports saving and querying trade records.
    """

    def save_trade(self, trade: Dict[str, Any]) -> str:
        """
        Save a trade record.

        Args:
            trade: Trade data dictionary

        Returns:
            Trade ID
        """
        ...

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trade by ID.

        Args:
            trade_id: Trade identifier

        Returns:
            Trade data or None if not found
        """
        ...

    def get_trades(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get trades matching filters.

        Args:
            filters: Optional query filters
            limit: Maximum number of trades to return

        Returns:
            List of trade records
        """
        ...

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent trades.

        Args:
            limit: Maximum number of trades

        Returns:
            List of recent trade records
        """
        ...


@runtime_checkable
class IStateRepository(Protocol):
    """
    Protocol for generic key-value state persistence.

    Used for caching, session state, and temporary data.
    """

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state by key."""
        ...

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set state with optional TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete state by key."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...


@runtime_checkable
class IDailyStatsRepository(Protocol):
    """
    Protocol for daily trading statistics persistence.
    """

    def get(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get daily stats for date (default: today)."""
        ...

    def save(self, stats: Dict[str, Any]) -> bool:
        """Save daily stats."""
        ...

    def increment_field(self, date: str, field: str, amount: float = 1.0) -> float:
        """Increment numeric field atomically."""
        ...


# =============================================================================
# EVENT BUS PROTOCOLS
# =============================================================================

EventListener = Callable[[Event], None]


@runtime_checkable
class IEventBus(Protocol):
    """
    Protocol for publish-subscribe event bus.

    Enables loose coupling between components through domain events.
    """

    def subscribe(self, event_type: str, listener: EventListener) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to
            listener: Callback function for events
        """
        ...

    def unsubscribe(self, event_type: str, listener: EventListener) -> None:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from
            listener: Callback function to remove
        """
        ...

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        ...

    def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously.

        Args:
            event: Event to publish
        """
        ...


# =============================================================================
# LOGGING PROTOCOLS
# =============================================================================

@runtime_checkable
class ILogger(Protocol):
    """
    Protocol for structured logging.
    """

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        ...

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        ...


@runtime_checkable
class ILoggerFactory(Protocol):
    """
    Protocol for creating logger instances.

    Factory Pattern: Create loggers with consistent configuration.
    """

    def get_logger(self, name: str) -> ILogger:
        """
        Get or create a logger for the given name.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        ...


# =============================================================================
# HEALTH CHECK PROTOCOLS
# =============================================================================

@runtime_checkable
class IHealthChecker(Protocol):
    """
    Protocol for health check operations.
    """

    def health_check(self) -> Dict[str, Any]:
        """
        Run health check.

        Returns:
            Dictionary with health status and details
        """
        ...

    @property
    def is_healthy(self) -> bool:
        """Quick health status check."""
        ...


# =============================================================================
# RISK MANAGEMENT PROTOCOLS
# =============================================================================

@runtime_checkable
class IRiskManager(Protocol):
    """
    Protocol for risk management operations.
    """

    def check_trade_allowed(
        self,
        signal: SignalType,
        confidence: float,
        position: float,
    ) -> Tuple[bool, str]:
        """
        Check if a trade is allowed by risk rules.

        Args:
            signal: Proposed trading signal
            confidence: Model confidence
            position: Current position

        Returns:
            Tuple of (allowed, reason)
        """
        ...

    def get_position_size(
        self,
        signal: SignalType,
        confidence: float,
    ) -> float:
        """
        Calculate position size for a trade.

        Args:
            signal: Trading signal
            confidence: Model confidence

        Returns:
            Position size (0 to 1)
        """
        ...


# =============================================================================
# CONFIGURATION PROTOCOLS
# =============================================================================

@runtime_checkable
class IConfigProvider(Protocol):
    """
    Protocol for configuration access.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (dot-notation supported)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        ...

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name

        Returns:
            Section as dictionary
        """
        ...


# =============================================================================
# CACHE PROTOCOLS
# =============================================================================

@runtime_checkable
class ICacheProvider(Protocol):
    """
    Protocol for caching operations.
    """

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        ...

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set cached value with TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete cached value."""
        ...

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: int = 300,
    ) -> T:
        """Get from cache or compute and cache."""
        ...


# =============================================================================
# ENSEMBLE STRATEGY PROTOCOLS
# =============================================================================

@runtime_checkable
class IEnsembleStrategy(Protocol):
    """
    Protocol for ensemble combination strategies.

    Strategy Pattern: Different ensemble methods can be plugged in.
    """

    @property
    def name(self) -> str:
        """Get strategy name identifier."""
        ...

    def combine(
        self,
        results: List[PredictionResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Combine multiple prediction results.

        Args:
            results: List of individual model results
            weights: Optional model weights

        Returns:
            Tuple of (action, confidence, combined_probs)
        """
        ...


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base types
    'Event',
    'SignalType',
    'PredictionResult',
    'EventListener',

    # Feature protocols
    'IFeatureBuilder',
    'IFeatureCalculator',
    'INormalizer',

    # Model protocols
    'IModelLoader',
    'IPredictor',
    'IEnsembleStrategy',

    # Repository protocols
    'ITradeRepository',
    'IStateRepository',
    'IDailyStatsRepository',

    # Event protocols
    'IEventBus',

    # Logging protocols
    'ILogger',
    'ILoggerFactory',

    # Health protocols
    'IHealthChecker',

    # Risk protocols
    'IRiskManager',

    # Config protocols
    'IConfigProvider',

    # Cache protocols
    'ICacheProvider',
]
