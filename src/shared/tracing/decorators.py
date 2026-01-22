"""
Tracing Decorators for USDCOP Trading Project.

This module provides decorators and utilities for adding distributed tracing
to functions and methods, with special support for ML operations.

Features:
    - @traced decorator for sync functions
    - @traced_async decorator for async functions
    - MLSpanBuilder for ML-specific spans
    - Helper functions for span attribute manipulation

Usage:
    from shared.tracing import traced, traced_async, MLSpanBuilder

    @traced(name="process_data", attributes={"data.type": "market"})
    def process_data(data):
        return transform(data)

    @traced_async(name="async_fetch")
    async def fetch_data():
        return await client.get("/data")

    # For ML operations
    ml_span = MLSpanBuilder()
    with ml_span.inference("model_v1", features_count=15):
        result = model.predict(features)
"""

import functools
import time
import logging
from typing import Optional, Dict, Any, Callable, TypeVar, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Type variables for generic decorators
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Any])


def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
    record_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to trace synchronous functions.

    Creates a span around the decorated function, automatically recording
    timing and optionally exceptions and return values.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to the span
        record_exception: Whether to record exceptions in the span
        record_result: Whether to record the result as an attribute

    Returns:
        Decorated function

    Example:
        @traced(name="calculate_features", attributes={"feature.version": "v2"})
        def calculate_features(data):
            # Processing...
            return features

        @traced(record_result=True)
        def get_signal(observation):
            return model.predict(observation)
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .otel_setup import get_tracer, record_exception as otel_record_exception

            tracer = get_tracer(func.__module__)

            with tracer.start_as_current_span(span_name) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                start_time = time.perf_counter()

                try:
                    result = func(*args, **kwargs)

                    # Record result if requested
                    if record_result and result is not None:
                        _record_result_attribute(span, result)

                    return result

                except Exception as e:
                    if record_exception:
                        otel_record_exception(e)
                    raise

                finally:
                    # Record execution time
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("execution.time_ms", elapsed_ms)

        return wrapper  # type: ignore

    return decorator


def traced_async(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
    record_result: bool = False,
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to trace asynchronous functions.

    Creates a span around the decorated async function, automatically recording
    timing and optionally exceptions and return values.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to the span
        record_exception: Whether to record exceptions in the span
        record_result: Whether to record the result as an attribute

    Returns:
        Decorated async function

    Example:
        @traced_async(name="fetch_market_data")
        async def fetch_market_data(symbol: str):
            return await client.get_ohlcv(symbol)

        @traced_async(attributes={"db.system": "postgresql"})
        async def query_trades():
            return await db.fetch_trades()
    """
    def decorator(func: AsyncF) -> AsyncF:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from .otel_setup import get_tracer, record_exception as otel_record_exception

            tracer = get_tracer(func.__module__)

            with tracer.start_as_current_span(span_name) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function metadata
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                start_time = time.perf_counter()

                try:
                    result = await func(*args, **kwargs)

                    # Record result if requested
                    if record_result and result is not None:
                        _record_result_attribute(span, result)

                    return result

                except Exception as e:
                    if record_exception:
                        otel_record_exception(e)
                    raise

                finally:
                    # Record execution time
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    span.set_attribute("execution.time_ms", elapsed_ms)

        return wrapper  # type: ignore

    return decorator


def _record_result_attribute(span, result: Any) -> None:
    """Record result as span attribute based on type."""
    if isinstance(result, (str, int, float, bool)):
        span.set_attribute("result.value", result)
    elif isinstance(result, dict):
        span.set_attribute("result.type", "dict")
        span.set_attribute("result.keys", str(list(result.keys())[:10]))
    elif isinstance(result, (list, tuple)):
        span.set_attribute("result.type", type(result).__name__)
        span.set_attribute("result.length", len(result))
    else:
        span.set_attribute("result.type", type(result).__name__)


class MLSpanBuilder:
    """
    Builder class for creating ML-specific spans.

    Provides context managers for common ML operations with pre-configured
    attributes relevant to machine learning inference and feature processing.

    Attributes tracked:
        - model_id: Model identifier
        - model_version: Model version
        - feature_version: Feature set version
        - feature_count: Number of features
        - confidence: Prediction confidence
        - prediction_type: Type of prediction (classification, regression)
        - inference_latency_ms: Inference timing

    Example:
        ml_span = MLSpanBuilder()

        # Track model inference
        with ml_span.inference("ppo_primary", feature_count=15) as span:
            signal, action, confidence = model.predict_signal(obs)
            span.set_attribute("ml.confidence", confidence)
            span.set_attribute("ml.signal", signal)

        # Track feature building
        with ml_span.feature_build("v2", feature_names=["rsi", "atr"]) as span:
            features = builder.build(data)
    """

    def __init__(self, tracer_name: str = "ml.operations"):
        """
        Initialize MLSpanBuilder.

        Args:
            tracer_name: Name for the tracer (default: "ml.operations")
        """
        self.tracer_name = tracer_name
        self._tracer = None

    @property
    def tracer(self):
        """Lazy load tracer."""
        if self._tracer is None:
            from .otel_setup import get_tracer
            self._tracer = get_tracer(self.tracer_name)
        return self._tracer

    @contextmanager
    def inference(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        feature_count: Optional[int] = None,
        deterministic: bool = True,
        **extra_attributes: Any,
    ):
        """
        Context manager for model inference spans.

        Args:
            model_id: Model identifier (e.g., "ppo_primary")
            model_version: Optional model version
            feature_count: Number of input features
            deterministic: Whether inference is deterministic
            **extra_attributes: Additional attributes to record

        Yields:
            Active span for adding additional attributes

        Example:
            with ml_span.inference("ppo_primary", feature_count=15) as span:
                action, confidence = model.predict(observation)
                span.set_attribute("ml.action", action)
                span.set_attribute("ml.confidence", confidence)
        """
        with self.tracer.start_as_current_span("ml.inference") as span:
            span.set_attribute("ml.model_id", model_id)
            span.set_attribute("ml.operation", "inference")
            span.set_attribute("ml.deterministic", deterministic)

            if model_version:
                span.set_attribute("ml.model_version", model_version)

            if feature_count is not None:
                span.set_attribute("ml.feature_count", feature_count)

            # Add any extra attributes
            for key, value in extra_attributes.items():
                span.set_attribute(f"ml.{key}", value)

            start_time = time.perf_counter()

            try:
                yield span
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("ml.inference_latency_ms", elapsed_ms)

    @contextmanager
    def feature_build(
        self,
        feature_version: str,
        feature_names: Optional[list] = None,
        source: str = "database",
        **extra_attributes: Any,
    ):
        """
        Context manager for feature building spans.

        Args:
            feature_version: Version of the feature set
            feature_names: List of feature names being built
            source: Data source (e.g., "database", "cache", "api")
            **extra_attributes: Additional attributes to record

        Yields:
            Active span for adding additional attributes

        Example:
            with ml_span.feature_build("v2", feature_names=FEATURE_ORDER) as span:
                features = builder.build_observation(data)
                span.set_attribute("ml.features_built", len(features))
        """
        with self.tracer.start_as_current_span("ml.feature_build") as span:
            span.set_attribute("ml.operation", "feature_build")
            span.set_attribute("ml.feature_version", feature_version)
            span.set_attribute("ml.data_source", source)

            if feature_names:
                span.set_attribute("ml.feature_count", len(feature_names))
                # Only store first 10 feature names to avoid huge spans
                span.set_attribute("ml.feature_names", str(feature_names[:10]))

            for key, value in extra_attributes.items():
                span.set_attribute(f"ml.{key}", value)

            start_time = time.perf_counter()

            try:
                yield span
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("ml.build_latency_ms", elapsed_ms)

    @contextmanager
    def model_predict(
        self,
        model_id: str,
        prediction_type: str = "continuous",
        observation_dim: Optional[int] = None,
        **extra_attributes: Any,
    ):
        """
        Context manager for model prediction spans.

        This is a more detailed version of inference() specifically for
        the prediction step.

        Args:
            model_id: Model identifier
            prediction_type: Type of prediction ("continuous", "discrete", "classification")
            observation_dim: Dimension of input observation
            **extra_attributes: Additional attributes to record

        Yields:
            Active span for adding additional attributes

        Example:
            with ml_span.model_predict("ppo_primary", observation_dim=15) as span:
                action, _ = model.predict(observation)
                signal = get_signal(action)
                span.set_attribute("ml.action_raw", action)
                span.set_attribute("ml.signal", signal)
        """
        with self.tracer.start_as_current_span("ml.model_predict") as span:
            span.set_attribute("ml.model_id", model_id)
            span.set_attribute("ml.operation", "predict")
            span.set_attribute("ml.prediction_type", prediction_type)

            if observation_dim is not None:
                span.set_attribute("ml.observation_dim", observation_dim)

            for key, value in extra_attributes.items():
                span.set_attribute(f"ml.{key}", value)

            start_time = time.perf_counter()

            try:
                yield span
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("ml.predict_latency_ms", elapsed_ms)

    @contextmanager
    def batch_inference(
        self,
        model_id: str,
        batch_size: int,
        **extra_attributes: Any,
    ):
        """
        Context manager for batch inference spans.

        Args:
            model_id: Model identifier
            batch_size: Number of observations in batch
            **extra_attributes: Additional attributes to record

        Yields:
            Active span for adding additional attributes

        Example:
            with ml_span.batch_inference("ppo_primary", batch_size=100) as span:
                results = model.predict_batch(observations)
                span.set_attribute("ml.predictions_count", len(results))
        """
        with self.tracer.start_as_current_span("ml.batch_inference") as span:
            span.set_attribute("ml.model_id", model_id)
            span.set_attribute("ml.operation", "batch_inference")
            span.set_attribute("ml.batch_size", batch_size)

            for key, value in extra_attributes.items():
                span.set_attribute(f"ml.{key}", value)

            start_time = time.perf_counter()

            try:
                yield span
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("ml.total_latency_ms", elapsed_ms)
                span.set_attribute(
                    "ml.latency_per_sample_ms",
                    elapsed_ms / batch_size if batch_size > 0 else 0
                )

    @contextmanager
    def signal_generation(
        self,
        model_id: str,
        threshold_long: float = 0.33,
        threshold_short: float = -0.33,
        **extra_attributes: Any,
    ):
        """
        Context manager for signal generation spans.

        Specifically for the action -> signal conversion step.

        Args:
            model_id: Model identifier
            threshold_long: Threshold for LONG signal
            threshold_short: Threshold for SHORT signal
            **extra_attributes: Additional attributes

        Yields:
            Active span for adding additional attributes

        Example:
            with ml_span.signal_generation("ppo_primary") as span:
                signal = engine.get_signal(action)
                span.set_attribute("ml.raw_action", action)
                span.set_attribute("ml.signal", signal)
        """
        with self.tracer.start_as_current_span("ml.signal_generation") as span:
            span.set_attribute("ml.model_id", model_id)
            span.set_attribute("ml.operation", "signal_generation")
            span.set_attribute("ml.threshold_long", threshold_long)
            span.set_attribute("ml.threshold_short", threshold_short)

            for key, value in extra_attributes.items():
                span.set_attribute(f"ml.{key}", value)

            start_time = time.perf_counter()

            try:
                yield span
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("ml.generation_latency_ms", elapsed_ms)


def add_ml_attributes(
    model_id: Optional[str] = None,
    feature_version: Optional[str] = None,
    confidence: Optional[float] = None,
    signal: Optional[str] = None,
    action: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """
    Helper function to add ML-specific attributes to the current span.

    Args:
        model_id: Model identifier
        feature_version: Feature set version
        confidence: Prediction confidence score
        signal: Trading signal (LONG, SHORT, HOLD)
        action: Raw action value
        **kwargs: Additional ml.* attributes

    Example:
        add_ml_attributes(
            model_id="ppo_primary",
            confidence=0.85,
            signal="LONG",
            action=0.67
        )
    """
    from .otel_setup import get_current_span

    span = get_current_span()
    if not span:
        return

    if model_id is not None:
        span.set_attribute("ml.model_id", model_id)

    if feature_version is not None:
        span.set_attribute("ml.feature_version", feature_version)

    if confidence is not None:
        span.set_attribute("ml.confidence", confidence)

    if signal is not None:
        span.set_attribute("ml.signal", signal)

    if action is not None:
        span.set_attribute("ml.action", action)

    for key, value in kwargs.items():
        span.set_attribute(f"ml.{key}", value)


def add_trading_attributes(
    symbol: Optional[str] = None,
    position: Optional[str] = None,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    pnl: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """
    Helper function to add trading-specific attributes to the current span.

    Args:
        symbol: Trading symbol (e.g., "USDCOP")
        position: Position type (LONG, SHORT, FLAT)
        entry_price: Entry price for trade
        exit_price: Exit price for trade
        pnl: Profit/Loss for trade
        **kwargs: Additional trading.* attributes

    Example:
        add_trading_attributes(
            symbol="USDCOP",
            position="LONG",
            entry_price=4250.50,
            pnl=15.25
        )
    """
    from .otel_setup import get_current_span

    span = get_current_span()
    if not span:
        return

    if symbol is not None:
        span.set_attribute("trading.symbol", symbol)

    if position is not None:
        span.set_attribute("trading.position", position)

    if entry_price is not None:
        span.set_attribute("trading.entry_price", entry_price)

    if exit_price is not None:
        span.set_attribute("trading.exit_price", exit_price)

    if pnl is not None:
        span.set_attribute("trading.pnl", pnl)

    for key, value in kwargs.items():
        span.set_attribute(f"trading.{key}", value)
