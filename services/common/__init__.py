"""
Services Common Module
======================
Shared utilities for all FastAPI services.

DRY Refactoring: Eliminates duplicated code across 6+ service files.

Modules:
    - database: Database connection pooling and utilities
    - config: Shared configuration management
    - metrics: Financial metrics calculations (Sharpe, Sortino, etc.)
    - validation: Input validation utilities (SOLID compliance)
    - trading_calendar: Colombian holiday and weekend validation
    - redis_streams_manager: Redis Streams for multi-model signal streaming
    - tracing: Distributed tracing with OpenTelemetry/Jaeger
    - prometheus_metrics: Prometheus metrics for observability

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
Updated: 2025-12-26 - Added Redis Streams manager
Updated: 2025-01-14 - Added tracing and prometheus metrics (FASE 8)
"""

from .config import (
    ServiceConfig,
    TradingHoursConfig,
    get_service_config,
    get_trading_hours,
)
from .database import (
    DatabaseConfig,
    execute_query,
    execute_query_df,
    get_connection_pool,
    get_db_config,
    get_db_connection,
)
from .prometheus_metrics import (
    # Gauges
    current_position_gauge,
    db_query_seconds,
    feature_calculation_seconds,
    feature_drift_gauge,
    get_metrics_app,
    # Histograms
    inference_latency_seconds,
    # Counters
    inference_requests_total,
    model_confidence_gauge,
    model_load_total,
    # Setup
    setup_prometheus_metrics,
    trade_signals_total,
)
from .redis_streams_manager import (
    RedisStreamsConfig,
    # Manager
    RedisStreamsManager,
    # SSE
    SSEAdapter,
    # Configuration
    StreamConfig,
    # Consumer
    StreamConsumer,
    # Producer
    StreamProducer,
    # Utilities
    StreamUtilities,
    get_consumer,
    get_producer,
    get_redis_streams_manager,
    get_sse_adapter,
    get_utilities,
)
from .tracing import (
    NoOpSpan,
    NoOpTracer,
    get_tracer,
    setup_tracing,
    trace_async_function,
    trace_function,
)
from .trading_calendar import (
    TradingCalendar,
    filter_trading_days,
    get_calendar,
    is_trading_day,
    should_skip_dag_task,
    trading_calendar,
    validate_and_filter,
    validate_dag_execution_date,
    validate_inference_time,
    validate_no_holidays,
    validate_training_data,
)
from .validation import (
    BARS_PER_SESSION,
    MAX_LIMIT,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    BarNumberParam,
    InferenceRequest,
    sanitize_identifier,
    validate_bar_number,
    validate_date_range,
    validate_limit,
    validate_observation,
    validate_position,
    validate_symbol,
    validate_timeframe,
)

__all__ = [
    # Database
    'get_db_config',
    'get_db_connection',
    'get_connection_pool',
    'execute_query',
    'execute_query_df',
    'DatabaseConfig',
    # Config
    'get_service_config',
    'get_trading_hours',
    'ServiceConfig',
    'TradingHoursConfig',
    # Validation
    'validate_symbol',
    'validate_timeframe',
    'validate_limit',
    'validate_date_range',
    'validate_bar_number',
    'validate_observation',
    'validate_position',
    'sanitize_identifier',
    'SUPPORTED_SYMBOLS',
    'SUPPORTED_TIMEFRAMES',
    'MAX_LIMIT',
    'BARS_PER_SESSION',
    'InferenceRequest',
    'BarNumberParam',
    # Trading Calendar
    'TradingCalendar',
    'get_calendar',
    'is_trading_day',
    'filter_trading_days',
    'validate_no_holidays',
    'validate_and_filter',
    'validate_dag_execution_date',
    'should_skip_dag_task',
    'validate_training_data',
    'validate_inference_time',
    'trading_calendar',
    # Redis Streams
    'StreamConfig',
    'RedisStreamsConfig',
    'RedisStreamsManager',
    'get_redis_streams_manager',
    'StreamProducer',
    'get_producer',
    'StreamConsumer',
    'get_consumer',
    'SSEAdapter',
    'get_sse_adapter',
    'StreamUtilities',
    'get_utilities',
    # Tracing (OpenTelemetry)
    'setup_tracing',
    'get_tracer',
    'trace_function',
    'trace_async_function',
    'NoOpTracer',
    'NoOpSpan',
    # Prometheus Metrics
    'inference_requests_total',
    'trade_signals_total',
    'model_load_total',
    'inference_latency_seconds',
    'feature_calculation_seconds',
    'db_query_seconds',
    'current_position_gauge',
    'model_confidence_gauge',
    'feature_drift_gauge',
    'setup_prometheus_metrics',
    'get_metrics_app',
]
