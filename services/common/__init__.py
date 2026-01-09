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

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
Updated: 2025-12-26 - Added Redis Streams manager
"""

from .database import (
    get_db_config,
    get_db_connection,
    get_connection_pool,
    execute_query,
    execute_query_df,
    DatabaseConfig,
)

from .config import (
    get_service_config,
    get_trading_hours,
    ServiceConfig,
    TradingHoursConfig,
)

from .validation import (
    validate_symbol,
    validate_timeframe,
    validate_limit,
    validate_date_range,
    validate_bar_number,
    validate_observation,
    validate_position,
    sanitize_identifier,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    MAX_LIMIT,
    BARS_PER_SESSION,
    InferenceRequest,
    BarNumberParam,
)

from .trading_calendar import (
    TradingCalendar,
    get_calendar,
    is_trading_day,
    filter_trading_days,
    validate_no_holidays,
    validate_and_filter,
    validate_dag_execution_date,
    should_skip_dag_task,
    validate_training_data,
    validate_inference_time,
    trading_calendar,
)

from .redis_streams_manager import (
    # Configuration
    StreamConfig,
    RedisStreamsConfig,
    # Manager
    RedisStreamsManager,
    get_redis_streams_manager,
    # Producer
    StreamProducer,
    get_producer,
    # Consumer
    StreamConsumer,
    get_consumer,
    # SSE
    SSEAdapter,
    get_sse_adapter,
    # Utilities
    StreamUtilities,
    get_utilities,
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
]
