# =============================================================================
# Centralized Logging Module
# =============================================================================
# Provides structured JSON logging for the USDCOP Trading Platform.
#
# Features:
#   - Structured JSON logging for Loki ingestion
#   - Automatic context propagation (request_id, service, etc.)
#   - Performance-optimized with caching
#   - Thread-safe logger factory
#
# Usage:
#   from src.core.logging import LoggerFactory, get_logger
#
#   # Configure once at application startup
#   LoggerFactory.configure(level="INFO", json_format=True)
#
#   # Get logger anywhere in the code
#   logger = get_logger(__name__)
#   logger.info("Processing request", request_id="abc123", duration_ms=45)
# =============================================================================

from src.core.logging.logger_factory import LoggerFactory, get_logger

__all__ = ["LoggerFactory", "get_logger"]
