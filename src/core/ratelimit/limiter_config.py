"""
Rate Limit Configuration
=======================
Configuration management for rate limiting across services.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass
import time

@dataclass
class RateLimitConfig:
    """Configuration for a specific rate limit."""
    rate: float          # Tokens per second
    burst: int           # Maximum burst capacity
    reset_interval: int  # Reset interval in seconds (0 = no reset)
    scope: str           # Scope: 'global', 'per_user', 'per_account'
    description: str     # Human-readable description

def _get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable."""
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

def _get_env_int(key: str, default: int) -> int:
    """Get int value from environment variable."""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

def get_rate_limits() -> Dict[str, RateLimitConfig]:
    """Get rate limit configurations from environment variables."""
    
    return {
        # MT5 API limits
        "mt5_api": RateLimitConfig(
            rate=_get_env_float("RL_MT5_RPS", 10.0),
            burst=_get_env_int("RL_MT5_BURST", 15),
            reset_interval=0,  # No daily reset
            scope="global",
            description="MT5 API requests per second"
        ),
        
        # Trade execution limits
        "trade_execution": RateLimitConfig(
            rate=_get_env_float("RL_EXEC_TRADES_PER_MIN", 5.0) / 60.0,  # Convert to per-second
            burst=_get_env_int("RL_EXEC_BURST", 5),
            reset_interval=0,
            scope="per_account",
            description="Trade executions per minute per account"
        ),
        
        # Data fetching limits
        "data_fetch": RateLimitConfig(
            rate=_get_env_float("RL_DATA_FETCH_PER_MIN", 100.0) / 60.0,  # Convert to per-second
            burst=_get_env_int("RL_DATA_BURST", 200),
            reset_interval=0,
            scope="global",
            description="Data fetch requests per minute"
        ),
        
        # Dashboard WebSocket updates
        "dashboard_ws": RateLimitConfig(
            rate=_get_env_float("RL_DASH_WS_RPS", 1.0),
            burst=_get_env_int("RL_DASH_WS_BURST", 5),
            reset_interval=0,
            scope="per_client",
            description="Dashboard WebSocket updates per second"
        ),
        
        # Health API limits
        "health_overview": RateLimitConfig(
            rate=_get_env_float("RL_HEALTH_OVERVIEW_RPS", 2.0),
            burst=_get_env_int("RL_HEALTH_OVERVIEW_BURST", 10),
            reset_interval=0,
            scope="per_client",
            description="Health overview API calls per second"
        ),
        
        "health_raw": RateLimitConfig(
            rate=_get_env_float("RL_HEALTH_RAW_RPS", 1.0),
            burst=_get_env_int("RL_HEALTH_RAW_BURST", 5),
            reset_interval=0,
            scope="per_client",
            description="Health raw data API calls per second"
        ),
        
        "health_metrics": RateLimitConfig(
            rate=_get_env_float("RL_HEALTH_METRICS_RPS", 1.0),
            burst=_get_env_int("RL_HEALTH_METRICS_BURST", 5),
            reset_interval=0,
            scope="per_client",
            description="Health metrics API calls per second"
        ),
        
        # Pipeline processing limits
        "pipeline_batch": RateLimitConfig(
            rate=_get_env_float("RL_PIPELINE_BATCH_PER_MIN", 20.0) / 60.0,
            burst=_get_env_int("RL_PIPELINE_BURST", 30),
            reset_interval=0,
            scope="global",
            description="Pipeline batch processing per minute"
        ),
        
        # Model inference limits
        "model_inference": RateLimitConfig(
            rate=_get_env_float("RL_MODEL_INFERENCE_RPS", 5.0),
            burst=_get_env_int("RL_MODEL_INFERENCE_BURST", 10),
            reset_interval=0,
            scope="global",
            description="Model inference requests per second"
        ),
        
        # Event publishing limits
        "event_publish": RateLimitConfig(
            rate=_get_env_float("RL_EVENT_PUBLISH_RPS", 50.0),
            burst=_get_env_int("RL_EVENT_PUBLISH_BURST", 100),
            reset_interval=0,
            scope="global",
            description="Event publishing per second"
        )
    }

def get_limit_for_service(service: str) -> RateLimitConfig:
    """Get rate limit configuration for a specific service."""
    limits = get_rate_limits()
    return limits.get(service, RateLimitConfig(
        rate=1.0,
        burst=10,
        reset_interval=0,
        scope="global",
        description="Default rate limit"
    ))

def get_redis_url() -> str:
    """Get Redis URL for rate limiting."""
    return os.getenv("RATE_LIMIT_REDIS_URL", "redis://redis:6379/3")

def get_rate_limit_headers(limit: float, remaining: float, retry_after: float = 0.0) -> Dict[str, str]:
    """Generate standard rate limit headers."""
    headers = {
        "X-RateLimit-Limit": str(int(limit)),
        "X-RateLimit-Remaining": str(max(0, int(remaining))),
    }
    
    if retry_after > 0:
        headers["Retry-After"] = f"{retry_after:.2f}"
        headers["X-RateLimit-Reset"] = str(int(time.time() + retry_after))
    
    return headers
