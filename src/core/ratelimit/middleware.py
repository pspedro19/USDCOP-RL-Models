"""
Rate Limiting Middleware
========================
FastAPI middleware and decorators for rate limiting.
"""
import asyncio
import functools
import logging
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse

from .redis_limiter import RedisTokenBucket, RateLimitExceeded
from .limiter_config import get_limit_for_service, get_rate_limit_headers

logger = logging.getLogger(__name__)

def rate_limit_decorator(service: str, endpoint: str = None, cost: int = 1):
    """
    Decorator for rate limiting functions.
    
    Args:
        service: Rate limit service name (e.g., 'mt5_api')
        endpoint: Specific endpoint identifier
        cost: Token cost for this operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get rate limit configuration
            config = get_limit_for_service(service)
            
            # Create endpoint identifier
            endpoint_id = endpoint or func.__name__
            
            # Get rate limiter
            limiter = await get_rate_limiter()
            
            # Try to consume tokens
            allowed, remaining, retry_after = await limiter.try_consume(
                scope=service,
                endpoint=endpoint_id,
                cost=cost,
                rate=config.rate,
                burst=config.burst,
                reset_interval=config.reset_interval
            )
            
            if not allowed:
                raise RateLimitExceeded(service, retry_after)
            
            # Call the original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def rate_limit_middleware(service: str, endpoint: str = None, cost: int = 1):
    """
    FastAPI dependency for rate limiting endpoints.
    
    Args:
        service: Rate limit service name
        endpoint: Specific endpoint identifier
        cost: Token cost for this operation
    """
    async def dependency(request: Request, response: Response):
        # Get rate limit configuration
        config = get_limit_for_service(service)
        
        # Create endpoint identifier
        endpoint_id = endpoint or request.url.path
        
        # Get client identifier (IP address or user ID)
        client_id = _get_client_id(request)
        
        # Get rate limiter
        limiter = await get_rate_limiter()
        
        # Try to consume tokens
        allowed, remaining, retry_after = await limiter.try_consume(
            scope=service,
            endpoint=f"{endpoint_id}:{client_id}",
            cost=cost,
            rate=config.rate,
            burst=config.burst,
            reset_interval=config.reset_interval
        )
        
        if not allowed:
            # Add rate limit headers
            headers = get_rate_limit_headers(config.rate, remaining, retry_after)
            for key, value in headers.items():
                response.headers[key] = value
            
            # Return 429 Too Many Requests
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "service": service,
                    "retry_after": retry_after,
                    "limit": config.rate,
                    "remaining": remaining
                }
            )
        
        # Add rate limit headers to successful response
        headers = get_rate_limit_headers(config.rate, remaining, retry_after)
        for key, value in headers.items():
            response.headers[key] = value
    
    return dependency

def _get_client_id(request: Request) -> str:
    """Extract client identifier from request."""
    # Try to get user ID from headers or query params
    user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
    if user_id:
        return str(user_id)
    
    # Fall back to IP address
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    return client_ip

async def get_rate_limiter() -> RedisTokenBucket:
    """Get the global rate limiter instance."""
    from .redis_limiter import get_rate_limiter as _get_limiter
    return await _get_limiter()

# Convenience functions for common rate limiting scenarios
def mt5_api_rate_limit(cost: int = 1):
    """Rate limit for MT5 API calls."""
    return rate_limit_middleware("mt5_api", cost=cost)

def trade_execution_rate_limit(cost: int = 1):
    """Rate limit for trade execution."""
    return rate_limit_middleware("trade_execution", cost=cost)

def data_fetch_rate_limit(cost: int = 1):
    """Rate limit for data fetching."""
    return rate_limit_middleware("data_fetch", cost=cost)

def dashboard_ws_rate_limit(cost: int = 1):
    """Rate limit for dashboard WebSocket updates."""
    return rate_limit_middleware("dashboard_ws", cost=cost)

def health_api_rate_limit(service: str, cost: int = 1):
    """Rate limit for health API endpoints."""
    return rate_limit_middleware(service, cost=cost)

# Decorator versions for non-FastAPI functions
def mt5_api_rate_limit_decorator(cost: int = 1):
    """Decorator for MT5 API rate limiting."""
    return rate_limit_decorator("mt5_api", cost=cost)

def trade_execution_rate_limit_decorator(cost: int = 1):
    """Decorator for trade execution rate limiting."""
    return rate_limit_decorator("trade_execution", cost=cost)

def data_fetch_rate_limit_decorator(cost: int = 1):
    """Decorator for data fetch rate limiting."""
    return rate_limit_decorator("data_fetch", cost=cost)
