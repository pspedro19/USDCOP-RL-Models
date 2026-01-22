"""
Rate limiting middleware using Redis.
"""

import time
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis

from app.core.config import settings
from app.core.exceptions import ErrorCode


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware.
    Uses Redis for distributed rate limiting.
    """

    def __init__(
        self,
        app,
        redis_url: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
    ):
        super().__init__(app)
        self.redis_url = redis_url or settings.redis_url
        self.requests_per_minute = requests_per_minute or settings.rate_limit_per_minute
        self._redis: Optional[redis.Redis] = None

    async def get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        if request.client:
            return request.client.host

        return "unknown"

    def get_rate_limit_key(self, request: Request) -> str:
        """Generate rate limit key for the request."""
        client_ip = self.get_client_ip(request)

        # If authenticated, use user ID instead of IP
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"rate_limit:user:{user_id}"

        return f"rate_limit:ip:{client_ip}"

    async def is_rate_limited(self, key: str) -> tuple[bool, int, int]:
        """
        Check if request is rate limited using sliding window.

        Returns:
            Tuple of (is_limited, current_count, retry_after)
        """
        try:
            redis_client = await self.get_redis()
            current_time = int(time.time())
            window_start = current_time - 60  # 1 minute window

            # Use sorted set for sliding window
            pipe = redis_client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Add current request
            pipe.zadd(key, {str(current_time): current_time})

            # Count requests in window
            pipe.zcard(key)

            # Set expiry on key
            pipe.expire(key, 120)  # 2 minutes to be safe

            results = await pipe.execute()
            current_count = results[2]

            if current_count > self.requests_per_minute:
                # Calculate retry after
                oldest = await redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = int(oldest[0][1])
                    retry_after = max(1, 60 - (current_time - oldest_time))
                else:
                    retry_after = 60

                return True, current_count, retry_after

            return False, current_count, 0

        except redis.RedisError:
            # If Redis is down, allow the request but log the error
            return False, 0, 0

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/health", "/"]:
            return await call_next(request)

        key = self.get_rate_limit_key(request)
        is_limited, count, retry_after = await self.is_rate_limited(key)

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "code": ErrorCode.RATE_LIMITED.value,
                    "message": "Rate limit exceeded",
                    "details": {
                        "retry_after": retry_after,
                        "limit": self.requests_per_minute,
                    },
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - count)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
