# services/inference_api/middleware/rate_limiter.py
"""
Enhanced Rate Limiter with Per-User Limits and Database Backend.

This module extends the base rate limiting with:
- Per-user rate limits (from API key configuration)
- Database-backed state (for distributed deployments)
- Integration with authentication middleware

Contract: CTR-RATE-001

Usage:
    from middleware.rate_limiter import AuthenticatedRateLimiter

    limiter = AuthenticatedRateLimiter(db_pool)
    await limiter.check_rate_limit(user_id, request)

Author: Trading Team / Claude Code
Date: 2026-01-16
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# Async Token Bucket Strategy
# =============================================================================

@dataclass
class AsyncTokenBucket:
    """Async-safe token bucket rate limiter.

    Token bucket algorithm allows burst traffic while maintaining
    an average rate limit over time.

    Attributes:
        max_tokens: Maximum bucket capacity (burst size)
        refill_rate: Tokens added per second
    """
    max_tokens: float
    refill_rate: float  # tokens per second

    _tokens: Dict[str, float] = field(default_factory=dict)
    _last_update: Dict[str, float] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        self._tokens = defaultdict(lambda: self.max_tokens)
        self._last_update = defaultdict(time.time)
        self._lock = asyncio.Lock()

    async def consume(self, key: str, tokens: float = 1.0) -> Tuple[bool, float]:
        """Try to consume tokens from the bucket.

        Args:
            key: Identifier for the bucket (usually user_id or IP)
            tokens: Number of tokens to consume (default 1)

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        async with self._lock:
            now = time.time()

            # Refill tokens based on elapsed time
            elapsed = now - self._last_update[key]
            self._tokens[key] = min(
                self.max_tokens,
                self._tokens[key] + elapsed * self.refill_rate
            )
            self._last_update[key] = now

            # Try to consume
            if self._tokens[key] >= tokens:
                self._tokens[key] -= tokens
                return True, 0.0

            # Calculate wait time
            tokens_needed = tokens - self._tokens[key]
            retry_after = tokens_needed / self.refill_rate

            return False, retry_after

    async def get_remaining(self, key: str) -> float:
        """Get remaining tokens for a key."""
        async with self._lock:
            return self._tokens.get(key, self.max_tokens)


# =============================================================================
# Per-User Rate Limiter
# =============================================================================

class AuthenticatedRateLimiter:
    """Rate limiter with per-user limits from database.

    Supports:
    - Default rate limits for anonymous users
    - Per-user rate limits from API key configuration
    - IP-based rate limiting fallback
    - Database-backed state for distributed deployments

    Attributes:
        db_pool: AsyncPG connection pool for rate limit state
        default_limit: Default requests per minute for anonymous users
        default_burst: Default burst size
    """

    def __init__(
        self,
        db_pool=None,
        default_limit: int = 100,
        default_burst: int = 20,
        use_database: bool = False,
    ):
        """Initialize rate limiter.

        Args:
            db_pool: Optional database pool for distributed state
            default_limit: Default requests per minute
            default_burst: Default burst size (max concurrent requests)
            use_database: Whether to use database for state (vs in-memory)
        """
        self.db_pool = db_pool
        self.default_limit = int(os.environ.get(
            "RATE_LIMIT_REQUESTS_PER_MINUTE",
            str(default_limit)
        ))
        self.default_burst = int(os.environ.get(
            "RATE_LIMIT_BURST",
            str(default_burst)
        ))
        self.use_database = use_database and db_pool is not None

        # In-memory rate limiters per user
        self._limiters: Dict[str, AsyncTokenBucket] = {}
        self._limiter_lock = asyncio.Lock()

        logger.info(
            f"AuthenticatedRateLimiter initialized: "
            f"default_limit={self.default_limit}/min, "
            f"default_burst={self.default_burst}, "
            f"use_database={self.use_database}"
        )

    async def _get_limiter(self, key: str, rate_limit: int) -> AsyncTokenBucket:
        """Get or create a rate limiter for a key."""
        async with self._limiter_lock:
            if key not in self._limiters:
                # Create new limiter with specified rate
                self._limiters[key] = AsyncTokenBucket(
                    max_tokens=float(min(rate_limit, self.default_burst * 2)),
                    refill_rate=rate_limit / 60.0  # Convert to per-second
                )
            return self._limiters[key]

    async def check_rate_limit(
        self,
        user_id: str,
        request: Request,
        rate_limit: Optional[int] = None
    ) -> Tuple[bool, Dict]:
        """Check if request is within rate limit.

        Args:
            user_id: User identifier (from auth middleware)
            request: FastAPI request object
            rate_limit: Optional custom rate limit (from API key)

        Returns:
            Tuple of (allowed, headers_dict)
            headers_dict contains rate limit headers to add to response
        """
        # Determine rate limit
        if rate_limit is None:
            # Try to get from request state (set by auth middleware)
            rate_limit = getattr(request.state, 'rate_limit', None)

        if rate_limit is None:
            rate_limit = self.default_limit

        # Use user_id if authenticated, otherwise use IP
        if user_id in ("public", "anonymous"):
            key = self._get_client_ip(request)
        else:
            key = f"user:{user_id}"

        # Check rate limit
        if self.use_database:
            allowed, retry_after = await self._check_database(key, rate_limit)
        else:
            limiter = await self._get_limiter(key, rate_limit)
            allowed, retry_after = await limiter.consume(key)

        # Build headers
        headers = {
            "X-RateLimit-Limit": str(rate_limit),
            "X-RateLimit-Remaining": str(max(0, int(rate_limit - 1) if allowed else 0)),
            "X-RateLimit-Reset": str(int(time.time() + 60)),
        }

        if not allowed:
            headers["Retry-After"] = str(int(retry_after) + 1)

        return allowed, headers

    async def _check_database(
        self,
        key: str,
        rate_limit: int
    ) -> Tuple[bool, float]:
        """Check rate limit using database state.

        This method is useful for distributed deployments where
        multiple API instances need to share rate limit state.
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Try to get current state
                row = await conn.fetchrow(
                    """
                    SELECT window_start, request_count, daily_count, daily_reset
                    FROM api_rate_limit_state
                    WHERE key_hash = $1
                    """,
                    key
                )

                now = datetime.utcnow()
                window_start = now.replace(second=0, microsecond=0)
                daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

                if row is None:
                    # Create new entry
                    await conn.execute(
                        """
                        INSERT INTO api_rate_limit_state
                        (key_hash, window_start, request_count, daily_count, daily_reset)
                        VALUES ($1, $2, 1, 1, $3)
                        """,
                        key, window_start, daily_reset
                    )
                    return True, 0.0

                # Check if window has rolled over
                if row['window_start'] < window_start:
                    # Reset window
                    await conn.execute(
                        """
                        UPDATE api_rate_limit_state
                        SET window_start = $2, request_count = 1, updated_at = NOW()
                        WHERE key_hash = $1
                        """,
                        key, window_start
                    )
                    return True, 0.0

                # Check if under limit
                if row['request_count'] < rate_limit:
                    await conn.execute(
                        """
                        UPDATE api_rate_limit_state
                        SET request_count = request_count + 1,
                            daily_count = daily_count + 1,
                            updated_at = NOW()
                        WHERE key_hash = $1
                        """,
                        key
                    )
                    return True, 0.0

                # Rate limited - calculate retry time
                elapsed = (now - row['window_start']).total_seconds()
                retry_after = 60.0 - elapsed

                return False, max(0, retry_after)

        except Exception as e:
            logger.error(f"Database rate limit check failed: {e}")
            # Fall back to allowing request on database error
            return True, 0.0

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# =============================================================================
# Middleware Integration
# =============================================================================

class AuthenticatedRateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware that respects per-user limits.

    Integrates with AuthMiddleware to apply custom rate limits
    based on API key configuration.

    Usage:
        app.add_middleware(
            AuthenticatedRateLimitMiddleware,
            rate_limiter=AuthenticatedRateLimiter(pool),
            exclude_paths=["/health", "/docs"]
        )
    """

    def __init__(
        self,
        app,
        rate_limiter: Optional[AuthenticatedRateLimiter] = None,
        exclude_paths: Optional[list] = None,
        enabled: bool = True,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter or AuthenticatedRateLimiter()
        self.exclude_paths = exclude_paths or [
            "/",
            "/health",
            "/ready",
            "/v1/health",
            "/api/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics",
        ]
        self.enabled = enabled and os.environ.get(
            "ENABLE_RATE_LIMITING", "true"
        ).lower() == "true"

        logger.info(
            f"AuthenticatedRateLimitMiddleware: enabled={self.enabled}"
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with authenticated rate limiting."""
        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip excluded paths
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Get user_id from auth middleware (set in request state)
        user_id = getattr(request.state, 'user_id', 'anonymous')

        # Check rate limit
        allowed, headers = await self.rate_limiter.check_rate_limit(
            user_id,
            request
        )

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {user_id} "
                f"({self.rate_limiter._get_client_ip(request)}) "
                f"on {request.method} {path}"
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after_seconds": int(headers.get("Retry-After", 60)),
                },
                headers=headers
            )

        # Process request and add rate limit headers to response
        response = await call_next(request)

        # Add rate limit headers
        for header, value in headers.items():
            response.headers[header] = value

        return response


# =============================================================================
# Factory Function
# =============================================================================

def create_authenticated_rate_limiter(
    db_pool=None,
    default_limit: int = 100,
    default_burst: int = 20,
    use_database: bool = False,
) -> AuthenticatedRateLimiter:
    """Factory function to create a rate limiter.

    Args:
        db_pool: Optional database pool for distributed state
        default_limit: Default requests per minute
        default_burst: Default burst size
        use_database: Whether to use database for state

    Returns:
        Configured AuthenticatedRateLimiter instance
    """
    return AuthenticatedRateLimiter(
        db_pool=db_pool,
        default_limit=default_limit,
        default_burst=default_burst,
        use_database=use_database,
    )
