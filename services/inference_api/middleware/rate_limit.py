"""
Rate Limiting Middleware - Strategy Pattern Implementation.

This module implements rate limiting using the Strategy Pattern,
allowing different rate limiting strategies to be easily swapped.

Patterns Used:
- Strategy Pattern: Different rate limiting algorithms
- Factory Pattern: Create appropriate rate limiter
- Singleton Pattern: Shared state across requests

SOLID Principles:
- SRP: Each class has one responsibility
- OCP: New strategies can be added without modifying existing code
- LSP: All strategies are interchangeable
- ISP: Minimal interface for strategies
- DIP: Depends on abstractions (Protocol)
"""

import time
import os
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol
from threading import Lock

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Protocol (Interface Segregation)
# =============================================================================

class RateLimitStrategy(Protocol):
    """Protocol for rate limiting strategies.

    Follows Interface Segregation Principle - minimal interface.
    """

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        ...

    def get_retry_after(self, key: str) -> int:
        """Get seconds until rate limit resets."""
        ...


# =============================================================================
# Rate Limit Strategies (Open/Closed Principle)
# =============================================================================

@dataclass
class TokenBucketStrategy:
    """Token Bucket rate limiting strategy.

    Allows burst traffic up to bucket capacity, then refills at steady rate.
    Good for APIs that need to handle occasional bursts.
    """

    requests_per_minute: int = 100
    burst_size: int = 20

    _buckets: Dict[str, float] = field(default_factory=dict)
    _last_update: Dict[str, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        self._buckets = defaultdict(lambda: float(self.burst_size))
        self._last_update = defaultdict(time.time)
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed using token bucket algorithm."""
        with self._lock:
            now = time.time()

            # Refill tokens based on time elapsed
            time_passed = now - self._last_update[key]
            refill_rate = self.requests_per_minute / 60.0
            tokens_to_add = time_passed * refill_rate

            self._buckets[key] = min(
                self.burst_size,
                self._buckets[key] + tokens_to_add
            )
            self._last_update[key] = now

            # Check if we have tokens available
            if self._buckets[key] >= 1:
                self._buckets[key] -= 1
                return True

            return False

    def get_retry_after(self, key: str) -> int:
        """Calculate seconds until a token is available."""
        with self._lock:
            if self._buckets[key] >= 1:
                return 0

            tokens_needed = 1 - self._buckets[key]
            refill_rate = self.requests_per_minute / 60.0

            return int(tokens_needed / refill_rate) + 1


@dataclass
class SlidingWindowStrategy:
    """Sliding Window rate limiting strategy.

    More accurate than fixed windows, prevents edge case bursts.
    Good for strict rate limiting requirements.
    """

    requests_per_minute: int = 100

    _requests: Dict[str, list] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        self._requests = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed using sliding window."""
        with self._lock:
            now = time.time()
            window_start = now - 60  # 1 minute window

            # Clean old requests
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]

            # Check if under limit
            if len(self._requests[key]) < self.requests_per_minute:
                self._requests[key].append(now)
                return True

            return False

    def get_retry_after(self, key: str) -> int:
        """Calculate seconds until oldest request expires."""
        with self._lock:
            if not self._requests[key]:
                return 0

            oldest = min(self._requests[key])
            retry_after = int(oldest + 60 - time.time())

            return max(0, retry_after) + 1


@dataclass
class FixedWindowStrategy:
    """Fixed Window rate limiting strategy.

    Simple and memory efficient. May allow bursts at window boundaries.
    Good for high-volume, less strict requirements.
    """

    requests_per_minute: int = 100

    _counts: Dict[str, int] = field(default_factory=dict)
    _window_start: Dict[str, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        self._counts = defaultdict(int)
        self._window_start = defaultdict(time.time)
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed using fixed window."""
        with self._lock:
            now = time.time()

            # Reset window if expired
            if now - self._window_start[key] >= 60:
                self._counts[key] = 0
                self._window_start[key] = now

            # Check if under limit
            if self._counts[key] < self.requests_per_minute:
                self._counts[key] += 1
                return True

            return False

    def get_retry_after(self, key: str) -> int:
        """Calculate seconds until window resets."""
        with self._lock:
            elapsed = time.time() - self._window_start[key]
            return max(0, int(60 - elapsed)) + 1


# =============================================================================
# Factory (Dependency Inversion)
# =============================================================================

def create_rate_limiter(
    strategy: str = "token_bucket",
    requests_per_minute: int = 100,
    burst_size: int = 20,
) -> RateLimitStrategy:
    """Factory function to create rate limiter with specified strategy.

    Args:
        strategy: One of 'token_bucket', 'sliding_window', 'fixed_window'
        requests_per_minute: Maximum requests per minute
        burst_size: Maximum burst size (token_bucket only)

    Returns:
        RateLimitStrategy implementation

    Example:
        limiter = create_rate_limiter(
            strategy="token_bucket",
            requests_per_minute=100,
            burst_size=20
        )
    """
    strategies = {
        "token_bucket": lambda: TokenBucketStrategy(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        ),
        "sliding_window": lambda: SlidingWindowStrategy(
            requests_per_minute=requests_per_minute,
        ),
        "fixed_window": lambda: FixedWindowStrategy(
            requests_per_minute=requests_per_minute,
        ),
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy]()


# =============================================================================
# Middleware (Single Responsibility)
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using configurable strategies.

    Follows Single Responsibility Principle - only handles rate limiting.
    Strategy is injected via constructor (Dependency Inversion).

    Example:
        app.add_middleware(
            RateLimitMiddleware,
            strategy=create_rate_limiter("token_bucket", 100, 20),
            exclude_paths=["/health", "/docs"]
        )
    """

    def __init__(
        self,
        app,
        strategy: Optional[RateLimitStrategy] = None,
        exclude_paths: Optional[list] = None,
        enabled: bool = True,
    ):
        super().__init__(app)

        # Load from environment or use defaults
        requests_per_minute = int(
            os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "100")
        )
        burst_size = int(os.environ.get("RATE_LIMIT_BURST", "20"))

        self.strategy = strategy or create_rate_limiter(
            strategy="token_bucket",
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        )

        self.exclude_paths = exclude_paths or [
            "/health",
            "/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
        ]

        self.enabled = enabled and os.environ.get(
            "ENABLE_RATE_LIMITING", "true"
        ).lower() == "true"

        logger.info(
            f"RateLimitMiddleware initialized: enabled={self.enabled}, "
            f"strategy={type(self.strategy).__name__}"
        )

    def _get_client_key(self, request: Request) -> str:
        """Extract client identifier for rate limiting.

        Uses X-Forwarded-For if behind proxy, otherwise client host.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip if disabled or excluded path
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Check rate limit
        client_key = self._get_client_key(request)

        if not self.strategy.is_allowed(client_key):
            retry_after = self.strategy.get_retry_after(client_key)

            logger.warning(
                f"Rate limit exceeded for {client_key} on {path}"
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(
                        getattr(self.strategy, "requests_per_minute", 100)
                    ),
                }
            )

        return await call_next(request)
