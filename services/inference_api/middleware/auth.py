# services/inference_api/middleware/auth.py
"""
API Authentication Middleware for USDCOP Inference Service.

This module provides authentication via API keys or JWT tokens.

Security Features:
    - API Key authentication (X-API-Key header)
    - JWT Bearer token authentication
    - Path exclusions for public endpoints
    - Secure key hashing (SHA-256)

Contract: CTR-AUTH-001

Usage:
    auth_middleware = AuthMiddleware(db_pool, jwt_secret)
    user_id = await auth_middleware.verify_request(request)

Author: Trading Team / Claude Code
Date: 2026-01-16
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# API Key header definition
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# JWT import with fallback
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed. JWT authentication disabled.")


class AuthMiddleware:
    """Authentication middleware for API protection.

    Supports two authentication methods:
    1. API Key (X-API-Key header) - validated against database
    2. JWT Bearer token (Authorization header) - validated with secret

    Certain paths are excluded from authentication (health, docs, etc.)

    Attributes:
        db_pool: AsyncPG connection pool for API key validation
        jwt_secret: Secret key for JWT token validation

    Example:
        auth = AuthMiddleware(pool, "my-secret-key")
        user_id = await auth.verify_request(request)
    """

    EXCLUDED_PATHS = {
        "/",
        "/health",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/openapi-export",
        "/metrics",
        "/api/v1/health",
        "/v1/health",
    }

    # Prefixes that should also be excluded (for pattern matching)
    EXCLUDED_PREFIXES = (
        "/docs",
        "/redoc",
    )

    def __init__(self, db_pool, jwt_secret: str, enabled: bool = True):
        """Initialize authentication middleware.

        Args:
            db_pool: AsyncPG connection pool
            jwt_secret: Secret key for JWT validation
            enabled: Whether authentication is enabled (default True)
        """
        self.db_pool = db_pool
        self.jwt_secret = jwt_secret
        self.enabled = enabled and os.environ.get(
            "ENABLE_AUTH", "true"
        ).lower() == "true"

        logger.info(f"AuthMiddleware initialized: enabled={self.enabled}")

    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from authentication."""
        # Exact match
        if path in self.EXCLUDED_PATHS:
            return True

        # Prefix match
        if path.startswith(self.EXCLUDED_PREFIXES):
            return True

        return False

    async def verify_request(self, request: Request) -> Optional[str]:
        """Verify request authentication.

        Checks API key or JWT token and returns user identifier.
        Public endpoints return "public" as user ID.

        Args:
            request: FastAPI Request object

        Returns:
            User ID string if authenticated, "public" for excluded paths

        Raises:
            HTTPException: 401 if authentication fails
        """
        # Skip auth if disabled
        if not self.enabled:
            return "anonymous"

        # Check if path is excluded
        if self._is_excluded_path(request.url.path):
            return "public"

        # Try API key first
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")

        if api_key:
            return await self._verify_api_key(api_key, request)
        elif auth_header and auth_header.startswith("Bearer "):
            return self._verify_jwt(auth_header[7:])

        # No authentication provided
        logger.warning(
            f"Missing authentication for {request.method} {request.url.path} "
            f"from {self._get_client_ip(request)}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication. Provide X-API-Key header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer, ApiKey"}
        )

    async def _verify_api_key(self, api_key: str, request: Request) -> str:
        """Verify API key against database.

        Args:
            api_key: Raw API key from header
            request: Request object for logging

        Returns:
            User ID associated with the API key

        Raises:
            HTTPException: 401 if key is invalid or inactive
        """
        # Hash the key for comparison
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Check if db_pool is available
        if self.db_pool is None:
            logger.warning("Database pool not available, using environment key check")
            return self._verify_env_api_key(api_key)

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT user_id, is_active, roles, rate_limit_per_minute
                    FROM api_keys
                    WHERE key_hash = $1
                    """,
                    key_hash
                )

            if not row:
                logger.warning(
                    f"Invalid API key attempt from {self._get_client_ip(request)} "
                    f"for {request.url.path}"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )

            if not row["is_active"]:
                logger.warning(
                    f"Inactive API key used by user {row['user_id']} "
                    f"from {self._get_client_ip(request)}"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key is deactivated"
                )

            # Update last_used_at timestamp (fire and forget)
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE api_keys SET last_used_at = NOW() WHERE key_hash = $1",
                        key_hash
                    )
            except Exception as e:
                logger.debug(f"Failed to update last_used_at: {e}")

            # Store rate limit in request state for rate limiter middleware
            request.state.rate_limit = row["rate_limit_per_minute"]
            request.state.user_roles = row["roles"]

            logger.debug(f"API key authenticated: user={row['user_id']}")
            return row["user_id"]

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database error during API key verification: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service temporarily unavailable"
            )

    def _verify_env_api_key(self, api_key: str) -> str:
        """Fallback API key verification using environment variable.

        Used when database is not available.
        """
        env_key = os.environ.get("API_KEY")
        if env_key and secrets.compare_digest(api_key, env_key):
            return "env_user"

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    def _verify_jwt(self, token: str) -> str:
        """Verify JWT bearer token.

        Args:
            token: JWT token string (without 'Bearer ' prefix)

        Returns:
            Subject (user ID) from token payload

        Raises:
            HTTPException: 401 if token is invalid or expired
        """
        if not JWT_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="JWT authentication not available"
            )

        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"],
                options={"require": ["exp", "sub"]}
            )

            # Check expiration (jwt.decode should handle this, but double-check)
            exp = datetime.fromtimestamp(payload["exp"])
            if datetime.utcnow() > exp:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )

            logger.debug(f"JWT authenticated: user={payload['sub']}")
            return payload["sub"]

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


def generate_api_key(prefix: str = "usdcop") -> Tuple[str, str]:
    """Generate a new API key and its hash.

    Creates a cryptographically secure API key with the format:
    {prefix}_{random_token}

    Args:
        prefix: Prefix for the key (default "usdcop")

    Returns:
        Tuple of (raw_key, key_hash)
        - raw_key: The API key to give to the user (store securely!)
        - key_hash: SHA-256 hash to store in database

    Example:
        key, hash = generate_api_key()
        # key = "usdcop_AbC123..." (give to user)
        # hash = "a1b2c3..." (store in database)
    """
    # Generate 32 bytes of random data, URL-safe base64 encoded
    random_part = secrets.token_urlsafe(32)
    key = f"{prefix}_{random_part}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    return key, key_hash


def create_jwt_token(
    user_id: str,
    secret: str,
    expires_in: timedelta = timedelta(hours=24),
    additional_claims: Optional[dict] = None
) -> str:
    """Create a JWT token for a user.

    Args:
        user_id: User identifier to include as 'sub' claim
        secret: Secret key for signing
        expires_in: Token validity duration (default 24 hours)
        additional_claims: Optional additional claims to include

    Returns:
        Encoded JWT token string

    Raises:
        RuntimeError: If PyJWT is not installed
    """
    if not JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed. Install with: pip install PyJWT")

    now = datetime.utcnow()
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + expires_in,
    }

    if additional_claims:
        payload.update(additional_claims)

    return jwt.encode(payload, secret, algorithm="HS256")


def get_key_prefix(api_key: str, length: int = 12) -> str:
    """Get a safe prefix of an API key for display/logging.

    Args:
        api_key: Full API key
        length: Number of characters to show (default 12)

    Returns:
        Key prefix with ellipsis, e.g., "usdcop_AbC1..."
    """
    if len(api_key) <= length:
        return api_key
    return api_key[:length] + "..."
