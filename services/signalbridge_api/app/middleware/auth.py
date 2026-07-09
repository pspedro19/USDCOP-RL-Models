"""
Authentication middleware and dependencies.
"""

import os
from datetime import datetime
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import ErrorCode
from app.core.login_security import TokenBlacklist
from app.core.security import verify_token
from app.models import User
from app.services.user import UserService

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)

# Dev mode flag - skip authentication when enabled.
# Hard guard: NEVER active in production, even if the env var is set. This
# prevents an accidental/leftover SIGNALBRIDGE_DEV_MODE=true from disabling auth
# on a live deployment (and makes load/security tests meaningful).
DEV_MODE = (
    os.getenv("SIGNALBRIDGE_DEV_MODE", "false").lower() == "true"
    and not settings.is_production
)

# Dev user id: fixed UUID so dev/prod identity types match (sb_users PKs are UUID —
# audit A8-11; the old int 1 diverged and broke FK-typed queries in dev).
DEV_USER_ID = "00000000-0000-4000-8000-0000000000de"  # fixed, valid UUID4-shaped


class DevUser:
    """Dummy user object for dev mode."""
    def __init__(self):
        self.id = DEV_USER_ID  # fixed UUID (A8-11)
        self.email = "admin@trading.usdcop.com"
        self.name = "Dev User"
        self.is_active = True
        self.is_verified = True
        # Dev bypass is a full admin so every gated route is reachable locally.
        self.status = "approved"
        self.role = "admin"
        self.must_reset_password = False
        self.last_login = datetime.utcnow()
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: Bearer token credentials
        db: Database session

    Returns:
        Authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    # In dev mode, return a dummy user without requiring authentication
    if DEV_MODE:
        return DevUser()

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "Missing authentication token",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = verify_token(token, token_type="access")

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "Invalid or expired token",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Reject revoked (logged-out) tokens.
    if await TokenBlacklist.is_blacklisted(payload.get("jti", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "Token has been revoked",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "Invalid token payload",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_service = UserService(db)
    user = await user_service.get_by_id(UUID(user_id))

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "User not found",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.

    Args:
        user: Current user from token

    Returns:
        Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": True,
                "code": ErrorCode.INSUFFICIENT_PERMISSIONS.value,
                "message": "Account is disabled",
            },
        )
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """
    Get current user if authenticated, None otherwise.
    Used for endpoints that work with or without authentication.
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for logging and request tracking.
    """

    async def dispatch(self, request: Request, call_next):
        # Add request ID for tracking
        request.state.request_id = request.headers.get(
            "X-Request-ID",
            str(UUID.uuid4()) if hasattr(UUID, 'uuid4') else "unknown",
        )

        # Process request
        response = await call_next(request)

        return response
