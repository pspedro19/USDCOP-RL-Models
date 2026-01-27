"""
Authentication middleware and dependencies.
"""

import os
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.database import get_db
from app.core.security import verify_token
from app.core.exceptions import AuthenticationError, ErrorCode
from app.models import User
from app.services.user import UserService


# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)

# Dev mode flag - skip authentication when enabled
DEV_MODE = os.getenv("SIGNALBRIDGE_DEV_MODE", "false").lower() == "true"

# Default dev user ID - use integer to match existing users table schema
DEV_USER_ID = 1  # Integer ID to match existing database schema


class DevUser:
    """Dummy user object for dev mode."""
    def __init__(self):
        self.id = DEV_USER_ID  # Integer ID
        self.email = "admin@trading.usdcop.com"
        self.name = "Dev User"
        self.is_active = True
        self.is_verified = True
        self.last_login = datetime.utcnow()
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
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
