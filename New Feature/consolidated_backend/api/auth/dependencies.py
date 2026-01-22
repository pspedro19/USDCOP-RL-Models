"""
FastAPI dependencies for authentication.

Provides dependency injection functions for protecting endpoints.
Users are now stored in PostgreSQL database (auth.users table).
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from .jwt_handler import decode_token, verify_token, TokenData
from ..database import get_user_by_username, verify_user_credentials

logger = logging.getLogger(__name__)

# OAuth2 scheme for token extraction from Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Optional OAuth2 scheme (doesn't raise error if token is missing)
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


class User(BaseModel):
    """User model for authenticated users."""
    username: str
    role: str = "viewer"
    is_active: bool = True


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user with username and password.

    Verifies credentials against PostgreSQL database using bcrypt password hashing.

    Args:
        username: The username to authenticate.
        password: The password to verify.

    Returns:
        User object if authentication successful, None otherwise.
    """
    try:
        # Verify credentials against database (handles bcrypt password verification)
        user_data = verify_user_credentials(username, password)

        if not user_data:
            return None

        return User(
            username=user_data["username"],
            role=user_data["role"],
            is_active=user_data["is_active"],
        )

    except Exception as e:
        logger.error(f"Authentication error for user '{username}': {e}")
        return None


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    FastAPI dependency to get the current authenticated user.

    This dependency extracts the JWT token from the Authorization header,
    validates it, and returns the associated user from PostgreSQL database.

    Args:
        token: JWT token extracted from Authorization header by OAuth2PasswordBearer.

    Returns:
        User object for the authenticated user.

    Raises:
        HTTPException: 401 Unauthorized if token is invalid or expired.

    Usage:
        @router.get("/protected")
        async def protected_route(current_user: User = Depends(get_current_user)):
            return {"message": f"Hello {current_user.username}"}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Verify token is valid
    if not verify_token(token):
        raise credentials_exception

    # Decode token to get user data
    token_data = decode_token(token)
    if token_data is None:
        raise credentials_exception

    username = token_data.username
    if username is None:
        raise credentials_exception

    # Get user from PostgreSQL database
    try:
        user_data = get_user_by_username(username)
    except Exception as e:
        logger.error(f"Database error fetching user '{username}': {e}")
        raise credentials_exception

    if user_data is None:
        raise credentials_exception

    if not user_data.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return User(
        username=user_data["username"],
        role=user_data["role"],
        is_active=user_data["is_active"],
    )


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme_optional)
) -> Optional[User]:
    """
    Optional authentication dependency.

    Returns the user if a valid token is provided, None otherwise.
    Does not raise an exception if no token is provided.

    Usage:
        @router.get("/optional-auth")
        async def optional_route(current_user: Optional[User] = Depends(get_current_user_optional)):
            if current_user:
                return {"message": f"Hello {current_user.username}"}
            return {"message": "Hello anonymous"}
    """
    if token is None:
        return None

    try:
        return await get_current_user(token)
    except HTTPException:
        return None


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to require admin role.

    Args:
        current_user: The authenticated user.

    Returns:
        User object if user has admin role.

    Raises:
        HTTPException: 403 Forbidden if user is not an admin.

    Usage:
        @router.delete("/admin-only")
        async def admin_route(admin: User = Depends(get_current_admin_user)):
            return {"message": "Admin access granted"}
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user
