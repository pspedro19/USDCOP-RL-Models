"""
Authentication endpoints.

Provides login and token refresh functionality.
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

from ..auth.jwt_handler import (
    create_access_token,
    decode_token,
    verify_token,
    get_token_expiry_seconds,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from ..auth.dependencies import (
    authenticate_user,
    get_current_user,
    User,
    oauth2_scheme,
)


router = APIRouter()


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Refresh token request model."""
    token: str


class UserResponse(BaseModel):
    """User response model (without sensitive data)."""
    username: str
    role: str
    is_active: bool


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT access token.

    This endpoint accepts OAuth2 password flow credentials.

    Args:
        form_data: OAuth2PasswordRequestForm with username and password.

    Returns:
        Token object containing access_token, token_type, and expires_in.

    Raises:
        HTTPException: 401 Unauthorized if credentials are invalid.

    Example:
        POST /auth/login
        Content-Type: application/x-www-form-urlencoded

        username=admin&password=admin123

        Response:
        {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer",
            "expires_in": 86400
        }
    """
    # Authenticate user
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
        }
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=get_token_expiry_seconds(),
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: User = Depends(get_current_user)):
    """
    Refresh an existing valid JWT token.

    Requires a valid token in the Authorization header.
    Returns a new token with refreshed expiration time.

    Args:
        current_user: The authenticated user from the current token.

    Returns:
        Token object containing new access_token.

    Raises:
        HTTPException: 401 Unauthorized if current token is invalid.

    Example:
        POST /auth/refresh
        Authorization: Bearer <current_token>

        Response:
        {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer",
            "expires_in": 86400
        }
    """
    # Create new access token with same user data
    access_token = create_access_token(
        data={
            "sub": current_user.username,
            "role": current_user.role,
        }
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=get_token_expiry_seconds(),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information.

    Args:
        current_user: The authenticated user.

    Returns:
        UserResponse with user details (excluding password).

    Example:
        GET /auth/me
        Authorization: Bearer <token>

        Response:
        {
            "username": "admin",
            "role": "admin",
            "is_active": true
        }
    """
    return UserResponse(
        username=current_user.username,
        role=current_user.role,
        is_active=current_user.is_active,
    )


@router.post("/verify")
async def verify_token_endpoint(token: str = Depends(oauth2_scheme)):
    """
    Verify if a token is valid.

    Args:
        token: JWT token from Authorization header.

    Returns:
        Dictionary with valid status and token data.

    Example:
        POST /auth/verify
        Authorization: Bearer <token>

        Response:
        {
            "valid": true,
            "username": "admin",
            "role": "admin"
        }
    """
    if not verify_token(token):
        return {
            "valid": False,
            "message": "Token is invalid or expired",
        }

    token_data = decode_token(token)
    if not token_data:
        return {
            "valid": False,
            "message": "Could not decode token",
        }

    return {
        "valid": True,
        "username": token_data.username,
        "exp": token_data.exp.isoformat() if token_data.exp else None,
        "iat": token_data.iat.isoformat() if token_data.iat else None,
    }
