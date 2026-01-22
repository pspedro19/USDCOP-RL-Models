"""
Authentication routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
)
from app.core.config import settings
from app.core.exceptions import AuthenticationError, ErrorCode
from app.contracts.auth import (
    LoginRequest,
    RegisterRequest,
    AuthToken,
    TokenRefreshRequest,
)
from app.contracts.user import UserProfile
from app.services.user import UserService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=AuthToken, status_code=status.HTTP_201_CREATED)
async def register(
    data: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user account.

    Returns access and refresh tokens upon successful registration.
    """
    user_service = UserService(db)

    # Create user
    user = await user_service.create(data)

    # Generate tokens
    token_data = {"sub": str(user.id), "email": user.email}

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return AuthToken(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/login", response_model=AuthToken)
async def login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user and return tokens.

    Returns access and refresh tokens upon successful authentication.
    """
    user_service = UserService(db)

    # Authenticate user
    user = await user_service.authenticate(
        email=data.email,
        password=data.password.get_secret_value(),
    )

    # Generate tokens
    token_data = {"sub": str(user.id), "email": user.email}

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return AuthToken(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=AuthToken)
async def refresh_token(
    data: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Refresh access token using refresh token.

    Returns new access and refresh tokens.
    """
    # Verify refresh token
    payload = verify_token(data.refresh_token, token_type="refresh")

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "Invalid or expired refresh token",
            },
        )

    user_id = payload.get("sub")
    email = payload.get("email")

    # Verify user still exists and is active
    user_service = UserService(db)
    user = await user_service.get_by_id(user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "code": ErrorCode.TOKEN_INVALID.value,
                "message": "User not found or inactive",
            },
        )

    # Generate new tokens
    token_data = {"sub": str(user.id), "email": user.email}

    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)

    return AuthToken(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/logout")
async def logout():
    """
    Logout user.

    Note: Token invalidation would require a token blacklist (Redis).
    For now, this is a no-op - client should discard tokens.
    """
    return {"message": "Logged out successfully"}
