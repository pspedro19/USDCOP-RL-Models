"""
User service for user management operations.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User, TradingConfig
from app.contracts.user import UserCreate, UserProfile, UserProfileUpdate
from app.contracts.auth import RegisterRequest
from app.core.security import get_password_hash, verify_password
from app.core.exceptions import (
    NotFoundError,
    ConflictError,
    AuthenticationError,
    ErrorCode,
)


class UserService:
    """Service for user-related operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def create(self, data: RegisterRequest) -> User:
        """
        Create a new user.

        Args:
            data: Registration data

        Returns:
            Created user

        Raises:
            ConflictError: If email already exists
        """
        # Check if email exists
        existing = await self.get_by_email(data.email)
        if existing:
            raise ConflictError(
                message="Email already registered",
                details={"email": data.email},
            )

        # Create user
        user = User(
            email=data.email.lower(),
            hashed_password=get_password_hash(data.password.get_secret_value()),
            name=data.name,
            is_active=True,
            is_verified=False,
        )

        self.db.add(user)
        await self.db.flush()

        # Create default trading config
        trading_config = TradingConfig(
            user_id=user.id,
            trading_enabled=False,
        )
        self.db.add(trading_config)

        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def authenticate(self, email: str, password: str) -> User:
        """
        Authenticate user with email and password.

        Args:
            email: User email
            password: User password

        Returns:
            Authenticated user

        Raises:
            AuthenticationError: If credentials are invalid
        """
        user = await self.get_by_email(email)

        if not user:
            raise AuthenticationError(
                message="Invalid email or password",
                error_code=ErrorCode.INVALID_CREDENTIALS,
            )

        if not verify_password(password, user.hashed_password):
            raise AuthenticationError(
                message="Invalid email or password",
                error_code=ErrorCode.INVALID_CREDENTIALS,
            )

        if not user.is_active:
            raise AuthenticationError(
                message="Account is disabled",
                error_code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            )

        # Update last login
        user.last_login = datetime.utcnow()
        await self.db.commit()

        return user

    async def update_profile(
        self,
        user_id: UUID,
        data: UserProfileUpdate,
    ) -> User:
        """
        Update user profile.

        Args:
            user_id: User ID
            data: Update data

        Returns:
            Updated user

        Raises:
            NotFoundError: If user not found
        """
        user = await self.get_by_id(user_id)

        if not user:
            raise NotFoundError(
                message="User not found",
                resource_type="User",
                resource_id=str(user_id),
            )

        # Check email uniqueness if changing
        if data.email and data.email.lower() != user.email:
            existing = await self.get_by_email(data.email)
            if existing:
                raise ConflictError(
                    message="Email already in use",
                    details={"email": data.email},
                )
            user.email = data.email.lower()

        if data.name:
            user.name = data.name

        user.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            True if successful

        Raises:
            NotFoundError: If user not found
            AuthenticationError: If current password is wrong
        """
        user = await self.get_by_id(user_id)

        if not user:
            raise NotFoundError(
                message="User not found",
                resource_type="User",
                resource_id=str(user_id),
            )

        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationError(
                message="Current password is incorrect",
                error_code=ErrorCode.INVALID_CREDENTIALS,
            )

        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        await self.db.commit()

        return True

    def to_profile(self, user: User) -> UserProfile:
        """Convert User model to UserProfile response."""
        return UserProfile(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.created_at,
            last_login=user.last_login,
            is_active=user.is_active,
            is_verified=user.is_verified,
        )
