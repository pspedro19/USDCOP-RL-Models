"""
User contracts.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, SecretStr

from .common import BaseContract


class UserBase(BaseModel):
    """Base user data."""

    email: EmailStr
    name: str = Field(min_length=2, max_length=100)


class UserCreate(UserBase):
    """User creation contract."""

    password: SecretStr = Field(min_length=8)


class UserProfile(BaseContract):
    """User profile response - matches spec UserProfile."""

    id: UUID
    email: EmailStr
    name: str
    created_at: datetime
    last_login: datetime | None = None
    is_active: bool = True
    is_verified: bool = False


class UserProfileUpdate(BaseModel):
    """User profile update contract."""

    name: str | None = Field(None, min_length=2, max_length=100)
    email: EmailStr | None = None


class UserInDB(UserProfile):
    """User model as stored in database (internal use)."""

    hashed_password: str
    updated_at: datetime | None = None
