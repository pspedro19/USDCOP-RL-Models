"""
User contracts.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr, SecretStr
from uuid import UUID

from .common import BaseContract, TimestampMixin


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
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False


class UserProfileUpdate(BaseModel):
    """User profile update contract."""

    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None


class UserInDB(UserProfile):
    """User model as stored in database (internal use)."""

    hashed_password: str
    updated_at: Optional[datetime] = None
