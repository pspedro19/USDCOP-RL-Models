"""
User contracts.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, SecretStr

from .common import BaseContract


class UserStatus(str, Enum):
    """Account approval lifecycle (SSOT — mirrored by the DB CHECK constraint in
    migration 053 and the TypeScript client). A new registration starts PENDING;
    an admin moves it to APPROVED or REJECTED."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class UserRole(str, Enum):
    """Authorization role (SSOT). Only ADMIN may reach the /api/admin/* routes."""

    USER = "user"
    ADMIN = "admin"


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
    status: UserStatus = UserStatus.APPROVED
    role: UserRole = UserRole.USER
    must_reset_password: bool = False


class UserProfileUpdate(BaseModel):
    """User profile update contract."""

    name: str | None = Field(None, min_length=2, max_length=100)
    email: EmailStr | None = None


class UserInDB(UserProfile):
    """User model as stored in database (internal use)."""

    hashed_password: str
    updated_at: datetime | None = None
