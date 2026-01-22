"""
Authentication contracts.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr, SecretStr, field_validator
import re

from .common import BaseContract


class LoginRequest(BaseModel):
    """Login request contract."""

    email: EmailStr
    password: SecretStr = Field(min_length=8)


class RegisterRequest(BaseModel):
    """Registration request contract."""

    email: EmailStr
    password: SecretStr = Field(min_length=8)
    name: str = Field(min_length=2, max_length=100)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: SecretStr) -> SecretStr:
        password = v.get_secret_value()
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", password):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", password):
            raise ValueError("Password must contain at least one digit")
        return v


class AuthToken(BaseContract):
    """Authentication token response - matches spec exactly."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token expiration time in seconds")


class TokenRefreshRequest(BaseModel):
    """Token refresh request contract."""

    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request contract."""

    current_password: SecretStr
    new_password: SecretStr = Field(min_length=8)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: SecretStr) -> SecretStr:
        password = v.get_secret_value()
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", password):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", password):
            raise ValueError("Password must contain at least one digit")
        return v


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str  # User ID
    email: Optional[str] = None
    exp: datetime
    iat: datetime
    type: str = "access"
