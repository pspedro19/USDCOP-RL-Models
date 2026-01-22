"""
Authentication module for USD/COP Forecasting API.

This module provides JWT-based authentication with:
- Token creation and verification
- User authentication
- FastAPI dependencies for protecting endpoints
"""

from .jwt_handler import (
    create_access_token,
    verify_token,
    decode_token,
    TokenData,
)
from .dependencies import get_current_user

__all__ = [
    "create_access_token",
    "verify_token",
    "decode_token",
    "TokenData",
    "get_current_user",
]
