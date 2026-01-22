"""
JWT Token Handler for authentication.

Uses python-jose with HS256 algorithm for token creation and verification.
SECRET_KEY is read from environment variable for security.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel
from jose import JWTError, jwt

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # Default 24 hours


class TokenData(BaseModel):
    """Token payload data model."""
    username: Optional[str] = None
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    sub: Optional[str] = None


class TokenPayload(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a new JWT access token.

    Args:
        data: Dictionary containing the claims to encode in the token.
              Should include 'sub' (subject/username) at minimum.
        expires_delta: Optional custom expiration time.
                      Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        Encoded JWT token as string.

    Example:
        >>> token = create_access_token({"sub": "admin", "role": "admin"})
        >>> print(token)
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
    """
    to_encode = data.copy()

    # Set expiration time
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    # Add standard claims
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    })

    # Encode the token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> bool:
    """
    Verify if a JWT token is valid.

    Args:
        token: The JWT token string to verify.

    Returns:
        True if token is valid, False otherwise.

    Example:
        >>> is_valid = verify_token(token)
        >>> if is_valid:
        ...     print("Token is valid")
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Check if token has expired
        exp = payload.get("exp")
        if exp is None:
            return False

        # Convert to datetime and check expiration
        exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
        if datetime.now(timezone.utc) > exp_datetime:
            return False

        # Check if subject exists
        if payload.get("sub") is None:
            return False

        return True

    except JWTError:
        return False


def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode a JWT token and extract its payload.

    Args:
        token: The JWT token string to decode.

    Returns:
        TokenData object with the decoded claims, or None if invalid.

    Raises:
        JWTError: If the token is invalid or expired (caught internally).

    Example:
        >>> token_data = decode_token(token)
        >>> if token_data:
        ...     print(f"User: {token_data.username}")
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        username: str = payload.get("sub")
        if username is None:
            return None

        # Extract expiration
        exp = payload.get("exp")
        exp_datetime = None
        if exp:
            exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)

        # Extract issued at
        iat = payload.get("iat")
        iat_datetime = None
        if iat:
            iat_datetime = datetime.fromtimestamp(iat, tz=timezone.utc)

        return TokenData(
            username=username,
            sub=username,
            exp=exp_datetime,
            iat=iat_datetime,
        )

    except JWTError:
        return None


def get_token_expiry_seconds() -> int:
    """
    Get the token expiry time in seconds.

    Returns:
        Number of seconds until token expires.
    """
    return ACCESS_TOKEN_EXPIRE_MINUTES * 60
