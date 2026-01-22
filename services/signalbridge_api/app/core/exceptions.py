"""
Custom exceptions for SignalBridge application.
Follows the error handling contract from the spec.
"""

from enum import Enum
from typing import Optional, Any, Dict
from fastapi import HTTPException, status


class ErrorCode(str, Enum):
    """Error codes as defined in the spec."""

    # Authentication Errors (1xxx)
    INVALID_CREDENTIALS = "AUTH_1001"
    TOKEN_EXPIRED = "AUTH_1002"
    TOKEN_INVALID = "AUTH_1003"
    INSUFFICIENT_PERMISSIONS = "AUTH_1004"

    # Validation Errors (2xxx)
    VALIDATION_ERROR = "VAL_2001"
    INVALID_EXCHANGE = "VAL_2002"
    INVALID_SYMBOL = "VAL_2003"
    INVALID_QUANTITY = "VAL_2004"

    # Exchange Errors (3xxx)
    EXCHANGE_CONNECTION_FAILED = "EXC_3001"
    EXCHANGE_AUTHENTICATION_FAILED = "EXC_3002"
    INSUFFICIENT_BALANCE = "EXC_3003"
    ORDER_REJECTED = "EXC_3004"
    RATE_LIMITED = "EXC_3005"

    # Vault Errors (4xxx)
    VAULT_ENCRYPTION_FAILED = "VLT_4001"
    VAULT_DECRYPTION_FAILED = "VLT_4002"
    VAULT_KEY_NOT_FOUND = "VLT_4003"

    # System Errors (5xxx)
    INTERNAL_ERROR = "SYS_5001"
    DATABASE_ERROR = "SYS_5002"
    CACHE_ERROR = "SYS_5003"
    QUEUE_ERROR = "SYS_5004"

    # Resource Errors (6xxx)
    NOT_FOUND = "RES_6001"
    ALREADY_EXISTS = "RES_6002"
    CONFLICT = "RES_6003"


class SignalBridgeException(Exception):
    """Base exception for all SignalBridge errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": True,
            "code": self.error_code.value,
            "message": self.message,
            "details": self.details,
        }

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict(),
        )


class AuthenticationError(SignalBridgeException):
    """Authentication-related errors."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: ErrorCode = ErrorCode.INVALID_CREDENTIALS,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class AuthorizationError(SignalBridgeException):
    """Authorization-related errors."""

    def __init__(
        self,
        message: str = "Access denied",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class ValidationError(SignalBridgeException):
    """Validation-related errors."""

    def __init__(
        self,
        message: str = "Validation error",
        error_code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class ExchangeError(SignalBridgeException):
    """Exchange-related errors."""

    def __init__(
        self,
        message: str = "Exchange error",
        error_code: ErrorCode = ErrorCode.EXCHANGE_CONNECTION_FAILED,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details,
        )


class VaultError(SignalBridgeException):
    """Vault/encryption-related errors."""

    def __init__(
        self,
        message: str = "Vault error",
        error_code: ErrorCode = ErrorCode.VAULT_ENCRYPTION_FAILED,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class RateLimitError(SignalBridgeException):
    """Rate limiting errors."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if retry_after:
            details = details or {}
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMITED,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
        )


class NotFoundError(SignalBridgeException):
    """Resource not found errors."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if resource_type or resource_id:
            details = details or {}
            if resource_type:
                details["resource_type"] = resource_type
            if resource_id:
                details["resource_id"] = resource_id

        super().__init__(
            message=message,
            error_code=ErrorCode.NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class ConflictError(SignalBridgeException):
    """Resource conflict errors."""

    def __init__(
        self,
        message: str = "Resource already exists",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.ALREADY_EXISTS,
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )
