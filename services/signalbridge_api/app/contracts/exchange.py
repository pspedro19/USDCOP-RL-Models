"""
Exchange contracts.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, SecretStr
from uuid import UUID

from .common import BaseContract


class SupportedExchange(str, Enum):
    """Supported exchanges as defined in spec."""

    MEXC = "mexc"
    BINANCE = "binance"


class ExchangeCredentialsBase(BaseModel):
    """Base exchange credentials."""

    exchange: SupportedExchange
    label: str = Field(min_length=1, max_length=100)
    is_testnet: bool = False


class ExchangeCredentialsCreate(ExchangeCredentialsBase):
    """Create exchange credentials contract."""

    api_key: SecretStr = Field(min_length=10)
    api_secret: SecretStr = Field(min_length=10)
    passphrase: Optional[SecretStr] = None  # For exchanges that require it


class ExchangeCredentialsUpdate(BaseModel):
    """Update exchange credentials contract."""

    label: Optional[str] = Field(None, min_length=1, max_length=100)
    api_key: Optional[SecretStr] = Field(None, min_length=10)
    api_secret: Optional[SecretStr] = Field(None, min_length=10)
    passphrase: Optional[SecretStr] = None
    is_active: Optional[bool] = None


class ExchangeCredentials(BaseContract):
    """Exchange credentials response (without secrets)."""

    id: UUID
    user_id: UUID
    exchange: SupportedExchange
    label: str
    is_testnet: bool = False
    is_active: bool = True
    is_valid: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_used: Optional[datetime] = None

    # Masked keys for display
    api_key_masked: str = Field(description="Masked API key (e.g., 'abc...xyz')")


class ExchangeBalance(BaseContract):
    """Exchange balance response."""

    asset: str
    free: float = Field(ge=0)
    locked: float = Field(ge=0)
    total: float = Field(ge=0)

    @property
    def available(self) -> float:
        return self.free


class ExchangeInfo(BaseContract):
    """Exchange information."""

    exchange: SupportedExchange
    name: str
    is_connected: bool = False
    supports_testnet: bool = True
    base_url: str
    testnet_url: Optional[str] = None
    rate_limit: int = Field(description="Requests per minute")
    features: List[str] = Field(default_factory=list)


class ExchangeValidationResult(BaseModel):
    """Result of exchange credentials validation."""

    is_valid: bool
    exchange: SupportedExchange
    permissions: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class ExchangeCredentialsWithSecret(ExchangeCredentials):
    """Exchange credentials with decrypted secrets (internal use only)."""

    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
