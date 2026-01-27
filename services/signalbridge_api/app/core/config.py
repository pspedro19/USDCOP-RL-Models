"""
Application configuration using Pydantic Settings.
Single Source of Truth (SSOT) for all configuration values.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="SignalBridge")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    secret_key: str = Field(default="change-me-in-production-min-32-chars")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/signalbridge"
    )
    database_sync_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/signalbridge"
    )
    database_pool_size: int = Field(default=10)
    database_max_overflow: int = Field(default=20)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    # JWT Configuration
    jwt_secret_key: str = Field(default="change-me-in-production-min-32-chars")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)

    # Vault Encryption (AES-256-GCM requires 32-byte key)
    vault_encryption_key: str = Field(default="change-me-32-byte-encryption-key")

    # Supabase (optional)
    supabase_url: Optional[str] = Field(default=None)
    supabase_key: Optional[str] = Field(default=None)
    supabase_jwt_secret: Optional[str] = Field(default=None)

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60)

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    # SignalBridge Configuration
    trading_mode: str = Field(default="PAPER")  # KILLED|DISABLED|SHADOW|PAPER|STAGING|LIVE
    # WebSocket predictions come from backtest-api (has WebSocket router)
    # MLOps inference API (8090) is HTTP-only for risk management
    inference_ws_url: str = Field(default="ws://usdcop-backtest-api:8000/api/v1/ws/predictions")
    inference_api_url: str = Field(default="http://usdcop-backtest-api:8000")

    # Position Sizing
    default_position_size_usd: float = Field(default=100.0)
    position_size_pct: float = Field(default=0.02)  # 2% of portfolio per trade
    max_position_size_usd: float = Field(default=1000.0)

    # Risk Defaults
    default_max_daily_loss_pct: float = Field(default=2.0)
    default_max_trades_per_day: int = Field(default=10)
    default_cooldown_minutes: int = Field(default=15)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
