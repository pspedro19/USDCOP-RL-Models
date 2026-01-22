"""
Shared module - Configuration, exceptions, utilities, schemas, secrets, tracing, and notifications.

Submodules:
    - config_loader: Configuration loading utilities
    - exceptions: Custom exception classes
    - schemas: Shared Pydantic schemas for API contracts
    - secrets: HashiCorp Vault client for secrets management
    - tracing: OpenTelemetry distributed tracing utilities
    - notifications: Slack notifications for trading events (P1-1)
"""

from .config_loader import ConfigLoader, get_config, load_feature_config
from .config_loader_adapter import ConfigLoaderAdapter
from .exceptions import (
    USDCOPError,
    ConfigurationError,
    FeatureCalculationError,
    ValidationError,
    ObservationDimensionError,
    FeatureMissingError,
    # Legacy aliases
    FeatureBuilderError,
    NormalizationError
)

# Schemas submodule (import on demand to avoid circular imports)
# Usage: from shared.schemas import TradeSchema, SignalType

# Secrets submodule (import on demand to avoid ImportError if hvac not installed)
# Usage: from shared.secrets import VaultClient, get_vault_client

# Tracing submodule (import on demand to avoid ImportError if otel not installed)
# Usage: from shared.tracing import init_tracing, traced, MLSpanBuilder

# Notifications submodule (import on demand to avoid ImportError if aiohttp not installed)
# Usage: from shared.notifications import SlackClient, send_slack_alert, AlertSeverity

__all__ = [
    'ConfigLoader',
    'ConfigLoaderAdapter',
    'get_config',
    'load_feature_config',
    'USDCOPError',
    'ConfigurationError',
    'FeatureCalculationError',
    'ValidationError',
    'ObservationDimensionError',
    'FeatureMissingError',
    # Legacy
    'FeatureBuilderError',
    'NormalizationError',
    # Submodules
    'schemas',
    'secrets',
    'tracing',
    'notifications',
]
