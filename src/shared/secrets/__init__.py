"""
USD/COP Trading System - Secrets Management Module
===================================================

This module provides secure secrets management with HashiCorp Vault integration
and fallback to environment variables.

Components:
    - VaultClient: Main client for interacting with Vault
    - VaultConfig: Configuration dataclass for VaultClient
    - get_vault_client: Helper function to get the singleton instance
    - require_secret: Decorator for injecting secrets into functions

Exceptions:
    - VaultError: Base exception for Vault-related errors
    - VaultConnectionError: Connection to Vault failed
    - VaultAuthenticationError: Authentication with Vault failed
    - VaultSecretNotFoundError: Requested secret not found

Usage:
    from src.shared.secrets import VaultClient, get_vault_client

    # Get the global client instance
    client = get_vault_client()

    # Get a secret
    api_key = client.get_secret("trading/twelvedata", "api_key_1")

    # Get TwelveData API key by index
    key = client.get_twelvedata_key(1)

    # Get database password
    password = client.get_database_password()

    # Get full database config
    db_config = client.get_database_config()

    # Health check
    health = client.health_check()

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2026-01-17
"""

from .vault_client import (
    # Main classes
    VaultClient,
    VaultConfig,
    CachedSecret,
    # Helper functions
    get_vault_client,
    require_secret,
    # Exceptions
    VaultError,
    VaultConnectionError,
    VaultAuthenticationError,
    VaultSecretNotFoundError,
)

__all__ = [
    # Main classes
    "VaultClient",
    "VaultConfig",
    "CachedSecret",
    # Helper functions
    "get_vault_client",
    "require_secret",
    # Exceptions
    "VaultError",
    "VaultConnectionError",
    "VaultAuthenticationError",
    "VaultSecretNotFoundError",
]

__version__ = "1.0.0"
