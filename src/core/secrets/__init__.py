"""
Secrets management module for USDCOP Trading System.

This module provides secure access to credentials and API keys through:
1. Docker secrets (/run/secrets/)
2. Local secret files (./secrets/)
3. Environment variables (fallback)
"""

from .secret_manager import SecretManager, get_secret

__all__ = ["SecretManager", "get_secret"]
