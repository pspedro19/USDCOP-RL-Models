"""
SecretManager - Secure secrets management for USDCOP Trading System.

This module provides a unified interface for accessing secrets from multiple sources:
1. Docker secrets (mounted at /run/secrets/)
2. Local secret files (./secrets/)
3. Environment variables (fallback)

Usage:
    from src.core.secrets import SecretManager, get_secret

    # Using the singleton instance
    password = get_secret("db_password")

    # Or create a custom instance
    secrets = SecretManager(secrets_dir="/custom/path")
    api_key = secrets.get_secret("api_key")

Security:
    - Secrets are never logged or exposed in error messages
    - Values are cached to minimize file reads
    - Supports both Docker secrets and local development
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Manages access to secrets from Docker secrets, local files, or environment variables.

    Priority order:
    1. Docker secrets (/run/secrets/<name>)
    2. Local secret files (./secrets/<name>.txt)
    3. Environment variables (<NAME> in uppercase)

    Attributes:
        docker_secrets_path: Path to Docker secrets (default: /run/secrets)
        local_secrets_path: Path to local secrets directory (default: ./secrets)
    """

    # Default paths
    DOCKER_SECRETS_PATH = Path("/run/secrets")
    DEFAULT_LOCAL_SECRETS_PATH = Path("secrets")

    # Singleton instance
    _instance: Optional["SecretManager"] = None

    def __init__(
        self,
        secrets_dir: Optional[str] = None,
        docker_secrets_path: Optional[str] = None,
        cache_secrets: bool = True
    ):
        """
        Initialize the SecretManager.

        Args:
            secrets_dir: Path to local secrets directory. If None, uses ./secrets
            docker_secrets_path: Path to Docker secrets. If None, uses /run/secrets
            cache_secrets: Whether to cache secret values (default: True)
        """
        self.docker_secrets_path = Path(docker_secrets_path) if docker_secrets_path else self.DOCKER_SECRETS_PATH

        # Resolve local secrets path relative to project root
        if secrets_dir:
            self.local_secrets_path = Path(secrets_dir)
        else:
            # Try to find secrets directory relative to this file or cwd
            self.local_secrets_path = self._find_secrets_dir()

        self._cache_enabled = cache_secrets
        self._cache: Dict[str, str] = {}

        logger.debug(f"SecretManager initialized with local_secrets_path={self.local_secrets_path}")

    def _find_secrets_dir(self) -> Path:
        """Find the secrets directory relative to the project root."""
        # Try common locations
        possible_paths = [
            Path.cwd() / "secrets",
            Path(__file__).parent.parent.parent.parent.parent / "secrets",  # Relative to this file
            Path("/app/secrets"),  # Docker container
            Path.home() / ".usdcop" / "secrets",  # User home
        ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                return path

        # Default to cwd/secrets even if it doesn't exist
        return Path.cwd() / "secrets"

    @classmethod
    def get_instance(cls) -> "SecretManager":
        """Get the singleton instance of SecretManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value by name.

        Lookup priority:
        1. Docker secrets (/run/secrets/<name>)
        2. Local secret files (./secrets/<name>.txt)
        3. Environment variables (<NAME> in uppercase)

        Args:
            name: The secret name (e.g., "db_password", "api_key")
            default: Default value if secret is not found

        Returns:
            The secret value, or default if not found

        Example:
            >>> secrets = SecretManager()
            >>> password = secrets.get_secret("db_password")
            >>> api_key = secrets.get_secret("api_key", default="")
        """
        # Check cache first
        if self._cache_enabled and name in self._cache:
            return self._cache[name]

        value = None

        # 1. Try Docker secrets
        value = self._read_docker_secret(name)

        # 2. Try local secret file
        if value is None:
            value = self._read_local_secret(name)

        # 3. Try environment variable
        if value is None:
            value = self._read_env_var(name)

        # Cache the value if found
        if value is not None and self._cache_enabled:
            self._cache[name] = value
            return value

        if value is None and default is not None:
            return default

        return value

    def _read_docker_secret(self, name: str) -> Optional[str]:
        """Read a secret from Docker secrets path."""
        secret_path = self.docker_secrets_path / name

        if secret_path.exists() and secret_path.is_file():
            try:
                with open(secret_path, "r", encoding="utf-8") as f:
                    value = f.read().strip()
                    if value:
                        logger.debug(f"Secret '{name}' loaded from Docker secrets")
                        return value
            except (IOError, PermissionError) as e:
                logger.warning(f"Failed to read Docker secret '{name}': {type(e).__name__}")

        return None

    def _read_local_secret(self, name: str) -> Optional[str]:
        """Read a secret from local secrets directory."""
        # Try with and without .txt extension
        possible_paths = [
            self.local_secrets_path / f"{name}.txt",
            self.local_secrets_path / name,
        ]

        for secret_path in possible_paths:
            if secret_path.exists() and secret_path.is_file():
                try:
                    with open(secret_path, "r", encoding="utf-8") as f:
                        value = f.read().strip()
                        if value:
                            logger.debug(f"Secret '{name}' loaded from local file")
                            return value
                except (IOError, PermissionError) as e:
                    logger.warning(f"Failed to read local secret '{name}': {type(e).__name__}")

        return None

    def _read_env_var(self, name: str) -> Optional[str]:
        """Read a secret from environment variable."""
        # Try exact name and uppercase version
        env_names = [name, name.upper(), name.lower()]

        for env_name in env_names:
            value = os.environ.get(env_name)
            if value:
                logger.debug(f"Secret '{name}' loaded from environment variable '{env_name}'")
                return value

        return None

    def get_secret_required(self, name: str) -> str:
        """
        Get a required secret value.

        Raises:
            ValueError: If the secret is not found

        Args:
            name: The secret name

        Returns:
            The secret value
        """
        value = self.get_secret(name)
        if value is None:
            raise ValueError(
                f"Required secret '{name}' not found. "
                f"Please create {self.local_secrets_path}/{name}.txt or set the {name.upper()} environment variable."
            )
        return value

    def clear_cache(self):
        """Clear the secrets cache."""
        self._cache.clear()

    def list_available_secrets(self) -> Dict[str, str]:
        """
        List all available secrets and their sources.

        Returns:
            Dictionary mapping secret names to their source (docker/local/env)
        """
        available = {}

        # Check Docker secrets
        if self.docker_secrets_path.exists():
            for path in self.docker_secrets_path.iterdir():
                if path.is_file():
                    available[path.name] = "docker"

        # Check local secrets
        if self.local_secrets_path.exists():
            for path in self.local_secrets_path.iterdir():
                if path.is_file() and not path.name.startswith("."):
                    name = path.stem if path.suffix == ".txt" else path.name
                    if name not in available:
                        available[name] = "local"

        return available

    # Convenience methods for common secrets
    def get_db_password(self) -> Optional[str]:
        """Get database password."""
        return self.get_secret("db_password") or self.get_secret("POSTGRES_PASSWORD")

    def get_redis_password(self) -> Optional[str]:
        """Get Redis password."""
        return self.get_secret("redis_password") or self.get_secret("REDIS_PASSWORD")

    def get_minio_secret_key(self) -> Optional[str]:
        """Get MinIO secret key."""
        return self.get_secret("minio_secret_key") or self.get_secret("MINIO_SECRET_KEY")

    def get_api_key(self, provider: str, index: int = 1) -> Optional[str]:
        """
        Get an API key for a provider.

        Args:
            provider: API provider name (e.g., "twelvedata", "deepseek")
            index: Key index for providers with multiple keys (default: 1)

        Returns:
            The API key or None
        """
        names_to_try = [
            f"{provider}_api_key_{index}",
            f"{provider}_api_key",
            f"API_KEY_{provider.upper()}_{index}",
            f"{provider.upper()}_API_KEY",
        ]

        for name in names_to_try:
            value = self.get_secret(name)
            if value:
                return value

        return None


# Module-level singleton access
_default_manager: Optional[SecretManager] = None


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret using the default SecretManager instance.

    This is a convenience function for quick access to secrets.

    Args:
        name: The secret name
        default: Default value if not found

    Returns:
        The secret value or default

    Example:
        >>> from src.core.secrets import get_secret
        >>> password = get_secret("db_password")
        >>> api_key = get_secret("twelvedata_api_key_1", default="")
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = SecretManager()
    return _default_manager.get_secret(name, default)


def get_secret_required(name: str) -> str:
    """
    Get a required secret using the default SecretManager instance.

    Raises:
        ValueError: If the secret is not found

    Args:
        name: The secret name

    Returns:
        The secret value
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = SecretManager()
    return _default_manager.get_secret_required(name)


# Backwards compatibility with simple get_secret function
@lru_cache(maxsize=128)
def get_secret_cached(name: str) -> Optional[str]:
    """Get a secret with caching (deprecated, use SecretManager instead)."""
    return get_secret(name)
