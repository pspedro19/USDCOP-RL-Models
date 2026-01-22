"""
USD/COP Trading System - HashiCorp Vault Client
=================================================

Production-ready Vault client with:
- AppRole authentication support
- Token-based authentication fallback
- Secret caching with configurable TTL
- Auto token renewal
- Fallback to environment variables when Vault unavailable
- Comprehensive error handling and logging

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Date: 2026-01-17
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from functools import wraps

# Flexible import for exceptions
try:
    from ..exceptions import USDCOPError
except ImportError:
    # When loaded standalone
    _shared_path = Path(__file__).parent.parent
    if str(_shared_path) not in sys.path:
        sys.path.insert(0, str(_shared_path))
    from exceptions import USDCOPError


# Configure module logger
logger = logging.getLogger(__name__)


class VaultError(USDCOPError):
    """Exception raised for Vault-related errors."""

    def __init__(self, message: str, vault_path: str = None, error_code: str = None):
        details = {}
        if vault_path:
            details["vault_path"] = vault_path
        if error_code:
            details["error_code"] = error_code
        super().__init__(message, details)


class VaultConnectionError(VaultError):
    """Exception raised when unable to connect to Vault."""
    pass


class VaultAuthenticationError(VaultError):
    """Exception raised when Vault authentication fails."""
    pass


class VaultSecretNotFoundError(VaultError):
    """Exception raised when a secret is not found in Vault."""
    pass


@dataclass
class CachedSecret:
    """Represents a cached secret with its expiration time."""
    value: Any
    expires_at: float
    path: str
    key: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if the cached secret has expired."""
        return time.time() > self.expires_at


@dataclass
class VaultConfig:
    """Configuration for VaultClient."""
    # Vault server URL
    vault_addr: str = field(default_factory=lambda: os.environ.get("VAULT_ADDR", "http://localhost:8200"))

    # Authentication method: 'approle', 'token', 'kubernetes'
    auth_method: str = field(default_factory=lambda: os.environ.get("VAULT_AUTH_METHOD", "approle"))

    # AppRole credentials
    role_id: Optional[str] = field(default_factory=lambda: os.environ.get("VAULT_ROLE_ID"))
    secret_id: Optional[str] = field(default_factory=lambda: os.environ.get("VAULT_SECRET_ID"))

    # Token authentication
    token: Optional[str] = field(default_factory=lambda: os.environ.get("VAULT_TOKEN"))

    # KV v2 secrets mount path
    mount_path: str = field(default_factory=lambda: os.environ.get("VAULT_MOUNT_PATH", "secret"))

    # Cache settings
    cache_ttl: int = field(default_factory=lambda: int(os.environ.get("VAULT_CACHE_TTL", "300")))

    # Connection settings
    timeout: int = field(default_factory=lambda: int(os.environ.get("VAULT_TIMEOUT", "30")))
    verify_ssl: bool = field(default_factory=lambda: os.environ.get("VAULT_VERIFY_SSL", "true").lower() == "true")

    # Fallback behavior
    fallback_to_env: bool = field(default_factory=lambda: os.environ.get("VAULT_FALLBACK_TO_ENV", "true").lower() == "true")

    # Auto renewal threshold (renew when token TTL is below this, in seconds)
    renewal_threshold: int = field(default_factory=lambda: int(os.environ.get("VAULT_RENEWAL_THRESHOLD", "300")))


class VaultClient:
    """
    Production-ready HashiCorp Vault client for the USDCOP Trading System.

    Features:
        - AppRole authentication (recommended for services)
        - Token-based authentication (for development/CLI)
        - Secret caching with TTL to reduce Vault load
        - Automatic token renewal before expiration
        - Graceful fallback to environment variables when Vault is unavailable

    Usage:
        # Using default configuration from environment
        client = VaultClient()

        # With custom configuration
        config = VaultConfig(vault_addr="http://vault:8200", cache_ttl=600)
        client = VaultClient(config=config)

        # Get a secret
        api_key = client.get_secret("trading/twelvedata", "api_key_1")

        # Get TwelveData API key by index
        key = client.get_twelvedata_key(1)

        # Get database password
        password = client.get_database_password()

    Environment Variables:
        VAULT_ADDR: Vault server URL (default: http://localhost:8200)
        VAULT_AUTH_METHOD: Authentication method (approle, token, kubernetes)
        VAULT_ROLE_ID: AppRole role ID
        VAULT_SECRET_ID: AppRole secret ID
        VAULT_TOKEN: Static token for token auth
        VAULT_MOUNT_PATH: KV v2 mount path (default: secret)
        VAULT_CACHE_TTL: Cache TTL in seconds (default: 300)
        VAULT_TIMEOUT: Connection timeout in seconds (default: 30)
        VAULT_VERIFY_SSL: Verify SSL certificates (default: true)
        VAULT_FALLBACK_TO_ENV: Fall back to env vars if Vault unavailable (default: true)
    """

    _instance: Optional["VaultClient"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[VaultConfig] = None) -> "VaultClient":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VaultClient, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[VaultConfig] = None):
        """
        Initialize the Vault client.

        Args:
            config: VaultConfig instance. If None, uses environment variables.
        """
        if self._initialized:
            return

        self.config = config or VaultConfig()
        self._client = None
        self._token: Optional[str] = None
        self._token_ttl: Optional[int] = None
        self._token_expires_at: Optional[float] = None
        self._cache: Dict[str, CachedSecret] = {}
        self._cache_lock = threading.RLock()
        self._renewal_timer: Optional[threading.Timer] = None
        self._vault_available: bool = False

        # Try to initialize Vault connection
        self._initialize_vault()
        self._initialized = True

    def _initialize_vault(self) -> None:
        """Initialize connection to Vault server."""
        try:
            import hvac
            self._hvac = hvac
        except ImportError:
            logger.warning(
                "hvac package not installed. Install with: pip install hvac. "
                "Vault integration disabled, falling back to environment variables."
            )
            self._vault_available = False
            return

        try:
            self._client = hvac.Client(
                url=self.config.vault_addr,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )

            # Authenticate based on configured method
            if self.config.auth_method == "approle":
                self._authenticate_approle()
            elif self.config.auth_method == "token":
                self._authenticate_token()
            elif self.config.auth_method == "kubernetes":
                self._authenticate_kubernetes()
            else:
                raise VaultAuthenticationError(
                    f"Unsupported authentication method: {self.config.auth_method}",
                    error_code="UNSUPPORTED_AUTH_METHOD"
                )

            # Verify authentication was successful
            if self._client.is_authenticated():
                self._vault_available = True
                logger.info(
                    f"Successfully connected to Vault at {self.config.vault_addr} "
                    f"using {self.config.auth_method} authentication"
                )
                # Schedule token renewal
                self._schedule_token_renewal()
            else:
                raise VaultAuthenticationError(
                    "Vault authentication failed",
                    error_code="AUTH_FAILED"
                )

        except VaultError:
            raise
        except Exception as e:
            logger.warning(
                f"Failed to connect to Vault: {e}. "
                "Falling back to environment variables if enabled."
            )
            self._vault_available = False
            if not self.config.fallback_to_env:
                raise VaultConnectionError(
                    f"Cannot connect to Vault and fallback is disabled: {e}",
                    error_code="CONNECTION_FAILED"
                )

    def _authenticate_approle(self) -> None:
        """Authenticate using AppRole method."""
        if not self.config.role_id or not self.config.secret_id:
            raise VaultAuthenticationError(
                "AppRole authentication requires VAULT_ROLE_ID and VAULT_SECRET_ID",
                error_code="MISSING_APPROLE_CREDENTIALS"
            )

        try:
            response = self._client.auth.approle.login(
                role_id=self.config.role_id,
                secret_id=self.config.secret_id,
            )
            self._token = response["auth"]["client_token"]
            self._token_ttl = response["auth"]["lease_duration"]
            self._token_expires_at = time.time() + self._token_ttl
            self._client.token = self._token
            logger.debug(f"AppRole authentication successful, token TTL: {self._token_ttl}s")
        except Exception as e:
            raise VaultAuthenticationError(
                f"AppRole authentication failed: {e}",
                error_code="APPROLE_AUTH_FAILED"
            )

    def _authenticate_token(self) -> None:
        """Authenticate using static token."""
        if not self.config.token:
            raise VaultAuthenticationError(
                "Token authentication requires VAULT_TOKEN environment variable",
                error_code="MISSING_TOKEN"
            )

        self._client.token = self.config.token
        self._token = self.config.token

        # Get token TTL
        try:
            token_info = self._client.auth.token.lookup_self()
            self._token_ttl = token_info["data"].get("ttl", 0)
            if self._token_ttl > 0:
                self._token_expires_at = time.time() + self._token_ttl
            logger.debug(f"Token authentication successful, TTL: {self._token_ttl}s")
        except Exception as e:
            logger.warning(f"Could not look up token TTL: {e}")

    def _authenticate_kubernetes(self) -> None:
        """Authenticate using Kubernetes service account."""
        role = os.environ.get("VAULT_K8S_ROLE", "trading-service")
        jwt_path = os.environ.get(
            "VAULT_K8S_JWT_PATH",
            "/var/run/secrets/kubernetes.io/serviceaccount/token"
        )

        try:
            with open(jwt_path, "r") as f:
                jwt = f.read().strip()

            response = self._client.auth.kubernetes.login(
                role=role,
                jwt=jwt,
            )
            self._token = response["auth"]["client_token"]
            self._token_ttl = response["auth"]["lease_duration"]
            self._token_expires_at = time.time() + self._token_ttl
            self._client.token = self._token
            logger.debug(f"Kubernetes authentication successful, token TTL: {self._token_ttl}s")
        except Exception as e:
            raise VaultAuthenticationError(
                f"Kubernetes authentication failed: {e}",
                error_code="K8S_AUTH_FAILED"
            )

    def _schedule_token_renewal(self) -> None:
        """Schedule automatic token renewal before expiration."""
        if not self._token_ttl or self._token_ttl == 0:
            logger.debug("Token has no TTL, skipping renewal scheduling")
            return

        # Calculate when to renew (threshold before expiration)
        renewal_in = max(1, self._token_ttl - self.config.renewal_threshold)

        # Cancel existing timer if any
        if self._renewal_timer:
            self._renewal_timer.cancel()

        self._renewal_timer = threading.Timer(renewal_in, self._renew_token)
        self._renewal_timer.daemon = True
        self._renewal_timer.start()
        logger.debug(f"Token renewal scheduled in {renewal_in} seconds")

    def _renew_token(self) -> None:
        """Renew the current token."""
        if not self._vault_available or not self._client:
            return

        try:
            response = self._client.auth.token.renew_self()
            self._token_ttl = response["auth"]["lease_duration"]
            self._token_expires_at = time.time() + self._token_ttl
            logger.info(f"Token renewed successfully, new TTL: {self._token_ttl}s")

            # Schedule next renewal
            self._schedule_token_renewal()

        except Exception as e:
            logger.error(f"Token renewal failed: {e}")
            # Try to re-authenticate
            try:
                self._initialize_vault()
            except Exception as auth_e:
                logger.error(f"Re-authentication failed: {auth_e}")
                self._vault_available = False

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get a value from cache if not expired."""
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and not cached.is_expired():
                logger.debug(f"Cache hit for {cache_key}")
                return cached.value
            elif cached:
                # Remove expired entry
                del self._cache[cache_key]
        return None

    def _set_in_cache(self, cache_key: str, value: Any, path: str, key: Optional[str] = None) -> None:
        """Store a value in cache with TTL."""
        with self._cache_lock:
            self._cache[cache_key] = CachedSecret(
                value=value,
                expires_at=time.time() + self.config.cache_ttl,
                path=path,
                key=key,
            )
            logger.debug(f"Cached {cache_key} for {self.config.cache_ttl}s")

    def _read_kv_secret(self, path: str) -> Dict[str, Any]:
        """
        Read a secret from KV v2 secrets engine.

        Args:
            path: Secret path (without mount point prefix)

        Returns:
            Dictionary of secret data

        Raises:
            VaultSecretNotFoundError: If secret not found
            VaultError: For other Vault errors
        """
        if not self._vault_available or not self._client:
            raise VaultConnectionError(
                "Vault is not available",
                vault_path=path,
                error_code="VAULT_UNAVAILABLE"
            )

        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.config.mount_path,
            )
            return response["data"]["data"]
        except self._hvac.exceptions.InvalidPath:
            raise VaultSecretNotFoundError(
                f"Secret not found at path: {path}",
                vault_path=path,
                error_code="SECRET_NOT_FOUND"
            )
        except Exception as e:
            raise VaultError(
                f"Failed to read secret: {e}",
                vault_path=path,
                error_code="READ_FAILED"
            )

    def _get_env_fallback(self, path: str, key: str) -> Optional[str]:
        """
        Get secret value from environment variable as fallback.

        The environment variable name is derived from the path and key:
        - trading/twelvedata -> api_key_1 -> TWELVEDATA_API_KEY_1
        - trading/database -> password -> POSTGRES_PASSWORD
        - trading/redis -> password -> REDIS_PASSWORD
        - trading/jwt -> secret -> JWT_SECRET

        Args:
            path: Secret path
            key: Secret key within the path

        Returns:
            Environment variable value or None
        """
        # Define path/key to env var mappings
        env_mappings = {
            ("trading/database", "password"): "POSTGRES_PASSWORD",
            ("trading/database", "user"): "POSTGRES_USER",
            ("trading/database", "host"): "POSTGRES_HOST",
            ("trading/database", "port"): "POSTGRES_PORT",
            ("trading/database", "database"): "POSTGRES_DB",
            ("trading/redis", "password"): "REDIS_PASSWORD",
            ("trading/redis", "host"): "REDIS_HOST",
            ("trading/redis", "port"): "REDIS_PORT",
            ("trading/jwt", "secret"): "JWT_SECRET",
            ("trading/minio", "access_key"): "MINIO_ACCESS_KEY",
            ("trading/minio", "secret_key"): "MINIO_SECRET_KEY",
            ("airflow", "fernet_key"): "AIRFLOW_FERNET_KEY",
            ("airflow", "secret_key"): "AIRFLOW_SECRET_KEY",
            ("airflow", "password"): "AIRFLOW_PASSWORD",
            ("airflow", "user"): "AIRFLOW_USER",
        }

        # Check direct mapping
        env_var = env_mappings.get((path, key))
        if env_var:
            value = os.environ.get(env_var)
            if value:
                logger.debug(f"Fallback to env var {env_var} for {path}/{key}")
                return value

        # Handle TwelveData API keys dynamically
        if "twelvedata" in path.lower():
            # Extract key index from key name (e.g., api_key_1, api_key_g1_1)
            key_lower = key.lower()
            if key_lower.startswith("api_key_"):
                suffix = key_lower.replace("api_key_", "").upper()
                # Try different env var patterns
                env_candidates = [
                    f"TWELVEDATA_API_KEY_{suffix}",
                    f"API_KEY_{suffix}",
                    f"API_KEY_G1_{suffix}",
                    f"API_KEY_G2_{suffix}",
                ]
                for env_var in env_candidates:
                    value = os.environ.get(env_var)
                    if value and not value.startswith("YOUR_"):
                        logger.debug(f"Fallback to env var {env_var} for {path}/{key}")
                        return value

        # Generic fallback: convert path/key to UPPER_SNAKE_CASE
        generic_key = f"{path}_{key}".upper().replace("/", "_").replace("-", "_")
        value = os.environ.get(generic_key)
        if value:
            logger.debug(f"Fallback to env var {generic_key} for {path}/{key}")
            return value

        return None

    def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """
        Get a secret from Vault with caching and environment fallback.

        Args:
            path: Secret path (e.g., "trading/database")
            key: Optional key within the secret. If None, returns entire secret dict.

        Returns:
            Secret value (string if key specified, dict if not)

        Raises:
            VaultSecretNotFoundError: If secret/key not found and no fallback
        """
        cache_key = f"{path}:{key}" if key else path

        # Check cache first
        cached_value = self._get_from_cache(cache_key)
        if cached_value is not None:
            return cached_value

        # Try Vault if available
        if self._vault_available:
            try:
                secret_data = self._read_kv_secret(path)
                if key:
                    if key in secret_data:
                        value = secret_data[key]
                        self._set_in_cache(cache_key, value, path, key)
                        return value
                    else:
                        raise VaultSecretNotFoundError(
                            f"Key '{key}' not found in secret at path: {path}",
                            vault_path=path,
                            error_code="KEY_NOT_FOUND"
                        )
                else:
                    self._set_in_cache(cache_key, secret_data, path)
                    return secret_data
            except VaultSecretNotFoundError:
                raise
            except VaultError as e:
                logger.warning(f"Vault error, trying fallback: {e}")

        # Fallback to environment variables
        if self.config.fallback_to_env and key:
            env_value = self._get_env_fallback(path, key)
            if env_value is not None:
                self._set_in_cache(cache_key, env_value, path, key)
                return env_value

        raise VaultSecretNotFoundError(
            f"Secret not found: {path}/{key or ''}" +
            (" (Vault unavailable and no env fallback)" if not self._vault_available else ""),
            vault_path=path,
            error_code="NOT_FOUND"
        )

    def get_twelvedata_key(self, index: int, group: Optional[str] = None) -> str:
        """
        Get a TwelveData API key by index.

        Args:
            index: Key index (1-8)
            group: Optional group identifier ('g1', 'g2', or None for legacy)

        Returns:
            TwelveData API key

        Raises:
            VaultSecretNotFoundError: If key not found
            ValueError: If index out of range
        """
        if not 1 <= index <= 8:
            raise ValueError(f"TwelveData key index must be 1-8, got {index}")

        if group:
            path = f"trading/twelvedata/{group.lower()}"
            key = f"api_key_{index}"
        else:
            path = "trading/twelvedata"
            key = f"api_key_{index}"

        return self.get_secret(path, key)

    def get_all_twelvedata_keys(self) -> Dict[str, str]:
        """
        Get all available TwelveData API keys.

        Returns:
            Dictionary mapping key identifiers to their values
        """
        keys = {}

        # Try to get keys from different sources
        sources = [
            ("trading/twelvedata", range(1, 9), ""),
            ("trading/twelvedata/g1", range(1, 9), "g1_"),
            ("trading/twelvedata/g2", range(1, 9), "g2_"),
        ]

        for path, indices, prefix in sources:
            for i in indices:
                key_name = f"api_key_{i}"
                try:
                    value = self.get_secret(path, key_name)
                    if value and not value.startswith("YOUR_"):
                        keys[f"{prefix}{key_name}"] = value
                except VaultSecretNotFoundError:
                    continue
                except Exception as e:
                    logger.debug(f"Error getting {path}/{key_name}: {e}")

        return keys

    def get_database_password(self) -> str:
        """
        Get the database password.

        Returns:
            Database password string
        """
        return self.get_secret("trading/database", "password")

    def get_database_config(self) -> Dict[str, Any]:
        """
        Get full database configuration.

        Returns:
            Dictionary with database configuration (user, password, host, port, database)
        """
        config_keys = ["user", "password", "host", "port", "database"]
        config = {}

        for key in config_keys:
            try:
                config[key] = self.get_secret("trading/database", key)
            except VaultSecretNotFoundError:
                # Use defaults for non-password fields
                defaults = {
                    "user": "trading_user",
                    "host": "localhost",
                    "port": "5432",
                    "database": "usdcop_trading",
                }
                if key in defaults:
                    config[key] = defaults[key]
                else:
                    raise

        # Ensure port is integer
        if "port" in config:
            config["port"] = int(config["port"])

        return config

    def get_database_url(self, dialect: str = "postgresql") -> str:
        """
        Get database connection URL.

        Args:
            dialect: Database dialect (postgresql, postgresql+asyncpg, etc.)

        Returns:
            Database connection URL string
        """
        config = self.get_database_config()
        return (
            f"{dialect}://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}"
        )

    def get_redis_password(self) -> str:
        """Get Redis password."""
        return self.get_secret("trading/redis", "password")

    def get_redis_config(self) -> Dict[str, Any]:
        """
        Get full Redis configuration.

        Returns:
            Dictionary with Redis configuration (host, port, password)
        """
        config = {}
        for key in ["host", "port", "password"]:
            try:
                config[key] = self.get_secret("trading/redis", key)
            except VaultSecretNotFoundError:
                defaults = {"host": "localhost", "port": "6379"}
                if key in defaults:
                    config[key] = defaults[key]
                else:
                    raise

        if "port" in config:
            config["port"] = int(config["port"])

        return config

    def get_jwt_secret(self) -> str:
        """Get JWT signing secret."""
        return self.get_secret("trading/jwt", "secret")

    def get_minio_credentials(self) -> Tuple[str, str]:
        """
        Get MinIO credentials.

        Returns:
            Tuple of (access_key, secret_key)
        """
        access_key = self.get_secret("trading/minio", "access_key")
        secret_key = self.get_secret("trading/minio", "secret_key")
        return access_key, secret_key

    def get_airflow_credentials(self) -> Dict[str, str]:
        """
        Get Airflow configuration secrets.

        Returns:
            Dictionary with fernet_key, secret_key, user, password
        """
        config = {}
        for key in ["fernet_key", "secret_key", "user", "password"]:
            try:
                config[key] = self.get_secret("airflow", key)
            except VaultSecretNotFoundError:
                if key in ["user"]:
                    config[key] = "admin"
                else:
                    raise
        return config

    def get_llm_api_key(self, provider: str) -> str:
        """
        Get LLM provider API key.

        Args:
            provider: Provider name (deepseek, anthropic, openai)

        Returns:
            API key string
        """
        return self.get_secret(f"trading/llm/{provider.lower()}", "api_key")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Vault connection.

        Returns:
            Dictionary with health status information
        """
        health = {
            "vault_available": self._vault_available,
            "vault_addr": self.config.vault_addr,
            "auth_method": self.config.auth_method,
            "fallback_enabled": self.config.fallback_to_env,
            "cache_size": len(self._cache),
            "cache_ttl": self.config.cache_ttl,
        }

        if self._vault_available and self._client:
            try:
                # Check Vault health
                vault_health = self._client.sys.read_health_status(method="GET")
                health["vault_initialized"] = vault_health.get("initialized", False)
                health["vault_sealed"] = vault_health.get("sealed", True)
                health["vault_version"] = vault_health.get("version", "unknown")

                # Check authentication status
                health["authenticated"] = self._client.is_authenticated()

                # Token TTL
                if self._token_expires_at:
                    health["token_ttl_remaining"] = max(0, int(self._token_expires_at - time.time()))
                else:
                    health["token_ttl_remaining"] = None

            except Exception as e:
                health["health_check_error"] = str(e)
        else:
            health["vault_status"] = "unavailable"

        return health

    def clear_cache(self, path: Optional[str] = None) -> int:
        """
        Clear cached secrets.

        Args:
            path: If provided, only clear secrets matching this path prefix.
                  If None, clear all cached secrets.

        Returns:
            Number of cache entries cleared
        """
        with self._cache_lock:
            if path is None:
                count = len(self._cache)
                self._cache.clear()
            else:
                keys_to_remove = [k for k in self._cache.keys() if k.startswith(path)]
                count = len(keys_to_remove)
                for k in keys_to_remove:
                    del self._cache[k]

        logger.info(f"Cleared {count} cached secrets")
        return count

    def close(self) -> None:
        """Clean up resources and cancel token renewal timer."""
        if self._renewal_timer:
            self._renewal_timer.cancel()
            self._renewal_timer = None

        self.clear_cache()
        logger.info("VaultClient closed")

    def __enter__(self) -> "VaultClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def get_vault_client(config: Optional[VaultConfig] = None) -> VaultClient:
    """
    Get the global VaultClient singleton instance.

    Args:
        config: Optional VaultConfig (only used on first call)

    Returns:
        VaultClient singleton instance
    """
    return VaultClient(config)


def require_secret(path: str, key: str):
    """
    Decorator to inject a secret into a function parameter.

    Usage:
        @require_secret("trading/database", "password")
        def connect_db(db_password: str):
            ...

    Args:
        path: Secret path in Vault
        key: Key within the secret
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get secret value
            client = get_vault_client()
            secret_value = client.get_secret(path, key)

            # Find the parameter name from function signature
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # Inject secret as first parameter if not already provided
            if len(args) < len(params):
                args = (secret_value,) + args
            elif params[0] not in kwargs:
                kwargs[params[0]] = secret_value

            return func(*args, **kwargs)
        return wrapper
    return decorator
