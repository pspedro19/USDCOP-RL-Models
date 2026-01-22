"""
Secrets Manager for Airflow DAGs
================================
P0-2: Secure credential access with Vault abstraction

Supports:
1. HashiCorp Vault (production)
2. Environment variables (development)
3. Airflow Connections (fallback)
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatabaseCredentials:
    host: str
    port: int
    database: str
    username: str
    password: str

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class APICredentials:
    api_key: str
    secret_key: Optional[str] = None
    base_url: Optional[str] = None


class SecretsManager:
    """
    Unified secrets access for Airflow DAGs.

    Priority:
    1. Vault (if VAULT_ADDR configured)
    2. Environment variables
    3. Airflow Connections
    """

    def __init__(self):
        self._vault_client = None
        self._vault_available = False
        self._init_vault()

    def _init_vault(self):
        """Initialize Vault client if available."""
        vault_addr = os.environ.get('VAULT_ADDR')
        vault_token = os.environ.get('VAULT_TOKEN')

        if vault_addr and vault_token:
            try:
                import hvac
                self._vault_client = hvac.Client(url=vault_addr, token=vault_token)
                if self._vault_client.is_authenticated():
                    self._vault_available = True
                    logger.info("Vault client initialized successfully")
                else:
                    logger.warning("Vault authentication failed, using fallback")
            except ImportError:
                logger.warning("hvac not installed, Vault unavailable")
            except Exception as e:
                logger.warning(f"Vault init failed: {e}")

    def get_database_credentials(self, db_name: str = "trading") -> DatabaseCredentials:
        """
        Get database credentials.

        Args:
            db_name: Name of database (trading, mlflow, etc.)
        """
        if self._vault_available:
            return self._get_db_from_vault(db_name)
        return self._get_db_from_env(db_name)

    def _get_db_from_vault(self, db_name: str) -> DatabaseCredentials:
        """Get database credentials from Vault."""
        secret_path = f"secret/data/databases/{db_name}"
        response = self._vault_client.secrets.kv.v2.read_secret_version(
            path=f"databases/{db_name}",
            mount_point="secret"
        )
        data = response['data']['data']

        return DatabaseCredentials(
            host=data['host'],
            port=int(data.get('port', 5432)),
            database=data['database'],
            username=data['username'],
            password=data['password'],
        )

    def _get_db_from_env(self, db_name: str) -> DatabaseCredentials:
        """Get database credentials from environment."""
        prefix = db_name.upper()

        # Try DATABASE_URL format first
        db_url = os.environ.get(f'{prefix}_DATABASE_URL') or os.environ.get('DATABASE_URL')
        if db_url:
            return self._parse_database_url(db_url)

        # Fallback to individual vars
        return DatabaseCredentials(
            host=os.environ.get(f'{prefix}_DB_HOST', 'localhost'),
            port=int(os.environ.get(f'{prefix}_DB_PORT', 5432)),
            database=os.environ.get(f'{prefix}_DB_NAME', db_name),
            username=os.environ.get(f'{prefix}_DB_USER', 'postgres'),
            password=os.environ.get(f'{prefix}_DB_PASSWORD', ''),
        )

    def _parse_database_url(self, url: str) -> DatabaseCredentials:
        """Parse DATABASE_URL format."""
        from urllib.parse import urlparse
        parsed = urlparse(url)

        return DatabaseCredentials(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 5432,
            database=parsed.path.lstrip('/'),
            username=parsed.username or 'postgres',
            password=parsed.password or '',
        )

    def get_api_credentials(self, service: str) -> APICredentials:
        """
        Get API credentials for external services.

        Args:
            service: Service name (twelvedata, slack, etc.)
        """
        if self._vault_available:
            return self._get_api_from_vault(service)
        return self._get_api_from_env(service)

    def _get_api_from_vault(self, service: str) -> APICredentials:
        """Get API credentials from Vault."""
        response = self._vault_client.secrets.kv.v2.read_secret_version(
            path=f"apis/{service}",
            mount_point="secret"
        )
        data = response['data']['data']

        return APICredentials(
            api_key=data['api_key'],
            secret_key=data.get('secret_key'),
            base_url=data.get('base_url'),
        )

    def _get_api_from_env(self, service: str) -> APICredentials:
        """Get API credentials from environment."""
        prefix = service.upper()

        return APICredentials(
            api_key=os.environ.get(f'{prefix}_API_KEY', ''),
            secret_key=os.environ.get(f'{prefix}_SECRET_KEY'),
            base_url=os.environ.get(f'{prefix}_BASE_URL'),
        )

    def get_mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI."""
        if self._vault_available:
            try:
                response = self._vault_client.secrets.kv.v2.read_secret_version(
                    path="mlflow/tracking",
                    mount_point="secret"
                )
                return response['data']['data']['tracking_uri']
            except Exception:
                pass

        return os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')


# Global singleton
_secrets_manager: Optional[SecretsManager] = None


def get_secrets() -> SecretsManager:
    """Get or create the global secrets manager."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions for common use cases
def get_trading_db() -> DatabaseCredentials:
    """Get trading database credentials."""
    return get_secrets().get_database_credentials("trading")


def get_twelvedata_key() -> str:
    """Get TwelveData API key."""
    return get_secrets().get_api_credentials("twelvedata").api_key


def get_slack_webhook() -> str:
    """Get Slack webhook URL."""
    return get_secrets().get_api_credentials("slack").api_key
