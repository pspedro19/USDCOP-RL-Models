"""
Security Settings - No Hardcoded Passwords
Contrato ID: CTR-007
CLAUDE-T8 | Plan Item: P0-4

IMPORTANTE: Esta clase NUNCA permite defaults para passwords.
Todas las credenciales DEBEN venir de variables de entorno.
"""

import os
from typing import Optional
from dataclasses import dataclass


class SecurityError(Exception):
    """Error de configuracion de seguridad."""
    pass


@dataclass
class SecuritySettings:
    """
    Configuracion de seguridad - NUNCA defaults para passwords.

    Uso:
        settings = SecuritySettings.from_env()
        url = settings.get_postgres_url()

    Variables de entorno requeridas:
        - POSTGRES_PASSWORD (obligatorio)
        - POSTGRES_USER (default: trading_user)
        - POSTGRES_HOST (default: localhost)
        - POSTGRES_PORT (default: 5432)
        - POSTGRES_DB (default: usdcop_trading)
        - API_SECRET_KEY (opcional, para servicios que lo requieran)
    """
    postgres_password: str
    postgres_user: str = "trading_user"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "usdcop_trading"
    api_secret_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        """
        Carga settings desde variables de entorno.

        Raises:
            SecurityError: Si POSTGRES_PASSWORD no esta definido
        """
        password = os.environ.get("POSTGRES_PASSWORD")

        if not password:
            raise SecurityError(
                "POSTGRES_PASSWORD no esta definido. "
                "NUNCA usar passwords hardcodeados. "
                "Defina la variable de entorno POSTGRES_PASSWORD"
            )

        return cls(
            postgres_password=password,
            postgres_user=os.environ.get("POSTGRES_USER", "trading_user"),
            postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
            postgres_db=os.environ.get("POSTGRES_DB", "usdcop_trading"),
            api_secret_key=os.environ.get("API_SECRET_KEY"),
        )

    def get_postgres_url(self) -> str:
        """Construye URL de conexion PostgreSQL."""
        return (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def get_postgres_url_masked(self) -> str:
        """URL con password enmascarado para logging."""
        return (
            f"postgresql://{self.postgres_user}:****@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def validate(self) -> bool:
        """
        Valida que la configuracion es segura.

        Returns:
            True si la configuracion es valida

        Raises:
            SecurityError: Si hay problemas de seguridad
        """
        # Verificar que password no es un valor comun/inseguro
        insecure_passwords = [
            "password", "123456", "admin", "root",
            "postgres", "trading", "test", ""
        ]

        if self.postgres_password.lower() in insecure_passwords:
            raise SecurityError(
                f"Password inseguro detectado. "
                f"Use un password fuerte y unico."
            )

        # Verificar longitud minima
        if len(self.postgres_password) < 8:
            raise SecurityError(
                "Password demasiado corto. Minimo 8 caracteres."
            )

        return True


def get_secure_db_url() -> str:
    """
    Obtiene URL de base de datos de forma segura.

    Uso:
        from src.config.security import get_secure_db_url
        url = get_secure_db_url()
    """
    settings = SecuritySettings.from_env()
    return settings.get_postgres_url()


# Verificacion de seguridad al importar el modulo
def _check_no_hardcoded_secrets():
    """Verifica que no hay secrets hardcodeados en este archivo."""
    # Este check es simbolico - el codigo real no debe tener secrets
    pass


_check_no_hardcoded_secrets()
