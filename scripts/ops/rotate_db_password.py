#!/usr/bin/env python3
"""
Database Password Rotation Script
==================================

Safely rotates database passwords with zero-downtime.
Updates Vault, database, and reloads application connections.

P2: Database Password Rotation

Usage:
    # Rotate password (generates new random password)
    python scripts/rotate_db_password.py

    # Rotate with specific password
    python scripts/rotate_db_password.py --password "new-secure-password"

    # Dry run (show what would be done)
    python scripts/rotate_db_password.py --dry-run

    # Rotate and update .env file
    python scripts/rotate_db_password.py --update-env

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import argparse
import logging
import os
import secrets
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import optional dependencies
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import hvac
    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class PasswordRotationConfig:
    """Configuration for password rotation."""

    def __init__(self):
        # Database connection (uses current password)
        self.db_host = os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.db_name = os.getenv("POSTGRES_DB", "usdcop")
        self.db_user = os.getenv("POSTGRES_USER", "admin")
        self.db_password = os.getenv("POSTGRES_PASSWORD", "")

        # Vault configuration (optional)
        self.vault_addr = os.getenv("VAULT_ADDR", "http://localhost:8200")
        self.vault_token = os.getenv("VAULT_TOKEN", "")
        self.vault_secret_path = "secret/data/database/postgres"

        # Docker configuration
        self.docker_compose_file = "docker-compose.yml"
        self.services_to_reload = [
            "trading-api",
            "analytics-api",
            "backtest-api",
            "mlops-inference-api",
            "airflow-scheduler",
            "airflow-webserver",
        ]

        # .env file path
        self.env_file = ".env"

        # Password requirements
        self.password_length = 32
        self.password_chars = string.ascii_letters + string.digits + "!@#$%^&*"


# =============================================================================
# Password Generation
# =============================================================================

def generate_secure_password(
    length: int = 32,
    chars: str = string.ascii_letters + string.digits + "!@#$%^&*"
) -> str:
    """
    Generate a cryptographically secure random password.

    Args:
        length: Password length (minimum 16)
        chars: Character set to use

    Returns:
        Secure random password
    """
    if length < 16:
        raise ValueError("Password length must be at least 16 characters")

    # Ensure password contains at least one of each required type
    password = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*"),
    ]

    # Fill the rest with random characters
    password.extend(secrets.choice(chars) for _ in range(length - 4))

    # Shuffle to avoid predictable positions
    random_list = list(password)
    secrets.SystemRandom().shuffle(random_list)

    return "".join(random_list)


# =============================================================================
# Database Operations
# =============================================================================

def test_database_connection(config: PasswordRotationConfig, password: str) -> bool:
    """
    Test database connection with given password.

    Args:
        config: Rotation configuration
        password: Password to test

    Returns:
        True if connection successful
    """
    if not PSYCOPG2_AVAILABLE:
        logger.warning("psycopg2 not available, skipping connection test")
        return True

    try:
        conn = psycopg2.connect(
            host=config.db_host,
            port=config.db_port,
            database=config.db_name,
            user=config.db_user,
            password=password,
            connect_timeout=5,
        )
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def update_database_password(
    config: PasswordRotationConfig,
    new_password: str,
) -> bool:
    """
    Update the database user password.

    Args:
        config: Rotation configuration
        new_password: New password to set

    Returns:
        True if successful
    """
    if not PSYCOPG2_AVAILABLE:
        logger.error("psycopg2 required for password update")
        return False

    try:
        conn = psycopg2.connect(
            host=config.db_host,
            port=config.db_port,
            database=config.db_name,
            user=config.db_user,
            password=config.db_password,
        )
        conn.autocommit = True

        with conn.cursor() as cur:
            # Use proper escaping for the password
            cur.execute(
                f"ALTER USER {config.db_user} WITH PASSWORD %s",
                (new_password,)
            )

        conn.close()
        logger.info(f"Database password updated for user: {config.db_user}")
        return True

    except Exception as e:
        logger.error(f"Failed to update database password: {e}")
        return False


# =============================================================================
# Vault Operations
# =============================================================================

def update_vault_secret(
    config: PasswordRotationConfig,
    new_password: str,
) -> bool:
    """
    Update password in HashiCorp Vault.

    Args:
        config: Rotation configuration
        new_password: New password to store

    Returns:
        True if successful
    """
    if not HVAC_AVAILABLE:
        logger.warning("hvac not available, skipping Vault update")
        return True

    if not config.vault_token:
        logger.warning("VAULT_TOKEN not set, skipping Vault update")
        return True

    try:
        client = hvac.Client(url=config.vault_addr, token=config.vault_token)

        if not client.is_authenticated():
            logger.error("Vault authentication failed")
            return False

        # Update secret
        client.secrets.kv.v2.create_or_update_secret(
            path=config.vault_secret_path.replace("secret/data/", ""),
            secret={
                "password": new_password,
                "username": config.db_user,
                "host": config.db_host,
                "port": str(config.db_port),
                "database": config.db_name,
                "rotated_at": datetime.utcnow().isoformat(),
            }
        )

        logger.info(f"Vault secret updated: {config.vault_secret_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to update Vault secret: {e}")
        return False


# =============================================================================
# Docker/Service Operations
# =============================================================================

def reload_docker_services(config: PasswordRotationConfig) -> bool:
    """
    Reload Docker services to pick up new password.

    Args:
        config: Rotation configuration

    Returns:
        True if successful
    """
    try:
        for service in config.services_to_reload:
            logger.info(f"Restarting service: {service}")
            result = subprocess.run(
                ["docker-compose", "-f", config.docker_compose_file,
                 "restart", service],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(f"Failed to restart {service}: {result.stderr}")

        logger.info("All services restarted")
        return True

    except Exception as e:
        logger.error(f"Failed to restart services: {e}")
        return False


def update_docker_secret(config: PasswordRotationConfig, new_password: str) -> bool:
    """
    Update Docker secret file.

    Args:
        config: Rotation configuration
        new_password: New password

    Returns:
        True if successful
    """
    secret_file = Path("secrets/db_password.txt")

    try:
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        secret_file.write_text(new_password)
        logger.info(f"Updated Docker secret: {secret_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to update Docker secret: {e}")
        return False


# =============================================================================
# Environment File Operations
# =============================================================================

def update_env_file(config: PasswordRotationConfig, new_password: str) -> bool:
    """
    Update password in .env file.

    Args:
        config: Rotation configuration
        new_password: New password

    Returns:
        True if successful
    """
    env_path = Path(config.env_file)

    if not env_path.exists():
        logger.warning(f".env file not found: {env_path}")
        return False

    try:
        # Read current content
        content = env_path.read_text()
        lines = content.split("\n")

        # Find and replace POSTGRES_PASSWORD line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("POSTGRES_PASSWORD="):
                lines[i] = f"POSTGRES_PASSWORD={new_password}"
                updated = True
                break

        if not updated:
            # Add if not found
            lines.append(f"POSTGRES_PASSWORD={new_password}")

        # Write back
        env_path.write_text("\n".join(lines))
        logger.info(f"Updated .env file: {env_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to update .env file: {e}")
        return False


# =============================================================================
# Main Rotation Logic
# =============================================================================

def rotate_password(
    config: PasswordRotationConfig,
    new_password: Optional[str] = None,
    dry_run: bool = False,
    update_env: bool = False,
) -> bool:
    """
    Perform complete password rotation.

    Args:
        config: Rotation configuration
        new_password: Specific password to use (generates if None)
        dry_run: If True, only show what would be done
        update_env: If True, update .env file

    Returns:
        True if rotation successful
    """
    # Generate new password if not provided
    if new_password is None:
        new_password = generate_secure_password(
            length=config.password_length,
            chars=config.password_chars,
        )

    logger.info("=" * 60)
    logger.info("DATABASE PASSWORD ROTATION")
    logger.info("=" * 60)
    logger.info(f"Database: {config.db_host}:{config.db_port}/{config.db_name}")
    logger.info(f"User: {config.db_user}")
    logger.info(f"New password length: {len(new_password)}")

    if dry_run:
        logger.info("\n[DRY RUN] Would perform the following actions:")
        logger.info("  1. Update database user password")
        logger.info("  2. Update Vault secret")
        logger.info("  3. Update Docker secret file")
        if update_env:
            logger.info("  4. Update .env file")
        logger.info("  5. Restart Docker services")
        logger.info("\n[DRY RUN] No changes made")
        return True

    # Step 1: Update database password
    logger.info("\nStep 1: Updating database password...")
    if not update_database_password(config, new_password):
        logger.error("Failed to update database password, aborting")
        return False

    # Step 2: Verify new password works
    logger.info("\nStep 2: Verifying new password...")
    if not test_database_connection(config, new_password):
        logger.error("New password verification failed!")
        return False

    # Step 3: Update Vault
    logger.info("\nStep 3: Updating Vault secret...")
    update_vault_secret(config, new_password)

    # Step 4: Update Docker secret
    logger.info("\nStep 4: Updating Docker secret file...")
    update_docker_secret(config, new_password)

    # Step 5: Update .env if requested
    if update_env:
        logger.info("\nStep 5: Updating .env file...")
        update_env_file(config, new_password)

    # Step 6: Restart services
    logger.info("\nStep 6: Restarting services...")
    reload_docker_services(config)

    logger.info("\n" + "=" * 60)
    logger.info("PASSWORD ROTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nNew password: {new_password[:4]}{'*' * (len(new_password) - 8)}{new_password[-4:]}")
    logger.info("\nIMPORTANT: Store this password securely!")

    return True


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rotate database password with zero-downtime"
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Specific password to use (generates random if not provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--update-env",
        action="store_true",
        help="Update .env file with new password",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=32,
        help="Password length (default: 32)",
    )

    args = parser.parse_args()

    config = PasswordRotationConfig()
    config.password_length = args.length

    if not config.db_password:
        logger.error("POSTGRES_PASSWORD environment variable not set")
        sys.exit(1)

    success = rotate_password(
        config=config,
        new_password=args.password,
        dry_run=args.dry_run,
        update_env=args.update_env,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
