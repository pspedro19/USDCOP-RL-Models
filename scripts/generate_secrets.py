#!/usr/bin/env python3
"""
Generate secret files for USDCOP Trading System.

This script creates secure secret files in the secrets/ directory
based on your .env file or generates new random secrets.

Usage:
    python scripts/generate_secrets.py           # Generate from .env
    python scripts/generate_secrets.py --random  # Generate random secrets
    python scripts/generate_secrets.py --help    # Show help

SECURITY: This script should only be run locally during initial setup.
"""

import argparse
import os
import secrets
import string
import sys
from pathlib import Path


# Map of secret file names to their environment variable names
SECRET_MAPPINGS = {
    "db_password.txt": "POSTGRES_PASSWORD",
    "redis_password.txt": "REDIS_PASSWORD",
    "minio_secret_key.txt": "MINIO_SECRET_KEY",
    "airflow_password.txt": "AIRFLOW_PASSWORD",
    "airflow_fernet_key.txt": "AIRFLOW_FERNET_KEY",
    "airflow_secret_key.txt": "AIRFLOW_SECRET_KEY",
    "grafana_password.txt": "GRAFANA_PASSWORD",
    "pgadmin_password.txt": "PGADMIN_PASSWORD",
}


def generate_random_password(length: int = 32) -> str:
    """Generate a cryptographically secure random password."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_fernet_key() -> str:
    """Generate a Fernet encryption key for Airflow."""
    try:
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()
    except ImportError:
        # Fallback: base64-encoded 32 bytes
        import base64
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()


def load_env_file(env_path: Path) -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    if not env_path.exists():
        return env_vars

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value

    return env_vars


def create_secret_file(secrets_dir: Path, filename: str, value: str) -> bool:
    """Create a secret file with the given value."""
    filepath = secrets_dir / filename

    # Don't overwrite existing files
    if filepath.exists():
        print(f"  [SKIP] {filename} already exists")
        return False

    # Write the secret
    with open(filepath, 'w') as f:
        f.write(value)

    # Set restrictive permissions on Unix
    try:
        os.chmod(filepath, 0o600)
    except (OSError, AttributeError):
        pass  # Windows doesn't support chmod the same way

    print(f"  [CREATE] {filename}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate secret files for USDCOP Trading System"
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help="Generate random secrets instead of reading from .env"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Overwrite existing secret files"
    )
    parser.add_argument(
        '--env-file',
        type=Path,
        default=None,
        help="Path to .env file (default: .env in project root)"
    )
    args = parser.parse_args()

    # Find project root and secrets directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    secrets_dir = project_root / "secrets"
    env_path = args.env_file or project_root / ".env"

    print("=" * 60)
    print("USDCOP Trading System - Secret File Generator")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Secrets directory: {secrets_dir}")
    print(f"Mode: {'Random generation' if args.random else 'From .env file'}")
    print()

    # Ensure secrets directory exists
    secrets_dir.mkdir(exist_ok=True)

    # Load environment variables if not using random mode
    env_vars = {}
    if not args.random:
        env_vars = load_env_file(env_path)
        if not env_vars:
            print(f"[WARNING] No .env file found at {env_path}")
            print("[WARNING] Using random generation as fallback")
            args.random = True

    # Generate secrets
    print("Generating secret files:")
    created_count = 0

    for filename, env_var in SECRET_MAPPINGS.items():
        filepath = secrets_dir / filename

        # Check if file already exists
        if filepath.exists() and not args.force:
            print(f"  [SKIP] {filename} already exists")
            continue

        # Get or generate value
        if args.random:
            if "fernet" in filename.lower():
                value = generate_fernet_key()
            else:
                value = generate_random_password()
        else:
            value = env_vars.get(env_var, "")
            # Skip placeholder values
            if not value or "CHANGE_ME" in value or "YOUR_" in value:
                value = generate_random_password()
                print(f"  [GENERATE] {filename} (placeholder detected, generating random)")

        # Create the file
        if args.force and filepath.exists():
            filepath.unlink()
            print(f"  [OVERWRITE] {filename}")

        if create_secret_file(secrets_dir, filename, value):
            created_count += 1

    print()
    print("=" * 60)
    print(f"Complete! Created {created_count} secret files.")
    print()
    print("Next steps:")
    print("1. Review the generated secrets in secrets/")
    print("2. Backup the secrets securely (offline storage)")
    print("3. Restart Docker services: docker-compose down && docker-compose up -d")
    print()
    print("SECURITY REMINDERS:")
    print("- Never commit secret files to git")
    print("- The secrets/ directory is protected by .gitignore")
    print("- Store backups in a secure location")
    print("=" * 60)


if __name__ == "__main__":
    main()
