#!/usr/bin/env python3
"""
API Key Generation Script
=========================

Generates API keys for the USDCOP Inference Service and optionally
stores them in the database.

Usage:
    # Generate key and display (no database)
    python scripts/generate_api_key.py

    # Generate and store in database
    python scripts/generate_api_key.py --name "Trading Bot" --user-id "bot_001"

    # Generate with custom options
    python scripts/generate_api_key.py --name "Admin Key" --user-id "admin" --roles admin,trader --rate-limit 1000

    # List existing keys
    python scripts/generate_api_key.py --list

    # Deactivate a key
    python scripts/generate_api_key.py --deactivate "usdcop_AbC1..."

Contract: CTR-AUTH-001

Author: Trading Team / Claude Code
Date: 2026-01-16
"""

import argparse
import asyncio
import hashlib
import os
import secrets
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "services"))

# Try to import from middleware
try:
    from inference_api.middleware.auth import generate_api_key, get_key_prefix
except ImportError:
    # Fallback implementation if import fails
    def generate_api_key(prefix: str = "usdcop") -> Tuple[str, str]:
        """Generate API key and hash."""
        random_part = secrets.token_urlsafe(32)
        key = f"{prefix}_{random_part}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash

    def get_key_prefix(api_key: str, length: int = 12) -> str:
        """Get display prefix of key."""
        return api_key[:length] + "..." if len(api_key) > length else api_key


# Database configuration
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "usdcop_trading")
DB_USER = os.environ.get("POSTGRES_USER", "")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")


def get_database_url() -> str:
    """Build database URL from environment."""
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def get_db_connection():
    """Get async database connection."""
    try:
        import asyncpg
        return await asyncpg.connect(get_database_url())
    except ImportError:
        print("Error: asyncpg not installed. Install with: pip install asyncpg")
        return None
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


async def store_api_key(
    key_hash: str,
    key_prefix: str,
    name: str,
    user_id: str,
    roles: list,
    rate_limit: int,
    description: Optional[str] = None,
    expires_days: Optional[int] = None,
) -> bool:
    """Store API key in database."""
    conn = await get_db_connection()
    if not conn:
        return False

    try:
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        await conn.execute(
            """
            INSERT INTO api_keys (
                key_hash, key_prefix, name, user_id, roles,
                rate_limit_per_minute, description, expires_at, created_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            key_hash,
            key_prefix,
            name,
            user_id,
            roles,
            rate_limit,
            description,
            expires_at,
            os.environ.get("USER", "script"),
        )
        return True
    except Exception as e:
        print(f"Error storing key: {e}")
        return False
    finally:
        await conn.close()


async def list_api_keys() -> None:
    """List all API keys from database."""
    conn = await get_db_connection()
    if not conn:
        return

    try:
        rows = await conn.fetch(
            """
            SELECT id, key_prefix, name, user_id, roles, rate_limit_per_minute,
                   is_active, expires_at, created_at, last_used_at, use_count
            FROM api_keys
            ORDER BY created_at DESC
            """
        )

        if not rows:
            print("No API keys found.")
            return

        print("\n" + "=" * 100)
        print("API KEYS")
        print("=" * 100)
        print(f"{'ID':<5} {'Prefix':<16} {'Name':<20} {'User':<12} {'Status':<10} {'Rate':<6} {'Uses':<8} {'Last Used'}")
        print("-" * 100)

        for row in rows:
            status = "ACTIVE" if row['is_active'] else "INACTIVE"
            if row['expires_at'] and row['expires_at'] < datetime.utcnow():
                status = "EXPIRED"

            last_used = row['last_used_at'].strftime("%Y-%m-%d %H:%M") if row['last_used_at'] else "Never"

            print(
                f"{row['id']:<5} "
                f"{row['key_prefix']:<16} "
                f"{row['name'][:20]:<20} "
                f"{row['user_id'] or 'N/A':<12} "
                f"{status:<10} "
                f"{row['rate_limit_per_minute']:<6} "
                f"{row['use_count']:<8} "
                f"{last_used}"
            )

        print("=" * 100)
        print(f"Total: {len(rows)} keys")

    except Exception as e:
        print(f"Error listing keys: {e}")
    finally:
        await conn.close()


async def deactivate_api_key(key_or_prefix: str) -> bool:
    """Deactivate an API key by prefix or full key."""
    conn = await get_db_connection()
    if not conn:
        return False

    try:
        # If full key provided, hash it
        if key_or_prefix.startswith("usdcop_") and len(key_or_prefix) > 20:
            key_hash = hashlib.sha256(key_or_prefix.encode()).hexdigest()
            result = await conn.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE key_hash = $1",
                key_hash
            )
        else:
            # Search by prefix
            result = await conn.execute(
                "UPDATE api_keys SET is_active = FALSE WHERE key_prefix LIKE $1",
                f"{key_or_prefix}%"
            )

        affected = int(result.split()[-1]) if result else 0
        if affected > 0:
            print(f"Deactivated {affected} key(s)")
            return True
        else:
            print("No matching keys found")
            return False

    except Exception as e:
        print(f"Error deactivating key: {e}")
        return False
    finally:
        await conn.close()


def generate_key_local(args) -> None:
    """Generate API key without database."""
    key, key_hash = generate_api_key()
    prefix = get_key_prefix(key)

    print("\n" + "=" * 70)
    print("NEW API KEY GENERATED")
    print("=" * 70)
    print(f"\nAPI Key (SAVE THIS - shown only once!):")
    print(f"  {key}")
    print(f"\nKey Hash (stored in database):")
    print(f"  {key_hash}")
    print(f"\nKey Prefix (for identification):")
    print(f"  {prefix}")

    if args.name:
        print(f"\nName: {args.name}")
    if args.user_id:
        print(f"User ID: {args.user_id}")
    if args.roles:
        print(f"Roles: {args.roles}")

    print("\n" + "-" * 70)
    print("To use this key, add the following header to your requests:")
    print(f"  X-API-Key: {key}")
    print("-" * 70)

    # If database credentials provided, offer to store
    if args.name and args.user_id and DB_USER and DB_PASSWORD:
        print("\nDatabase credentials detected. Use --store to save to database.")

    print("=" * 70 + "\n")


async def generate_and_store(args) -> None:
    """Generate API key and store in database."""
    key, key_hash = generate_api_key()
    prefix = get_key_prefix(key)

    roles = args.roles.split(",") if args.roles else ["trader"]

    print("\n" + "=" * 70)
    print("GENERATING AND STORING API KEY")
    print("=" * 70)

    success = await store_api_key(
        key_hash=key_hash,
        key_prefix=prefix,
        name=args.name,
        user_id=args.user_id,
        roles=roles,
        rate_limit=args.rate_limit,
        description=args.description,
        expires_days=args.expires_days,
    )

    if success:
        print(f"\nAPI Key stored successfully!")
        print(f"\nAPI Key (SAVE THIS - shown only once!):")
        print(f"  {key}")
        print(f"\nDetails:")
        print(f"  Name: {args.name}")
        print(f"  User ID: {args.user_id}")
        print(f"  Roles: {roles}")
        print(f"  Rate Limit: {args.rate_limit} req/min")
        if args.expires_days:
            print(f"  Expires: {args.expires_days} days")

        print("\n" + "-" * 70)
        print("To use this key, add the following header to your requests:")
        print(f"  X-API-Key: {key}")
        print("-" * 70)
    else:
        print("\nFailed to store API key in database.")
        print("Key generated (not stored):")
        print(f"  {key}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and manage API keys for USDCOP Inference Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate key (display only)
  python generate_api_key.py

  # Generate and store in database
  python generate_api_key.py --name "Trading Bot" --user-id "bot_001" --store

  # Generate with custom rate limit and roles
  python generate_api_key.py --name "Admin" --user-id "admin" --roles admin,trader --rate-limit 1000 --store

  # List all keys
  python generate_api_key.py --list

  # Deactivate a key
  python generate_api_key.py --deactivate "usdcop_AbC1..."
        """
    )

    # Generation options
    parser.add_argument(
        "--name", "-n",
        help="Human-readable name for the key (required for --store)"
    )
    parser.add_argument(
        "--user-id", "-u",
        help="User ID to associate with the key (required for --store)"
    )
    parser.add_argument(
        "--roles", "-r",
        default="trader",
        help="Comma-separated roles (default: trader). Options: trader,admin,readonly"
    )
    parser.add_argument(
        "--rate-limit", "-l",
        type=int,
        default=100,
        help="Rate limit in requests per minute (default: 100)"
    )
    parser.add_argument(
        "--description", "-d",
        help="Optional description for the key"
    )
    parser.add_argument(
        "--expires-days",
        type=int,
        help="Number of days until key expires (optional)"
    )
    parser.add_argument(
        "--store", "-s",
        action="store_true",
        help="Store the key in the database (requires --name and --user-id)"
    )

    # Management options
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all API keys from database"
    )
    parser.add_argument(
        "--deactivate",
        metavar="KEY",
        help="Deactivate an API key by prefix or full key"
    )

    args = parser.parse_args()

    # Handle list/deactivate commands
    if args.list:
        asyncio.run(list_api_keys())
        return 0

    if args.deactivate:
        success = asyncio.run(deactivate_api_key(args.deactivate))
        return 0 if success else 1

    # Handle generation
    if args.store:
        if not args.name or not args.user_id:
            print("Error: --name and --user-id required when using --store")
            return 1

        if not DB_USER or not DB_PASSWORD:
            print("Error: Database credentials not configured.")
            print("Set POSTGRES_USER and POSTGRES_PASSWORD environment variables.")
            return 1

        asyncio.run(generate_and_store(args))
    else:
        generate_key_local(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
