#!/usr/bin/env python3
"""
Database Table Verification Script (Legacy)
============================================

DEPRECATED: Use db_migrate.py instead for full migration support.

This script is kept for backwards compatibility only.
All functionality has been moved to db_migrate.py.

Usage:
    python scripts/db_migrate.py              # Run migrations
    python scripts/db_migrate.py --validate   # Validate tables
    python scripts/db_migrate.py --status     # Show status
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ensure_tables():
    """Legacy function - delegates to db_migrate.validate_tables()."""
    logger.warning("ensure_tables() is deprecated. Use db_migrate.validate_tables() instead.")
    try:
        from scripts.db_migrate import validate_tables
        return await validate_tables()
    except ImportError:
        # Fallback for when running outside project root
        import asyncio
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from db_migrate import validate_tables
        return await validate_tables()


def main():
    """Legacy CLI - redirects to db_migrate.py."""
    logger.warning("=" * 60)
    logger.warning("ensure_db_tables.py is DEPRECATED")
    logger.warning("Use: python scripts/db_migrate.py")
    logger.warning("=" * 60)

    try:
        from scripts.db_migrate import main as migrate_main
        migrate_main()
    except ImportError:
        import asyncio
        result = asyncio.run(ensure_tables())
        sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
