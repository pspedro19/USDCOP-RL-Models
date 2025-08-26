"""
Database package for USDCOP Trading RL System

This package contains database schemas, initialization scripts, and database utilities.
"""

from pathlib import Path

# Database schema files
DATABASE_SCHEMA_DIR = Path(__file__).parent

# SQL files (PostgreSQL versions - for reference)
INIT_DATABASES_SQL = DATABASE_SCHEMA_DIR / "init-databases.sql"
TRADING_SCHEMA_SQL = DATABASE_SCHEMA_DIR / "trading-schema.sql"

# SQLite versions (currently used)
INIT_DATABASES_SQLITE = DATABASE_SCHEMA_DIR / "init-databases-sqlite.sql"
TRADING_SCHEMA_SQLITE = DATABASE_SCHEMA_DIR / "trading-schema-sqlite.sql"

__all__ = [
    "DATABASE_SCHEMA_DIR",
    "INIT_DATABASES_SQL", 
    "TRADING_SCHEMA_SQL",
    "INIT_DATABASES_SQLITE",
    "TRADING_SCHEMA_SQLITE"
]
