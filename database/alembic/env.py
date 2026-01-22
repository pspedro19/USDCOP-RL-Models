"""
Alembic Environment Configuration for USD/COP RL Trading System.

This module configures Alembic migrations to work with SQLAlchemy ORM models.
It supports both online (connected to database) and offline (SQL generation) modes.

Environment Variables:
    DATABASE_URL: PostgreSQL connection string
        Example: postgresql://user:pass@localhost:5432/usdcop_trading

Usage:
    # Run migrations
    alembic upgrade head

    # Generate new migration
    alembic revision --autogenerate -m "description"

    # Generate SQL without applying
    alembic upgrade head --sql > migration.sql
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Add project root to Python path for imports
# This allows importing src.models.orm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the ORM models and Base
from src.models.orm import Base

# Alembic Config object - provides access to .ini file values
config = context.config

# Setup Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata


def get_database_url() -> str:
    """
    Get database URL from environment or config.

    Priority:
    1. DATABASE_URL environment variable
    2. sqlalchemy.url from alembic.ini

    Returns:
        PostgreSQL connection string

    Raises:
        ValueError: If no database URL is configured
    """
    # First, try environment variable
    url = os.environ.get("DATABASE_URL")

    if url:
        # Handle Heroku-style postgres:// URLs (convert to postgresql://)
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    # Fall back to alembic.ini config
    url = config.get_main_option("sqlalchemy.url")

    if not url or url == "postgresql://user:password@localhost:5432/usdcop_trading":
        raise ValueError(
            "DATABASE_URL environment variable not set and no valid "
            "sqlalchemy.url configured in alembic.ini. "
            "Please set DATABASE_URL or update alembic.ini."
        )

    return url


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This generates SQL statements without connecting to the database,
    useful for review or deployment to restricted environments.

    Configures the context with just a URL and not an Engine.
    Calls to context.execute() emit the given string to the script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an Engine and connects to the database to run migrations.
    This is the standard mode for development and production deployments.
    """
    url = get_database_url()

    # Create engine with connection pooling disabled for migrations
    # (we only need a single connection)
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
        # Ensure proper handling of schema changes
        isolation_level="AUTOCOMMIT",
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Enable autogenerate comparison features
            compare_type=True,
            compare_server_default=True,
            # Include object names in autogenerate
            include_object=include_object,
            # Render Python types in migrations
            render_as_batch=True,  # Better support for SQLite in dev
        )

        with context.begin_transaction():
            context.run_migrations()


def include_object(
    object_: object,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: object | None,
) -> bool:
    """
    Filter function to control which database objects are included in migrations.

    This prevents alembic from trying to manage tables that are created
    by other systems (e.g., TimescaleDB hypertables, Airflow metadata).

    Args:
        object_: The database object being considered
        name: Object name
        type_: Object type (table, column, index, etc.)
        reflected: Whether object was reflected from database
        compare_to: Object being compared against (for modifications)

    Returns:
        True if the object should be included in migrations
    """
    # Skip TimescaleDB internal tables
    if name and name.startswith("_timescaledb"):
        return False

    # Skip tables in schemas we don't manage
    if type_ == "table":
        # Only manage tables in public schema that match our models
        # Skip dw schema (managed by separate SQL migrations)
        if hasattr(object_, "schema") and object_.schema == "dw":
            return False

    # Skip specific tables managed by other systems
    excluded_tables = {
        "alembic_version",  # Managed by alembic itself
        "macro_indicators_daily",  # Managed by SQL migrations
        "usdcop_m5_ohlcv",  # TimescaleDB hypertable
        "api_keys",  # Managed by SQL migrations
        "api_key_usage_log",  # Managed by SQL migrations
        "api_rate_limit_state",  # Managed by SQL migrations
    }

    if type_ == "table" and name in excluded_tables:
        return False

    return True


# Determine which mode to run
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
