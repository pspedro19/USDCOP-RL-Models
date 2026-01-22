"""
Database Module - Connection Management
=======================================
Provides database session management for the feature store and other modules.

This module provides a centralized way to manage database connections using
SQLAlchemy for ORM operations and raw psycopg2 for performance-critical queries.

Usage:
    from src.database import get_db_session

    with get_db_session() as session:
        result = session.execute(text("SELECT * FROM inference_features_5m"))

Author: Trading Team
Version: 1.0.0
Created: 2026-01-17
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)


# Database configuration from environment
DATABASE_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "database": os.environ.get("POSTGRES_DB", "trading"),
    "user": os.environ.get("POSTGRES_USER", "trading"),
    "password": os.environ.get("POSTGRES_PASSWORD", ""),
}


def get_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    return (
        f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
    )


# Lazy initialization of SQLAlchemy engine and session factory
_engine = None
_SessionFactory = None


def _get_engine():
    """Get or create SQLAlchemy engine (lazy initialization)."""
    global _engine
    if _engine is None:
        try:
            from sqlalchemy import create_engine
            connection_string = get_connection_string()
            _engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            logger.info("Database engine initialized")
        except ImportError:
            logger.warning("SQLAlchemy not installed - using psycopg2 fallback")
            raise
    return _engine


def _get_session_factory():
    """Get or create SQLAlchemy session factory (lazy initialization)."""
    global _SessionFactory
    if _SessionFactory is None:
        from sqlalchemy.orm import sessionmaker
        engine = _get_engine()
        _SessionFactory = sessionmaker(bind=engine)
    return _SessionFactory


@contextmanager
def get_db_session() -> Generator:
    """
    Context manager for database sessions.

    Provides a SQLAlchemy session that automatically commits on success
    and rolls back on exception.

    Usage:
        with get_db_session() as session:
            result = session.execute(text("SELECT * FROM table"))
            for row in result:
                print(row)

    Yields:
        SQLAlchemy Session object

    Raises:
        Exception: Re-raises any exception after rollback
    """
    try:
        SessionFactory = _get_session_factory()
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except ImportError:
        # Fallback to psycopg2 if SQLAlchemy not available
        logger.warning("SQLAlchemy not available, using psycopg2 fallback")
        with get_psycopg2_connection() as conn:
            yield conn


@contextmanager
def get_psycopg2_connection():
    """
    Context manager for raw psycopg2 connections.

    Use this for performance-critical queries or when SQLAlchemy overhead
    is not desired.

    Usage:
        with get_psycopg2_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM table")
            rows = cur.fetchall()

    Yields:
        psycopg2 connection object
    """
    import psycopg2

    conn = psycopg2.connect(
        host=DATABASE_CONFIG["host"],
        port=DATABASE_CONFIG["port"],
        database=DATABASE_CONFIG["database"],
        user=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"],
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


__all__ = [
    "get_db_session",
    "get_psycopg2_connection",
    "get_connection_string",
    "DATABASE_CONFIG",
]
