"""
Database connection module for PostgreSQL.

Provides database connection utilities for the FastAPI application,
including connection pooling and helper functions for user authentication.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# =============================================================================
# Database Configuration
# =============================================================================

# Read from environment with fallback to DATABASE_URL or individual components
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Parse DATABASE_URL format: postgresql://user:password@host:port/database
    url = DATABASE_URL.replace("postgresql://", "")
    user_pass, host_db = url.split("@")
    DB_USER, DB_PASSWORD = user_pass.split(":")
    host_port, DB_NAME = host_db.split("/")
    DB_HOST, DB_PORT = host_port.split(":") if ":" in host_port else (host_port, "5432")
else:
    # Use individual environment variables
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "pipeline_db")
    DB_USER = os.getenv("DB_USER", "pipeline")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "pipeline_secret_change_me")

# Connection pool settings
POOL_MIN_CONNECTIONS = int(os.getenv("DB_POOL_MIN", "1"))
POOL_MAX_CONNECTIONS = int(os.getenv("DB_POOL_MAX", "10"))

# =============================================================================
# Connection Pool
# =============================================================================

_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def get_connection_pool() -> pool.ThreadedConnectionPool:
    """
    Get or create the database connection pool.

    Returns:
        ThreadedConnectionPool instance for database connections.
    """
    global _connection_pool

    if _connection_pool is None:
        try:
            _connection_pool = pool.ThreadedConnectionPool(
                POOL_MIN_CONNECTIONS,
                POOL_MAX_CONNECTIONS,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            logger.info(f"Database connection pool created: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            raise

    return _connection_pool


def close_connection_pool():
    """Close the database connection pool."""
    global _connection_pool

    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Database connection pool closed")


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.

    Yields a connection from the pool and returns it when done.
    Handles exceptions and connection cleanup.

    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
    """
    pool = get_connection_pool()
    conn = None

    try:
        conn = pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            pool.putconn(conn)


@contextmanager
def get_db_cursor(cursor_factory=RealDictCursor):
    """
    Context manager for database cursors.

    Yields a cursor and handles connection management.
    Uses RealDictCursor by default for dictionary-like row access.

    Usage:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM auth.users WHERE username = %s", (username,))
            user = cursor.fetchone()
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()


# =============================================================================
# User Database Functions
# =============================================================================

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a user from the database by username.

    Args:
        username: The username to look up.

    Returns:
        Dictionary with user data if found, None otherwise.
        Keys: id, username, password_hash, role, is_active, created_at, updated_at
    """
    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, username, password_hash, role, is_active, created_at, updated_at
                FROM auth.users
                WHERE username = %s
                """,
                (username,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    except Exception as e:
        logger.error(f"Error fetching user '{username}': {e}")
        return None


def verify_user_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Verify user credentials against the database.

    This function fetches the user by username and verifies the password
    using bcrypt comparison. Returns user data if valid, None otherwise.

    Args:
        username: The username to authenticate.
        password: The plain-text password to verify.

    Returns:
        Dictionary with user data (without password_hash) if valid, None otherwise.
    """
    import bcrypt

    user = get_user_by_username(username)

    if not user:
        return None

    if not user.get("is_active", False):
        return None

    # Verify password using bcrypt directly
    try:
        password_bytes = password.encode('utf-8')
        hash_bytes = user["password_hash"].encode('utf-8')
        if not bcrypt.checkpw(password_bytes, hash_bytes):
            return None
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return None

    # Return user data without the password hash
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "is_active": user["is_active"],
    }


def create_user(username: str, password: str, role: str = "viewer") -> Optional[Dict[str, Any]]:
    """
    Create a new user in the database.

    Args:
        username: The username for the new user.
        password: The plain-text password (will be hashed).
        role: The user's role (default: 'viewer').

    Returns:
        Dictionary with new user data if created, None if failed.
    """
    import bcrypt

    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO auth.users (username, password_hash, role, is_active)
                VALUES (%s, %s, %s, true)
                RETURNING id, username, role, is_active, created_at
                """,
                (username, password_hash, role)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    except Exception as e:
        logger.error(f"Error creating user '{username}': {e}")
        return None


def update_user_password(username: str, new_password: str) -> bool:
    """
    Update a user's password.

    Args:
        username: The username of the user to update.
        new_password: The new plain-text password (will be hashed).

    Returns:
        True if updated successfully, False otherwise.
    """
    import bcrypt

    password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

    try:
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                UPDATE auth.users
                SET password_hash = %s, updated_at = CURRENT_TIMESTAMP
                WHERE username = %s
                """,
                (password_hash, username)
            )
            return cursor.rowcount > 0

    except Exception as e:
        logger.error(f"Error updating password for user '{username}': {e}")
        return False
