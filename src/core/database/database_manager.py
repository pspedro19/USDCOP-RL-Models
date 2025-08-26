"""
Database Manager for USDCOP Trading RL System

Handles database initialization, schema management, and database operations.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from . import INIT_DATABASES_SQLITE, TRADING_SCHEMA_SQLITE

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations and schema"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: Optional[sqlite3.Connection] = None
        
    def initialize_database(self) -> bool:
        """Initialize the database with schemas"""
        try:
            # Create connection
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            
            # Read and execute initialization SQL
            if INIT_DATABASES_SQLITE.exists():
                with open(INIT_DATABASES_SQLITE, 'r', encoding='utf-8') as f:
                    init_sql = f.read()
                self.connection.executescript(init_sql)
                logger.info("Database initialization completed")
            
            # Read and execute trading schema SQL
            if TRADING_SCHEMA_SQLITE.exists():
                with open(TRADING_SCHEMA_SQLITE, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                self.connection.executescript(schema_sql)
                logger.info("Trading schema created successfully")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_connection(self):
        """Get database connection as context manager"""
        from contextlib import contextmanager
        
        @contextmanager
        def _get_conn():
            # Always create a new connection for thread safety
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                conn.row_factory = sqlite3.Row
                yield conn
            except Exception as e:
                logger.error(f"Database connection error: {e}")
                yield None
            finally:
                if conn:
                    conn.close()
        
        return _get_conn()
    
    def get_raw_connection(self):
        """Get a raw connection for pandas operations"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            return conn
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            return None
    
    def execute_query(self, query: str, params: tuple = ()) -> Optional[list]:
        """Execute a query and return results"""
        try:
            with self.get_connection() as conn:
                if not conn:
                    return None
                    
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return None
                    
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

def get_database_manager(db_path: str = "data/trading.db") -> DatabaseManager:
    """Get a database manager instance"""
    return DatabaseManager(db_path)
