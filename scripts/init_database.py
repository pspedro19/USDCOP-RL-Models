#!/usr/bin/env python3
"""
Database Initialization Script for USDCOP Trading RL System

This script initializes the database with the required schemas.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.database.database_manager import DatabaseManager

def main():
    """Initialize the database"""
    # Setup simple logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting database initialization...")
    
    try:
        # Initialize database
        with DatabaseManager() as db_manager:
            success = db_manager.initialize_database()
            
            if success:
                logger.info("✅ Database initialized successfully!")
                logger.info(f"Database location: {db_manager.db_path}")
            else:
                logger.error("❌ Database initialization failed!")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        sys.exit(1)
    
    logger.info("Database initialization completed!")

if __name__ == "__main__":
    main()
