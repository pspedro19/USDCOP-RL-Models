"""
Database Integration Layer
==========================
Provides unified database access for all components with proper error handling,
transaction management, and event publishing.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from contextlib import contextmanager
import pandas as pd

from .database_manager import DatabaseManager

# Try to import event bus, but make it optional
try:
    from ..events.bus import event_bus, Event, EventType
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    event_bus = None
    Event = None
    EventType = None

logger = logging.getLogger(__name__)


class DatabaseIntegration:
    """Unified database integration for all components"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            try:
                self.db_manager = DatabaseManager()
                self.initialized = True
                logger.info("Database integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database integration: {e}")
                self.db_manager = None
                self.initialized = False
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        if not self.db_manager:
            logger.error("Database manager not initialized")
            yield None
            return
        
        # Use raw connection for pandas compatibility
        conn = self.db_manager.get_raw_connection()
        if not conn:
            logger.error("Failed to get database connection")
            yield None
            return
            
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            conn.close()
    
    def save_market_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Save market data to database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return False
            
        try:
            with self.transaction() as conn:
                if conn is None:
                    return False
                    
                # Add metadata columns
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                df_copy['timeframe'] = timeframe
                df_copy['inserted_at'] = datetime.now(timezone.utc)
                
                # Save to database
                df_copy.to_sql('market_data', conn, if_exists='append', index=False)
                
                # Publish event if available
                if HAS_EVENT_BUS and event_bus:
                    try:
                        event_bus.publish(Event(
                            event=EventType.DATA_SAVED.value,
                            source='db_integration',
                            ts=datetime.now(timezone.utc).isoformat(),
                            payload={'symbol': symbol, 'timeframe': timeframe, 'rows': len(df)}
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to publish event: {e}")
                
                logger.info(f"Saved {len(df)} rows of market data for {symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save market data: {e}")
            return False
    
    def save_features(self, df: pd.DataFrame, run_id: str) -> bool:
        """Save engineered features to database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return False
            
        try:
            with self.transaction() as conn:
                if conn is None:
                    return False
                    
                df_copy = df.copy()
                df_copy['run_id'] = run_id
                df_copy['created_at'] = datetime.now(timezone.utc)
                
                df_copy.to_sql('features', conn, if_exists='append', index=False)
                
                logger.info(f"Saved {len(df)} rows of features for run {run_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            return False
    
    def save_trades(self, trades: List[Dict[str, Any]], strategy_id: str) -> bool:
        """Save trade history to database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return False
            
        try:
            with self.transaction() as conn:
                if conn is None:
                    return False
                    
                df = pd.DataFrame(trades)
                df['strategy_id'] = strategy_id
                df['executed_at'] = datetime.now(timezone.utc)
                
                df.to_sql('trades', conn, if_exists='append', index=False)
                
                # Publish event if available
                if HAS_EVENT_BUS and event_bus:
                    try:
                        event_bus.publish(Event(
                            event=EventType.TRADE_EXECUTED.value,
                            source='db_integration',
                            ts=datetime.now(timezone.utc).isoformat(),
                            payload={'strategy_id': strategy_id, 'trade_count': len(trades)}
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to publish event: {e}")
                
                logger.info(f"Saved {len(trades)} trades for strategy {strategy_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save trades: {e}")
            return False
    
    def save_model_metrics(self, metrics: Dict[str, Any], model_id: str) -> bool:
        """Save model training metrics to database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return False
            
        try:
            with self.transaction() as conn:
                if conn is None:
                    return False
                    
                df = pd.DataFrame([metrics])
                df['model_id'] = model_id
                df['recorded_at'] = datetime.now(timezone.utc)
                
                df.to_sql('model_metrics', conn, if_exists='append', index=False)
                
                logger.info(f"Saved metrics for model {model_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save model metrics: {e}")
            return False
    
    def save_backtest_results(self, results: Dict[str, Any], backtest_id: str) -> bool:
        """Save backtest results to database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return False
            
        try:
            with self.transaction() as conn:
                if conn is None:
                    return False
                    
                df = pd.DataFrame([results])
                df['backtest_id'] = backtest_id
                df['created_at'] = datetime.now(timezone.utc)
                
                df.to_sql('backtest_results', conn, if_exists='append', index=False)
                
                logger.info(f"Saved backtest results for {backtest_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
            return False
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Retrieve latest market data from database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return pd.DataFrame()
            
        try:
            query = """
                SELECT * FROM market_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time DESC
                LIMIT ?
            """
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
                return df
                
        except Exception as e:
            logger.error(f"Failed to retrieve market data: {e}")
            return pd.DataFrame()
    
    def get_model_metrics(self, model_id: str = None, limit: int = 100) -> pd.DataFrame:
        """Retrieve model metrics from database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return pd.DataFrame()
            
        try:
            if model_id:
                query = """
                    SELECT * FROM model_metrics 
                    WHERE model_id = ?
                    ORDER BY recorded_at DESC
                    LIMIT ?
                """
                params = (model_id, limit)
            else:
                query = """
                    SELECT * FROM model_metrics 
                    ORDER BY recorded_at DESC
                    LIMIT ?
                """
                params = (limit,)
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
                
        except Exception as e:
            logger.error(f"Failed to retrieve model metrics: {e}")
            return pd.DataFrame()
    
    def get_trades(self, strategy_id: str = None, start_date: datetime = None, 
                   end_date: datetime = None) -> pd.DataFrame:
        """Retrieve trades from database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return pd.DataFrame()
            
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if strategy_id:
                query += " AND strategy_id = ?"
                params.append(strategy_id)
            
            if start_date:
                query += " AND executed_at >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND executed_at <= ?"
                params.append(end_date)
            
            query += " ORDER BY executed_at DESC"
            
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
                
        except Exception as e:
            logger.error(f"Failed to retrieve trades: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """Clean up old data from database"""
        if not self.db_manager:
            logger.error("Database not initialized")
            return False
            
        try:
            cutoff_date = datetime.now(timezone.utc) - pd.Timedelta(days=days_to_keep)
            
            with self.transaction() as conn:
                if conn is None:
                    return False
                    
                # Clean up old market data
                conn.execute(
                    "DELETE FROM market_data WHERE inserted_at < ?",
                    (cutoff_date,)
                )
                
                # Clean up old features
                conn.execute(
                    "DELETE FROM features WHERE created_at < ?",
                    (cutoff_date,)
                )
                
                logger.info(f"Cleaned up data older than {days_to_keep} days")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.db_manager:
            return {"error": "Database not initialized"}
            
        try:
            stats = {}
            
            with self.db_manager.get_connection() as conn:
                # Count records in each table
                tables = ['market_data', 'features', 'trades', 'model_metrics', 'backtest_results']
                
                for table in tables:
                    try:
                        result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                        stats[f"{table}_count"] = result[0] if result else 0
                    except:
                        stats[f"{table}_count"] = 0
                
                # Get database size
                try:
                    result = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()").fetchone()
                    stats['database_size_bytes'] = result[0] if result else 0
                except:
                    stats['database_size_bytes'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {"error": str(e)}


# Singleton instance
db_integration = DatabaseIntegration()


# Helper functions for easy access
def save_market_data(df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
    """Save market data using singleton instance"""
    return db_integration.save_market_data(df, symbol, timeframe)


def save_trades(trades: List[Dict[str, Any]], strategy_id: str) -> bool:
    """Save trades using singleton instance"""
    return db_integration.save_trades(trades, strategy_id)


def save_model_metrics(metrics: Dict[str, Any], model_id: str) -> bool:
    """Save model metrics using singleton instance"""
    return db_integration.save_model_metrics(metrics, model_id)


def get_latest_data(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Get latest data using singleton instance"""
    return db_integration.get_latest_data(symbol, timeframe, limit)