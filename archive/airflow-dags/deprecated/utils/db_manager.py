"""
Database Manager for USDCOP Trading System
==========================================

Provides PostgreSQL connectivity and data operations for the trading pipeline.
Replaces MinIO storage for structured data with PostgreSQL tables.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
from typing import Dict, List, Optional, Any, Tuple
import os
from datetime import datetime, timedelta
import json
import hashlib
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import numpy as np
from pathlib import Path

class DatabaseManager:
    """Manages PostgreSQL connections and operations for the trading system"""

    def __init__(self, connection_params: Optional[Dict[str, str]] = None):
        """
        Initialize database manager with connection parameters

        Args:
            connection_params: Dict with host, port, database, user, password
                              If None, will use environment variables
        """
        if connection_params is None:
            # Use environment variables (from docker-compose)
            self.connection_params = {
                'host': os.getenv('POSTGRES_HOST', 'postgres'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
                'user': os.getenv('POSTGRES_USER', 'admin'),
                'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
            }
        else:
            self.connection_params = connection_params

        self.connection_string = (
            f"postgresql://{self.connection_params['user']}:"
            f"{self.connection_params['password']}@"
            f"{self.connection_params['host']}:"
            f"{self.connection_params['port']}/"
            f"{self.connection_params['database']}"
        )

        # Initialize SQLAlchemy engine for pandas operations
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine"""
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            logging.info("✅ Database engine initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize database engine: {e}")
            raise

    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            from sqlalchemy import text

            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logging.info("✅ Database connection test successful")
                return True
        except Exception as e:
            logging.error(f"❌ Database connection test failed: {e}")
            return False

    def insert_market_data(self, df: pd.DataFrame, run_id: str, batch_id: str) -> int:
        """
        DEPRECATED: Use insert_ohlcv_data() instead

        This method references the old market_data table which has been removed.
        Use insert_ohlcv_data() for the unified usdcop_m5_ohlcv table.

        Args:
            df: DataFrame with market data (time, open, high, low, close, volume, source)
            run_id: Pipeline run identifier
            batch_id: Batch identifier

        Returns:
            Number of records inserted
        """
        logging.warning("⚠️ DEPRECATED: insert_market_data() references removed table. Use insert_ohlcv_data()")
        raise DeprecationWarning("Table market_data has been removed. Use usdcop_m5_ohlcv instead.")
        try:
            if df.empty:
                logging.warning("Empty dataframe provided for market data insertion")
                return 0

            # Prepare data for insertion
            df_copy = df.copy()

            # Ensure required columns exist
            required_cols = ['time', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df_copy.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Rename time column to datetime and timestamp for database compatibility
            if 'time' in df_copy.columns:
                df_copy = df_copy.rename(columns={'time': 'datetime'})
                # Also populate timestamp column for TimescaleDB partitioning
                df_copy['timestamp'] = df_copy['datetime']

            # Add metadata columns
            df_copy['pipeline_run_id'] = run_id
            df_copy['batch_id'] = batch_id
            df_copy['created_at'] = datetime.now()
            df_copy['updated_at'] = datetime.now()

            # Ensure volume column exists
            if 'volume' not in df_copy.columns:
                df_copy['volume'] = 0

            # Set default values for missing columns
            if 'symbol' not in df_copy.columns:
                df_copy['symbol'] = 'USDCOP'
            if 'timeframe' not in df_copy.columns:
                df_copy['timeframe'] = '5min'
            if 'source' not in df_copy.columns:
                df_copy['source'] = 'twelvedata'
            if 'timezone' not in df_copy.columns:
                df_copy['timezone'] = 'America/Bogota'
            if 'trading_session' not in df_copy.columns:
                # Determine if timestamp is within trading hours (8 AM - 12:55 PM COT)
                df_copy['trading_session'] = (
                    (df_copy['datetime'].dt.hour >= 8) &
                    (df_copy['datetime'].dt.hour < 13)
                )

            # Select only the columns that exist in the database table
            db_columns = [
                'symbol', 'timeframe', 'datetime', 'open', 'high', 'low', 'close',
                'volume', 'source', 'timezone', 'trading_session', 'created_at',
                'updated_at', 'batch_id', 'pipeline_run_id'
            ]
            df_to_insert = df_copy[db_columns]

            # Insert data using pandas to_sql with ON CONFLICT handling
            insert_count = df_to_insert.to_sql(
                'market_data',
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )

            logging.info(f"✅ Inserted {len(df_to_insert)} market data records to PostgreSQL")
            return len(df_to_insert)

        except Exception as e:
            logging.error(f"❌ Error inserting market data: {e}")
            logging.error(f"DataFrame shape: {df.shape}")
            logging.error(f"DataFrame columns: {list(df.columns)}")
            raise

    def insert_pipeline_run(self, run_id: str, pipeline_name: str, status: str = 'running',
                           config: Optional[Dict] = None) -> int:
        """
        Insert or update pipeline run record

        Args:
            run_id: Unique run identifier
            pipeline_name: Name of the pipeline
            status: Run status ('running', 'success', 'failed')
            config: Pipeline configuration dictionary

        Returns:
            Pipeline run database ID
        """
        try:
            from sqlalchemy import text

            pipeline_data = {
                'run_id': run_id,
                'pipeline_name': pipeline_name,
                'execution_date': datetime.now(),
                'start_time': datetime.now(),
                'status': status,
                'config': json.dumps(config) if config else '{}',
                'created_at': datetime.now()
            }

            # Check if run already exists
            with self.engine.begin() as conn:  # Use begin() for automatic transaction handling
                result = conn.execute(
                    text("SELECT id FROM pipeline_runs WHERE run_id = :run_id"),
                    {"run_id": run_id}
                ).fetchone()

                if result:
                    # Update existing run
                    conn.execute(
                        text("""UPDATE pipeline_runs
                               SET status = :status, config = :config
                               WHERE run_id = :run_id"""),
                        {
                            "status": status,
                            "config": json.dumps(config) if config else '{}',
                            "run_id": run_id
                        }
                    )
                    pipeline_id = result[0]
                    logging.info(f"✅ Updated pipeline run: {run_id}")
                else:
                    # Insert new run
                    result = conn.execute(
                        text("""INSERT INTO pipeline_runs
                               (run_id, pipeline_name, execution_date, start_time, status, config, created_at)
                               VALUES (:run_id, :pipeline_name, :execution_date, :start_time, :status, :config, :created_at)
                               RETURNING id"""),
                        {
                            "run_id": pipeline_data['run_id'],
                            "pipeline_name": pipeline_data['pipeline_name'],
                            "execution_date": pipeline_data['execution_date'],
                            "start_time": pipeline_data['start_time'],
                            "status": pipeline_data['status'],
                            "config": pipeline_data['config'],
                            "created_at": pipeline_data['created_at']
                        }
                    )
                    pipeline_id = result.fetchone()[0]
                    logging.info(f"✅ Created pipeline run: {run_id}")

                return pipeline_id

        except Exception as e:
            logging.error(f"❌ Error managing pipeline run: {e}")
            raise

    def update_pipeline_run_status(self, run_id: str, status: str,
                                  records_processed: int = 0, error_message: str = None):
        """Update pipeline run status and metrics"""
        try:
            from sqlalchemy import text

            with self.engine.begin() as conn:
                if error_message:
                    conn.execute(
                        text("""UPDATE pipeline_runs
                               SET status = :status, end_time = NOW(), records_processed = :records_processed,
                                   error_message = :error_message
                               WHERE run_id = :run_id"""),
                        {
                            "status": status,
                            "records_processed": records_processed,
                            "error_message": error_message,
                            "run_id": run_id
                        }
                    )
                else:
                    conn.execute(
                        text("""UPDATE pipeline_runs
                               SET status = :status, end_time = NOW(), records_processed = :records_processed
                               WHERE run_id = :run_id"""),
                        {
                            "status": status,
                            "records_processed": records_processed,
                            "run_id": run_id
                        }
                    )
                logging.info(f"✅ Updated pipeline run {run_id} status to {status}")

        except Exception as e:
            logging.error(f"❌ Error updating pipeline run status: {e}")
            raise

    def insert_api_usage(self, api_key_name: str, endpoint: str, success: bool = True,
                        response_time_ms: int = None, status_code: int = None,
                        error_message: str = None, credits_used: int = 1):
        """Track API usage for monitoring and rate limiting"""
        try:
            usage_data = {
                'api_key_name': api_key_name,
                'endpoint': endpoint,
                'success': success,
                'response_time_ms': response_time_ms,
                'status_code': status_code,
                'error_message': error_message,
                'credits_used': credits_used,
                'request_datetime': datetime.now(),
                'created_at': datetime.now()
            }

            df_usage = pd.DataFrame([usage_data])
            df_usage.to_sql('api_usage', self.engine, if_exists='append', index=False)

            logging.debug(f"📊 Logged API usage for {api_key_name}")

        except Exception as e:
            logging.error(f"❌ Error logging API usage: {e}")
            # Don't raise - API usage logging shouldn't break the pipeline

    def insert_data_quality_check(self, run_id: str, check_name: str, check_type: str,
                                 status: str, expected_value: float = None,
                                 actual_value: float = None, details: Dict = None):
        """Insert data quality check results"""
        try:
            quality_data = {
                'run_id': run_id,
                'check_name': check_name,
                'check_type': check_type,
                'status': status,
                'expected_value': expected_value,
                'actual_value': actual_value,
                'details': json.dumps(details) if details else '{}',
                'created_at': datetime.now()
            }

            df_quality = pd.DataFrame([quality_data])
            df_quality.to_sql('data_quality_checks', self.engine, if_exists='append', index=False)

            logging.info(f"✅ Logged quality check: {check_name} - {status}")

        except Exception as e:
            logging.error(f"❌ Error logging data quality check: {e}")
            raise

    def get_latest_market_data(self, symbol: str = 'USD/COP', limit: int = 100) -> pd.DataFrame:
        """
        Retrieve latest market data records from usdcop_m5_ohlcv table

        Updated to use unified table usdcop_m5_ohlcv instead of deprecated market_data
        """
        try:
            # Convert USDCOP to USD/COP for compatibility
            if symbol == 'USDCOP':
                symbol = 'USD/COP'

            query = """
                SELECT * FROM usdcop_m5_ohlcv
                WHERE symbol = %s
                ORDER BY time DESC
                LIMIT %s
            """

            df = pd.read_sql(query, self.engine, params=(symbol, limit))
            logging.info(f"✅ Retrieved {len(df)} latest OHLCV records from usdcop_m5_ohlcv")
            return df

        except Exception as e:
            logging.error(f"❌ Error retrieving OHLCV data: {e}")
            return pd.DataFrame()

    def get_market_data_stats(self, symbol: str = 'USD/COP') -> Dict:
        """
        Get market data statistics from usdcop_m5_ohlcv table

        Updated to use unified table usdcop_m5_ohlcv instead of deprecated market_data
        """
        try:
            from sqlalchemy import text

            # Convert USDCOP to USD/COP for compatibility
            if symbol == 'USDCOP':
                symbol = 'USD/COP'

            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT
                        COUNT(*) as total_records,
                        MIN(time) as earliest_date,
                        MAX(time) as latest_date,
                        COUNT(DISTINCT DATE(time)) as trading_days,
                        100.0 as trading_session_pct
                    FROM usdcop_m5_ohlcv
                    WHERE symbol = :symbol
                """), {"symbol": symbol}).fetchone()

                if result:
                    stats = {
                        'total_records': result[0],
                        'earliest_date': str(result[1]) if result[1] else None,
                        'latest_date': str(result[2]) if result[2] else None,
                        'trading_days': result[3],
                        'trading_session_percentage': 100.0  # All data is now within trading hours
                    }
                    logging.info(f"✅ Retrieved OHLCV stats for {symbol} from usdcop_m5_ohlcv")
                    return stats
                else:
                    return {}

        except Exception as e:
            logging.error(f"❌ Error retrieving OHLCV stats: {e}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """Enhanced cleanup of old data based on comprehensive retention policies"""
        try:
            cleanup_results = {}

            with self.engine.begin() as conn:
                # Clean up old API usage records (keep 30 days)
                result = conn.execute(
                    text("DELETE FROM api_usage WHERE request_datetime < NOW() - INTERVAL '30 days'")
                )
                cleanup_results['api_usage'] = result.rowcount

                # Clean up old pipeline runs (keep 90 days)
                result = conn.execute(
                    text("DELETE FROM pipeline_runs WHERE created_at < NOW() - INTERVAL '90 days'")
                )
                cleanup_results['pipeline_runs'] = result.rowcount

                # Clean up expired backup metadata
                result = conn.execute(
                    text("DELETE FROM backup_metadata WHERE is_active = false AND updated_at < NOW() - INTERVAL '7 days'")
                )
                cleanup_results['backup_metadata'] = result.rowcount

                # Clean up processed ready signals (keep 7 days)
                result = conn.execute(
                    text("DELETE FROM ready_signals WHERE status IN ('processed', 'expired') AND updated_at < NOW() - INTERVAL '7 days'")
                )
                cleanup_results['ready_signals'] = result.rowcount

                # Clean up resolved data gaps (keep 30 days)
                result = conn.execute(
                    text("DELETE FROM data_gaps WHERE resolution_status = 'resolved' AND resolved_at < NOW() - INTERVAL '30 days'")
                )
                cleanup_results['data_gaps'] = result.rowcount

                # Clean up old pipeline health records (keep 30 days)
                result = conn.execute(
                    text("DELETE FROM pipeline_health WHERE measured_at < NOW() - INTERVAL '30 days'")
                )
                cleanup_results['pipeline_health'] = result.rowcount

                # Clean up old system metrics (keep based on category)
                result = conn.execute(text("""
                    DELETE FROM system_metrics
                    WHERE (
                        (category = 'api' AND measured_at < NOW() - INTERVAL '7 days') OR
                        (category = 'pipeline' AND measured_at < NOW() - INTERVAL '30 days') OR
                        (category = 'trading' AND measured_at < NOW() - INTERVAL '90 days') OR
                        (category = 'system' AND measured_at < NOW() - INTERVAL '14 days')
                    )
                """))
                cleanup_results['system_metrics'] = result.rowcount

                # Clean up old data quality checks (keep 60 days)
                result = conn.execute(
                    text("DELETE FROM data_quality_checks WHERE created_at < NOW() - INTERVAL '60 days'")
                )
                cleanup_results['data_quality_checks'] = result.rowcount

                # Clean up old user sessions (expired)
                result = conn.execute(
                    text("DELETE FROM user_sessions WHERE expires_at < NOW()")
                )
                cleanup_results['user_sessions'] = result.rowcount

                # Clean up old websocket connections (disconnected > 24 hours)
                result = conn.execute(
                    text("DELETE FROM websocket_connections WHERE status = 'disconnected' AND disconnected_at < NOW() - INTERVAL '24 hours'")
                )
                cleanup_results['websocket_connections'] = result.rowcount

                total_deleted = sum(cleanup_results.values())
                logging.info(f"✅ Comprehensive cleanup completed: {total_deleted} total records deleted")
                logging.info(f"📊 Cleanup breakdown: {cleanup_results}")

                return cleanup_results

        except Exception as e:
            logging.error(f"❌ Error during comprehensive cleanup: {e}")
            raise

    def setup_data_retention_policies(self) -> Dict[str, str]:
        """Setup and document data retention policies"""
        retention_policies = {
            'usdcop_m5_ohlcv': 'Permanent retention (core trading data)',
            'trading_signals': 'Permanent retention (performance analysis)',
            'trading_performance': 'Permanent retention (historical performance)',
            'pipeline_runs': '90 days (operational logs)',
            'api_usage': '30 days (rate limiting and monitoring)',
            'backup_metadata': 'Active backups + 7 days after deactivation',
            'ready_signals': '7 days after processing (coordination logs)',
            'data_gaps': '30 days after resolution (quality tracking)',
            'pipeline_health': '30 days (operational monitoring)',
            'system_metrics': '7-90 days (varies by category)',
            'data_quality_checks': '60 days (quality history)',
            'user_sessions': 'Until expiration (security)',
            'websocket_connections': '24 hours after disconnection (connection logs)',
            'users': 'Permanent (unless manually deleted)'
        }

        try:
            # Log current retention policies
            for table, policy in retention_policies.items():
                logging.info(f"📋 {table}: {policy}")

            return retention_policies

        except Exception as e:
            logging.error(f"❌ Error setting up retention policies: {e}")
            return {}

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            stats = {}

            with self.engine.connect() as conn:
                # Table sizes and record counts
                result = conn.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        n_tup_ins as total_inserts,
                        n_tup_upd as total_updates,
                        n_tup_del as total_deletes,
                        n_live_tup as live_rows,
                        n_dead_tup as dead_rows,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                    ORDER BY n_live_tup DESC
                """))

                table_stats = []
                for row in result:
                    table_stats.append({
                        'table_name': row.tablename,
                        'live_rows': row.live_rows,
                        'dead_rows': row.dead_rows,
                        'total_inserts': row.total_inserts,
                        'total_updates': row.total_updates,
                        'total_deletes': row.total_deletes,
                        'last_vacuum': row.last_vacuum,
                        'last_analyze': row.last_analyze
                    })

                stats['table_statistics'] = table_stats

                # Database size information
                result = conn.execute(text("""
                    SELECT
                        pg_size_pretty(pg_database_size(current_database())) as database_size,
                        pg_size_pretty(pg_total_relation_size('usdcop_m5_ohlcv')) as ohlcv_table_size,
                        pg_size_pretty(pg_indexes_size('usdcop_m5_ohlcv')) as ohlcv_index_size
                """))

                size_info = result.fetchone()
                if size_info:
                    stats['database_size'] = {
                        'total_database_size': size_info.database_size,
                        'ohlcv_table_size': size_info.ohlcv_table_size,
                        'ohlcv_index_size': size_info.ohlcv_index_size
                    }

                # Recent activity summary
                result = conn.execute(text("""
                    SELECT
                        'usdcop_m5_ohlcv' as table_name,
                        COUNT(*) as recent_records,
                        MAX(created_at) as latest_record
                    FROM usdcop_m5_ohlcv
                    WHERE created_at >= NOW() - INTERVAL '24 hours'

                    UNION ALL

                    SELECT
                        'pipeline_runs' as table_name,
                        COUNT(*) as recent_records,
                        MAX(created_at) as latest_record
                    FROM pipeline_runs
                    WHERE created_at >= NOW() - INTERVAL '24 hours'

                    UNION ALL

                    SELECT
                        'api_usage' as table_name,
                        COUNT(*) as recent_records,
                        MAX(created_at) as latest_record
                    FROM api_usage
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """))

                recent_activity = []
                for row in result:
                    recent_activity.append({
                        'table_name': row.table_name,
                        'recent_records': row.recent_records,
                        'latest_record': row.latest_record
                    })

                stats['recent_activity'] = recent_activity

                logging.info("✅ Retrieved comprehensive database statistics")
                return stats

        except Exception as e:
            logging.error(f"❌ Error retrieving database statistics: {e}")
            return {}

    # ========================================
    # BACKUP METADATA MANAGEMENT
    # ========================================

    def create_backup_record(self, file_path: str, data_start_date: datetime,
                           data_end_date: datetime, record_count: int,
                           file_size_bytes: int, pipeline_run_id: str = None,
                           backup_type: str = 'incremental', **kwargs) -> str:
        """Create a backup metadata record"""
        try:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            file_name = Path(file_path).name

            # Calculate checksum if file exists
            checksum = None
            if os.path.exists(file_path):
                checksum = self._calculate_file_checksum(file_path)

            backup_data = {
                'backup_id': backup_id,
                'file_path': file_path,
                'file_name': file_name,
                'backup_type': backup_type,
                'symbol': kwargs.get('symbol', 'USDCOP'),
                'timeframe': kwargs.get('timeframe', '5min'),
                'data_start_date': data_start_date,
                'data_end_date': data_end_date,
                'file_size_bytes': file_size_bytes,
                'record_count': record_count,
                'checksum': checksum,
                'pipeline_run_id': pipeline_run_id,
                'compression_type': kwargs.get('compression_type', 'gzip'),
                'backup_method': kwargs.get('backup_method', 'database_export'),
                'retention_days': kwargs.get('retention_days', 30)
            }

            df_backup = pd.DataFrame([backup_data])
            df_backup.to_sql('backup_metadata', self.engine, if_exists='append', index=False)

            logging.info(f"✅ Created backup record: {backup_id}")
            return backup_id

        except Exception as e:
            logging.error(f"❌ Error creating backup record: {e}")
            raise

    def update_backup_validation(self, backup_id: str, validation_status: str,
                               completeness_pct: float = None,
                               quality_score: float = None,
                               validation_errors: List = None):
        """Update backup validation results"""
        try:
            with self.engine.begin() as conn:
                update_data = {
                    'validation_status': validation_status,
                    'updated_at': datetime.now()
                }

                if completeness_pct is not None:
                    update_data['completeness_pct'] = completeness_pct
                if quality_score is not None:
                    update_data['data_quality_score'] = quality_score
                if validation_errors is not None:
                    update_data['validation_errors'] = json.dumps(validation_errors)

                # Build SET clause dynamically
                set_clause = ', '.join([f"{k} = :{k}" for k in update_data.keys()])
                query = f"UPDATE backup_metadata SET {set_clause} WHERE backup_id = :backup_id"
                update_data['backup_id'] = backup_id

                conn.execute(text(query), update_data)
                logging.info(f"✅ Updated backup validation for {backup_id}: {validation_status}")

        except Exception as e:
            logging.error(f"❌ Error updating backup validation: {e}")
            raise

    def get_backup_metadata(self, backup_id: str = None, days_back: int = 7) -> pd.DataFrame:
        """Retrieve backup metadata records"""
        try:
            if backup_id:
                query = "SELECT * FROM backup_metadata WHERE backup_id = %s"
                params = (backup_id,)
            else:
                query = """
                    SELECT * FROM backup_metadata
                    WHERE created_at >= %s AND is_active = true
                    ORDER BY created_at DESC
                """
                params = (datetime.now() - timedelta(days=days_back),)

            df = pd.read_sql(query, self.engine, params=params)
            logging.info(f"✅ Retrieved {len(df)} backup metadata records")
            return df

        except Exception as e:
            logging.error(f"❌ Error retrieving backup metadata: {e}")
            return pd.DataFrame()

    def cleanup_expired_backups(self) -> int:
        """Clean up expired backup records"""
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text("""
                    UPDATE backup_metadata
                    SET is_active = false
                    WHERE expire_date < NOW() AND is_active = true
                """))

                expired_count = result.rowcount
                logging.info(f"✅ Marked {expired_count} backups as expired")
                return expired_count

        except Exception as e:
            logging.error(f"❌ Error cleaning up expired backups: {e}")
            raise

    # ========================================
    # READY SIGNALS COORDINATION
    # ========================================

    def create_ready_signal(self, pipeline_run_id: str, signal_type: str,
                          data_start_time: datetime, data_end_time: datetime,
                          records_available: int, completeness_pct: float,
                          **kwargs) -> str:
        """Create a ready signal for L0→WebSocket coordination"""
        try:
            signal_id = f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            signal_data = {
                'signal_id': signal_id,
                'pipeline_run_id': pipeline_run_id,
                'signal_type': signal_type,
                'symbol': kwargs.get('symbol', 'USDCOP'),
                'timeframe': kwargs.get('timeframe', '5min'),
                'data_start_time': data_start_time,
                'data_end_time': data_end_time,
                'records_available': records_available,
                'completeness_pct': completeness_pct,
                'quality_score': kwargs.get('quality_score'),
                'priority': kwargs.get('priority', 1),
                'metadata': json.dumps(kwargs.get('metadata', {})),
                'dependencies': json.dumps(kwargs.get('dependencies', [])),
                'expires_at': kwargs.get('expires_at', datetime.now() + timedelta(hours=1))
            }

            df_signal = pd.DataFrame([signal_data])
            df_signal.to_sql('ready_signals', self.engine, if_exists='append', index=False)

            logging.info(f"✅ Created ready signal: {signal_id} ({signal_type})")
            return signal_id

        except Exception as e:
            logging.error(f"❌ Error creating ready signal: {e}")
            raise

    def acknowledge_ready_signal(self, signal_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a ready signal"""
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text("""
                    UPDATE ready_signals
                    SET status = 'acknowledged',
                        acknowledged_by = :acknowledged_by,
                        acknowledged_at = NOW()
                    WHERE signal_id = :signal_id AND status = 'pending'
                """), {
                    'signal_id': signal_id,
                    'acknowledged_by': acknowledged_by
                })

                if result.rowcount > 0:
                    logging.info(f"✅ Acknowledged ready signal: {signal_id} by {acknowledged_by}")
                    return True
                else:
                    logging.warning(f"⚠️ Ready signal {signal_id} not found or already processed")
                    return False

        except Exception as e:
            logging.error(f"❌ Error acknowledging ready signal: {e}")
            raise

    def complete_ready_signal(self, signal_id: str, processed_by: str,
                            processing_duration_ms: int = None) -> bool:
        """Mark a ready signal as processed"""
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text("""
                    UPDATE ready_signals
                    SET status = 'processed',
                        processed_by = :processed_by,
                        processed_at = NOW(),
                        processing_duration_ms = :processing_duration_ms
                    WHERE signal_id = :signal_id AND status = 'acknowledged'
                """), {
                    'signal_id': signal_id,
                    'processed_by': processed_by,
                    'processing_duration_ms': processing_duration_ms
                })

                if result.rowcount > 0:
                    logging.info(f"✅ Completed ready signal: {signal_id} by {processed_by}")
                    return True
                else:
                    logging.warning(f"⚠️ Ready signal {signal_id} not found or not acknowledged")
                    return False

        except Exception as e:
            logging.error(f"❌ Error completing ready signal: {e}")
            raise

    def get_pending_ready_signals(self, signal_type: str = None, limit: int = 50) -> pd.DataFrame:
        """Get pending ready signals"""
        try:
            if signal_type:
                query = """
                    SELECT * FROM ready_signals
                    WHERE status = 'pending' AND signal_type = %s
                    AND expires_at > NOW()
                    ORDER BY priority ASC, created_at ASC
                    LIMIT %s
                """
                params = (signal_type, limit)
            else:
                query = """
                    SELECT * FROM ready_signals
                    WHERE status = 'pending' AND expires_at > NOW()
                    ORDER BY priority ASC, created_at ASC
                    LIMIT %s
                """
                params = (limit,)

            df = pd.read_sql(query, self.engine, params=params)
            logging.info(f"✅ Retrieved {len(df)} pending ready signals")
            return df

        except Exception as e:
            logging.error(f"❌ Error retrieving pending ready signals: {e}")
            return pd.DataFrame()

    def cleanup_expired_signals(self) -> int:
        """Clean up expired ready signals"""
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text("""
                    UPDATE ready_signals
                    SET status = 'expired'
                    WHERE status IN ('pending', 'acknowledged') AND expires_at < NOW()
                """))

                expired_count = result.rowcount
                logging.info(f"✅ Marked {expired_count} ready signals as expired")
                return expired_count

        except Exception as e:
            logging.error(f"❌ Error cleaning up expired signals: {e}")
            raise

    # ========================================
    # DATA GAP MANAGEMENT
    # ========================================

    def detect_data_gaps(self, start_date: datetime, end_date: datetime,
                        symbol: str = 'USDCOP', timeframe: str = '5min') -> List[Dict]:
        """Detect data gaps in the specified time range"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM find_data_gaps(:start_date, :end_date, :symbol)
                """), {
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbol': symbol
                })

                gaps = []
                for row in result:
                    gap_data = {
                        'gap_start': row.gap_start,
                        'gap_end': row.gap_end,
                        'gap_duration': str(row.gap_duration),
                        'missing_points': row.missing_points
                    }
                    gaps.append(gap_data)

                logging.info(f"✅ Detected {len(gaps)} data gaps for {symbol} from {start_date} to {end_date}")
                return gaps

        except Exception as e:
            logging.error(f"❌ Error detecting data gaps: {e}")
            return []

    def create_data_gap_record(self, gap_start: datetime, gap_end: datetime,
                             missing_points: int, detected_by: str,
                             detection_method: str, **kwargs) -> str:
        """Create a data gap record"""
        try:
            gap_id = f"gap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            gap_duration = gap_end - gap_start

            # Calculate expected points based on 5-minute intervals
            expected_points = int(gap_duration.total_seconds() / 300)  # 300 seconds = 5 minutes

            gap_data = {
                'gap_id': gap_id,
                'symbol': kwargs.get('symbol', 'USDCOP'),
                'timeframe': kwargs.get('timeframe', '5min'),
                'source': kwargs.get('source', 'twelvedata'),
                'gap_start': gap_start,
                'gap_end': gap_end,
                'gap_duration': gap_duration,
                'missing_points': missing_points,
                'expected_points': expected_points,
                'detected_by': detected_by,
                'detection_method': detection_method,
                'detection_run_id': kwargs.get('detection_run_id'),
                'gap_type': kwargs.get('gap_type', 'data_provider'),
                'severity': kwargs.get('severity', 'medium'),
                'impact_score': kwargs.get('impact_score', 50.0),
                'context_data': json.dumps(kwargs.get('context_data', {}))
            }

            df_gap = pd.DataFrame([gap_data])
            df_gap.to_sql('data_gaps', self.engine, if_exists='append', index=False)

            logging.info(f"✅ Created data gap record: {gap_id}")
            return gap_id

        except Exception as e:
            logging.error(f"❌ Error creating data gap record: {e}")
            raise

    def update_gap_fill_status(self, gap_id: str, fill_status: str,
                             fill_method: str = None, filled_points: int = None,
                             fill_quality: str = None) -> bool:
        """Update the fill status of a data gap"""
        try:
            with self.engine.begin() as conn:
                update_data = {
                    'fill_status': fill_status,
                    'gap_id': gap_id
                }

                set_clauses = ['fill_status = :fill_status']

                if fill_method:
                    update_data['fill_method'] = fill_method
                    set_clauses.append('fill_method = :fill_method')

                if filled_points is not None:
                    update_data['filled_points'] = filled_points
                    set_clauses.append('filled_points = :filled_points')

                if fill_quality:
                    update_data['fill_quality'] = fill_quality
                    set_clauses.append('fill_quality = :fill_quality')

                if fill_status == 'filled':
                    set_clauses.append('filled_at = NOW()')
                    if filled_points is not None:
                        # Calculate success rate
                        set_clauses.append('fill_success_rate = (filled_points * 100.0 / missing_points)')

                query = f"UPDATE data_gaps SET {', '.join(set_clauses)} WHERE gap_id = :gap_id"
                result = conn.execute(text(query), update_data)

                if result.rowcount > 0:
                    logging.info(f"✅ Updated gap fill status: {gap_id} -> {fill_status}")
                    return True
                else:
                    logging.warning(f"⚠️ Gap {gap_id} not found")
                    return False

        except Exception as e:
            logging.error(f"❌ Error updating gap fill status: {e}")
            raise

    def get_unfilled_gaps(self, severity: str = None, limit: int = 100) -> pd.DataFrame:
        """Get unfilled data gaps"""
        try:
            if severity:
                query = """
                    SELECT * FROM data_gaps
                    WHERE fill_status IN ('detected', 'filling')
                    AND severity = %s
                    AND resolution_status = 'open'
                    ORDER BY severity DESC, gap_start ASC
                    LIMIT %s
                """
                params = (severity, limit)
            else:
                query = """
                    SELECT * FROM data_gaps
                    WHERE fill_status IN ('detected', 'filling')
                    AND resolution_status = 'open'
                    ORDER BY
                        CASE severity
                            WHEN 'critical' THEN 1
                            WHEN 'high' THEN 2
                            WHEN 'medium' THEN 3
                            WHEN 'low' THEN 4
                        END,
                        gap_start ASC
                    LIMIT %s
                """
                params = (limit,)

            df = pd.read_sql(query, self.engine, params=params)
            logging.info(f"✅ Retrieved {len(df)} unfilled gaps")
            return df

        except Exception as e:
            logging.error(f"❌ Error retrieving unfilled gaps: {e}")
            return pd.DataFrame()

    # ========================================
    # PIPELINE HEALTH MONITORING
    # ========================================

    def record_pipeline_health(self, pipeline_name: str, component: str,
                             status: str, health_score: float, **kwargs) -> str:
        """Record pipeline health metrics"""
        try:
            check_id = f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            health_data = {
                'check_id': check_id,
                'pipeline_name': pipeline_name,
                'component': component,
                'status': status,
                'health_score': health_score,
                'response_time_ms': kwargs.get('response_time_ms'),
                'error_rate': kwargs.get('error_rate', 0.0),
                'throughput_records_per_min': kwargs.get('throughput_records_per_min'),
                'cpu_usage_pct': kwargs.get('cpu_usage_pct'),
                'memory_usage_pct': kwargs.get('memory_usage_pct'),
                'disk_usage_pct': kwargs.get('disk_usage_pct'),
                'connectivity_status': kwargs.get('connectivity_status'),
                'data_freshness_min': kwargs.get('data_freshness_min'),
                'queue_length': kwargs.get('queue_length'),
                'active_connections': kwargs.get('active_connections'),
                'alert_level': kwargs.get('alert_level', 'none'),
                'issues': json.dumps(kwargs.get('issues', [])),
                'recommendations': json.dumps(kwargs.get('recommendations', [])),
                'check_metadata': json.dumps(kwargs.get('check_metadata', {})),
                'environment': kwargs.get('environment', 'production'),
                'version': kwargs.get('version')
            }

            df_health = pd.DataFrame([health_data])
            df_health.to_sql('pipeline_health', self.engine, if_exists='append', index=False)

            logging.info(f"✅ Recorded pipeline health: {pipeline_name}/{component} - {status}")
            return check_id

        except Exception as e:
            logging.error(f"❌ Error recording pipeline health: {e}")
            raise

    def get_pipeline_health_summary(self, hours_back: int = 24) -> pd.DataFrame:
        """Get pipeline health summary"""
        try:
            query = """
                SELECT
                    pipeline_name,
                    component,
                    status,
                    AVG(health_score) as avg_health_score,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(error_rate) as avg_error_rate,
                    COUNT(*) as check_count,
                    MAX(measured_at) as last_check
                FROM pipeline_health
                WHERE measured_at >= %s
                GROUP BY pipeline_name, component, status
                ORDER BY pipeline_name, component
            """

            since_time = datetime.now() - timedelta(hours=hours_back)
            df = pd.read_sql(query, self.engine, params=(since_time,))

            logging.info(f"✅ Retrieved pipeline health summary for last {hours_back} hours")
            return df

        except Exception as e:
            logging.error(f"❌ Error retrieving pipeline health summary: {e}")
            return pd.DataFrame()

    # ========================================
    # UTILITY FUNCTIONS
    # ========================================

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logging.error(f"❌ Error calculating checksum for {file_path}: {e}")
            return None

    def get_data_quality_summary(self, run_id: str = None, days_back: int = 7) -> Dict:
        """Get comprehensive data quality summary"""
        try:
            with self.engine.connect() as conn:
                if run_id:
                    # Quality summary for specific run
                    result = conn.execute(text("""
                        SELECT
                            COUNT(*) as total_checks,
                            COUNT(CASE WHEN status = 'passed' THEN 1 END) as passed_checks,
                            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_checks,
                            COUNT(CASE WHEN status = 'warning' THEN 1 END) as warning_checks,
                            AVG(CASE WHEN actual_value IS NOT NULL THEN actual_value END) as avg_actual_value,
                            AVG(CASE WHEN expected_value IS NOT NULL THEN expected_value END) as avg_expected_value
                        FROM data_quality_checks
                        WHERE run_id = :run_id
                    """), {'run_id': run_id})
                else:
                    # Quality summary for recent period
                    since_date = datetime.now() - timedelta(days=days_back)
                    result = conn.execute(text("""
                        SELECT
                            COUNT(*) as total_checks,
                            COUNT(CASE WHEN status = 'passed' THEN 1 END) as passed_checks,
                            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_checks,
                            COUNT(CASE WHEN status = 'warning' THEN 1 END) as warning_checks,
                            AVG(CASE WHEN actual_value IS NOT NULL THEN actual_value END) as avg_actual_value,
                            AVG(CASE WHEN expected_value IS NOT NULL THEN expected_value END) as avg_expected_value
                        FROM data_quality_checks
                        WHERE created_at >= :since_date
                    """), {'since_date': since_date})

                row = result.fetchone()
                if row:
                    summary = {
                        'total_checks': row.total_checks or 0,
                        'passed_checks': row.passed_checks or 0,
                        'failed_checks': row.failed_checks or 0,
                        'warning_checks': row.warning_checks or 0,
                        'pass_rate': round((row.passed_checks or 0) * 100.0 / max(row.total_checks or 1, 1), 2),
                        'avg_actual_value': float(row.avg_actual_value) if row.avg_actual_value else None,
                        'avg_expected_value': float(row.avg_expected_value) if row.avg_expected_value else None
                    }

                    logging.info(f"✅ Retrieved data quality summary: {summary['pass_rate']}% pass rate")
                    return summary
                else:
                    return {}

        except Exception as e:
            logging.error(f"❌ Error retrieving data quality summary: {e}")
            return {}

    def refresh_materialized_views(self) -> bool:
        """Refresh all materialized views"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT refresh_trading_views()"))
                logging.info("✅ Refreshed all materialized views")
                return True

        except Exception as e:
            logging.error(f"❌ Error refreshing materialized views: {e}")
            return False

    def close(self):
        """Close database connections"""
        try:
            if self.engine:
                self.engine.dispose()
                logging.info("✅ Database connections closed")
        except Exception as e:
            logging.error(f"❌ Error closing database connections: {e}")


# Convenience function for pipeline usage
def get_db_manager() -> DatabaseManager:
    """Get a database manager instance with default connection parameters"""
    return DatabaseManager()


# Enhanced test function
def test_database_connection():
    """Comprehensive test of database connectivity and enhanced operations"""
    try:
        db = DatabaseManager()

        # Test connection
        if not db.test_connection():
            return False

        print("🔍 Testing enhanced database functionality...")

        # Test market data stats
        stats = db.get_market_data_stats()
        print(f"📊 Market data stats: {stats}")

        # Test API usage logging
        db.insert_api_usage(
            api_key_name='TEST_KEY',
            endpoint='/test',
            success=True,
            response_time_ms=100,
            status_code=200
        )

        # Test pipeline run creation
        test_run_id = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pipeline_id = db.insert_pipeline_run(
            run_id=test_run_id,
            pipeline_name='test_pipeline',
            status='running',
            config={'test': True}
        )
        print(f"✅ Created test pipeline run: {test_run_id}")

        # Test backup record creation
        backup_id = db.create_backup_record(
            file_path='/tmp/test_backup.csv',
            data_start_date=datetime.now() - timedelta(hours=1),
            data_end_date=datetime.now(),
            record_count=100,
            file_size_bytes=1024,
            pipeline_run_id=test_run_id
        )
        print(f"✅ Created test backup record: {backup_id}")

        # Test ready signal creation
        signal_id = db.create_ready_signal(
            pipeline_run_id=test_run_id,
            signal_type='data_ready',
            data_start_time=datetime.now() - timedelta(minutes=30),
            data_end_time=datetime.now(),
            records_available=100,
            completeness_pct=95.5
        )
        print(f"✅ Created test ready signal: {signal_id}")

        # Test data gap detection (this will return empty since we don't have real data gaps)
        gaps = db.detect_data_gaps(
            start_date=datetime.now() - timedelta(hours=2),
            end_date=datetime.now()
        )
        print(f"🔍 Detected {len(gaps)} data gaps (expected 0 for test)")

        # Test pipeline health recording
        health_check_id = db.record_pipeline_health(
            pipeline_name='test_pipeline',
            component='data_ingestion',
            status='healthy',
            health_score=95.0,
            response_time_ms=150
        )
        print(f"❤️ Recorded pipeline health: {health_check_id}")

        # Test data quality summary
        quality_summary = db.get_data_quality_summary(run_id=test_run_id)
        print(f"✅ Quality summary: {quality_summary}")

        # Test database statistics
        db_stats = db.get_database_statistics()
        print(f"📈 Database statistics retrieved: {len(db_stats.get('table_statistics', []))} tables")

        # Test retention policies setup
        retention_policies = db.setup_data_retention_policies()
        print(f"📋 Retention policies configured: {len(retention_policies)} tables")

        # Update pipeline status to completed
        db.update_pipeline_run_status(test_run_id, 'success', records_processed=100)
        print(f"✅ Updated pipeline run status to success")

        print("🎉 All enhanced database tests completed successfully")
        return True

    except Exception as e:
        print(f"❌ Enhanced database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

def test_enhanced_features():
    """Test specific enhanced features"""
    try:
        db = DatabaseManager()

        print("🧪 Testing enhanced features...")

        # Test materialized view refresh
        refresh_success = db.refresh_materialized_views()
        print(f"🔄 Materialized views refresh: {'✅' if refresh_success else '❌'}")

        # Test comprehensive cleanup
        cleanup_results = db.cleanup_old_data()
        print(f"🧹 Cleanup completed: {cleanup_results}")

        # Test health summary
        health_summary = db.get_pipeline_health_summary(hours_back=24)
        print(f"❤️ Health summary: {len(health_summary)} components checked")

        return True

    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        return False
    finally:
        db.close()


if __name__ == "__main__":
    test_database_connection()