"""
DWH Helper Module
=================
Helper functions for inserting and managing data in the Kimball Data Warehouse.

This module provides:
- Dimension management (SCD1 and SCD2)
- Fact table insertions
- Audit logging
- Data validation
- Error handling

Usage in DAGs:
    from utils.dwh_helper import DWHHelper

    dwh = DWHHelper(conn)

    # Get or insert dimension
    symbol_id = dwh.get_or_insert_dim_symbol('USD/COP')

    # Insert fact
    dwh.insert_fact_l0_acquisition(run_id, metrics_dict)
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as Connection
from psycopg2.extras import execute_batch, RealDictCursor

logger = logging.getLogger(__name__)


class DWHHelper:
    """Helper class for Data Warehouse operations."""

    def __init__(self, conn: Connection):
        """
        Initialize DWH Helper.

        Args:
            conn: psycopg2 connection object
        """
        self.conn = conn
        self.cur = conn.cursor()

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    @staticmethod
    def generate_sha256(data: Union[str, dict]) -> str:
        """
        Generate SHA256 hash for data lineage tracking.

        Args:
            data: String or dict to hash

        Returns:
            SHA256 hash as hex string
        """
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def utc_to_cot(ts_utc: datetime) -> datetime:
        """Convert UTC timestamp to Colombian time (COT)."""
        # Note: This is a simplified conversion. In production, use pytz
        from datetime import timedelta
        return ts_utc - timedelta(hours=5)

    def log_operation(
        self,
        schema_name: str,
        table_name: str,
        operation: str,
        rows_affected: int,
        dag_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> None:
        """
        Log DWH operation to audit log.

        Args:
            schema_name: Schema name (e.g., 'dw')
            table_name: Table name (e.g., 'fact_l0_acquisition')
            operation: Operation type ('INSERT', 'UPDATE', 'DELETE')
            rows_affected: Number of rows affected
            dag_id: Airflow DAG ID
            run_id: Airflow run ID
            task_id: Airflow task ID
        """
        query = """
            INSERT INTO dw.audit_log (
                schema_name, table_name, operation, rows_affected,
                dag_id, run_id, task_id, execution_date
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """

        try:
            self.cur.execute(query, (
                schema_name, table_name, operation, rows_affected,
                dag_id, run_id, task_id
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
            self.conn.rollback()

    # ========================================================================
    # DIMENSION MANAGEMENT (SCD1)
    # ========================================================================

    def get_or_insert_dim_symbol(
        self,
        symbol_code: str,
        base_currency: str = None,
        quote_currency: str = None,
        symbol_type: str = 'forex',
        exchange: str = None
    ) -> int:
        """
        Get or insert symbol dimension (SCD Type 1).

        Args:
            symbol_code: Symbol code (e.g., 'USD/COP')
            base_currency: Base currency (e.g., 'USD')
            quote_currency: Quote currency (e.g., 'COP')
            symbol_type: Symbol type ('forex', 'crypto', 'stock')
            exchange: Exchange name

        Returns:
            symbol_id (int)
        """
        # Try to get existing
        self.cur.execute(
            "SELECT symbol_id FROM dw.dim_symbol WHERE symbol_code = %s",
            (symbol_code,)
        )
        result = self.cur.fetchone()

        if result:
            return result[0]

        # Parse symbol if currencies not provided
        if not base_currency or not quote_currency:
            parts = symbol_code.split('/')
            if len(parts) == 2:
                base_currency, quote_currency = parts
            else:
                base_currency = symbol_code[:3]
                quote_currency = symbol_code[3:]

        # Insert new
        self.cur.execute("""
            INSERT INTO dw.dim_symbol (
                symbol_code, base_currency, quote_currency, symbol_type, exchange
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (symbol_code) DO UPDATE SET updated_at = NOW()
            RETURNING symbol_id
        """, (symbol_code, base_currency, quote_currency, symbol_type, exchange))

        symbol_id = self.cur.fetchone()[0]
        self.conn.commit()

        logger.info(f"Inserted/retrieved dim_symbol: {symbol_code} -> {symbol_id}")
        return symbol_id

    def get_or_insert_dim_source(
        self,
        source_name: str,
        source_type: str = 'api',
        api_endpoint: Optional[str] = None,
        cost_per_call: float = 0.0,
        rate_limit_per_min: Optional[int] = None
    ) -> int:
        """
        Get or insert source dimension (SCD Type 1).

        Args:
            source_name: Source name (e.g., 'twelvedata')
            source_type: Source type ('api', 'file', 'stream')
            api_endpoint: API endpoint URL
            cost_per_call: Cost per API call
            rate_limit_per_min: Rate limit per minute

        Returns:
            source_id (int)
        """
        # Try to get existing
        self.cur.execute(
            "SELECT source_id FROM dw.dim_source WHERE source_name = %s",
            (source_name,)
        )
        result = self.cur.fetchone()

        if result:
            return result[0]

        # Insert new
        self.cur.execute("""
            INSERT INTO dw.dim_source (
                source_name, source_type, api_endpoint, cost_per_call, rate_limit_per_min
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source_name) DO UPDATE SET updated_at = NOW()
            RETURNING source_id
        """, (source_name, source_type, api_endpoint, cost_per_call, rate_limit_per_min))

        source_id = self.cur.fetchone()[0]
        self.conn.commit()

        logger.info(f"Inserted/retrieved dim_source: {source_name} -> {source_id}")
        return source_id

    def get_time_id(self, ts_utc: datetime) -> Optional[int]:
        """
        Get time_id from dim_time_5m for a given UTC timestamp.

        Args:
            ts_utc: UTC timestamp

        Returns:
            time_id (int) or None if not found
        """
        self.cur.execute(
            "SELECT time_id FROM dw.dim_time_5m WHERE ts_utc = %s",
            (ts_utc,)
        )
        result = self.cur.fetchone()
        return result[0] if result else None

    def get_or_insert_dim_feature(
        self,
        feature_name: str,
        feature_type: str,
        tier: int = 1,
        lag_bars: int = 0,
        normalization_method: str = 'median_mad'
    ) -> int:
        """
        Get or insert feature dimension (SCD Type 1).

        Args:
            feature_name: Feature name (e.g., 'obs_00')
            feature_type: Feature type ('momentum', 'volatility', 'shape', etc.)
            tier: Feature tier (1 or 2)
            lag_bars: Anti-leakage lag in bars
            normalization_method: Normalization method

        Returns:
            feature_id (int)
        """
        # Try to get existing
        self.cur.execute(
            "SELECT feature_id FROM dw.dim_feature WHERE feature_name = %s",
            (feature_name,)
        )
        result = self.cur.fetchone()

        if result:
            return result[0]

        # Insert new
        self.cur.execute("""
            INSERT INTO dw.dim_feature (
                feature_name, feature_type, tier, lag_bars, normalization_method
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (feature_name) DO UPDATE SET updated_at = NOW()
            RETURNING feature_id
        """, (feature_name, feature_type, tier, lag_bars, normalization_method))

        feature_id = self.cur.fetchone()[0]
        self.conn.commit()

        logger.info(f"Inserted/retrieved dim_feature: {feature_name} -> {feature_id}")
        return feature_id

    def get_or_insert_dim_indicator(
        self,
        indicator_name: str,
        indicator_family: str,
        params: Optional[dict] = None
    ) -> int:
        """
        Get or insert indicator dimension (SCD Type 1).

        Args:
            indicator_name: Indicator name (e.g., 'RSI', 'MACD')
            indicator_family: Indicator family ('momentum', 'trend', 'volatility')
            params: Parameters as dict

        Returns:
            indicator_id (int)
        """
        # Try to get existing
        self.cur.execute(
            "SELECT indicator_id FROM dw.dim_indicator WHERE indicator_name = %s",
            (indicator_name,)
        )
        result = self.cur.fetchone()

        if result:
            return result[0]

        # Insert new
        params_json = json.dumps(params) if params else None
        self.cur.execute("""
            INSERT INTO dw.dim_indicator (
                indicator_name, indicator_family, params
            ) VALUES (%s, %s, %s)
            ON CONFLICT (indicator_name) DO UPDATE SET params = EXCLUDED.params
            RETURNING indicator_id
        """, (indicator_name, indicator_family, params_json))

        indicator_id = self.cur.fetchone()[0]
        self.conn.commit()

        logger.info(f"Inserted/retrieved dim_indicator: {indicator_name} -> {indicator_id}")
        return indicator_id

    # ========================================================================
    # DIMENSION MANAGEMENT (SCD2)
    # ========================================================================

    def get_or_insert_dim_model_scd2(
        self,
        model_id: str,
        model_name: str,
        algorithm: str,
        version: str,
        architecture: Optional[str] = None,
        framework: Optional[str] = None,
        hyperparams: Optional[dict] = None,
        sha256_hash: Optional[str] = None,
        is_production: bool = False
    ) -> int:
        """
        Get or insert model dimension (SCD Type 2).

        For SCD2, we:
        1. Check if a current version exists with same hash
        2. If hash differs, expire old version and insert new

        Args:
            model_id: Model ID (natural key, e.g., 'rl_ppo_v1.2')
            model_name: Model name
            algorithm: Algorithm (e.g., 'PPO', 'DQN')
            version: Version string
            architecture: Architecture description
            framework: Framework name
            hyperparams: Hyperparameters as dict
            sha256_hash: Model artifact hash
            is_production: Whether this is the production model

        Returns:
            model_sk (int) - Surrogate key
        """
        # Try to get current version
        self.cur.execute("""
            SELECT model_sk, sha256_hash
            FROM dw.dim_model
            WHERE model_id = %s AND is_current = TRUE
        """, (model_id,))
        result = self.cur.fetchone()

        if result:
            existing_sk, existing_hash = result
            # If hash matches, return existing
            if existing_hash == sha256_hash:
                return existing_sk

            # Hash differs - expire old version
            self.cur.execute("""
                UPDATE dw.dim_model
                SET valid_to = NOW(), is_current = FALSE
                WHERE model_sk = %s
            """, (existing_sk,))

        # Insert new version
        hyperparams_json = json.dumps(hyperparams) if hyperparams else None
        self.cur.execute("""
            INSERT INTO dw.dim_model (
                model_id, model_name, algorithm, architecture, framework,
                version, hyperparams, sha256_hash, is_production,
                valid_from, is_current
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), TRUE)
            RETURNING model_sk
        """, (
            model_id, model_name, algorithm, architecture, framework,
            version, hyperparams_json, sha256_hash, is_production
        ))

        model_sk = self.cur.fetchone()[0]
        self.conn.commit()

        logger.info(f"Inserted new version of dim_model: {model_id} -> SK {model_sk}")
        return model_sk

    # ========================================================================
    # FACT TABLE INSERTIONS - L0
    # ========================================================================

    def insert_fact_bars_bulk(
        self,
        bars: List[Dict[str, Any]],
        symbol_id: int,
        source_id: int
    ) -> int:
        """
        Bulk insert OHLCV bars into fact_bar_5m.

        Args:
            bars: List of bar dicts with keys: ts_utc, open, high, low, close, volume
            symbol_id: Symbol ID from dim_symbol
            source_id: Source ID from dim_source

        Returns:
            Number of rows inserted
        """
        if not bars:
            return 0

        # Prepare data with time_id lookup
        records = []
        for bar in bars:
            time_id = self.get_time_id(bar['ts_utc'])
            if not time_id:
                logger.warning(f"No time_id found for {bar['ts_utc']}, skipping")
                continue

            records.append((
                symbol_id,
                time_id,
                bar['ts_utc'],
                bar['open'],
                bar['high'],
                bar['low'],
                bar['close'],
                bar.get('volume', 0),
                source_id
            ))

        if not records:
            return 0

        # Bulk insert with ON CONFLICT DO NOTHING (idempotent)
        query = """
            INSERT INTO dw.fact_bar_5m (
                symbol_id, time_id, ts_utc, open, high, low, close, volume, source_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol_id, ts_utc) DO NOTHING
        """

        execute_batch(self.cur, query, records, page_size=1000)
        self.conn.commit()

        rows_inserted = len(records)
        logger.info(f"Inserted {rows_inserted} bars into fact_bar_5m")

        return rows_inserted

    def insert_fact_l0_acquisition(
        self,
        run_id: str,
        symbol_id: int,
        source_id: int,
        execution_date: datetime,
        metrics: Dict[str, Any],
        dag_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> None:
        """
        Insert L0 acquisition metrics into fact_l0_acquisition.

        Args:
            run_id: Unique run ID
            symbol_id: Symbol ID
            source_id: Source ID
            execution_date: Execution date
            metrics: Dict with keys: fetch_mode, date_range_start, date_range_end,
                     rows_fetched, rows_inserted, stale_rate_pct, coverage_pct, etc.
            dag_id: Airflow DAG ID
            task_id: Airflow task ID
        """
        query = """
            INSERT INTO dw.fact_l0_acquisition (
                run_id, symbol_id, source_id, execution_date,
                fetch_mode, date_range_start, date_range_end,
                rows_fetched, rows_inserted, rows_duplicated, rows_rejected,
                stale_rate_pct, coverage_pct, gaps_detected,
                duration_sec, api_calls_count, api_cost_usd,
                quality_passed, minio_manifest_path, dag_id, task_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (run_id) DO UPDATE SET
                rows_fetched = EXCLUDED.rows_fetched,
                rows_inserted = EXCLUDED.rows_inserted,
                quality_passed = EXCLUDED.quality_passed
        """

        self.cur.execute(query, (
            run_id,
            symbol_id,
            source_id,
            execution_date,
            metrics.get('fetch_mode', 'unknown'),
            metrics.get('date_range_start'),
            metrics.get('date_range_end'),
            metrics.get('rows_fetched', 0),
            metrics.get('rows_inserted', 0),
            metrics.get('rows_duplicated', 0),
            metrics.get('rows_rejected', 0),
            metrics.get('stale_rate_pct', 0.0),
            metrics.get('coverage_pct', 0.0),
            metrics.get('gaps_detected', 0),
            metrics.get('duration_sec', 0),
            metrics.get('api_calls_count', 0),
            metrics.get('api_cost_usd', 0.0),
            metrics.get('quality_passed', False),
            metrics.get('minio_manifest_path'),
            dag_id,
            task_id
        ))

        self.conn.commit()
        self.log_operation('dw', 'fact_l0_acquisition', 'INSERT', 1, dag_id, run_id, task_id)

        logger.info(f"Inserted fact_l0_acquisition: {run_id}")

    # ========================================================================
    # FACT TABLE INSERTIONS - L1
    # ========================================================================

    def insert_fact_l1_quality(
        self,
        date_cot: datetime,
        symbol_id: int,
        metrics: Dict[str, Any],
        run_id: Optional[str] = None,
        dag_id: Optional[str] = None
    ) -> None:
        """
        Insert L1 quality metrics into fact_l1_quality.

        Args:
            date_cot: Date in Colombian time
            symbol_id: Symbol ID
            metrics: Dict with quality metrics
            run_id: Run ID
            dag_id: DAG ID
        """
        query = """
            INSERT INTO dw.fact_l1_quality (
                date_cot, symbol_id,
                total_episodes, accepted_episodes, rejected_episodes,
                grid_300s_ok, repeated_ohlc_rate_pct, gaps_over_1_interval,
                coverage_pct, status_passed,
                run_id, minio_manifest_path, dag_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date_cot, symbol_id) DO UPDATE SET
                total_episodes = EXCLUDED.total_episodes,
                accepted_episodes = EXCLUDED.accepted_episodes,
                status_passed = EXCLUDED.status_passed
        """

        self.cur.execute(query, (
            date_cot,
            symbol_id,
            metrics.get('total_episodes', 0),
            metrics.get('accepted_episodes', 0),
            metrics.get('rejected_episodes', 0),
            metrics.get('grid_300s_ok', False),
            metrics.get('repeated_ohlc_rate_pct', 0.0),
            metrics.get('gaps_over_1_interval', 0),
            metrics.get('coverage_pct', 0.0),
            metrics.get('status_passed', False),
            run_id,
            metrics.get('minio_manifest_path'),
            dag_id
        ))

        self.conn.commit()
        self.log_operation('dw', 'fact_l1_quality', 'INSERT', 1, dag_id, run_id, None)

        logger.info(f"Inserted fact_l1_quality for {date_cot}")

    # ========================================================================
    # FACT TABLE INSERTIONS - L5 (Serving)
    # ========================================================================

    def insert_fact_signal_5m_bulk(
        self,
        signals: List[Dict[str, Any]],
        model_sk: int,
        symbol_id: int,
        dag_id: Optional[str] = None
    ) -> int:
        """
        Bulk insert trading signals into fact_signal_5m.

        Args:
            signals: List of signal dicts with keys: ts_utc, action, confidence, q_hold, q_buy, q_sell, etc.
            model_sk: Model surrogate key
            symbol_id: Symbol ID
            dag_id: DAG ID

        Returns:
            Number of rows inserted
        """
        if not signals:
            return 0

        # Prepare data with time_id lookup
        records = []
        for signal in signals:
            time_id = self.get_time_id(signal['ts_utc'])
            if not time_id:
                logger.warning(f"No time_id found for {signal['ts_utc']}, skipping")
                continue

            records.append((
                model_sk,
                symbol_id,
                time_id,
                signal['ts_utc'],
                signal['action'],
                signal.get('confidence'),
                signal.get('q_hold'),
                signal.get('q_buy'),
                signal.get('q_sell'),
                signal.get('epsilon'),
                signal.get('reason_code'),
                signal.get('latency_ms')
            ))

        if not records:
            return 0

        # Bulk insert
        query = """
            INSERT INTO dw.fact_signal_5m (
                model_sk, symbol_id, time_id, ts_utc,
                action, confidence, q_hold, q_buy, q_sell,
                epsilon, reason_code, latency_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_sk, symbol_id, ts_utc) DO NOTHING
        """

        execute_batch(self.cur, query, records, page_size=1000)
        self.conn.commit()

        rows_inserted = len(records)
        logger.info(f"Inserted {rows_inserted} signals into fact_signal_5m")

        if dag_id:
            self.log_operation('dw', 'fact_signal_5m', 'INSERT', rows_inserted, dag_id, None, None)

        return rows_inserted

    def insert_fact_inference_latency(
        self,
        model_sk: int,
        date_cot: datetime,
        metrics: Dict[str, Any],
        dag_id: Optional[str] = None
    ) -> None:
        """
        Insert inference latency metrics into fact_inference_latency.

        Args:
            model_sk: Model surrogate key
            date_cot: Date in Colombian time
            metrics: Dict with latency metrics
            dag_id: DAG ID
        """
        query = """
            INSERT INTO dw.fact_inference_latency (
                model_sk, date_cot,
                latency_p50_ms, latency_p95_ms, latency_p99_ms,
                e2e_latency_p99_ms, throughput_eps, inference_count,
                dag_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_sk, date_cot) DO UPDATE SET
                latency_p50_ms = EXCLUDED.latency_p50_ms,
                latency_p95_ms = EXCLUDED.latency_p95_ms,
                latency_p99_ms = EXCLUDED.latency_p99_ms,
                e2e_latency_p99_ms = EXCLUDED.e2e_latency_p99_ms,
                throughput_eps = EXCLUDED.throughput_eps,
                inference_count = EXCLUDED.inference_count
        """

        self.cur.execute(query, (
            model_sk,
            date_cot,
            metrics.get('latency_p50_ms'),
            metrics.get('latency_p95_ms'),
            metrics.get('latency_p99_ms'),
            metrics.get('e2e_latency_p99_ms'),
            metrics.get('throughput_eps'),
            metrics.get('inference_count', 0),
            dag_id
        ))

        self.conn.commit()
        self.log_operation('dw', 'fact_inference_latency', 'INSERT', 1, dag_id, None, None)

        logger.info(f"Inserted fact_inference_latency for model_sk={model_sk}, date={date_cot}")

    # ========================================================================
    # FACT TABLE INSERTIONS - L6 (Backtesting)
    # ========================================================================

    def insert_dim_backtest_run(
        self,
        run_id: str,
        model_sk: int,
        symbol_id: int,
        split: str,
        date_range_start: datetime,
        date_range_end: datetime,
        execution_date: datetime,
        initial_capital: float = 100000.0,
        features_sha256: Optional[str] = None,
        dataset_sha256: Optional[str] = None,
        model_sha256: Optional[str] = None,
        minio_manifest_path: Optional[str] = None
    ) -> int:
        """
        Insert backtest run dimension.

        Args:
            run_id: Unique run ID
            model_sk: Model surrogate key
            symbol_id: Symbol ID
            split: Split ('train', 'val', 'test')
            date_range_start: Start date of backtest
            date_range_end: End date of backtest
            execution_date: Execution date
            initial_capital: Initial capital
            features_sha256: Features hash
            dataset_sha256: Dataset hash
            model_sha256: Model hash
            minio_manifest_path: MinIO manifest path

        Returns:
            run_sk (int) - Surrogate key
        """
        query = """
            INSERT INTO dw.dim_backtest_run (
                run_id, model_sk, symbol_id, split,
                date_range_start, date_range_end, initial_capital,
                features_sha256, dataset_sha256, model_sha256,
                execution_date, minio_manifest_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                execution_date = EXCLUDED.execution_date
            RETURNING run_sk
        """

        self.cur.execute(query, (
            run_id,
            model_sk,
            symbol_id,
            split,
            date_range_start,
            date_range_end,
            initial_capital,
            features_sha256,
            dataset_sha256,
            model_sha256,
            execution_date,
            minio_manifest_path
        ))

        run_sk = self.cur.fetchone()[0]
        self.conn.commit()

        logger.info(f"Inserted dim_backtest_run: {run_id} -> SK {run_sk}")
        return run_sk

    def insert_fact_trades_bulk(
        self,
        trades: List[Dict[str, Any]],
        run_sk: int,
        dag_id: Optional[str] = None
    ) -> int:
        """
        Bulk insert trades into fact_trade.

        Args:
            trades: List of trade dicts
            run_sk: Backtest run surrogate key
            dag_id: DAG ID

        Returns:
            Number of rows inserted
        """
        if not trades:
            return 0

        records = []
        for trade in trades:
            records.append((
                run_sk,
                trade['trade_id'],
                trade['side'],
                trade['entry_time'],
                trade['exit_time'],
                trade.get('duration_bars', 0),
                trade['entry_px'],
                trade['exit_px'],
                trade.get('quantity', 1.0),
                trade['pnl'],
                trade.get('pnl_pct'),
                trade.get('pnl_bps'),
                trade.get('costs', 0.0),
                trade.get('reason_entry'),
                trade.get('reason_exit')
            ))

        query = """
            INSERT INTO dw.fact_trade (
                run_sk, trade_id, side, entry_time, exit_time, duration_bars,
                entry_px, exit_px, quantity, pnl, pnl_pct, pnl_bps, costs,
                reason_entry, reason_exit
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_sk, trade_id) DO NOTHING
        """

        execute_batch(self.cur, query, records, page_size=1000)
        self.conn.commit()

        rows_inserted = len(records)
        logger.info(f"Inserted {rows_inserted} trades into fact_trade")

        if dag_id:
            self.log_operation('dw', 'fact_trade', 'INSERT', rows_inserted, dag_id, None, None)

        return rows_inserted

    def insert_fact_perf_daily_bulk(
        self,
        daily_perfs: List[Dict[str, Any]],
        run_sk: int,
        dag_id: Optional[str] = None
    ) -> int:
        """
        Bulk insert daily performance into fact_perf_daily.

        Args:
            daily_perfs: List of daily performance dicts
            run_sk: Backtest run surrogate key
            dag_id: DAG ID

        Returns:
            Number of rows inserted
        """
        if not daily_perfs:
            return 0

        records = []
        for perf in daily_perfs:
            records.append((
                run_sk,
                perf['date_cot'],
                perf.get('daily_return'),
                perf.get('cumulative_return'),
                perf.get('equity'),
                perf.get('trades_count', 0),
                perf.get('wins_count', 0),
                perf.get('losses_count', 0),
                perf.get('daily_pnl'),
                perf.get('daily_costs', 0.0),
                perf.get('drawdown_pct')
            ))

        query = """
            INSERT INTO dw.fact_perf_daily (
                run_sk, date_cot, daily_return, cumulative_return, equity,
                trades_count, wins_count, losses_count, daily_pnl, daily_costs,
                drawdown_pct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_sk, date_cot) DO NOTHING
        """

        execute_batch(self.cur, query, records, page_size=1000)
        self.conn.commit()

        rows_inserted = len(records)
        logger.info(f"Inserted {rows_inserted} daily performance records into fact_perf_daily")

        if dag_id:
            self.log_operation('dw', 'fact_perf_daily', 'INSERT', rows_inserted, dag_id, None, None)

        return rows_inserted

    def insert_fact_perf_summary(
        self,
        run_sk: int,
        split: str,
        metrics: Dict[str, Any],
        dag_id: Optional[str] = None
    ) -> None:
        """
        Insert backtest summary metrics into fact_perf_summary.

        Args:
            run_sk: Backtest run surrogate key
            split: Split ('train', 'val', 'test')
            metrics: Dict with performance metrics
            dag_id: DAG ID
        """
        query = """
            INSERT INTO dw.fact_perf_summary (
                run_sk, split,
                total_return, cagr, volatility,
                sharpe_ratio, sortino_ratio, calmar_ratio,
                max_drawdown, max_drawdown_duration_days,
                total_trades, win_rate, profit_factor,
                avg_win, avg_loss, total_costs, costs_pct_of_pnl,
                is_production_ready, dag_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_sk, split) DO UPDATE SET
                total_return = EXCLUDED.total_return,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                sortino_ratio = EXCLUDED.sortino_ratio,
                max_drawdown = EXCLUDED.max_drawdown,
                win_rate = EXCLUDED.win_rate
        """

        self.cur.execute(query, (
            run_sk,
            split,
            metrics.get('total_return'),
            metrics.get('cagr'),
            metrics.get('volatility'),
            metrics.get('sharpe_ratio'),
            metrics.get('sortino_ratio'),
            metrics.get('calmar_ratio'),
            metrics.get('max_drawdown'),
            metrics.get('max_drawdown_duration_days'),
            metrics.get('total_trades', 0),
            metrics.get('win_rate'),
            metrics.get('profit_factor'),
            metrics.get('avg_win'),
            metrics.get('avg_loss'),
            metrics.get('total_costs', 0.0),
            metrics.get('costs_pct_of_pnl'),
            metrics.get('is_production_ready', False),
            dag_id
        ))

        self.conn.commit()
        self.log_operation('dw', 'fact_perf_summary', 'INSERT', 1, dag_id, None, None)

        logger.info(f"Inserted fact_perf_summary for run_sk={run_sk}, split={split}")

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def close(self):
        """Close cursor and connection."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ============================================================================
# STANDALONE UTILITY FUNCTIONS
# ============================================================================

def get_dwh_connection() -> Connection:
    """
    Get PostgreSQL connection for DWH operations.

    Returns:
        psycopg2 connection
    """
    import os

    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'usdcop_trading'),
        user=os.getenv('POSTGRES_USER', 'admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'admin123')
    )

    return conn


def validate_fact_data(fact_name: str, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate fact data before insertion.

    Args:
        fact_name: Name of fact table (e.g., 'fact_l0_acquisition')
        data: Data dict to validate

    Returns:
        (is_valid, error_message)
    """
    # Add validation logic per fact table
    required_fields = {
        'fact_l0_acquisition': ['run_id', 'rows_fetched', 'rows_inserted', 'quality_passed'],
        'fact_l1_quality': ['date_cot', 'total_episodes', 'status_passed'],
        # Add more as needed
    }

    if fact_name not in required_fields:
        return True, None  # No validation defined

    missing = [f for f in required_fields[fact_name] if f not in data]
    if missing:
        return False, f"Missing required fields: {missing}"

    return True, None
