# -*- coding: utf-8 -*-
"""
UpsertService v2.0 - Unified UPSERT for 4-Table Architecture.

Contract: CTR-L0-UPSERT-002

Routes UPSERT operations to the correct table based on variable frequency:
- Daily variables  → macro_indicators_daily
- Monthly variables → macro_indicators_monthly
- Quarterly variables → macro_indicators_quarterly

Features:
- Automatic routing by frequency from SSOT
- Date normalization (month/quarter start for non-daily)
- FFill limit tracking per table
- Anti-leakage compliance
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# TABLE CONFIGURATION (from CTR-L0-4TABLE-001)
# =============================================================================

TABLE_CONFIG = {
    'daily': {
        'table': 'macro_indicators_daily',
        'date_col': 'fecha',
        'ffill_limit': 5,
        'date_transform': lambda d: d,  # No transform
    },
    'monthly': {
        'table': 'macro_indicators_monthly',
        'date_col': 'fecha',
        'ffill_limit': 35,
        'date_transform': lambda d: d.replace(day=1),  # First day of month
    },
    'quarterly': {
        'table': 'macro_indicators_quarterly',
        'date_col': 'fecha',
        'ffill_limit': 95,
        'date_transform': lambda d: pd.Timestamp(d).to_period('Q').to_timestamp().date(),
    },
}


def _get_variable_frequency(variable_name: str) -> Literal['daily', 'monthly', 'quarterly']:
    """
    Get frequency for a variable from SSOT.

    Falls back to naming convention if SSOT not available.
    """
    # Try SSOT first
    try:
        import sys
        from pathlib import Path

        # Add src to path if needed
        project_root = Path(__file__).parent.parent.parent.parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from data.macro_ssot import MacroSSOT
        ssot = MacroSSOT()
        var_def = ssot.get_variable(variable_name)

        if var_def:
            freq = var_def.identity.frequency
            if freq in ('daily', 'monthly', 'quarterly'):
                return freq
    except ImportError:
        pass

    # Fallback: infer from naming convention
    if '_m_' in variable_name or variable_name.endswith('_m'):
        return 'monthly'
    elif '_q_' in variable_name or variable_name.endswith('_q'):
        return 'quarterly'
    else:
        return 'daily'


class UpsertService:
    """
    Unified UPSERT service for time-series data (DRY).

    Features:
    - Configurable number of records to upsert
    - Automatic conflict resolution
    - Batch execution for performance
    - Support for multiple tables
    - Frequency-based routing (v2.0)
    """

    def __init__(
        self,
        conn,
        table: str,
        date_col: str = 'fecha',
        schema: str = 'public'
    ):
        """
        Initialize UpsertService.

        Args:
            conn: Database connection (psycopg2 or similar)
            table: Target table name
            date_col: Date column for conflict resolution
            schema: Database schema
        """
        self.conn = conn
        self.table = table
        self.date_col = date_col
        self.schema = schema
        self._full_table = f"{schema}.{table}" if schema else table

    def upsert_last_n(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n: int = 5
    ) -> Dict[str, Any]:
        """
        UPSERT the last N records from DataFrame.

        Used for realtime updates to correct recent data.

        Args:
            df: DataFrame with data (must have date_col)
            columns: Data columns to upsert (excluding date)
            n: Number of recent records to upsert

        Returns:
            Dict with statistics: rows_affected, success, error
        """
        if df is None or df.empty:
            return {'rows_affected': 0, 'success': True, 'message': 'Empty DataFrame'}

        # Ensure date column exists
        if self.date_col not in df.columns and df.index.name != self.date_col:
            return {
                'rows_affected': 0,
                'success': False,
                'error': f"Missing date column: {self.date_col}"
            }

        # Reset index if date is in index
        if df.index.name == self.date_col:
            df = df.reset_index()

        # Take last N records
        df_upsert = df.tail(n).copy()

        return self._execute_upsert(df_upsert, columns)

    def upsert_range(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """
        UPSERT all records in DataFrame (for backfill).

        Args:
            df: DataFrame with data
            columns: Data columns to upsert

        Returns:
            Dict with statistics
        """
        if df is None or df.empty:
            return {'rows_affected': 0, 'success': True, 'message': 'Empty DataFrame'}

        # Reset index if date is in index
        if df.index.name == self.date_col:
            df = df.reset_index()

        return self._execute_upsert(df, columns)

    def _execute_upsert(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """Execute the actual UPSERT operation."""
        try:
            # Filter columns to only those present in DataFrame
            available_cols = [c for c in columns if c in df.columns]
            if not available_cols:
                return {
                    'rows_affected': 0,
                    'success': False,
                    'error': 'No matching columns in DataFrame'
                }

            all_cols = [self.date_col] + available_cols

            # Build dynamic query
            placeholders = ', '.join(['%s'] * len(all_cols))
            update_clause = ', '.join([
                f"{col} = EXCLUDED.{col}"
                for col in available_cols
            ])

            query = f"""
                INSERT INTO {self._full_table} ({', '.join(all_cols)})
                VALUES ({placeholders})
                ON CONFLICT ({self.date_col}) DO UPDATE SET
                    {update_clause},
                    updated_at = NOW()
            """

            # Prepare data tuples
            data = []
            for _, row in df.iterrows():
                values = [row[self.date_col]] + [
                    None if pd.isna(row.get(c)) else row.get(c)
                    for c in available_cols
                ]
                data.append(tuple(values))

            # Execute batch
            cur = self.conn.cursor()
            from psycopg2.extras import execute_batch
            execute_batch(cur, query, data, page_size=100)
            self.conn.commit()
            cur.close()

            logger.info(
                "[UpsertService] Upserted %d rows to %s",
                len(data), self._full_table
            )

            return {
                'rows_affected': len(data),
                'success': True,
                'columns': available_cols
            }

        except Exception as e:
            logger.error("[UpsertService] UPSERT failed: %s", e)
            self.conn.rollback()
            return {
                'rows_affected': 0,
                'success': False,
                'error': str(e)
            }

    def get_latest_date(self, column: Optional[str] = None) -> Optional[datetime]:
        """Get the most recent date in the table."""
        try:
            query = f"""
                SELECT MAX({self.date_col})
                FROM {self._full_table}
            """
            if column:
                query += f" WHERE {column} IS NOT NULL"

            cur = self.conn.cursor()
            cur.execute(query)
            result = cur.fetchone()
            cur.close()

            return result[0] if result and result[0] else None

        except Exception as e:
            logger.warning("[UpsertService] Failed to get latest date: %s", e)
            return None

    def get_record_count(self) -> int:
        """Get total record count in table."""
        try:
            query = f"SELECT COUNT(*) FROM {self._full_table}"
            cur = self.conn.cursor()
            cur.execute(query)
            result = cur.fetchone()
            cur.close()
            return result[0] if result else 0
        except Exception:
            return 0


# =============================================================================
# V2.0: FREQUENCY-ROUTED UPSERT SERVICE
# =============================================================================

class FrequencyRoutedUpsertService:
    """
    Upsert service that routes to correct table based on variable frequency.

    Contract: CTR-L0-UPSERT-002

    Usage:
        service = FrequencyRoutedUpsertService(conn)
        result = service.upsert_variable('volt_vix_usa_d_vix', df, n=15)
        # Automatically routes to macro_indicators_daily

        result = service.upsert_variable('polr_fed_funds_usa_m_fedfunds', df, n=15)
        # Automatically routes to macro_indicators_monthly
    """

    def __init__(self, conn, schema: str = 'public'):
        """
        Initialize FrequencyRoutedUpsertService.

        Args:
            conn: Database connection (psycopg2)
            schema: Database schema
        """
        self.conn = conn
        self.schema = schema

        # Initialize UpsertService instances for each table
        self._services: Dict[str, UpsertService] = {}
        for freq, config in TABLE_CONFIG.items():
            self._services[freq] = UpsertService(
                conn=conn,
                table=config['table'],
                date_col=config['date_col'],
                schema=schema
            )

    def upsert_variable(
        self,
        variable_name: str,
        df: pd.DataFrame,
        n: int = 15
    ) -> Dict[str, Any]:
        """
        UPSERT a variable to the correct table based on its frequency.

        Args:
            variable_name: Variable name (e.g., 'volt_vix_usa_d_vix')
            df: DataFrame with date column and variable values
            n: Number of recent records to upsert

        Returns:
            Dict with: rows_affected, success, table, frequency
        """
        if df is None or df.empty:
            return {
                'rows_affected': 0,
                'success': True,
                'message': 'Empty DataFrame',
                'variable': variable_name
            }

        # Determine frequency and target table
        frequency = _get_variable_frequency(variable_name)
        config = TABLE_CONFIG[frequency]
        service = self._services[frequency]

        logger.debug(
            "[FrequencyRouter] %s → %s (%s)",
            variable_name, config['table'], frequency
        )

        # Transform dates for monthly/quarterly
        df = df.copy()
        date_col = config['date_col']

        if df.index.name == date_col or df.index.name == 'fecha':
            df = df.reset_index()

        # Find the date column
        if date_col not in df.columns:
            # Try to find date-like column
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'fecha' in c.lower()]
            if date_cols:
                df = df.rename(columns={date_cols[0]: date_col})
            else:
                return {
                    'rows_affected': 0,
                    'success': False,
                    'error': f"No date column found for {variable_name}"
                }

        # Apply date transformation for monthly/quarterly
        if frequency in ('monthly', 'quarterly'):
            transform = config['date_transform']
            df[date_col] = df[date_col].apply(
                lambda d: transform(pd.to_datetime(d).date()) if pd.notna(d) else None
            )
            # Deduplicate by transformed date (keep last)
            df = df.drop_duplicates(subset=[date_col], keep='last')

        # Execute upsert
        result = service.upsert_last_n(df, [variable_name], n=n)

        # Enrich result
        result['variable'] = variable_name
        result['frequency'] = frequency
        result['table'] = config['table']
        result['ffill_limit'] = config['ffill_limit']

        return result

    def upsert_batch(
        self,
        data: Dict[str, pd.DataFrame],
        n: int = 15
    ) -> Dict[str, Any]:
        """
        UPSERT multiple variables, routing each to correct table.

        Args:
            data: Dict of variable_name -> DataFrame
            n: Number of recent records per variable

        Returns:
            Dict with aggregated statistics per table
        """
        results = {
            'daily': {'success': 0, 'failed': 0, 'rows': 0},
            'monthly': {'success': 0, 'failed': 0, 'rows': 0},
            'quarterly': {'success': 0, 'failed': 0, 'rows': 0},
            'details': []
        }

        for variable_name, df in data.items():
            result = self.upsert_variable(variable_name, df, n=n)

            freq = result.get('frequency', 'daily')
            if result.get('success'):
                results[freq]['success'] += 1
                results[freq]['rows'] += result.get('rows_affected', 0)
            else:
                results[freq]['failed'] += 1

            results['details'].append(result)

        # Summary logging
        for freq in ('daily', 'monthly', 'quarterly'):
            stats = results[freq]
            if stats['success'] + stats['failed'] > 0:
                logger.info(
                    "[FrequencyRouter] %s: %d success, %d failed, %d rows",
                    freq, stats['success'], stats['failed'], stats['rows']
                )

        return results

    def get_table_for_variable(self, variable_name: str) -> str:
        """Get the target table name for a variable."""
        frequency = _get_variable_frequency(variable_name)
        return TABLE_CONFIG[frequency]['table']

    def get_ffill_limit(self, variable_name: str) -> int:
        """Get FFill limit for a variable based on its frequency."""
        frequency = _get_variable_frequency(variable_name)
        return TABLE_CONFIG[frequency]['ffill_limit']
