# -*- coding: utf-8 -*-
"""
SeedService - Restore from backup and align data to current date.

Workflow:
1. Detect latest backup (CSV, CSV.GZ, Parquet)
2. Restore backup with UPSERT (preserves newer data)
3. Detect gap between backup and today
4. Extract delta from all sources
5. UPSERT delta to align with current date

Usage:
    seed_service = SeedService(registry)
    stats = seed_service.restore_and_align(backup_path)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from extractors.registry import ExtractorRegistry
from .upsert_service import UpsertService

logger = logging.getLogger(__name__)


class SeedService:
    """
    Service for restoring data from backups and aligning to current date.

    This service ensures data continuity when:
    - Starting fresh from a backup
    - Recovering from data loss
    - Bootstrapping a new environment
    """

    def __init__(
        self,
        registry: ExtractorRegistry = None,
        conn=None,
        table: str = 'macro_indicators_daily'
    ):
        """
        Initialize SeedService.

        Args:
            registry: ExtractorRegistry instance (or creates new)
            conn: Database connection
            table: Target table name
        """
        self.registry = registry or ExtractorRegistry()
        self.conn = conn
        self.table = table
        self._upsert = None

    def _get_upsert_service(self) -> UpsertService:
        """Get or create UpsertService."""
        if self._upsert is None:
            self._upsert = UpsertService(self.conn, self.table)
        return self._upsert

    def restore_and_align(
        self,
        backup_path: Path,
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Restore from backup and align data to current date.

        Args:
            backup_path: Path to backup file
            variables: Optional list of variables to align (default: all)

        Returns:
            Dict with statistics: restored, aligned, errors
        """
        stats = {
            'restored': 0,
            'aligned': 0,
            'variables_aligned': [],
            'errors': [],
            'backup_date': None,
            'current_date': datetime.now().date()
        }

        # 1. Restore backup
        if backup_path.exists():
            try:
                df_backup = self._load_backup(backup_path)
                stats['backup_date'] = df_backup['fecha'].max().date() if 'fecha' in df_backup.columns else None

                # UPSERT backup data
                upsert = self._get_upsert_service()
                columns = [c for c in df_backup.columns if c != 'fecha']
                result = upsert.upsert_range(df_backup, columns)
                stats['restored'] = result.get('rows_affected', 0)

                logger.info(
                    "[SeedService] Restored %d rows from backup dated %s",
                    stats['restored'], stats['backup_date']
                )
            except Exception as e:
                stats['errors'].append(f"Backup restore failed: {e}")
                logger.error("[SeedService] Backup restore failed: %s", e)
                stats['backup_date'] = datetime(2020, 1, 1).date()
        else:
            logger.warning("[SeedService] No backup found at %s", backup_path)
            stats['backup_date'] = datetime(2020, 1, 1).date()

        # 2. Detect gap and align
        gap_start = datetime.combine(
            stats['backup_date'] + timedelta(days=1),
            datetime.min.time()
        ) if stats['backup_date'] else datetime(2020, 1, 1)

        today = datetime.now()

        if gap_start.date() < today.date():
            logger.info(
                "[SeedService] Aligning gap: %s -> %s",
                gap_start.strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d")
            )

            # Get variables to align
            vars_to_align = variables or self.registry.get_all_variables()

            for variable in vars_to_align:
                try:
                    result = self.registry.extract_variable(
                        variable,
                        start_date=gap_start,
                        end_date=today
                    )

                    if result.success and result.data is not None and not result.data.empty:
                        upsert = self._get_upsert_service()
                        upsert_result = upsert.upsert_range(result.data, [variable])
                        if upsert_result.get('success'):
                            stats['aligned'] += upsert_result.get('rows_affected', 0)
                            stats['variables_aligned'].append(variable)
                            logger.info(
                                "[SeedService] Aligned %s: %d rows",
                                variable, upsert_result.get('rows_affected', 0)
                            )
                    else:
                        stats['errors'].append(f"{variable}: {result.error or 'No data'}")

                except Exception as e:
                    stats['errors'].append(f"{variable}: {e}")
                    logger.warning("[SeedService] Failed to align %s: %s", variable, e)

        return stats

    def full_extraction(
        self,
        start_date: str = '2020-01-01',
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform full extraction from start_date to today.

        Used when no backup is available.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            variables: Optional list of variables (default: all)

        Returns:
            Dict with statistics
        """
        stats = {
            'extracted': 0,
            'variables': [],
            'errors': []
        }

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.now()

        vars_to_extract = variables or self.registry.get_all_variables()

        for variable in vars_to_extract:
            try:
                result = self.registry.extract_variable(
                    variable,
                    start_date=start_dt,
                    end_date=end_dt
                )

                if result.success and result.data is not None and not result.data.empty:
                    upsert = self._get_upsert_service()
                    upsert_result = upsert.upsert_range(result.data, [variable])
                    if upsert_result.get('success'):
                        stats['extracted'] += upsert_result.get('rows_affected', 0)
                        stats['variables'].append(variable)
                else:
                    stats['errors'].append(f"{variable}: {result.error or 'No data'}")

            except Exception as e:
                stats['errors'].append(f"{variable}: {e}")
                logger.warning("[SeedService] Failed to extract %s: %s", variable, e)

        return stats

    def _load_backup(self, path: Path) -> pd.DataFrame:
        """Load backup file (CSV, CSV.GZ, or Parquet)."""
        suffix = path.suffix.lower()

        if suffix == '.parquet':
            return pd.read_parquet(path)
        elif suffix == '.gz':
            return pd.read_csv(path, compression='gzip', parse_dates=['fecha'])
        elif suffix == '.csv':
            return pd.read_csv(path, parse_dates=['fecha'])
        else:
            raise ValueError(f"Unsupported backup format: {suffix}")

    def find_latest_backup(self, backup_dir: Path, pattern: str = 'macro_*.csv.gz') -> Optional[Path]:
        """Find the most recent backup file matching pattern."""
        backups = list(backup_dir.glob(pattern))
        if not backups:
            return None
        return max(backups, key=lambda p: p.stat().st_mtime)

    def detect_gaps(self, variable: str) -> List[tuple]:
        """
        Detect gaps in data for a variable.

        Returns:
            List of (start_date, end_date) tuples for each gap
        """
        # This would query the database to find missing dates
        # Simplified implementation
        return []
