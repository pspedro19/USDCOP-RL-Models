"""
Macro Merge Service
===================

Handles merging of extracted macro data and database operations.
Implements UPSERT logic and forward-fill functionality.

Contract: CTR-L0-MERGE-001

Operations:
    - merge_and_upsert: Combine all source data and write to database
    - forward_fill: Apply bounded forward-fill for missing values
    - generate_readiness_report: Create daily data readiness report

Version: 1.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

from utils.dag_common import get_db_connection

# Import contracts
from contracts.l0_data_contracts import (
    L0XComKeys,
    MACRO_INDICATOR_REGISTRY,
    FFillConfig,
    DEFAULT_FFILL_CONFIG,
    IndicatorReadiness,
    IndicatorReadinessStatus,
    DailyDataReadinessReport,
    DataSourceType,
)

logger = logging.getLogger(__name__)


class MacroMergeService:
    """
    Service for merging extracted macro data and database operations.

    Handles combining data from all sources, upserting to database,
    and applying forward-fill for missing values.
    """

    # Source to column mapping for release_date determination
    SOURCE_RELEASE_OFFSETS = {
        'fred': 1,
        'twelvedata': 0,
        'investing': 0,
        'banrep': 0,
        'bcrp': 0,
        'fedesarrollo': 15,
        'dane': 45,
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize merge service.

        Args:
            config_path: Path to configuration (for release offsets)
        """
        self.config_path = config_path

    def merge_and_upsert(self, **context) -> Dict[str, Any]:
        """
        Merge all sources and upsert to database with release_date tracking.

        Pulls data from XCom for all sources, merges by date,
        and performs UPSERT to macro_indicators_daily table.

        Args:
            **context: Airflow context (includes 'ti' for XCom)

        Returns:
            Dictionary with merge statistics
        """
        ti = context.get('ti')
        if not ti:
            logger.error("No task instance in context")
            return {'status': 'error', 'message': 'No task instance'}

        # Pull data from all sources via XCom
        source_data = self._pull_all_xcom_data(ti)

        # Merge all data with source tracking
        all_data = self._merge_source_data(source_data)

        if not all_data:
            logger.warning("No data to upsert")
            return {'status': 'no_data', 'inserted': 0, 'updated': 0}

        logger.info(f"Merging {len(all_data)} dates from {len(source_data)} sources")

        # Upsert to database
        stats = self._upsert_to_database(all_data)

        return stats

    def _pull_all_xcom_data(self, ti) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Pull data from XCom for all sources.

        Args:
            ti: Task instance

        Returns:
            Dictionary mapping source name to date -> column -> value
        """
        source_data = {}

        xcom_mappings = [
            (L0XComKeys.FRED_DATA, 'fred'),
            (L0XComKeys.TWELVEDATA_DATA, 'twelvedata'),
            (L0XComKeys.INVESTING_DATA, 'investing'),
            (L0XComKeys.BANREP_DATA, 'banrep'),
            (L0XComKeys.EMBI_DATA, 'bcrp'),
            (L0XComKeys.FEDESARROLLO_DATA, 'fedesarrollo'),
            (L0XComKeys.DANE_DATA, 'dane'),
        ]

        for xcom_key, source_name in xcom_mappings:
            data = ti.xcom_pull(key=xcom_key.value) or {}
            if data:
                source_data[source_name] = data
                logger.info(f"  Pulled {len(data)} dates from {source_name}")

        return source_data

    def _merge_source_data(
        self,
        source_data: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Tuple[float, str]]]:
        """
        Merge data from all sources with source tracking.

        Args:
            source_data: Source name -> date -> column -> value

        Returns:
            date -> column -> (value, source)
        """
        all_data: Dict[str, Dict[str, Tuple[float, str]]] = {}

        for source_name, dates_data in source_data.items():
            for date_str, columns in dates_data.items():
                if date_str not in all_data:
                    all_data[date_str] = {}
                for column, value in columns.items():
                    all_data[date_str][column] = (value, source_name)

        return all_data

    def _upsert_to_database(
        self,
        all_data: Dict[str, Dict[str, Tuple[float, str]]]
    ) -> Dict[str, Any]:
        """
        Upsert merged data to database.

        Args:
            all_data: Merged data with source tracking

        Returns:
            Statistics dictionary
        """
        conn = get_db_connection()
        cur = conn.cursor()

        total_updated = 0
        total_skipped = 0
        today = datetime.now().date()

        try:
            for date_str, columns in all_data.items():
                if not columns:
                    continue

                fecha = datetime.strptime(date_str, '%Y-%m-%d').date()

                # Filter out future dates (API timezone issues)
                if fecha > today:
                    logger.warning(f"Skipping future date: {fecha}")
                    total_skipped += 1
                    continue

                for column, (value, source) in columns.items():
                    # Determine release_date based on source
                    offset = self.SOURCE_RELEASE_OFFSETS.get(source, 0)
                    release_date = fecha + timedelta(days=offset)

                    try:
                        cur.execute(f"""
                            INSERT INTO macro_indicators_daily (fecha, {column}, release_date)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (fecha) DO UPDATE SET
                                {column} = EXCLUDED.{column},
                                release_date = COALESCE(macro_indicators_daily.release_date, EXCLUDED.release_date),
                                updated_at = NOW()
                            WHERE macro_indicators_daily.{column} IS NULL
                               OR macro_indicators_daily.{column} != EXCLUDED.{column}
                        """, [fecha, value, release_date])

                        if cur.rowcount > 0:
                            total_updated += 1

                    except Exception as e:
                        logger.warning(f"Error upserting {column} for {fecha}: {e}")

            conn.commit()
            logger.info(f"Database updated: {total_updated} values changed")

        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cur.close()
            conn.close()

        return {
            'status': 'success',
            'dates': len(all_data),
            'updated': total_updated,
            'skipped': total_skipped,
        }

    def forward_fill(self, **context) -> Dict[str, Any]:
        """
        Apply bounded forward fill respecting per-indicator limits.

        Uses MACRO_INDICATOR_REGISTRY to determine max FFILL days per indicator.

        Args:
            **context: Airflow context

        Returns:
            Dictionary with fill statistics
        """
        conn = get_db_connection()
        cur = conn.cursor()

        config = DEFAULT_FFILL_CONFIG
        total_filled = 0
        exceeded_columns = []
        fill_details = {}

        try:
            for column, metadata in MACRO_INDICATOR_REGISTRY.items():
                # Get max FFILL days based on publication schedule
                max_days = config.get_max_days_for_schedule(metadata.schedule)

                # Get last known value with its date
                cur.execute(f"""
                    SELECT fecha, {column}
                    FROM macro_indicators_daily
                    WHERE {column} IS NOT NULL
                    ORDER BY fecha DESC
                    LIMIT 1
                """)

                result = cur.fetchone()
                if not result:
                    fill_details[column] = {
                        "filled": 0,
                        "exceeded": 0,
                        "max_days": max_days,
                        "status": "no_data"
                    }
                    continue

                last_date, last_value = result

                # Fill only up to max_days from last known value
                cur.execute(f"""
                    UPDATE macro_indicators_daily
                    SET {column} = %s,
                        ffilled_from_date = %s,
                        updated_at = NOW()
                    WHERE fecha > %s
                      AND fecha <= %s + INTERVAL '%s days'
                      AND fecha <= CURRENT_DATE
                      AND {column} IS NULL
                """, [last_value, last_date, last_date, last_date, max_days])

                filled = cur.rowcount
                total_filled += filled

                # Count rows that exceed the limit
                cur.execute(f"""
                    SELECT COUNT(*) FROM macro_indicators_daily
                    WHERE fecha > %s + INTERVAL '%s days'
                      AND fecha <= CURRENT_DATE
                      AND {column} IS NULL
                """, [last_date, max_days])
                exceeded = cur.fetchone()[0]

                if exceeded > 0:
                    exceeded_columns.append(column)
                    if config.on_limit_exceeded == "warn":
                        logger.warning(
                            f"[FFILL] {column}: {exceeded} rows exceed "
                            f"{max_days}-day limit (schedule: {metadata.schedule.value})"
                        )

                fill_details[column] = {
                    "filled": filled,
                    "exceeded": exceeded,
                    "max_days": max_days,
                    "last_value": float(last_value) if last_value else None,
                    "last_date": str(last_date),
                }

                if filled > 0:
                    logger.info(f"[FFILL] {metadata.display_name}: {filled} rows filled (max {max_days} days)")

            conn.commit()
            logger.info(f"Bounded forward fill completed: {total_filled} total rows filled")
            if exceeded_columns:
                logger.warning(f"Columns exceeding FFILL limits: {exceeded_columns}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Forward fill error: {e}")
            raise
        finally:
            cur.close()
            conn.close()

        # Push result to XCom
        result = {
            'status': 'success',
            'filled': total_filled,
            'exceeded_columns': exceeded_columns,
            'fill_details': fill_details,
        }

        ti = context.get('ti')
        if ti:
            ti.xcom_push(key='ffill_result', value=result)

        return result

    def generate_readiness_report(self, **context) -> Dict[str, Any]:
        """
        Generate comprehensive data readiness report.

        Creates DailyDataReadinessReport as the SINGLE SOURCE OF TRUTH
        for whether data is ready for inference.

        Args:
            **context: Airflow context

        Returns:
            Dictionary with readiness status
        """
        ti = context.get('ti')

        # Get FFILL result from previous task
        ffill_result = {}
        if ti:
            ffill_result = ti.xcom_pull(key='ffill_result') or {}

        conn = get_db_connection()
        cur = conn.cursor()

        indicator_details = []
        today = date.today()

        try:
            for column, metadata in MACRO_INDICATOR_REGISTRY.items():
                # Get current value for this indicator
                cur.execute(f"""
                    SELECT {column}, fecha
                    FROM macro_indicators_daily
                    WHERE fecha = CURRENT_DATE
                """)
                row = cur.fetchone()

                if row and row[0] is not None:
                    value, fecha = row

                    # Check if this was FFILLed
                    fill_info = ffill_result.get('fill_details', {}).get(column, {})
                    is_ffilled = fill_info.get('filled', 0) > 0
                    ffill_days = (today - fecha).days if fecha else 0

                    if not is_ffilled and ffill_days == 0:
                        status = IndicatorReadinessStatus.FRESH
                    elif ffill_days <= metadata.max_ffill_days:
                        status = IndicatorReadinessStatus.FFILLED
                    else:
                        status = IndicatorReadinessStatus.STALE

                    indicator_details.append(IndicatorReadiness(
                        column_name=column,
                        display_name=metadata.display_name,
                        status=status,
                        latest_value=float(value),
                        latest_observation_date=fecha,
                        age_days=ffill_days,
                        is_ffilled=is_ffilled,
                        ffill_days=ffill_days,
                        ffill_within_limit=ffill_days <= metadata.max_ffill_days
                    ))
                else:
                    indicator_details.append(IndicatorReadiness(
                        column_name=column,
                        display_name=metadata.display_name,
                        status=IndicatorReadinessStatus.MISSING
                    ))

            # Build readiness report
            report = DailyDataReadinessReport(
                indicator_details=indicator_details,
                indicators_fresh=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.FRESH),
                indicators_ffilled=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.FFILLED),
                indicators_stale=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.STALE),
                indicators_missing=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.MISSING),
                indicators_error=sum(1 for d in indicator_details if d.status == IndicatorReadinessStatus.ERROR),
                ffill_applied=ffill_result.get('filled', 0) > 0,
                ffill_total_rows=ffill_result.get('filled', 0),
                ffill_exceeded_limit=len(ffill_result.get('exceeded_columns', [])),
            )

            # Log comprehensive summary
            summary = report.to_summary_dict()
            logger.info("=" * 60)
            logger.info("DAILY DATA READINESS REPORT")
            logger.info("=" * 60)
            logger.info(f"Date: {summary['date']}")
            logger.info(f"Ready for Inference: {'YES' if summary['ready'] else 'NO'}")
            logger.info(f"Readiness Score: {summary['score']}")
            logger.info(f"Indicators: Fresh={summary['fresh']}, FFilled={summary['ffilled']}, "
                       f"Stale={summary['stale']}, Missing={summary['missing']}, Errors={summary['errors']}")

            if report.blocking_issues:
                logger.error("BLOCKING ISSUES:")
                for issue in report.blocking_issues:
                    logger.error(f"  - {issue}")

            if report.warnings:
                logger.warning("WARNINGS:")
                for warn in report.warnings:
                    logger.warning(f"  - {warn}")

            # Push to XCom
            if ti:
                ti.xcom_push(key='readiness_report', value=summary)
                ti.xcom_push(key='is_ready_for_inference', value=report.is_ready_for_inference)

            return {
                'status': 'success',
                'is_ready': report.is_ready_for_inference,
                'score': report.readiness_score,
                'summary': summary,
            }

        except Exception as e:
            logger.error(f"Readiness report error: {e}")
            return {'status': 'error', 'message': str(e), 'is_ready': False}
        finally:
            cur.close()
            conn.close()


# =============================================================================
# Airflow Task Functions
# =============================================================================

def merge_and_upsert(**context) -> Dict[str, Any]:
    """Airflow task for merge and upsert."""
    service = MacroMergeService()
    return service.merge_and_upsert(**context)


def apply_forward_fill(**context) -> Dict[str, Any]:
    """Airflow task for forward fill."""
    service = MacroMergeService()
    return service.forward_fill(**context)


def generate_readiness_report(**context) -> Dict[str, Any]:
    """Airflow task for readiness report."""
    service = MacroMergeService()
    return service.generate_readiness_report(**context)
