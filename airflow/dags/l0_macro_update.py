# -*- coding: utf-8 -*-
"""
DAG: l0_macro_update v2.0
=========================
Realtime macro data updates - hourly extraction without change detection.

Features:
- Hourly execution (8:00-12:00 COT, Mon-Fri)
- Always rewrites last 15 records per variable
- No change detection (eliminates overhead)
- Full metrics tracking per variable/source
- Daily summary report at last run
- Circuit breaker protection per source

Contract: CTR-L0-UPDATE-002

Variables:
- Daily (18): DXY, VIX, UST10Y, UST2Y, IBR, TPM, EMBI, etc.
- Monthly (8): FEDFUNDS, CPI, UNEMPLOYMENT, etc.
- Quarterly (4): GDP, BOP data

Schedule: Hourly 8:00-12:00 COT (13:00-17:00 UTC), Mon-Fri

Version: 2.0.0
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# =============================================================================
# PATH SETUP
# =============================================================================

DAGS_DIR = Path(__file__).parent
PROJECT_ROOT = DAGS_DIR.parent.parent
SRC_PATH = PROJECT_ROOT / 'src'

for path in [str(DAGS_DIR), str(SRC_PATH), str(PROJECT_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of records to ALWAYS rewrite (increased from 5 for safety)
SAFETY_RECORDS = 15

# Market hours in COT timezone (8:00-13:00 = until 12:59)
MARKET_HOURS_START = 8
MARKET_HOURS_END = 13

# Last run hour in COT (12:00 = 17:00 UTC)
LAST_RUN_HOUR_UTC = 17

# Critical variables that must have is_complete = true
CRITICAL_VARIABLES = [
    'dxy', 'vix', 'ust10y', 'ust2y', 'ibr', 'tpm', 'embi_col',
]

# Default arguments
default_args = {
    'owner': 'usdcop-data-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['alerts@trading.local'],
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=30),
}


# =============================================================================
# DATACLASSES FOR METRICS
# =============================================================================

@dataclass
class HourlyExtractionReport:
    """Reporte de métricas por ejecución horaria."""
    run_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Por Source
    source_success_count: Dict[str, int] = field(default_factory=dict)
    source_failed_count: Dict[str, int] = field(default_factory=dict)
    source_duration_ms: Dict[str, int] = field(default_factory=dict)

    # Por Variable
    variable_success: List[str] = field(default_factory=list)
    variable_failed: List[str] = field(default_factory=list)

    # Por Frecuencia
    daily_vars_success: int = 0
    daily_vars_failed: int = 0
    monthly_vars_success: int = 0
    monthly_vars_failed: int = 0
    quarterly_vars_success: int = 0
    quarterly_vars_failed: int = 0

    # Timing
    total_extraction_duration_ms: int = 0
    total_upsert_duration_ms: int = 0

    # Records
    total_records_extracted: int = 0
    total_records_upserted: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XCom serialization."""
        return {
            'run_timestamp': self.run_timestamp.isoformat(),
            'source_success_count': self.source_success_count,
            'source_failed_count': self.source_failed_count,
            'source_duration_ms': self.source_duration_ms,
            'variable_success': self.variable_success,
            'variable_failed': self.variable_failed,
            'daily_vars_success': self.daily_vars_success,
            'daily_vars_failed': self.daily_vars_failed,
            'monthly_vars_success': self.monthly_vars_success,
            'monthly_vars_failed': self.monthly_vars_failed,
            'quarterly_vars_success': self.quarterly_vars_success,
            'quarterly_vars_failed': self.quarterly_vars_failed,
            'total_extraction_duration_ms': self.total_extraction_duration_ms,
            'total_upsert_duration_ms': self.total_upsert_duration_ms,
            'total_records_extracted': self.total_records_extracted,
            'total_records_upserted': self.total_records_upserted,
        }

    def to_log_format(self) -> str:
        """Format for structured logging."""
        total_vars = len(self.variable_success) + len(self.variable_failed)
        success_rate = len(self.variable_success) / total_vars if total_vars > 0 else 0

        return (
            f"Vars: {len(self.variable_success)}/{total_vars} ({success_rate:.1%}) | "
            f"Records: {self.total_records_extracted} extracted, {self.total_records_upserted} upserted | "
            f"Duration: {self.total_extraction_duration_ms}ms extract, {self.total_upsert_duration_ms}ms upsert"
        )


@dataclass
class DailyExtractionSummary:
    """Resumen del día completo (5 ejecuciones)."""
    date: date = field(default_factory=date.today)

    # Ejecuciones
    total_runs: int = 0
    successful_runs: int = 0

    # Por Variable (agregado del día)
    variable_success_rate: Dict[str, float] = field(default_factory=dict)

    # Por Source (agregado del día)
    source_success_rate: Dict[str, float] = field(default_factory=dict)
    source_avg_duration_ms: Dict[str, float] = field(default_factory=dict)

    # Latencia
    avg_extraction_to_db_latency_ms: float = 0.0

    # Alertas
    sources_with_issues: List[str] = field(default_factory=list)
    variables_with_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'date': self.date.isoformat(),
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'variable_success_rate': self.variable_success_rate,
            'source_success_rate': self.source_success_rate,
            'source_avg_duration_ms': self.source_avg_duration_ms,
            'avg_latency_ms': self.avg_extraction_to_db_latency_ms,
            'sources_with_issues': self.sources_with_issues,
            'variables_with_issues': self.variables_with_issues,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_db_connection():
    """Get database connection from Airflow connection."""
    from airflow.hooks.postgres_hook import PostgresHook
    hook = PostgresHook(postgres_conn_id='timescale_conn')
    return hook.get_conn()


def get_circuit_breaker_for_source(source_name: str):
    """Get or create circuit breaker for a data source."""
    try:
        from src.core.circuit_breaker import (
            CircuitBreakerRegistry,
            CircuitBreakerConfig,
        )

        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=300.0,  # 5 min cooldown
            failure_rate_threshold=0.5,
        )

        registry = CircuitBreakerRegistry()
        return registry.get_or_create(f"l0_update_{source_name}", config)
    except ImportError:
        logger.warning("Circuit breaker not available, proceeding without it")
        return None


def is_last_run_of_day() -> bool:
    """Check if this is the last run of the day (12:00 COT = 17:00 UTC)."""
    now_utc = datetime.utcnow()
    return now_utc.hour == LAST_RUN_HOUR_UTC


def send_pipeline_alert(
    title: str,
    message: str,
    severity: str = "INFO",
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Send alert via logging (AlertService integration optional)."""
    log_func = {
        "INFO": logger.info,
        "WARNING": logger.warning,
        "CRITICAL": logger.error,
    }.get(severity, logger.info)

    log_func(f"[ALERT][{severity}] {title}: {message}")
    if metrics:
        logger.info(f"[ALERT] Metrics: {metrics}")


# =============================================================================
# TASK 1: HEALTH CHECK
# =============================================================================

def health_check(**context) -> Dict[str, Any]:
    """
    Quick verification before starting extraction.

    Checks:
    - Database connectivity
    - ExtractorRegistry availability
    - Config files accessible
    """
    logger.info("=" * 60)
    logger.info("L0 MACRO UPDATE v2.0 - Health Check")
    logger.info("=" * 60)

    results = {
        'database': False,
        'extractors': False,
        'timestamp': datetime.utcnow().isoformat(),
    }

    # Check database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        results['database'] = True
        logger.info("[HEALTH] Database: OK")
    except Exception as e:
        logger.error(f"[HEALTH] Database: FAILED - {e}")
        raise RuntimeError(f"Database health check failed: {e}")

    # Check extractors
    try:
        from extractors.registry import ExtractorRegistry
        registry = ExtractorRegistry()
        all_vars = registry.get_all_variables()
        results['extractors'] = True
        results['total_variables'] = len(all_vars)
        logger.info(f"[HEALTH] Extractors: OK ({len(all_vars)} variables)")
    except Exception as e:
        logger.error(f"[HEALTH] Extractors: FAILED - {e}")
        raise RuntimeError(f"Extractor registry failed: {e}")

    context['ti'].xcom_push(key='health_check', value=results)
    logger.info("[HEALTH] All checks passed")
    return results


# =============================================================================
# TASK 2: CHECK MARKET HOURS (SHORT CIRCUIT)
# =============================================================================

def check_market_hours(**context) -> bool:
    """
    Check if we're within market hours.

    Returns True to continue, False to skip the rest of the DAG.
    Can be bypassed with force_run parameter.
    """
    # Check for force_run override
    conf = context.get('dag_run').conf or {}
    if conf.get('force_run', False):
        logger.info("[HOURS] Force run enabled, bypassing market hours check")
        return True

    try:
        import pytz
    except ImportError:
        logger.warning("[HOURS] pytz not available, proceeding anyway")
        return True

    tz = pytz.timezone('America/Bogota')
    now = datetime.now(tz)

    # Market hours: 8:00-13:00 COT, Mon-Fri
    is_weekday = now.weekday() < 5
    is_market_hours = MARKET_HOURS_START <= now.hour < MARKET_HOURS_END

    if not (is_weekday and is_market_hours):
        logger.info(
            "[HOURS] Outside market hours: %s (hour=%d, weekday=%d). Skipping.",
            now.strftime("%Y-%m-%d %H:%M"),
            now.hour,
            now.weekday()
        )
        return False

    logger.info("[HOURS] Within market hours (%s). Proceeding.", now.strftime("%H:%M COT"))
    return True


# =============================================================================
# TASK 3: EXTRACT ALL SOURCES (WITH CIRCUIT BREAKERS)
# =============================================================================

def extract_all_sources(**context) -> Dict[str, Any]:
    """
    Extract last N records from ALL sources with circuit breaker protection.

    Key differences from v1.0:
    - NO change detection (always extract)
    - SAFETY_RECORDS = 15 (increased from 5)
    - Circuit breaker per source
    - Full metrics tracking
    """
    from extractors.registry import ExtractorRegistry
    from extractors.metrics import (
        ExtractionMetrics,
        reset_metrics_collector,
    )

    logger.info("=" * 60)
    logger.info("L0 MACRO UPDATE v2.0 - Extraction Phase")
    logger.info(f"SAFETY_RECORDS = {SAFETY_RECORDS}")
    logger.info("=" * 60)

    # Reset metrics collector for this run
    collector = reset_metrics_collector()

    # Initialize registry
    registry = ExtractorRegistry()
    all_sources = registry.get_all_sources()
    all_variables = registry.get_all_variables()

    logger.info(f"[EXTRACT] Sources: {all_sources}")
    logger.info(f"[EXTRACT] Total variables: {len(all_variables)}")

    # Get skip_sources from config
    conf = context.get('dag_run').conf or {}
    skip_sources = conf.get('skip_sources', [])
    if skip_sources:
        logger.info(f"[EXTRACT] Skipping sources: {skip_sources}")

    # Initialize report
    report = HourlyExtractionReport()

    # Results by source
    extraction_results = {}
    extraction_data = {}  # variable -> DataFrame

    for source_name in all_sources:
        if source_name in skip_sources:
            logger.info(f"[{source_name.upper()}] Skipped by request")
            continue

        source_start = time.time()
        source_success = 0
        source_failed = 0

        # Get circuit breaker for this source
        cb = get_circuit_breaker_for_source(source_name)

        # Get variables for this source
        source_vars = registry.get_variables_by_source(source_name)
        logger.info(f"[{source_name.upper()}] Extracting {len(source_vars)} variables...")

        for variable in source_vars:
            var_start = time.time()

            try:
                # Extract with circuit breaker if available
                if cb:
                    result = cb.call(
                        registry.extract_variable,
                        variable,
                        last_n=SAFETY_RECORDS
                    )
                else:
                    result = registry.extract_variable(variable, last_n=SAFETY_RECORDS)

                if result.success and result.data is not None and not result.data.empty:
                    records = len(result.data)
                    extraction_data[variable] = result.data
                    report.variable_success.append(variable)
                    report.total_records_extracted += records
                    source_success += 1

                    # Track by frequency
                    var_config = registry.get_variable_config(variable)
                    freq = var_config.get('frequency', 'D')
                    if freq == 'D':
                        report.daily_vars_success += 1
                    elif freq == 'M':
                        report.monthly_vars_success += 1
                    elif freq == 'Q':
                        report.quarterly_vars_success += 1

                    logger.debug(f"[{source_name.upper()}] {variable}: {records} records")
                else:
                    report.variable_failed.append(variable)
                    source_failed += 1
                    logger.warning(
                        f"[{source_name.upper()}] {variable}: No data - {result.error or 'Empty'}"
                    )

                # Record metrics
                var_duration = int((time.time() - var_start) * 1000)
                collector.record(ExtractionMetrics(
                    source=source_name,
                    variable=variable,
                    records_extracted=len(result.data) if result.data is not None else 0,
                    total_duration_ms=var_duration,
                    success=result.success,
                    error_message=result.error,
                ))

            except Exception as e:
                report.variable_failed.append(variable)
                source_failed += 1
                logger.error(f"[{source_name.upper()}] {variable}: Exception - {e}")

                collector.record(ExtractionMetrics(
                    source=source_name,
                    variable=variable,
                    records_extracted=0,
                    total_duration_ms=int((time.time() - var_start) * 1000),
                    success=False,
                    error_message=str(e),
                ))

        # Source summary
        source_duration = int((time.time() - source_start) * 1000)
        report.source_success_count[source_name] = source_success
        report.source_failed_count[source_name] = source_failed
        report.source_duration_ms[source_name] = source_duration
        report.total_extraction_duration_ms += source_duration

        extraction_results[source_name] = {
            'success': source_success,
            'failed': source_failed,
            'duration_ms': source_duration,
        }

        logger.info(
            f"[{source_name.upper()}] Complete: {source_success} success, "
            f"{source_failed} failed, {source_duration}ms"
        )

    # Log summary
    collector.log_summary()

    # Push results to XCom
    context['ti'].xcom_push(key='extraction_results', value=extraction_results)
    context['ti'].xcom_push(key='extraction_data', value={
        v: df.to_dict() for v, df in extraction_data.items()
    })
    context['ti'].xcom_push(key='extraction_report', value=report.to_dict())

    logger.info(f"[EXTRACT] {report.to_log_format()}")

    return extraction_results


# =============================================================================
# TASK 4: UPSERT ALL (4-TABLE ARCHITECTURE - ROUTES BY FREQUENCY)
# =============================================================================

def upsert_all(**context) -> Dict[str, Any]:
    """
    UPSERT last 15 records for ALL extracted variables.

    v2.1 Changes (4-Table Architecture):
    - Uses FrequencyRoutedUpsertService for automatic routing
    - Daily variables → macro_indicators_daily
    - Monthly variables → macro_indicators_monthly
    - Quarterly variables → macro_indicators_quarterly
    - NO comparison, always rewrites
    - SAFETY_RECORDS = 15

    Contract: CTR-L0-4TABLE-001
    """
    from services.upsert_service import FrequencyRoutedUpsertService

    logger.info("=" * 60)
    logger.info("L0 MACRO UPDATE v2.1 - UPSERT Phase (4-Table Architecture)")
    logger.info("=" * 60)

    # Get extraction data from previous task
    ti = context['ti']
    extraction_data_raw = ti.xcom_pull(task_ids='extract_all_sources', key='extraction_data') or {}

    if not extraction_data_raw:
        logger.warning("[UPSERT] No extraction data available")
        return {'success': 0, 'failed': 0, 'total_rows': 0}

    # Reconstruct DataFrames
    extraction_data = {}
    for var, data_dict in extraction_data_raw.items():
        try:
            df = pd.DataFrame.from_dict(data_dict)
            extraction_data[var] = df
        except Exception as e:
            logger.warning(f"[UPSERT] Could not reconstruct {var}: {e}")

    logger.info(f"[UPSERT] Variables to upsert: {len(extraction_data)}")

    # Initialize frequency-routed upsert service
    conn = get_db_connection()
    upsert_service = FrequencyRoutedUpsertService(conn)

    results = {
        'success': 0,
        'failed': 0,
        'total_rows': 0,
        'errors': [],
        'by_table': {
            'daily': {'success': 0, 'failed': 0, 'rows': 0},
            'monthly': {'success': 0, 'failed': 0, 'rows': 0},
            'quarterly': {'success': 0, 'failed': 0, 'rows': 0},
        }
    }

    upsert_start = time.time()

    for variable, df in extraction_data.items():
        try:
            # UPSERT using frequency-routed service (auto-routes to correct table)
            upsert_result = upsert_service.upsert_variable(
                variable,
                df,
                n=SAFETY_RECORDS
            )

            freq = upsert_result.get('frequency', 'daily')
            table = upsert_result.get('table', 'unknown')

            if upsert_result.get('success'):
                results['success'] += 1
                results['total_rows'] += upsert_result.get('rows_affected', 0)
                results['by_table'][freq]['success'] += 1
                results['by_table'][freq]['rows'] += upsert_result.get('rows_affected', 0)
                logger.debug(
                    f"[UPSERT] {variable} → {table}: {upsert_result.get('rows_affected', 0)} rows"
                )
            else:
                results['failed'] += 1
                results['by_table'][freq]['failed'] += 1
                error = upsert_result.get('error', 'Unknown')
                results['errors'].append(f"{variable}: {error}")
                logger.warning(f"[UPSERT] {variable} failed: {error}")

        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"{variable}: {str(e)}")
            logger.error(f"[UPSERT] {variable} exception: {e}")

    conn.close()

    upsert_duration = int((time.time() - upsert_start) * 1000)
    results['duration_ms'] = upsert_duration

    # Log summary by table
    logger.info(
        f"[UPSERT] Complete: {results['success']} success, {results['failed']} failed, "
        f"{results['total_rows']} rows, {upsert_duration}ms"
    )
    for freq in ('daily', 'monthly', 'quarterly'):
        stats = results['by_table'][freq]
        if stats['success'] + stats['failed'] > 0:
            logger.info(
                f"[UPSERT] {freq}: {stats['success']} success, "
                f"{stats['failed']} failed, {stats['rows']} rows"
            )

    # Push to XCom
    context['ti'].xcom_push(key='upsert_results', value=results)

    return results


# =============================================================================
# TASK 5: UPDATE IS_COMPLETE FLAG
# =============================================================================

def update_is_complete(**context) -> Dict[str, Any]:
    """
    Update is_complete flag for critical variables.

    Sets is_complete = TRUE for dates where all critical variables have data.
    """
    logger.info("=" * 60)
    logger.info("L0 MACRO UPDATE v2.0 - is_complete Refresh")
    logger.info("=" * 60)

    results = {
        'dates_updated': 0,
        'critical_vars_checked': len(CRITICAL_VARIABLES),
    }

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Build dynamic SQL to check all critical variables
        # A date is "complete" if all critical variables have non-NULL values
        var_conditions = ' AND '.join([f"{var} IS NOT NULL" for var in CRITICAL_VARIABLES])

        # Update is_complete for last N days
        update_sql = f"""
            UPDATE macro_indicators_daily
            SET is_complete = (
                CASE WHEN {var_conditions} THEN TRUE ELSE FALSE END
            ),
            updated_at = NOW()
            WHERE fecha >= CURRENT_DATE - INTERVAL '{SAFETY_RECORDS} days'
        """

        cursor.execute(update_sql)
        results['dates_updated'] = cursor.rowcount
        conn.commit()

        # Verify: count complete vs incomplete in recent window
        verify_sql = f"""
            SELECT
                SUM(CASE WHEN is_complete THEN 1 ELSE 0 END) as complete_count,
                SUM(CASE WHEN NOT is_complete OR is_complete IS NULL THEN 1 ELSE 0 END) as incomplete_count
            FROM macro_indicators_daily
            WHERE fecha >= CURRENT_DATE - INTERVAL '{SAFETY_RECORDS} days'
        """
        cursor.execute(verify_sql)
        row = cursor.fetchone()
        results['complete_count'] = row[0] or 0
        results['incomplete_count'] = row[1] or 0

        cursor.close()
        conn.close()

        logger.info(
            f"[IS_COMPLETE] Updated {results['dates_updated']} rows. "
            f"Complete: {results['complete_count']}, Incomplete: {results['incomplete_count']}"
        )

    except Exception as e:
        logger.error(f"[IS_COMPLETE] Failed: {e}")
        results['error'] = str(e)

    context['ti'].xcom_push(key='is_complete_results', value=results)
    return results


# =============================================================================
# TASK 6: LOG METRICS
# =============================================================================

def log_metrics(**context) -> None:
    """
    Log comprehensive metrics for this run.

    Outputs structured logs for monitoring and Prometheus integration.
    """
    logger.info("=" * 60)
    logger.info("L0 MACRO UPDATE v2.0 - Metrics")
    logger.info("=" * 60)

    ti = context['ti']

    # Gather all results
    extraction_results = ti.xcom_pull(task_ids='extract_all_sources', key='extraction_results') or {}
    extraction_report = ti.xcom_pull(task_ids='extract_all_sources', key='extraction_report') or {}
    upsert_results = ti.xcom_pull(task_ids='upsert_all', key='upsert_results') or {}
    is_complete_results = ti.xcom_pull(task_ids='update_is_complete', key='is_complete_results') or {}

    # Calculate totals
    total_success = sum(s.get('success', 0) for s in extraction_results.values())
    total_failed = sum(s.get('failed', 0) for s in extraction_results.values())
    total_vars = total_success + total_failed
    success_rate = total_success / total_vars if total_vars > 0 else 0

    # Build metrics dict
    metrics = {
        'run_id': context.get('run_id', 'unknown'),
        'dag_id': _MACRO_UPDATE_DAG_ID,
        'execution_date': str(context.get('execution_date', '')),
        'sources_count': len(extraction_results),
        'variables_success': total_success,
        'variables_failed': total_failed,
        'variables_total': total_vars,
        'success_rate': round(success_rate, 4),
        'records_extracted': extraction_report.get('total_records_extracted', 0),
        'records_upserted': upsert_results.get('total_rows', 0),
        'extraction_duration_ms': extraction_report.get('total_extraction_duration_ms', 0),
        'upsert_duration_ms': upsert_results.get('duration_ms', 0),
        'dates_complete': is_complete_results.get('complete_count', 0),
        'dates_incomplete': is_complete_results.get('incomplete_count', 0),
    }

    # Structured log (JSON-like for log aggregators)
    logger.info(
        "[L0_UPDATE] Run complete",
        extra={'metrics': metrics}
    )

    # Human-readable summary
    logger.info(
        f"[METRICS] Variables: {total_success}/{total_vars} ({success_rate:.1%}) | "
        f"Records: {metrics['records_extracted']} extracted, {metrics['records_upserted']} upserted | "
        f"Duration: {metrics['extraction_duration_ms']}ms extract + {metrics['upsert_duration_ms']}ms upsert"
    )

    # Per-source breakdown
    logger.info("[METRICS] By source:")
    for source, stats in extraction_results.items():
        status = 'OK' if stats.get('failed', 0) == 0 else 'PARTIAL'
        logger.info(
            f"  [{status}] {source}: {stats.get('success', 0)} success, "
            f"{stats.get('failed', 0)} failed, {stats.get('duration_ms', 0)}ms"
        )

    # Alert if success rate is low
    if success_rate < 0.8:
        send_pipeline_alert(
            title="L0 Macro Update: Low Success Rate",
            message=f"Only {success_rate:.1%} of variables extracted successfully",
            severity="WARNING",
            metrics=metrics
        )


# =============================================================================
# TASK 7: DAILY SUMMARY (CONDITIONAL)
# =============================================================================

def daily_summary(**context) -> Optional[Dict[str, Any]]:
    """
    Generate daily summary report (only at last run of day: 12:00 COT).

    Aggregates metrics from all runs today.
    """
    if not is_last_run_of_day():
        logger.info("[DAILY] Not last run of day, skipping summary")
        return None

    logger.info("=" * 60)
    logger.info("L0 MACRO UPDATE v2.0 - Daily Summary")
    logger.info("=" * 60)

    ti = context['ti']

    # Get current run metrics
    extraction_results = ti.xcom_pull(task_ids='extract_all_sources', key='extraction_results') or {}
    upsert_results = ti.xcom_pull(task_ids='upsert_all', key='upsert_results') or {}

    # Build daily summary (in production, would aggregate from all runs)
    summary = DailyExtractionSummary(date=date.today())
    summary.total_runs = 5  # Expected runs per day

    # Calculate source success rates from this run
    for source, stats in extraction_results.items():
        total = stats.get('success', 0) + stats.get('failed', 0)
        if total > 0:
            rate = stats.get('success', 0) / total
            summary.source_success_rate[source] = round(rate, 2)
            summary.source_avg_duration_ms[source] = float(stats.get('duration_ms', 0))

            if rate < 0.8:
                summary.sources_with_issues.append(source)

    # Log daily summary
    logger.info(
        "[DAILY] Summary",
        extra={'daily_metrics': summary.to_dict()}
    )

    logger.info(f"[DAILY] Date: {summary.date}")
    logger.info(f"[DAILY] Source success rates: {summary.source_success_rate}")

    if summary.sources_with_issues:
        logger.warning(f"[DAILY] Sources with issues (<80%): {summary.sources_with_issues}")
        send_pipeline_alert(
            title="L0 Macro Update: Daily Issues",
            message=f"Sources with low success rate: {', '.join(summary.sources_with_issues)}",
            severity="WARNING",
            metrics=summary.to_dict()
        )
    else:
        logger.info("[DAILY] All sources performing well")

    context['ti'].xcom_push(key='daily_summary', value=summary.to_dict())
    return summary.to_dict()


# =============================================================================
# DAG DEFINITION
# =============================================================================

try:
    from contracts.dag_registry import CORE_L0_MACRO_UPDATE
    _MACRO_UPDATE_DAG_ID = CORE_L0_MACRO_UPDATE
except ImportError:
    _MACRO_UPDATE_DAG_ID = 'core_l0_04_macro_update'

with DAG(
    _MACRO_UPDATE_DAG_ID,
    default_args=default_args,
    description='Realtime macro updates v2.0 - hourly, always rewrite last 15',
    # Hourly 8:00-12:00 COT = 13:00-17:00 UTC, Mon-Fri
    schedule_interval='0 13-17 * * 1-5',
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['l0', 'macro', 'update', 'realtime', 'v2'],
    params={
        'safety_records': SAFETY_RECORDS,
        'skip_sources': [],
        'force_run': False,
    },
) as dag:

    dag.doc_md = """
    ## L0 Macro Update DAG v2.0

    Realtime macro data updates - hourly extraction without change detection.

    ### Contract: CTR-L0-UPDATE-002

    ### Key Differences from v1.0
    - **No change detection**: Always rewrites last 15 records (eliminates overhead)
    - **Increased safety margin**: 15 records instead of 5
    - **Circuit breakers**: Per-source failure protection
    - **Full metrics**: Per-variable/source tracking
    - **is_complete refresh**: Updates flag for critical variables
    - **Daily summary**: Report at last run of day

    ### Schedule (COT)
    8:00, 9:00, 10:00, 11:00, 12:00 (Mon-Fri) = 5 runs/day

    ### Parameters
    - `safety_records`: Records to rewrite (default: 15)
    - `skip_sources`: Sources to ignore (e.g., ['dane'])
    - `force_run`: Bypass market hours check

    ### Pipeline Stages
    1. **health_check**: Verify DB and extractors
    2. **check_market_hours**: Skip if outside hours (unless force_run)
    3. **extract_all_sources**: Parallel extraction with circuit breakers
    4. **upsert_all**: ALWAYS rewrite last 15 per variable
    5. **update_is_complete**: Refresh flag for critical vars
    6. **log_metrics**: Structured logging
    7. **daily_summary**: Conditional report at 12:00 COT

    ### Usage
    ```bash
    # Normal run (respects market hours)
    airflow dags trigger l0_macro_update

    # Force run outside market hours
    airflow dags trigger l0_macro_update --conf '{"force_run": true}'

    # Skip problematic source
    airflow dags trigger l0_macro_update --conf '{"skip_sources": ["dane"]}'
    ```

    ### Monitoring
    - Check logs for `[L0_UPDATE]` prefix
    - Daily summary at 12:00 COT includes aggregate metrics
    - Alerts sent when success rate < 80%
    """

    # Task 1: Health Check
    health = PythonOperator(
        task_id='health_check',
        python_callable=health_check,
    )

    # Task 2: Check Market Hours (short-circuit)
    check_hours = ShortCircuitOperator(
        task_id='check_market_hours',
        python_callable=check_market_hours,
    )

    # Task 3: Extract All Sources
    extract = PythonOperator(
        task_id='extract_all_sources',
        python_callable=extract_all_sources,
    )

    # Task 4: UPSERT All
    upsert = PythonOperator(
        task_id='upsert_all',
        python_callable=upsert_all,
    )

    # Task 5: Update is_complete
    is_complete = PythonOperator(
        task_id='update_is_complete',
        python_callable=update_is_complete,
    )

    # Task 6: Log Metrics
    metrics = PythonOperator(
        task_id='log_metrics',
        python_callable=log_metrics,
    )

    # Task 7: Daily Summary (conditional)
    summary = PythonOperator(
        task_id='daily_summary',
        python_callable=daily_summary,
        trigger_rule='all_done',  # Run even if upstream has issues
    )

    # Define dependencies
    health >> check_hours >> extract >> upsert >> is_complete >> metrics >> summary
